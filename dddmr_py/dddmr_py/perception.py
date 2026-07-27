"""3D 感知层: 体素降采样 / 法向量估计 / 地面(可站立面)提取 / 静态代价层.

对应 dddmr_perception_3d 中 static layer 的职责:
把一张 PCD 地图变成"带代价的可通行节点集合", 供全局规划器建图搜索.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import PlannerConfig


@dataclass
class GroundMap:
    """可通行面(节点)集合及其属性."""

    nodes: np.ndarray            # (M,3) 节点坐标
    normals: np.ndarray          # (M,3) 单位法向量 (nz>0)
    slope: np.ndarray            # (M,)  坡度 (rad)
    roughness: np.ndarray        # (M,)  局部平面残差 (m)
    clearance: np.ndarray        # (M,)  头顶净空高度 (m)
    cost: np.ndarray             # (M,)  静态代价 0~lethal_cost
    lethal: np.ndarray           # (M,)  bool, True=不可通行
    obstacles: np.ndarray        # (K,3) 障碍点(非地面点)
    kdtree: cKDTree              # 节点 3D kd-tree
    config: PlannerConfig

    @property
    def free_ratio(self) -> float:
        return float((~self.lethal).mean()) if len(self.lethal) else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<GroundMap nodes={len(self.nodes)} free={self.free_ratio:.1%} "
                f"obstacles={len(self.obstacles)}>")


# --------------------------------------------------------------------------
def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """体素质心降采样. voxel<=0 时原样返回."""
    if voxel <= 0 or len(xyz) == 0:
        return np.asarray(xyz, dtype=np.float64)
    keys = np.floor(np.asarray(xyz) / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inv, xyz)
    return sums / counts[:, None]


def estimate_normals(xyz: np.ndarray, k: int = 16, radius: float = 0.0,
                     chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """PCA 法向量估计.

    返回 (normals(N,3) 单位向量且 nz>=0, residual(N,) 最小特征值的平方根≈平面残差).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    residual = np.zeros(n)
    if n < 3:
        return normals, residual
    tree = cKDTree(xyz)
    k = int(min(max(k, 4), n))

    for beg in range(0, n, chunk):
        end = min(beg + chunk, n)
        if radius > 0:
            # 半径近邻: 用 k 近邻再按半径裁剪, 保证向量化
            dist, idx = tree.query(xyz[beg:end], k=k, workers=-1)
            valid = dist <= radius
            valid[:, 0] = True
            idx = np.where(valid, idx, idx[:, :1])
        else:
            _, idx = tree.query(xyz[beg:end], k=k, workers=-1)
        nb = xyz[idx]                                    # (c,k,3)
        centered = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("ikj,ikl->ijl", centered, centered) / max(k - 1, 1)
        evals, evecs = np.linalg.eigh(cov)               # 升序
        nrm = evecs[:, :, 0]
        flip = nrm[:, 2] < 0
        nrm[flip] *= -1.0
        normals[beg:end] = nrm
        residual[beg:end] = np.sqrt(np.clip(evals[:, 0], 0.0, None))
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norm_len > 1e-12, norm_len, 1.0)
    return normals, residual


# --------------------------------------------------------------------------
def _cylinder_neighbors(query_xy: np.ndarray, tree_xy: cKDTree, radius: float):
    """返回 (query_index, target_index) 的扁平配对, 便于向量化后处理."""
    lists = tree_xy.query_ball_point(query_xy, radius, workers=-1)
    lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    if lens.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    flat = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if l])
    qidx = np.repeat(np.arange(len(lists), dtype=np.int64), lens)
    return qidx, flat


def build_ground_map(xyz: np.ndarray, cfg: Optional[PlannerConfig] = None,
                     verbose: bool = False) -> GroundMap:
    """把原始点云处理成 GroundMap (可通行节点 + 静态代价)."""
    cfg = cfg or PlannerConfig()
    pts = voxel_downsample(np.asarray(xyz, dtype=np.float64), cfg.voxel_size)
    if verbose:
        print(f"[perception] 降采样: {len(xyz)} -> {len(pts)} 点 "
              f"(voxel={cfg.voxel_size} m)")

    normals, residual = estimate_normals(pts, k=cfg.normal_k, radius=cfg.normal_radius)
    slope = np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0))

    # 1) 可站立面候选: 坡度 + 局部平整度
    max_slope = np.deg2rad(cfg.max_slope_deg)
    surface = (slope <= max_slope) & (residual <= cfg.max_roughness)
    if not surface.any():
        raise RuntimeError("未找到任何可站立面, 请放宽 max_slope_deg / max_roughness")

    nodes = pts[surface]
    node_normals = normals[surface]
    node_slope = slope[surface]
    node_rough = residual[surface]
    obstacles = pts[~surface]
    if verbose:
        print(f"[perception] 可站立面候选 {len(nodes)} 个, 障碍点 {len(obstacles)} 个")

    # 2) 净空(clearance): 机器人本体圆柱内是否有点.
    #    高度用"相对局部切平面"的高度, 否则斜坡上的上坡点会被误判成头顶障碍.
    clearance = np.full(len(nodes), np.inf)
    if len(pts):
        tree_all_xy = cKDTree(pts[:, :2])
        qi, ti = _cylinder_neighbors(nodes[:, :2], tree_all_xy, cfg.robot_radius)
        if len(qi):
            h = np.einsum("ij,ij->i", pts[ti] - nodes[qi], node_normals[qi])
            above = h > cfg.max_step
            if above.any():
                np.minimum.at(clearance, qi[above], h[above])

    # 3) 静态代价: 障碍点在"机器人身体高度带"内 -> 膨胀
    cost = np.zeros(len(nodes))
    lethal = np.zeros(len(nodes), dtype=bool)
    if len(obstacles):
        tree_obs_xy = cKDTree(obstacles[:, :2])
        qi, ti = _cylinder_neighbors(nodes[:, :2], tree_obs_xy, cfg.inflation_radius)
        if len(qi):
            dz = np.einsum("ij,ij->i", obstacles[ti] - nodes[qi], node_normals[qi])
            band = (dz > cfg.max_step) & (dz < cfg.robot_height)
            qi, ti = qi[band], ti[band]
            if len(qi):
                d = np.linalg.norm(obstacles[ti, :2] - nodes[qi, :2], axis=1)
                nearest = np.full(len(nodes), np.inf)
                np.minimum.at(nearest, qi, d)
                inside = nearest <= cfg.inflation_radius
                lethal |= nearest <= cfg.robot_radius
                decay = (cfg.lethal_cost - 1.0) * np.exp(
                    -cfg.cost_scaling_factor * np.clip(nearest - cfg.robot_radius, 0, None))
                cost = np.where(inside, decay, 0.0)
    cost = np.where(lethal, cfg.lethal_cost, cost)

    # 4) 净空不足(钻不过去) -> 致命
    low_ceiling = clearance < cfg.robot_height
    lethal |= low_ceiling
    cost = np.where(lethal, cfg.lethal_cost, cost)

    if verbose:
        print(f"[perception] 致命节点 {int(lethal.sum())} 个 "
              f"(其中本体碰撞/低净空 {int(low_ceiling.sum())} 个), "
              f"可通行率 {100.0 * (~lethal).mean():.1f}%")

    return GroundMap(
        nodes=nodes, normals=node_normals, slope=node_slope, roughness=node_rough,
        clearance=clearance, cost=cost, lethal=lethal, obstacles=obstacles,
        kdtree=cKDTree(nodes), config=cfg,
    )
