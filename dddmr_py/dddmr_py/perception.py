"""3D 感知层: 体素降采样 / 法向量估计 / 地面(可站立面)提取 / 静态代价层.

对应 dddmr_perception_3d 中 static layer 的职责:
把一张 PCD 地图变成"带代价的可通行节点集合", 供全局规划器建图搜索.

优化版要点 (详见 OPTIMIZATION_REPORT.md):
  * 体素降采样走一维哈希键 + bincount
  * 2.5D 分层预筛, 法向量只算在候选点上
  * 净空/膨胀两处的定半径查询改成内存有界的定长 kNN, 不再爆内存
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import PlannerConfig
from .fastops import (estimate_normals, first_match_in_radius, layer_bottom_mask,
                      voxel_downsample)


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


def build_ground_map(xyz: np.ndarray, cfg: Optional[PlannerConfig] = None,
                     verbose: bool = False) -> GroundMap:
    """把原始点云处理成 GroundMap (可通行节点 + 静态代价)."""
    cfg = cfg or PlannerConfig()
    fdt = np.float32 if cfg.dtype_32bit else np.float64
    n_in = len(xyz)

    pts = voxel_downsample(xyz, cfg.voxel_size, chunk=cfg.chunk_size, dtype=fdt)
    if verbose:
        print(f"[perception] 降采样: {n_in} -> {len(pts)} 点 (voxel={cfg.voxel_size} m)")

    pts64 = np.asarray(pts, dtype=np.float64)   # KD-tree 内部就是 float64, 复用一份
    tree_all = cKDTree(pts64)

    # ---------------- 1) 可站立面候选 ----------------
    max_slope = np.deg2rad(cfg.max_slope_deg)
    if cfg.fast_surface_prefilter:
        # 只有"每个 XY 柱体内每一层的最低点"才可能站得住人.
        # 其余点(墙面/家具侧面/天花板)直接归为障碍, 无需求法向量.
        cand = np.flatnonzero(layer_bottom_mask(pts64, cfg.voxel_size, cfg.layer_gap,
                                                band=cfg.max_step))
        nrm_c, res_c = estimate_normals(pts64, k=cfg.normal_k, radius=cfg.normal_radius,
                                        chunk=cfg.chunk_size, query_idx=cand,
                                        tree=tree_all, workers=cfg.workers)
        slope_c = np.arccos(np.clip(np.abs(nrm_c[:, 2]), 0.0, 1.0))
        keep = (slope_c <= max_slope) & (res_c <= cfg.max_roughness)
        surface = np.zeros(len(pts64), dtype=bool)
        surface[cand[keep]] = True
        node_idx = cand[keep]
        node_normals = nrm_c[keep]
        node_slope = slope_c[keep]
        node_rough = res_c[keep]
        if verbose:
            print(f"[perception] 2.5D 预筛: {len(pts64)} -> {len(cand)} 个候选 "
                  f"({100.0 * len(cand) / max(len(pts64), 1):.1f}%), 只对候选算法向量")
        del nrm_c, res_c, slope_c
    else:
        normals, residual = estimate_normals(pts64, k=cfg.normal_k, radius=cfg.normal_radius,
                                             chunk=cfg.chunk_size, tree=tree_all,
                                             workers=cfg.workers)
        slope = np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0))
        surface = (slope <= max_slope) & (residual <= cfg.max_roughness)
        node_idx = np.flatnonzero(surface)
        node_normals = normals[node_idx]
        node_slope = slope[node_idx]
        node_rough = residual[node_idx]
        del normals, residual, slope

    if not surface.any():
        raise RuntimeError("未找到任何可站立面, 请放宽 max_slope_deg / max_roughness")

    nodes = pts64[node_idx]
    obstacles = pts64[~surface]
    n_nodes = len(nodes)
    if verbose:
        print(f"[perception] 可站立面候选 {n_nodes} 个, 障碍点 {len(obstacles)} 个")

    # ---------------- 2) 净空 (clearance) ----------------
    # 机器人本体圆柱内, 是否存在"高于脚下 max_step 但低于车高"的点.
    # 注意: 判据本质是存在性 (clearance < robot_height), 定长 kNN 给出的
    #      最近命中点即可给出与暴力搜索一致的结论.
    clearance = np.full(n_nodes, np.inf)
    _, hit_dz = first_match_in_radius(
        pts64, nodes, node_normals, cfg.robot_radius,
        dz_lo=cfg.max_step, dz_hi=cfg.robot_height,
        k0=cfg.neighbor_k0, kmax=cfg.neighbor_kmax, chunk=cfg.chunk_size,
        workers=cfg.workers)
    low_ceiling = ~np.isnan(hit_dz)
    clearance[low_ceiling] = hit_dz[low_ceiling]
    del hit_dz

    # ---------------- 3) 静态代价: 身体高度带内的障碍 -> 膨胀 ----------------
    cost = np.zeros(n_nodes, dtype=np.float64)
    lethal = np.zeros(n_nodes, dtype=bool)
    if len(obstacles):
        nearest, _ = first_match_in_radius(
            obstacles, nodes, node_normals, cfg.inflation_radius,
            dz_lo=cfg.max_step, dz_hi=cfg.robot_height,
            k0=cfg.neighbor_k0, kmax=cfg.neighbor_kmax, chunk=cfg.chunk_size,
            workers=cfg.workers)
        inside = nearest <= cfg.inflation_radius
        lethal |= nearest <= cfg.robot_radius
        decay = (cfg.lethal_cost - 1.0) * np.exp(
            -cfg.cost_scaling_factor * np.clip(nearest - cfg.robot_radius, 0, None))
        cost = np.where(inside, decay, 0.0)
        del nearest, decay, inside

    # ---------------- 4) 净空不足(钻不过去) -> 致命 ----------------
    lethal |= low_ceiling
    cost = np.where(lethal, cfg.lethal_cost, cost)

    if verbose:
        print(f"[perception] 致命节点 {int(lethal.sum())} 个 "
              f"(其中本体碰撞/低净空 {int(low_ceiling.sum())} 个), "
              f"可通行率 {100.0 * (~lethal).mean():.1f}%")

    return GroundMap(
        nodes=nodes,
        normals=np.ascontiguousarray(node_normals, dtype=fdt),
        slope=np.ascontiguousarray(node_slope, dtype=fdt),
        roughness=np.ascontiguousarray(node_rough, dtype=fdt),
        clearance=np.ascontiguousarray(clearance, dtype=fdt),
        cost=np.ascontiguousarray(cost, dtype=fdt),
        lethal=lethal,
        obstacles=np.ascontiguousarray(obstacles, dtype=fdt),
        kdtree=cKDTree(nodes), config=cfg,
    )
