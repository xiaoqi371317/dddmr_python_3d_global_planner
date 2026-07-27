"""全局规划器: PCD 地图 -> 可通行图 -> 3D 路径.

用法::

    from dddmr_py import GlobalPlanner3D, PlannerConfig

    planner = GlobalPlanner3D.from_pcd("map.pcd", PlannerConfig(robot_radius=0.35))
    path = planner.make_plan(start=(0, 0, 0), goal=(12, 3, 3.0))
    print(path.length, path.poses)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import PlannerConfig
from .graph import NavGraph, build_graph
from .pcd_io import read_pcd
from .perception import GroundMap, build_ground_map
from .search import SearchResult, astar

Point = Sequence[float]


@dataclass
class Path:
    """规划结果. points: (N,3); yaw: (N,); normals: (N,3)."""

    points: np.ndarray
    yaw: np.ndarray
    normals: np.ndarray
    cost: float = 0.0
    expanded: int = 0
    planning_time: float = 0.0
    success: bool = True
    message: str = ""
    raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def climb(self) -> float:
        """累计爬升高度 (m)."""
        if len(self.points) < 2:
            return 0.0
        dz = np.diff(self.points[:, 2])
        return float(dz[dz > 0].sum())

    @property
    def poses(self) -> np.ndarray:
        """(N,4) 的 [x, y, z, yaw]."""
        return np.column_stack([self.points, self.yaw])

    def quaternions(self) -> np.ndarray:
        """(N,4) 的 [qx,qy,qz,qw]: 机体 z 轴对齐地面法向量, 航向为 yaw."""
        out = np.zeros((len(self.points), 4))
        for i, (n, yaw) in enumerate(zip(self.normals, self.yaw)):
            n = n / (np.linalg.norm(n) + 1e-12)
            # 由 yaw 得到期望前向, 再用法向量正交化, 组成旋转矩阵
            fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            fwd = fwd - n * float(np.dot(fwd, n))
            if np.linalg.norm(fwd) < 1e-6:
                fwd = np.array([1.0, 0.0, 0.0])
            fwd /= np.linalg.norm(fwd)
            left = np.cross(n, fwd)
            R = np.column_stack([fwd, left, n])
            out[i] = _mat_to_quat(R)
        return out

    def to_csv(self, path: str) -> str:
        arr = np.column_stack([self.points, self.yaw])
        header = "x,y,z,yaw"
        np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.6f")
        return path

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "success": self.success,
            "message": self.message,
            "length": self.length,
            "climb": self.climb,
            "cost": self.cost,
            "expanded": self.expanded,
            "planning_time": self.planning_time,
            "poses": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": float(y)}
                for p, y in zip(self.points, self.yaw)
            ],
        }
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<Path success={self.success} pts={len(self.points)} "
                f"len={self.length:.2f}m climb={self.climb:.2f}m "
                f"t={self.planning_time * 1000:.0f}ms>")


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([qx, qy, qz, qw])


class GlobalPlanner3D:
    """3D 全局规划器 (dddmr_global_planner 的纯 Python 复现)."""

    def __init__(self, cloud: np.ndarray, config: Optional[PlannerConfig] = None,
                 verbose: bool = False):
        self.config = config or PlannerConfig()
        self.verbose = verbose
        self.cloud = np.asarray(cloud, dtype=np.float64)
        t0 = time.time()
        self.ground: GroundMap = build_ground_map(self.cloud, self.config, verbose)
        self.graph: NavGraph = build_graph(self.ground, self.config, verbose)
        self.build_time = time.time() - t0
        if verbose:
            print(f"[planner] 地图预处理完成, 用时 {self.build_time:.2f}s")

    # ------------------------------------------------------------------
    @classmethod
    def from_pcd(cls, pcd_path: str, config: Optional[PlannerConfig] = None,
                 verbose: bool = False) -> "GlobalPlanner3D":
        cloud = read_pcd(pcd_path)
        if verbose:
            print(f"[planner] 读取 {pcd_path}: {len(cloud)} 点")
        return cls(cloud.xyz, config, verbose)

    # ------------------------------------------------------------------
    def snap(self, point: Point, radius: Optional[float] = None) -> int:
        """把任意 3D 点吸附到最近的可通行节点, 返回节点索引; 失败返回 -1."""
        radius = self.config.snap_radius if radius is None else radius
        p = np.asarray(point, dtype=np.float64).reshape(3)
        idx = self.ground.kdtree.query_ball_point(p, radius, workers=-1)
        if not idx:
            return -1
        idx = np.asarray(idx, dtype=np.int64)
        free = idx[~self.ground.lethal[idx]]
        if len(free) == 0:
            return -1
        # 同等距离下优先选代价低的节点
        d = np.linalg.norm(self.ground.nodes[free] - p, axis=1)
        score = d + 0.01 * self.ground.cost[free]
        return int(free[int(np.argmin(score))])

    # ------------------------------------------------------------------
    def make_plan(self, start: Point, goal: Point,
                  heuristic_weight: Optional[float] = None) -> Path:
        """规划从 start 到 goal 的 3D 路径."""
        t0 = time.time()
        s = self.snap(start)
        g = self.snap(goal)
        if s < 0 or g < 0:
            who = "起点" if s < 0 else "终点"
            return self._fail(f"{who}附近 {self.config.snap_radius} m 内没有可通行节点", t0)

        res: SearchResult = astar(self.graph, s, g, heuristic_weight)
        if not res.success:
            return self._fail(res.reason, t0, expanded=res.expanded)

        raw = self.ground.nodes[np.asarray(res.path_idx, dtype=np.int64)]
        pts = raw.copy()
        if self.config.shortcut:
            pts = self._shortcut(pts)
        pts = self._resample(pts, self.config.path_resolution)
        pts = self._smooth(pts)
        pts = self._project_to_ground(pts)
        yaw = self._compute_yaw(pts)
        normals = self._surface_normals(pts)

        return Path(points=pts, yaw=yaw, normals=normals, cost=res.cost,
                    expanded=res.expanded, planning_time=time.time() - t0,
                    success=True, message="ok", raw_points=raw)

    def _fail(self, msg: str, t0: float, expanded: int = 0) -> Path:
        if self.verbose:
            print(f"[planner] 规划失败: {msg}")
        return Path(points=np.empty((0, 3)), yaw=np.empty(0),
                    normals=np.empty((0, 3)), cost=math.inf, expanded=expanded,
                    planning_time=time.time() - t0, success=False, message=msg)

    # ------------------------------------------------------------------
    def _segment_feasible(self, p: np.ndarray, q: np.ndarray) -> bool:
        """判断两点之间的直线段是否贴着可通行面且无碰撞."""
        cfg = self.config
        d = q - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return True
        d_xy = float(np.linalg.norm(d[:2]))
        if abs(d[2]) > max(np.tan(np.deg2rad(cfg.max_slope_deg)) * d_xy, cfg.max_step):
            return False
        n_samples = max(int(dist / (0.5 * cfg.voxel_size)) + 1, 2)
        samples = p + np.linspace(0.0, 1.0, n_samples)[:, None] * d
        tol = max(1.5 * cfg.voxel_size, cfg.voxel_size + cfg.max_step)
        dists, idx = self.ground.kdtree.query(samples, k=1, workers=-1)
        if np.any(dists > tol):
            return False
        return not np.any(self.ground.lethal[idx])

    def _shortcut(self, pts: np.ndarray, lookahead: int = 60) -> np.ndarray:
        """拉直(string pulling): 贪心跳过中间点. lookahead 限制单次跳跃跨度以控制耗时."""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            j = min(i + lookahead, n - 1)
            advanced = False
            while j > i + 1:
                if self._segment_feasible(pts[i], pts[j]):
                    out.append(pts[j])
                    i = j
                    advanced = True
                    break
                j -= 1
            if not advanced:
                i += 1
                out.append(pts[i])
        return np.asarray(out)

    @staticmethod
    def _resample(pts: np.ndarray, step: float) -> np.ndarray:
        """按弧长等间隔重采样."""
        if len(pts) < 2 or step <= 0:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-9:
            return pts
        targets = np.arange(0.0, total, step)
        targets = np.append(targets, total)
        out = np.column_stack([np.interp(targets, s, pts[:, k]) for k in range(3)])
        return out

    def _smooth(self, pts: np.ndarray) -> np.ndarray:
        """带约束的梯度下降平滑 (保持首尾不动)."""
        cfg = self.config
        if len(pts) < 3 or cfg.smooth_iterations <= 0:
            return pts
        new = pts.copy()
        for _ in range(cfg.smooth_iterations):
            for i in range(1, len(pts) - 1):
                cand = new[i].copy()
                cand += cfg.smooth_weight_data * (pts[i] - cand)
                cand += cfg.smooth_weight_smooth * (new[i - 1] + new[i + 1] - 2.0 * cand)
                d, idx = self.ground.kdtree.query(cand, k=1)
                if d <= max(1.5 * cfg.voxel_size, cfg.max_step) and not self.ground.lethal[idx]:
                    new[i] = cand
        return new

    def _project_to_ground(self, pts: np.ndarray) -> np.ndarray:
        """把路径点投影回可通行面 (用最近若干节点的加权高度)."""
        if len(pts) == 0:
            return pts
        k = min(4, len(self.ground.nodes))
        d, idx = self.ground.kdtree.query(pts, k=k, workers=-1)
        d = np.atleast_2d(d)
        idx = np.atleast_2d(idx)
        w = 1.0 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        z = (self.ground.nodes[idx][:, :, 2] * w).sum(axis=1)
        out = pts.copy()
        out[:, 2] = z
        return out

    @staticmethod
    def _compute_yaw(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.empty(0)
        if len(pts) == 1:
            return np.zeros(1)
        d = np.diff(pts[:, :2], axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        return np.append(yaw, yaw[-1])

    def _surface_normals(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.empty((0, 3))
        _, idx = self.ground.kdtree.query(pts, k=1, workers=-1)
        return self.ground.normals[np.atleast_1d(idx)]

    # ------------------------------------------------------------------
    def plan_through(self, waypoints: Iterable[Point]) -> List[Path]:
        """多点巡逻: 依次规划相邻航点之间的路径."""
        wp = [np.asarray(p, dtype=np.float64) for p in waypoints]
        return [self.make_plan(wp[i], wp[i + 1]) for i in range(len(wp) - 1)]

    def stats(self) -> dict:
        gm = self.ground
        return {
            "cloud_points": int(len(self.cloud)),
            "nodes": int(len(gm.nodes)),
            "free_nodes": int((~gm.lethal).sum()),
            "lethal_nodes": int(gm.lethal.sum()),
            "obstacle_points": int(len(gm.obstacles)),
            "edges": int(self.graph.n_edges),
            "build_time_s": round(self.build_time, 3),
            "z_range": [float(gm.nodes[:, 2].min()), float(gm.nodes[:, 2].max())],
        }
