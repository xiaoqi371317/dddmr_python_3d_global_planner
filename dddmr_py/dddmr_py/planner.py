"""全局规划器: PCD 地图 -> 可通行图 -> 3D 路径.

用法::

    from dddmr_py import GlobalPlanner3D, PlannerConfig

    planner = GlobalPlanner3D.from_pcd("map.pcd", PlannerConfig(robot_radius=0.35))
    path = planner.make_plan(start=(0, 0, 0), goal=(12, 3, 3.0))
    print(path.length, path.poses)

优化版改动:
  * 预处理结果可落盘缓存 (cache_dir), 二次加载毫秒级
  * 拉直/平滑的 KD-tree 查询全部批量化 (原版在 Python 双重循环里逐点查)
  * 不再长期持有原始点云 (原版 self.cloud 对 200 万点地图白占 48 MB)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import cache as _cache
from .config import PlannerConfig
from .graph import NavGraph, build_graph
from .pcd_io import read_pcd
from .perception import GroundMap, build_ground_map
from .search import ALTHeuristic, SearchResult, astar, bidirectional_astar

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
        """(N,4) 的 [qx,qy,qz,qw]: 机体 z 轴对齐地面法向量, 航向为 yaw (全向量化)."""
        n = np.asarray(self.normals, dtype=np.float64)
        if len(n) == 0:
            return np.empty((0, 4))
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        yaw = np.asarray(self.yaw, dtype=np.float64)
        fwd = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))])
        fwd -= n * np.einsum("ij,ij->i", fwd, n)[:, None]
        bad = np.linalg.norm(fwd, axis=1) < 1e-6
        fwd[bad] = np.array([1.0, 0.0, 0.0])
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
        left = np.cross(n, fwd)
        R = np.stack([fwd, left, n], axis=2)          # (N,3,3), 列为基向量
        return _mat_to_quat_batch(R)

    def to_csv(self, path: str) -> str:
        arr = np.column_stack([self.points, self.yaw])
        np.savetxt(path, arr, delimiter=",", header="x,y,z,yaw", comments="", fmt="%.6f")
        return path

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "success": self.success, "message": self.message,
            "length": self.length, "climb": self.climb, "cost": self.cost,
            "expanded": self.expanded, "planning_time": self.planning_time,
            "poses": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": float(y)}
                      for p, y in zip(self.points, self.yaw)],
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


def _mat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    """批量旋转矩阵 -> 四元数 (Shepperd 四分支法, 向量化)."""
    n = len(R)
    out = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = tr > 0
    c1 = (~c0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    c2 = (~c0) & (~c1) & (R[:, 1, 1] > R[:, 2, 2])
    c3 = ~(c0 | c1 | c2)
    for mask, which in ((c0, 0), (c1, 1), (c2, 2), (c3, 3)):
        if not mask.any():
            continue
        M = R[mask]
        if which == 0:
            s = np.sqrt(tr[mask] + 1.0) * 2
            qw, qx = 0.25 * s, (M[:, 2, 1] - M[:, 1, 2]) / s
            qy, qz = (M[:, 0, 2] - M[:, 2, 0]) / s, (M[:, 1, 0] - M[:, 0, 1]) / s
        elif which == 1:
            s = np.sqrt(1.0 + M[:, 0, 0] - M[:, 1, 1] - M[:, 2, 2]) * 2
            qw, qx = (M[:, 2, 1] - M[:, 1, 2]) / s, 0.25 * s
            qy, qz = (M[:, 0, 1] + M[:, 1, 0]) / s, (M[:, 0, 2] + M[:, 2, 0]) / s
        elif which == 2:
            s = np.sqrt(1.0 + M[:, 1, 1] - M[:, 0, 0] - M[:, 2, 2]) * 2
            qw, qx = (M[:, 0, 2] - M[:, 2, 0]) / s, (M[:, 0, 1] + M[:, 1, 0]) / s
            qy, qz = 0.25 * s, (M[:, 1, 2] + M[:, 2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[:, 2, 2] - M[:, 0, 0] - M[:, 1, 1]) * 2
            qw, qx = (M[:, 1, 0] - M[:, 0, 1]) / s, (M[:, 0, 2] + M[:, 2, 0]) / s
            qy, qz = (M[:, 1, 2] + M[:, 2, 1]) / s, 0.25 * s
        out[mask] = np.column_stack([qx, qy, qz, qw])
    return out


class GlobalPlanner3D:
    """3D 全局规划器 (dddmr_global_planner 的纯 Python 复现)."""

    def __init__(self, cloud: np.ndarray, config: Optional[PlannerConfig] = None,
                 verbose: bool = False, _prebuilt=None):
        self.config = config or PlannerConfig()
        self.verbose = verbose
        self.n_cloud_points = 0 if cloud is None else len(cloud)
        self.alt: Optional[ALTHeuristic] = None
        # numba 首次编译约 3.8s (之后走磁盘缓存 ~0.2s). 丢到后台线程去,
        # 与感知/建图并行, 用户感知不到这段编译时间.
        if self.config.search_backend in ("auto", "numba"):
            import threading
            from .search import HAVE_NUMBA, _warmup_numba
            if HAVE_NUMBA:
                threading.Thread(target=_warmup_numba, daemon=True).start()
        t0 = time.time()
        if _prebuilt is not None:
            self.ground, self.graph = _prebuilt
        else:
            self.ground: GroundMap = build_ground_map(cloud, self.config, verbose)
            self.graph: NavGraph = build_graph(self.ground, self.config, verbose)
        self.build_time = time.time() - t0
        # 原始点云不再长期持有 (原版 self.cloud 对大图是纯浪费)
        if self.config.alt_landmarks > 0:
            self.alt = ALTHeuristic.build(self.graph, self.config.alt_landmarks, verbose)
        if verbose:
            print(f"[planner] 地图预处理完成, 用时 {self.build_time:.2f}s")

    # ------------------------------------------------------------------
    @classmethod
    def from_pcd(cls, pcd_path: str, config: Optional[PlannerConfig] = None,
                 verbose: bool = False) -> "GlobalPlanner3D":
        cfg = config or PlannerConfig()
        cpath = _cache.cache_path(pcd_path, cfg)
        if cpath:
            hit = _cache.load(cpath, cfg)
            if hit is not None:
                if verbose:
                    print(f"[planner] 命中预处理缓存 {cpath}")
                return cls(None, cfg, verbose, _prebuilt=hit)
        dt = np.float32 if cfg.dtype_32bit else np.float64
        cloud = read_pcd(pcd_path, dtype=dt)
        if verbose:
            print(f"[planner] 读取 {pcd_path}: {len(cloud)} 点 ({cloud.xyz.dtype})")
        obj = cls(cloud.xyz, cfg, verbose)
        del cloud
        if cpath:
            try:
                _cache.save(cpath, obj.ground, obj.graph)
                if verbose:
                    print(f"[planner] 预处理结果已缓存到 {cpath}")
            except OSError:
                pass
        return obj

    # ------------------------------------------------------------------
    def snap(self, point: Point, radius: Optional[float] = None) -> int:
        """把任意 3D 点吸附到最近的可通行节点, 返回节点索引; 失败返回 -1.

        原版用 query_ball_point 取回半径内全部点再筛; 这里用定长 kNN,
        并按"距离 + 代价"打分, 结果一致但不构造 Python list.
        """
        cfg = self.config
        radius = cfg.snap_radius if radius is None else radius
        p = np.asarray(point, dtype=np.float64).reshape(1, 3)
        k = min(64, len(self.ground.nodes))
        while True:
            d, idx = self.ground.kdtree.query(p, k=k, distance_upper_bound=radius,
                                              workers=cfg.workers)
            d = np.atleast_2d(d)[0]
            idx = np.atleast_1d(np.atleast_2d(idx)[0])
            valid = np.isfinite(d) & (idx < len(self.ground.nodes))
            d, idx = d[valid], idx[valid]
            if len(idx) == 0:
                return -1
            free = ~self.ground.lethal[idx]
            if free.any():
                d, idx = d[free], idx[free]
                score = d + 0.01 * np.asarray(self.ground.cost[idx], dtype=np.float64)
                return int(idx[int(np.argmin(score))])
            if len(idx) < k or k >= len(self.ground.nodes):
                return -1                       # 半径内全是致命节点
            k = min(k * 4, len(self.ground.nodes))

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

        if self.config.bidirectional and self.alt is None:
            res: SearchResult = bidirectional_astar(self.graph, s, g, heuristic_weight)
        else:
            res = astar(self.graph, s, g, heuristic_weight, alt=self.alt)
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
        return Path(points=np.empty((0, 3)), yaw=np.empty(0), normals=np.empty((0, 3)),
                    cost=math.inf, expanded=expanded, planning_time=time.time() - t0,
                    success=False, message=msg)

    # ------------------------------------------------------------------
    def _segments_feasible(self, p: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """**批量**判断 p -> qs[j] 的直线段是否贴地且无碰撞.

        原版 ``_segment_feasible`` 每次只查一条线段, 在 string-pulling 里被
        调用 O(路径点数 x lookahead) 次, 每次还带 workers=-1 的线程池启动开销.
        这里把一个 i 的所有候选 j 合并成一次 KD 查询.
        """
        cfg = self.config
        d = qs - p
        dist = np.linalg.norm(d, axis=1)
        d_xy = np.linalg.norm(d[:, :2], axis=1)
        ok = np.abs(d[:, 2]) <= np.maximum(
            np.tan(np.deg2rad(cfg.max_slope_deg)) * d_xy, cfg.max_step)
        ok &= dist > 1e-9
        if not ok.any():
            return ok
        step = 0.5 * cfg.voxel_size
        n_s = int(max(dist[ok].max() / step, 1)) + 1
        t = np.linspace(0.0, 1.0, n_s)                       # (n_s,)
        cand = np.flatnonzero(ok)
        samples = (p[None, None, :] + t[None, :, None] * d[cand][:, None, :])
        flat = samples.reshape(-1, 3)
        tol = max(1.5 * cfg.voxel_size, cfg.voxel_size + cfg.max_step)
        dd, ii = self.ground.kdtree.query(flat, k=1, distance_upper_bound=tol * 4,
                                          workers=cfg.workers)
        dd = dd.reshape(len(cand), n_s)
        ii = np.minimum(ii.reshape(len(cand), n_s), len(self.ground.nodes) - 1)
        good = (dd <= tol).all(axis=1) & (~self.ground.lethal[ii]).all(axis=1)
        ok[cand] = good
        return ok

    def _segment_feasible(self, p: np.ndarray, q: np.ndarray) -> bool:
        """单段版本 (保持原版 API 兼容)."""
        return bool(self._segments_feasible(np.asarray(p, float),
                                            np.asarray(q, float).reshape(1, 3))[0])

    def _shortcut(self, pts: np.ndarray, lookahead: int = 60) -> np.ndarray:
        """拉直(string pulling): 贪心跳过中间点, 每步只做一次批量可行性检查."""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        i, n = 0, len(pts)
        while i < n - 1:
            hi = min(i + lookahead, n - 1)
            cand = pts[i + 2:hi + 1]
            j = i + 1
            if len(cand):
                feas = self._segments_feasible(pts[i], cand)
                hits = np.flatnonzero(feas)
                if len(hits):
                    j = i + 2 + int(hits[-1])      # 取能跳到的最远点
            out.append(pts[j])
            i = j
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
        targets = np.append(np.arange(0.0, total, step), total)
        return np.column_stack([np.interp(targets, s, pts[:, k]) for k in range(3)])

    def _smooth(self, pts: np.ndarray) -> np.ndarray:
        """带约束的路径平滑.

        原版在 (迭代 x 路径点) 的双重 Python 循环里逐点做 KD 查询;
        这里改成 Jacobi 式更新: 每轮算出全部候选点后做**一次**批量查询.
        """
        cfg = self.config
        if len(pts) < 3 or cfg.smooth_iterations <= 0:
            return pts
        new = pts.copy()
        tol = max(1.5 * cfg.voxel_size, cfg.max_step)
        for _ in range(cfg.smooth_iterations):
            cand = new.copy()
            interior = cand[1:-1]
            interior = interior + cfg.smooth_weight_data * (pts[1:-1] - interior)
            interior = interior + cfg.smooth_weight_smooth * (
                new[:-2] + new[2:] - 2.0 * interior)
            d, idx = self.ground.kdtree.query(interior, k=1, workers=cfg.workers)
            idx = np.minimum(idx, len(self.ground.nodes) - 1)
            accept = (d <= tol) & (~self.ground.lethal[idx])
            new[1:-1] = np.where(accept[:, None], interior, new[1:-1])
        return new

    def _project_to_ground(self, pts: np.ndarray) -> np.ndarray:
        """把路径点投影回可通行面 (用最近若干节点的加权高度)."""
        if len(pts) == 0:
            return pts
        k = min(4, len(self.ground.nodes))
        d, idx = self.ground.kdtree.query(pts, k=k, workers=self.config.workers)
        d = np.atleast_2d(d)
        idx = np.atleast_2d(idx)
        w = 1.0 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        out = pts.copy()
        out[:, 2] = (self.ground.nodes[idx][:, :, 2] * w).sum(axis=1)
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
        _, idx = self.ground.kdtree.query(pts, k=1, workers=self.config.workers)
        return self.ground.normals[np.atleast_1d(idx)]

    # ------------------------------------------------------------------
    def plan_through(self, waypoints: Iterable[Point]) -> List[Path]:
        """多点巡逻: 依次规划相邻航点之间的路径."""
        wp = [np.asarray(p, dtype=np.float64) for p in waypoints]
        return [self.make_plan(wp[i], wp[i + 1]) for i in range(len(wp) - 1)]

    def stats(self) -> dict:
        gm = self.ground
        return {
            "cloud_points": int(self.n_cloud_points),
            "nodes": int(len(gm.nodes)),
            "free_nodes": int((~gm.lethal).sum()),
            "lethal_nodes": int(gm.lethal.sum()),
            "obstacle_points": int(len(gm.obstacles)),
            "edges": int(self.graph.n_edges),
            "graph_mb": round(self.graph.nbytes() / 1e6, 2),
            "build_time_s": round(self.build_time, 3),
            "z_range": [float(gm.nodes[:, 2].min()), float(gm.nodes[:, 2].max())],
        }
