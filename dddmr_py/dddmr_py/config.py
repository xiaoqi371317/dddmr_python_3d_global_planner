"""规划器参数配置 (对应 dddmr 中 perception_3d / global_planner 的 yaml 参数).

相对原版新增了 [性能] 分组的参数, 全部有安全默认值;
把 ``fast_*`` 全部关掉即可退化回原版语义 (见 ``PlannerConfig.exact()``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


@dataclass
class PlannerConfig:
    # ---------------- 机器人本体 ----------------
    robot_radius: float = 0.35          # 机器人外接圆半径 (m)
    robot_height: float = 1.20          # 机器人高度, 用于净空(过桥/桌下/多层)判断 (m)
    max_slope_deg: float = 25.0         # 可通行最大坡度 (deg)
    max_step: float = 0.25              # 可跨越最大台阶高度 (m)
    max_roughness: float = 0.10         # 局部平面拟合残差上限 (m), 越大越颠簸

    # ---------------- 地图预处理 ----------------
    voxel_size: float = 0.10            # 体素降采样边长 (m); 决定图节点密度
    normal_k: int = 16                  # 法向量估计的近邻点数
    normal_radius: float = 0.0          # >0 时改用半径近邻估计法向量 (m)
    layer_gap: float = 0.40             # 同一 XY 栅格内区分不同楼层/结构层的高差 (m)

    # ---------------- 代价地图 (静态层) ----------------
    inflation_radius: float = 0.55      # 障碍物膨胀半径 (m), 应 > robot_radius
    cost_scaling_factor: float = 3.0    # 膨胀代价指数衰减系数, 越大衰减越快
    lethal_cost: float = 254.0          # 致命代价值
    unknown_cost: float = 0.0           # 未知区域代价

    # ---------------- 图构建 ----------------
    connection_radius: float = 2.2      # 邻接半径 (m); <=0 时自动取 2.6*voxel_size
    #  ^^ 与上游保持一致. 原版会把半径内 ~530 个候选全部枚举再截断(97.7% 浪费);
    #     优化版改用 kNN, 半径只作为上界, 大半径几乎不增加开销, 故保留语义.
    max_neighbors: int = 12             # 每个节点最多保留的邻居数 (严格生效)
    neighbor_oversample: int = 3        # kNN 过采样倍数, 抵消坡度/台阶过滤造成的邻居损失
    edge_collision_check: bool = True   # 对长边沿线做碰撞校验(原版只查端点, 会穿墙)

    # ---------------- 搜索 ----------------
    heuristic_weight: float = 1.0       # 1.0=标准A*, 0=Dijkstra, >1=加权A*(更快但次优)
    cost_weight: float = 0.02           # 代价对边权的影响: w = len*(1 + cost_weight*cost)
    snap_radius: float = 1.5            # 起终点吸附到最近可通行节点的搜索半径 (m)

    # ---------------- 路径后处理 ----------------
    path_resolution: float = 0.10       # 输出路径重采样间隔 (m)
    smooth_iterations: int = 3          # 路径平滑(带约束的梯度下降)迭代次数
    smooth_weight_data: float = 0.35    # 平滑: 贴近原路径的权重
    smooth_weight_smooth: float = 0.35  # 平滑: 曲线光滑的权重
    shortcut: bool = True               # 是否做拉直(string-pulling)剪枝

    # ================ [性能] 以下为优化版新增 ================
    dtype_32bit: bool = True            # 点云/代价用 float32, 图索引用 int32 -> 内存减半
    fast_surface_prefilter: bool = False  # 2.5D 分层预筛(实验性): 只对"每层底部附近"
    #  ^^ 默认关闭. 实测只省掉约 6% 的感知耗时, 却会在多层/楼梯结构上丢节点
    #     (building.pcd voxel=0.2 下丢了 23% 的可通行节点). 收益与风险不匹配.
    #     地图是单层平坦场景时可以打开.
    neighbor_k0: int = 24               # 定半径归约的初始 k (自适应升档)
    neighbor_kmax: int = 512            # 定半径归约的 k 上限
    chunk_size: int = 200_000           # 分块处理粒度, 控制峰值内存
    search_backend: str = "auto"        # auto | numba | python | scipy
    bidirectional: bool = False         # 双向 A* (无 ALT 时对长距离更快)
    alt_landmarks: int = 0              # ALT 地标数, 0=关闭; 8~16 对迷宫型地图收益大
    prune_components: bool = True       # 用连通分量预判不可达, 避免搜索整张图
    cache_dir: str = ""                 # 预处理缓存目录, 空=不缓存 (推荐 ".dddmr_cache")
    workers: int = -1                   # KD-tree 并行线程数, -1=全部核心

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @classmethod
    def compat(cls, **kw) -> "PlannerConfig":
        """与上游语义逐项对齐的配置: 关闭会改变可达性的安全增强,
        只保留纯实现级提速. 用于 A/B 对比或需要复刻上游结果时."""
        base = dict(edge_collision_check=False,
                    dtype_32bit=False, neighbor_k0=64, neighbor_kmax=4096)
        base.update(kw)
        return cls(**base)

    exact = compat   # 旧名保留

    @classmethod
    def from_yaml(cls, path: str) -> "PlannerConfig":
        """读取 yaml (无 pyyaml 时退化为简易 key: value 解析)."""
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(yaml.safe_load(fh) or {})
        except ImportError:
            data: Dict[str, Any] = {}
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.split("#")[0].strip()
                    if not line or ":" not in line:
                        continue
                    k, _, v = line.partition(":")
                    v = v.strip()
                    if not v:
                        continue
                    try:
                        data[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except ValueError:
                        data[k.strip()] = v.lower() in ("true", "yes", "on")
            return cls.from_dict(data)

    # ------------------------------------------------------------------
    def map_fingerprint(self) -> str:
        """只包含影响"地图预处理结果"的参数, 用于缓存键 (搜索/后处理参数变化不失效)."""
        keys = ("robot_radius", "robot_height", "max_slope_deg", "max_step",
                "max_roughness", "voxel_size", "normal_k", "normal_radius", "layer_gap",
                "inflation_radius", "cost_scaling_factor", "lethal_cost",
                "connection_radius", "max_neighbors", "cost_weight",
                "dtype_32bit", "fast_surface_prefilter")
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]
