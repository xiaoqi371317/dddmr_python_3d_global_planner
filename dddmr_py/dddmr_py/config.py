"""规划器参数配置 (对应 dddmr 中 perception_3d / global_planner 的 yaml 参数)."""

from __future__ import annotations

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
    max_neighbors: int = 12             # 每个节点最多保留的邻居数

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

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
