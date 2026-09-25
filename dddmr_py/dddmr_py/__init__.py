"""dddmr_py (fast) — dddmr 3D 导航栈(全局规划部分)的纯 Python 复现 · 性能优化版.

输入一张 PCD 点云地图, 发布 3D 导航目标点, 直接得到一条 3D 路径.
不依赖 ROS, 只需要 numpy + scipy (numba 可选, 装了自动启用).

与上游 API 完全兼容, 可直接替换 import.
"""

from .config import PlannerConfig
from .graph import NavGraph, build_graph
from .mapgen import make_demo_map
from .pcd_io import PointCloud, iter_pcd_chunks, read_pcd, write_pcd
from .perception import GroundMap, build_ground_map
from .fastops import estimate_normals, voxel_downsample
from .planner import GlobalPlanner3D, Path
from .search import (ALTHeuristic, astar, bidirectional_astar, dijkstra,
                     reachable_set, HAVE_NUMBA)

__version__ = "2.0.0"

__all__ = [
    "PlannerConfig", "PointCloud", "read_pcd", "write_pcd", "iter_pcd_chunks",
    "GroundMap", "build_ground_map", "estimate_normals", "voxel_downsample",
    "NavGraph", "build_graph", "astar", "bidirectional_astar", "dijkstra",
    "reachable_set", "ALTHeuristic", "HAVE_NUMBA",
    "GlobalPlanner3D", "Path", "make_demo_map", "__version__",
]
