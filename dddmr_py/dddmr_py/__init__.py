"""dddmr_py — dddmr 3D 导航栈(全局规划部分)的纯 Python 复现.

输入一张 PCD 点云地图, 发布 3D 导航目标点, 直接得到一条 3D 路径.
不依赖 ROS, 只需要 numpy + scipy.
"""

from .config import PlannerConfig
from .graph import NavGraph, build_graph
from .mapgen import make_demo_map
from .pcd_io import PointCloud, read_pcd, write_pcd
from .perception import GroundMap, build_ground_map, estimate_normals, voxel_downsample
from .planner import GlobalPlanner3D, Path
from .search import astar, dijkstra, reachable_set

__version__ = "1.0.0"

__all__ = [
    "PlannerConfig", "PointCloud", "read_pcd", "write_pcd",
    "GroundMap", "build_ground_map", "estimate_normals", "voxel_downsample",
    "NavGraph", "build_graph", "astar", "dijkstra", "reachable_set",
    "GlobalPlanner3D", "Path", "make_demo_map", "__version__",
]
