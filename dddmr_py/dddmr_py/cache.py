"""地图预处理结果的磁盘缓存.

感知 + 建图是纯函数: (点云文件内容, 影响地图的参数) -> (GroundMap, NavGraph).
把结果存成 npz, 第二次起直接加载, 对"反复改起终点做规划"的调试流程
是最直接的一笔提速 (实测秒级 -> 毫秒级).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import PlannerConfig
from .graph import NavGraph
from .perception import GroundMap

_CACHE_VERSION = 2


def _file_signature(path: str) -> str:
    """用 (大小, mtime, 头尾各 64KB 的哈希) 作为文件指纹, 不必读全文."""
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
        if st.st_size > 131072:
            fh.seek(-65536, os.SEEK_END)
            h.update(fh.read(65536))
    return h.hexdigest()[:16]


def cache_path(map_path: str, cfg: PlannerConfig) -> Optional[str]:
    if not cfg.cache_dir:
        return None
    os.makedirs(cfg.cache_dir, exist_ok=True)
    try:
        sig = _file_signature(map_path)
    except OSError:
        return None
    name = os.path.splitext(os.path.basename(map_path))[0]
    return os.path.join(cfg.cache_dir,
                        f"{name}.{sig}.{cfg.map_fingerprint()}.v{_CACHE_VERSION}.npz")


def save(path: str, gmap: GroundMap, graph: NavGraph) -> None:
    # 注意: np.savez 对不以 .npz 结尾的路径会自动追加后缀, 所以临时名也要带 .npz
    tmp = path + ".tmp.npz"
    np.savez(tmp,
             nodes=gmap.nodes, normals=gmap.normals, slope=gmap.slope,
             roughness=gmap.roughness, clearance=gmap.clearance, cost=gmap.cost,
             lethal=gmap.lethal, obstacles=gmap.obstacles,
             indptr=graph.indptr, indices=graph.indices, weights=graph.weights,
             component=(graph.component if graph.component is not None
                        else np.zeros(0, np.int32)))
    os.replace(tmp, path)


def load(path: str, cfg: PlannerConfig) -> Optional[Tuple[GroundMap, NavGraph]]:
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        nodes = z["nodes"]
        tree = cKDTree(nodes)
        gmap = GroundMap(nodes=nodes, normals=z["normals"], slope=z["slope"],
                         roughness=z["roughness"], clearance=z["clearance"],
                         cost=z["cost"], lethal=z["lethal"], obstacles=z["obstacles"],
                         kdtree=tree, config=cfg)
        comp = z["component"]
        graph = NavGraph(indptr=z["indptr"], indices=z["indices"], weights=z["weights"],
                         nodes=nodes, lethal=gmap.lethal, cost=gmap.cost, kdtree=tree,
                         config=cfg, component=comp if len(comp) else None)
        return gmap, graph
    except Exception:
        return None
