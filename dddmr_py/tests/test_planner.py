"""优化版单元测试. 运行: python -m pytest tests/ -v  (或 python tests/test_planner.py)"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dddmr_py import (GlobalPlanner3D, PlannerConfig, astar, bidirectional_astar,
                      build_graph, build_ground_map, make_demo_map, read_pcd,
                      voxel_downsample, write_pcd)
from dddmr_py.fastops import estimate_normals, symmetric_eig3


# ---------------------------------------------------------------- 数值内核
def test_symmetric_eig3_matches_lapack():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 3, 3))
    cov = np.matmul(A.transpose(0, 2, 1), A)
    ev_min, vec = symmetric_eig3(cov)
    ref = np.linalg.eigvalsh(cov)[:, 0]
    assert np.abs(ev_min - ref).max() < 1e-9, "最小特征值应与 LAPACK 一致"
    # 特征向量验证: cov @ v ≈ lambda * v
    resid = np.einsum("ijk,ik->ij", cov, vec) - ev_min[:, None] * vec
    assert np.abs(resid).max() < 1e-7, "特征向量残差过大"


def test_voxel_downsample_equivalent_to_reference():
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=3.0, size=(20000, 3))
    v = 0.25
    fast = voxel_downsample(pts, v, dtype=np.float64)
    # 参考实现 (原版做法)
    keys = np.floor(pts / v).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3))
    np.add.at(sums, inv, pts)
    ref = sums / cnt[:, None]
    assert len(fast) == len(ref), f"体素数不一致 {len(fast)} vs {len(ref)}"
    d, _ = cKDTree(ref).query(fast, k=1)
    assert d.max() < 1e-9, "体素质心应与参考实现逐点一致"


def test_voxel_downsample_chunked_matches_whole():
    rng = np.random.default_rng(2)
    pts = rng.normal(scale=2.0, size=(50000, 3))
    a = voxel_downsample(pts, 0.2, chunk=0, dtype=np.float64)
    b = voxel_downsample(pts, 0.2, chunk=7000, dtype=np.float64)
    assert len(a) == len(b)
    d, _ = cKDTree(a).query(b, k=1)
    assert d.max() < 1e-9, "流式分块结果应与整体处理一致"


def test_estimate_normals_on_plane():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5, 5, size=(3000, 2))
    pts = np.column_stack([xy, np.zeros(len(xy))])
    n, res = estimate_normals(pts, k=16)
    assert np.abs(np.abs(n[:, 2]) - 1.0).max() < 1e-6, "平面上法向量应为 ±z"
    assert res.max() < 1e-6, "平面残差应接近 0"


def test_estimate_normals_subset_query():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(2000, 3))
    idx = np.array([5, 100, 777, 1500])
    n_all, r_all = estimate_normals(pts, k=12)
    n_sub, r_sub = estimate_normals(pts, k=12, query_idx=idx)
    assert np.abs(n_sub - n_all[idx]).max() < 1e-9
    assert np.abs(r_sub - r_all[idx]).max() < 1e-9


# ---------------------------------------------------------------- PCD IO
def test_pcd_roundtrip_binary_and_ascii():
    cloud = make_demo_map()[:5000]
    for binary in (True, False):
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as fh:
            path = fh.name
        try:
            write_pcd(path, cloud, binary=binary)
            back = read_pcd(path, dtype=np.float64)
            assert len(back) == len(cloud)
            assert np.abs(back.xyz - cloud).max() < 1e-4, f"binary={binary} 往返失真"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------- 感知 / 图
def _demo_planner(**kw):
    cfg = PlannerConfig(voxel_size=0.15, **kw)
    return GlobalPlanner3D(make_demo_map(), cfg)


def test_ground_map_basic():
    gm = build_ground_map(make_demo_map(), PlannerConfig(voxel_size=0.15))
    assert len(gm.nodes) > 100
    assert gm.normals.shape == (len(gm.nodes), 3)
    assert np.all(gm.normals[:, 2] >= -1e-9), "法向量应统一朝上"
    assert 0.0 < gm.free_ratio <= 1.0


def test_graph_respects_max_neighbors():
    cfg = PlannerConfig(voxel_size=0.15, max_neighbors=8)
    gm = build_ground_map(make_demo_map(), cfg)
    g = build_graph(gm, cfg)
    deg = np.diff(g.indptr)
    # 对称化后一个节点的度可能超过 K (别人把它选为邻居), 但出边必须 <= K
    assert deg.max() <= 8 * 3, "度数异常膨胀"
    assert g.component is not None, "应计算连通分量"


def _edge_arrays(g):
    deg = np.diff(g.indptr)
    return np.repeat(np.arange(len(g.nodes)), deg), g.indices


def test_graph_edges_keep_clearance_from_lethal():
    """核心安全断言.

    优化版给出的保证是: 图中任意一条边上的**任意一点**, 到最近致命节点的
    距离都不小于 0.5*voxel (半个体素, 即致命节点的势力范围).
    这里用比建图时更密的采样去验证这个保证.
    """
    cfg = PlannerConfig(voxel_size=0.15, edge_collision_check=True)
    gm = build_ground_map(make_demo_map(), cfg)
    g = build_graph(gm, cfg)
    src, dst = _edge_arrays(g)
    L = np.linalg.norm(g.nodes[dst] - g.nodes[src], axis=1)
    longe = np.flatnonzero(L > 1.5 * cfg.voxel_size)
    leth = np.flatnonzero(gm.lethal)
    if not len(longe) or not len(leth):
        return
    tree_l = cKDTree(g.nodes[leth])
    tree_f = cKDTree(g.nodes[~gm.lethal])
    t = np.linspace(0, 1, 64)[1:-1]            # 远比建图时(1/4 体素)更密
    worst = np.inf
    for beg in range(0, len(longe), 20000):
        e = longe[beg:beg + 20000]
        p, q = g.nodes[src[e]], g.nodes[dst[e]]
        s = (p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]).reshape(-1, 3)
        dl, _ = tree_l.query(s, k=1, workers=-1)
        df, _ = tree_f.query(s, k=1, workers=-1)
        worst = min(worst, float((dl - df).min()))
    assert worst >= 0.0, (
        f"存在边上的点更靠近致命节点 (d_leth-d_free={worst:.4f} < 0)")


def test_collision_check_removes_wall_piercing_edges():
    """关掉碰撞校验时应能观察到穿障边, 打开后应被清除 —— 证明这道校验有效."""
    gm = build_ground_map(make_demo_map(), PlannerConfig(voxel_size=0.15))
    g_off = build_graph(gm, PlannerConfig(voxel_size=0.15, edge_collision_check=False))
    g_on = build_graph(gm, PlannerConfig(voxel_size=0.15, edge_collision_check=True))
    assert g_on.n_edges < g_off.n_edges, "碰撞校验应当剔除掉一部分边"

    def pierce_count(g):
        src, dst = _edge_arrays(g)
        L = np.linalg.norm(g.nodes[dst] - g.nodes[src], axis=1)
        e = np.flatnonzero(L > 1.5 * 0.15)
        if not len(e):
            return 0
        tree = cKDTree(g.nodes)
        t = np.linspace(0, 1, 32)[1:-1]
        p, q = g.nodes[src[e]], g.nodes[dst[e]]
        s = p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]
        _, ii = tree.query(s.reshape(-1, 3), k=1, workers=-1)
        return int(gm.lethal[ii.reshape(len(e), -1)].any(axis=1).sum())

    off, on = pierce_count(g_off), pierce_count(g_on)
    assert off > 0, "未开校验时本应存在穿障边(用于证明该校验确有必要)"
    assert on == 0, f"开启校验后仍有 {on} 条穿障边"


# ---------------------------------------------------------------- 搜索
def test_astar_backends_agree():
    p = _demo_planner()
    free = np.flatnonzero(~p.ground.lethal)
    s, g = int(free[0]), int(free[len(free) // 2])
    ra = astar(p.graph, s, g, backend="python")
    rb = astar(p.graph, s, g, backend="numba")
    assert ra.success == rb.success
    if ra.success:
        assert abs(ra.cost - rb.cost) < 1e-6, "两个后端代价必须一致"
        assert ra.expanded == rb.expanded, "两个后端扩展数必须一致"


def test_astar_optimal_equals_dijkstra():
    p = _demo_planner()
    free = np.flatnonzero(~p.ground.lethal)
    s, g = int(free[3]), int(free[-4])
    ra = astar(p.graph, s, g, heuristic_weight=1.0)
    rd = astar(p.graph, s, g, heuristic_weight=0.0)
    if ra.success:
        assert abs(ra.cost - rd.cost) < 1e-6, "A*(h可采纳) 应与 Dijkstra 同解"


def test_bidirectional_matches_astar():
    p = _demo_planner()
    free = np.flatnonzero(~p.ground.lethal)
    s, g = int(free[1]), int(free[-2])
    ra = astar(p.graph, s, g)
    rb = bidirectional_astar(p.graph, s, g)
    assert ra.success == rb.success
    if ra.success:
        assert abs(ra.cost - rb.cost) / max(ra.cost, 1e-9) < 1e-6


def test_alt_heuristic_admissible_and_faster():
    from dddmr_py.search import ALTHeuristic
    cfg = PlannerConfig(voxel_size=0.15)
    gm = build_ground_map(make_demo_map(), cfg)
    g = build_graph(gm, cfg)
    alt = ALTHeuristic.build(g, 6)
    free = np.flatnonzero(~gm.lethal)
    s, t = int(free[0]), int(free[-1])
    r_plain = astar(g, s, t)
    r_alt = astar(g, s, t, alt=alt)
    assert r_plain.success == r_alt.success
    if r_plain.success:
        assert abs(r_plain.cost - r_alt.cost) < 1e-6, "ALT 可采纳, 解必须最优"


def test_disconnected_goal_fails_fast():
    p = _demo_planner()
    comp = p.graph.component
    counts = np.bincount(comp)
    big = int(np.argmax(counts))
    small = np.flatnonzero(counts == counts[counts < counts[big]].max()) if (counts < counts[big]).any() else []
    if not len(small):
        return
    s = int(np.flatnonzero(comp == big)[0])
    t = int(np.flatnonzero(comp == int(small[0]))[0])
    r = astar(p.graph, s, t)
    assert not r.success
    assert r.expanded == 0, "不连通时应 O(1) 失败, 而不是搜遍全图"


# ---------------------------------------------------------------- 端到端
def test_make_plan_produces_valid_path():
    p = _demo_planner()
    nodes = p.ground.nodes[~p.ground.lethal]
    ok = False
    for i in range(0, min(len(nodes), 400), 40):
        path = p.make_plan(nodes[0], nodes[-1 - i])
        if path.success:
            ok = True
            assert len(path.points) >= 2
            assert path.length > 0
            assert path.yaw.shape == (len(path.points),)
            assert path.normals.shape == (len(path.points), 3)
            q = path.quaternions()
            assert np.abs(np.linalg.norm(q, axis=1) - 1.0).max() < 1e-6, "四元数应归一"
            step = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
            assert step.max() < 1.0, "重采样后不应有大跳变"
            break
    assert ok, "demo 地图上至少应有一条可行路径"


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        pcd = os.path.join(d, "m.pcd")
        write_pcd(pcd, make_demo_map())
        cfg = PlannerConfig(voxel_size=0.15, cache_dir=os.path.join(d, "cache"))
        p1 = GlobalPlanner3D.from_pcd(pcd, cfg)
        p2 = GlobalPlanner3D.from_pcd(pcd, cfg)          # 应命中缓存
        assert len(p1.ground.nodes) == len(p2.ground.nodes)
        assert np.abs(p1.ground.nodes - p2.ground.nodes).max() < 1e-9
        assert p1.graph.n_edges == p2.graph.n_edges
        assert p2.build_time < max(p1.build_time, 1e-6), "命中缓存应更快"


def test_config_fingerprint_sensitivity():
    a = PlannerConfig(voxel_size=0.1)
    b = PlannerConfig(voxel_size=0.2)
    c = PlannerConfig(voxel_size=0.1, heuristic_weight=5.0)
    assert a.map_fingerprint() != b.map_fingerprint(), "改地图参数应使缓存失效"
    assert a.map_fingerprint() == c.map_fingerprint(), "只改搜索参数不应使缓存失效"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} 通过")
    sys.exit(1 if fails else 0)
