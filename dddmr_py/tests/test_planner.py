"""自检脚本: 可直接 `python tests/test_planner.py`, 也可用 pytest 运行."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dddmr_py import (GlobalPlanner3D, PlannerConfig, make_demo_map,  # noqa: E402
                      read_pcd, write_pcd)
from dddmr_py.mapgen import UPPER_Z  # noqa: E402


def _planner(cfg=None):
    return GlobalPlanner3D(make_demo_map(), cfg or PlannerConfig())


def test_pcd_roundtrip():
    cloud = make_demo_map()[:5000]
    for binary in (True, False):
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as fh:
            path = fh.name
        write_pcd(path, cloud, binary=binary)
        back = read_pcd(path).xyz
        assert back.shape == cloud.shape, (back.shape, cloud.shape)
        assert np.allclose(back, cloud, atol=1e-3), "PCD 读写不一致"
        os.unlink(path)
    print("✓ PCD 读写 (ascii/binary) 一致")


def test_plan_to_upper_level():
    p = _planner()
    path = p.make_plan((1, 1, 0), (21, 10, UPPER_Z))
    assert path.success, path.message
    assert abs(path.points[-1][2] - UPPER_Z) < 0.3, "终点没有落在二层"
    assert path.climb > 2.5, f"爬升不足: {path.climb}"
    print(f"✓ 上二层: {path.length:.2f} m, 爬升 {path.climb:.2f} m, "
          f"{len(path.points)} 个位姿")
    return p, path


def test_plan_same_xy_different_level():
    """同一个 (x,y), z 不同 -> 应规划到不同楼层, 路径明显不同."""
    p = _planner()
    up = p.make_plan((1, 1, 0), (21, 10, UPPER_Z))
    low = p.make_plan((1, 1, 0), (21, 10, 0.0))
    assert up.success and low.success
    assert up.points[-1][2] > 2.5 and low.points[-1][2] < 0.5
    assert low.climb < 1.0, "一层路径不应该有明显爬升"
    print(f"✓ 3D 目标点区分楼层: 上层 {up.length:.2f} m / 下层 {low.length:.2f} m")


def test_path_is_continuous_and_safe():
    p, path = test_plan_to_upper_level()
    seg = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
    assert seg.max() < 0.5, f"路径存在跳变: {seg.max():.3f} m"

    d, idx = p.ground.kdtree.query(path.points, k=1)
    assert d.max() < 0.4, f"路径偏离可通行面: {d.max():.3f} m"
    assert not p.ground.lethal[idx].any(), "路径穿过了致命(碰撞)区域"

    d_xy = np.linalg.norm(np.diff(path.points[:, :2], axis=0), axis=1)
    d_z = np.abs(np.diff(path.points[:, 2]))
    slope = np.degrees(np.arctan2(d_z, np.maximum(d_xy, 1e-6)))
    # 允许少量离散化噪声超限
    assert np.percentile(slope, 98) < p.config.max_slope_deg + 10, \
        f"路径坡度超限: p98={np.percentile(slope, 98):.1f}°"
    print(f"✓ 路径连续/贴地/无碰撞 (最大步长 {seg.max():.3f} m, "
          f"p98 坡度 {np.percentile(slope, 98):.1f}°)")


def test_low_clearance_is_blocked():
    """桌子下方净空 0.75 m < 机器人高度 -> 不可通行."""
    p = _planner()
    idx = p.snap((2.0, 11.0, 0.0), radius=0.4)
    assert idx < 0, "桌子下方不应该被判为可通行"
    print("✓ 低净空区域(桌下)被正确判为不可通行")


def test_unreachable_goal_reports_failure():
    p = _planner()
    path = p.make_plan((1, 1, 0), (100, 100, 0))
    assert not path.success and "可通行节点" in path.message
    print(f"✓ 非法目标点正确报错: {path.message}")


def test_dijkstra_matches_astar():
    p = _planner()
    a = p.make_plan((1, 1, 0), (14, 3, 0), heuristic_weight=1.0)
    d = p.make_plan((1, 1, 0), (14, 3, 0), heuristic_weight=0.0)
    assert a.success and d.success
    assert abs(a.cost - d.cost) < 1e-6, f"A* 与 Dijkstra 代价不一致: {a.cost} vs {d.cost}"
    assert a.expanded <= d.expanded
    print(f"✓ A* 与 Dijkstra 结果一致 (扩展节点 {a.expanded} vs {d.expanded})")


def test_waypoints():
    p = _planner()
    paths = p.plan_through([(1, 1, 0), (14, 3, 0), (21, 10, UPPER_Z)])
    assert all(x.success for x in paths)
    print(f"✓ 多航点巡逻: {len(paths)} 段, 总长 {sum(x.length for x in paths):.2f} m")


if __name__ == "__main__":
    tests = [test_pcd_roundtrip, test_plan_same_xy_different_level,
             test_path_is_continuous_and_safe, test_low_clearance_is_blocked,
             test_unreachable_goal_reports_failure, test_dijkstra_matches_astar,
             test_waypoints]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"✗ {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 项通过")
    raise SystemExit(1 if failed else 0)
