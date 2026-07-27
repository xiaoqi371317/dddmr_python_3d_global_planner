"""命令行入口: python -m dddmr_py <子命令>"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .config import PlannerConfig
from .mapgen import make_demo_map
from .pcd_io import read_pcd, write_pcd
from .planner import GlobalPlanner3D


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="yaml 参数文件")
    p.add_argument("--voxel", type=float, help="体素降采样边长 (m)")
    p.add_argument("--robot-radius", type=float, help="机器人半径 (m)")
    p.add_argument("--robot-height", type=float, help="机器人高度 (m)")
    p.add_argument("--max-slope", type=float, help="最大可通行坡度 (deg)")
    p.add_argument("--max-step", type=float, help="最大可跨越台阶 (m)")
    p.add_argument("--inflation", type=float, help="障碍膨胀半径 (m)")
    p.add_argument("--heuristic-weight", type=float, help="A* 启发式权重, 0=Dijkstra")
    p.add_argument("--snap-radius", type=float, help="起终点吸附半径 (m)")


def _config_from_args(args) -> PlannerConfig:
    cfg = PlannerConfig.from_yaml(args.config) if getattr(args, "config", None) else PlannerConfig()
    mapping = {
        "voxel": "voxel_size", "robot_radius": "robot_radius", "robot_height": "robot_height",
        "max_slope": "max_slope_deg", "max_step": "max_step", "inflation": "inflation_radius",
        "heuristic_weight": "heuristic_weight", "snap_radius": "snap_radius",
    }
    for arg_name, field_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg, field_name, val)
    return cfg


def _load_planner(map_path: str, cfg: PlannerConfig, verbose: bool = True) -> GlobalPlanner3D:
    if map_path in ("demo", "-"):
        print("[cli] 使用内置合成地图 (双层楼+斜坡+门洞+桌子)")
        return GlobalPlanner3D(make_demo_map(), cfg, verbose=verbose)
    return GlobalPlanner3D.from_pcd(map_path, cfg, verbose=verbose)


def cmd_plan(args) -> int:
    cfg = _config_from_args(args)
    planner = _load_planner(args.map, cfg)
    path = planner.make_plan(args.start, args.goal)
    if not path.success:
        print(f"[cli] 规划失败: {path.message}", file=sys.stderr)
        return 2
    print(f"[cli] 规划成功: {len(path.points)} 个位姿, 长度 {path.length:.2f} m, "
          f"爬升 {path.climb:.2f} m, 扩展 {path.expanded} 节点, "
          f"耗时 {path.planning_time * 1000:.0f} ms")
    if args.out:
        path.to_csv(args.out)
        print(f"[cli] 路径已写入 {args.out}")
    if args.json:
        path.to_json(args.json)
        print(f"[cli] 路径 JSON 已写入 {args.json}")
    if args.viz:
        from .viz import render_html
        render_html(planner.ground, path, args.viz, start=args.start, goal=args.goal,
                    map_name=args.map)
        print(f"[cli] 可视化已写入 {args.viz}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    cfg = _config_from_args(args)
    planner = _load_planner(args.map, cfg)
    serve(planner, host=args.host, port=args.port, robot_pose=args.start, map_name=args.map)
    return 0


def cmd_goal(args) -> int:
    from .server import publish_goal
    res = publish_goal(args.goal[0], args.goal[1], args.goal[2],
                       host=args.host, port=args.port, start=args.start)
    if res.get("success"):
        print(f"[cli] 目标点已发布, 路径长度 {res['length']:.2f} m, "
              f"爬升 {res['climb']:.2f} m, 耗时 {res['planning_time'] * 1000:.0f} ms, "
              f"位姿数 {len(res['path']['x'])}")
    else:
        print(f"[cli] 规划失败: {res.get('message')}", file=sys.stderr)
        return 2
    if args.out:
        arr = np.column_stack([res["path"]["x"], res["path"]["y"],
                               res["path"]["z"], res["path"]["yaw"]])
        np.savetxt(args.out, arr, delimiter=",", header="x,y,z,yaw", comments="", fmt="%.6f")
        print(f"[cli] 路径已写入 {args.out}")
    return 0


def cmd_info(args) -> int:
    cfg = _config_from_args(args)
    planner = _load_planner(args.map, cfg)
    print(json.dumps(planner.stats(), ensure_ascii=False, indent=2))
    if args.viz:
        from .viz import render_html
        render_html(planner.ground, None, args.viz, map_name=args.map)
        print(f"[cli] 可视化已写入 {args.viz}")
    return 0


def cmd_demo(args) -> int:
    cloud = make_demo_map()
    write_pcd(args.out, cloud, binary=not args.ascii)
    print(f"[cli] 已生成合成地图 {args.out} ({len(cloud)} 点)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dddmr_py", description="dddmr 3D 导航栈的纯 Python 复现: PCD 点云 -> 3D 全局路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="离线规划一条 3D 路径")
    sp.add_argument("map", help="PCD 地图路径, 或写 demo 使用内置合成地图")
    sp.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    sp.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    sp.add_argument("--out", help="输出 CSV 路径")
    sp.add_argument("--json", help="输出 JSON 路径")
    sp.add_argument("--viz", help="输出交互式 HTML 可视化")
    _add_config_args(sp)
    sp.set_defaults(func=cmd_plan)

    sv = sub.add_parser("serve", help="启动目标点发布服务 + 浏览器可视化")
    sv.add_argument("map", help="PCD 地图路径, 或写 demo")
    sv.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"), help="机器人初始位姿")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    _add_config_args(sv)
    sv.set_defaults(func=cmd_serve)

    gp = sub.add_parser("goal", help="向运行中的服务发布 3D 目标点")
    gp.add_argument("goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    gp.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    gp.add_argument("--host", default="127.0.0.1")
    gp.add_argument("--port", type=int, default=8000)
    gp.add_argument("--out", help="输出 CSV 路径")
    gp.set_defaults(func=cmd_goal)

    ip = sub.add_parser("info", help="查看地图统计 (节点数/可通行率/高程范围)")
    ip.add_argument("map")
    ip.add_argument("--viz", help="输出交互式 HTML 可视化")
    _add_config_args(ip)
    ip.set_defaults(func=cmd_info)

    dp = sub.add_parser("demo", help="生成合成测试点云 PCD")
    dp.add_argument("--out", default="demo_map.pcd")
    dp.add_argument("--ascii", action="store_true", help="以 ascii 格式写出")
    dp.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
