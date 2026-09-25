"""最小示例: PCD 地图 -> 3D 路径."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dddmr_py import GlobalPlanner3D, PlannerConfig, make_demo_map, write_pcd

if __name__ == "__main__":
    write_pcd("demo_map.pcd", make_demo_map())

    cfg = PlannerConfig(
        robot_radius=0.35, robot_height=1.20,
        voxel_size=0.10,
        max_neighbors=18,           # 18~24 时路径长度与上游基本一致
        cache_dir=".dddmr_cache",   # 二次加载直接命中缓存
    )
    planner = GlobalPlanner3D.from_pcd("demo_map.pcd", cfg, verbose=True)
    print(planner.stats())

    path = planner.make_plan(start=(0, 0, 0), goal=(8, 8, 0))
    if path.success:
        print(f"长度 {path.length:.2f} m  爬升 {path.climb:.2f} m  "
              f"位姿 {len(path.points)} 个  耗时 {path.planning_time*1000:.0f} ms")
        path.to_csv("path.csv")
        from dddmr_py.viz import render_html
        render_html(planner.ground, path, "path.html")
        print("已输出 path.csv / path.html")
    else:
        print("规划失败:", path.message)
