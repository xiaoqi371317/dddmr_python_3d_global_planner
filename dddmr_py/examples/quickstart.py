"""最小示例: PCD -> 3D 目标点 -> 路径."""

import numpy as np

from dddmr_py import GlobalPlanner3D, PlannerConfig, make_demo_map, write_pcd

# 0) 准备一张 PCD 地图 (这里用合成地图; 换成自己的 map.pcd 即可)
write_pcd("demo_map.pcd", make_demo_map())

# 1) 载入地图并做感知/建图 (只需一次, 之后可反复规划)
cfg = PlannerConfig(robot_radius=0.35, robot_height=1.2, max_slope_deg=25.0)
planner = GlobalPlanner3D.from_pcd("demo_map.pcd", cfg, verbose=True)
print(planner.stats())

# 2) 发布 3D 导航目标点 -> 直接得到路径
path = planner.make_plan(start=(1, 1, 0), goal=(21, 10, 3.0))
print(path)                       # <Path success=True ...>
print("路径长度 %.2f m, 累计爬升 %.2f m" % (path.length, path.climb))
print("前 3 个位姿 [x y z yaw]:\n", np.round(path.poses[:3], 3))

# 3) 导出
path.to_csv("path.csv")
path.to_json("path.json")

from dddmr_py.viz import render_html     # noqa: E402
render_html(planner.ground, path, "path.html", start=(1, 1, 0), goal=(21, 10, 3.0))
print("已生成 path.csv / path.json / path.html")

# 4) 同一 (x,y) 上下两层: 用目标点的 z 区分
low = planner.make_plan((1, 1, 0), (21, 10, 0.0))
print("去二层 %.2f m (爬升 %.2f) / 去一层同一位置 %.2f m (爬升 %.2f)"
      % (path.length, path.climb, low.length, low.climb))
