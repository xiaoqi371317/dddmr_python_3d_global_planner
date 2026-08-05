# dddmr_py — dddmr 3D 导航栈的纯 Python 复现

输入一张 **PCD 点云地图**，发布一个 **3D 导航目标点 (x, y, z)**，直接得到一条 **3D 路径**。
不依赖 ROS，只需要 `numpy` + `scipy`。

参考对象是 [dddmr_navigation](https://github.com/dfl-rlab/dddmr_navigation)（DDDMobileRobot 的 3D 导航栈）
中「感知静态层 + 全局规划器」这两块的算法思路，用 Python 重写并做了工程化封装。

## Demo

<p align="center">
  <img src="assets/demo1.png" alt="Demo 1" width="90%" />
</p>

<p align="center">
  <img src="assets/demo2.png" alt="Demo 2" width="90%" />
</p>

<p align="center">
  <img src="assets/demo3.png" alt="Demo 3" width="90%" />
</p>

<p align="center">
  <img src="assets/demo4.png" alt="Demo 3" width="90%" />
</p>

| dddmr (ROS 2, C++) | 本项目 (纯 Python) |
|---|---|
| `dddmr_perception_3d` 静态层 | `perception.py`：体素降采样 / 法向量 / 可站立面提取 / 本体碰撞 / 膨胀代价 |
| `dddmr_global_planner` | `graph.py` + `search.py`：3D 邻接图 + A\*/Dijkstra |
| `dddmr_p2p_move_base` 目标点接口 | `server.py`：HTTP `POST /goal`（等价于 `/move_base_simple/goal`）|
| RViz 显示 | `viz.py`：自包含 WebGL 网页，点云上单击即可发布目标点 |
| `dddmr_lego_loam` / `mcl_3dl` / 局部规划 | **未包含**（本项目只做静态地图上的全局规划，见「范围与局限」）|

---

## 1. 安装

```bash
pip install numpy scipy          # 全部依赖
# 可选: cd dddmr_py && pip install -e .   （安装后可直接用 dddmr-py 命令）
```

Python ≥ 3.9。

## 2. 三种用法

### (a) 命令行：一次性规划

```bash
# 先生成一张合成测试地图（双层 + 斜坡 + 门洞 + 桌子）
python -m dddmr_py demo --out demo_map.pcd

# 规划: 从一层 (1,1,0) 到二层 (21,10,3)
python -m dddmr_py plan demo_map.pcd \
    --start 1 1 0 --goal 21 10 3 \
    --out path.csv --json path.json --viz path.html
```

输出：

```
[perception] 降采样: 36618 -> 32510 点 (voxel=0.1 m)
[perception] 可站立面候选 15047 个, 障碍点 17463 个
[perception] 致命节点 1719 个 (其中本体碰撞/低净空 ...), 可通行率 88.6%
[graph]      节点 15047 个, 双向边 106070 条 (半径 0.26 m)
[cli] 规划成功: 257 个位姿, 长度 25.50 m, 爬升 3.13 m, 扩展 6168 节点, 耗时 126 ms
```

把 `demo_map.pcd` 换成自己的地图即可。`--map` 写 `demo` 可以不落盘直接用内置地图。

### (b) Python API

```python
from dddmr_py import GlobalPlanner3D, PlannerConfig

cfg = PlannerConfig(robot_radius=0.35, robot_height=1.2, max_slope_deg=25.0)
planner = GlobalPlanner3D.from_pcd("map.pcd", cfg)   # 预处理一次

path = planner.make_plan(start=(1, 1, 0), goal=(21, 10, 3.0))
if path.success:
    print(path.length, path.climb)      # 路径长度 / 累计爬升
    print(path.poses)                   # (N,4) 的 [x, y, z, yaw]
    print(path.quaternions())           # (N,4) 四元数, 机体 z 轴贴合地面法向量
    path.to_csv("path.csv")

planner.plan_through([(1,1,0), (14,3,0), (21,10,3)])  # 多航点巡逻
```

### (c) 交互式：在浏览器里发布 3D 目标点

```bash
python -m dddmr_py serve demo_map.pcd --start 1 1 0 --port 8000
```

浏览器打开 <http://localhost:8000>：拖拽旋转、滚轮缩放，**在点云上单击就发布一个 3D 目标点**，
路径实时画出来。也可以从命令行或任意 HTTP 客户端发布：

```bash
python -m dddmr_py goal 21 10 3 --port 8000        # 命令行发布
curl -X POST localhost:8000/goal -H 'Content-Type: application/json' \
     -d '{"x":21,"y":10,"z":3.0}'                  # 等价于 ROS 的 /move_base_simple/goal
```

| 接口 | 说明 |
|---|---|
| `GET  /` | 交互式 3D 可视化页面 |
| `POST /goal` | `{"x":..,"y":..,"z":..}` → 从当前位姿规划到该目标点，返回路径 |
| `POST /plan` | `{"start":[..],"goal":[..]}` → 指定起点规划 |
| `POST /pose` | 设置机器人当前位姿 |
| `GET  /stats` | 地图统计 |

---

## 3. 算法流程

```
PCD 点云
  │
  ├─ ① 体素降采样 (voxel_size)
  ├─ ② PCA 法向量 + 局部平面残差 (normal_k)
  ├─ ③ 可站立面提取:  坡度 ≤ max_slope_deg 且 残差 ≤ max_roughness
  │        └─ 其余点 = 障碍点
  ├─ ④ 机器人本体碰撞: 以节点为底、半径 robot_radius、高 robot_height 的圆柱内有点 → 致命
  │        高度按「相对局部切平面」计算，斜坡上的上坡点不会被误判成头顶障碍
  ├─ ⑤ 障碍膨胀代价: cost = (lethal-1)·exp(-cost_scaling·(d - robot_radius))
  │        d ≤ robot_radius → 致命；只统计落在机器人身体高度带内的障碍点
  ├─ ⑥ 建图: 半径近邻 (connection_radius, 默认 2.6×voxel)，逐边检查
  │        坡度约束 |dz| ≤ tan(max_slope)·d_xy、台阶约束 |dz| ≤ max_step、端点非致命
  │        边权 w = 3D 长度 × (1 + cost_weight × 平均代价)
  ├─ ⑦ 搜索: A*（启发式 = 3D 欧氏距离 × heuristic_weight；0 = Dijkstra，>1 = 加权 A*）
  └─ ⑧ 后处理: string-pulling 拉直 → 等间隔重采样 → 带约束平滑 → 投影回地面 → 计算 yaw / 姿态
```

**为什么是 3D 的**：节点直接取自点云表面而不是 2D 栅格，所以同一个 (x, y) 上可以有多层可通行面
（楼板、桥面、高架平台）。目标点的 **z 决定去哪一层**——内置示例地图里
`(21, 10, 3.0)` 是走斜坡上二层（25.5 m，爬升 3.13 m），`(21, 10, 0.0)` 是留在一层从二层平台**下方**穿过（24.7 m，无爬升）。

## 4. 主要参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `robot_radius` | 0.35 | 机器人外接圆半径，决定膨胀致命范围 |
| `robot_height` | 1.20 | 机器人高度，决定能否从桌下/桥下/楼板下穿过 |
| `max_slope_deg` | 25.0 | 可通行最大坡度 |
| `max_step` | 0.12 | 可跨越最大台阶 |
| `max_roughness` | 0.05 | 局部平面拟合残差上限（越小越挑剔） |
| `voxel_size` | 0.10 | 体素边长；决定节点密度、精度与耗时 |
| `inflation_radius` | 0.55 | 膨胀半径（应 > `robot_radius`） |
| `cost_scaling_factor` | 3.0 | 膨胀代价指数衰减系数 |
| `connection_radius` | 自动 | 建图邻接半径，`<=0` 时取 `2.6×voxel_size` |
| `heuristic_weight` | 1.0 | `0`=Dijkstra，`1`=标准 A\*，`>1`=加权 A\*（更快、次优） |
| `cost_weight` | 0.02 | 代价对边权的影响：越大越"贴着中线走" |
| `snap_radius` | 1.5 | 起终点吸附到最近可通行节点的半径 |
| `path_resolution` | 0.10 | 输出路径重采样间隔 |
| `shortcut` / `smooth_iterations` | true / 3 | 拉直与平滑 |

可用 `--config xxx.yaml`（见 `examples/config_example.yaml`）或 `PlannerConfig(...)` 传入，
命令行还提供 `--voxel / --robot-radius / --max-slope / --max-step / --inflation / --heuristic-weight` 等快捷开关。

## 5. 用自己的 PCD 地图

1. **坐标系**：假定 **z 轴向上**（重力方向）。若地图是相机坐标系，先旋转到 z-up。
2. **格式**：支持 PCD v0.7 的 `ascii` / `binary` / `binary_compressed`（内置 LZF 解压，无需 `python-lzf`）。
3. **调参顺序**：先 `voxel_size`（大地图用 0.15~0.2 提速）→ `robot_height` / `robot_radius`
   → `max_slope_deg` / `max_step` → `inflation_radius`。
4. **先看一眼地图**再规划：

   ```bash
   python -m dddmr_py info map.pcd --viz map.html
   ```

   绿色=可通行、橙红=接近障碍、深红=不可通行、灰色=障碍点云。如果该走的地方是深红，
   多半是 `robot_height` 或 `max_slope_deg` 设得太严。
5. **规划失败排查**：
   - `起点/终点附近 X m 内没有可通行节点` → 目标点没落在可站立面上，或该处被判致命；增大 `snap_radius` 或放宽参数。
   - `起终点不连通` → 两块可通行面之间没有满足坡度/台阶约束的连接；检查斜坡坡度、`max_step`、
     或适当增大 `connection_radius`。

## 6. 输出

- `Path.poses` → `(N,4)` 的 `[x, y, z, yaw]`；`Path.quaternions()` → 贴合地面法向量的姿态四元数
- `path.to_csv()` / `path.to_json()`；服务模式返回同结构 JSON
- `Path.length` 路径长度、`Path.climb` 累计爬升、`Path.planning_time` 规划耗时

## 7. 性能（合成地图 36.6k 点 / 15k 节点 / 10.6 万条边，单核笔记本级 CPU）

| 阶段 | 耗时 |
|---|---|
| 地图预处理（降采样+法向量+代价+建图） | ≈ 0.45 s（只做一次） |
| 单次 A\* 规划（跨楼层 25 m） | ≈ 0.1 s |

预处理耗时大致与点数线性相关；`voxel_size` 加倍约能省 4 倍时间。

## 8. 范围与局限

本项目复现的是 **静态地图上的 3D 全局规划**，不包含 dddmr 的建图（LeGO-LOAM）、定位（MCL-3DL）、
局部规划/避障与语义感知（YOLO+TensorRT）。因此：

- 不处理动态障碍：地图变化需要重新构建 `GlobalPlanner3D`（或改造成增量更新）。
- 路径是几何路径，未考虑机器人运动学/动力学；差速/阿克曼底盘需要在下游做轨迹跟踪。
- 可站立面提取基于「法向量 + 平整度」的几何判据，没有语义（如玻璃、栅格、水面）判断。
- 孤立的水平面（桌面、台面）也会被当作可通行面，只是通常与主图不连通，不影响规划结果。

如果之后要接回 ROS 2：把 `GlobalPlanner3D.make_plan()` 包一层节点，订阅
`/move_base_simple/goal`、发布 `nav_msgs/Path` 即可，`server.py` 就是这个接口的无 ROS 版本。

## 9. 目录结构

```
dddmr_py/
├── dddmr_py/
│   ├── config.py       # PlannerConfig 参数
│   ├── pcd_io.py       # PCD 读写 (ascii/binary/binary_compressed + 纯 Python LZF)
│   ├── perception.py   # 降采样 / 法向量 / 可站立面 / 本体碰撞 / 膨胀代价
│   ├── graph.py        # 3D 邻接图 (CSR)
│   ├── search.py       # A* / Dijkstra / 可达集
│   ├── planner.py      # GlobalPlanner3D + Path
│   ├── viz.py          # 自包含 WebGL 可视化 (无外部依赖)
│   ├── server.py       # 3D 目标点发布服务 (HTTP)
│   ├── mapgen.py       # 合成测试地图
│   └── cli.py          # 命令行
├── examples/           # quickstart.py / config_example.yaml
├── tests/test_planner.py
└── README.md
```

## 10. 自检

```bash
python tests/test_planner.py
```

覆盖：PCD 读写往返、跨楼层规划、同 (x,y) 不同 z 区分楼层、路径连续/贴地/无碰撞、
低净空区域不可通行、非法目标点报错、A\* 与 Dijkstra 代价一致、多航点巡逻。
