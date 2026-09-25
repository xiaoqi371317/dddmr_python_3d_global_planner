# dddmr_py · v2.0 Fast 性能优化版

[`xiaoqi371317/dddmr_python_3d_global_planner`](https://github.com/xiaoqi371317/dddmr_python_3d_global_planner)
的性能重写版：**输入 PCD 点云地图 → 发布 3D 目标点 → 得到 3D 路径**，不依赖 ROS。

与上游 **API 完全兼容**，`import dddmr_py` 直接替换即可。

---

## 一句话总结

> 原版在 85 万点的地图上会 **MemoryError**；这一版在 200 万点、2 GB 内存的机器上
> 3.7 秒跑完预处理，峰值 450 MB。顺带修掉了一个会让规划出的路径**从墙里穿过去**的 bug。

| | 原版 | 本版 |
|---|---|---|
| 85 万点 @ voxel=0.1 | **MemoryError** | 6.3 s / 460 MB |
| 200 万点 @ voxel=0.15 | **MemoryError** | 3.7 s / 450 MB |
| 预处理（85 万点 @ 0.2，兼容模式） | 4.13 s | 0.81 s（**5.1x**） |
| A\* 搜索 ×20 | 0.86 s | 0.17 s（**4.9x**） |
| 峰值内存 | 699 MB | 269 MB（**2.6x**） |
| 图中穿墙边 | 长边里 **26.4%** | 0（有 Lipschitz 证明的校验） |

完整的分析过程、实测数据和等价性验证见 **[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)**。

本次发布的变更摘要见 [CHANGELOG.md](CHANGELOG.md)。

## 新增示例地图：`luoxuan.pcd`

仓库新增螺旋场景点云 `luoxuan.pcd`（231,885 个原始点）。建议先查看统计信息与可视化：

```bash
python -m dddmr_py info luoxuan.pcd --viz luoxuan.html --cache-dir .dddmr_cache
```

启动交互式规划服务：

```bash
python -m dddmr_py serve luoxuan.pcd --start 0 0 0 --port 8000 --cache-dir .dddmr_cache
```

在浏览器打开 <http://localhost:8000>；`0.0.0.0` 仅表示服务监听所有网卡，不是浏览器访问地址。

---

## 安装

```bash
pip install -r requirements.txt     # numpy + scipy (numba 可选但强烈建议)
# 或
pip install -e .[fast]
```

`numba` 是可选依赖：装了就自动把 A\* 主循环 JIT 成机器码（实测 18x），
没装则回退到优化过的纯 Python 实现（仍有 2.2x）。

---

## 快速开始

```python
from dddmr_py import GlobalPlanner3D, PlannerConfig

cfg = PlannerConfig(
    robot_radius=0.35, robot_height=1.20,
    voxel_size=0.10,
    max_neighbors=18,           # 18~24 时路径长度与上游基本一致
    cache_dir=".dddmr_cache",   # 预处理结果落盘, 二次加载毫秒级
)
planner = GlobalPlanner3D.from_pcd("map.pcd", cfg, verbose=True)
path = planner.make_plan(start=(0, 0, 0), goal=(12, 3, 3.0))

print(path.length, path.climb, path.poses)   # (N,4) 的 [x,y,z,yaw]
path.to_csv("path.csv")
```

命令行：

```bash
python -m dddmr_py plan map.pcd --start 0 0 0 --goal 12 3 3 \
       --viz out.html --cache-dir .dddmr_cache
python -m dddmr_py serve map.pcd --port 8000     # 浏览器点选目标点
python -m dddmr_py info map.pcd                  # 地图统计
```

---

## 主要改动

| 模块 | 原版做法 | 本版做法 | 实测 |
|---|---|---|---|
| `graph.build_graph` | `query_pairs(2.2m)` 枚举全部点对再截断到 12 个（**97.7% 被丢弃**） | 直接 kNN + 过采样 + 自适应升档 | **24.1x** |
| `perception` 定半径查询 | `query_ball_point` 返回 Python list-of-lists（**OOM 元凶**） | z-slab 分层 + 定长 kNN 归约，内存有上界 | 不再 OOM |
| `voxel_downsample` | `np.unique(axis=0)` + `np.add.at` | 一维哈希键 + `bincount` | **10.0x** |
| 法向量特征分解 | `einsum` + `np.linalg.eigh` | 批量 `matmul` + 3×3 解析特征分解 | **4.3x**（误差 2.5e-16） |
| A\* 主循环 | 纯 Python heapq + numpy 标量索引 | numba JIT / 优化 Python / scipy 三后端 | **18.2x** |
| 不可达目标 | 扩展整张图才报错 | 连通分量 **O(1)** 预判 | — |
| 路径拉直/平滑 | 循环里逐点查 KD-tree | 全部批量化 | 显著 |
| 存储 | 恒 float64 | float32 + int32 CSR + memmap | 内存 **2.6x** |
| 预处理 | 每次重算 | 按内容+参数指纹落盘缓存 | 秒级 → 毫秒级 |

---

## 新增配置项

```python
PlannerConfig(
    # ---- 性能 ----
    dtype_32bit=True,            # float32 点云 / int32 图索引
    chunk_size=200_000,          # 分块粒度, 控制峰值内存
    search_backend="auto",       # auto | numba | python | scipy
    cache_dir=".dddmr_cache",    # 预处理缓存目录, 空=不缓存
    workers=-1,                  # KD-tree 线程数

    # ---- 搜索效率 ----
    prune_components=True,       # 连通分量预判, 不可达时 O(1) 失败
    alt_landmarks=0,             # ALT 地标数, 迷宫型地图建议 8~16
    bidirectional=False,         # 双向 A*

    # ---- 正确性 / 质量 ----
    edge_collision_check=True,   # 长边碰撞校验 (修穿墙 bug)
    neighbor_oversample=3,       # kNN 过采样, 抵消坡度过滤的邻居损失
    max_neighbors=12,            # 严格生效 (原版实际是约 2 倍)
)
```

---

## ⚠️ 两处行为变化（都是修 bug，请先读）

### 1. 路径可能变长——因为原来那条是穿墙的

原版建图只校验边的**两个端点**，中间不管。抽查 3000 条 >0.5 m 的边，
**26.4% 直接从致命（碰撞）节点上方穿过**。

把这些边删掉再跑原版 A\*：

```
原版图(含穿障长边)   11.92 m     ← 穿墙
碰撞检查后的安全图   15.47 m     ← 真正的安全最优解
本版(默认配置)       15.27 m     ← 安全, 且拉直后短于图最优
```

本版默认开启 `edge_collision_check`，判据基于 `d_lethal`/`d_free` 的 1-Lipschitz
性质，**保证整条线段无漏检**（详见报告 §3.1）。

副作用：部分目标点会**如实地**变为不可达（原先只能靠穿墙到达）。
实测 60 组起终点，原版判定 56 组可达，安全真值是 29 组，本版给出 29 组 ✓。

### 2. `max_neighbors` 现在严格生效

原版因为 top-k 截断分别作用在 `query_pairs` 的两列上，
`max_neighbors=12` 实际保留平均 **17.7** 个邻居。本版严格保留最近 K 个。

想复刻上游的边密度和路径长度：

| `max_neighbors` | 平均度数 | 相对原版路径长度 |
|---|---|---|
| 12（默认） | 11.0 | 中位 +3.45% |
| **18** | 16.5 | **中位 +0.35%** |
| 24 | 22.0 | 中位 −0.78% |

### 想完全复刻上游行为

```python
cfg = PlannerConfig.compat(voxel_size=0.10, max_neighbors=18)
```

`compat()` 关闭所有会改变可达性的增强，只保留纯实现级提速
（此模式下节点集与 lethal 标记与原版 **100% 逐点一致**）。

---

## 验证

```bash
python tests/test_planner.py                    # 18 项单元测试

# A/B 基准 (两侧各跑独立子进程, 内存互不污染)
python benchmarks/bench.py --orig /path/to/upstream/dddmr_py \
       --map building.pcd --voxel 0.20

# 与原版逐层核对 (节点集 / lethal / 连通性 / 路径长度)
python benchmarks/verify.py --orig /path/to/upstream/dddmr_py \
       --map building.pcd --voxel 0.20
```

`verify.py` 在兼容模式下的输出：

```
1) 感知层节点集      逐点重合 25597/25597 (100.000%)
2) 致命标记          共同节点上一致率 100.000%
3) 图连通性          200 组随机起终点, 可达性判定一致 191 (95.5%)
4) 端到端路径长度    成功性不一致 0 条; 相对长度差中位 +3.45%
```

---

## 目录结构

```
dddmr_py/
  config.py       参数配置 (新增 [性能] 分组 + compat() 兼容模式)
  fastops.py      数值内核: 体素 / 法向量 / 解析特征分解 / 定半径归约
  pcd_io.py       PCD 读写 (float32 / memmap / 快速 LZF)
  perception.py   地面提取 + 静态代价层
  graph.py        kNN 建图 + 长边碰撞校验 + 连通分量
  search.py       A* (numba/python/scipy) + 双向 + ALT 地标
  planner.py      端到端规划器 + 批量后处理
  cache.py        预处理结果落盘缓存
  cli.py / server.py / viz.py / mapgen.py
benchmarks/
  bench.py        A/B 性能基准
  verify.py       等价性验证
tests/            18 项单元测试
```

---

## 下一步还能做什么

1. 障碍膨胀改用 2D 距离变换（EDT），`O(N log N)` → `O(G)`
2. 连续目标点场景做增量搜索（D\* Lite / LPA\*），复用上一次的搜索树
3. 开阔区域自适应粗化图节点（四叉/八叉），节点数可降一个数量级
4. `connection_radius` 按局部点密度逐节点自适应

---

原项目参考 [dddmr_navigation](https://github.com/dfl-rlab/dddmr_navigation)（DDDMobileRobot 的 3D 导航栈）。
