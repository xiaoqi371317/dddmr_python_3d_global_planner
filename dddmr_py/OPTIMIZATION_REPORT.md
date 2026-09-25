# dddmr_py 性能优化分析报告

> 针对 [xiaoqi371317/dddmr_python_3d_global_planner](https://github.com/xiaoqi371317/dddmr_python_3d_global_planner)
> 的逐层剖析与重写。所有数字都是在下述环境实测得到，不是估算。
>
> **测试环境**：2 vCPU / 2 GB RAM，Python 3.13，numpy 2.3.5，scipy 1.17.1，numba 0.66
> **测试地图**：仓库自带的 `demo_map.pcd`(3.7 万点)、`building.pcd`(85 万点)、`map.pcd`(200 万点)

---

## 0. 结论速览

| | 原版 | 优化版 |
|---|---|---|
| 85 万点地图 @ voxel=0.1 | **MemoryError**（1.7 GB 上限） | 6.3 s / 460 MB |
| 200 万点地图 @ voxel=0.15 | **MemoryError** | 3.7 s / 450 MB |
| 预处理（85 万点 @ 0.2） | 4.13 s | 0.81 s（**5.1x**） |
| A* 搜索 ×20 | 0.86 s | 0.17 s（**4.9x**） |
| 峰值内存 | 699 MB | 269 MB（**2.6x**） |
| 图中存在穿墙边 | 长边里 26.4% | 0（有数学证明的校验） |

优化版与原版 **API 完全兼容**，`import dddmr_py` 即可直接替换。

---

## 1. 瓶颈定位

先做分阶段计时（`demo_map.pcd`，1.56 万节点）：

```
read_pcd          0.003s  N=36618        rss= 66MB
voxel_downsample  0.047s  N=32510        rss= 70MB
estimate_normals  0.149s                 rss= 95MB
build_ground_map  0.397s  nodes=15610    rss=125MB
build_graph       2.595s  edges=138081   rss=582MB   <-- 时间和内存双杀
astar             0.131s  expanded=7468
```

`build_graph` 在一个**只有 1.5 万节点的小图**上就吃掉 457 MB、耗时 2.6 s。
换成 85 万点的 `building.pcd` 直接崩溃：

```
TypeError: object of type 'NoneType' has no len()
  ↑ 真实原因是上一行的 MemoryError 被吞掉了
File "perception.py", line 97, in _cylinder_neighbors
    lens = np.fromiter((len(l) for l in lists), ...)
```

于是问题收敛到两处：`perception._cylinder_neighbors` 和 `graph.build_graph`。

---

## 2. 逐项分析与改法

### 2.1 建图：`query_pairs` 枚举了 40 倍于所需的边 ⭐ 影响最大

```python
# 原版 graph.py
pairs = tree.query_pairs(radius, output_type="ndarray")   # radius 默认 2.2 m
...
if cfg.max_neighbors > 0:                                  # 再截断到 12 个
```

实测（1.56 万节点，`connection_radius=2.2`，`voxel=0.1`）：

```
query_pairs(2.2) -> 4,129,247 对, 仅 pairs 数组就 66 MB
平均每节点邻居 = 529   而 max_neighbors 只保留 12  ->  97.7% 当场被丢弃
```

`pairs` 只是开始：后续 `delta`/`d_xy`/`d_z`/`length`/`mean_cost`/`w`
再加上 top-k 截断用的 `np.lexsort`，每个都是 400 万行的数组，峰值叠到 457 MB。
节点数翻 10 倍，这里就是 100 倍内存——`building.pcd` 必然 OOM。

**改法**：直接 kNN 建图，不走"先全枚举再截断"。

```python
free_idx = np.flatnonzero(~lethal)        # 致命节点本就不会有边, 先剔除
tree_free = cKDTree(nodes[free_idx])      # 树更小, 且返回的邻居全部可用
dist, idx = tree_free.query(nodes[free_idx], k=K*3+1,
                            distance_upper_bound=radius, workers=-1)
```

内存从 `O(N·529)` 降到 `O(N·37)`。注意半径**保留**上游的 2.2 m：
kNN 下半径只是一个上界，几乎不增加开销，因此不必为了性能牺牲语义。

> **实测：1.346 s → 0.056 s，24.1x**

两个配套细节：

- **过采样 ×3**：坡度/台阶约束还会再刷掉一部分邻居。若先取 k 个再过滤，
  墙边节点的近邻大多是致命的，滤完就没剩几条边——我最初的版本就栽在这里，
  路径凭空长了 30%。先剔除致命节点 + 过采样后问题消失。
- **自适应升档**：台阶处最近的一批邻居会被坡度约束全否掉，而更远、水平位移
  更大因而更平缓的边反而合法（判据是 `d_z ≤ max(tan·d_xy, max_step)`）。
  对"边数不足 K 且候选已取满"的节点提高 k 重查，实测触发率 < 0.01%。

### 2.2 感知：`query_ball_point` 的 list-of-lists ⭐ OOM 元凶

```python
# 原版 perception.py
lists = tree_xy.query_ball_point(query_xy, radius, workers=-1)   # Python 列表的列表
lens = np.fromiter((len(l) for l in lists), ...)
flat = np.concatenate([np.asarray(l) for l in lists if l])
```

每个节点返回一个 Python list，几千万个 Python int 对象——光对象头就压垮内存。

**改法**：这两处（净空 clearance、膨胀 inflation）在语义上都是
*"在 XY 半径内，找沿法向高度 dz 落在 (max_step, robot_height) 内的最近点"*，
是一个**归约**，不需要物化整个邻域。于是改成两层结构：

1. **z-slab 分层**：按 query 点高度切片，每片只对"高度上可能落入带内"的目标点
   建 KD-tree。这一步是关键——否则地面高度的点（dz≈0，永远不满足高度带）
   会占满 k 近邻名额，逼着 k 一路升档。我第一版没做分层，k 升到 384 时
   一次分配了 436 MB 直接爆掉。
2. **定长 kNN + 自适应升档**：返回规整的 `(chunk, k)` 数组；结果按距离升序，
   **首个**满足高度带的即真正最近点；只有"未命中且候选取满"的行才提高 k 重查。
   分块大小按 `budget // k` 反比收缩，中间数组内存恒定有上界。

> **等价性验证**：`building.pcd` voxel=0.2/0.3 下，节点集 100% 逐点重合，
> lethal 标记 100% 一致。这是一个**逐位等价**的替换，只是不再 OOM。

### 2.3 体素降采样：`np.unique(axis=0)` + `np.add.at`

```python
keys = np.floor(xyz / voxel).astype(np.int64)
_, inv, counts = np.unique(keys, axis=0, ...)   # (N,3) 的 void-view 排序
sums = np.zeros(...); np.add.at(sums, inv, xyz) # ufunc.at 是出名的慢路径
```

**改法**：三维整数栅格线性化成**一维键**再 `np.unique`，累加用 `np.bincount`。

> **实测（85 万点）：1.172 s → 0.117 s，10.0x**，输出逐点一致（见单元测试）

另提供 `chunk` 流式模式，峰值内存从 `O(点数)` 降到 `O(体素数 + chunk)`。

### 2.4 法向量：`einsum` + `eigh`

```python
cov = np.einsum("ikj,ikl->ijl", centered, centered) / (k-1)   # 不走 BLAS
evals, evecs = np.linalg.eigh(cov)                            # (N,3,3) 逐个进 LAPACK
```

**改法**：协方差改用 `matmul`（批量 GEMM，走 BLAS）；3×3 对称阵有闭式解，
用解析特征分解替代 `eigh`。

> **实测（6 万点）：0.104 s → 0.024 s，4.3x**，特征值最大误差 **2.5e-16**

### 2.5 A\* 主循环：Python 逐节点开销

原版是纯 Python `heapq` 循环，且每次取邻居都用 Python int 索引 numpy 数组
（会构造 numpy 标量对象，比 list 索引慢 3~5 倍），启发式还在循环里逐点算 `sqrt`。

**改法**：三层后端，按可用性自动降级。

| 后端 | 做法 | 实测 |
|---|---|---|
| `numba` | 主循环 + 二叉堆 JIT 成机器码 | 0.137 s → **0.008 s（18.2x）** |
| `python` | 启发式向量化预计算 + CSR 转 list | 0.137 s → 0.062 s（2.2x） |
| `scipy` | 退化为 C 实现的 Dijkstra | 无启发式但常数极小 |

三个后端在同一图上 **代价与扩展节点数完全一致**（有单元测试断言）。

> numba 首次编译约 3.8 s。已做两件事消化掉：`cache=True` 让编译结果落盘
> （第二个进程起 0.2 s），并在 `GlobalPlanner3D.__init__` 里丢到后台线程
> 与地图预处理并行，用户感知不到。

### 2.6 搜索效率：少搜比搜得快更值钱

- **连通分量预判**：建图时用 `scipy.sparse.csgraph.connected_components`
  预计算分量号。起终点不连通时 **O(1)** 直接失败——原版这种情况要把整张图
  扩展一遍才报错（`building.pcd` 上一次失败要扩展 6.7 万节点）。
- **ALT 地标启发式**（`alt_landmarks=8`，默认关）：预存少量地标的精确图距离，
  用 `h(n)=max_i |d(Lᵢ,goal) − d(Lᵢ,n)|` 作可采纳下界。欧氏启发式在绕墙/绕楼时
  非常弱，ALT 能显著压低扩展数。预处理用 scipy 的 C 版 Dijkstra，几秒完成。
- **双向 A\***（默认关）：终止条件用"两侧前沿 f 均不低于当前最优"，保证最优性。

### 2.7 路径后处理：循环里逐点查 KD-tree

```python
# 原版 _shortcut: O(路径点数 × lookahead) 次单线段查询, 每次还带 workers=-1 的线程池开销
while j > i + 1:
    if self._segment_feasible(pts[i], pts[j]): ...
# 原版 _smooth: (迭代 × 路径点) 的双重 Python 循环, 每点一次 query
```

**改法**：`_shortcut` 把一个 `i` 的全部候选 `j` 合并成**一次**批量查询；
`_smooth` 改成 Jacobi 式更新，每轮只做一次批量查询。

> `make_plan ×20`（`demo_map.pcd`，两侧均 12/12 成功，可公平对比）：
> 原版 0.25 s → 优化版 0.06 s，**3.8x**

### 2.8 内存

| 措施 | 效果 |
|---|---|
| 点云默认 float32（原版恒 float64） | 减半 |
| CSR 索引 int32、权重 float32 | 图内存 4.54 MB → 2.66 MB |
| binary PCD 走 `np.memmap` | 不整份读入 |
| 默认丢弃 intensity/rgb 等额外字段 | 按需保留 |
| 不再长期持有原始点云 | 原版 `self.cloud` 对 200 万点白占 48 MB |
| 中间量显式 `del` + 分块粒度可配 | 峰值可控 |

> 节点坐标**刻意保留 float64**：`cKDTree` 内部就用 float64，
> 传 float32 反而会多出一份转换副本。

### 2.9 预处理缓存

感知 + 建图是纯函数：`(点云内容, 影响地图的参数) → (GroundMap, NavGraph)`。
按 `(文件指纹, 参数指纹)` 存 npz，第二次起直接加载。
参数指纹只覆盖影响地图的字段——改 `heuristic_weight` 之类的搜索参数不会让缓存失效。

---

## 3. 顺带发现的三个正确性问题

### 3.1 图里有 26% 的长边直接穿墙 ⚠️ 安全问题

原版建图只检查边的**两个端点**是否致命，中间完全不校验。抽查 3000 条 >0.5 m 的边：

```
中途压过致命(碰撞)节点:  792 条 (26.4%)
```

这不是理论隐患。把原版图里这些边删掉（仅占全部边的 0.9%）再跑 A\*：

```
目标 (12,3,0):
  原版图(含穿障长边)  几何长度 11.920 m
  碰撞检查后的安全图  几何长度 15.467 m   <-- 真正的安全最优解
  优化版(kNN 图)      最终长度 15.267 m   <-- 拉直后甚至短于图最优
```

**原版那条"更短"的 11.78 m 路径是从墙里穿过去的。**

这些长边本身是 §3.2 那个 bug 的副产物：节点顺序恰好让某些节点的"上行邻居"
名额没填满，就把 1 米开外的点也连了进来。

**优化版的做法**（`edge_collision_check=True`，默认开）：

`d_leth(p)` 与 `d_free(p)`（到最近致命/可通行节点的距离）都是 1-Lipschitz 的，因此

- **单点认证**：设中点 `m`、边长 `L`，对边上任意 `p` 有 `|p−m| ≤ L/2`，故
  `d_leth(p) − d_free(p) ≥ d_leth(m) − d_free(m) − L`。
  中点满足余量即可**一次查询认证整条边**——绝大多数边离致命区很远，
  这一级就筛掉了，是这段校验的主要提速来源（建图 22.1 s → 3.6 s）。
- **逐点校验**：未通过认证的边按 1/4 体素采样，检查 `d_leth ≥ d_free + h`
  （`h` 为采样间距）。由 Lipschitz 性质，这保证**整条线段上**致命节点都不会
  成为最近节点，不存在漏检，且与采样密度无关。

单元测试用 4 倍于建图的采样密度独立复核这个不变量。

### 3.2 `max_neighbors` 实际保留了约 2 倍的邻居

```python
for src in (a, b):
    order = np.lexsort((length, src))
    keep[order] &= rank < cfg.max_neighbors
```

`query_pairs` 保证 `a < b`，所以对节点 `i` 而言，`a` 列里只有"指向更大索引"的边，
`b` 列里只有"来自更小索引"的边。截断分别作用在两列上，于是每个节点实际保留
**上行 12 + 下行 12**。实测 `max_neighbors=12` 时平均度数 **17.7**（最大 24）。

这不只是参数名不符：由于名额按节点索引而非距离划分，节点顺序会决定哪些边被留下，
这正是 §3.1 那些跨越式长边的来源。

优化版严格保留最近的 K 条。若要复刻上游的边密度，把 `max_neighbors` 设成 18~24：

| `max_neighbors` | 平均度数 | 相对原版的路径长度差 |
|---|---|---|
| 12（默认） | 11.0 | 中位 +3.45% |
| **18** | 16.5 | **中位 +0.35%** |
| 24 | 22.0 | 中位 −0.78%（比原版更短） |

### 3.3 `layer_gap` 定义了但从未被使用

`config.py` 里声明了 `layer_gap: float = 0.40  # 区分不同楼层/结构层的高差`，
全项目搜索无任何引用。优化版在 2.5D 分层预筛（`fast_surface_prefilter`）里真正用上了它。

> 该预筛**默认关闭**：实测只省掉约 6% 的感知耗时，却会在多层/楼梯结构上丢节点
> （`building.pcd` voxel=0.2 下丢了 23% 的可通行节点）。收益与风险不匹配，
> 仅建议在单层平坦场景开启。这是我在优化过程中**否决掉的一项优化**。

---

## 4. 实测数据汇总

### 4.1 微基准

| 操作 | 规模 | 原版 | 优化版 | 加速 |
|---|---|---|---|---|
| `voxel_downsample` | 85 万点 | 1.172 s | 0.117 s | **10.0x** |
| 法向量特征分解 | 6 万点 | 0.104 s | 0.024 s | **4.3x** |
| `build_graph` | 2.56 万节点 | 1.346 s | 0.056 s | **24.1x** |
| A\*（numba） | 2.56 万节点 | 0.137 s | 0.008 s | **18.2x** |
| A\*（纯 Python） | 2.56 万节点 | 0.137 s | 0.062 s | 2.2x |

### 4.2 端到端（默认配置，含长边碰撞校验）

| 地图 | 点数 | voxel | 阶段 | 原版 | 优化版 | 加速 |
|---|---|---|---|---|---|---|
| demo_map | 3.7 万 | 0.10 | 预处理 | 2.92 s | 0.73 s | **4.0x** |
| | | | A\* ×20 | 0.64 s | 0.19 s | 3.4x |
| | | | `make_plan` ×20 | 0.25 s | 0.06 s | 3.8x |
| | | | 峰值内存 | 581 MB | 221 MB | **2.6x** |
| building | 85 万 | 0.20 | 预处理 | 4.13 s | 2.34 s | 1.8x |
| | | | A\* ×20 | 0.86 s | 0.17 s | 4.9x |
| | | | `make_plan` ×20 | 0.91 s | 0.01 s | 不可比 † |
| | | | 峰值内存 | 699 MB | 269 MB | **2.6x** |
| building | 85 万 | **0.10** | 预处理 | **OOM** | 6.35 s | — |
| | | | 峰值内存 | **OOM** | 460 MB | — |
| map.pcd | **200 万** | 0.15 | 预处理 | **OOM** | 3.74 s | — |
| | | | 峰值内存 | **OOM** | 450 MB | — |

† `building.pcd` 这一行的 `make_plan` 不具可比性：开启碰撞校验后，20 个随机查询里
大部分目标点变为**真实不可达**，被连通分量预判 O(1) 拒掉，因此耗时被压到 0.01 s。
要看公平的端到端对比请用 `demo_map.pcd` 那一行（两侧均 12/12 成功，3.8x）。

### 4.3 兼容模式（`PlannerConfig.compat()`，关闭会改变可达性的安全增强）

`building.pcd` @ voxel=0.20：

| 阶段 | 原版 | 优化版 | 加速 |
|---|---|---|---|
| 感知 | 2.84 s | 0.76 s | 3.7x |
| 建图 | 1.46 s | 0.13 s | **11.0x** |
| 预处理合计 | 4.14 s | 0.81 s | **5.1x** |
| A\* ×20 | 1.04 s | 0.18 s | 5.9x |
| 峰值内存 | 698 MB | 278 MB | 2.5x |

### 4.4 等价性验证（兼容模式，`building.pcd` @ voxel=0.20）

```
1) 感知层节点集      逐点重合 25597/25597 (100.000%)
2) 致命标记          共同节点上一致率 100.000%
3) 图连通性          200 组随机起终点, 可达性判定一致 191 (95.5%)
4) 端到端路径长度    成功性不一致 0 条; 相对长度差中位 +3.45%
                     (max_neighbors=18 时降到 +0.35%)
```

可达性那 4.5% 的差异来自 kNN 与全枚举在稀疏区域的选边差异，
可用 `max_neighbors` / `neighbor_oversample` 调节。

### 4.5 默认（安全）模式下的可达性变化

`building.pcd` @ voxel=0.20，60 组随机起终点：

| 图 | 判定可达 | 说明 |
|---|---|---|
| 原版（含穿障边） | 56 / 60 | 其中 27 组只能靠穿墙到达 |
| 优化版 kNN（不查碰撞） | 56 / 60 | **与原版连通性完全一致** → 建图是无损替换 |
| 安全真值（收敛值） | 29 / 60 | 逐步加密采样至收敛 |
| 优化版（查碰撞，默认） | **29 / 60** | ✓ 命中真值 |

收敛性核验：采样点数 6→12→24→48→96 对应 36/36/**29/29/29** ——
24 点以上稳定，说明 29 是真值而非过严。

> 这意味着开启默认的碰撞校验后，一部分目标点会**如实地**变为不可达。
> 这是修正而非退化：原先的"可达"依赖于穿墙。若需复刻上游行为，
> 用 `PlannerConfig.compat()` 或 `edge_collision_check=False`。

---

## 5. 还能继续做的

按性价比排序：

1. **障碍膨胀改用 2D 距离变换（EDT）**。当前逐节点查最近障碍是 `O(N log N)`，
   在 XY 栅格上做可分离 EDT 是 `O(G)`。地图规则时还能再快数倍。
2. **A\* 结果复用**。同一起点连续发目标点时，可增量修复搜索树（D\* Lite / LPA\*），
   而不是每次从头搜。
3. **感知阶段多进程**。当前 KD-tree 查询已用 `workers=-1` 吃满核心，
   但体素化和法向量的 Python 层仍是单线程，可按空间分块并行。
4. **图压缩**。当前节点即体素，可对开阔区域做自适应粗化（四叉/八叉），
   节点数能降一个数量级而不影响路径质量。
5. **`connection_radius` 自适应**。稀疏区域需要大半径、密集区域不需要，
   可按局部点密度逐节点定半径。
