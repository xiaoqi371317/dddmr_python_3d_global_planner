# 更新说明

## v2.0.0 — Fast 性能优化版

本版本将规划器升级为面向大规模 PCD 点云的 Fast 实现，并保持既有的
`import dddmr_py`、命令行和 HTTP 服务接口可用。

### 性能与内存

- 建图由半径内全量点对枚举改为 kNN 建图，避免大图上生成大量最终会被丢弃的边。
- 感知层采用分层、分块的定长近邻归约，降低大地图上的峰值内存并避免 Python 邻居列表造成的内存耗尽。
- 体素降采样、法向量计算和路径后处理改为批量化实现；图索引和代价数据默认使用更紧凑的数值类型。
- A* 支持 `numba` JIT、优化 Python 和 Scipy 三种后端；安装 `.[fast]` 后会自动优先使用 JIT 后端。
- 加入按地图内容与参数指纹保存的 `.dddmr_cache` 预处理缓存，重复规划同一张 PCD 地图可直接复用建图结果。

### 正确性与规划质量

- 默认启用长边碰撞检查，移除端点可通行但中途穿过障碍的图边。
- 新增连通分量预判，不可达的起终点会快速返回而不必遍历整张图。
- `max_neighbors` 现在严格控制每个节点保留的近邻数；推荐 `18` 至 `24` 以接近旧版的边密度。
- 通过 `--compat` 或 `PlannerConfig.compat()` 可关闭会改变可达性的安全增强，以便复现旧版行为。

### 新功能与资源

- 新增 `fastops.py`、`cache.py`，以及性能测试与等价性验证相关实现。
- 新增 `luoxuan.pcd` 螺旋场景示例地图（231,885 点）。
- 新增 [dddmr_py/OPTIMIZATION_REPORT.md](dddmr_py/OPTIMIZATION_REPORT.md)，记录基准、实现细节和兼容性验证。

### 升级方式

```bash
cd dddmr_py
pip install -e .[fast]
python -m dddmr_py info ../luoxuan.pcd --cache-dir .dddmr_cache
```

Python 3.9 及以上可用。没有安装 `numba` 时，程序仍会降级使用优化后的非 JIT 搜索后端。
