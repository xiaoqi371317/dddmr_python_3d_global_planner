"""图搜索: A* / 加权 A* / Dijkstra (基于 CSR 邻接结构)."""

from __future__ import annotations

import heapq
import math
from typing import List, Optional

import numpy as np

from .graph import NavGraph


class SearchResult:
    def __init__(self, path_idx: List[int], cost: float, expanded: int, success: bool,
                 reason: str = ""):
        self.path_idx = path_idx
        self.cost = cost
        self.expanded = expanded
        self.success = success
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<SearchResult success={self.success} nodes={len(self.path_idx)} "
                f"cost={self.cost:.2f} expanded={self.expanded}>")


def astar(graph: NavGraph, start: int, goal: int,
          heuristic_weight: Optional[float] = None,
          max_expansions: int = 5_000_000) -> SearchResult:
    """在 NavGraph 上做 A* 搜索, 启发式为 3D 欧氏距离.

    heuristic_weight = 0  -> Dijkstra (最优, 最慢)
    heuristic_weight = 1  -> 标准 A* (最优)
    heuristic_weight > 1  -> 加权 A* (次优但更快)
    """
    hw = graph.config.heuristic_weight if heuristic_weight is None else heuristic_weight
    nodes = graph.nodes
    indptr, indices, weights = graph.indptr, graph.indices, graph.weights
    n = len(nodes)
    if not (0 <= start < n and 0 <= goal < n):
        return SearchResult([], math.inf, 0, False, "起点或终点索引越界")
    if start == goal:
        return SearchResult([start], 0.0, 0, True)

    goal_xyz = nodes[goal]
    g = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)
    g[start] = 0.0

    def h(i: int) -> float:
        if hw <= 0.0:
            return 0.0
        d = nodes[i] - goal_xyz
        return hw * math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

    open_heap = [(h(start), 0.0, start)]
    expanded = 0

    while open_heap:
        f, gc, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        closed[cur] = True
        expanded += 1
        if cur == goal:
            path, node = [], cur
            while node != -1:
                path.append(int(node))
                node = parent[node]
            path.reverse()
            return SearchResult(path, float(g[goal]), expanded, True)
        if expanded > max_expansions:
            return SearchResult([], math.inf, expanded, False, "超出最大扩展节点数")

        beg, end = indptr[cur], indptr[cur + 1]
        for k in range(beg, end):
            nxt = int(indices[k])
            if closed[nxt]:
                continue
            ng = gc + weights[k]
            if ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

    return SearchResult([], math.inf, expanded, False, "起终点不连通(可能被障碍或断层隔开)")


def dijkstra(graph: NavGraph, start: int, goal: int) -> SearchResult:
    """Dijkstra = 启发式权重为 0 的 A*."""
    return astar(graph, start, goal, heuristic_weight=0.0)


def reachable_set(graph: NavGraph, start: int, max_cost: float = math.inf) -> np.ndarray:
    """从 start 出发的可达节点(可用于检查地图连通性/多层是否互通)."""
    n = graph.n_nodes
    dist = np.full(n, np.inf)
    dist[start] = 0.0
    heap = [(0.0, start)]
    done = np.zeros(n, dtype=bool)
    while heap:
        d, cur = heapq.heappop(heap)
        if done[cur]:
            continue
        done[cur] = True
        beg, end = graph.indptr[cur], graph.indptr[cur + 1]
        for k in range(beg, end):
            nxt = int(graph.indices[k])
            nd = d + graph.weights[k]
            if nd < dist[nxt] and nd <= max_cost:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist
