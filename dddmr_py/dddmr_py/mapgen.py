"""生成合成测试点云: 双层结构 + 斜坡 + 隔墙门洞 + 桌子(低净空) + 柱子/矮障碍.

用于在没有真实 PCD 地图时验证 3D 规划效果. 场景 (单位 m):

  * 一层地面 z=0, 范围 x∈[0,24], y∈[0,14] (斜坡投影区域挖空)
  * 隔墙 x=5, 仅 y∈[6,7.6] 留门洞
  * 斜坡 x∈[8,16], y∈[9,13], 由 z=0 升到 z=3 (坡度约 20.6°), 两侧有护栏
  * 二层平台 z=3, 范围 x∈[16,24], y∈[6,14]; 平台正下方的一层区域净空 3 m, 仍可通行
    -> 同一 (x,y) 上下两层都可走, 只能靠 3D 目标点的 z 来区分
  * 桌子: 桌面 z=0.75 (下方净空不足, 不可通行)
  * 柱子 + 0.9 m 矮墙障碍
"""

from __future__ import annotations

import numpy as np

RAMP_X0, RAMP_X1 = 8.0, 16.0
RAMP_Y0, RAMP_Y1 = 9.0, 13.0
UPPER_Z = 3.0


def ramp_height(x: np.ndarray | float):
    return (np.asarray(x) - RAMP_X0) * (UPPER_Z / (RAMP_X1 - RAMP_X0))


def _plane(x0, x1, y0, y1, z_fn, density=40.0, rng=None, noise=0.005):
    rng = rng or np.random.default_rng(0)
    n = max(int((x1 - x0) * (y1 - y0) * density), 4)
    x = rng.uniform(x0, x1, n)
    y = rng.uniform(y0, y1, n)
    z = z_fn(x, y) + rng.normal(0.0, noise, n)
    return np.column_stack([x, y, z])


def _wall(x0, y0, x1, y1, z_bottom, z_top, density=40.0, rng=None, noise=0.005):
    """竖直墙面; z_bottom/z_top 可以是常数, 也可以是关于 (x,y) 的函数."""
    rng = rng or np.random.default_rng(1)
    length = float(np.hypot(x1 - x0, y1 - y0))
    height = 1.0
    if not callable(z_bottom) and not callable(z_top):
        height = max(z_top - z_bottom, 0.05)
    n = max(int(length * height * density), 8)
    t = rng.uniform(0, 1, n)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    zb = z_bottom(x, y) if callable(z_bottom) else np.full(n, z_bottom)
    zt = z_top(x, y) if callable(z_top) else np.full(n, z_top)
    z = zb + rng.uniform(0, 1, n) * (zt - zb)
    return np.column_stack([x + rng.normal(0, noise, n), y + rng.normal(0, noise, n), z])


def _cylinder(cx, cy, r, z0, z1, density=40.0, rng=None):
    rng = rng or np.random.default_rng(2)
    n = max(int(2 * np.pi * r * (z1 - z0) * density), 12)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(z0, z1, n)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), z])


def make_demo_map(seed: int = 7, density: float = 45.0) -> np.ndarray:
    """返回 (N,3) 合成点云."""
    rng = np.random.default_rng(seed)
    parts = []

    # 一层地面 (挖掉斜坡投影)
    ground = _plane(0, 24, 0, 14, lambda x, y: np.zeros_like(x), density, rng)
    in_ramp = ((ground[:, 0] >= RAMP_X0) & (ground[:, 0] <= RAMP_X1) &
               (ground[:, 1] >= RAMP_Y0) & (ground[:, 1] <= RAMP_Y1))
    parts.append(ground[~in_ramp])

    # 外墙
    for (x0, y0, x1, y1) in [(0, 0, 24, 0), (0, 14, 24, 14), (0, 0, 0, 14), (24, 0, 24, 14)]:
        parts.append(_wall(x0, y0, x1, y1, 0.0, 4.2, density, rng))

    # 隔墙 x=5, 门洞 y∈[6,7.6]
    parts.append(_wall(5, 0, 5, 6.0, 0.0, 2.6, density, rng))
    parts.append(_wall(5, 7.6, 5, 14, 0.0, 2.6, density, rng))

    # 斜坡 + 两侧护栏
    parts.append(_plane(RAMP_X0, RAMP_X1, RAMP_Y0, RAMP_Y1,
                        lambda x, y: ramp_height(x), density, rng))
    for y_rail in (RAMP_Y0, RAMP_Y1):
        parts.append(_wall(RAMP_X0, y_rail, RAMP_X1, y_rail,
                           lambda x, y: ramp_height(x),
                           lambda x, y: ramp_height(x) + 1.0, density, rng))

    # 二层平台 z=3 (x∈[16,24], y∈[6,14]) 与斜坡顶部无缝相接
    parts.append(_plane(16, 24, 6, 14, lambda x, y: np.full_like(x, UPPER_Z), density, rng))
    # 二层临空边护栏 (斜坡入口 y∈[9,13] 处留空)
    parts.append(_wall(16, 6, 16, 9, UPPER_Z, UPPER_Z + 1.0, density, rng))
    parts.append(_wall(16, 13, 16, 14, UPPER_Z, UPPER_Z + 1.0, density, rng))
    parts.append(_wall(16, 6, 24, 6, UPPER_Z, UPPER_Z + 1.0, density, rng))

    # 桌子: 桌面 0.75 m + 四条腿 (机器人钻不过去)
    parts.append(_plane(1, 3, 10, 12, lambda x, y: np.full_like(x, 0.75), density, rng))
    for (cx, cy) in [(1.1, 10.1), (2.9, 10.1), (1.1, 11.9), (2.9, 11.9)]:
        parts.append(_cylinder(cx, cy, 0.05, 0.0, 0.75, density * 3, rng))

    # 柱子
    for (cx, cy) in [(7.0, 3.0), (13.0, 5.0), (20.0, 2.5)]:
        parts.append(_cylinder(cx, cy, 0.35, 0.0, 2.6, density, rng))
    # 二层上的柱子
    parts.append(_cylinder(20.0, 10.0, 0.3, UPPER_Z, UPPER_Z + 1.2, density, rng))

    # 一层 (二层平台正下方) 的 0.9 m 矮墙障碍
    parts.append(_wall(18, 7.0, 18, 12.0, 0.0, 0.9, density, rng))
    parts.append(_wall(18, 12.0, 22, 12.0, 0.0, 0.9, density, rng))

    cloud = np.vstack(parts)
    rng.shuffle(cloud)
    return cloud
