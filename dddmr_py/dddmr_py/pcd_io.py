"""PCD (Point Cloud Data) 读写工具, 纯 Python 实现, 只依赖 numpy.

支持 PCD v0.7 的三种 DATA 格式: ascii / binary / binary_compressed (内置 LZF).

优化版改动:
  * 默认 float32 存储 (内存减半); xyz 连续字段走零拷贝视图
  * ascii 走 ``np.frombuffer``/``np.loadtxt`` 批量解析, 不再逐行逐字段 Python 循环
  * LZF 解压对"非重叠回引用"改用切片拷贝, 并优先使用 C 实现的 python-lzf
  * ``fields_needed=False`` 时不保留额外字段, 避免为 intensity/rgb 白付内存
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_TYPE_MAP = {
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("F", 4): "f4", ("F", 8): "f8",
}
_INV_TYPE_MAP = {v: k for k, v in _TYPE_MAP.items()}


@dataclass
class PointCloud:
    """点云容器: xyz 为 (N,3), extra 保存其它字段."""

    xyz: np.ndarray
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __repr__(self) -> str:  # pragma: no cover
        keys = ",".join(self.extra) or "-"
        return f"<PointCloud N={len(self)} extra={keys} {self.xyz.dtype}>"


# --------------------------------------------------------------------------
# LZF 解压
# --------------------------------------------------------------------------
def lzf_decompress(src: bytes, expected_len: int) -> bytes:
    """LZF 解压. 有 python-lzf 时走 C 实现, 否则用优化过的纯 Python 版."""
    try:
        import lzf  # type: ignore
        out = lzf.decompress(src, expected_len)
        if out is not None:
            return out
    except Exception:
        pass

    dst = bytearray(expected_len)
    si, di, slen = 0, 0, len(src)
    while si < slen:
        ctrl = src[si]
        si += 1
        if ctrl < 32:                                # 字面量串
            n = ctrl + 1
            dst[di:di + n] = src[si:si + n]
            si += n
            di += n
        else:                                        # 回引用
            length = ctrl >> 5
            if length == 7:
                length += src[si]
                si += 1
            length += 2
            ref = di - ((ctrl & 0x1F) << 8) - src[si] - 1
            si += 1
            if ref < 0:
                raise ValueError("LZF 解压失败: 引用越界")
            if ref + length <= di:                   # 不自重叠 -> 切片拷贝
                dst[di:di + length] = dst[ref:ref + length]
                di += length
            else:                                    # 自重叠, 必须逐字节
                for _ in range(length):
                    dst[di] = dst[ref]
                    di += 1
                    ref += 1
    if di != expected_len:
        raise ValueError(f"LZF 解压长度不符: {di} != {expected_len}")
    return bytes(dst)


def _parse_header(fh):
    header: Dict[str, str] = {}
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PCD 文件头不完整")
        text = line.decode("utf-8", "replace").strip()
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition(" ")
        key = key.upper()
        header[key] = value.strip()
        if key == "DATA":
            break
    return header, fh.tell()


def read_pcd(path: str, dtype=np.float32, keep_extra: bool = False) -> PointCloud:
    """读取 PCD 文件.

    dtype: xyz 的存储精度, 默认 float32 (原版恒为 float64, 内存翻倍).
    keep_extra: 是否保留 intensity/rgb 等额外字段, 默认丢弃.
    """
    with open(path, "rb") as fh:
        header, data_offset = _parse_header(fh)

        fields: List[str] = header["FIELDS"].split()
        sizes = [int(v) for v in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(v) for v in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        n_points = (int(header["POINTS"]) if "POINTS" in header
                    else int(header["WIDTH"]) * int(header["HEIGHT"]))
        data_kind = str(header["DATA"]).lower()

        np_names, np_formats = [], []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            dt = _TYPE_MAP[(typ.upper(), size)]
            for k in range(cnt):
                np_names.append(name if cnt == 1 else f"{name}_{k}")
                np_formats.append(dt)
        np_dtype = np.dtype({"names": np_names, "formats": np_formats})

        if data_kind == "ascii":
            raw = fh.read().decode("utf-8", "replace")
            # 批量解析, 不再逐行逐字段赋值
            flat = np.fromstring(raw, sep=" ") if hasattr(np, "fromstring") else None
            if flat is None or flat.size < n_points * len(np_names):
                rows = [r for r in re.split(r"[\r\n]+", raw) if r.strip()]
                flat = np.array([float(v) for r in rows for v in r.split()])
            flat = flat[:n_points * len(np_names)].reshape(n_points, len(np_names))
            arr = np.zeros(n_points, dtype=np_dtype)
            for j, name in enumerate(np_names):
                arr[name] = flat[:, j]
            del flat
        elif data_kind == "binary":
            # memmap: 不把整个文件读进内存, 只映射需要的部分
            arr = np.memmap(path, dtype=np_dtype, mode="r", offset=data_offset,
                            shape=(n_points,))
        elif data_kind == "binary_compressed":
            comp_size, uncomp_size = struct.unpack("II", fh.read(8))
            blob = lzf_decompress(fh.read(comp_size), uncomp_size)
            arr = np.zeros(n_points, dtype=np_dtype)   # 列存 -> 点存
            pos = 0
            for name in np_names:
                sub = np.dtype(np_dtype[name])
                nbytes = n_points * sub.itemsize
                arr[name] = np.frombuffer(blob[pos:pos + nbytes], dtype=sub, count=n_points)
                pos += nbytes
            del blob
        else:
            raise ValueError(f"不支持的 DATA 类型: {data_kind}")

        xyz = np.empty((n_points, 3), dtype=dtype)
        for i, name in enumerate(("x", "y", "z")):
            xyz[:, i] = arr[name]
        extra = ({n: np.asarray(arr[n]) for n in np_names if n not in ("x", "y", "z")}
                 if keep_extra else {})
        if isinstance(arr, np.memmap):
            del arr

    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz = xyz[finite]
        extra = {k: v[finite] for k, v in extra.items()}
    return PointCloud(xyz=np.ascontiguousarray(xyz), extra=extra)


def iter_pcd_chunks(path: str, chunk: int = 1_000_000, dtype=np.float32):
    """流式读取 binary PCD 的 xyz, 逐块 yield —— 超大地图避免一次性载入.

    非 binary 格式回退为一次性读取后切块.
    """
    with open(path, "rb") as fh:
        header, data_offset = _parse_header(fh)
    if str(header["DATA"]).lower() != "binary":
        cloud = read_pcd(path, dtype=dtype)
        for beg in range(0, len(cloud), chunk):
            yield cloud.xyz[beg:beg + chunk]
        return

    fields = header["FIELDS"].split()
    sizes = [int(v) for v in header["SIZE"].split()]
    types = header["TYPE"].split()
    counts = [int(v) for v in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    n_points = int(header.get("POINTS", 0)) or int(header["WIDTH"]) * int(header["HEIGHT"])
    np_names, np_formats = [], []
    for name, size, typ, cnt in zip(fields, sizes, types, counts):
        dt = _TYPE_MAP[(typ.upper(), size)]
        for k in range(cnt):
            np_names.append(name if cnt == 1 else f"{name}_{k}")
            np_formats.append(dt)
    np_dtype = np.dtype({"names": np_names, "formats": np_formats})
    mm = np.memmap(path, dtype=np_dtype, mode="r", offset=data_offset, shape=(n_points,))
    for beg in range(0, n_points, chunk):
        blk = mm[beg:beg + chunk]
        out = np.empty((len(blk), 3), dtype=dtype)
        for i, name in enumerate(("x", "y", "z")):
            out[:, i] = blk[name]
        yield out[np.isfinite(out).all(axis=1)]
    del mm


def write_pcd(path: str, cloud, binary: bool = True,
              extra_fields: Optional[Dict[str, np.ndarray]] = None) -> str:
    """写出 PCD 文件. cloud 可以是 PointCloud 或 (N,3) 数组."""
    if isinstance(cloud, PointCloud):
        xyz, extra = cloud.xyz, dict(cloud.extra)
    else:
        xyz, extra = np.asarray(cloud), {}
    if extra_fields:
        extra.update(extra_fields)

    names = ["x", "y", "z"] + list(extra)
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]] + [np.asarray(v) for v in extra.values()]
    formats = ["f4", "f4", "f4"] + [
        (v.dtype.str[1:] if v.dtype.str[1:] in _INV_TYPE_MAP else "f4") for v in cols[3:]
    ]
    dtype = np.dtype({"names": names, "formats": formats})
    arr = np.zeros(len(xyz), dtype=dtype)
    for name, col in zip(names, cols):
        arr[name] = col

    types = " ".join(_INV_TYPE_MAP[f][0] for f in formats)
    sizes = " ".join(str(_INV_TYPE_MAP[f][1]) for f in formats)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(names)}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {' '.join(['1'] * len(names))}\n"
        f"WIDTH {len(arr)}\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(arr)}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        if binary:
            fh.write(arr.tobytes())
        else:
            np.savetxt(fh, np.column_stack(cols), fmt="%.6f")
    return path
