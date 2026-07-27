"""PCD (Point Cloud Data) 读写工具, 纯 Python 实现, 只依赖 numpy.

支持 PCD v0.7 的三种 DATA 格式:
  * ascii
  * binary
  * binary_compressed (内置 LZF 解压, 无需 python-lzf)
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

_TYPE_MAP = {
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("F", 4): "f4", ("F", 8): "f8",
}
_INV_TYPE_MAP = {v: k for k, v in _TYPE_MAP.items()}


@dataclass
class PointCloud:
    """点云容器: xyz 为 (N,3) float32/64, extra 保存其它字段."""

    xyz: np.ndarray
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        keys = ",".join(self.extra) or "-"
        return f"<PointCloud N={len(self)} extra={keys}>"


# --------------------------------------------------------------------------
# LZF 解压 (libLZF, PCL 的 binary_compressed 使用的算法)
# --------------------------------------------------------------------------
def lzf_decompress(src: bytes, expected_len: int) -> bytes:
    """纯 Python 版 LZF 解压. 与 liblzf 的 lzf_decompress 行为一致."""
    dst = bytearray(expected_len)
    si, di, slen = 0, 0, len(src)
    while si < slen:
        ctrl = src[si]
        si += 1
        if ctrl < 32:  # literal run: 后面 ctrl+1 个原始字节
            n = ctrl + 1
            dst[di:di + n] = src[si:si + n]
            si += n
            di += n
        else:  # back reference
            length = ctrl >> 5
            if length == 7:
                length += src[si]
                si += 1
            ref = di - ((ctrl & 0x1F) << 8) - src[si] - 1
            si += 1
            if ref < 0:
                raise ValueError("LZF 解压失败: 引用越界")
            # 逐字节复制 (可能自重叠, 不能用切片)
            for _ in range(length + 2):
                dst[di] = dst[ref]
                di += 1
                ref += 1
    if di != expected_len:
        raise ValueError(f"LZF 解压长度不符: {di} != {expected_len}")
    return bytes(dst)


def _parse_header(fh) -> (dict, int):
    header: Dict[str, object] = {}
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


def read_pcd(path: str) -> PointCloud:
    """读取 PCD 文件, 返回 PointCloud."""
    with open(path, "rb") as fh:
        header, data_offset = _parse_header(fh)

        fields: List[str] = header["FIELDS"].split()
        sizes = [int(v) for v in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(v) for v in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        n_points = int(header["POINTS"]) if "POINTS" in header else \
            int(header["WIDTH"]) * int(header["HEIGHT"])
        data_kind = str(header["DATA"]).lower()

        np_names, np_formats = [], []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            dt = _TYPE_MAP[(typ.upper(), size)]
            for k in range(cnt):
                np_names.append(name if cnt == 1 else f"{name}_{k}")
                np_formats.append(dt)
        dtype = np.dtype({"names": np_names, "formats": np_formats})

        if data_kind == "ascii":
            raw = fh.read().decode("utf-8", "replace")
            rows = [r for r in re.split(r"[\r\n]+", raw) if r.strip()]
            arr = np.zeros(len(rows), dtype=dtype)
            for i, row in enumerate(rows):
                vals = row.split()
                for j, name in enumerate(np_names):
                    arr[name][i] = float(vals[j])
            arr = arr[:n_points] if len(arr) >= n_points else arr
        elif data_kind == "binary":
            buf = fh.read(n_points * dtype.itemsize)
            arr = np.frombuffer(buf, dtype=dtype, count=n_points)
        elif data_kind == "binary_compressed":
            comp_size, uncomp_size = struct.unpack("II", fh.read(8))
            blob = lzf_decompress(fh.read(comp_size), uncomp_size)
            # binary_compressed 是"按列存储"的, 需要转回按点存储
            arr = np.zeros(n_points, dtype=dtype)
            pos = 0
            for name in np_names:
                sub = np.dtype(dtype[name])
                nbytes = n_points * sub.itemsize
                arr[name] = np.frombuffer(blob[pos:pos + nbytes], dtype=sub, count=n_points)
                pos += nbytes
        else:
            raise ValueError(f"不支持的 DATA 类型: {data_kind}")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    extra = {n: np.asarray(arr[n])[finite] for n in np_names if n not in ("x", "y", "z")}
    return PointCloud(xyz=xyz[finite], extra=extra)


def write_pcd(path: str, cloud: "PointCloud | np.ndarray", binary: bool = True,
              extra_fields: Dict[str, np.ndarray] | None = None) -> str:
    """写出 PCD 文件. cloud 可以是 PointCloud 或 (N,3) 数组."""
    if isinstance(cloud, PointCloud):
        xyz, extra = cloud.xyz, dict(cloud.extra)
    else:
        xyz, extra = np.asarray(cloud, dtype=np.float64), {}
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
            for row in arr:
                fh.write((" ".join(f"{v:.6f}" for v in row) + "\n").encode())
    return path
