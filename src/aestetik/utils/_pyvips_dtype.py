"""Mapping from libvips band formats to numpy dtypes.

Shared by the morphology and visualization helpers that wrap pyvips
``Image`` outputs into numpy arrays. Kept here so the lookup table has
a single source of truth.
"""
from __future__ import annotations

import numpy as np

FORMAT_TO_DTYPE: dict[str, type] = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}

__all__ = ["FORMAT_TO_DTYPE"]
