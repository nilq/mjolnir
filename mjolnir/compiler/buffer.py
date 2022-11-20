"""A module containing abstract LLVM array buffer."""
import ctypes
import math
from typing import Iterable, TypeVar

import numpy as np

from mjolnir.compiler.types import DataType


class LLVMBuffer:
    """LLVM buffer for turning arrays into cool low-level arrays."""

    def __init__(self, content: Iterable, dtype: DataType) -> None:
        content: np.ndarray = (
            content
            if isinstance(content, np.ndarray)
            else np.asarray(content, dtype.numpy)
        )

        self.shape: (int, ...) = content.shape
        self.buffer = (ctypes.c_float * math.prod(self.shape))()

        ctypes.memmove(
            self.buffer,
            content.astype(dtype.numpy).ctypes.data,
            math.prod(self.shape) * 4,
        )

    def to_numpy(self):
        return (
            np.ctypeslib.as_array(self.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .copy()
        )
