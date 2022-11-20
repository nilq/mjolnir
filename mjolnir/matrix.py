"""A module for the more complete matrix implementation."""
from __future__ import annotations

import math
import numpy as np
from mjolnir.tensor import Tensor
from mjolnir.compiler.types import BrainFloat16

from typing import Iterable


class Matrix(Tensor):
    """The Matrix."""
    def __init__(self, content: Iterable, requires_grad: bool = False, dtype=BrainFloat16()):
        super().__init__(content=content, requires_grad=requires_grad, dtype=dtype)

    def transpose(self) -> None:
        self._buffer.buffer = (
            np.ctypeslib.as_array(self._buffer.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .swapaxes(-1, -2)
        )

    @property
    def T(self) -> Matrix:
        out = Matrix(self.content)
        out.transpose()

        return out

    def __repr__(self):
        return f"Matrix<{self.shape}>"
