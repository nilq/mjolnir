"""A somewhat lacking tensor module."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np

from mjolnir.compiler.buffer import LLVMBuffer
from mjolnir.compiler import graph
from mjolnir.compiler.types import BrainFloat16, DataType

Operable: type = Union["Node", "Tensor", LLVMBuffer]


class Tensor:
    def __init__(
        self,
        content: Iterable,
        requires_grad: bool = False,
        dtype: DataType = BrainFloat16(),
    ) -> None:
        self.gradient: Optional[Tensor] = None
        self.requires_grad = requires_grad
        self.dtype = dtype

        self._buffer = LLVMBuffer(content, dtype=dtype)

    @classmethod
    def zeros(
        cls, shape: Iterable[int], dtype: Optional[DataType] = None, **kwargs
    ) -> Tensor:
        return cls(np.zeros(shape, dtype=dtype), **kwargs)

    @classmethod
    def ones(
        cls, shape: Iterable[int], dtype: Optional[DataType] = None, **kwargs
    ) -> Tensor:
        return cls(np.ones(shape, dtype=dtype), **kwargs)

    @classmethod
    def eye(
        cls, shape: int | Iterable[int], dtype: Optional[DataType] = None, **kwargs
    ) -> Tensor:
        rows, cols = (shape, shape) if isinstance(shape, int) else shape
        return cls(np.eye(rows, cols, dtype), **kwargs)

    @classmethod
    def empty(
        cls, shape: Iterable[int], dtype: Optional[DataType] = None, **kwargs
    ) -> Tensor:
        return cls(np.empty(shape, dtype=dtype), **kwargs)

    @property
    def content(self):
        return (
            np.ctypeslib.as_array(self._buffer.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .copy()
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._buffer.shape

    def __repr__(self) -> str:
        return f"Tensor<{self.shape}>"


def _build_binary_op(op: str):
    def binary(self, other):
        return graph.Node(op=op, incoming=[self, other], shape=self.shape, dtype=self.dtype)

    return binary

for name, op in {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}.items():
    setattr(Tensor, f"__{name}__", _build_binary_op(op))
    setattr(Tensor, f"__r{name}__", _build_binary_op(op))
