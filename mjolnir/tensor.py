import math
from typing import TYPE_CHECKING, Optional, Union, Iterable

from . import compiler, graph
import numpy as np

if TYPE_CHECKING:
    from .graph import Node

Operable = Union[list, np.ndarray, "Node", "Tensor"]


class Tensor:
    def __init__(self, content: list | np.ndarray, requires_grad: bool = False) -> None:
        self.gradient: Optional["Tensor"] = None
        self.requires_grad = requires_grad

        self._buffer = compiler.LLVMBuffer(content)

    @classmethod
    def zeros(cls, shape: Iterable[int], **kwargs) -> "Tensor":
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, shape: Iterable[int], **kwargs) -> "Tensor":
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def eye(cls, shape: int | Iterable[int], **kwargs) -> "Tensor":
        rows, cols = (shape, shape) if isinstance(shape, int) else shape
        return cls(np.eye(rows, cols, dtype=np.float32), **kwargs)

    @classmethod
    def empty(cls, shape: Iterable[int], **kwargs) -> "Tensor":
        return cls(np.empty(shape, dtype=np.float32), **kwargs)

    @property
    def content(self):
        return (
            np.ctypeslib.as_array(self._buffer.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .copy()
        )

    @property
    def shape(self) -> (int, int):
        return self._buffer.shape

    def __add__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "+")
        return out

    def __sub__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "-")
        return out

    def __mul__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "*")
        return out

    def __truediv__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "/")
        return out

    def __radd__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "+")
        return out

    def __rsub__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "-")
        return out

    def __rmul__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "*")
        return out

    def __rtruediv__(self, other: Operable) -> "Node":
        out = self._binary_op(other, "/")
        return out

    def __repr__(self) -> str:
        return f"Tensor<{self.shape}>"

    def _binary_op(self, other: Operable, op: str) -> "Node":
        return graph.Node(op=op, incoming=(self, other), shape=self.shape)