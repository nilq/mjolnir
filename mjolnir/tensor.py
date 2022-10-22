import math
from typing import Optional, Union

import compiler
import numpy as np
from compiler import LLVMBuffer, Node

Operable = Union[list, np.ndarray, Node, "Tensor"]


class Tensor:
    def __init__(self, content: list | np.ndarray, requires_grad: bool = False) -> None:
        self.gradient: Optional["Tensor"] = None
        self.requires_grad = requires_grad

        self.buffer = LLVMBuffer(content)

    @property
    def content(self):
        return (
            np.ctypeslib.as_array(self.buffer.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .copy()
        )

    @property
    def shape(self) -> (int, int):
        return self.buffer.shape

    def __add__(self, other: Operable):
        out = self._binary_op(other, "+")
        return out

    def __mul__(self, other: Operable):
        out = self._binary_op(other, "*")
        return out

    def __radd__(self, other: Operable):
        out = self._binary_op(other, "+")
        return out

    def __rmul__(self, other: Operable):
        out = self._binary_op(other, "*")
        return out

    def __repr__(self):
        return f"Tensor<{self.shape}>"

    def _binary_op(self, other: Operable, op: str) -> Node:
        return Node(op=op, incoming=(self, other))

    def _jit_binary_op(
        self, other: Union[list, np.ndarray, "Tensor"], op: str
    ) -> ("Tensor", "Tensor"):
        other: "Tensor" = other if isinstance(other, Tensor) else Tensor(other)
        target: "Tensor" = Tensor(np.zeros(self.shape))

        return other, compiler.tensor_llvm_op(target, [self, other], op)


if __name__ == "__main__":
    a = Tensor([[1, 2], [3, 4]])
    d = Tensor([[2, 2], [2, 2]])

    test = d * (d * d) * d

    print(test)

    mod = compiler.create_module("tissemand")
    test2 = compiler.jit_flat_tensor_op(mod, test, a.shape)

    c = Tensor(test2.numpy())

    print()
    print(c)
    print(c.buffer.numpy())
