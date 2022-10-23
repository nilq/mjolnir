import itertools
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from . import compiler, tensor

if TYPE_CHECKING:
    from .compiler import LLVMBuffer
    from llvmlite import ir


global_module: Optional["ir.Module"] = None


@dataclass
class Node:
    op: str
    incoming: (Union["Node", "LLVMBuffer"], ...)
    shape: (int, ...)

    def __post_init__(self):
        global global_module

        if global_module is None:
            global_module = compiler.create_module("global_module")

    def buffers(self) -> list["LLVMBuffer"]:
        return list(
            itertools.chain(
                *[
                    [x._buffer] if not isinstance(x, Node) else x.buffers()
                    for x in self.incoming
                ]
            )
        )

    @functools.cached_property
    def tensor(self) -> tensor.Tensor:
        return tensor.Tensor(
            compiler.jit_flat_tensor_op(
                global_module, self, self.incoming[0].shape
            ).numpy()
        )

    def print(self) -> None:
        def _print(node: Node, level: int = 0) -> None:
            print('  ' * level + f"({node.op})" + ":")

            new_level: int = level + 1

            for incoming in node.incoming:
                if isinstance(incoming, Node):
                    _print(incoming, new_level)
                else:
                    print("  " * new_level + str(incoming))

        _print(self)

    @classmethod
    def _build_binary_op(cls, op: str):
        def binary(self, other):
            return cls(op=op, incoming=(self, other), shape=self.shape)
        return binary


for name, op in [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/")]:
    setattr(Node, f"__{name}__", Node._build_binary_op(op))
    setattr(Node, f"__r{name}__", Node._build_binary_op(op))
