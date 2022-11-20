"""Module containing simple computation graph."""
from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from mjolnir import tensor
from mjolnir.compiler.types import DataType
from mjolnir.compiler import compiler
from llvmlite import ir


global_module: Optional["ir.Module"] = None


NodeType = Union[tensor.Tensor, "Node"]


@dataclass
class Node:
    op: str
    incoming: list[Node | compiler.LLVMBuffer]
    shape: tuple[int, ...]
    dtype: DataType

    def __post_init__(self):
        global global_module

        if global_module is None:
            global_module = compiler.populate_global_module("global_module")

    def buffers(self) -> list[compiler.LLVMBuffer]:
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
            compiler.jit_graph(
                global_module, self, self.incoming[0].shape
            ).to_numpy()
        )

    def print(self) -> None:
        def _print(node: Node, level: int = 0) -> None:
            print("  " * level + f"({node.op})" + ":")

            new_level: int = level + 1

            for incoming in node.incoming:
                if isinstance(incoming, Node):
                    _print(incoming, new_level)
                else:
                    print("  " * new_level + str(incoming))

        _print(self)

    @classmethod
    def build_binary_op(cls, op: str):
        def binary(self, other):
            return cls(op=op, incoming=[self, other], shape=self.shape, dtype=self.dtype)

        return binary


for name, op in [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/")]:
    setattr(Node, f"__{name}__", Node.build_binary_op(op))
    setattr(Node, f"__r{name}__", Node.build_binary_op(op))
