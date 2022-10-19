import ctypes

from llvmlite import ir
from typing import Union

import llvmlite.binding as llvm
import ctypes

import ops


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

_target: llvm.Target = llvm.Target.from_default_triple()

target_machine: llvm.TargetMachine = _target.create_target_machine()
engine: llvm.ExecutionEngine = llvm.create_mcjit_compiler(
    llvm.parse_assembly(""), target_machine
)
optimizer: llvm.ModulePassManager = llvm.ModulePassManager()

builder = llvm.PassManagerBuilder()
builder.opt_level = 3
builder.loop_vectorize = True
builder.populate(optimizer)


def llvm_op(buffer, op):
    arg_type = ir.FloatType()

    fn_type: ir.FunctionType = ir.FunctionType(ir.FloatType(), [arg_type] * len(buffer))
    module: ir.Module = ir.Module(name=__file__)
    fn: ir.Function = ir.Function(module, fn_type, name="op")

    body = fn.append_basic_block(name="entry")

    body_builder = ir.IRBuilder(body)
    body_builder.ret(ops.OPS_LOOKUP[op](body_builder, *fn.args))

    llvm_ir = str(module)

    llvm_module = llvm.parse_assembly(llvm_ir)
    llvm_module.verify()

    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    fn_pointer = engine.get_function_address("op")

    c_fn = ctypes.CFUNCTYPE(ctypes.c_float, *[ctypes.c_float] * len(buffer))(fn_pointer)
    result = c_fn(*buffer)

    engine.remove_module(llvm_module)

    return result


class Value:
    """Stupid JIT'ed auto-grad'ed value."""

    def __init__(
        self, content: float, children: ("Value", ...) = (), op: str = ""
    ) -> None:
        self.content = content
        self.gradient = 0.0

        self._children = set(children)
        self._op = op
        self._backward = lambda: None

    def backward(self):
        topology = []
        visited = set()

        def walk(node: "Value"):
            if node not in visited:
                visited.add(node)

                for child in node._children:
                    walk(child)

                topology.append(node)

        walk(self)

        self.gradient = 1.0

        for node in reversed(topology):
            node._backward()

    def llvm_binary_op(
        self, other: Union[float, "Value"], op: str
    ) -> ("Value", "Value"):
        other: "Value" = isinstance(other, Value) and other or Value(other)
        return other, Value(
            llvm_op([self.content, other.content], op), (self, other), op
        )

    def __add__(self, other) -> "Value":
        other, out = self.llvm_binary_op(other, "+")

        def backward() -> None:
            self.gradient += out.gradient
            other.gradient += out.gradient

        out._backward = backward

        return out

    def __mul__(self, other) -> "Value":
        other, out = self.llvm_binary_op(other, "*")

        def backward() -> None:
            self.gradient += other.gradient * out.gradient
            other.gradient += self.gradient * out.gradient

        out._backward = backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(other, (int, float)), "TODO: Value powers."

        out: "Value" = Value(llvm_op([self.content, other], "**"), (self,), "**")

        def backward() -> None:
            self.gradient += (other * self.content ** (other - 1)) * out.gradient

        out._backward = backward

        return out

    def __truediv__(self, other):
        return self * other ** (-1)

    def __repr__(self):
        return f"Value({self.content}, gradient={self.gradient})"


if __name__ == "__main__":
    import math

    a = Value(math.pi)
    b = Value(2)

    c = a**2 / (b * 10)

    c.backward()

    print(c)
