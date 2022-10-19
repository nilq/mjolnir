import ctypes
import math
import numpy as np
from llvmlite import ir
from typing import Union, Iterable

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


def discrete_llvm_op(buffer, op) -> float:
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


def tensor_llvm_op(target: "Tensor", sources: list["Tensor"], op: str):
        module = ir.Module(name=__file__)

        # LLVM to store resulting value of operation applied to given sources.
        # All the inputs are pointers, first one is the target.
        fn = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.FloatType())] * (1 + len(sources))), name="op")

        start_builder = ir.IRBuilder(fn.append_basic_block(name="entry"))
        body_builder = ir.IRBuilder(fn.append_basic_block(name="inner"))
        exit_builder = ir.IRBuilder(fn.append_basic_block(name="exit"))

        start_builder.branch(body_builder._block)
        exit_builder.ret_void()

        start_i = ir.Constant(ir.IntType(32), 0)
        end_i = ir.Constant(ir.IntType(32), math.prod(target.shape))

        phi = body_builder.phi(ir.IntType(32))
        phi.add_incoming(start_i, start_builder.block)

        ###

        r, a, b = fn.args
        ap = body_builder.load(body_builder.gep(a, [phi]))
        bp = body_builder.load(body_builder.gep(b, [phi]))

        val = ops.OPS_LOOKUP[op](body_builder, ap, bp)
        body_builder.store(val, body_builder.gep(r, [phi]))

        ###

        phi_new = body_builder.add(phi, ir.Constant(ir.IntType(32), 1))
        phi.add_incoming(phi_new, body_builder.block)

        body_builder.cbranch(body_builder.icmp_unsigned("==", phi, end_i), exit_builder.block, body_builder.block)

        llvm_ir = str(module)

        print(llvm_ir)

        llvm_module = llvm.parse_assembly(llvm_ir)
        llvm_module.verify()

        engine.add_module(llvm_module)
        engine.finalize_object()
        engine.run_static_constructors()

        fn_pointer = engine.get_function_address("op")
        source_buffers = [target._buffer] + [x._buffer for x in sources]

        arg_types = [ctypes.POINTER(ctypes.c_float) for _ in sources]
        cfunc = ctypes.CFUNCTYPE(*arg_types)(fn_pointer)

        cfunc(*source_buffers)

        return target


class Tensor:
    def __init__(self, content: Union[list, np.ndarray]) -> None:
        content = content if isinstance(content, np.ndarray) else np.asarray(content, np.float32)

        self.shape: (float, ...) = content.shape
        self._buffer = (ctypes.c_float * math.prod(self.shape))()

        ctypes.memmove(self._buffer, content.astype(np.float32).ctypes.data, math.prod(self.shape) * 4)

    def llvm_binary_op(
        self, other: Union[list, np.ndarray, "Tensor"], op: str
    ) -> ("Tensor", "Tensor"):
        other: "Tensor" = other if isinstance(other, Tensor) else Tensor(other)
        target: "Tensor" = Tensor(np.zeros(self.shape))
        return other, tensor_llvm_op(target, [self, other], op)

    def __add__(self, other: Union[list, np.ndarray, "Tensor"]):
        other, out = self.llvm_binary_op(other, "+")
        return out

    def __repr__(self):
        return f"Tensor{self.shape}"

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
            discrete_llvm_op([self.content, other.content], op), (self, other), op
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

        out: "Value" = Value(discrete_llvm_op([self.content, other], "**"), (self,), "**")

        def backward() -> None:
            self.gradient += (other * self.content ** (other - 1)) * out.gradient

        out._backward = backward

        return out

    def __truediv__(self, other):
        return self * other ** (-1)

    def __repr__(self):
        return f"Value({self.content}, gradient={self.gradient})"


if __name__ == "__main__":
    # Add these two tensors ...
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[1, 1], [1, 1]])

    test = a + b

    # This is definitely correct.
    # I am the new linear algebra professor.
    print("Result:", list(test._buffer))
