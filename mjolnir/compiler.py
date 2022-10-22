import ctypes
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

if TYPE_CHECKING:
    from tensor import Tensor


class LLVMBuffer:
    def __init__(self, content: list | np.ndarray) -> None:
        content: np.ndarray = (
            content
            if isinstance(content, np.ndarray)
            else np.asarray(content, np.float32)
        )

        self.shape: (int, ...) = content.shape
        self.buffer = (ctypes.c_float * math.prod(self.shape))()

        ctypes.memmove(
            self.buffer,
            content.astype(np.float32).ctypes.data,
            math.prod(self.shape) * 4,
        )

    def numpy(self):
        return (
            np.ctypeslib.as_array(self.buffer)[: math.prod(self.shape)]
            .reshape(self.shape)
            .copy()
        )


@dataclass
class Node:
    op: str
    incoming: (Union["Node", LLVMBuffer], ...)

    def buffers(self) -> list[LLVMBuffer]:
        return list(
            itertools.chain(
                *[
                    [x.buffer] if not isinstance(x, Node) else x.buffers()
                    for x in self.incoming
                ]
            )
        )


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


def ir_float(n: float) -> ir.Constant:
    return ir.Constant(ir.FloatType(), n)


def ir_int32(n: int) -> ir.Constant:
    return ir.Constant(ir.IntType(32), n)


BINARY_OPS = {
    "+": lambda ir_builder, *args: ir_builder.fadd(*args),
    "-": lambda ir_builder, *args: ir_builder.fsub(*args),
    "*": lambda ir_builder, *args: ir_builder.fmul(*args),
    "/": lambda ir_builder, *args: ir_builder.fdiv(*args),
    "**": lambda ir_builder, *args: ir_builder.call(
        ir_builder._block.module.declare_intrinsic("llvm.pow", [ir.FloatType()]), args
    ),
}

UNARY_OPS = {
    "relu": lambda ir_builder, x: ir_builder.select(
        ir_builder.fcmp_ordered("<=", ir_float(0.0), x), x, ir_float(0)
    ),
}

OPS_LOOKUP = {"": lambda ir_builder, *args: args, **BINARY_OPS, **UNARY_OPS}  # Noop.


def _build_primitive_op(module: ir.Module, op: str) -> None:
    fn = ir.Function(
        module,
        ir.FunctionType(
            ir.VoidType(),
            [ir.PointerType(ir.FloatType())] * 3
            + [ir.IntType(32)],  # -> (target, a, b, length)
        ),
        name=f"primitive[{op}]",
    )

    start_builder = ir.IRBuilder(fn.append_basic_block(name="entry"))
    body_builder = ir.IRBuilder(fn.append_basic_block(name="inner"))
    exit_builder = ir.IRBuilder(fn.append_basic_block(name="exit"))

    start_builder.branch(body_builder._block)
    exit_builder.ret_void()

    start_i = ir.Constant(ir.IntType(32), 0)

    phi = body_builder.phi(ir.IntType(32))
    phi.add_incoming(start_i, start_builder.block)

    target_val, a, b, length = fn.args

    ap = body_builder.load(body_builder.gep(a, [phi]))
    bp = body_builder.load(body_builder.gep(b, [phi]))

    val = OPS_LOOKUP[op](body_builder, ap, bp)
    body_builder.store(val, body_builder.gep(target_val, [phi]))

    phi_new = body_builder.add(phi, ir.Constant(ir.IntType(32), 1))
    phi.add_incoming(phi_new, body_builder.block)

    body_builder.cbranch(
        body_builder.icmp_unsigned("==", phi, length),
        exit_builder.block,
        body_builder.block,
    )


def create_module(name: str = __file__) -> ir.Module:
    module: ir.Module = ir.Module(name=name)

    for op in BINARY_OPS:
        _build_primitive_op(module, op)

    return module


def flatten_buffer_operations(buffers: list[LLVMBuffer], *args):
    def compile_tree_into(
        target,
        body_builder: ir.IRBuilder,
        node: Union["Tensor", Node],
        shape,
        level: int = 0,
    ):
        nonlocal buffers

        buffer_left, buffer_right = None, None

        if isinstance(node.incoming[0], Node):
            buffer_left = body_builder.alloca(
                ir.FloatType(), math.prod(shape), name=f"tmp_{level}_left"
            )
            # Compiling downwards, storing into left value buffer
            compile_tree_into(
                buffer_left, body_builder, node.incoming[0], shape, level + 1
            )

        if isinstance(node.incoming[1], Node):
            buffer_right = body_builder.alloca(
                ir.FloatType(), math.prod(shape), name=f"tmp_{level}_right"
            )

            compile_tree_into(
                buffer_right, body_builder, node.incoming[1], shape, level + 1
            )

        operand_left = buffer_left if buffer_left else buffers[level]
        operand_right = buffer_right if buffer_right else buffers[level + 1]

        length = ir.Constant(ir.IntType(32), math.prod(shape) - 1)

        body_builder.call(
            body_builder.module.globals[f"primitive[{node.op}]"],
            [target, operand_left, operand_right, length],
        )

    compile_tree_into(*args)


def jit_flat_tensor_op(module: ir.Module, root: Node, shape: (int, ...)) -> LLVMBuffer:
    result: LLVMBuffer = LLVMBuffer(np.zeros(shape))
    buffers = root.buffers()

    print(buffers)

    fn = ir.Function(
        module,
        ir.FunctionType(
            ir.VoidType(),
            [ir.PointerType(ir.FloatType())] * (1 + len(buffers)),  # target, *buffers
        ),
        name="main",
    )

    start_builder = ir.IRBuilder(fn.append_basic_block(name="entry"))
    body_builder = ir.IRBuilder(fn.append_basic_block(name="inner"))
    exit_builder = ir.IRBuilder(fn.append_basic_block(name="exit"))

    start_builder.branch(body_builder._block)
    exit_builder.ret_void()

    target = fn.args[0]
    flatten_buffer_operations(fn.args[1:], target, body_builder, root, shape)

    body_builder.branch(exit_builder.block)

    llvm_ir = str(module)

    print(llvm_ir)

    llvm_module = llvm.parse_assembly(llvm_ir)
    llvm_module.verify()

    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    fn_pointer = engine.get_function_address("main")
    source_buffers = [result.buffer] + [x.buffer for x in buffers]

    arg_types = [ctypes.POINTER(ctypes.c_float) for _ in source_buffers]

    c_func = ctypes.CFUNCTYPE(*arg_types)(fn_pointer)
    c_func(*source_buffers)

    engine.remove_module(llvm_module)

    return result
