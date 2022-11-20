"""The compiler."""
import ctypes
import math

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir

from mjolnir.compiler import ops
from mjolnir.compiler import graph
from mjolnir.compiler.buffer import LLVMBuffer
from typing import Iterable

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


def build_rolling_function(
    module: ir.Module,
    name: str,
    parameter_types: list[ir.types.Type],
    llvm_compile_func,
) -> None:
    fn = ir.Function(module, ir.FunctionType(ir.VoidType(), parameter_types), name=name)

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

    val = llvm_compile_func(body_builder, ap, bp)

    body_builder.store(val, body_builder.gep(target_val, [phi]))

    phi_new = body_builder.add(phi, ir.Constant(ir.IntType(32), 1))
    phi.add_incoming(phi_new, body_builder.block)

    body_builder.cbranch(
        body_builder.icmp_unsigned("==", phi, length),
        exit_builder.block,
        body_builder.block,
    )


def flatten_graph(buffers: list[LLVMBuffer], *args):
    def compile_tree_into(
        target,
        body_builder: ir.IRBuilder,
        root: graph.NodeType,
        shape,
        level: int = 0,
    ):
        nonlocal buffers

        buffer_left, buffer_right = None, None

        if isinstance(root.incoming[0], graph.Node):
            buffer_left = body_builder.alloca(
                ir.FloatType(), math.prod(shape), name=f"tmp_{level}_left"
            )
            # Compiling downwards, storing into left value buffer
            compile_tree_into(
                buffer_left, body_builder, root.incoming[0], shape, level + 1
            )

        if isinstance(root.incoming[1], graph.Node):
            buffer_right = body_builder.alloca(
                ir.FloatType(), math.prod(shape), name=f"tmp_{level}_right"
            )

            compile_tree_into(
                buffer_right, body_builder, root.incoming[1], shape, level + 1
            )

        operand_left = buffer_left if buffer_left else buffers[level]
        operand_right = buffer_right if buffer_right else buffers[level + 1]

        length = ir.Constant(ir.IntType(32), math.prod(shape) - 1)

        body_builder.call(
            body_builder.module.globals[f"{root.op}"],
            [target, operand_left, operand_right, length],
        )

    compile_tree_into(*args)


def jit_graph(module: ir.Module, root: graph.Node, shape: Iterable[int]) -> LLVMBuffer:
    result: LLVMBuffer = LLVMBuffer(np.empty(shape), root.dtype)
    buffers = root.buffers()

    # Scary, scary.
    if "main" in module.globals:
        del module.globals["main"]
        module.scope._useset.remove("main")

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
    flatten_graph(fn.args[1:], target, body_builder, root, shape)

    body_builder.branch(exit_builder.block)

    llvm_ir = str(module)

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


def populate_global_module(name: str) -> ir.Module:
    module: ir.Module = ir.Module(name)

    # Parameter specifications for function taking: pointers to target, A, B, and integer length.
    parameter_types = [ir.PointerType(ir.FloatType())] * 3 + [ir.IntType(32)]

    for op, llvm_func in ops.LOOKUP.items():
        build_rolling_function(module, op, parameter_types, llvm_func)

    return module
