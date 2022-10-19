from llvmlite import ir


def ir_float(n: float) -> ir.Constant:
    return ir.Constant(ir.FloatType(), n)


OPS_LOOKUP = {
    "": lambda ir_builder, *args: args,
    "+": lambda ir_builder, *args: ir_builder.fadd(*args),
    "-": lambda ir_builder, *args: ir_builder.fsub(*args),
    "*": lambda ir_builder, *args: ir_builder.fmul(*args),
    "/": lambda ir_builder, *args: ir_builder.fdiv(*args),
    "**": lambda ir_builder, *args: ir_builder.call(
        ir_builder._block.module.declare_intrinsic("llvm.pow", [ir.FloatType()]), args
    ),
    "relu": lambda ir_builder, x: ir_builder.select(
        ir_builder.fcmp_ordered("<=", ir_float(0.0), x), x, ir_float(0)
    ),
}
