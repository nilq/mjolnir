"""A module containing basic operator mappings."""
LOOKUP = {
    "+": lambda ir_builder, *args: ir_builder.fadd(*args),
    "-": lambda ir_builder, *args: ir_builder.fsub(*args),
    "*": lambda ir_builder, *args: ir_builder.fmul(*args),
    "/": lambda ir_builder, *args: ir_builder.fdiv(*args),
}

