"""Internal type menu card."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from llvmlite.ir import types as ir_types
from llvmlite.ir.types import Type, _as_half, _BaseFloatType, _format_double


# Create missing brain-float LLVMLite type.
class BrainFloatType(_BaseFloatType):
    """Type for brain floating-point format."""

    null = "0.0"
    intrinsic_name = "bfloat"

    def __str__(self) -> str:
        return "bfloat"

    def format_constant(self, value):
        return _format_double(_as_half(value))


# I've to do everything yourself.
BrainFloatType._create_instance()


class DataType(ABC):
    """Generic data-type class."""

    @property
    @abstractmethod
    def llvm(self) -> Type:
        """Get corresponding LLVM type.

        Returns:
            Type: LLVMLite IR type.
        """

    @property
    @abstractmethod
    def numpy(self) -> np.dtype:
        """Get corresponding NumPy type.

        Returns:
            np.dtype: NumPy data-type.
        """


class BrainFloat16(DataType):
    llvm = BrainFloatType()
    numpy = np.float32


class Float32(DataType):
    llvm = ir_types.FloatType()
    numpy = np.float32


class Float64(DataType):
    llvm = ir_types.DoubleType()
    numpy = np.float64


Float = BrainFloat16 | Float32 | Float64


class Bool(DataType):
    llvm = ir_types.IntType(1)
    numpy = np.bool_


class Int8(DataType):
    llvm = ir_types.IntType(8)
    numpy = np.int8


class Int16(DataType):
    llvm = ir_types.IntType(16)
    numpy = np.int16


class Int32(DataType):
    llvm = ir_types.IntType(32)
    numpy = np.int32


class Int64(DataType):
    llvm = ir_types.IntType(64)
    numpy = np.int64


Int = Int8 | Int16 | Int32 | Int64
