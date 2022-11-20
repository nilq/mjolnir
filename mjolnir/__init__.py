from . import compiler, graph, tensor
from .tensor import Tensor

__all__ = ["compiler", "tensor", "graph"]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("This is not how to matrix-multiply.")

    output: Tensor = Tensor.empty((a.shape[0], b.shape[-1]))

    return output
