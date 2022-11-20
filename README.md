# mjolnir

A (for now) quite primitive, but JIT'ed autograd.

## Roadmap

- [x] Simple tensors.
- [ ] Gradients.
- [ ] Criterions/optimizers.
- [ ] Torch-like modules.

## For now

### Simple tensor operations

```python
from mjolnir.tensor import Tensor

a = Tensor([[1, 2], [3, 4]])
b = Tensor.ones((2, 2))

summed = (a + b).tensor
```

### Computation graph

```python
from mjolnir.tensor import Tensor

a = Tensor.ones([5, 5])
b = Tensor.eye([5, 5])
c = Tensor([[1, 2, 3, 4, 5]] * 5)

graph = ((a + b + c) * (c - b - a))
graph.print()
```

```
(*):
  (+):
    (+):
      Tensor<(5, 5)>
      Tensor<(5, 5)>
    Tensor<(5, 5)>
  (-):
    (-):
      Tensor<(5, 5)>
      Tensor<(5, 5)>
    Tensor<(5, 5)>
```
