# mjolnir

A (for now) quite primitive, but JIT'ed autograd.

## Plan

- [ ] ~~Values~~ Proper *Tensors*.
- [ ] Torch-like nn-modules.
- [ ] Convolutions.
- [ ] Criterions, optimizers.

## For now

### Values with gradients

```python
import math
from mjolnir import Value

a = Value(math.pi)
b = Value(2)

c = a**2 / (b * 10)

c.backward()

print(c)
```

```
Value(0.49348026514053345, gradient=1.0)
```

### Tensors with no Jacobians

Needs proper views, graphs with gradient computations and more ... 

```python
from mjolnir import Tensor

a = Tensor([[1, 2], [3, 4]])
b = Tensor([[1, 1], [1, 1]])

print(a + b)
```

```
Tensor(2, 2)
```