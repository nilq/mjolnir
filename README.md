# mjolnir

A (for now) quite primitive, but JIT'ed autograd.

## Plan

- [ ] ~~Values~~ *Tensors*.
- [ ] Torch-like nn-modules.
- [ ] Convolutions.
- [ ] Criterions, optimizers.

## For now

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