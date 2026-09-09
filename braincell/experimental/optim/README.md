# Experimental gradient interfaces

```python
from braincell.experimental import optim
```

`gradients.py` provides additive rollout and trajectory gradient engines with
`method="bptt"` and `method="rtrl"`. `_forward_sensitivity.py` supplies private
functionalization and exact forward-sensitivity recurrence. Parameter selection,
sharing, transforms and materialization come from `braincell.trainable`.

API contracts and examples: [Experimental Workflows](../../../docs/design/optim/current/experimental-workflows.md).
Scientific checks: [Optimization validation](../../../validation/optim/README.md).
Performance: [Gradient scaling](../../../benchmarks/performance/optim_gradient_scaling/README.md).

```bash
pytest -q braincell/experimental/optim
```
