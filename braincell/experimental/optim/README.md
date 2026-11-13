# Experimental gradient interfaces

```python
from braincell.experimental import optim
```

`gradients.py` provides additive rollout and trajectory gradient engines with
`method="bptt"` and `method="rtrl"`. `_forward_sensitivity.py` supplies private
functionalization and exact forward-sensitivity recurrence. Parameter selection,
sharing, transforms and materialization come from `braincell.trainable`.

For a cable Cell, every scan builds the axial operator (the CV operator of the
generic solvers, the DHS coefficients of `staggered` and `dhs_voltage`) once,
from its initial states, and passes it to every step; RTRL passes its tangent
along. `engine.axial_operator_mode` reports the choice after `prepare`:

| Mode | When | The operator is built |
| --- | --- | --- |
| `"rollout"` | the step does not change the geometry or an optimizer root | once per scan |
| `"step"` | the step writes a state the operator is built from, or a root | at every step |
| `"none"` | no operator to hoist: a lumped Cell or a reduction model has none; a `Network` builds its own | by the target, if it has one (a `Network`: at every step) |

Scientific checks: [Optimization validation](../../../validation/optim/README.md).
Performance: [Gradient scaling](../../../benchmarks/performance/optim_gradient_scaling/README.md).

```bash
pytest -q braincell/experimental/optim
```
