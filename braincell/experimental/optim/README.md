# Experimental gradient interfaces

```python
from braincell.experimental import optim
```

`gradients.py` provides additive rollout and trajectory gradient engines with
`method="bptt"` and `method="rtrl"`. `_forward_sensitivity.py` supplies private
functionalization and exact forward-sensitivity recurrence. Parameter selection,
sharing, transforms and materialization come from `braincell.trainable`.

For a cable Cell, and for each cable Cell of a `Network` target whose step calls
`update()`, every scan builds the axial operator (the CV operator of the
generic solvers, the DHS coefficients of `staggered` and `dhs_voltage`) once,
from its initial states, and passes it to every step; RTRL passes its tangent
along. `engine.axial_operator_mode` reports the choice after `prepare`:

| Mode | When | The operator is built |
| --- | --- | --- |
| `"rollout"` | the step does not change the geometry or an optimizer root | once per scan (in a `Network`, at every step for a Cell whose geometry the step writes) |
| `"step"` | the step writes a state every operator is built from, or a root | at every step |
| `"none"` | no operator to hoist: a lumped Cell or a reduction model has none | — |

The step's own writes decide: the refresh of the trainable parameters that
`Network.update()` makes at every step does not count, and only the states
whose values reach the operator do.

Scientific checks: [Optimization validation](../../../validation/optim/README.md).
Performance: [Gradient scaling](../../../benchmarks/performance/optim_gradient_scaling/README.md).

```bash
pytest -q braincell/experimental/optim
```
