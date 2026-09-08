# Experimental Optimization Core

This package contains the model-independent optimization interfaces currently
being evaluated for a future `braincell.optim` package.

- `gradients.py` exposes fixed-shape rollout and trajectory gradient engines
  with `method="bptt"` and `method="rtrl"`.
- `_forward_sensitivity.py` owns the low-level functionalization, parameter
  coordinates, and exact forward-sensitivity recurrence. It is private to the
  experimental implementation.
- The adjacent tests define the candidate API contract. Scientific validation,
  scaling, and training comparisons live in separate experimental directories.

Parameter selection, sharing through `group_by`, transforms, and runtime
materialization already belong to `braincell.trainable`. The gradient engines
consume `target.trainables`; this package does not duplicate that API.

```bash
pytest -q examples/experimental/optim
```
