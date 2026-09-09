# Optimization accuracy

- [Gradient correctness](gradient_correctness/README.md) compares BPTT and exact
  RTRL on one-CV, multi-CV, autapse and bidirectional models.
- [Training agreement](training_comparison/README.md) compares losses, gradients
  and parameter trajectories for matched updates.
- `_workload.py` supplies shared HH model construction and rollout preparation
  for training agreement and performance scaling. It does not run timing suites.

```bash
pytest -q validation/optim
python -m validation.optim.training_comparison.rtrl_bptt --help
```

The gradient engines come from `braincell.experimental.optim`. Scaling results
and resource measurement belong to [benchmarks](../../benchmarks/performance/optim_gradient_scaling/README.md).
Model contracts and interpretation are maintained in [Optimization Design](../../docs/design/optim/TODO.md).
