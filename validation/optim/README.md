# Optimization accuracy

Model builders initialize the final grid before selecting and registering
trainable parameters. The differentiated initializer materializes current
parameters, resets dynamic states, and then starts the rollout. Full Cell
`reset()` requires fresh registration and a new gradient program.

- [Gradient correctness](gradient_correctness/README.md) compares BPTT and exact
  RTRL on one-CV, multi-CV, autapse and bidirectional models.
- [Training agreement](training_comparison/README.md) compares losses, gradients
  and parameter trajectories for matched updates.
- [Nonlinear Pattern Separation](nonlinear_pattern_separation/README.md) checks fixed-grid cable
  gradients, runtime-derived values and geometry grouping, and records
  Nonlinear Pattern Separation 的 Jaxley reproduction comparisons in its [results](nonlinear_pattern_separation/results/README.md).
- `_workload.py` supplies shared HH model construction and rollout preparation
  for training agreement and performance scaling. It does not run timing suites.

```bash
pytest -q validation/optim
python -m validation.optim.training_comparison.rtrl_bptt --help
```

The gradient engines come from `braincell.experimental.optim`. Scaling results
and resource measurement belong to [benchmarks](../../benchmarks/performance/optim_gradient_scaling/README.md).
Model contracts and interpretation are maintained in [Optimization Design](../../docs/design/optim/TODO.md).
