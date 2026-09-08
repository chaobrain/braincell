# Gradient Scaling Experiments

This directory measures BPTT and exact RTRL across CV count, recurrent state,
parameter directions, protocol batch, and independent seed lanes.

- `benchmark.py` runs BrainCell HH scaling suites in isolated workers.
- `controlled_complexity.py` separates state, parameter, and time complexity.
- `report.py` aggregates existing CSV/JSON results without running JAX.
- `analysis.ipynb` produces the detailed analysis and figures.
- `profile_case.py` adapts the fixed-shape gradient workload to the shared
  `examples/profiling/profile_simulation.py` harness.
- Generated traces, tables, and figures live under ignored `artifacts/`.

```bash
pytest -q examples/experimental/optim_gradient_scaling
```
