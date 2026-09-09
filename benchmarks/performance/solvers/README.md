# DHS Solver Performance

Measure isolated DHS tree-solver execution across CV count and population size.
`bench_dhs_levels.py` constructs a synthetic tree and compares a whole compiled solve
with level-wise compiled execution. `bench_dhs_levels_test.py` checks solver behavior.
Requires BrainCell, BrainState, BrainUnit and a JAX installation for the selected backend.

```bash
python -m benchmarks.performance.solvers.bench_dhs_levels \
  --platform <cpu-or-gpu> --n-cv <count> --popsize <count> \
  --warmup <count> --repeat <count> \
  --out benchmarks/performance/solvers/artifacts/levels.json
```

`--dtype` selects float32 or float64. `--trace-dir` captures a separate trace;
`--profile-barrier` changes synchronization for diagnosis. Inspect `--help` for the
whole-solve and level execution options before selecting a protocol. Compare equivalent
numerical problems; level timing and whole-solve timing have different launch costs.
Generated JSON and traces belong in ignored `artifacts/`; reviewed historical summaries
belong in `results/` when available. Trace analysis is provided by the shared
[profiling tools](../../profiling/README.md).

Before measuring, follow the [execution rules](../../AGENTS.md): confirm the full
configuration matrix, independent rounds, first executions, warmups, timed repeats,
extra validation and time budget. Commands below are templates, not authorization.
Replace angle-bracket placeholders before running from the repository root.
