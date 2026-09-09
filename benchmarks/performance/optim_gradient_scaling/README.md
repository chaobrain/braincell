# Gradient Scaling Experiments

Measure BPTT and exact RTRL across CV count, recurrent state, parameter directions,
protocol batch and independent seed lanes. Requires BrainCell and its JAX/BrainState
stack; GPU suites require a GPU-enabled JAX installation.

| Script | Purpose |
| --- | --- |
| `benchmark.py` | Isolated HH scaling suites |
| `controlled_complexity.py` | Separate state, parameter and time complexity |
| `report.py` | Aggregate stored CSV/JSON without running simulations |
| `analysis.ipynb` | Detailed analysis and figures from stored results |
| `profile_case.py` | Adapt gradient workloads to the shared profiling harness |

```bash
python -m benchmarks.performance.optim_gradient_scaling.benchmark run \
  --suite <suite> --gpu <physical-index> --repeats <count> \
  --output-dir benchmarks/performance/optim_gradient_scaling/artifacts/run
python -m benchmarks.performance.optim_gradient_scaling.controlled_complexity run \
  --suite <suite> --gpu <physical-index> --repeats <count> --replicates <count> \
  --output-dir benchmarks/performance/optim_gradient_scaling/artifacts/controlled
python -m benchmarks.performance.optim_gradient_scaling.report --help
```

Workers use the current Python interpreter unless `--python` selects another.
`--dry-run` prepares the execution commands without starting workers; it writes a local
manifest. `--resume` reuses successful stored trials, so verify their protocol first.
Report output defaults to the input artifacts directory; explicit output paths are supported.
See [RESULTS](RESULTS.md) for reviewed historical evidence. The report currently recognizes
the named historical suite directories; arbitrary output directories need that documented
layout under the selected artifact root. Test modules include numerical model execution;
do not treat the entire directory as a report-only test command.

Before measuring, follow the [execution rules](../../AGENTS.md): confirm the full
configuration matrix, independent rounds, first executions, warmups, timed repeats,
extra validation and time budget. Commands below are templates, not authorization.
Replace angle-bracket placeholders before running from the repository root.
