# Gradient Scaling Experiments

Measure BPTT and exact RTRL across CV count, recurrent state, parameter directions,
protocol batch and independent seed lanes. Requires BrainCell and its JAX/BrainState
stack; GPU suites require a GPU-enabled JAX installation.

| Script | Purpose |
| --- | --- |
| `benchmark.py` | Isolated HH scaling suites |
| `controlled_complexity.py` | Separate state, parameter and time complexity |
| `report.py` | Aggregate stored CSV/JSON without running simulations |
| `plot_hh_crossover.py` | Plot the three HH parameter slices, runtime ratios, memory and compilation from saved pairs |
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

## HH crossover coarse scan

The `hh_crossover` suite fixes full HH paint and measures `C = 1, 21, 41, 81`.
For each CV count it trains Leak, Leak+K, then Leak+K+Na conductance scales per CV:
`Nx = 4C`, `Ntheta = C, 2C, 3C`. Parameters are shared over the protocol batch.
The suite fixes batch=16, seeds=16, duration=40 ms, dt=0.025 ms and float64.
It uses 12 configurations, two isolated method workers each, and one independent
round. Workers are serial, with method order alternating between configurations.

```bash
python -m benchmarks.performance.optim_gradient_scaling.benchmark run \
  --suite hh_crossover --gpu <physical-index> \
  --repeats <timed-count> --warmups <extra-warmup-count> --no-gpu-monitor \
  --worker-timeout-seconds <worker-limit> --budget-seconds <total-limit> \
  --output-dir <new-artifact-directory>
```

`--warmups` counts **additional** executions after the separately recorded first
execution. Each worker also generates one target forward rollout before compiling
the gradient kernel. Thus W extra warmups and R timed calls produce `1+W+R`
gradient calls plus one target rollout per worker. Output comparison reuses the
first gradient result. There are no separate validation rollouts or automatic retries.
The default DHS differentiation mode is generic JAX AD. The experimental
explicit DHS JVP can be selected with `--dhs-jvp-mode explicit` for correctness
or diagnostic comparisons; it is not the default speed path.
The timeout includes worker startup, preparation, compilation and all executions;
the total budget additionally includes orchestration and aggregation.
Use `--cv-values 1 21 41` to select those existing suite points. A zero worker
timeout or total budget disables that limit. Suite workers disable JAX persistent
compilation caching so compilation measurements do not reuse disk cache entries.

Trial JSON preserves first, warmup and timed durations, median/IQR/min/p90,
completed-call counters, compilation time, dimensions and XLA memory analysis.
`materialization_mode` records the mapping schedule selected by the gradient engine.
Compilation time covers the outer gradient JIT tracing/lowering/compilation;
model preparation and target compilation occur beforehand. With `--no-gpu-monitor`,
no device activity or process-memory polling runs. XLA temporary bytes and logical
RTRL carry are not measured process peak memory.

Individual `worker` commands can add `--compile-diagnostics` to time JAX tracing,
lowering and XLA compilation separately. This requires a JAX JIT object supporting
`.trace()`. The worker reuses that same traced/lowered computation and saves StableHLO
plus nested Jaxpr operation counts after timing. It does not add model executions;
`--warmups` and `--repeats` still determine execution counts. Normal suite workers
retain the existing combined compilation path unless diagnostics are explicitly enabled.
Core measurements are saved before optional graph export; an export failure is recorded
under `diagnostics_status` and does not discard completed timing or output data.

`paired_results.csv` and `summary.md` are regenerated beside raw trials after each
worker; they include all 12 pairs, including pending or failed ones. A valid pair
requires finite outputs, total and local losses agreeing within rtol=1e-7,
atol=1e-8, and gradient relative-L2 error <=1e-6 or max absolute error <=1e-8.
A runtime ratio within [0.9, 1.1] is only a proximity candidate. IQR describes
within-process spread; this single round does not establish cross-process stability.

`manifest.json`, `source.patch`, `source_files.json` and `actual_counts.json` record
the protocol, command order, environment, source fingerprint and completed calls.
After interruption, completed-call counters are lower bounds: an in-flight execution
or a target completed during unfinished preparation may not have been recorded.
The crossover suite requires a fresh directory and rejects `--resume`. A dry run
also writes a manifest, so use a separate directory for it.

Plot a completed HH scan using NumPy and Matplotlib, with no model execution:

```bash
python -m benchmarks.performance.optim_gradient_scaling.plot_hh_crossover \
  <completed-artifact-directory> --context '<measured environment and protocol>'
```

The input must contain validated `paired_results.csv` and matching `trials/*.json`
with saved timing samples. The three slices are `Ntheta=Nx/4`, `Nx/2`, and `3Nx/4`.
Figures default to the input directory's `figures/`; `--output-dir` overrides it.
PNG, SVG and PDF versions and source hashes are saved. Error bars use actual Q25–Q75;
connecting lines do not establish an unmeasured crossover boundary. Incomplete or
invalid pairs and duplicate slice points are rejected.

Orchestration tests that do not execute a neuron model:

```bash
python -m pytest benchmarks/performance/optim_gradient_scaling/benchmark_test.py \
  -k CrossoverProtocolTest
```
