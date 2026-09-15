# Geometry initialization

Compare fresh-cell startup before and after the differentiable geometry refactor.
The driver loads an explicitly selected BrainCell checkout, so both revisions
can use the same measurement script without reinstalling the package.
The design and staged acceptance criteria are in
[Nonlinear Pattern Separation validation design](../../../docs/design/optim/proposals/geometry-training.md).

The saved comparisons measure the earlier array-preparation prototype, before
the current runtime geometry lifecycle and trainable integration. Descriptions
of the measured candidate below refer to those historical revisions. The drivers
can select a checkout explicitly; historical results do not measure the current
branch. In particular, current runtime arrays may carry a population axis even
when their values are shared across members.

The [results index](results/README.md) separates reviewed findings from local raw
outputs. Nonlinear Pattern Separation 是 accuracy/reproduction validation，Jaxley training agreement 不属于本目录；本目录只记录 geometry initialization performance。
[geometry validation](../../../validation/optim/nonlinear_pattern_separation/README.md).
These measurements do not establish the cost of dynamic geometry training.

## Workload and timing

`benchmark.py` builds a soma with two tapering daughter branches, passive leak,
one population member and a specified total CV count. CVs are divided as evenly
as possible across the three branches with `CVPerBranchList`. Each trial:

1. Builds fresh morphology, Cell and mechanism declarations.
2. Accesses `cell.cvs`, timing initial discretization.
3. Calls `init_state`, including its morphology clone and rediscretization.
4. Compiles and executes exactly one `staggered` step of 0.025 ms in CPU or GPU float64.

These phases are separate timings. `init_state` does not reuse the discretization
measured in step 2. First-step time includes lazy solver setup, tracing,
compilation and execution, and is **not** steady-state simulation throughput.
Every trial uses a fresh Cell and first JIT execution; a warmup is a complete
trial, not an extra rollout attached to each trial. Device work is synchronized.
The uniform passive voltage is checked against its analytic implicit-Euler
update using the already-produced sample, with no extra model execution.
Persistent JAX compilation caching is disabled across source trees.
GPU selection uses `--gpu` before importing JAX; allocation preallocation is disabled.

The candidate stores the CV area, capacitance density and half-CV resistances
as unit-carrying JAX arrays. Point/ion/clamp numeric vectors are also transferred
as whole arrays. Node-tree coefficients and the reduced axial operator use JAX
array arithmetic; integer routing remains NumPy. This measures numerical array
preparation, not a public nonlinear-pattern-separation API. Differentiating arbitrary
morphology clipping, binding length/radius/Ra to derived arrays, and propagating
trainable updates through all caches/ion geometry remain later integration work.

## Running

Use an environment with the selected checkout's dependencies (JAX, BrainState,
BrainUnit and NumPy). Start a fresh interpreter and specify every count explicitly.
Agree on the execution plan under [benchmark rules](../../AGENTS.md) before running.

```bash
python benchmarks/performance/geometry_initialization/benchmark.py \
  --source-root /path/to/baseline-checkout --label baseline_cpu --platform cpu \
  --sizes 12 128 512 1024 --warmup 1 --repeat 5 \
  --output benchmarks/performance/geometry_initialization/artifacts/baseline.json
```

Run the same driver with the candidate checkout, another label and a fresh
output path. For each checkout also run `--platform gpu --gpu DEVICE_INDEX`.
Execute serially on the same otherwise-idle devices. Counts in this example
describe the protocol; a new session must still obtain authorization.

For that protocol there are four processes (two revisions × two backends), four
scale results per process, and six trials per scale. Each process constructs 24 fresh
models, performs 24 `init_state` calls and 24 first-JIT single-step executions.
There are 96 model executions overall, 80 measured and 16 warmup; the first
execution is included in warmup. There are no additional warmups, reset calls,
training updates, validation rollouts, profiling runs or automatic retries.
Each of the 80 measured trials produces four phase durations (320 values).

Limits: up to 10 minutes per process and 40 minutes total, with no
automatic retries. Use an external process timeout to enforce the agreed limit.
This is a coarse one-process-per-revision-and-backend comparison; it cannot establish
cross-process variability. Larger-scale first-use kernels can reuse in-process
primitives from earlier scales; record and preserve the scale order.

## Outputs and checks

The driver refuses to overwrite an output. JSON preserves every completed sample,
including warmups, plus source hashes, revision/dirty status, actual import path,
software/device information, configured execution counts and phase summaries.
Report both the first warmup sample for each scale and warmed trial medians:
new device kernels may compile during initialization, before the explicit JIT
phase, and excluding those warmups alone would understate first-use cost.
Ordinary exceptions save a failure message and completed samples. A hard process
kill leaves the last flushed sample; inspect `status` before interpreting a run.
Generated JSON stays in ignored `artifacts/`. Reviewed performance findings belong
in Git-tracked `results/`. The completed CPU/GPU array-preparation comparison,
including the repaired CPU baseline run, is recorded in [results](results/arrays-cpu-gpu.md).

`reporting.py` regenerates the timing table and figure from completed JSON inputs,
without importing BrainCell or rerunning models. It defaults to an input-adjacent
`report/` directory; `--output` explicitly selects another directory.

`benchmark_test.py` checks execution accounting, warmup exclusion and input
validation with fake trial results. It never starts a performance measurement.

```bash
pytest -q benchmarks/performance/geometry_initialization/benchmark_test.py
```

## Population and steady simulation

`population.py` separates population startup from repeated execution of the same
compiled rollout. It compares the same passive fork across source revisions,
with a nonuniform initial voltage over both CVs and population members. Geometry
is shared across the population; this is not a network of separately constructed
Cells, a training batch, or independently trainable per-member morphology.

The reduced matrix is CPU/GPU × baseline/candidate × CV=128 ×
pop_size=1/10/100/1000. Each of the four serial processes handles four cases. Each
case constructs and initializes **one** Cell, then reuses one compiled 100-step
rollout (dt=0.025 ms, duration=2.5 ms): first compilation/execution once, one
additional warmup and five timed executions. Reset happens before each of those
seven runs, with its own timing outside the rollout timer. The new initialization
metric is a single first-use sample per case, not the five repeated-init samples
in `benchmark.py`; do not mix those statistics.

The reduced matrix comprises 16 initializations, 16 first compiled executions,
16 extra warmups, 80 timed rollouts, 112 resets and 11,200 population timesteps.
There are no additional simulation validations, backward passes or automatic
retries. The limit is 15 minutes per process and 60 minutes total.
Obtain authorization for these execution counts before starting a new measurement.

Compilation caches are cleared before constructing each case, not between its
seven runs. Persistent compilation caching is disabled. All live device arrays
are synchronized at timing boundaries; the synchronization overhead is included.
The already-executed voltages are checked for shape, finite float64 values,
passive bounds, the analytic decay of the population voltage offset, and identical
results after reset. Final voltages are saved once per case for baseline/candidate
comparison without extra simulations. Plain Python loops enumerate independent
cases and timed calls; all repeated simulation steps use a compiled BrainState
`for_loop`.

Each case records declaration/initialization time, first compiled execution,
every reset and rollout time, population-step latency, per-cell-step cost and
aggregate cell-step throughput. It also records voltage and geometry array
shapes and logical State byte counts. Logical byte counts include broadcast
views and are not physical peak RAM/VRAM measurements.

```bash
python benchmarks/performance/geometry_initialization/population.py \
  --source-root /path/to/baseline-checkout --label baseline_cpu_population \
  --platform cpu --sizes 128 --pop-sizes 1 10 100 1000 \
  --steps 100 --warmup 1 --repeat 5 \
  --output benchmarks/performance/geometry_initialization/artifacts/baseline_cpu_population.json
```

Repeat with the candidate source and, for GPU, `--platform gpu --gpu DEVICE_INDEX`,
using distinct output names. A hard timeout preserves the last flushed case/run;
failed or unfinished cases are not interpreted as successful timing samples.

`population_report.py` reads four completed scans, checks matching protocols and
saved final voltages across revisions/backends, and generates `population.png`,
`timings.md` and `checks.json`. It never imports BrainCell or executes a model.
Inputs must cover baseline/arrays on CPU/GPU at one common CV size. Baseline
labels start with `baseline`; the other two inputs represent the array candidate.
Raw JSON and its referenced voltage NPZ files must both remain available.
By default the report goes under the first input's `population_report/` directory.

```bash
python benchmarks/performance/geometry_initialization/population_report.py \
  artifacts/baseline_cpu_population.json artifacts/arrays_cpu_population.json \
  artifacts/baseline_gpu_population.json artifacts/arrays_gpu_population.json \
  --output artifacts/population_report
```

Adjust input paths to where the driver wrote them. The reviewed reduced scan is
recorded in [128-CV population results](results/population-128.md).

`population_test.py` checks counts and failures with fake cases, and runs an
untimed 3-CV/2-member/3-step correctness example twice around reset. It never
invokes the performance measurement routine.

```bash
pytest -q benchmarks/performance/geometry_initialization/population_test.py
```
