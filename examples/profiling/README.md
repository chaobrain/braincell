# BrainCell Profiling Examples

This directory contains developer-oriented profiling helpers for BrainCell
examples.  The helpers live outside the public `braincell` API and are intended
for local performance diagnosis.

## Supported Workloads

- `neuron_compare_cell`: BrainCell-only runs for
  `examples/neuron_compare/cell/*`.
- `cerebellar_probability_network`: script form of
  `examples/multi_compartment/cerebellar_probability_network_demo.ipynb`.

## Basic Usage

Profile a single cell example:

```bash
python examples/profiling/profile_simulation.py \
  --case neuron_compare_cell \
  --cell pc_ma2024 \
  --duration-ms 10 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 3 \
  --out /tmp/profile_pc.json
```

Profile the cerebellar probability network at a small scale:

```bash
python examples/profiling/profile_simulation.py \
  --case cerebellar_probability_network \
  --scale tiny \
  --populations GrC,GoC \
  --duration-ms 0.1 \
  --dt-ms 0.1 \
  --warmup 1 \
  --repeat 1 \
  --out /tmp/profile_network.json
```

## Trace And Memory Profiles

JAX dispatch is asynchronous, so the harness blocks on simulation results before
stopping each run timer.  Use `--trace-dir` to collect a JAX profiler trace:

```bash
python examples/profiling/profile_simulation.py \
  --case neuron_compare_cell \
  --cell grc_ma2020 \
  --trace-dir /tmp/braincell-trace
```

For run-time attribution, open the generated Perfetto trace and search for
`braincell:` scopes. The hot run path is annotated with nested scopes such as
`braincell:cell_run:update_dynamics`, `braincell:staggered:dhs_voltage_step`,
`braincell:dhs:forward_elimination`, and
`braincell:membrane_current:channel_currents`.

For a command-line summary of GPU events attributed to BrainCell scopes, parse
the generated XPlane file:

```bash
python examples/profiling/parse_xplane_trace.py \
  --trace-dir /tmp/braincell-trace \
  --mode leaf
```

Use `--mode leaf` to charge each GPU event to the deepest matching scope, or
`--mode inclusive` to charge it to every matching scope in the XLA path. These
totals are GPU event-duration sums, not critical-path wall time; parallel
streams can make the sum larger than the outer `steady_run` wall time.

GPU run trace for a Purkinje cell population:

```bash
python examples/profiling/profile_simulation.py \
  --platform cuda \
  --case neuron_compare_cell \
  --cell pc_ma2024 \
  --population-size 32 \
  --duration-ms 10 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 3 \
  --trace-dir /tmp/braincell-trace-pc-pop32 \
  --trace-phase steady \
  --trace-device-only
```

For finer attribution while diagnosing kernel placement, disable GPU command
buffers in the profiling process:

```bash
XLA_FLAGS='--xla_gpu_enable_command_buffer=' python examples/profiling/profile_simulation.py \
  --platform cuda \
  --case neuron_compare_cell \
  --cell pc_ma2024 \
  --population-size 32 \
  --duration-ms 1 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 1 \
  --trace-dir /tmp/braincell-trace-pc-pop32-no-command-buffer \
  --trace-phase steady \
  --trace-device-only
```

This command-buffer setting is for profiler attribution only. It can change
kernel scheduling and should not be used as a steady-state performance baseline.

## DHS Level Utilization

Use the toy DHS level benchmark to diagnose whether narrow late tree levels
underfill the GPU. The `level-jit` mode launches one compiled function per
level so profiler scopes remain visible; use it for attribution, not as an
end-to-end DHS performance baseline.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
python examples/profiling/bench_dhs_levels.py \
  --platform gpu \
  --execution-mode level-jit \
  --n-cv 1024 \
  --popsize 32 \
  --warmup 2 \
  --repeat 5 \
  --profile-barrier \
  --trace-dir /tmp/dhs-level-pop32 \
  --out /tmp/dhs-level-pop32.json
```

Then print the per-level GPU table:

```bash
python examples/profiling/parse_xplane_trace.py \
  --trace-dir /tmp/dhs-level-pop32 \
  --scope-prefix braincell:dhs_toy \
  --mode leaf \
  --dhs-level-table \
  --out /tmp/dhs-level-pop32-summary.json
```

The key column is `us/work`, computed as GPU event time divided by
`width * popsize`. A rising `us/work` near the root means the late levels are
paying a fixed kernel/occupancy cost while doing less parallel work.

To inspect the real DHS forward elimination inside a cell run, enable the
profiling-only real-level scopes:

```bash
BRAINCELL_PROFILE_DHS_LEVELS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
python examples/profiling/profile_simulation.py \
  --platform cuda \
  --case neuron_compare_cell \
  --cell pc_ma2024 \
  --population-size 32 \
  --duration-ms 0.1 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 1 \
  --trace-dir /tmp/pc-real-dhs-levels \
  --trace-phase steady \
  --trace-device-only
```

Parse the real levels with:

```bash
python examples/profiling/parse_xplane_trace.py \
  --trace-dir /tmp/pc-real-dhs-levels \
  --scope-prefix braincell:dhs \
  --mode leaf \
  --dhs-level-table
```

For real DHS rows, `items` means eliminated edges in that level, and `work`
means `items * batch`. When available, `occ%`, `grid`, and `block` come from
XLA `kernel_details`.

GPU run trace for a small network subset:

```bash
JAX_PLATFORMS=gpu python examples/profiling/profile_simulation.py \
  --case cerebellar_probability_network \
  --scale tiny \
  --populations GrC,GoC,PC \
  --duration-ms 10 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 3 \
  --spike-recording population \
  --trace-dir /tmp/braincell-trace-network
```

Use `--device-memory-profile` to save device memory snapshots after warmup and
the final steady run:

```bash
python examples/profiling/profile_simulation.py \
  --case neuron_compare_cell \
  --cell grc_ma2020 \
  --device-memory-profile /tmp/braincell-memory
```

Host memory uses `tracemalloc` by default.  If `psutil` is installed, RSS is
reported as well.  No extra dependency is required for basic timing.

## Nsight Systems Capture Range

Use `--cuda-profiler-range` with the Nsight Systems CUDA profiler API capture
range when a full JAX trace is unnecessary or too large. The range brackets
only steady runs, after workload creation and warmup:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o /tmp/braincell-neuron-compare \
  python examples/profiling/profile_simulation.py \
    --case neuron_compare_cell \
    --platform cuda \
    --cell pc_ma2024 \
    --population-size 32 \
    --duration-ms 10 \
    --dt-ms 0.01 \
    --warmup 1 \
    --repeat 1 \
    --cuda-profiler-range
```

## Notes

- The tool profiles the BrainCell/JAX path only; it does not run NEURON
  comparison baselines.
- The first warmup run includes JIT compilation and should be interpreted
  separately from steady-state runs.
- The network adapter exposes `tiny`, `small`, and `notebook` scales, plus
  `--populations` for profiling a subnetwork first.  Start with
  `--scale tiny --populations GrC,GoC` before using the full notebook-sized
  workload.
