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

GPU run trace for a Purkinje cell population:

```bash
JAX_PLATFORMS=gpu python examples/profiling/profile_simulation.py \
  --case neuron_compare_cell \
  --cell pc_ma2024 \
  --population-size 32 \
  --duration-ms 10 \
  --dt-ms 0.01 \
  --warmup 1 \
  --repeat 3 \
  --trace-dir /tmp/braincell-trace-pc-pop32
```

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

## Notes

- The tool profiles the BrainCell/JAX path only; it does not run NEURON
  comparison baselines.
- The first warmup run includes JIT compilation and should be interpreted
  separately from steady-state runs.
- The network adapter exposes `tiny`, `small`, and `notebook` scales, plus
  `--populations` for profiling a subnetwork first.  Start with
  `--scale tiny --populations GrC,GoC` before using the full notebook-sized
  workload.
