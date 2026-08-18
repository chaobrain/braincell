# Channel Layout Cost Benchmark

## Goal

Quantify the memory and compute cost of BrainCell's dense runtime layout for
region-limited density channels, and estimate the benefit of a compact packed
layout without changing runtime behavior.

The benchmark uses the current runtime as the source of truth and keeps a
synthetic packed comparison separate. The packed comparison must use the
existing `Na_HH1952`, `K_HH1952`, and `IL` implementations and BrainCell's
existing integration step; it must not duplicate channel equations.

## Fixture

Construct one morphology with four branches:

- soma: 4 CVs
- dendrite A: 32 CVs
- dendrite B: 32 CVs
- axon: 32 CVs

`CVPerBranchList((4, 32, 32, 32))` gives exactly 100 CVs. The runtime point
count is recorded rather than assumed. Population sizes are 1, 10, 100, and
1000.

The representative mixed layout paints:

- `Na_HH1952` on the soma: 4 active CVs
- `K_HH1952` on the proximal half of dendrite A: 16 active CVs
- `IL` on dendrite B: 32 active CVs

Additional profiles cover 4, 16, 32, and 100 CVs so cost can be plotted
against active fraction. A global control paints every channel on all 100 CVs.

## Measurements

Report these memory categories independently:

1. declaration buffers retained by `CellRuntimeState.state_buffers`
2. runtime channel parameter arrays
3. mutable channel gate values
4. layout masks or packed indices
5. XLA argument, output, and temporary bytes from `memory_analysis()`

The compact projection preserves each observed dtype and replaces the trailing
`n_point` dimension by `n_active`. It is an estimate of persistent
channel-owned storage, not a claim about allocator-resident GPU memory.

For compute measurements:

- obtain compiler estimates with `lower().cost_analysis()`
- inspect optimized HLO with `compile().as_text()` and count fusion
  computations/instructions
- force synchronization with `jax.block_until_ready()`
- discard two initial calls and report the median of at least seven timed calls
- execute each timing configuration in a fresh subprocess so JIT caches and
  allocator state do not leak across configurations

The notebook records Python, JAX, backend, device, precision, CV count, point
count, and active counts alongside every result. CPU measurements describe
relative behavior only; GPU measurements are optional and skipped cleanly
when no usable GPU is available.

## Packed Reference

The packed microbenchmark is not a runtime implementation. It gathers active
voltage and ion information, updates compact channel states, evaluates current,
and scatter-adds current into full point space. The dense reference uses full
point-shaped states with masked conductance/current, matching the current
runtime behavior.

Before timing, compare dense and packed active-point gate values and currents
for multiple steps. Packed inactive current must remain exactly zero.

## Interpretation

The notebook must distinguish the following conclusions:

- masking current establishes numerical inactivity but does not skip gate work
- setting inactive derivatives to zero would freeze state but would not, by
  itself, remove elementwise computation
- moving density channels from point space to CV space removes topology-only
  points but does not solve region sparsity
- a hybrid dense/packed runtime can keep dense storage for broad coverage and
  use compact storage for local mechanisms
- channels sharing an active index set should share gather/scatter work

No production API or runtime behavior changes are part of this benchmark.
