# NetStim Connection Runtime

## Goal

Replace the implicit colocated `cell.place(..., NetStim(...))` route with an
explicit event-source/connection model that works for one cell and
one-dimensional cell populations. Network lowering remains free to use its
separate network-level topology-table representation.

## Public contract

- `NetStim(size=1, start=..., number=..., interval=..., noise=0, seed=None,
  name=None)` is a standalone event-source group, not a point mechanism.
- NetStim parameters accept scalars or exact `(size,)` values. Physical fields
  require units. `noise` follows NEURON's fixed-plus-negative-exponential
  interval definition. For nonzero noise, `start` is the most likely first
  event time and the realized first event adds a negative-exponential wait
  with mean `noise * interval`. `seed=None` uses standalone root seed zero; an
  explicit seed fully controls the source group.
- `NetStim` has no output weight. `Connection` owns synapse-compatible weight and
  delay.
- `cell.cv_midpoints` returns an ordered `LocsetMask`. Two-dimensional integer
  indexing produces a `LocsetBatch` whose leading dimension aligns with a Cell
  population selection.
- Existing `place()` remains the only placement command. A Locset expression or
  one-dimensional mask broadcasts; a LocsetBatch is zipped with selected cell
  indices.
- `cell.synapses` exposes logical synapse instances. Selection by declaration,
  name, or stable ID is supported; selected scalar or row-aligned parameter
  columns can be updated before initialization.
- `connect(name, source=..., synapse=..., weight=..., delay=...)` appends
  Cell-owned routing rows. It supports scalar fan-out/fan-in and equal-size
  zip. Arbitrary mappings use duplicate-preserving source and synapse views.
- `cell.connections` is the unified row view. String indexing selects a
  connect name; numeric, slice, and boolean indexing always select rows.

## Runtime contract

- A NetStim source group has a reproducible schedule controlled by its seed.
  Its noise distribution follows NEURON, but the BrainState random backend does
  not promise the same event sequence as NEURON for the same integer seed.
- Events are generated at continuous times and assigned to the nearest
  fixed-step boundary, matching NEURON fixed-step delivery. Exact half-step
  ties use the later boundary; non-grid events are not dropped and multiple
  events in one step accumulate.
- Connections lower into columnar source indices, synapse runtime indices, weights,
  and delay steps. Multiple connections may share a synapse instance and add their
  due payloads.
- Connection queues persist across continued `run()` calls and reset with the Cell.
- Structural edits remain declaration-time only.

## Compatibility

`cell.place(..., NetStim(...))` is removed. Existing deterministic NetStim
examples and tests migrate to standalone NetStim plus named Connection rows. Existing
Network direct-connection APIs keep their public behavior while their delivery code
continues to share event scatter/delay utilities.

## Verification

- Constructor, unit, shape, noise, seed, and reset tests for NetStim.
- Locset indexing, batch placement, stable ordering, and duplicate tests.
- Heterogeneous synapse parameter and logical-instance view tests.
- Connection pairing, unit validation, delay, shared synapse, continued-run, and
  explicit removal tests.
- A CV-pair population test with `P = N * (N - 1) / 2`, `2P` synapses, and
  `2P` heterogeneous NetStim sources.
- A runnable notebook under `examples/multi_compartment/` showing construction,
  inspection, voltage output, CPU scaling, and optional GPU scaling.
- A benchmark must report declaration/init/compile/warm-run time and persistent
  source/connection/synapse storage, then identify the dominant performance costs.

## Implemented result

- The CV-pair notebook is
  `examples/multi_compartment/netstim_heterogeneous_connections.ipynb`; it executes
  a five-CV HH population with 10 CV-pair cells, 20 independently parameterized
  ExpSyn instances, and 20 heterogeneous NetStim connection rows.
- `examples/multi_compartment/netstim_connection_benchmark.py` provides isolated
  CPU or CUDA processes. On the development A100, heterogeneous populations of
  64 and 256 cells used 8,320 and 33,280 persistent bytes respectively for the
  measured source/connection/synapse columns. Warm three-step runs were about 400
  and 414 ms; at these deliberately short runs, Python/JAX launch overhead is
  still dominant, so these numbers are validation data rather than a throughput
  claim.
- CPU notebook runs show identical measured storage for broadcast and packed
  heterogeneous layouts when both own exactly two synapses per cell. Storage
  grows linearly (520 bytes at size 4 and 2,080 bytes at size 16); the packed
  representation avoids padding when per-cell counts differ, but it cannot
  reduce storage when the logical work is already uniform.
