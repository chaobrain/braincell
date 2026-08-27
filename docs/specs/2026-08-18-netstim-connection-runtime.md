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
- A runnable notebook under `examples/multi_compartment/` showing source and
  target registration, synapse placement, connection inspection, split runs,
  sparse events, and recorded voltage/conductance output.
- A focused NEURON comparison must replay the realized BrainCell event times so
  event routing and numerical integration are tested independently of RNG choice.

## Implemented result

- `examples/multi_compartment/network.ipynb` is the maintained end-to-end
  tutorial. It registers Cell, NetStim, and EventSequence populations; connects
  multiple sources to two independent target populations; and verifies split
  and continuous runs.
- `examples/neuron_compare/synapse/netstim_heterogeneous_compare.ipynb` preserves
  the focused five-CV / 10-cell / 20-ExpSyn heterogeneous NetStim comparison. It
  replays 40 realized events in NEURON and checks the aligned voltage traces.
- The old mixed tutorial/benchmark files were retired. Performance profiling is
  intentionally kept separate from the maintained user workflow and numerical
  comparison notebooks.
