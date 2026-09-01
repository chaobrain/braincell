# Cell Population View

## Summary

`CellView` is a lightweight population selection returned by `cell[index]`.
The root `Cell` remains the only owner of morphology, discretization caches,
BrainState modules, and runtime state. A view stores only the root reference
and stable population indices; reading population-shaped values gathers the
selected rows on demand.

## Public behavior

- `cell[index]`, slices, and one-dimensional integer selections return a
  `CellView`. Negative indices are normalized and repeated fancy indices are
  deduplicated in first-occurrence order.
- Views can be indexed again. The second selection is relative to the first.
- Shared declarations and topology (`morpho`, CV policy/tree, node tree,
  solver configuration, and paint rules) are exposed without copying.
- `point_placements`, `place_rules`, `synapses`, and contact groups are
  filtered to declarations or logical instances that affect a selected cell.
- Initialized voltage, spike, runtime ion fields, and runtime mechanism fields
  are gathered along the population axis for read-only inspection.
- `CellView.set(V_init=..., V_th=...)` and the corresponding properties set
  declaration-time values for selected population members. Values are
  concrete voltage quantities and broadcast within the already-fixed
  `(selected_cell, n_cv)` shape. They do not change morphology or layout.
- A callable `V_init` is materialized only by root `Cell.init_state()`;
  declaration-time view inspection never invokes it or consumes its RNG.
- `CellView.place(...)` retains the existing heterogeneous packed placement
  behavior and supports point mechanisms only.

## Deliberate v1 limits

- `CellView.paint(...)` always raises with guidance to use root
  `Cell.paint(...)`. Density mechanisms, including ions and channels, remain
  population-wide and keep dense `(population_size, n_point)` storage.
- Morphology, branch geometry, CV policy, solver configuration, lifecycle, and
  runtime mutation are root-only. Population-specific morphology and density
  masks are deferred.
- View access does not cache gathered arrays. Creating a view is constant-space
  apart from its explicit fancy-index tuple; reading a selected runtime field
  performs the corresponding gather.

## Validation

Tests cover selection composition, compatibility exports, shared-object
identity, filtered declarations and synapse instances, heterogeneous point
placement, selected voltage declarations, initialized state inspection,
runtime ion inspection, paint rejection, phase guards, and unchanged root
broadcast behavior.
