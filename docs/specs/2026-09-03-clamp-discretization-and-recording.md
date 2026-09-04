# Clamp Discretization and Recording

## Fixed-step contract

For a main step starting at absolute time `t`, every current clamp is sampled
exactly once at `t + 0.5 * dt`. The resulting current is held constant for the
whole main step. Solver stages, including Runge-Kutta stages, consume that
cached value and do not evaluate clamp functions at their local stage times.

Clamp windows are left-closed and right-open. `CurrentClamp.durations` are
contiguous segment lengths beginning at `delay`; they are independent of the
duration passed to `run()` and need not be integer multiples of `dt`.

The runtime keeps both per-logical-clamp component currents and their
point-space scatter-add. Clamp recordings read these caches rather than
re-evaluating a declaration. The point-space cache is the value routed to the
midpoint density and DHS boundary-current solver paths.

## Selection and recording

`cell.clamps` returns a composable `ClampView`. Integer and array indexing are
positional within the current view, declaration objects select by identity,
`by_id()` selects stable logical IDs, and strings select clamp types:

```python
cell.clamps["CurrentClamp"]
cell.clamps["CurrentClamp"][1]
cell.clamps.by_type(CurrentClamp)
```

Current clamp declarations do not have semantic names, so `ClampView` has no
`by_name()` operation. A view recording produces one `SampleBlock` whose rows
are the selected logical clamps without spatial reduction. Its time axis is
the actual midpoint sampling time.

Spatial recording uses `observe.clamp_current()`. Its default `reduce="sum"`
groups only within each selected `(population, CV)`; `reduce="none"` retains
one row per matching logical clamp. Both modes read the per-step component
cache. Results use the existing `result.samples` namespace.

## Floating-point boundaries

Piecewise current selection uses one ordered segment lookup rather than
summing independently constructed interval masks. Comparisons snap values
within a small dtype-aware ULP tolerance to the same boundary, with starts
included and ends excluded. This prevents decimal cumulative durations from
creating gaps, overlaps, or an extra terminal sample.

## Deferred work

`ClampView` is intentionally generic enough to include voltage clamps later.
Voltage commands, feedback current, controller state, and solver constraints
are not part of this change and require a separate specification.

Exact boundary-point voltage dependence is currently implemented only by the
staggered/DHS voltage solve. Derivative-based explicit solvers still project
point mechanisms onto CV midpoint voltage and do not expose boundary rows or
re-evaluate mechanisms at Runge-Kutta substeps. Extending those solvers needs a
separate point-state and stage-sampling design; it is intentionally TODO rather
than approximated by the DHS implementation.
