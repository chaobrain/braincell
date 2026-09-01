# Quality cleanup of `braincell.vis`

Reuse, simplification, efficiency, and altitude cleanup across the 33
source modules of `braincell/vis` (~9,200 lines). No public API change,
no intended change to rendered output.

## Scope and method

Four independent reviews swept the package, one per angle. Their
findings were deduplicated — the same three clusters (value-scale
policy, unit stripping, 2D/3D helper duplication) surfaced in three of
the four reviews — and the ones with a concrete, behaviour-preserving
fix were applied.

Two invariants governed every edit:

1. **Rendered output must not change.** Layout geometry is verified
   bit-identical against a pre-change snapshot (see below).
2. **Where two code paths disagreed, the disagreement was preserved,
   not silently unified.** Several duplicated helpers had drifted; the
   shared helper takes the differing input from its caller rather than
   imposing one answer. Divergences worth a decision are listed under
   *Deliberately not changed*.

## Verification

- `braincell/vis/layout` output snapshotted for 11 (fixture, mode,
  family, root_layout) combinations across `grc.swc` and `io.swc`,
  before and after: **bit-identical** (`np.array_equal` on every
  concatenated `segment_points_um`).
- `pytest braincell/` — 2537 passed, 0 failed, 19 skipped, matching the
  pre-change baseline exactly.
- `ruff check` and `ruff format --check` clean across the package.

## Performance

The layout snapshot (11 cases) went from **10.95 s to 0.36 s**. The
dominant term was collision scoring:

| Case | Before | After |
|---|---|---|
| `io.swc` stem/tree | 6.60 s | 0.21 s |
| `io.swc` stem/frustum | 4.11 s | 0.10 s |

### `layout/_collision.py` — scalar rewrite

The pair-scoring predicates ran `numpy.linalg.norm` and `numpy.dot` on
*2-element* vectors, spending microseconds of dispatch on nanoseconds of
arithmetic. Scoring one 114-branch morphology issued ~17M `norm` calls.

Each predicate is now a `_scalar` core over loose floats, with a thin
array-taking wrapper retained for the public/test-facing signature.
`math.sqrt(dx*dx + dy*dy)` is used rather than `math.hypot` precisely
because it reproduces `norm`'s IEEE operation sequence bit-for-bit.
`_pair_score_scalar` also opens with an AABB reject: when the boxes are
more than `margin_um` apart on either axis the score is provably 0.0, so
the expensive predicates are skipped — which is the common case, since
the spatial hash returns whole cells. `_SegmentSpatialHash` stores
`(x0, y0, x1, y1)` float tuples instead of ndarray pairs.

### Other efficiency fixes

- `layout/_geometry.py` — the segment-point walk is a prefix sum;
  replaced with `np.cumsum(..., out=...)`. Runs once per stem placement
  *candidate*, so the saving multiplies.
- `layout/_cache.py` — `_metric_key` ran a per-element
  `round(float(x), 6)` genexpr on every cache **hit**; now one
  `np.round(...).tobytes()` per array.
- `layout/_stem.py` — `tuple(layouts.values())[-N:]` copied every
  layout placed so far on each call (quadratic); replaced with
  `_recent_layouts`, which `islice`s only the window.
- `scene2d.py` / `scene3d.py` — `morpho.branch(index=i)` in a loop
  re-sorts the node table each iteration (O(n² log n)); switched to
  iterating `morpho.branches`, keyed by `branch_view.index`.
- `scene2d.py` — frustum quads built one `np.vstack` per segment, in two
  copies; now one vectorized `_segment_quads_um` per branch.
- `movie.py` — the 2D path materialised all `T` frames of resolved
  scalars before the first frame drew, and rebuilt a loop-invariant
  branch-index list inside the loop. Now hoisted and resolved lazily
  per frame.
- `point_topology.py` — BFS used `list.pop(0)`; now `deque.popleft`.

## Reuse and simplification

- **`_values.py` is now the single home for value/overlay normalisation.**
  It gained `resolve_overlay_values`, `with_unit_label` (via
  `dataclasses.replace`, so a new `ValueSpec` field cannot be dropped),
  `resolve_value_limits`, and `compose_colorbar_label`. `scene2d` and
  `scene3d` had four near-identical private copies of the first two;
  those are gone.
- **One unit stripper.** `_values._strip_quantity` replaces the four
  variants in `traces`, `movie`, and `point_topology` — the last of
  which duck-typed on `to_decimal`/`unit` and so disagreed with the
  others about what counts as a quantity. The dead
  `try: import brainunit / except ModuleNotFoundError` guards are
  removed; the package imports brainunit unconditionally elsewhere.
- **One colour-scale policy.** `resolve_value_limits` replaces the
  bound-resolution and degenerate-range padding open-coded in all three
  backends and `movie`. It takes the scalar arrays from its caller, so
  each backend keeps feeding it the arrays it actually renders.
- **One availability probe.** `backend.module_available` replaces the
  identical 5-line `find_spec`/`sys.modules` body in all three backends.
- **One palette decode.** `config.rgb_to_float` replaces three
  byte-identical `_rgb_to_float` definitions.
- **One VTK cell-walk.** `backend_plotly._iter_batch_polylines` replaces
  two copies of the `[n, i0…i{n-1}]` decoder, each of which carried a
  dead cursor variable the other did not.
- `config._copy_defaults` re-listed all 15 `VisDefaults` fields by hand
  — a field forgotten there leaks across a `theme()` block. Now
  `dataclasses.replace` with explicit copies of only the two mutable
  dicts. The two colour-map merge blocks collapse into `_merge_color_map`.
- Dead code removed: `point_topology._normalize_scalar_values` (no
  callers), a tautological `show_colorbar` expression in `plot2d`, a
  redundant `elif` arm in `movie._save_animation_2d`, a no-op branch and
  its `n_branches` local in `backend_pyvista`, a discarded
  `sample_layout_branch` tangent in `layout/_fan` (now
  `point_on_layout_branch`), and six unused imports.
- `point_topology._VALID_COLOR_MODES` / `_VALID_COVERAGE_MODES` are
  derived from their `Literal` aliases via `typing.get_args` rather than
  restating each value.

## Deliberately not changed

These are real findings whose fix would change behaviour or exceed the
remit of a quality pass. Each needs a decision, not a refactor.

- **`layout/_stem.py:319` uses a hardcoded collision window of 48**
  while the tree path reads `layout_config.stem_collision_window`
  (default 24) — so that config knob silently does nothing on the
  `shape='frustum'` path. Preserved, and now commented at the site.
  Aligning them changes rendered geometry.
- **The matplotlib backend derives its colour scale from segment-midpoint
  values; PyVista and Plotly derive it from point values.** The same
  `ValueSpec` therefore yields a slightly wider scale in 3D than in 2D.
  `resolve_value_limits` deliberately does not unify this — each caller
  still passes its own arrays.
- **`point_topology` pads a degenerate colour range by `+1.0`** where
  everything else uses `±0.5`. Left alone for the same reason.
- **`movie._format_time` renders `"t = <frame> × dt"`** — a placeholder
  that ignores the `dt` the caller threads in, despite the docstring
  promising elapsed time. The unused parameter is removed; whether to
  render real time is a product decision.
- **`traces.plot_traces` builds the 2D scene twice** and reconstructs
  centerlines from rendered polygons, defaulting to `layout="stem"`
  where `plot2d` defaults to `"fan"`. Fixing it properly means letting
  `OverlaySpec` carry per-point marker colours.
- **`layout/_legacy.py` carries ~175 lines its own docstring marks as
  preserved-for-history.** Deleting is a maintainer's call, not a
  cleanup.
- **Larger structural items** left for a dedicated change: the
  duplicated stem/stem-linear walker (~180 lines), a `LayoutFamily`
  registry to replace the family list spread across five places,
  `export.py`'s second backend registry, and unifying the 2D/3D
  arc-length samplers.
