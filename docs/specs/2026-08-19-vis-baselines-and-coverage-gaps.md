# Retire the dead pytest-mpl baselines and cover the three untested modules

## Context

`docs/specs/2026-08-19-test-file-naming-compliance.md` enforced the co-located
`<module>_test.py` rule across `braincell/`. It closed with two defects it had surfaced but
deliberately did not fix, on the grounds that neither was caused by a rename. This change
fixes both.

The naming work is what made them visible. While every test lived in a catch-all file, "no
sibling test file" was not a signal anyone could read.

## Defect 1 — a baseline suite that had never compared anything

`visual_regression_test.py` carried 12 `pytest-mpl` image comparisons pointed at
`braincell/vis/_baseline_images`. **That directory has never existed in the repository.**

Worse, nothing would have noticed. The full picture:

- CI installs `.[testing]`, which included `pytest-mpl`, then runs plain `pytest braincell/`
  — with **no `--mpl` flag** on any of the three OS legs.
- Without `--mpl`, `pytest.mark.mpl_image_compare` is inert. The test body runs, builds a
  figure, returns it, and passes.
- So in CI the 12 tests were figure constructors with no assertions; locally, where
  `pytest-mpl` is absent, they skipped outright.

Either way, zero baselines were ever compared, and the missing directory never bit anyone.

### Why not just commit the baselines

That was the obvious fix and it is the wrong one here. Baseline PNGs are sensitive to font
rasterization and matplotlib version. CI runs a Linux/macOS/Windows matrix; baselines
generated on one machine at `tolerance=25` RMS would fail on the other two the moment
`--mpl` was switched on. Reinstating real pixel regression needs committed baselines *plus*
a Linux-only job pinned to a known matplotlib — a larger piece of work with ongoing
regeneration cost. That option is recorded in `docs/design/TODO.md` rather than half-done here.

### What was actually done

Auditing the 12 against existing coverage found **8 were redundant**:

| Dropped | Already covered by |
|---|---|
| `test_dendrogram_baseline` | `morphometry_test.py::PlotDendrogramTest` — line count, x-axis extent, palette |
| `test_topology_baseline` | `morphometry_test.py::PlotTopologyTest` — branch-order bounds, palette |
| `test_sholl_baseline` | `morphometry_test.py::ShollAnalysisTest` |
| `test_branch_order_histogram_baseline` | `morphometry_test.py::BranchOrderHistogramTest` — bar count *and* heights |
| `test_compare_layouts_baseline` | `compare2d_test.py::CompareLayouts2dTest` — 4 axes, titles, artists |
| `test_stem_frustum_baseline` | `backend_matplotlib_test.py::test_matplotlib_backend_renders_frustum_scene` |
| `test_stem_line_baseline` | `backend_matplotlib_test.py::test_matplotlib_backend_can_render_into_existing_axes` |
| `test_projected_scene_baseline` | `backend_matplotlib_test.py::test_matplotlib_backend_renders_projected_scene` |

Rewriting those as "smoke tests with assertions" would have produced 8 weaker copies of
tests that already assert more. They were deleted.

The remaining 4 reached paths nothing else did, and became
`backend_matplotlib_test.py::MatplotlibLayoutAndColorbarTest` with real artist assertions:

- `layout="balloon"` end-to-end render — no other test renders this family through matplotlib
- `layout="radial_360"` end-to-end render — likewise
- the colourbar, observable only with `show_colorbar` at its default (the one existing values
  test passes `show_colorbar=False`), asserted via the second figure axes and its label
- `vmin`/`vmax` reaching the collection's `Normalize`, pinned against the data range

`compare2d_test.py` gained a `figsize` assertion — the single thing its baseline test covered
uniquely.

`image_comparison()` and `PYTEST_MPL_AVAILABLE` left `vis/_testing.py`; `pytest-mpl` left the
`testing` extra.

## Defect 2 — three modules with no sibling test file

### `braincell/vis/scene_test.py` (20 tests)

`scene.py` is 13 KB, almost all frozen dataclasses that the scene builders construct
incidentally. The tests target the parts that carry behaviour rather than the constructors:

- `BranchValues.segment_values` midpoint rule, including the `size <= 1` branch that returns
  `.copy()` so a caller mutating the result cannot reach back into the branch's own array.
- `OverlaySpec.values_spec()` normalization — `None` through, an existing `ValueSpec` returned
  **by identity**, a bare array wrapped with default styling.
- The eight config delegators resolve **at call time**. `scene.py` imports them from `config`
  at module import; binding the values there instead of the functions would silently staleize
  every scene builder, since they all read colours and alphas through this module. Covered for
  both `configure_defaults()` and the `theme()` context manager.
- `RenderScene2D`/`RenderScene3D` container defaults, and that `RenderRequest`'s two
  `default_factory` fields do not alias across instances.

### `braincell/filter/cache_test.py` (17 tests)

Investigating why `cache.py` had no tests produced the answer: **`SelectionCache`'s three
dictionaries are never read or written anywhere in the tree.** `locset.py:89` does `_ = cache`
to silence the unused argument. The five expressions that would populate them —
`RadiusRangeRegion`, `TreeDistanceRegion`, `EuclideanDistanceRegion`, `SubtreeRegion`,
`StepSamples` — all raise `NotImplementedError`.

So there is no memoization to test, and writing tests that implied otherwise would have been
worse than none. What *is* real and load-bearing is the plumbing: every composite region and
locset expression threads the caller's cache down to its leaves, and nothing checked it. A spy
`RegionExpr`/`LocsetExpr` records the cache each leaf receives, driven through union,
intersection, difference, complement, a nested composite, `UniformSamples`, and `LocsetSetOp`.

Also pinned: omitting the cache yields `None` at the leaves rather than a per-leaf instance,
which would look like memoization while sharing nothing. And each of the five reserved
expressions gets a `NotImplementedError` assertion, so implementing one without wiring up the
matching cache dict becomes a visible edit rather than a silent omission.

### `braincell/io/swc/rules_test.py` (75 tests)

17 KB, 19 rules and 5 helpers. `reader_test.py` exercises all of it through `SwcReader`, which
is correct under the splitting criterion, but that left no test pinning any individual rule's
issue code, its `stop_processing` behaviour, or its report-only path.

The rules are pure functions over a `_SwcContext`, itself a plain dataclass, so every case is
built directly rather than round-tripped through a temp file. Local
`_context`/`_raw`/`_unparsed`/`_parsed` builders; no `_testing.py`, since `reader_test.py`
drives `SwcReader` and has nothing to share — the same reasoning that dropped `filter/_testing.py`
last change.

Notable cases: the three-way `fix_applied` conjunction in `_add_warning`/`_add_error` (each
falsifying leg), both branches of the duplicate-xyzr merge loop, `contour_soma_ids` following
the sequential renumbering, `apply_swc_rules` halting on `stop_processing`, and `rule_contour`
— deliberately a no-op — asserted to add nothing while remaining registered in `SWC_RULES`,
which is what the comment in the source asks for.

**Verified non-vacuous by mutation.** Making `rule_no_soma_samples` always return early fails
exactly 1 test; hardcoding `fix_applied=True` fails exactly 3.

## Test count

No longer a fixed invariant, so the delta is stated instead:

```
2276  after the naming change
 -12  pytest-mpl baselines removed
  +5  4 matplotlib layout/colorbar + 1 compare2d figsize
 +20  vis/scene_test.py
 +17  filter/cache_test.py
 +75  io/swc/rules_test.py
----
2381
```

Skips in `braincell/vis/` drop from 15 to 3; the survivors are plotly and pyvista.

## Not done

- **Coverage numbers were not measured.** `pytest-cov` segfaults at import in this
  environment, on both `scene_test.py` alone and the full suite. The behavioural surface of
  `scene.py` (both `segment_values` branches, all three `values_spec` paths, all eight
  delegators) is reached directly, and `rules_test.py` was mutation-checked instead.
- **`vis/scene.py`'s dataclass field defaults remain partly unexecuted**, which is expected —
  they are data declarations, not logic.
- **Pixel regression is gone, not replaced.** If it is wanted back, it needs committed
  baselines and a `--mpl` CI job; see `docs/design/TODO.md`.

Per the convention this repository now follows, the previous spec is a dated record and was
not edited.

## Addendum — recorded after code review

Two things this document under-reported, added here rather than left to be rediscovered:

- **The `CONTRIBUTING.md` edits went beyond what either spec enumerated.** Alongside the two
  single-test example commands the naming spec listed, this change also rewrote the `testing`
  row of the extras table and the optional-dependency skip paragraph (which now names
  hypothesis, pyvista, and plotly instead of pytest-mpl). Both follow from the pytest-mpl
  removal above and are correct; neither spec said so.
- **The missing-baseline finding no longer has an in-code home.** It was originally written
  into `image_comparison()`'s docstring, and that function was deleted here. `docs/design/TODO.md` and
  these specs are now the only record that `braincell/vis/_baseline_images/` never existed.

The follow-up work from that review is in `docs/specs/2026-08-19-code-review-followups.md`.
