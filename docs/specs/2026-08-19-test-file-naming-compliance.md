# Enforce the co-located `<module>_test.py` rule across `braincell/`

## Problem

`AGENTS.md` rule 10 requires every source module `foo.py` to keep its tests in a
sibling `foo_test.py`. Twenty-four test files under `braincell/` violated it:
their stem named no source module at all.

```
braincell/_compute/_layering_test.py            braincell/io/asc/asc_test.py
braincell/_compute/spatial_params_test.py       braincell/io/swc/swc_test.py
braincell/_compute/topology_test.py             braincell/ion/_docstring_test.py
braincell/_discretization/build_test.py         braincell/network/runtime_test.py
braincell/_multi_compartment/cell_vis_discrete_test.py   braincell/network/topology_test.py
braincell/_multi_compartment/cell_vis_node_test.py       braincell/vis/layout/_property_test.py
braincell/channel/_deprecation_test.py          braincell/vis/perf_benchmark_test.py
braincell/channel/_docstring_test.py            braincell/vis/vis_geometry_test.py
braincell/filter/filter_branch_filters_test.py  braincell/vis/vis_plot_test.py
braincell/filter/filter_locset_test.py          braincell/vis/vis_pyvista_real_files_test.py
braincell/filter/filter_region_test.py          braincell/vis/vis_real_files_test.py
braincell/filter/filter_vis_test.py             braincell/vis/visual_regression_test.py
```

The cost was not cosmetic. Catch-all names hid real coverage gaps:

- `braincell/network/` had 7 modules and 0 matching test files; `braincell/filter/`
  had 5 modules and 0 matching test files. Every test lived in `runtime_test.py`,
  `filter_region_test.py`, and friends.
- `braincell/vis/` had 9 modules with no sibling test file, including
  `backend_matplotlib.py` (20 KB), `scene2d.py` (23 KB), and `scene3d.py` (13 KB).
- `filter/cache.py` had zero direct tests; `network/delivery.py` was exercised by
  2 of 41 tests. Neither was visible while the tests were filed under names that
  corresponded to nothing.

The repository already paid for a naming slip once: a bare `test.py` matched
neither collection pattern and silently hid 72 SWC/ASC tests.

## Decisions

Three came from clarifying questions before any code was written:

1. **Package-scope guard tests** — those with genuinely no module counterpart —
   go to `<package>/__init___test.py`, one per package.
2. **Scope is `braincell/` only.** `examples/` and `dev/` are untouched;
   `examples/neuron_compare/cable/tests/` is a documented `pyproject.toml`
   exception and `dev/` is gitignored scratch.
3. **Files spanning several modules are split by target module**, not renamed
   wholesale.

A fourth, applied throughout to keep the split mechanical:

4. **Splitting criterion.** A test moves to `X_test.py` when it calls `X`'s API
   *directly*. A test that drives a higher-level entry point — `SwcReader`,
   `Cell`, `Network.run` — stays with that entry point's module even when the
   behaviour under test originates deeper. This is what keeps ~11 soma-*geometry*
   tests in `io/swc/reader_test.py` while only the 3 tests that call
   `is_contour_soma` / `is_special_three_point_soma` move to `soma_test.py`.

## Invariant

`pytest braincell/ --collect-only` reported **2276 tests** before the first edit
and must report 2276 after the last. Nothing may be dropped, merged away, or
silently un-collected. The count was re-checked after every commit.

This matters more than it sounds: `pytest-mpl`, `pyvista`, and `plotly` are absent
in the development environment, so those suites *skip*. A skip guard that stops
matching looks identical to a passing run. Only the collected count catches it.

## What changed

### Renames (history preserved via `git mv`)

| From | To |
|---|---|
| `filter/filter_locset_test.py` | `filter/locset_test.py` |
| `io/asc/asc_test.py` | `io/asc/reader_test.py` |
| `io/swc/swc_test.py` | `io/swc/reader_test.py` |
| `_compute/_layering_test.py` | `_compute/__init___test.py` |
| `ion/_docstring_test.py` | `ion/__init___test.py` |
| `channel/_docstring_test.py` (+ `_deprecation_test.py` merged in) | `channel/__init___test.py` |

### Splits

| Source | Destinations |
|---|---|
| `_compute/topology_test.py` | `_compute/scheduling_test.py`, `_discretization/node_build_test.py` |
| `_compute/spatial_params_test.py` | `_compute/layouts_test.py` |
| `_discretization/build_test.py` (1004 lines, 22 classes) | `mechanism_test.py`, `geometry_test.py`, `base_test.py` |
| `_multi_compartment/cell_vis_{discrete,node}_test.py` | `_multi_compartment/cell_test.py` |
| `filter/filter_branch_filters_test.py` + `filter/filter_region_test.py` | `filter/region_test.py`, `filter/locset_test.py`, `filter/helper_test.py` |
| `filter/filter_vis_test.py` | `morph/morphology_test.py`, `morph/branch_test.py`, `vis/plot3d_test.py` |
| `io/swc/reader_test.py` | carve out `io/swc/soma_test.py` |
| `network/runtime_test.py` + `network/topology_test.py` | `core_test.py`, `lowering_test.py`, `edges_test.py`, `projections_test.py`, `engine_test.py`, `delivery_test.py` |
| `vis/vis_geometry_test.py` | `scene3d_test.py`, `scene2d_test.py`, and the six `layout/_*_test.py` |
| `vis/vis_plot_test.py` | `plot2d_test.py`, `plot3d_test.py`, `backend_test.py`, `backend_matplotlib_test.py`, `compare2d_test.py`, `config_test.py`, `scene2d_test.py`, `scene3d_test.py`, `_values_test.py` |
| `vis/visual_regression_test.py` | `plot2d_test.py`, `morphometry_test.py`, `compare2d_test.py` |
| `vis/perf_benchmark_test.py` | `layout/_dispatch_test.py`, `scene2d_test.py`, `plot2d_test.py` |
| `vis/vis_real_files_test.py` | `scene3d_test.py` |
| `vis/vis_pyvista_real_files_test.py` | `backend_pyvista_test.py` |
| `vis/layout/_property_test.py` | `vis/layout/_dispatch_test.py` |

`filter/filter_vis_test.py`'s subject was never `filter/` at all: it tests
`Morphology.select` / `.vis2d` / `.vis3d`. Filter symbols are only its inputs.
That is why 11 of its 14 tests landed in `morph/`.

`network/delivery_test.py` is a thin 2-test file. That is an honest signal that
`delivery.py`'s ring-buffer machinery is only tested implicitly through
`Network.run`. It was left thin rather than padded.

### New shared helpers

`_testing.py` modules (leading underscore, so pytest never collects them):

- `braincell/_discretization/_testing.py` — `build_geo`, `make_branch`,
  `make_cable`, `make_single_branch_morpho`, `make_two_branch_morpho`
- `braincell/network/_testing.py` — 8 cell / solver / tree builders shared by all
  six destination files
- `braincell/vis/_testing.py` — gained `VisDefaultsResetMixin`,
  `make_four_type_tree`, `image_comparison`, `PYTEST_MPL_AVAILABLE`,
  `PYTEST_BENCHMARK_AVAILABLE`, the `FIXTURE_DIR` / `VALID_SWC_FIXTURES` /
  `ALLOWED_TYPES` constants that were duplicated verbatim across two files, and
  `FakeBackend` folded in from the deleted `vis/_test_helper.py`

`braincell/vis/_test_helper.py` was deleted: a 1-symbol module whose job
`_testing.py` already has. It had three importers, all repointed.

Two rival module-level `_build_tree()` helpers in `filter/` had *different*
geometry (3 branches vs 4). They were renamed `_soma_dend_axon_tree` and
`_soma_dend_axon_tuft_tree` before the files could be merged, rather than one
silently winning.

`perf_benchmark_test.py::_synthetic_tree` was byte-for-byte identical to
`vis/_testing.make_deep_chain_tree` and was dropped in favour of it.

### Optional-dependency guards

These are the part of a split most likely to go wrong quietly, because a lost
guard turns into a *skip*, not a failure.

- `visual_regression_test.py` had a class-level
  `@unittest.skipUnless(_pytest_mpl_available)`. Each destination class carries
  its own now, keyed on `PYTEST_MPL_AVAILABLE`.
- `perf_benchmark_test.py` had a module-level
  `pytestmark = skipif(not find_spec("pytest_benchmark"))`. A module-level mark
  would now skip each destination file *entirely*, since the destinations also
  hold ordinary tests. It became a per-function `@_needs_benchmark`.
- The same file's autouse `_clear_layout_cache` fixture would likewise have
  applied to every unrelated test in each destination. It became an explicitly
  requested `clean_layout_cache` parameter on the five benchmark functions.
- The benchmarks stay plain pytest functions, never `TestCase` methods: pytest
  cannot inject the function-scoped `benchmark` fixture into a `TestCase`.
- `layout/_property_test.py`'s `_hypothesis_available` guard and no-op `given`
  stub followed into `_dispatch_test.py`.
- `vis_pyvista_real_files_test.py`'s runtime `self.skipTest("pyvista is not
  installed.")` path followed into `backend_pyvista_test.py`.

Verified after the move: with `pytest-benchmark` present, `--benchmark-only`
still runs all 7 benchmarks.

## Deviations from the plan

Stated here rather than left implicit:

- **`filter/_testing.py` was not created.** Once the two `_build_tree` helpers
  were given distinct names, each had exactly one consumer. A shared module would
  have been indirection with no sharing.
- **`test_tree_and_frustum_ignore_real_points_geometry` was kept whole** in
  `scene2d_test.py` rather than split across a layout half and a scene half.
  Splitting one method into two is a rewrite, and it would move the collected
  count off 2276.
- **`layout/_property_test.py` went to `_dispatch_test.py`, not
  `__init___test.py`.** It was bucketed as a guard test when the
  `__init___test.py` decision was made, but it drives `build_layout_branches_2d`
  — it has a real module target. `__init___test.py` is reserved for files with
  genuinely no counterpart.
- **`config_test.py` keeps both restore-on-exception tests.** The plan flagged
  them as possible duplicates. They are not: the existing one asserts matplotlib
  `rcParams` are restored by `publication_theme()`, the incoming one asserts
  `branch_type_colors` is restored by `vis.theme()`.
- **`docs/specs/` was not rewritten.** Earlier specs name files this change
  renamed, but a dated spec is a record of what was done at the time. Falsifying
  it to match today's tree would destroy the thing that makes the directory
  useful. Only living documentation — `AGENTS.md`, `CONTRIBUTING.md`, `docs/design/TODO.md`,
  `docs/design/`, `pyproject.toml` — was updated.

## Defects surfaced, not introduced

1. **`braincell/vis/_baseline_images/` does not exist.** `visual_regression_test.py`
   documented it as the pytest-mpl baseline directory. Even with the plugin
   installed and `--mpl` passed, every comparison fails on a missing baseline —
   the suite has never actually compared anything. Recorded in the
   `image_comparison()` docstring along with the regeneration command.
   Generating the PNGs is a separate change.
2. **`vis/scene.py` (13 KB) has no direct coverage** in any of the five files
   that were split, and still has none. This is now visible: there is no
   `vis/scene_test.py`, and the absence reads as an absence rather than being
   masked by a catch-all filename.
3. `braincell/io/swc/rules.py` and `braincell/filter/cache.py` likewise have no
   sibling test file; their behaviour is asserted only through `SwcReader` and
   through the region/locset evaluators respectively. Correct under the
   splitting criterion, and now legible as gaps.

## Documentation updated

- `AGENTS.md` — three new clauses in Testing: the stem must name a real sibling;
  package-scope guards go in `<package>/__init___test.py`; optional-dependency
  guards travel with the tests they protect. Fixture-path example corrected to
  `reader_test.py`.
- `CONTRIBUTING.md` — both single-test example commands.
- `docs/design/io-swc-reader-invariants.md` — the "tests to re-run" lists, which
  also still named `_discretization/lower_test.py` (long since `base_test.py`).
- `docs/design/TODO.md` — the `vis/` module inventory and the M6 Phase 4 status entries.
- `pyproject.toml` — the `python_files` comment.

## Verification

```bash
# every *_test.py names a sibling module or the package __init__
python3 - <<'EOF'
import pathlib
bad = []
for f in sorted(pathlib.Path('braincell').rglob('*_test.py')):
    stem = f.name[:-len('_test.py')]
    if stem == '__init__' and (f.parent / '__init__.py').exists():
        continue
    if not (f.parent / f'{stem}.py').exists():
        bad.append(str(f))
print('VIOLATIONS:', *bad, sep='\n  ') if bad else print('OK: no violations')
EOF

pytest braincell/ --collect-only -q | tail -1   # must be 2276
pytest braincell/ -q
pytest braincell/vis/ --benchmark-only -q       # 7 benchmarks still run
pre-commit run --all-files
```

Result: zero violations, 2276 collected, full suite green, 7 benchmarks run.
