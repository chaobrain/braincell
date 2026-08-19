# Close the code-review findings on the test-naming branch

## Context

A two-axis review ran over `main..HEAD` of `worktree-test-naming-compliance` — the 12 commits
covering `docs/specs/2026-08-19-test-file-naming-compliance.md` and
`docs/specs/2026-08-19-vis-baselines-and-coverage-gaps.md`. One axis checked the diff against
the repository's documented standards plus a fixed set of Fowler code smells; the other checked
it against those two specs.

**No hard violations.** Both axes independently confirmed the substance: all 107 test modules
name a real sibling, the three sanctioned `<package>/__init___test.py` guards are the only
exceptions, all 21 new files carry the verbatim 2026 licence header, no optional-dependency
skip guard was lost in the splits, `2276 − 12 + 5 + 20 + 17 + 75 = 2381` is exact, and every
factual claim in both specs survives re-verification against the tree.

What came back was seven judgement calls. This change closes all seven.

## Standards findings

### Triplicated benchmark scaffolding

`vis/layout/_dispatch_test.py`, `vis/plot2d_test.py`, and `vis/scene2d_test.py` each carried a
character-for-character copy of the same 8-line banner comment, the same `_needs_benchmark`
skip mark, and the same `clean_layout_cache` fixture body.

The mark moved to `vis/_testing.py` as `needs_benchmark`, next to the
`PYTEST_BENCHMARK_AVAILABLE` probe it is built from.

The fixture moved to a **new `braincell/vis/conftest.py`** rather than to `_testing.py`. A
fixture is the one helper kind that cannot travel by import: pytest resolves it by name from
the collected `conftest.py` chain, so putting it in `_testing.py` would force each consumer to
carry an import that exists only to satisfy the lookup — and a `# noqa: F401` to stop ruff
deleting it. `vis/conftest.py` sits above `vis/layout/`, so `_dispatch_test.py` inherits it.

The fixture imports `matplotlib` inside its body rather than at module scope. `conftest.py` is
imported when pytest collects the directory, so a top-level import of an optional dependency
would turn a matplotlib-less install into a collection error for the whole of `braincell/vis/`
instead of for the individual modules that need it — and `CONTRIBUTING.md` requires the vis
dependencies stay lazy regardless.

This is the change most able to fail silently: a fixture pytest cannot find turns benchmarks
into errors, and a mark that evaluates wrongly turns them into skips. Verified by
`pytest braincell/vis/ --benchmark-only`, still **7 passed**.

### `FIXTURE_DIR` defined four times at two depths

`vis/_testing.py`, `io/swc/reader_test.py`, `io/asc/reader_test.py`, and `io/checkpoint_test.py`
each recomputed `Path(__file__).resolve().parents[N] / "data" / "morphology"` — `N` being 2 or
3 depending on how deep the file sits, which is exactly the kind of constant that goes quietly
wrong when a test file moves between package levels. `ALLOWED_TYPES` had three copies.

**New `braincell/io/_testing.py`** owns `FIXTURE_DIR`, `VALID_SWC_FIXTURES`, and
`ALLOWED_TYPES`. The io package is the right owner — the fixtures are SWC and ASC files its
readers parse. `vis/_testing.py` re-exports all three, so `backend_pyvista_test.py` and
`scene3d_test.py` still import from one place and were not touched.

`AGENTS.md` §Testing previously taught counting `parents[N]` by hand and named those four files
as worked examples. It now says to import the constant, and explains why the hand-rolled form
is a latent bug.

### Stale docstring

`make_four_type_tree`'s docstring still justified itself by "what the baseline-image regression
figures need" — figures deleted two commits earlier in the same branch. Repointed at its real
consumers.

### Thin `filter/helper_test.py`

It asserted three `__all__` memberships against a module exporting seventeen names. The
interval algebra underneath is what every `RegionSetOp` in `region.py` delegates to, so a sign
error there surfaces as a subtly wrong selection rather than an exception — `region_test.py`
drives it through the expression tree, but nothing pinned the arithmetic.

Now 31 tests over pure `tuple[int, float, float]` triples, no `Morphology` needed, so every
expected result is checkable by eye: merge/absorb/drop-zero-length and the `_clip_norm_x`
epsilon tolerance in `normalize_region_intervals`; union across operands; intersection dropping
a branch present on only one side and treating a shared endpoint as no overlap; difference
splitting a range in two when the subtrahend is strictly interior, and its asymmetry;
complement against `n_branches`, including that an out-of-range index on the input cannot
conjure a branch into the result. The `__all__` test was widened to the full export set.

## Spec findings

### A deleted test's replacement did not, in fact, assert more

The sharpest finding. The previous spec justified deleting eight pytest-mpl tests on the
grounds that "rewriting those as smoke tests with assertions would have produced 8 weaker
copies of tests that already assert more" — then cited
`backend_matplotlib_test.py::test_matplotlib_backend_renders_projected_scene` as the
replacement for `test_projected_scene_baseline`. That test was:

```python
axes = plot2d(tree, layout="projected", shape="line", ...)
self.assertIsInstance(axes, matplotlib.axes.Axes)
```

— precisely the assertion-free figure constructor the spec was criticising. The other seven
replacements hold up on inspection.

It now pins the projection itself: `make_node_tree()` is one soma branch running
`(0,0,0) -> (10,0,1)`, so the default xy projection must produce one line with xdata `[0, 10]`
and ydata `[0, 0]` — z dropped — on equal-aspect axes.

### The second-weakest test

`ShollAnalysisTest::test_plot_sholl_returns_ax_with_one_line` asserted
`assertGreaterEqual(len(ax.lines), 1)`. Renamed to `test_plot_sholl_draws_the_computed_profile`
and now asserts exactly one line whose x/y data equal `compute_sholl_profile(tree,
step_um=5.0).radii_um` / `.intersections`, plus both axis labels — so `plot_sholl` and
`compute_sholl_profile` are held to the same `step_um`.

### Unenumerated `CONTRIBUTING.md` edits

Recorded as an addendum on the previous spec rather than fixed; the edits were correct, just
not listed. See that document.

## Not done

- **The mutation claim in the previous spec is still unverified by the review**, because
  checking it requires editing `io/swc/rules.py`. It was performed and recorded when that spec
  was written; nothing in this change touches those tests.
- **Coverage numbers remain unmeasured.** `pytest-cov` still segfaults at import in this
  environment.
- `braincell/_compute/*_test.py` uses relative imports (`from ._testing import ...`), which
  contradicts `AGENTS.md` §Import style. Pre-existing and untouched by either change — left for
  a focused pass rather than widened into this one.

## Verification

```
pytest braincell/vis/ --benchmark-only -q     # 7 passed, 322 skipped
pytest braincell/filter/helper_test.py -q     # 31 passed
pytest braincell/ -q                          # full suite
pre-commit run --all-files
```
