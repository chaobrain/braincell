# Quality cleanup of `braincell.morph`

Reuse, simplification, efficiency, and altitude cleanup of the morphology
package (`braincell/morph`: `morphology.py` 2,257 lines, `branch.py` 1,140,
`_spatial.py` 216, `__init__.py` 30, plus 1,594 lines of co-located tests).
Breaking API changes are in scope and were explicitly authorised for this
pass.

This is iteration 2 of a module-by-module sweep. PR #137's package-wide pass
already took morph's hot paths — it memoised `_ordered_node_ids_by`,
`_branch_index_map`, `branch_by_order`, and `has_full_point_geometry`, and
rewrote `_spatial.build()` to walk `morpho.edges` once. This pass therefore
looks for what those left behind, and for the structural problems a
performance pass does not surface.

## Scope and method

Four independent reviews swept the package, one per angle (reuse,
simplification, efficiency, altitude). Findings were deduplicated; where three
reviews independently converged on the same defect it is treated as
load-bearing. Every claim acted on below was re-verified against the code by
running it, not by reading the report.

Two reported claims did **not** survive that re-verification and are recorded
here so they are not reintroduced:

- Two reviews claimed four or five `Morphology` names were shadowable. Only
  **two** actually are (`topo`, `from_neuromorpho`). `save_checkpoint`,
  `load_checkpoint`, and `diam_arc_mean` are already rejected, because
  `Branch` happens to define them too and `_BRANCH_RESERVED_NAMES` is derived
  from `dir(Branch)`. The defect is real; its blast radius was overstated.
- The claim that `Morphology.topo()`'s recursion could be fixed by reusing
  `braincell.vis._traversal.iter_depth_first` does not hold: that helper
  returns a flat list of `MorphoBranch` objects, while `_format_topology`
  needs per-node prefix and is-last bookkeeping to draw the tree. The fix is
  an explicit stack, not a call to the existing helper.

Three invariants govern every edit, carried over from PR #137 and iteration 1:

1. **The test suite stays green.** Baseline on `main` @ `4c7c572` before any
   change: 2,713 passed, 15 skipped, 338 subtests, 0 failed (181.94 s).
   `braincell/morph` alone: 82 passed, 18 subtests (7.75 s).
2. **Every performance claim carries a measurement.** No fix lands on the
   strength of "this looks faster".
3. **Where two code paths disagree, the disagreement is preserved, not
   silently unified.** Divergences worth a decision are listed under
   *Deliberately not changed*.

## Bugs fixed

Both were found by review and reproduced before being fixed, per the working
agreement's "write a test that reproduces it, then fix until it passes".

### `Morphology.topo()` raised `RecursionError` on deep morphologies

`_format_topology` recursed once per branch. Reproduced on a 1,200-branch
chain against the default recursion limit of 1,000:

```
chain of 1200 branches, recursionlimit=1000
topo() -> RecursionError
```

This is the same hazard `docs/specs/2026-08-13-vis-iterative-tree-walks.md`
fixed for the rendering paths ("a morphology deeper than roughly 400 branches
raised `RecursionError`"); that spec scoped itself to `braincell/vis/`, so the
package that *owns* the tree kept the bug. `_format_topology` becomes an
explicit-stack loop, with a regression test at a depth that overflows the
recursive form.

### Two branch names silently shadowed the API

`_validate_public_name`'s whole job is to reject a branch name that would
shadow a `Morphology` attribute. Its `_MORPHO_RESERVED_NAMES` set is a
hand-typed copy of `Morphology`'s public surface and had drifted, so:

```
attach(child_name="topo")             -> ACCEPTED
morph.topo                            -> <bound method>   # branch unreachable
morph.branch(name="topo")             -> the branch       # but it is in the tree
attach(child_name="from_neuromorpho") -> ACCEPTED, same shadowing
```

The branch is in the tree and reachable by `branch(name=...)`, but attribute
access silently returns the method, because `__getattr__` only fires when
normal lookup fails.

The fix is at the depth the file already demonstrates three lines below the
defect: `_BRANCH_RESERVED_NAMES` is *derived* from `dir(Branch)`. The three
hand-typed literals (`_MORPHO_RESERVED_NAMES`, `_MORPHO_BRANCH_RESERVED_NAMES`,
`_MORPHO_METRIC_PROPERTY_NAMES` — 55 lines of names) collapse into one derived
`_ALL_RESERVED_NAMES`; the metric names needed no clause of their own, because
they are already properties on `Morphology` and so already in `dir()`. The set
is computed at the *end* of the module, the earliest point at which `dir()` can
see both classes.

One clause is not derivable and is unioned in explicitly:
`_MORPHO_BRANCH_PUBLIC_ATTRS` — the five names (`branch`, `name`, `parent_id`,
`parent_x`, `child_x`) that `MorphoBranch.__getattr__` serves dynamically and
that therefore never appear in `dir(MorphoBranch)`. Deriving from `dir()` alone
would have *regressed* those five from protected to shadowable; this was caught
by checking empirically rather than by reading, and is now pinned by a test.

This is a **behaviour tightening**: `topo` and `from_neuromorpho` become
rejected branch names. Every future public method on `Morphology` is now
covered automatically instead of reopening the hole.

## Breaking changes

### `Morphology.path_length_to_root` and `Morphology.shortest_path_length` are deleted

52 lines whose entire body is `raise NotImplementedError`, verified by calling
both. They have zero callers anywhere in `braincell/`, `examples/`, or the
notebooks, and `pyproject.toml`'s `report.exclude_also = ["raise
NotImplementedError"]` means coverage never flagged them either — they were
doubly invisible. Their two bullets in `docs/design/interface-map.md` go with
them.

Deleting rather than implementing is a deliberate scope decision. The
capability does exist in the package — `_spatial.MorphologySpatialGeometry`
runs multi-source Dijkstra and is what `braincell/filter`,
`braincell/network`, and `braincell/_discretization` already import for these
numbers — but wiring it up is feature work, not simplification. See
*Deliberately not changed*.

### `Morphology.diam_arc_mean` is deleted

`2.0 * self.mean_radius` with zero callers on a `Morphology` (every one of the
40+ `diam_arc_mean` hits repo-wide is on a `Branch`, `MorphoBranch`, or a CV
object) and never executed under the test suite.

### `Morphology._get_branch` is deleted

Zero callers, including its own definition site's file.

## What was fixed

### Reuse — one home per concept

- **The frustum formulas existed twice.** Lateral area
  `π(r₀+r₁)√(L²+(r₁−r₀)²)`, volume `πL(r₀²+r₀r₁+r₁²)/3`, and the
  length-weighted mean radius are written character-for-character identically
  in `Branch` (over one branch's segments) and in `Morphology` (over the
  concatenated arrays). `morphology_test.py` asserts the two agree, so the
  copies are load-bearing on each other: a fix to `Branch.areas` — for
  instance to the zero-length jump-segment end-cap area that `branch.py`
  documents as surprising — would silently leave `Morphology.total_area`
  disagreeing. Module-level pure-NumPy helpers in `branch.py` now back both.
  `Morphology` keeps its concatenated fast path; only the formula moved.
- **`Morphology.branch(index=...)` inlined `_node_id_from_index`**, down to
  the identical `IndexError` message. Collapsing it also revives that helper's
  `order` parameter, which every existing call site left at its default.
- **`MorphoBranch.branch_id` duplicated `MorphoBranch.index`'s body** rather
  than returning it, while its own docstring called itself an alias.
- **`Morphology.max_branch_order` re-derived `MorphoBranch.branch_order`**;
  `morphology_test.py` already asserted the two agree.
- **`_parse_attachment_key` re-implemented `_validate_parent_x` /
  `_validate_child_x`**, with byte-identical message strings, and had already
  diverged: it calls `float(key)` first, so an out-of-range integer reports
  `2.0` through `morph.soma[2]` and `2` through `attach(parent_x=2)`.
- **`Branch.from_lengths` and `Branch.from_points` shared a verbatim
  `_BRANCH_TYPE`/`_UNSET` resolution block**, differing only in the method
  name inside the error string.
- **`morph` had no `_testing.py`**, the only domain package without one, even
  though its own test files had reached for per-`TestCase` fixture helpers
  three separate times because there was nowhere shared to put them, and even
  though the morph tests already import *upward* into `braincell.vis._testing`
  for `FakeBackend`.

### Simplification

- **Five unreachable guards and tails.** `_point_on_branch_at_x_um`'s
  `points is None` raise (its caller has just asserted full point geometry for
  every branch, root included) and its post-loop `return points_um[-1]` (the
  loop condition guarantees the last iteration returns); `mean_radius`'s
  `total length must be > 0` (every `Branch` guarantees it at construction and
  a `Morphology` always has at least one branch); `_root_attach_distances_um`'s
  `parent_id is None` continue (the root is already skipped two lines above,
  and only the root has a null parent); `_spatial._interpolate_scalar`'s
  trailing `last_positive is None` raise. Each was confirmed unexecuted by a
  coverage run over 624 tests.
- **`_root_branch_attach_x()` returned the constant `0.0`** through two call
  sites.
- **`Morphology.edges` used a one-element-tuple rebinding trick** to bind a
  loop variable, and redid N dict lookups per access instead of using the
  memoised `self.branches`. Fixing it makes `_ordered_node_ids()` — a one-line
  wrapper whose only caller it was — dead.
- **`Branch.vis2d` / `vis3d` re-forwarded to `plot2d` / `plot3d` instead of
  calling `Morphology.vis2d` / `vis3d`.** Both already build
  `Morphology.from_root(self, name="soma")`, then re-list a strict subset of
  the arguments the `Morphology` methods forward and re-implement the same
  `show` handling. `Branch.vis2d` had already silently fallen behind on six
  parameters. Every default the `Morphology` methods pass explicitly
  (`ax=None`, `notebook=None`, `jupyter_backend=None`,
  `min_branch_angle_deg=25.0`, `root_layout="type_split"`) was checked against
  `plot2d`'s own signature and matches, so delegating is behaviour-identical.
- **A dead skip guard** in `branch_test.py` tested `if jnp is None` under an
  unconditional `import jax.numpy as jnp` — if jax were absent, collection
  would already have failed at the import.
- **41 copies of five branch literals** in `morphology_test.py` (the same
  `Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um,
  type="soma")` appeared 21 times), which made arbitrary dimensions read as
  though they were significant. They move to the new
  `braincell/morph/_testing.py` as `make_soma` / `make_dendrite` /
  `make_basal` / `make_apical` / `make_axon`; a test that genuinely depends on
  a radius still passes it explicitly.
- **`make_deep_chain_tree` lived in `braincell/vis/_testing.py`.** It builds a
  `Morphology` and contains nothing visualization-specific, and the morph
  tests need it for the `topo()` regression above. It moves to
  `braincell/morph/_testing.py`, and `vis/_testing.py` re-exports it — the
  same pattern that file already uses for `FIXTURE_DIR` from
  `braincell/io/_testing.py`.

### Documentation that described behaviour the code does not have

- **28 references to a class named `Morpho`**, which does not exist and has
  not for some time. These are not only comments: `Morphology.__repr__`
  printed `Morpho(root='soma', n_branches=1, ...)`, and six user-facing error
  messages named a symbol no user can look up (`"reserved by the Morpho
  API"`, `"already exists in this Morpho"`, `"Morpho.select(...) expects"`).
  Nine `See Also` entries and four Sphinx roles resolved to nothing.
- **Five docstrings pointed at `braincell.morph.vis.configure(...)`**, which
  is wrong twice: there is no `braincell.morph.vis` module, and the public
  name is `braincell.vis.configure_defaults`.
- **Two docstrings enumerated the 2-D layouts omitting `"fan"`** — which is
  the default, named as such three lines later in the same docstring.
- **`Branch.from_lengths` documented a `ValueError` on radius
  discontinuities** that it does not raise: `_canonicalize_segments` inserts a
  zero-length jump segment instead, as `branch_test.py` already proves.

## Efficiency — measured, or not landed

Measured on `data/morphology/CA1.swc` (681 branches, 39,277 points), median of
7 repeats, a freshly parsed morphology per repeat so no cache is shared across
samples. `before` is `main` @ `4c7c572`; `after` is this branch. Both runs
print the result of every benchmark, and **all eight digests are identical**
across the two trees — 39,277 points, total length 16,094.912745159878 µm,
x-range 490.03 µm, 681 branches, 6,800 edges, a 67,885-character `topo()`, and
an interpolated radius sum of 4,144.2883 µm. That equality is the evidence
these are speedups and not behaviour changes.

| Benchmark | Before | After | Speedup |
|---|---:|---:|---:|
| `metric.as_dict()` | 253.62 ms | 56.86 ms | **4.5×** |
| `clone_morpho` | 259.55 ms | 59.13 ms | **4.4×** |
| `interpolate_branch`, 256 sites × 86 branches | 595.03 ms | 201.03 ms | **3.0×** |
| `Branch.points` over every branch | 53.19 ms | 19.15 ms | **2.8×** |
| `x_range` | 52.46 ms | 21.03 ms | **2.5×** |
| `Branch.length` over every branch | 12.32 ms | 8.15 ms | **1.5×** |
| `edges` × 10 | 5.70 ms | 5.61 ms | 1.0× |
| `topo()` | 0.47 ms | 0.45 ms | 1.0× |

Four changes produced those numbers:

- **`Branch.points` did unit-aware math on a two-line hot path.**
  `u.math.allclose` and `u.math.concatenate` each re-derive dimensions and
  tree-map over their arguments on every call, and `points` is reached by
  essentially every whole-morphology consumer. Decoding to plain NumPy once
  and doing the continuity check and the concatenate there is the single
  largest contributor; it is why `metric`, `clone_morpho`, and `x_range` all
  move together.
- **`_all_segment_arrays_um()` was recomputed per aggregate.** `mean_radius`,
  `total_area`, `total_volume`, and each of `x_range` / `y_range` / `z_range`
  rebuilt the whole-tree arrays independently, so `metric.as_dict()` paid for
  them six times over. Both it and the new `_all_points_um()` are memoised on
  the instance and cleared from `_invalidate_derived_caches`, which is already
  the single choke point for cache invalidation (`_register_node` is the only
  place `_nodes` grows).
- **`Branch.length` decoded both radius arrays to discard them.** It called
  the shared `_segment_arrays_um()` helper and used one of its three outputs.
- **`_interpolate_scalar` re-ran five `to_decimal` conversions per site.**
  None of the five depend on the site, and `interpolate_branch` calls it once
  per requested `branch_x`. The decode moves into a `_BranchGeometryUm` object
  built once per branch; the per-site arithmetic stays in `at()`. The flat
  output views are also hoisted out of the loop rather than re-derived per
  index.

Two candidates were **measured and not landed**, recorded so they are not
re-proposed:

- **`Morphology.edges`** — the reuse fix (iterate `self.branches` instead of
  redoing N dict lookups) landed for clarity, but at 5.70 → 5.61 ms over ten
  accesses it is not a performance change and is not claimed as one.
- **`_path_node_ids` walks and `MorphoBranch.__getattr__` dispatch** were both
  profiled after being flagged. Neither is hot enough on a real reconstruction
  to justify restructuring; `__getattr__` in particular only fires for the five
  dynamic attributes, since normal lookup handles everything else.

## Deliberately not changed

Recorded so the next reader does not re-litigate them, and so later iterations
of this sweep know what is waiting.

- **The public tree-distance API.** `MorphologySpatialGeometry` in the
  package's own `_spatial.py` computes exactly what the two deleted stubs
  promised, and `braincell/filter`, `braincell/network`, and
  `braincell/_discretization` all import that **private** module to get it.
  Making it public — either by backing methods on `Morphology` or by renaming
  `_spatial.py` to `spatial.py` and exporting it — is a public-surface
  decision affecting three other packages. **Deferred to iteration 14**, whose
  remit is exactly cross-module layering and public surface.
- **`morph/__init__.py` exports four names and not `Morphology`, `Branch`, or
  `Soma`.** Roughly 50 import sites across the repo — including
  `braincell/__init__.py` itself — therefore spell submodule paths like
  `braincell.morph.morphology`. This is already tracked as an open question in
  `docs/design/interface-map.md`. Exporting the names is cheap; normalising
  the call sites touches ~30 files. **Deferred to iteration 14.**
- **`braincell/vis/_traversal.py` owns the tree-walk primitive for
  `MorphoBranch`.** It imports `from braincell.morph import MorphoBranch` and
  contains nothing visualization-specific, so a renderer package owns a
  geometry-package primitive. Moving it to `morph/` costs a module move, a
  test move, and two import updates. It is a cross-package relocation, the
  same shape as PR #136, so it belongs with the layering pass.
  **Deferred to iteration 14.** The `topo()` crash it would have helped with
  is fixed independently above, since the existing helper does not carry the
  prefix bookkeeping `_format_topology` needs anyway.
- **`Morphology._swc_type_map()` imports `braincell.io.swc.types` from inside
  a method** to rank branch types, so the canonical ordering of a morphology
  concept is defined by a file-format reader's integer table, reached through
  the package's only relative import and past two `__init__.py` surfaces.
  `morph/branch.py` already owns the type vocabulary twice
  (`_ALLOWED_BRANCH_TYPES`, `_BRANCH_TYPE_TO_CLASS`). Untangling this means
  deciding which of those three is canonical, which is a design decision about
  `branch.py`'s public vocabulary rather than a refactor. **Deferred to
  iteration 14.**
- **`filter/cache.py` reads `Morphology._revision`** via
  `getattr(morpho, "_revision", None)`, and `morphology.py` documents the
  coupling from its side while publishing it only as a private attribute. The
  one-line fix is a public `revision` property, but it is an addition to the
  public surface and belongs with the item above. **Deferred to iteration 14.**
- **`show=True` has no observable effect in `vis3d`.** The docstring says the
  backend's show method is called; the code calls `matplotlib.pyplot.show()`,
  and the 3-D backend is PyVista, which never creates a matplotlib figure.
  Whether to correct the documentation or make the parameter real is a
  behaviour decision about `braincell/vis`, not a `morph` refactor.
- **`MorphoBranch.parent_id` vs the filter's `parent_id`.** They deliberately
  mean different things (parent node id vs parent branch index); PR #137
  examined and kept the distinction.
- **The `Morpho` → `Morphology` rename stops at this package.** 28 references
  inside `braincell/morph/` are fixed here, plus the 11 in
  `examples/multi_compartment/morphology-checkpoint.ipynb`, which had to move
  in this PR because one of them is a stored `__repr__` output this change
  invalidates. The remaining stale spellings are *not* callers of anything
  changed here — they are independent error strings in other packages — so
  they travel with their own iterations: 16 in `braincell/filter/`
  (**iteration 3**, next), and 7 in `braincell/io/` plus 9 across
  `braincell/vis/` (**iteration 14**). The two tests that assert on those
  strings (`io/checkpoint_test.py:318`, `vis/layout/_dispatch_test.py:45`) use
  substring matching, so they stay green either way.
- **`morphology_test.py` imports `FakeBackend` from `braincell.vis._testing`.**
  Unlike `make_deep_chain_tree`, that double is genuinely a render-backend
  object and belongs in `vis`. The morph tests needing it at all is a symptom
  of `Morphology.vis2d` living on the model class; that is a public-surface
  question. **Deferred to iteration 14.**

## Behaviour changes beyond the deletions

- **`interpolate_branch` on a zero-length branch now raises before the loop,
  not inside it.** `_BranchGeometryUm.__init__` performs the `total <= 0`
  check once per branch, so `interpolate_branch(morpho, i, [])` — an empty
  site list on a degenerate branch — raises `ValueError` where it previously
  returned empty arrays. Raising is the more defensible answer for a branch
  that cannot be interpolated at any site, and no caller passes an empty site
  list.
- **Error-message wording.** `"reserved by the Morpho API"` becomes
  `"reserved by the Morphology API"`, and `_parse_attachment_key` now reports
  an out-of-range integer the same way `attach(parent_x=...)` does, since both
  go through the same check.

## Verification

```
$ pytest braincell/morph -q
82 passed, 1 warning, 18 subtests passed in 4.56s

$ pytest braincell/ -q
2718 passed, 15 skipped, 411 warnings, 389 subtests passed in 180.57s (0:03:00)

$ pre-commit run --files <the 9 changed files>
... all hooks Passed
```

The suite **gained** tests rather than losing any, and the arithmetic closes
exactly against the `main` @ `4c7c572` baseline of 2,713 passed / 338 subtests:

| | Baseline | This branch | Delta |
|---|---:|---:|---:|
| tests passed | 2,713 | 2,718 | **+5** |
| subtests passed | 338 | 389 | **+51** |
| skipped | 15 | 15 | 0 |
| failed | 0 | 0 | 0 |

Every one of those deltas is accounted for by the two new regression classes,
with nothing else moving:

| New test | Tests | Subtests |
|---|---:|---:|
| `MorphoReservedNameTest.test_names_that_used_to_slip_through_are_rejected` | 1 | 2 |
| `MorphoReservedNameTest.test_dynamic_morpho_branch_attributes_are_reserved` | 1 | 5 |
| `MorphoReservedNameTest.test_every_public_attribute_of_both_classes_is_reserved` | 1 | 44 |
| `MorphoDeepTreeTest.test_topo_formats_a_1200_branch_chain` | 1 | 0 |
| `MorphoDeepTreeTest.test_deep_chain_aggregates_do_not_recurse` | 1 | 0 |
| **total** | **5** | **51** |

Both bugs were reproduced on `main` @ `4c7c572` before the fix and re-checked
after, using the identical script against each tree:

```
baseline pkg: /mnt/d/codes/projects/braincell/braincell/__init__.py
BASELINE topo: RecursionError -> maximum recursion depth exceeded
BASELINE accepted: topo -> shadows method
BASELINE accepted: from_neuromorpho -> shadows method

after pkg: .../worktrees/simplify-morph/braincell/__init__.py
topo type: str len: 2887285   lines: 1199
rejected: topo             -> Branch name 'topo' is reserved by the Morphology API.
rejected: from_neuromorpho -> Branch name 'from_neuromorpho' is reserved by the Morphology API.
rejected: branch           -> Branch name 'branch' is reserved by the Morphology API.
rejected: name             -> Branch name 'name' is reserved by the Morphology API.
rejected: parent_x         -> Branch name 'parent_x' is reserved by the Morphology API.
```

The last three lines are the near-miss: deriving the reserved set from `dir()`
alone would have left them accepted, because `MorphoBranch.__getattr__` serves
them dynamically.
