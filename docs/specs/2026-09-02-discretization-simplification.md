# `braincell/_discretization` simplification

Iteration 9 of the module-by-module simplification sweep. Written before any
code changed; the Verification section at the end is filled in afterwards.

## Baseline

`braincell/_discretization` is the static, declaration-time layer between
`braincell.morph` (the geometric tree) and `braincell._compute` (the runtime
lowering layer). It resolves a CV policy into branch-wise bounds, assembles the
physical CV facts, attaches normalized mechanism declarations, and derives the
point-space node tree.

| module | lines | role |
| --- | --- | --- |
| `geometry.py` | 736 | frusta, CV physical facts, morphology/bounds validation |
| `mechanism.py` | 702 | paint/place rule normalization and lowering onto CVs |
| `base.py` | 694 | the frozen record types and `build_discretization` |
| `policy.py` | 520 | the five CV policies |
| `node_build.py` | 453 | CV tree to node tree |
| `context.py` | 111 | one `CVContext` per CV |
| `__init__.py` | 75 | export surface |
| `_testing.py` | 73 | shared fixtures |

Tests: `pytest braincell/_discretization -q` -> **117 passed in 12.33s**.

`build_discretization` cost, min of 5 runs, on a soma with N tapering basal
dendrites:

| branches | CVs | nodes | build |
| --- | --- | --- | --- |
| 5 | 10 | 16 | 2.5 ms |
| 41 | 328 | 370 | 72.9 ms |
| 121 | 3025 | 3147 | 670.8 ms |

Roughly linear at ~0.22 ms per CV. For scale, `Cell.init_state()` on the same
41-branch morphology measured 150.7 ms in iteration 8, so this layer is a large
share of cell construction rather than a rounding error.

## Findings

Every claim below was re-verified independently of the review that raised it.
"Verified by running" means a script executed against this worktree;
"verified by reading" means source comparison or a control-flow argument.

### 1. `locate_node_on_branch` is a dead subsystem, and it is the only reason a per-branch array is built

`git grep locate_node_on_branch` over **all tracked files** returns its own
`def` (`node_build.py:288`), its `__all__` entry, `docs/design/TODO.md`, and three lines
of `docs/design/module-dependency-map.md`. No code caller, no test.

It is also the only reader of `NodeTree.branch_endpoint_node_id`: that field is
produced at `node_build.py:211`, stored at `:284`, and read only at `:316` and
`:324` — both inside `locate_node_on_branch`. So an `(n_branch, 2)` int32 array
is built and carried on every discretization for a function nothing calls.

`branch_endpoint_node_id_by_x` is a *different*, live thing —
`_resolve_attachment_node` reads it during the build. It stays.

### 2. One CV-locating algorithm under three names

| site | form |
| --- | --- |
| `base.py:498` | `locate_cv_on_branch(ids, cvs, *, x, epsilon=_EPS_PARAM)` — imported by `network/lowering.py`, `network/event.py`, `_multi_compartment/{field_resolution,selection}.py` |
| `geometry.py:570` | a second `locate_cv_on_branch(ids, geos, *, x)` — no empty-`ids` guard, hardcoded epsilon, plus a trailing fallback loop |
| `node_build.py:397` | `_locate_branch_cv_by_x` — 27 lines whose body is `return locate_cv_on_branch(ids, cvs, x=x, epsilon=epsilon)` |

Both real implementations read only `.prox`/`.dist`, so the `_GeoCV`/`CV` split
is not a reason for two of them.

Verified by running: 16,065 probes across five policies — a 401-point uniform
sweep per branch plus every exact CV boundary at ±1e-10, ±1e-9 and ±1e-8 —
gave **0 disagreements**, neither implementation ever raised, and the
`geometry` fallback loop **never fired**. The review reproduced this
independently at 506,285 probes over random non-uniform tilings, also with 0
disagreements and 0 fallback hits, and instrumented the whole
`_discretization`/`_multi_compartment`/`network` suite: 402 calls, 1 reaching
the fallback, from `geometry_test.py:312`, which deliberately passes a
non-tiling bound — where both versions raise `ValueError` anyway.

That fallback is unreachable by construction: `validate_bounds`
(`geometry.py:497`) guarantees each branch tiling is contiguous, starts at 0.0,
ends at 1.0, and has every CV wider than `EPS_PARAM`, under which the first
loop always returns.

### 3. Two hand-written copies of formulas `morph/branch.py` declares as the single definition

`braincell/morph/branch.py:40-46` carries an explicit comment: these frustum
functions "are the single definition of that geometry ... Keeping the formulas
here is what makes `Morphology.total_area == sum(b.branch.area)` a consequence
rather than a coincidence that two hand-written copies have to maintain."

`_discretization/geometry.py` is that second copy, twice:

- `_lateral_area_um2` (`:330`) sums `pi * (r0 + r1) * sqrt(L^2 + (r1 - r0)^2)`
  in a Python loop — the body of `frustum_areas_um2` (`branch.py:49`).
- `_arc_weighted_mean_diam_um` (`:371`) computes `sum((r0 + r1) * L) / sum(L)`,
  which is `2 *` `length_weighted_mean_radius_um` (`branch.py:96`).

Verified by reading (formula identity) and by running: over 76 CVs spanning
four policies on a morphology with a taper, a 3-D-point branch and a
zero-length jump segment, **max relative error 0.000e+00** for both.

This is the highest-consequence duplication in the package: `_lateral_area_um2`
decides membrane area, so a divergence silently changes every conductance, and
`diam_arc_mean` feeds the calcium shell models.

### 4. `NodeEdgeRole.r_axial` is written for every edge role and read nowhere

`git grep -E '\.r_axial\b'` over all tracked files: the only hits are
`CV.r_axial` (a documented public field, printed in
`docs/tutorials/cell.ipynb:303` — **live, keep**) and
`cv.r_axial_prox`/`cv.r_axial_dist` in `quad/_staggered.py:561`. There is no
reader of `role.r_axial` anywhere, and no `asdict`/`fields` reflection that
could reach it.

`_role_axial_resistance` (`node_build.py:442`) allocates a `u.Quantity` per edge
role at every discretization to populate it. Its body is, verbatim, the select
that `_staggered.py:561` performs anyway:

```python
resistance = cv.r_axial_prox if role.half == "prox" else cv.r_axial_dist
```

Its trailing `raise ValueError(f"Unsupported half {half!r}.")` is unreachable:
`half` reaches it only from `_entry_half_for_walk`/`_exit_half_for_walk`, which
return `"prox"` or `"dist"` and nothing else.

### 5. `Discretization.locate_cv` has no caller

`git grep '\.locate_cv('` over all tracked files finds exactly one call,
`geometry.locate_cv(...)` at `mechanism.py:684` — that is
`CVGeometryResult.locate_cv`, a different class. The `Discretization` method has
no caller, no test, and no doc reference. Every consumer that needs this
imports `locate_cv_on_branch` from `base` directly.

### 6. The package default cable property is defined twice

`policy._DEFAULT_D_LAMBDA_CABLE` (`:39`) and `mechanism._DEFAULT_CABLE` (`:66`)
are distinct objects that compare **equal** (verified by running). `DLambda`
sizes CVs from one while lowering applies the other, so editing one makes the
discretization silently no longer match the electrics it was sized for, with no
test to catch it.

### 7. Four copies of the tolerance constants

`EPS_PARAM`/`_EPS_PARAM = 1e-9` is declared in `geometry.py:32`, `policy.py:31`,
`base.py:71` and `node_build.py:47`; `EPS_LEN_UM = 1e-6` in `geometry.py:33` and
`policy.py:32`. All copies compare equal (verified by running). `mechanism.py`
already does the right thing and imports `EPS_PARAM` from `geometry`.

### 8. `context.py` pays for a second input shape that never arrives

`build_cv_contexts`' docstring promises "``_GeoCV`` records **or** finalized
public ``CV`` records", and pays for it with `getattr(source, "midpoint", ...)`
and a `_source_quantity(source, quantity_name, scalar_name, unit)` shim called
six times per CV. Both call sites (`base.py:586`, `mechanism.py:581`) pass
`geometry.geos`, and `git grep build_cv_contexts` finds no others.

`_GeoCV` declares `midpoint` and the scalar spellings (`length_um`,
`lateral_area_um2`, `r_mid_um`, ...) and **none** of the `Quantity` spellings
(`length`, `area`, `radius_mid`, ...). So `_source_quantity`'s first branch is
statically unreachable, and the `midpoint` default — `0.5 * (prox + dist)` — is
an eagerly evaluated default argument, computed once per CV and always
discarded.

The `contexts[i] is None` check at `:101-103` is likewise unreachable: the loop
runs `len(cvs)` times and each iteration either raises or writes a distinct
in-range index, so N distinct indices into N slots leave none empty.

### 9. The CV midpoint radius is computed twice, and one answer is thrown away

`context.py:76` calls `interpolate_branch(morpho, branch_id, midpoint)` and
discards its first return value with `_`, then reads the same number out of
`_GeoCV.r_mid_um`, which `geometry._midpoint_radius_um` computed by walking the
frusta. Verified by running: over 100 CVs across four policies the two agree to
**8.9e-16 um**.

### 10. `_bounds_from_d_lambda` re-inlines `_uniform_bounds_for_count`

`policy.py:431` and `policy.py:399` are byte-identical (verified by
`inspect.getsource`), and outputs match for `n_cv = 1..200`. The sibling
`_bounds_from_max_len_um` calls the helper correctly — this is the one call site
of two that forgot.

### 11. Four one-line helpers in `node_build` that are one boolean

`_entry_half_for_walk`/`_entry_position_for_walk` have byte-identical bodies
differing only in return annotation, as do `_exit_half_for_walk`/
`_exit_position_for_walk`; and `_exit_*` is the exact complement of `_entry_*`.
Verified by running over `{0.0, -0.0, 1e-12, 1e-9, 1e-9+1e-15, 0.25, 0.5, 0.75,
1.0}`: all three identities hold on every probe. Both `_half_` variants are
re-called inside the per-CV loop (`:199`, `:205`) with an `attach_x` that is
fixed for the whole branch.

### 12. `__all__` entries that publish private names

`geometry.__all__` lists `_Frustum`, `_GeoCV`, `_build_frusta` and
`_lateral_area_um2`; `node_build.__all__` lists `_EPS_PARAM` and
`_locate_branch_cv_by_x` (verified by running). `__all__` is the declared public
surface, so an underscore entry contradicts itself and makes `import *`
re-export internals. The symbols are genuinely used by siblings and tests — the
entries go, not the symbols.

### 13. `_GeoCV.midpoint` is always derivable

Verified by running: `midpoint == 0.5 * (prox + dist)` for **100/100** CVs
across four policies, 0 violations. As a stored field it must be hand-copied in
the 17-field rebuild at `geometry.py:723` and in the test literal at
`geometry_test.py:290-308`.

### 14. `build_cv_geometry` retypes all 17 `_GeoCV` fields to change two

`geometry.py:710-731` is a 22-line generator that rebuilds every field by hand
solely to fill `parent_cv` and `children_cv`. Any field added to `_GeoCV` later
must be added here too or it is silently dropped.

## The bug this iteration found

Finding 8 above led to a live correctness defect, so it is written up on its
own. `_RegionCache.points` keyed its memo on `id(locset)` while retaining no
reference to the key, and `build_cv_mechanisms` (`mechanism.py:668`) feeds it
freshly built `LocsetBatch` members from inside a generator expression:

```python
location_groups = zip(
    (cache.points(rule.locset[index]) for index in range(len(rule.locset))),
    ((int(index),) for index in rule.population_indices),
)
```

Each member is unreferenced the moment `points` returns, CPython reuses the
freed address for the next one, and the cache hands member *n* member *n-1*'s
locations. Reproduced end to end through the public API — four population
members, each declared on its own branch and coordinate:

```
expected: {0: (0, 0.1), 1: (1, 0.9), 2: (0, 0.2), 3: (1, 0.8)}
actual  : {0: (0, 0.1), 1: (1, 0.9), 2: (1, 0.9), 3: (1, 0.9)}
```

Two of four members get their synapse on the wrong branch and the wrong CV,
silently. The existing `LocsetBatch` coverage never caught it because it
drives `.loc()`, which does not go through this cache, and because the two
`_RegionCache` tests hold their locset in a local — which keeps the address
unique and hides the aliasing.

Per AGENTS.md rule 4 the regression tests were written and confirmed failing
first: one unit test in `mechanism_test.py`, and two end-to-end tests in
`_multi_compartment/cell_test.py` (the second failed as `{0: 3, 1: 4, 2: 4}`,
two members sharing a CV).

`_RegionCache.intervals` carried the same `id()` pattern. It is safe today
only because `paint_rules` happens to hold its regions alive for the whole
build — an invariant nothing states or enforces. Both are now value-keyed.

## Changes

Each numbered item refers to the finding above.

**Correctness**

- `_RegionCache` keys both memos by expression **value**, via two
  `filter.SelectionCache` instances rather than a second memo
  implementation. `SelectionCache` already provides exactly this — value
  keying, a morphology-revision guard, and a fall-through for unhashable
  payloads. Its docstring is broadened to say that a consumer deriving
  something *from* a mask may memoize that too in its own instance.

**Duplication removed** (2, 3, 6, 7, 10, 11)

- One `locate_cv_on_branch`, in `base.py`. The `geometry.py` copy and the
  `node_build._locate_branch_cv_by_x` pass-through are gone; `geometry.py`
  imports the survivor. The removed copy's trailing fallback loop went with
  it: `validate_bounds` guarantees each branch tiling is contiguous, starts
  at 0.0, ends at 1.0, and has every CV wider than `EPS_PARAM`, under which
  the first loop always returns — measured at 0 hits across 522,350 probes
  and the whole test suite.
- `geometry._lateral_area_um2` and `_arc_weighted_mean_diam_um` now call
  `morph.branch.frustum_areas_um2` and `length_weighted_mean_radius_um`,
  the module that declares itself the single definition of frustum
  geometry. A new `_frustum_arrays` helper unpacks frusta into the arrays
  those take. Verified bit-identical: **max relative error 0.000e+00** over
  45 CVs across four policies, on a morphology with a taper, a
  multi-segment branch and a zero-length radius jump.
- One `DEFAULT_CABLE`, in `base.py`. `DLambda` sized CVs from
  `policy._DEFAULT_D_LAMBDA_CABLE` while lowering applied
  `mechanism._DEFAULT_CABLE`; they were equal, and nothing tested that.
- One copy of `EPS_PARAM` / `EPS_LEN_UM` / `EPS_AREA_UM2`, in `base.py` —
  the only module in the package that imports no sibling at module scope.
  `geometry`, `policy`, `node_build` and `mechanism` import them.
- `_bounds_from_d_lambda` calls `_uniform_bounds_for_count` instead of
  re-inlining its body.
- The four one-line walk helpers in `node_build` collapse to one
  `walk_from_prox` boolean computed once per branch, next to the
  `ordered_cv_ids` reversal it already governed.

**Dead code removed** (1, 4, 5, 12)

- `locate_node_on_branch`, its private callee `_locate_node_id_on_branch`,
  `_build_branch_endpoint_node_id`, and `NodeTree.branch_endpoint_node_id`.
  Nothing called any of them; the field's only readers were inside the dead
  function, so an `(n_branch, 2)` int32 array was built on every
  discretization for a function with no caller. The live, differently named
  `branch_endpoint_node_id_by_x` stays.
- `NodeEdgeRole.r_axial` and `_role_axial_resistance`, which allocated a
  `u.Quantity` per edge role on every discretization. Nothing read the
  field; the one consumer that needs the value (`quad/_staggered.py:561`)
  performs the same `prox`/`dist` select on the CV itself.
- `Discretization.locate_cv`, a third spelling no caller picked.
- Private names removed from `geometry.__all__` and `node_build.__all__`.
  The symbols stay — they are used by siblings and tests — but `__all__` is
  the declared public surface and an underscore entry contradicts itself.

**Simplified** (8, 13, 14)

- `build_cv_contexts` takes `_GeoCV` only. Its docstring promised finalized
  `CV` records too and paid for the promise with a `_source_quantity` shim
  called six times per CV — but both call sites pass
  `CVGeometryResult.geos`, and `_GeoCV` declares only the scalar spellings,
  so the `CV` branch was statically unreachable. The `contexts[i] is None`
  check went too: the loop writes `len(cvs)` distinct in-range indices into
  `len(cvs)` slots, so none can be left empty. `u.um**2` is hoisted out of
  the per-CV loop.
- `_GeoCV.midpoint` becomes a property. It equalled `0.5 * (prox + dist)`
  for 100/100 CVs across four policies, and as a stored field it had to be
  hand-copied in the 17-field rebuild and in every test literal.
- `build_cv_geometry` finalizes with `dataclasses.replace(geo, parent_cv=…,
  children_cv=…)` instead of retyping all 17 fields to change two. A field
  added to `_GeoCV` later can no longer be silently dropped there.

**Tests**

- The locator tests move to `base_test.py`, the surviving definition's
  sibling, merged from the two files that tested the two deleted copies.
  Two cases are added: the half-open internal-boundary convention, and the
  empty-tiling guard.

## Breaking changes

All are internal to `braincell` and every in-repo caller is updated in this
PR. No deprecation shims or aliases were added; the old spellings are gone.

| Removed | Replacement |
| --- | --- |
| `_discretization.node_build.locate_node_on_branch` | none — no caller existed |
| `_discretization.base.Discretization.locate_cv` | `locate_cv_on_branch(cv_tree.branch_to_cv_ids[b], cvs, x=…)` |
| `_discretization.geometry.locate_cv_on_branch` | `_discretization.base.locate_cv_on_branch` |
| `_discretization.node_build._locate_branch_cv_by_x` | `_discretization.base.locate_cv_on_branch` |
| `NodeTree.branch_endpoint_node_id` | none — no reader outside the deleted function |
| `NodeEdgeRole.r_axial` | `cvs[role.cv_id].r_axial_prox` / `.r_axial_dist` |
| `_GeoCV.midpoint` as a constructor argument | derived; drop the argument |
| `geometry.EPS_PARAM`, `EPS_LEN_UM`, `EPS_AREA_UM2` | same names from `_discretization.base` |
| `policy._DEFAULT_D_LAMBDA_CABLE`, `mechanism._DEFAULT_CABLE` | `_discretization.base.DEFAULT_CABLE` |
| `node_build._EPS_PARAM` | `_discretization.base.EPS_PARAM` |
| `build_cv_contexts` accepting public `CV` records | pass `CVGeometryResult.geos` |

`docs/design/module-dependency-map.md` still lists `locate_node_on_branch`
in three places; those lines are updated in this PR.

## Considered and declined

**Finding 9 — the CV midpoint radius is computed twice.** The measurement
holds: `interpolate_branch`'s discarded first return and `_GeoCV.r_mid_um`
agree to **8.9e-16 um** over 100 CVs. But `r_mid_um` is the single stored
value that feeds *both* `CV.radius_mid` (in `base.py`) and
`CVContext.radius_mid` (in `context.py`). Switching `context.py` to the
interpolated value would leave those two public fields computed by two
different code paths — a worse duplication than the discard it removes. The
right fix is to pick one source for both, which is a `base.py` + `context.py`
change better made with iteration 11's view of `CV` construction.

**The claimed cache-hit speedup.** The review that raised the `_RegionCache`
keying also predicted "+12.3 ms per duplicated region at 3405 CVs". It does
not reproduce: **1160.1 ms → 1157.7 ms** at 4 `AllRegion()` paint rules on a
3003-CV morphology, inside the run-to-run noise. The reason is that
`_compute_intervals` evaluates through `self._selection`, which was already
value-keyed, so the outer `id()` cache only ever saved the cheap regrouping
loop over `mask.intervals`. The change is justified by the correctness bug
alone and is not claimed as an optimization.

## Deferred to later iterations of the sweep

- **Iteration 11 (`_multi_compartment`).** `field_resolution.locset_cv_ids`
  rebuilds a branch-to-CV map locally that `CVTree.branch_to_cv_ids` already
  holds. Also carried: the coverage-fraction divergence — `vis` highlights
  use length-based overlap while physics uses area-based, giving 0.500000 vs
  0.704545 on a tapering dendrite (they agree exactly with no taper).
- **Iteration 14 (whole package).** `morph._spatial` is imported across a
  package boundary by `context.py`.

## Verification

Run from the worktree with `PYTHONPATH=$PWD JAX_PLATFORMS=cpu`.

Regression tests written first, against unmodified code:

```
braincell/_discretization/mechanism_test.py    1 failed, 38 deselected
braincell/_multi_compartment/cell_test.py      2 failed, 85 deselected
  AssertionError: {0: (0, 0.1), 1: (1, 0.9), 2: (1, 0.9), 3: (1, 0.9)} != {0: (0, 0.1), 1: (1, ...
  AssertionError: 2 != 3 : {0: 3, 1: 4, 2: 4}
```

After the fix:

```
$ pytest braincell/_discretization -q
120 passed, 9 warnings in 8.42s

$ pytest braincell/ -q
2804 passed, 15 skipped, 408 warnings, 410 subtests passed in 233.84s (0:03:53)
```

Baseline was 117 and 2799 passed; the delta is the six added tests less the
duplicates merged when the locator tests were consolidated.

Numerical equivalence of the frustum-formula reuse, over 45 CVs across four
policies on a morphology with a taper, a multi-segment branch and a
zero-length radius jump:

```
max rel error, area     : 0.000e+00
max rel error, mean diam: 0.000e+00
```

Build cost is unchanged — this iteration removed duplication, not work:

| branches | CVs | before | after | per CV |
| --- | --- | --- | --- | --- |
| 5 | 10 | 2.5 ms | 2.8 ms | — |
| 41 | 328 / 361 | 72.9 ms | 80.3 ms | 0.2223 → 0.2224 ms |
| 121 | 3025 / 3003 | 670.8 ms | 677.7 ms | 0.2217 → 0.2257 ms |
