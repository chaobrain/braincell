# Quality cleanup of `braincell.filter`

Reuse, simplification, efficiency, and altitude cleanup of the region/locset
selection package (`braincell/filter`: `helper.py` 725 lines, `locset.py` 699,
`_sampling.py` 468, `region.py` 238, `density.py` 187, `cache.py` 121,
`metric.py` 56, `__init__.py` 94, plus 1,692 lines of co-located tests).
Breaking API changes are in scope and were explicitly authorised for this pass.

This is iteration 3 of a module-by-module sweep, after `quad` (#138) and
`morph` (#139).

## Scope and method

Four independent reviews swept the package, one per angle (reuse,
simplification, efficiency, altitude). Findings were deduplicated; where three
or four reviews independently converged on the same defect it is treated as
load-bearing. Every claim acted on below was re-verified by running it, not by
reading the report.

Three invariants govern every edit, carried over from the previous iterations:

1. **The test suite stays green.** Baseline on `main` @ `5d79fac` before any
   change: `braincell/filter` 126 passed, 17 subtests (8.71 s); whole package
   2,718 passed, 15 skipped, 389 subtests.
2. **Every performance claim carries a measurement**, taken by this pass rather
   than inherited from a report.
3. **Where two code paths disagree, the disagreement is preserved, not
   silently unified.** Divergences worth a decision are listed under
   *Deliberately not changed*.

### The six `NotImplementedError` stubs are kept

`RadiusRangeRegion`, `TreeDistanceRegion`, `EuclideanDistanceRegion`,
`SubtreeRegion`, `RegionAnchors`, and `StepSamples` are exported public classes
whose entire body is `raise NotImplementedError`. Every reference in the
repository is their own definition, export plumbing, a test asserting they
raise, or documentation stating that they do not work — there is no caller.

They are nonetheless a **deliberate, maintained reservation**, not an
oversight: `docs/tutorials/filter.ipynb` says verbatim "Exported but not
implemented yet", `docs/concepts/regions_locsets.ipynb` gives them table rows,
and `docs/design/interface-map.md` lists them. Deleting an advertised roadmap
is a product decision rather than a simplification, so **this pass leaves all
six classes, their exports, and all four documentation surfaces untouched.**

What does go is the dead state they accreted: see *`SelectionCache` sheds three
fields nothing writes*, below.

## Bugs fixed

### Two entry points into `LocsetMask` round coordinates by different rules

`LocsetMask.__init__` normalizes through `helper.normalize_locset_points`,
while `LocsetMask.from_columns` normalizes through
`locset._normalize_location_columns`. Both enforce the same four rules, but
they derive the rounding precision differently — verified by running them:

```
helper rounding: digits = _round_digits_from_epsilon(epsilon)
                 normalized.append((branch, round(x, digits)))
locset rounding: np.round(np.clip(normalized_x, 0.0, 1.0), decimals=12)
_round_digits_from_epsilon(EPSILON) = 12
```

The two agree today only because `EPSILON` happens to be `1e-12`. Change
`helper.EPSILON` and the row path re-rounds while the column path does not —
and `LocsetMask.__eq__` / `__hash__` are built on exactly those rounded rows,
so the same locset built two ways would stop comparing equal. The column path
now derives its precision from the same constant.

### `EmptyRegion` accepts arguments every sibling rejects

Fourteen of the fifteen concrete `evaluate` implementations guard their
argument; `EmptyRegion.evaluate` does not. Verified:

```
EmptyRegion().evaluate("not a morphology")  -> ACCEPTED (no guard)
AllRegion().evaluate("not a morphology")    -> TypeError: AllRegion expects Morpho, got str.
```

The guard consolidation below closes this as a side effect, which is the point
of fixing it at the base class rather than adding a fifteenth copy.

## Breaking changes

### `SelectionCache` sheds three fields nothing writes

`tree_distance_to_root`, `euclidean_distance_to_root`, and
`branch_radius_summary` are public dataclass fields with **no write site
anywhere** in `braincell/`, `examples/`, or `docs/` — the class's own docstring
says they are "reserved ... nothing populates them yet". They are the reason
`SelectionCache` cannot be frozen, and they carry a docstring paragraph plus
three tests that only assert empty dicts are empty.

Whoever implements one of the reserved region types can add the field it
actually needs; reserving three guesses in advance has already aged badly, in
that the concept `tree_distance_to_root` names is a per-branch, lower-fidelity
re-declaration of what `MorphologySpatialGeometry` already computes per node.

### `helper.branch_points_locations` is deleted

A one-line `return fork_points_locations(morpho, epsilon=epsilon)` described as
a compatibility alias, with zero callers in `braincell/`, `examples/`, or
`docs/`. The alias users actually see is `BranchPoints = ForkPoints` in
`locset.py`, and `ForkPoints.evaluate` calls `fork_points_locations` directly.

It is kept alive solely by `helper_test.py`, whose `__all__` allowlist names
it — a test asserting that a dead function is still exported. The function, its
`__all__` entry, and the allowlist string all go.

### `evaluate` becomes a template method on both expression base classes

`RegionExpr.evaluate` and `LocsetExpr.evaluate` become concrete: they perform
the morphology type check once and delegate to a new abstract
`_evaluate(morpho, cache)`. The public calling convention is unchanged —
`expr.evaluate(morpho, cache)` still works for all eight external call sites in
`_discretization`, `_multi_compartment`, `network`, and `vis`.

**What breaks:** anything subclassing `RegionExpr` / `LocsetExpr` and
overriding `evaluate` must rename its override to `_evaluate`. There are no
such subclasses outside the package; inside it, the three test doubles in
`cache_test.py` are updated in this PR.

### `LocsetMask` and `LocsetBatch` are no longer `is_dataclass()`

Both carried `@dataclass(init=False, eq=False, repr=False)` over zero annotated
fields, so the decorator generated no method — `__init__`, `__eq__`, and
`__repr__` are all hand-written, and `dataclasses.fields()` returns `()`. Its
sole effect was to set `__dataclass_fields__ = {}`, which flips
`dataclasses.is_dataclass()` to `True`.

**What breaks:** `is_dataclass(mask)` now returns `False`, as does
`is_dataclass(LocsetMask)`. `dataclasses.fields(mask)` and `asdict(mask)` no
longer work at all — previously they "worked" by returning `()` and `{}`,
silently reporting a fully-populated mask as empty.

That silent-empty behaviour was load-bearing in one place, and wrongly:
`braincell/mech/_params.py:_to_hashable` branches on `is_dataclass(value)` and
builds its key from `fields(value)`, so **every** `LocsetMask` hashed to the
identical key `("LocsetMask", ())` regardless of its contents. Any two distinct
masks used as mechanism parameters would collide in that cache. After this
change they fall through to the identity-keyed default and no longer collide.
The two other `fields()`-driven introspection sites, `_compute/layouts.py` and
`_compute/table.py`, only ever see mechanisms, never masks.

One in-repo caller asserted the old behaviour:
`morph/morphology_test.py::test_morpho_select_accepts_locset_expr` opened with
`assertTrue(is_dataclass(selected))`. Its sibling region test two lines above
asserts only that `select` delegates to `evaluate`, so the assertion was an
inconsistent extra rather than a stated contract; it becomes
`assertIsInstance(selected, LocsetMask)`, which is what `Morphology.select`'s
`Returns` section actually promises.

### Error messages name `Morphology`, not `Morpho`

Fifteen guards said `"<ClassName> expects Morpho, got ..."`, naming a class
that does not exist. Consolidating them fixes the wording in one place and
derives the class name from `type(self).__name__` instead of hand-typing it.
The sixteenth reference is a `:meth:`Morpho.attach`` Sphinx role in
`cache.py` that resolves to nothing.

This completes for `braincell/filter` the rename that iteration 2 scoped to
`braincell/morph`; the remaining 16 occurrences in `braincell/io` and
`braincell/vis` travel with iteration 14. No test asserts on these strings.

## What was fixed

### Reuse — one home per concept

- **The morphology type-guard was copy-pasted 15 times**, each hand-typing its
  own class name into the message. Adding an expression type silently got no
  guard, which is exactly how `EmptyRegion` ended up without one.
- **The `branch(x)` display-name format was defined twice** in one file, and
  the copy whose docstring claims "the label format is defined here rather than
  at each call site" is no longer the only definition — nor the faster one, as
  it resolves each branch through `morpho.branch(index=...)` (which re-runs
  mutual-exclusion validation) while the other hoists the memoised
  `morpho.branches`. `resolved_display_names` now delegates.
- **The location normalizer existed twice** (see *Bugs fixed*).
- **`_density_array` and `_log_density_array`** were line-for-line identical
  apart from two message nouns and one extra non-negativity check.
- **`number` / `seed` validation was written twice**, in `helper.py` and
  `_sampling.py`, with byte-identical message strings.
- **`filter` had no `_testing.py`**, so the canonical soma literal was retyped
  in four test modules and the same two-branch tree in three. It gets one,
  re-exporting the `braincell/morph/_testing.py` builders — the sanctioned
  pattern, already used by `braincell/vis/_testing.py`.

### Simplification

- **Every set-op operand was normalized twice.** `RegionSetOp.evaluate` and
  `LocsetSetOp.evaluate` normalize each operand before handing it to
  `union_/intersect_/difference_*`, all of which normalize internally.
  Verified by execution that the outer pass is redundant for all six
  operations.
- **`LocsetConcatOp` re-normalized its accumulator once per operand**, making
  the fold O(k²m) over already-validated data where one final pass suffices.
- **Two `@dataclass(init=False, eq=False, repr=False)` decorators on classes
  with zero annotated fields.** Confirmed at runtime: `dataclasses.fields()`
  returns `()` for both and every dunder is hand-written. Removing them is a
  breaking change with a real bug behind it — see *Breaking changes* above.
- **A provably dead ternary** in `uniform_samples_from_region`: its `length_um`
  comes from entries that `_sample_entries` appends only when
  `length_um > epsilon`, so the `<= epsilon` arm cannot execute.
- **Dead validation in `density._field`**: both call sites are frozen classes
  constructed at exactly one site each, and both of those already apply the
  identical check.
- **`metric.py`'s four accessors are one function copy-pasted four times**, and
  all six of its `getattr(context, "literal")` calls are plain attribute access
  written the long way.

### Documentation that described behaviour the code does not have

- `LocsetBatch`'s class docstring documents `branch_id` / `branch_x` /
  `display_names` as constructor parameters; verified at runtime that
  `LocsetBatch(branch_id=..., branch_x=...)` raises
  `TypeError: LocsetBatch() takes no arguments`. They belong on `from_columns`.
- `locset.py` documents a `morpho : braincell.morph.Morphology` that is not
  importable under that path — `braincell/morph/__init__.py` does not export
  it. `cache.py` already spells the resolvable `braincell.Morphology`.
- `density.py` uses a bare `` :class:`SamplingContext` `` role in a module that
  never imports the name, so Sphinx cannot resolve it.

### Test layout

`density.py` had no sibling `density_test.py`; its coverage was scattered
across `_sampling_test.py` and `metric_test.py`, against AGENTS.md rule 10.
It gets one.

## Efficiency — measured, or not landed

Benchmarked on `data/morphology/CA1.swc` (681 branches, 38,150 selected
components), median of 5 runs, same machine, before/after trees differing only
by this branch. Every row prints a digest of its result; **all five digests are
identical before and after**, which is the correctness evidence for the
speedups.

| Benchmark | Before | After | Change | Digest |
|---|---:|---:|---|---:|
| `branch_in`, 8 candidates | 31.54 ms | 9.44 ms | **3.3× faster** | 8 |
| `branch_range` | 15.84 ms | 9.37 ms | **1.7× faster** | 436 |
| `sample(density=None)`, 2k | 303.50 ms | 292.88 ms | see below | 2000 |
| `uniform_samples`, 10k | 46.90 ms | 45.80 ms | unchanged (noise) | 10000 |
| `fork_points_locations` | 5.46 ms | 6.03 ms | unchanged (noise) | 337 |

**What made the first two faster.** `helper._matches_in` and `_matches_range`
called `normalize_param` on the *candidate set* (resp. the low/high bounds)
once per branch, re-doing identical unit conversion 681 times for values that
never change across the loop. Both now memoise per unit, in a dict keyed by the
`brainunit` `Unit` — verified first that `Unit` is hashable and equality-stable.
Per-unit rather than hoisted-once, so a mixed-unit candidate list stays correct.

**The third row needs an honest reading.** The end-to-end delta above is 10.6 ms,
but a second run of the identical benchmark measured 21 ms — the two disagree
because both are noise-level against a ~300 ms total dominated by
`_build_components`, which this pass deliberately leaves alone. Timing the
skipped work in isolation settles it:

```
MorphologySpatialGeometry.build :   14.13 ms
_branch_identities              :    6.90 ms
total skipped when density=None :   21.02 ms
```

So the saving is a real and reproducible **21.02 ms per density-free
`sample()`** — a multi-source Dijkstra plus a per-component identity scan, both
built unconditionally and then used only on the `density is not None` path. The
end-to-end row understates it; neither end-to-end number should be quoted as
the result.

**Two rows are reported as unchanged, not as wins.** Neither
`uniform_samples_from_region` nor `fork_points_locations` was targeted for
speed, and `fork_points` measured marginally *slower*. Both deltas are within
run-to-run variation and no claim is made in either direction.

## Deliberately not changed

Recorded so the next reader does not re-litigate them, and so later iterations
know what is waiting.

- **The six `NotImplementedError` stubs** — see *Scope and method*.
- **Vectorising `_build_components`.** It emits 38,150 components for 681
  branches, and for `measure in {"normalized", "length"}` with `density=None`
  the jacobian is constant per branch, so the per-segment split buys nothing on
  that path. A vectorised draw measured far faster in review. It is not landed
  here because it is a redesign of the sampling core, and `_sampling_test.py`
  pins exact RNG draw sequences — the risk profile belongs in a change whose
  own spec is about sampling, not in a cleanup pass.
- **Deleting the log-density fast path.** `_log_density_array`, the
  `log_shift` branch, and `_builtin_log_shift` (~45 lines, plus a
  `# type: ignore` reach-in to `density.py`'s privates) return `0.0` for both
  non-deprecated builtin densities and matter only for `density.exponential`,
  which is deprecated. Removing it is a behaviour decision about a deprecation
  timeline, not a refactor.
- **`Morphology.revision` as a public property.** `cache.py` reads
  `getattr(morpho, "_revision", None)` — another package's private, through a
  default that silently degrades to "never invalidate" for a non-`Morphology`.
  `morph` knows about the coupling and documents it from its side, but
  publishes it privately. Adding the property touches `braincell/morph`, and
  iteration 2 already deferred the neighbouring public-surface questions.
  **Deferred to iteration 14.**
- **Memoising `MorphologySpatialGeometry` on `Morphology`.** It is rebuilt from
  scratch at three call sites across the package (`_sampling`,
  `_discretization/context`, `network/pairing`) for a tree that cannot change
  between attaches. The natural fix is a sixth entry in `Morphology`'s existing
  `_invalidate_derived_caches` set, keyed on the revision above — so it is
  blocked on the same deferral. **Deferred to iteration 14.**
- **Publishing a per-branch segment table from `morph`.** `_build_components`
  decodes branch geometry to µm arrays in a spelling that already exists twice
  in `braincell/morph` (`Branch._segment_arrays_um`,
  `_spatial._BranchGeometryUm`). The filter copy is not gratuitous — it is
  vectorised within a segment because `scipy` calls it thousands of times,
  which the `morph` versions are not. The fix is to push a vectorised segment
  table *down* into `morph`, which is cross-package. **Deferred to iteration
  14.**
- **`RegionMask` carries no normalization invariant**, so three consumers
  outside the package re-normalize and re-group by hand — and two of them
  disagree about whether to normalize at all
  (`_multi_compartment/cell.py` does, `_discretization/mechanism.py` does not).
  Establishing the invariant in `RegionMask` is right, but changing what
  `mechanism.py` sees is a behaviour change in another package. **Deferred to
  iteration 14.**
- **`helper.py` should be `_helper.py`, and `_sampling.py` should not be
  underscored** given `__init__.py` exports `SamplingContext` from it. The
  rename is blocked on `_multi_compartment/cell.py`, which deep-imports
  `braincell.filter.helper`. **Deferred to iteration 14.**
- **`braincell.filter` is missing from `braincell/__init__.py`'s `__all__`**
  although it is imported there and AGENTS.md names it a public domain package.
  A one-line fix in a file that belongs to iteration 13.
- **`CVContext` spells `branch_x` / `radius` as `midpoint` / `radius_mid`**,
  which is the entire reason `metric.branch_x` and `metric.radius` carry
  fallback chains. The fix is an alias property in `braincell/mech/_context.py`
  — **iteration 4's territory**, not this one.
- **`epsilon=` is threaded through 15 helper functions and never varied**; no
  call site anywhere passes a non-default. Collapsing it would delete ~20
  parameters, but `helper.py` is internal plumbing whose signature churn is
  better bundled with the `_helper.py` rename above.

## Verification

```
$ pytest braincell/filter -q
138 passed, 1 warning, 25 subtests passed in 6.24s

$ pytest braincell/ -q
2730 passed, 15 skipped, 411 warnings, 397 subtests passed in 175.64s

$ pre-commit run --files <17 changed files>
... ruff (legacy alias) .... Passed
... ruff format .......... Passed          (all hooks Passed or Skipped)
```

The counts reconcile exactly against the baseline `main` left at after
iteration 2 (2,718 passed / 389 subtests):

- `braincell/filter` goes 126 → 138 passed. That is **+16** from the new
  `density_test.py`, **−3** from the deleted `SelectionCacheFieldsTest`, and
  **−1** for a test moved out of `metric_test.py` into `density_test.py`.
  Package total 2,718 + 12 = **2,730**. ✔
- Subtests go 17 → 25 in `filter`, so 389 + 8 = **397**. ✔

**One failure was found and fixed during verification, not before it.** The
first full-suite run came back `1 failed, 2729 passed` on
`morph/morphology_test.py::MorphoSelectAndVisTest::test_morpho_select_accepts_locset_expr`
— the `filter`-only run had been green, so the regression was only visible
package-wide. It was genuinely caused by this branch (the dropped `@dataclass`
decorators) and is resolved in the *Breaking changes* section above rather than
by reverting; investigating it is what surfaced the `_to_hashable` key-collapse
bug that decorator was hiding.
