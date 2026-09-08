# `braincell/mech` simplification

Iteration 4 of the module-by-module simplification sweep. Predecessors:
`quad` (#138), `morph` (#139), `filter` (#140).

## Scope and method

`braincell/mech` is 4,017 lines across 19 files — 11 source modules and 8
co-located test modules. Four independent quality reviews were run over the
package (reuse, simplification, efficiency, altitude), their findings
deduplicated, and each surviving one re-verified **by execution** rather than
by reading, before anything was edited.

Two invariants govern this pass, carried forward from iteration 3:

1. Breaking changes are allowed, but every in-repo caller moves in the same
   commit, and no deprecation shim, alias, or `warnings.warn` bridge is left
   behind. The old spelling is deleted outright.
2. Every performance claim carries a measurement taken by this pass, not one
   inherited from a review report.

### The constraint that shaped this iteration

`braincell.mech` is a **leaf package**: it imports nothing from `braincell`.
This is stated in `mech/__init__.py` and verified here —
`grep -rn "^from braincell\|^import braincell" braincell/mech/*.py` returns
nothing for non-test sources.

That single fact vetoes a whole class of otherwise-attractive reuse:

- The `_typing.py` aliases (`Initializer`, `ArrayLike`, …) that AGENTS.md
  requires elsewhere **cannot** be used here — importing `braincell._typing`
  executes `braincell/__init__.py`, which imports `mech`. `mech` therefore
  annotates with `Any` where the rest of the package would use an alias, and
  that stays.
- `_misc.normalize_param`, `_misc.validate_time_quantity`, and
  `filter.helper._is_quantity` are equally unreachable, even though `mech`
  contains near-copies of all three.

`brainunit` is *not* `braincell`, so `u.get_mantissa` and friends remain fair
game — that distinction is what makes one of the reuse fixes below legal.

## Bugs fixed

Both were found while consolidating, both are reproduced by a new test first.

### `Channel(..., params={...})` silently discarded the caller's parameters

`Density.__init__` captures `**params`, so a caller writing the natural-looking
`Channel("IL", params={"g_max": ...})` did not set `g_max`. They created a
parameter *literally named* `"params"` whose value was the dict. Verified:

```
>>> Channel('IL', params={'g_max': 1.0*u.mS/u.cm**2}).params
{'params': {'g_max': Quantity(1., "mS / cm^2")}}
```

The failure is silent — no exception, and `g_max` quietly falls back to its
default at lowering time. `Synapse.__init__` already guarded exactly this
mistake with a clear `TypeError`; `Channel` and `Ion` did not. The guard moves
to `Density.__init__` so all three declaration families behave alike.

### `with_coverage` bypassed the validation `__init__` advertises

`Density.__init__` enforces `coverage_area_fraction ∈ [0, 1]`, but
`with_coverage` routed through `_replace`, which only does `float(...)`. The
invariant held on one construction path and not the other, so
`density.with_coverage(7.0)` produced an object `Density(...)` would have
rejected. `with_integration` is the counter-example that got it right — it
re-runs its normalizer before `_replace`. Normalization now lives in one place
that both paths go through.

## Breaking changes

### Zero-caller API deleted

Each was confirmed to have no caller in `braincell/`, `examples/`, or `docs/`
outside its own test:

| Symbol | Note |
|---|---|
| `Density.identity`, `Synapse.identity` | see below |
| `Density.with_params`, `Density.with_name` | `with_coverage` is live and stays |
| `Density.with_integration` | |
| `Params.without` | `Params.with_updates` is live and stays |
| `CableProperty.with_updates` | |
| `MechanismRegistry.add_alias` | `unregister` / `contains` are live and stay |
| `CurrentClamp.target_index` + `_normalize_target_index` | field documented "Reserved"; no reader |
| `mech.Synapse` (the deprecated alias) | its 2 callers in `network/_testing.py` move to the then-canonical `SynapseSpec` |

> **Later note.** The last row reads backwards today. `SynapseSpec` was itself
> renamed to `Synapse` shortly after this refactor landed, reclaiming the name this
> spec had just freed — see
> [`2026-09-02-mech-synapse-rename.md`](2026-09-02-mech-synapse-rename.md). The row
> above is left in its original terms because it records what was true at the time.

`identity` earns a specific note. Both copies documented themselves as
returning `(instance_name, class_name)` "for table views", but the actual table
view — `_compute/table.py:194` — builds `(class_name, instance_name)`, the
opposite order, inline, and never calls `identity`. The property was not merely
unused; its docstring pointed at a real consumer that had silently disagreed
with it. `instance_name` has 30+ production readers and stays.

### One exception type for "not a non-empty string"

The same condition was validated 11 times across `_density.py` and `_point.py`
with **three different exception types** — `ValueError` in the probes,
`TypeError` in `Density.name` / `Channel.ion_name` / `Synapse.name`,
`ValueError` in `Synapse.synapse_type`. A caller writing `except` had no
way to be right.

One helper now owns the check and picks by cause, following the Python
convention the package was already half-following: **`TypeError` when the value
is not a `str`, `ValueError` when it is an empty `str`.** No test in the
package asserted on these types, so nothing in-repo depended on the old
inconsistency.

### `CVContext.diam_mid` becomes a derived property

It was a stored field whose sole construction site passed exactly
`2.0 * radius_mid`, so two fields had to agree and nothing enforced it. Read
access is unchanged — a user callable writing `ctx.diam_mid` still works — but
it is no longer a constructor parameter.

## Efficiency — measured, or not landed

Median of 7 runs, same machine, before/after trees differing only by this
branch (`main` at `48fdd26` checked out into a scratch worktree for the
"before" column). Each row prints a digest; **all five digests match across
both columns.** Iteration counts are taken from the real workload: 3,405
control volumes is a CA1 cell at `CVPerBranch(5)`, and 2,046 is the measured
number of `Params.__hash__` calls in one `CellRuntimeState.from_cell` build.

| Benchmark | Before | After | Change |
|---|---:|---:|---|
| `Params.__hash__` ×2046, 500-element array param | 19.57 ms | 0.11 ms | **178× faster** |
| `Params.__hash__` ×2046, scalar params | 2.16 ms | 0.12 ms | **18× faster** |
| `Channel.__hash__` ×2046 | 2.51 ms | 1.25 ms | **2.0× faster** |
| `ScalarEventInput.validate_payload` ×20000 | 68.86 ms | 36.94 ms | **1.9× faster** |
| `CableProperty` rebuilt ×3405 CVs | 259.29 ms | 218.75 ms | **40.5 ms saved** |

**Why `Params` hashing was the big one.** `Params` is immutable, but
`__hash__` re-walked every value through `_to_hashable` on *every call*, and
`__eq__` built two complete converted dicts each time. `__init__` was already
computing that exact walk to validate hashability — and throwing it away. It
now keeps the converted view and the resulting hash. The array row is the
point: `_to_hashable` turns a Quantity array into a Python tuple via
`reshape(-1).tolist()`, so the old cost scaled with *array length × number of
hash calls*. A 500-element `g_max` painted over a cell went quadratic in CV
count; it is now paid once at construction.

**`validate_payload` was doing arithmetic to answer a metadata question.** It
called `payload.to_decimal(self.unit)` and discarded the result — a real array
multiply per call. The check is purely dimensional, so it now compares
`payload.dim` against the contract's. Honest scoping: the supported run paths
(`_multi_compartment/run.py`, `network/engine.py`) drive the step under
`brainstate.transform.for_loop` + `jit`, where XLA's DCE already eliminated the
dead convert. This 1.9× is real only for eager `Cell.update()` loops — tests,
notebooks, single-step drivers — and under jit it removes one dead node per
layout from the jaxpr rather than any runtime cost.

**`CableProperty` is the smallest relative win and the most conditional one.**
The paint lowering pass rebuilds a `CableProperty` per control volume from
fields it has *already* canonicalized, so `_coerce_temperature` re-did a decimal
conversion, an `asarray`, and a fresh `Quantity` allocation for a value that was
already in exactly the target form. Instrumenting the guard showed **8,178 hits
and 0 misses** across a full build+init — every single call was redundant. The
15% saving here is bounded by the fact that the other three cable fields still
go through the same rebuild; this pass did not touch those.

**Not claimed.** `_resolve_class_name`'s registry scan is ~185× slower than a
reverse index but ran 3 times in a full build, all short-circuiting before
touching the registry — microseconds. It is listed under *Deliberately not
changed* rather than reported as a win.

## What was fixed

### Reuse

- **`_synapse_schema._decimal` re-implemented `brainunit.get_mantissa`.**
  `value.to_decimal(value.unit)` converts a quantity to its own unit, which is
  the mantissa. Verified equal across scalar/array/bare/int-dtype cases; the
  private helper is deleted in favour of the documented API the rest of the
  repo already uses.
- **`CableProperty` was hand-built at 17 sites repo-wide** from the identical
  three-field literal. `braincell/_discretization/_testing.py` already had
  `make_cable` with byte-matching defaults, but `mech` owns `CableProperty`, so
  the builder moves to a new `braincell/mech/_testing.py` and `_discretization`
  re-exports it — the pattern `filter/_testing.py` uses for `morph`.
- **`_CHANNEL` / `_ION` string literals** in `_density.py` duplicated
  `_CATEGORY_CHANNEL` / `_CATEGORY_ION` in the sibling `_registry.py`, which
  `_density.py` already imports from. Two sources of truth for the strings that
  key the registry.

### Simplification

- **`Params` re-implemented five `collections.abc.Mapping` mixins** it already
  inherits (`keys`/`values`/`items`/`get`/`__contains__`), and its
  `__getitem__` wrapped a dict lookup in `try/except KeyError: raise
  KeyError(key) from None` — verified at runtime that the bare lookup already
  produces exactly `KeyError('zz')`. Its `__eq__` had two branches differing
  only in `other._items.items()` vs `other.items()`, which are the same thing
  for a `Mapping`.
- **`_to_hashable` had two array branches producing byte-identical output.**
  For an `np.ndarray`, `np.asarray(value) is value` and cannot raise, so the
  generic shape/dtype branch reproduces the ndarray branch exactly. Verified
  for 1-D, 2-D, and 0-D arrays.
- **`Density` enumerated its six fields five times** — `__init__`, `__eq__`,
  `__hash__`, `__repr__`, `_replace` — and every subclass had to re-enumerate
  all five. The sharpest symptom was `Channel.__repr__` doing string surgery on
  its parent's output (`base = super().__repr__(); base = base[:-1]`) to splice
  two fields in before the closing paren, and `Channel.__hash__` hashing a hash.
  The field list is now derived once by walking `__slots__` up the MRO.
- **The four probe classes repeated the same `__post_init__` preamble**, each
  passing its own class name to the deprecation warning as a string literal
  that a rename would silently desync, then repeating the same optional-`name`
  check. A `_LegacyProbe` base now owns both — deriving the name from
  `type(self).__name__` — and each subclass implements only `_validate` for its
  own fields.
- **`SineClamp.__post_init__` was six near-identical coercions**, two
  near-identical positivity guards, and five `object.__setattr__` lines; it
  becomes two table-driven loops.
- **`_coerce_scalar_quantity` was a strict subset of `_coerce_quantity`** and,
  despite its name, never checked scalar-ness. One parameterized function.
- **`MechanismRegistry.names()` and `.items()` each duplicated their whole body**
  across the `category is None` / `not None` branches.
- **`CVContext.position` duplicated `local_position`'s body** rather than
  delegating — two error messages for one condition. Its sibling
  `SamplingContext.position` in `filter` already delegates; iteration 3 made it
  do so.
- **`if cls is Density or not cls.category:`** — `Density.category` is `""`, so
  the second clause already covers the first.

### Docstrings describing behaviour the code does not have

- `_density.py` cross-referenced `braincell.cv._lower` as the module that sets
  `coverage_area_fraction`. There is no `braincell/cv/` package; the real site
  is `_discretization/mechanism.py`. It was the only `braincell.cv` reference
  in the repository.
- `_base.py` cross-referenced `braincell.mech.Probe`. No such class exists —
  the real ones are `StateProbe` / `MechanismProbe` / `CurrentProbe`.
- `Density` documented `params : Mapping or None` as a constructor parameter,
  which is the bug fixed above.

## Deliberately not changed

Recorded so the next reader does not re-litigate them, and so later iterations
inherit the reasoning rather than the rediscovery.

1. **`metric.branch_x` / `metric.radius` keep their two-name fallback lists.**
   Iteration 3 deferred to this iteration the question of whether `CVContext`
   should expose `branch_x` / `radius` directly so `filter/metric.py`'s
   fallbacks collapse to one name each. Answer: no. `CVContext`'s names are
   coherent triples — `prox`/`midpoint`/`dist` and
   `radius_prox`/`radius_mid`/`radius_dist`. Renaming `midpoint` to `branch_x`
   would break both triples to save two list entries, and adding an alias
   property would *add* a second spelling, which is the opposite of the goal.
   The fallback list in `metric.py` is exactly the right adapter between two
   internally-coherent vocabularies, and it is documented public API
   (`examples/multi_compartment/synapse.ipynb`).

2. **`ProbeMechanism` is dead but not deleted here.** Zero construction sites
   outside `_point_test.py`, yet four production dispatch branches exist for it
   in `_compute/layouts.py` and `_compute/table.py`, plus entries in
   `docs/apis/`, `docs/tutorials/mech.ipynb`, and `interface-map.md`. Deleting
   it is a `_compute` change wearing a `mech` hat; it belongs to **iteration 8**
   (`_compute`), which will be reworking that dispatch chain anyway.

3. **`_warn_legacy_probe` stays.** It fires on 222 in-repo construction sites,
   including the repo's own `network/_testing.py` fixtures, with no
   `filterwarnings` entry to damp it — genuinely noisy. But the warning names a
   real replacement (`Cell.record(..., braincell.observe.*)`), and the fix is
   to *perform* the migration across 222 sites, not to silence the signal.
   That is a migration, not a simplification.

4. **`Junction` has no construction site outside its own tests**, but
   `docs/design/TODO.md` tracks "Junction runtime wiring" as planned work. Flagged, not
   deleted.

5. **The `Density` / `Synapse` duplication is not unified.** They are the
   same "registry-keyed, named, params-carrying declaration" implemented twice,
   and `Synapse` lacking a `category` is why three separate consumer chains
   in `_compute` each need a parallel branch. A shared `RegistryKeyed` mixin is
   the right end state, but it changes `_compute/layouts.py`, `_compute/table.py`,
   and `_discretization` together — **iteration 14**.

6. **Probe default-name derivation stays where it is.** `Density` and
   `Synapse` have `instance_name`; the probe classes do not, so
   `_discretization/mechanism.py` and `_compute/table.py` each hand-roll the
   same three suffix formulas. Giving the probes a `default_name` is a genuine
   altitude fix, but the payoff lands in two other packages — **iteration 14**.

7. **`MechanismRegistry` and `quad`'s `IntegratorRegistry` are not unified.**
   Structurally similar, semantically divergent: `mech` namespaces by category
   and has no metadata/override/deprecation; `quad` has all three and no
   category namespace. That is a redesign, not a reuse fix.

8. **`_registry._resolve_class_name`'s linear scan is left alone.** It sorts an
   entire registry category to answer one identity question, which is ~185×
   slower than a reverse-index lookup — but it was called 3 times in a full
   CA1 build, all of which short-circuited before touching the registry. The
   absolute saving is microseconds; adding and maintaining a `_by_class` index
   costs more than it returns.

9. **`Mechanism` (`_base.py`) is left in place, with its docstring corrected.**
   The marker has zero `isinstance` call sites outside `mech` — verified — and
   `CableProperty`, which `Cell.paint` accepts alongside `Density`, does not
   inherit it. So it is currently an abstraction with no clients. Both fixes
   (give it a real job by including `CableProperty`, or delete it) change
   `_discretization/mechanism.py`'s `paint` signature, so the decision travels
   with **iteration 14**.

## Verification

```
$ pytest braincell/mech -q
177 passed, 23 warnings, 4 subtests passed in 4.26s

$ pytest braincell/ -q
2724 passed, 15 skipped, 411 warnings, 401 subtests passed in 173.11s

$ pre-commit run --files <24 changed files>
... ruff (legacy alias) .... Passed
... ruff format .......... Passed          (all hooks Passed or Skipped)
```

Against the baseline `main` at `48fdd26` (2,730 passed / 397 subtests), the
package total moves to **2,724 passed / 401 subtests** — a net of **−6 tests
and +4 subtests**, every one of which is accounted for:

| Module | Δ | Detail |
|---|---:|---|
| `_density_test.py` | **+3** | −3 tests for the deleted `with_params` / `with_name` / `with_integration`; +1 copy-preserves-subclass-fields; +1 `with_coverage` range regression (2 subtests); +4 in the new `DensityParamsKeywordTest` |
| `_params_test.py` | −2 | the two `Params.without` tests |
| `_registry_test.py` | −3 | the three `add_alias` tests |
| `_cable_test.py` | −1 | −1 `with_updates`; two temperature-rejection tests merged into one with 2 subtests (−1); +1 canonical-passthrough |
| `_point_test.py` | −3 | the deprecated `Synapse` alias test and the two `target_index` tests |
| **Total** | **−6** | subtests **+4** = 2 (`with_coverage`) + 2 (temperature) |

`braincell/mech` alone goes 183 → 177, matching the same −6.

**One cross-package failure surfaced during verification and was fixed, not
worked around.** The first full-suite run returned
`filter/metric_test.py::test_cv_context_maps_midpoint_and_midpoint_radius` —
it constructed a `CVContext` with an explicit `diam_mid=2.0 * u.um`, which is
no longer a constructor parameter. The test now omits it and additionally
asserts that the derived value equals `2 * radius_mid`, which is the property
the change introduces. The `mech`-only run was green, so this was visible only
package-wide — the same lesson iteration 3 recorded.
