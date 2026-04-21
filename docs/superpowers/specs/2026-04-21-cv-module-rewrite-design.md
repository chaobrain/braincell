# braincell.cv — full rewrite design

**Date:** 2026-04-21
**Scope:** `braincell/cv/` (~1727 LOC across 4 source files + 1 test file)
**Driver:** clean up correctness bugs, narrow public surface, restructure internals into one compact two-stage pipeline.

---

## 1. Motivation

Audit of current `braincell/cv/` surfaced a cluster of correctness, API, and testability problems:

### Correctness bugs

1. `_scale_density_for_coverage` swallows all exceptions during `g_max` scaling and silently falls back to `coverage_area_fraction`. Unit mismatches and init-callable failures are hidden. Two representations of the same physical scaling coexist → runtime bug magnet.
2. `EPSILON = 1e-12` is applied to both **normalized** x ∈ [0, 1] and **physical** μm lengths/areas. Comparing μm² area to 1e-12 is meaningless. Mixed-dimension tolerance.
3. `_interval_owns_boundary` zero-length-segment ownership is only correct for the `dist == 1.0` case. Interior CV boundaries with exact-hit floating points can misattribute segment radii.
4. `_locate_branch_cv_by_x` silently snaps non-owning `x` to `ids[-1]`. Bug masking.
5. `_build_frusta` returns `()` for zero-length branches. CV gets zero length / area / axial factor, no error. Runtime inherits degenerate compartments.
6. Duplicate density paint on the same region silently accumulates → conductance doubles. Only `CableProperty` gets deduped.
7. Radii ≤ 0 propagate `inf` axial resistance. No early validation.
8. No post-build consistency check: no cycle detection, no parent-id validation, no bounds-coverage check.
9. `children_cv = tuple(sorted(...))` sorts by integer id, losing morphological order.
10. `DLambda` docstring claims “raises `NotImplementedError` today” but implementation works. Stale doc.
11. Place-rule auto-generated names can collide (two locsets with the same `display_name`) → silent runtime-dedup burden.
12. `build_cv_geo` TypeError message says “expects Morpho” — stale rename; type is `Morphology`.

### Architectural friction

- Mutable `CVMech` accumulator threaded through three in-place mutators (`init_cv_mech` → `apply_paint_rules` → `apply_place_rules`). Tight coupling, hard to test isolated stages.
- `_frusta` is a private field on the public frozen `CV`. Debug reconstruction (`as_branch`) mixes presentation into the data class.
- Region masks are re-evaluated on every paint / place rule (and again inside the policy for DLambda). Wasted work on complex filters.
- `Quantity = Any`, `tuple[object, ...]` fields — lost static typing.
- Public surface leaks internal names: `_multi_compartment/build.py` imports `assemble_cv`, `build_cv_geo`, `apply_paint_rules`, `init_cv_mech` directly.
- No tests for `_cv.py`, `_geo.py`, or `_mech.py`. Only `_policy.py` tested.

### Goals

1. Single, clean public API: `CV`, `CVPolicy`, `CVPerBranch`, `MaxCVLen`, `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`. Everything else internal.
2. One functional lowering pipeline. No mutable accumulator.
3. Fix every correctness bug listed above.
4. Typed epsilons — one per physical dimension.
5. Co-located tests per source file, plus property-based tests for key invariants.
6. Migrate `_multi_compartment/cell.py` and `_multi_compartment/build.py` to a single `build_cvs(...)` entry point. No transitional aliases.

---

## 2. Non-goals

- No new `DLambda` math. Keep today’s uniform-cable-per-branch restriction; document explicitly.
- No new CV policies.
- No runtime-side changes in `braincell.compute` or `braincell._multi_compartment` beyond the import migration needed to consume the new entry point.
- No vectorization / JAX-ification of the lowering pipeline (still Python-level).

---

## 3. Architecture

### Layout (approach B — two-stage compact)

```
braincell/cv/
  __init__.py       # public exports
  _cv.py            # CV frozen dataclass + build_cvs(...) entry
  _policy.py        # CVPolicy + subclasses (bug-fixed)
  _lower.py         # PaintRule/PlaceRule + frustum math + region cache +
                    # coverage + one-shot lower(...) function. Internal
                    # _GeoCV / _MechBucket never leave this module.
  _debug.py         # cv_to_branch(cv, morpho) reconstruction
  _cv_test.py
  _policy_test.py
  _lower_test.py
  _debug_test.py
```

### Public exports (`__init__.py`)

```python
from ._cv import CV
from ._policy import (
    CompositeByTypePolicy, CVPerBranch, CVPolicy,
    CVPolicyByTypeRule, DLambda, MaxCVLen,
)

__all__ = [
    "CV", "CompositeByTypePolicy", "CVPerBranch", "CVPolicy",
    "CVPolicyByTypeRule", "DLambda", "MaxCVLen",
]
```

`PaintRule`, `PlaceRule`, `build_cvs`, and normalize/merge helpers are imported by `braincell._multi_compartment` from `braincell.cv._cv` / `braincell.cv._lower` as internal-public. They are NOT in `__all__`.

### Data flow

```
Cell.paint / Cell.place  →  (morpho, policy, paint_rules, place_rules)
                                ↓
                     build_cvs (in _cv.py)
                                ↓
                     _lower.lower(...):
                       1. _validate_morpho           (raise on bad radii / empty branches)
                       2. _RegionCache(morpho)       (one evaluate per region)
                       3. policy.resolve_cv_bounds   (reads paint_rules for DLambda)
                       4. _validate_bounds           (cover [0,1], no overlap/gap)
                       5. _build_geo                 (pure — returns fresh tuple + ids_by_branch)
                       6. _validate_connectivity     (parent exists, no cycles, mono order)
                       7. _build_mech                (pure — returns per-CV buckets)
                       8. _validate_names            (point-mechanism name uniqueness)
                       9. _assemble                  (freeze each geo+bucket → CV)
                                ↓
                     tuple[CV, ...]  →  consumed by _multi_compartment/build.py
```

### `CV` frozen dataclass

```python
@dataclass(frozen=True)
class CV:
    id: int
    branch_id: int
    branch_type: str
    prox: float                       # normalized [0, 1] on branch
    dist: float
    parent_cv: int | None
    children_cv: tuple[int, ...]      # morphological order
    length: u.Quantity                # μm
    area: u.Quantity                  # μm² lateral
    cm: u.Quantity                    # uF/cm²
    ra: u.Quantity                    # ohm·cm
    v: u.Quantity                     # mV
    temp: u.Quantity                  # kelvin (normalized at boundary)
    r_axial: u.Quantity               # ohm
    r_axial_prox: u.Quantity
    r_axial_dist: u.Quantity
    radius_prox: u.Quantity           # precomputed (was a @property)
    radius_mid: u.Quantity            # precomputed (was diam_mid / 2)
    radius_dist: u.Quantity           # precomputed
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]

    @property
    def region(self) -> RegionMask:
        return RegionMask(((self.branch_id, self.prox, self.dist),))
```

Removed fields: `_frusta` (no longer carried).
Removed methods: `as_branch` (replaced by `cv._debug.cv_to_branch(cv, morpho)`).
Removed properties: `radius_prox` / `radius_dist` / `diam_mid` as @property accessors — now plain fields, precomputed once during lowering. Users wanting diameter compute `2 * cv.radius_mid`.

### `build_cvs` entry point

```python
def build_cvs(
    morpho: Morphology,
    *,
    policy: CVPolicy,
    paint_rules: tuple[PaintRule, ...] = (),
    place_rules: tuple[PlaceRule, ...] = (),
) -> tuple[CV, ...]: ...
```

Single entry. `_multi_compartment/cell.py` and `_multi_compartment/build.py` import this plus `PaintRule`, `PlaceRule`, `default_paint_rules`, `normalize_paint_rules`, `normalize_place_rule`, `merge_paint_rules`, `merge_place_rules` from `braincell.cv._lower`.

---

## 4. `_lower.py` internals

### Typed epsilon constants

```python
EPS_PARAM    = 1e-9     # normalized x ∈ [0, 1]
EPS_LEN_UM   = 1e-6     # μm lengths
EPS_AREA_UM2 = 1e-9     # μm² areas
```

Each helper picks the right one. No more mixed-dimension `1e-12`.

### Transient types (module-private)

```python
@dataclass(frozen=True)
class _Frustum:
    prox: float
    dist: float
    length_um: float
    r_prox_um: float
    r_dist_um: float
    point_prox_um: np.ndarray | None
    point_dist_um: np.ndarray | None

@dataclass(frozen=True)
class _GeoCV:
    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    midpoint: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length_um: float
    lateral_area_um2: float
    axial_factor_total_per_cm: float
    axial_factor_prox_per_cm: float
    axial_factor_dist_per_cm: float
    r_prox_um: float
    r_mid_um: float
    r_dist_um: float

@dataclass
class _MechBucket:
    cable: CableProperty
    density_by_key: dict[tuple[str, str], Density]   # (region_id, class_name) → last-wins
    points: list[Point]
```

`_GeoCV` and `_MechBucket` are transient locals inside `lower(...)`. Never returned from `_lower.py`.

### Rule dataclasses (internal-public)

```python
@dataclass(frozen=True)
class PaintRule:
    region: RegionExpr
    mechanism: CableProperty | Density

@dataclass(frozen=True)
class PlaceRule:
    locset: LocsetExpr
    mechanisms: tuple[Point, ...]
    site: Literal["mid"] = "mid"
```

### Normalize + merge

```python
def normalize_paint_rules(region, mechanisms) -> tuple[PaintRule, ...]
def normalize_place_rule(locset, mechanisms) -> PlaceRule
def default_paint_rules() -> tuple[PaintRule, ...]    # AllRegion + default cable
def merge_paint_rules(existing, incoming) -> tuple[PaintRule, ...]
def merge_place_rules(existing, incoming) -> tuple[PlaceRule, ...]
```

`merge_paint_rules` — last-wins keyed by `(region, _mechanism_key(mechanism))`:
- `CableProperty` key: `("__cable__",)`
- `Density` key: `("__density__", class_name)`
- New incoming rule replaces existing rule with the same key on the same region. Different regions → both rules kept.

`merge_place_rules` — today’s dedup by full equality.

### Region mask cache

```python
class _RegionCache:
    def __init__(self, morpho: Morphology): ...
    def intervals(self, region: RegionExpr) -> dict[int, tuple[tuple[float, float], ...]]
    def points(self, locset: LocsetExpr) -> tuple[tuple[int, float, str], ...]
```

Keyed by `id(region)` (regions are typically hashable, but `id()` avoids any hashability assumption). Collapses N region evaluations → 1. Used by `_lower.lower` AND passed through into `policy.resolve_cv_bounds` for DLambda (optional kwarg, default `None` for backward compat of the policy signature).

### Lowering pipeline

```python
def lower(
    morpho: Morphology,
    *,
    policy: CVPolicy,
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
) -> tuple[CV, ...]:
    _validate_morpho(morpho)
    cache = _RegionCache(morpho)
    bounds = policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)
    _validate_bounds(bounds, morpho)
    geos, ids_by_branch = _build_geo(morpho, bounds)
    _validate_connectivity(geos, ids_by_branch, morpho)
    buckets = _build_mech(
        morpho, geos, ids_by_branch,
        paint_rules=paint_rules, place_rules=place_rules, cache=cache,
    )
    _validate_names(buckets)
    return tuple(_assemble(g, b) for g, b in zip(geos, buckets))
```

Every stage pure-functional. No hidden mutation.

### Frustum math (bug-fixed)

```python
def _build_frusta(branch, *, prox: float, dist: float) -> tuple[_Frustum, ...]:
    # Validate: 0 <= prox < dist <= 1 via EPS_PARAM.
    # Validate: every segment radius > 0. Raise ValueError on <= 0.
    # Reject zero-total-length branches before this call (_validate_morpho).
    # Boundary ownership: x == dist belongs to the NEXT CV for interior
    # boundaries; last CV on branch owns x == 1.0 exactly.
    # Zero-length segments at interior boundaries are dropped — they contribute
    # no area / length / axial factor. Retained only when prox == 0 or
    # dist == 1 so branch-root / branch-tip junction points preserve their
    # 3D coordinates for visualization.
```

Boundary ownership rule (single source of truth, replaces both
`_interval_owns_boundary` and `_locate_branch_cv_by_x`):

> For a CV with normalized range `[prox, dist]`: it owns `x` when
> `prox - EPS_PARAM <= x < dist - EPS_PARAM`, except when `dist >= 1.0 - EPS_PARAM`
> (last CV on branch), which owns up to and including `x == 1.0`.

Lookup helper:

```python
def _locate_cv_on_branch(ids, geos, *, x: float) -> int:
    # Raises ValueError if no CV on this branch owns x. Never silently snaps.
```

### Frustum scalar helpers

```python
def _midpoint_radius_um(frusta) -> float
def _lateral_area_um2(frusta) -> float
def _axial_factor_per_cm(frusta) -> float        # raises if any r <= 0
def _split_frusta(frusta, *, x: float) -> tuple[tuple[_Frustum, ...], tuple[_Frustum, ...]]
```

No `inf` returns. Validation happens early (`_validate_morpho` + `_build_frusta`), so by the time a helper runs the inputs are known-good.

### Coverage fraction (single representation)

```python
def _coverage_fraction(geo: _GeoCV, intervals) -> float:
    # Sum lateral area of clipped intervals / geo.lateral_area_um2.
    # Clamped to [0.0, 1.0] with EPS_PARAM tolerance.
    # Total area is guaranteed > EPS_AREA_UM2 by _validate_morpho + strict
    # _build_geo, so no divide-by-zero guard is needed here.

def _apply_density(bucket, mechanism, *, region_id, fraction):
    key = (region_id, mechanism.class_name)
    if fraction >= 1.0 - EPS_PARAM:
        bucket.density_by_key[key] = mechanism
    else:
        bucket.density_by_key[key] = mechanism.with_coverage(fraction)
    # Ions (category != "channel"): stored unchanged under the same key.
    # No g_max multiplication. No try/except.
```

`Density.with_coverage(fraction)` is assumed present (already used today). Runtime in `braincell.compute` consumes `coverage_area_fraction` when converting density → point mechanism. That contract is already in place — this rewrite just stops the dual-path.

### Point-mechanism lowering (unique names, raise on collision)

```python
def _apply_place(bucket, mechanism, *, display_name, seen_names: set[str]):
    named = _resolve_point_name(mechanism, display_name=display_name)
    if named.name in seen_names:
        raise ValueError(
            f"Duplicate point-mechanism name {named.name!r} from place rule "
            f"at {display_name!r}. Supply an explicit name= argument to "
            f"disambiguate."
        )
    seen_names.add(named.name)
    bucket.points.append(named)
```

`seen_names` is a single set carried across the whole lowering, not per-CV — two clamps on two CVs with the same auto-generated name still raise.

`_resolve_point_name` is unchanged from today: autogenerated names for `StateProbe` / `MechanismProbe` / `CurrentProbe`; others return as-is.

### Connectivity validation

```python
def _validate_connectivity(geos, ids_by_branch, morpho):
    # 1. Each non-first CV on a branch has parent_cv = (prior CV on same branch).
    # 2. Each first CV on a branch is either:
    #    - on morpho.root → parent_cv is None
    #    - otherwise → parent_cv is a CV on parent branch resolved via
    #      morpho.edges + _locate_cv_on_branch(edge.parent_x).
    # 3. No cycle (DFS from roots, mark-visited).
    # 4. `children_cv` listed in morpho.edges traversal order, not sorted by id.
    # 5. Every CV id referenced as parent_cv / children_cv is a valid index.
    # Raises ValueError with branch_id + cv_id on any violation.
```

### Bounds validation

```python
def _validate_bounds(bounds, morpho):
    # For each branch:
    #  - Non-empty tuple of (prox, dist) pairs.
    #  - Each 0 <= prox < dist <= 1, within EPS_PARAM.
    #  - Sorted: bounds[i].dist <= bounds[i+1].prox (no overlap / no gap).
    #  - First prox == 0.0, last dist == 1.0 (within EPS_PARAM).
    # Raises ValueError pinpointing the offending branch.
```

### Morphology validation

```python
def _validate_morpho(morpho):
    # Every branch has total length > EPS_LEN_UM.
    # Every segment radius > 0.
    # Raises ValueError with branch_id on any violation.
```

---

## 5. `_policy.py` changes

- Stay inside `CVPolicy` hierarchy; public API unchanged.
- `DLambda` docstring rewritten: remove “placeholder” language, document uniform-cable-per-branch restriction explicitly with the error message users will hit.
- `_sorted_unique_coords` uses `EPS_PARAM` consistently.
- `_cable_signature` stays `(Ra, Cm)`. Resting potential / temperature variations are explicitly allowed; document this in the class docstring.
- `CVPolicyByTypeRule` post-init unchanged (already validates). Add a test that empty `branch_types` after `__post_init__` is impossible (first branch already filters).
- `CompositeByTypePolicy` unchanged other than EPS renames.

No signature changes. No new policies.

---

## 6. `_debug.py`

```python
def cv_to_branch(cv: CV, morpho: Morphology) -> Branch:
    """Reconstruct a standalone Branch from a CV's normalized range.

    Slices `morpho.branches[cv.branch_id]` from `cv.prox` to `cv.dist`
    using the same frustum math as `_lower._build_frusta`, then returns a
    typed subclass via `branch_class_for_type(cv.branch_type)`.
    """
```

Implementation calls a shared `_build_frusta` (exposed from `_lower` as module-private helper imported by `_debug`). No duplication of clipping math.

`braincell.vis` can later re-export this via its own debug surface. For now `from braincell.cv._debug import cv_to_branch` is enough.

Call sites of old `CV.as_branch()`: one match — `_debug_test.py` exercise. Migration is a single import change for any downstream users.

---

## 7. Migration (`_multi_compartment/`)

### Before

```python
from braincell.cv._cv import assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    PaintRule, PlaceRule, apply_paint_rules, apply_place_rules,
    default_paint_rules, init_cv_mech, merge_paint_rules, merge_place_rules,
    normalize_paint_rules, normalize_place_rule,
)
from braincell.cv._policy import CVPerBranch, CVPolicy
```

### After

```python
from braincell.cv import CV, CVPolicy, CVPerBranch
from braincell.cv._cv import build_cvs
from braincell.cv._lower import (
    PaintRule, PlaceRule,
    default_paint_rules, normalize_paint_rules, normalize_place_rule,
    merge_paint_rules, merge_place_rules,
)
```

Inside `_multi_compartment/build.py` and `_multi_compartment/cell.py` the 3-step `build_cv_geo` / `init_cv_mech` / `apply_paint_rules` / `apply_place_rules` / `assemble_cv` dance collapses to:

```python
cvs = build_cvs(
    morpho,
    policy=cell.cv_policy,
    paint_rules=cell.paint_rules,
    place_rules=cell.place_rules,
)
```

`compute/_point_tree.py` import of `from braincell.cv._cv import CV` becomes `from braincell.cv import CV`. Its import of `from braincell.cv._geo import EPSILON` is dropped — it should pick the appropriate typed epsilon from its own module, not reach into cv internals. (Out of scope for the rewrite: call site already inline-redefines a tolerance; audit separately.)

Tests for `_multi_compartment` re-run green with the new entry point.

---

## 8. Error model

Every failure raises a `ValueError` or `TypeError` with:

- the offending branch_id / cv_id / rule index where applicable,
- the actionable remediation (“Supply an explicit `name=` to disambiguate”, “Unify Ra/cm within the branch or use another cv_policy”, “CV bounds must satisfy 0 <= prox < dist <= 1, got …”).

No silent degradation. No `inf` fall-throughs. No `except Exception: pass`.

---

## 9. Testing

One `*_test.py` per source module, plus property-based tests for the invariants that matter most.

### `_cv_test.py`

- CV field defaults / immutability.
- `CV.region` round-trip.
- Precomputed `radius_prox / radius_mid / radius_dist` match hand-calculated expected values on a uniform frustum, a tapered frustum, and a multi-segment branch.
- `build_cvs` smoke test: 1-branch, 3-branch, deep tree (10 branches).

### `_policy_test.py`

All tests from today, plus:

- `DLambda` with non-uniform Ra on half a branch → raises with expected message.
- `CompositeByTypePolicy` round-trip over mixed-type tree.
- `CVPolicyByTypeRule` validation of bogus branch types (already covered).

### `_lower_test.py`

- Paint-rule merge: Density with same (region, class_name) replaces prior.
- Paint-rule merge: Density with same class_name on different regions both retained.
- CableProperty last-wins behavior unchanged.
- Coverage fraction: full-overlap → 1.0 (no scaling stored); half-overlap → 0.5 stored on `coverage_area_fraction`.
- Coverage fraction for ions: stored unchanged (category != "channel").
- Point-mechanism name collision between two locsets with same display_name → raises.
- Point-mechanism explicit name already set → no autogeneration, collision still detected.
- Connectivity: linear 3-branch tree parents form a chain.
- Connectivity: branched tree children order matches morpho.edges traversal order.
- Bounds validation: overlap / gap / missing coverage → raises.
- Morphology validation: zero-length branch → raises; radius = 0 → raises.
- Frustum boundary ownership: interior CV boundary x == dist belongs to next CV.
- Frustum boundary ownership: x == 1.0 belongs to last CV.
- `_RegionCache` evaluates each region once (assert call count via monkeypatched Region).

### Property-based tests (`_lower_test.py`)

Using `hypothesis`:

- For a random uniform-branch morpho + random `CVPerBranch` count k: sum of per-CV coverage over an AllRegion paint equals 1.0 ± EPS_PARAM.
- For any valid bounds: sum of per-CV lengths = branch total length ± EPS_LEN_UM.
- Every CV id appears exactly once as a child or as a branch root, never twice.

### `_debug_test.py`

- `cv_to_branch(cv, morpho)` reconstructs a Branch whose total length matches `cv.length`.
- Reconstructed branch type matches `cv.branch_type`.
- On a CV spanning multiple source segments, reconstructed branch has the right number of segments.

---

## 10. Performance expectations

- One region evaluate per region (today: N per region → ~Nx speed-up on complex filters).
- Fewer Python-level dict/list allocations during build (single pass over geos, single pass over paint rules).
- Precomputed radii eliminate per-property frustum walks on every `cv.radius_*` access.
- No JIT / vectorization. Still O(n_cv * n_paint_rules) Python.

Acceptable: today’s lowering is already fast enough for ~10k CV morphologies; the rewrite keeps the same asymptotic cost while eliminating hot-path allocations.

---

## 11. Open risks

1. Downstream code relying on `CV._frusta` or `CV.as_branch` — need to grep. Expected: only `_multi_compartment`, `compute`, and existing tests consume `CV`. None reach `_frusta` today (search done, no hits).
2. `Density.with_coverage(...)` availability — already used in today’s code, confirmed by grep. No new API required from `braincell.mech`.
3. Strict validation may break loading of noisy real-world SWC files that have zero-radius segments or zero-length branches. The IO readers (`braincell.io.swc`, `braincell.io.asc`) already normalize these via `SwcReport` / `AscReport` warnings, so the assumption holds. Any surviving degenerate morphology is genuine user error — strict raise is the right behavior. If a regression surfaces in real morphologies, the fix is at the IO layer, not here.
4. `compute/_point_tree.py` imports `EPSILON` from `cv._geo` today. That import disappears. `_point_tree.py` needs its own tolerance constant (or picks up `EPS_PARAM` from a shared location). Included in migration.

---

## 12. Implementation phases

1. Skeleton + policy migration. Rewrite `_policy.py` with typed epsilons, unchanged public behavior. Green on existing `_policy_test.py`.
2. `_lower.py`. Port PaintRule / PlaceRule / normalize / merge. Write `_RegionCache`. Write validation helpers. Write `_build_frusta` / `_build_geo` / `_build_mech` / `_assemble` pure functions. Write `lower(...)` composer. Write `_lower_test.py` alongside.
3. `_cv.py`. New `CV` dataclass (drop `_frusta`, add `radius_*` fields). Write `build_cvs(...)`. Write `_cv_test.py`.
4. `_debug.py` + test. Delete old `CV.as_branch`.
5. Migrate `_multi_compartment/cell.py` and `_multi_compartment/build.py` to `build_cvs`.
6. Fix `compute/_point_tree.py` EPSILON import.
7. Delete old `_geo.py` and `_mech.py`. Update `__init__.py`.
8. Full suite green. Run property tests.
