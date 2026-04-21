# braincell.cv Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `braincell/cv/` into a two-stage compact pipeline with typed epsilons, strict validation, single coverage representation, and co-located per-module tests — while preserving the public API (`CV`, `CVPolicy`, `CVPerBranch`, `MaxCVLen`, `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`).

**Architecture:** `braincell/cv/` becomes `{__init__.py, _cv.py, _policy.py, _lower.py, _debug.py}` plus co-located `*_test.py`. `_cv.py` exposes the `build_cvs(...)` entry; `_lower.py` contains one functional `lower(...)` composer with pure helpers for frustum math, rule normalize/merge, region caching, coverage, and validation; `_debug.py` reconstructs a `Branch` from a `CV` for visualization.

**Tech Stack:** Python 3.13, pytest (unittest.TestCase), `brainunit` for quantities, `jax` (CPU), `hypothesis` for property tests.

**Spec:** `docs/superpowers/specs/2026-04-21-cv-module-rewrite-design.md`.

**Pre-flight test command:**
```bash
pytest braincell/cv/ braincell/_multi_compartment/ braincell/compute/ -q
```
Expected at the start of the plan: everything green.

---

## Phase 0: Preparation

### Task 0.1: Baseline green test run

**Files:**
- None (read-only)

- [ ] **Step 1: Run the pre-flight suite**

```bash
pytest braincell/cv/ braincell/_multi_compartment/ braincell/compute/ -q
```
Expected: PASS. Record counts.

- [ ] **Step 2: Record baseline counts in a scratch note**

```bash
pytest braincell/cv/ braincell/_multi_compartment/ braincell/compute/ --collect-only -q | tail -3
```
Expected: number of collected tests. Note them down; we compare at the end.

---

## Phase 1: Policy module rewrite (behavior-preserving)

Rewrite `_policy.py` with typed epsilons and refreshed docstrings. Behavior unchanged; existing `_policy_test.py` must stay green.

### Task 1.1: Introduce typed epsilons + move `_policy.py` to new constants

**Files:**
- Modify: `braincell/cv/_policy.py`

- [ ] **Step 1: Open the file and replace the module-level `EPSILON = 1e-12` with typed constants**

Replace:
```python
EPSILON = 1e-12
```
with:
```python
EPS_PARAM = 1e-9        # tolerance for normalized x in [0, 1]
EPS_LEN_UM = 1e-6       # tolerance for physical μm lengths
```

- [ ] **Step 2: Audit every `EPSILON` use in `_policy.py` and migrate**

Every `EPSILON` reference in `_policy.py` is parametric (normalized-x comparisons): replace with `EPS_PARAM`. Specifically update these call sites:

- `_sorted_unique_coords`: `if not out or abs(value - out[-1]) > EPSILON:` → `> EPS_PARAM`
- `_last_cable_covering`: `if prox - EPSILON <= x <= dist + EPSILON:` → `prox - EPS_PARAM <= x <= dist + EPS_PARAM`
- `_same_cable_signature`: keep `atol=EPS_PARAM, rtol=EPS_PARAM` but note that Ra (ohm·cm) and Cm (uF/cm²) comparisons are scale-dependent. Switch to `np.isclose(..., atol=1e-9, rtol=1e-6)`:
  ```python
  def _same_cable_signature(lhs, rhs) -> bool:
      return bool(
          np.isclose(lhs[0], rhs[0], atol=1e-9, rtol=1e-6)
          and np.isclose(lhs[1], rhs[1], atol=1e-9, rtol=1e-6)
      )
  ```
- `_bounds_from_max_len_um`: `if branch_len_um <= max_len_um + EPSILON:` → `<= max_len_um + EPS_LEN_UM` (these are μm).
- `_bounds_from_max_len_um`: `n_cv = int(np.ceil((branch_len_um / max_len_um) - EPSILON))` → `- EPS_PARAM` (dimensionless ratio).
- `_bounds_from_d_lambda`: `n_cv = int(np.ceil((electrotonic_length / d_lambda) - EPSILON))` → `- EPS_PARAM` (dimensionless).

- [ ] **Step 3: Update the `DLambda` class docstring to remove stale placeholder language**

Replace the `DLambda` class docstring with:

```python
    """Splits each branch using the NEURON-style d_lambda discretization.

    Computes ``lambda_f`` per segment from branch diameter, ``Ra``, ``cm``, and
    ``frequency``. Sums the electrotonic length of the branch, divides by
    ``d_lambda``, and rounds up to the next integer (optionally promoted to an
    odd count when ``keep_odd`` is ``True``).

    This policy requires uniform ``(Ra, cm)`` **within each branch**. Painting
    a different ``CableProperty`` onto a sub-interval of a branch raises
    ``ValueError``. Resting-potential and temperature variations within a
    branch are explicitly allowed — only the two cable values that enter
    ``lambda_f`` are checked for uniformity.
    """
```

- [ ] **Step 4: Run the policy tests**

Run: `pytest braincell/cv/_policy_test.py -v`
Expected: PASS (same count as baseline).

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_policy.py
git commit -m "refactor(cv): split EPSILON into typed EPS_PARAM / EPS_LEN_UM"
```

---

## Phase 2: Build `_lower.py` scaffolding (rules + caches)

`_lower.py` is new. Build it incrementally, TDD. The file ends up containing: rule dataclasses, normalize/merge helpers, region cache, transient types, frustum math, validation, and the `lower(...)` composer.

### Task 2.1: Create empty `_lower.py` with the rule dataclasses

**Files:**
- Create: `braincell/cv/_lower.py`
- Create: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Write failing test for PaintRule / PlaceRule**

Create `braincell/cv/_lower_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (... same header as other files ...)

import unittest

import brainunit as u

from braincell.cv._lower import PaintRule, PlaceRule
from braincell.filter import AllRegion, BranchSlice, UniformSamples
from braincell.mech import CableProperty, Channel, CurrentClamp


class PaintAndPlaceRuleTest(unittest.TestCase):
    def test_paint_rule_is_frozen_and_equal_by_value(self) -> None:
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        r1 = PaintRule(region=AllRegion(), mechanism=cable)
        r2 = PaintRule(region=AllRegion(), mechanism=cable)
        self.assertEqual(r1, r2)
        with self.assertRaises(Exception):
            r1.region = AllRegion()  # type: ignore[misc]

    def test_place_rule_default_site_is_mid(self) -> None:
        rule = PlaceRule(
            locset=UniformSamples(n=1, seed=0),
            mechanisms=(CurrentClamp.step(0.2 * u.nA, 10 * u.ms),),
        )
        self.assertEqual(rule.site, "mid")
```

- [ ] **Step 2: Run it, see it fail**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: FAIL — `ModuleNotFoundError: braincell.cv._lower`.

- [ ] **Step 3: Create `braincell/cv/_lower.py` with the rule dataclasses**

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (... same header as other files ...)

"""Pure-functional control-volume lowering.

One entry point: :func:`lower`. All helpers in this module are pure
functions operating on immutable or locally-scoped data. Internal
types ``_Frustum``, ``_GeoCV``, ``_MechBucket`` never leave this
module — the final ``tuple[CV, ...]`` is the only output.
"""

from dataclasses import dataclass
from typing import Literal

from braincell.filter import LocsetExpr, RegionExpr
from braincell.mech import CableProperty, Density, Point

__all__ = ["PaintRule", "PlaceRule"]


@dataclass(frozen=True)
class PaintRule:
    """Normalized ``Cell.paint(...)`` declaration.

    Binds one :class:`RegionExpr` to either :class:`CableProperty` or
    :class:`Density`. Stored in declaration order by ``Cell``.
    """

    region: RegionExpr
    mechanism: CableProperty | Density


@dataclass(frozen=True)
class PlaceRule:
    """Normalized ``Cell.place(...)`` declaration.

    Maps one :class:`LocsetExpr` to a tuple of point mechanisms. Only
    ``site='mid'`` is currently supported.
    """

    locset: LocsetExpr
    mechanisms: tuple[Point, ...]
    site: Literal["mid"] = "mid"
```

- [ ] **Step 4: Run the test, see it pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): add _lower.py with PaintRule/PlaceRule dataclasses"
```

### Task 2.2: Add `normalize_paint_rules` + default + tests

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Write failing tests**

Append to `_lower_test.py`:

```python
from braincell.cv._lower import default_paint_rules, normalize_paint_rules
from braincell.filter import AllRegion, BranchSlice
from braincell.mech import Channel, Ion


class NormalizePaintRulesTest(unittest.TestCase):
    def test_default_paint_rules_has_one_cable_on_all_region(self) -> None:
        rules = default_paint_rules()
        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0].region, AllRegion)
        self.assertIsInstance(rules[0].mechanism, CableProperty)

    def test_normalize_rejects_non_region_expr(self) -> None:
        with self.assertRaises(TypeError):
            normalize_paint_rules(
                "not a region",  # type: ignore[arg-type]
                (Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV),),
            )

    def test_normalize_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_paint_rules(AllRegion(), ())

    def test_normalize_accepts_cable_and_density(self) -> None:
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        rules = normalize_paint_rules(AllRegion(), (cable, ch))
        self.assertEqual(len(rules), 2)
        self.assertIs(rules[0].mechanism, cable)
        self.assertIs(rules[1].mechanism, ch)

    def test_normalize_rejects_point_mechanism(self) -> None:
        from braincell.mech import CurrentClamp
        with self.assertRaises(TypeError):
            normalize_paint_rules(
                AllRegion(),
                (CurrentClamp.step(0.1 * u.nA, 10 * u.ms),),
            )

    def test_normalize_accepts_ion(self) -> None:
        rules = normalize_paint_rules(AllRegion(), (Ion("SodiumFixed"),))
        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0].mechanism, Ion)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::NormalizePaintRulesTest -v`
Expected: FAIL — `ImportError` for `default_paint_rules` / `normalize_paint_rules`.

- [ ] **Step 3: Implement in `_lower.py`**

Append to `_lower.py`:

```python
import brainunit as u

from braincell.filter import AllRegion

__all__ = [
    "PaintRule",
    "PlaceRule",
    "default_paint_rules",
    "normalize_paint_rules",
]

_DEFAULT_CABLE = CableProperty(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


def default_paint_rules() -> tuple[PaintRule, ...]:
    return (PaintRule(region=AllRegion(), mechanism=_DEFAULT_CABLE),)


def normalize_paint_rules(
    region: RegionExpr,
    mechanisms: tuple[object, ...],
) -> tuple[PaintRule, ...]:
    if not isinstance(region, RegionExpr):
        raise TypeError(
            f"Cell.paint(...) expects RegionExpr, got {type(region).__name__!s}."
        )
    if len(mechanisms) == 0:
        raise ValueError("Cell.paint(...) expects at least one mechanism.")

    rules: list[PaintRule] = []
    for mechanism in mechanisms:
        if isinstance(mechanism, (CableProperty, Density)):
            rules.append(PaintRule(region=region, mechanism=mechanism))
            continue
        raise TypeError(
            "Cell.paint(...) mechanisms must be CableProperty or Density "
            f"(use braincell.mech.Channel / Ion), got {type(mechanism).__name__!s}."
        )
    return tuple(rules)
```

Merge the earlier `__all__` with the one above — keep only this one.

- [ ] **Step 4: Run all lower tests, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): normalize_paint_rules + default_paint_rules"
```

### Task 2.3: Add `normalize_place_rule` + tests

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import normalize_place_rule


class NormalizePlaceRuleTest(unittest.TestCase):
    def test_rejects_non_locset(self) -> None:
        with self.assertRaises(TypeError):
            normalize_place_rule(
                "not a locset",  # type: ignore[arg-type]
                (CurrentClamp.step(0.1 * u.nA, 10 * u.ms),),
            )

    def test_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_place_rule(UniformSamples(n=1, seed=0), ())

    def test_rejects_non_point_mechanism(self) -> None:
        with self.assertRaises(TypeError):
            normalize_place_rule(
                UniformSamples(n=1, seed=0),
                (Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV),),
            )

    def test_returns_place_rule_with_site_mid(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        rule = normalize_place_rule(UniformSamples(n=1, seed=0), (clamp,))
        self.assertEqual(rule.site, "mid")
        self.assertEqual(rule.mechanisms, (clamp,))
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::NormalizePlaceRuleTest -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

Append to `_lower.py`:

```python
def normalize_place_rule(
    locset: LocsetExpr,
    mechanisms: tuple[object, ...],
) -> PlaceRule:
    if not isinstance(locset, LocsetExpr):
        raise TypeError(
            f"Cell.place(...) expects LocsetExpr, got {type(locset).__name__!s}."
        )
    if len(mechanisms) == 0:
        raise ValueError("Cell.place(...) expects at least one point mechanism.")

    normalized: list[Point] = []
    for mechanism in mechanisms:
        if not isinstance(mechanism, Point):
            raise TypeError(
                "Cell.place(...) mechanisms must be Point instances, "
                f"got {type(mechanism).__name__!s}."
            )
        normalized.append(mechanism)
    return PlaceRule(locset=locset, mechanisms=tuple(normalized), site="mid")
```

Add `"normalize_place_rule"` to `__all__`.

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): normalize_place_rule"
```

### Task 2.4: Add `merge_paint_rules` with last-wins density dedup

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import merge_paint_rules


class MergePaintRulesTest(unittest.TestCase):
    def _cable(self, cm=1.0, ra=100.0):
        return CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=cm * (u.uF / u.cm ** 2),
            axial_resistivity=ra * (u.ohm * u.cm),
        )

    def test_cable_same_region_replaces(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=self._cable(cm=1.0))
        r2 = PaintRule(region=AllRegion(), mechanism=self._cable(cm=2.0))
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].mechanism, r2.mechanism)

    def test_cable_different_regions_kept(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=self._cable(cm=1.0))
        r2 = PaintRule(
            region=BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            mechanism=self._cable(cm=2.0),
        )
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_same_region_and_class_replaces(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        d2 = Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].mechanism, d2)

    def test_density_different_classes_both_kept(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        d2 = Channel("INa_Ba2002", g_max=0.05 * (u.mS / u.cm ** 2), E=50 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_different_regions_both_kept(self) -> None:
        d = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d)
        r2 = PaintRule(region=BranchSlice(branch_index=0, prox=0.0, dist=1.0), mechanism=d)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::MergePaintRulesTest -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

Append to `_lower.py`:

```python
def _paint_key(rule: PaintRule) -> tuple[object, str, str]:
    """Key for last-wins dedup: (region, kind, class_name_if_density)."""
    if isinstance(rule.mechanism, CableProperty):
        return (rule.region, "cable", "")
    return (rule.region, "density", rule.mechanism.class_name)


def merge_paint_rules(
    existing: tuple[PaintRule, ...],
    incoming: tuple[PaintRule, ...],
) -> tuple[PaintRule, ...]:
    """Append incoming paint rules with last-wins dedup.

    Two rules collide when they share the same region AND the same kind:
    - CableProperty on the same region → incoming replaces existing.
    - Density with the same class_name on the same region → incoming
      replaces existing.

    Rules targeting different regions, or two Density rules for different
    channel classes, all survive.
    """
    merged: list[PaintRule] = list(existing)
    for rule in incoming:
        new_key = _paint_key(rule)
        merged = [item for item in merged if _paint_key(item) != new_key]
        merged.append(rule)
    return tuple(merged)
```

Add `"merge_paint_rules"` to `__all__`.

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): merge_paint_rules with last-wins density dedup"
```

### Task 2.5: Add `merge_place_rules`

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing test**

```python
from braincell.cv._lower import merge_place_rules


class MergePlaceRulesTest(unittest.TestCase):
    def test_exact_duplicate_dropped(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        r = normalize_place_rule(UniformSamples(n=1, seed=0), (clamp,))
        merged = merge_place_rules((r,), (r,))
        self.assertEqual(len(merged), 1)

    def test_different_rules_both_kept(self) -> None:
        clamp_a = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        clamp_b = CurrentClamp.step(0.2 * u.nA, 10 * u.ms)
        r1 = normalize_place_rule(UniformSamples(n=1, seed=0), (clamp_a,))
        r2 = normalize_place_rule(UniformSamples(n=1, seed=0), (clamp_b,))
        merged = merge_place_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::MergePlaceRulesTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `_lower.py`:

```python
def merge_place_rules(
    existing: tuple[PlaceRule, ...],
    incoming: tuple[PlaceRule, ...],
) -> tuple[PlaceRule, ...]:
    """Append incoming place rules, dropping exact duplicates."""
    merged: list[PlaceRule] = list(existing)
    for rule in incoming:
        if rule in merged:
            continue
        merged.append(rule)
    return tuple(merged)
```

Add `"merge_place_rules"` to `__all__`.

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): merge_place_rules with exact-duplicate dedup"
```

### Task 2.6: Add `_RegionCache`

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing test**

```python
from braincell.cv._lower import _RegionCache
from braincell import Branch, Morphology
from braincell.filter import AllRegion


class RegionCacheTest(unittest.TestCase):
    def _morpho(self) -> Morphology:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 3.0] * u.um, type="soma")
        return Morphology.from_root(soma, name="soma")

    def test_intervals_evaluated_once(self) -> None:
        morpho = self._morpho()
        region = AllRegion()
        calls = {"n": 0}
        original_evaluate = region.evaluate

        def counting_evaluate(morpho, cache=None):
            calls["n"] += 1
            return original_evaluate(morpho, cache)

        region.evaluate = counting_evaluate  # type: ignore[method-assign]
        cache = _RegionCache(morpho)
        a = cache.intervals(region)
        b = cache.intervals(region)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(a, b)
        self.assertIn(0, a)
        self.assertEqual(a[0], ((0.0, 1.0),))
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::RegionCacheTest -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

Append to `_lower.py`:

```python
from braincell.filter.cache import SelectionCache
from braincell.morph.morphology import Morphology


class _RegionCache:
    """Per-build cache of region / locset evaluation outputs.

    Keyed by ``id(expr)`` so even non-hashable exprs work. Sharing a
    single ``SelectionCache`` for morphology-derived intermediates
    (distance-to-root etc.) across all lookups.
    """

    def __init__(self, morpho: Morphology) -> None:
        self._morpho = morpho
        self._selection = SelectionCache()
        self._region_by_id: dict[int, dict[int, tuple[tuple[float, float], ...]]] = {}
        self._locset_by_id: dict[int, tuple[tuple[int, float, str], ...]] = {}

    def intervals(
        self, region: RegionExpr
    ) -> dict[int, tuple[tuple[float, float], ...]]:
        key = id(region)
        cached = self._region_by_id.get(key)
        if cached is not None:
            return cached
        mask = region.evaluate(self._morpho, self._selection)
        grouped: dict[int, list[tuple[float, float]]] = {}
        for branch_id, prox, dist in mask.intervals:
            grouped.setdefault(int(branch_id), []).append((float(prox), float(dist)))
        result = {bid: tuple(ranges) for bid, ranges in grouped.items()}
        self._region_by_id[key] = result
        return result

    def points(self, locset: LocsetExpr) -> tuple[tuple[int, float, str], ...]:
        key = id(locset)
        cached = self._locset_by_id.get(key)
        if cached is not None:
            return cached
        mask = locset.evaluate(self._morpho, self._selection)
        result = tuple(
            (int(branch), float(x), str(name))
            for (branch, x), name in zip(mask.points, mask.display_names)
        )
        self._locset_by_id[key] = result
        return result
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): _RegionCache for one-evaluate-per-region"
```

---

## Phase 3: Frustum math + geometry build

### Task 3.1: Typed epsilons + transient types in `_lower.py`

**Files:**
- Modify: `braincell/cv/_lower.py`

- [ ] **Step 1: Add constants and transient types**

Near the top of `_lower.py`, after imports:

```python
EPS_PARAM = 1e-9         # normalized x ∈ [0, 1]
EPS_LEN_UM = 1e-6        # μm lengths
EPS_AREA_UM2 = 1e-9      # μm² areas


@dataclass(frozen=True)
class _Frustum:
    prox: float
    dist: float
    length_um: float
    r_prox_um: float
    r_dist_um: float
    point_prox_um: "np.ndarray | None"
    point_dist_um: "np.ndarray | None"


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
    density_by_key: dict[tuple[object, str], Density]
    points: list[Point]
```

Also add `import numpy as np` at the top if not already present.

- [ ] **Step 2: Verify import of numpy**

Run: `python -c "from braincell.cv import _lower"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add braincell/cv/_lower.py
git commit -m "feat(cv): typed epsilons + transient types in _lower"
```

### Task 3.2: `_build_frusta` — strict clipping + ownership rule

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import _build_frusta


class BuildFrustaTest(unittest.TestCase):
    def _branch(self, lengths, radii):
        return Branch.from_lengths(
            lengths=lengths * u.um,
            radii=radii * u.um,
            type="dendrite",
        )

    def test_full_branch_single_segment(self) -> None:
        branch = self._branch([10.0], [2.0, 3.0])
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 10.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_half_branch_clips_length_and_interpolates_radius(self) -> None:
        branch = self._branch([10.0], [2.0, 4.0])
        frusta = _build_frusta(branch, prox=0.0, dist=0.5)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 5.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_multi_segment_branch(self) -> None:
        branch = self._branch([4.0, 6.0], [1.0, 2.0, 3.0])
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 2)
        self.assertAlmostEqual(frusta[0].length_um, 4.0)
        self.assertAlmostEqual(frusta[1].length_um, 6.0)

    def test_rejects_reversed_bounds(self) -> None:
        branch = self._branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.6, dist=0.4)

    def test_rejects_equal_bounds(self) -> None:
        branch = self._branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.5, dist=0.5)

    def test_rejects_out_of_range(self) -> None:
        branch = self._branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=-0.1, dist=0.5)
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.5, dist=1.1)

    def test_rejects_nonpositive_radius(self) -> None:
        branch = self._branch([10.0], [0.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.0, dist=1.0)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::BuildFrustaTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_build_frusta` and helpers**

Append to `_lower.py`:

```python
from braincell.morph.branch import Branch


def _build_frusta(
    branch: Branch,
    *,
    prox: float,
    dist: float,
) -> tuple[_Frustum, ...]:
    if not (0.0 - EPS_PARAM <= prox < dist - EPS_PARAM and dist <= 1.0 + EPS_PARAM):
        raise ValueError(
            f"CV bounds must satisfy 0 <= prox < dist <= 1, got {(prox, dist)!r}."
        )
    prox = max(0.0, min(1.0, float(prox)))
    dist = max(0.0, min(1.0, float(dist)))

    lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    radii_prox_um = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    radii_dist_um = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)

    if np.any(radii_prox_um <= 0.0) or np.any(radii_dist_um <= 0.0):
        raise ValueError(
            f"All branch radii must be > 0 (branch type={branch.type!r})."
        )

    total_length_um = float(np.sum(lengths_um))
    if total_length_um <= EPS_LEN_UM:
        raise ValueError(
            f"Branch total length must be > {EPS_LEN_UM} μm "
            f"(got {total_length_um} μm, type={branch.type!r})."
        )

    points_proximal = (
        np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
        if branch.points_proximal is not None
        else None
    )
    points_distal = (
        np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)
        if branch.points_distal is not None
        else None
    )

    start_um = prox * total_length_um
    end_um = dist * total_length_um
    segment_starts_um = np.concatenate(([0.0], np.cumsum(lengths_um)[:-1]))
    segment_ends_um = segment_starts_um + lengths_um

    frusta: list[_Frustum] = []
    for seg_idx, seg_length_um in enumerate(lengths_um):
        seg_start_um = float(segment_starts_um[seg_idx])
        seg_end_um = float(segment_ends_um[seg_idx])
        if seg_length_um <= EPS_LEN_UM:
            continue  # drop zero-length interior segments; validation handled overall length

        left_um = max(seg_start_um, start_um)
        right_um = min(seg_end_um, end_um)
        if right_um - left_um <= EPS_LEN_UM:
            continue

        t0 = (left_um - seg_start_um) / float(seg_length_um)
        t1 = (right_um - seg_start_um) / float(seg_length_um)
        r_seg_prox = float(radii_prox_um[seg_idx])
        r_seg_dist = float(radii_dist_um[seg_idx])
        r0_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t0
        r1_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t1

        x0 = max(prox, min(dist, left_um / total_length_um))
        x1 = max(prox, min(dist, right_um / total_length_um))
        point0 = None
        point1 = None
        if points_proximal is not None and points_distal is not None:
            p_prox = points_proximal[seg_idx]
            p_dist = points_distal[seg_idx]
            point0 = p_prox + (p_dist - p_prox) * t0
            point1 = p_prox + (p_dist - p_prox) * t1

        frusta.append(
            _Frustum(
                prox=float(x0),
                dist=float(x1),
                length_um=float(right_um - left_um),
                r_prox_um=r0_um,
                r_dist_um=r1_um,
                point_prox_um=point0,
                point_dist_um=point1,
            )
        )

    if len(frusta) == 0:
        raise ValueError(
            f"CV [{prox}, {dist}] produced no frusta on branch of length "
            f"{total_length_um} μm."
        )
    return tuple(frusta)
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py::BuildFrustaTest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): _build_frusta with strict clipping and validation"
```

### Task 3.3: Frustum scalar helpers (area, axial factor, midpoint radius, boundary radii)

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import (
    _axial_factor_per_cm,
    _boundary_radii_um,
    _lateral_area_um2,
    _midpoint_radius_um,
)


class FrustumScalarsTest(unittest.TestCase):
    def _single(self, length_um, r0, r1):
        return (
            _Frustum(
                prox=0.0, dist=1.0, length_um=length_um,
                r_prox_um=r0, r_dist_um=r1,
                point_prox_um=None, point_dist_um=None,
            ),
        )

    def test_lateral_area_cylinder(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)
        # cylinder: π·(r0+r1)·slant, slant=L because dr=0
        area = _lateral_area_um2(frusta)
        self.assertAlmostEqual(area, 40.0 * 3.141592653589793, places=6)

    def test_axial_factor_uniform(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)   # 10 μm = 1e-3 cm
        # factor = L/(π·r0·r1) in cm·cm⁻²·cm⁻¹ = cm⁻¹
        expected = 1e-3 / (3.141592653589793 * 2e-4 * 2e-4)
        self.assertAlmostEqual(_axial_factor_per_cm(frusta), expected, places=4)

    def test_midpoint_radius_uniform(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)
        self.assertAlmostEqual(_midpoint_radius_um(frusta), 2.0)

    def test_midpoint_radius_tapered(self) -> None:
        frusta = self._single(10.0, 2.0, 4.0)
        self.assertAlmostEqual(_midpoint_radius_um(frusta), 3.0)

    def test_boundary_radii(self) -> None:
        frusta = self._single(10.0, 2.0, 4.0)
        r0, r1 = _boundary_radii_um(frusta)
        self.assertAlmostEqual(r0, 2.0)
        self.assertAlmostEqual(r1, 4.0)

    def test_empty_frusta_raises(self) -> None:
        with self.assertRaises(ValueError):
            _boundary_radii_um(())
        with self.assertRaises(ValueError):
            _midpoint_radius_um(())
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::FrustumScalarsTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `_lower.py`:

```python
def _boundary_radii_um(frusta: tuple[_Frustum, ...]) -> tuple[float, float]:
    if len(frusta) == 0:
        raise ValueError("Cannot resolve boundary radii from empty frusta.")
    return frusta[0].r_prox_um, frusta[-1].r_dist_um


def _lateral_area_um2(frusta: tuple[_Frustum, ...]) -> float:
    total = 0.0
    pi = float(np.pi)
    for piece in frusta:
        slant = float(np.sqrt(piece.length_um ** 2 + (piece.r_dist_um - piece.r_prox_um) ** 2))
        total += pi * (piece.r_prox_um + piece.r_dist_um) * slant
    return total


def _axial_factor_per_cm(frusta: tuple[_Frustum, ...]) -> float:
    """Return Σ L_cm / (π · r_prox_cm · r_dist_cm) across the frusta."""
    factor = 0.0
    pi = float(np.pi)
    for piece in frusta:
        length_cm = piece.length_um * 1e-4
        r0_cm = piece.r_prox_um * 1e-4
        r1_cm = piece.r_dist_um * 1e-4
        if r0_cm <= 0.0 or r1_cm <= 0.0:
            raise ValueError(
                "Axial factor requires strictly positive radii; validation slipped."
            )
        factor += length_cm / (pi * r0_cm * r1_cm)
    return factor


def _midpoint_radius_um(frusta: tuple[_Frustum, ...]) -> float:
    if len(frusta) == 0:
        raise ValueError("Cannot resolve midpoint radius from empty frusta.")
    total_length_um = sum(piece.length_um for piece in frusta)
    if total_length_um <= EPS_LEN_UM:
        return 0.5 * (frusta[0].r_prox_um + frusta[-1].r_dist_um)
    target = 0.5 * total_length_um
    walked = 0.0
    for piece in frusta:
        next_walked = walked + piece.length_um
        if next_walked >= target - EPS_LEN_UM:
            if piece.length_um <= EPS_LEN_UM:
                return 0.5 * (piece.r_prox_um + piece.r_dist_um)
            ratio = max(0.0, min(1.0, (target - walked) / piece.length_um))
            return piece.r_prox_um + (piece.r_dist_um - piece.r_prox_um) * ratio
        walked = next_walked
    return frusta[-1].r_dist_um
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): frustum scalar helpers (area, axial factor, radii)"
```

### Task 3.4: `_split_frusta` for prox/dist axial-factor halves

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing test**

```python
from braincell.cv._lower import _split_frusta


class SplitFrustaTest(unittest.TestCase):
    def test_split_at_midpoint(self) -> None:
        frustum = _Frustum(
            prox=0.0, dist=1.0, length_um=10.0,
            r_prox_um=2.0, r_dist_um=4.0,
            point_prox_um=None, point_dist_um=None,
        )
        left, right = _split_frusta((frustum,), x=0.5)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertAlmostEqual(left[0].length_um, 5.0)
        self.assertAlmostEqual(right[0].length_um, 5.0)
        self.assertAlmostEqual(left[0].r_dist_um, 3.0)
        self.assertAlmostEqual(right[0].r_prox_um, 3.0)

    def test_split_at_boundary_puts_all_on_one_side(self) -> None:
        frustum = _Frustum(
            prox=0.0, dist=1.0, length_um=10.0,
            r_prox_um=2.0, r_dist_um=4.0,
            point_prox_um=None, point_dist_um=None,
        )
        left, right = _split_frusta((frustum,), x=1.0)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 0)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::SplitFrustaTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _split_frusta(
    frusta: tuple[_Frustum, ...],
    *,
    x: float,
) -> tuple[tuple[_Frustum, ...], tuple[_Frustum, ...]]:
    left: list[_Frustum] = []
    right: list[_Frustum] = []
    for piece in frusta:
        p0 = piece.prox
        p1 = piece.dist
        if p1 <= x + EPS_PARAM:
            left.append(piece)
            continue
        if p0 >= x - EPS_PARAM:
            right.append(piece)
            continue
        span = p1 - p0
        if span <= EPS_PARAM:
            continue
        ratio = max(0.0, min(1.0, (x - p0) / span))

        length_left = piece.length_um * ratio
        length_right = piece.length_um - length_left
        r_mid = piece.r_prox_um + (piece.r_dist_um - piece.r_prox_um) * ratio
        point_mid = None
        if piece.point_prox_um is not None and piece.point_dist_um is not None:
            point_mid = piece.point_prox_um + (piece.point_dist_um - piece.point_prox_um) * ratio

        if length_left > EPS_LEN_UM:
            left.append(_Frustum(
                prox=p0, dist=x, length_um=length_left,
                r_prox_um=piece.r_prox_um, r_dist_um=r_mid,
                point_prox_um=piece.point_prox_um, point_dist_um=point_mid,
            ))
        if length_right > EPS_LEN_UM:
            right.append(_Frustum(
                prox=x, dist=p1, length_um=length_right,
                r_prox_um=r_mid, r_dist_um=piece.r_dist_um,
                point_prox_um=point_mid, point_dist_um=piece.point_dist_um,
            ))
    return tuple(left), tuple(right)
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): _split_frusta for prox/dist axial halves"
```

### Task 3.5: `_build_geo` — per-branch CV geometry (no topology yet)

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import _build_geo


class BuildGeoTest(unittest.TestCase):
    def _morpho_single(self):
        soma = Branch.from_lengths(
            lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma"
        )
        return Morphology.from_root(soma, name="soma")

    def test_single_branch_one_cv(self) -> None:
        morpho = self._morpho_single()
        bounds = (((0.0, 1.0),),)
        geos, ids = _build_geo(morpho, bounds)
        self.assertEqual(len(geos), 1)
        self.assertEqual(ids, {0: (0,)})
        g = geos[0]
        self.assertEqual(g.id, 0)
        self.assertEqual(g.branch_id, 0)
        self.assertEqual(g.branch_type, "soma")
        self.assertAlmostEqual(g.prox, 0.0)
        self.assertAlmostEqual(g.dist, 1.0)
        self.assertIsNone(g.parent_cv)
        self.assertEqual(g.children_cv, ())
        self.assertAlmostEqual(g.length_um, 10.0)

    def test_single_branch_two_cvs_chain(self) -> None:
        morpho = self._morpho_single()
        bounds = (((0.0, 0.5), (0.5, 1.0)),)
        geos, ids = _build_geo(morpho, bounds)
        self.assertEqual(len(geos), 2)
        self.assertEqual(geos[0].parent_cv, None)
        self.assertEqual(geos[0].children_cv, (1,))
        self.assertEqual(geos[1].parent_cv, 0)
        self.assertEqual(geos[1].children_cv, ())
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::BuildGeoTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _build_geo(
    morpho: Morphology,
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[_GeoCV, ...], dict[int, tuple[int, ...]]]:
    cv_ids_by_branch: dict[int, tuple[int, ...]] = {}
    geos: list[_GeoCV] = []
    parent_by_cv: list[int | None] = []
    children_by_cv: list[list[int]] = []

    cv_id = 0
    for branch_id, branch in enumerate(morpho.branches):
        branch_bounds = bounds_by_branch[branch_id]
        ids: list[int] = []
        for prox, dist in branch_bounds:
            frusta = _build_frusta(branch, prox=float(prox), dist=float(dist))
            length_um = sum(p.length_um for p in frusta)
            area_um2 = _lateral_area_um2(frusta)
            factor_total = _axial_factor_per_cm(frusta)
            midpoint = 0.5 * (float(prox) + float(dist))
            left, right = _split_frusta(frusta, x=midpoint)
            factor_prox = _axial_factor_per_cm(left) if left else 0.0
            factor_dist = _axial_factor_per_cm(right) if right else 0.0
            r_prox, r_dist = _boundary_radii_um(frusta)
            r_mid = _midpoint_radius_um(frusta)

            geos.append(_GeoCV(
                id=cv_id,
                branch_id=branch_id,
                branch_type=branch.type,
                prox=float(prox),
                dist=float(dist),
                midpoint=midpoint,
                parent_cv=None,
                children_cv=(),
                length_um=length_um,
                lateral_area_um2=area_um2,
                axial_factor_total_per_cm=factor_total,
                axial_factor_prox_per_cm=factor_prox,
                axial_factor_dist_per_cm=factor_dist,
                r_prox_um=r_prox,
                r_mid_um=r_mid,
                r_dist_um=r_dist,
            ))
            parent_by_cv.append(None)
            children_by_cv.append([])
            ids.append(cv_id)
            cv_id += 1
        cv_ids_by_branch[branch_id] = tuple(ids)

    # Chain CVs on each branch.
    for ids in cv_ids_by_branch.values():
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            parent_by_cv[right_id] = left_id
            children_by_cv[left_id].append(right_id)

    # Wire branch-root CVs to parent branches in traversal order.
    for edge in morpho.edges:
        parent_ids = cv_ids_by_branch[edge.parent.index]
        child_ids = cv_ids_by_branch[edge.child.index]
        parent_cv = _locate_cv_on_branch(parent_ids, geos, x=float(edge.parent_x))
        child_cv = child_ids[0]   # branch-root CV is always the first
        if parent_by_cv[child_cv] is None:
            parent_by_cv[child_cv] = parent_cv
        if child_cv not in children_by_cv[parent_cv]:
            children_by_cv[parent_cv].append(child_cv)

    finalized = tuple(
        _GeoCV(
            id=geo.id,
            branch_id=geo.branch_id,
            branch_type=geo.branch_type,
            prox=geo.prox,
            dist=geo.dist,
            midpoint=geo.midpoint,
            parent_cv=parent_by_cv[geo.id],
            children_cv=tuple(children_by_cv[geo.id]),
            length_um=geo.length_um,
            lateral_area_um2=geo.lateral_area_um2,
            axial_factor_total_per_cm=geo.axial_factor_total_per_cm,
            axial_factor_prox_per_cm=geo.axial_factor_prox_per_cm,
            axial_factor_dist_per_cm=geo.axial_factor_dist_per_cm,
            r_prox_um=geo.r_prox_um,
            r_mid_um=geo.r_mid_um,
            r_dist_um=geo.r_dist_um,
        )
        for geo in geos
    )
    return finalized, cv_ids_by_branch


def _locate_cv_on_branch(
    ids: tuple[int, ...],
    geos: list[_GeoCV] | tuple[_GeoCV, ...],
    *,
    x: float,
) -> int:
    if x <= 0.0 + EPS_PARAM:
        return ids[0]
    if x >= 1.0 - EPS_PARAM:
        return ids[-1]
    for cv_id in ids:
        geo = geos[cv_id]
        if geo.prox - EPS_PARAM <= x < geo.dist - EPS_PARAM:
            return cv_id
    for cv_id in ids:
        geo = geos[cv_id]
        if abs(x - geo.dist) <= EPS_PARAM:
            return cv_id
    raise ValueError(
        f"x={x!r} not owned by any CV in branch; "
        f"bounds are {[(geos[i].prox, geos[i].dist) for i in ids]!r}."
    )
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): _build_geo + _locate_cv_on_branch"
```

### Task 3.6: Two-branch geo wiring test + fix if needed

**Files:**
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Add cross-branch wiring test**

```python
class BuildGeoCrossBranchTest(unittest.TestCase):
    def test_two_branch_parent_pointer(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 3.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        bounds = (((0.0, 0.5), (0.5, 1.0)), ((0.0, 1.0),))
        geos, ids = _build_geo(tree, bounds)
        self.assertEqual(len(geos), 3)
        # child branch root CV (index 2) parent is last CV on soma (index 1)
        self.assertEqual(geos[2].parent_cv, 1)
        self.assertIn(2, geos[1].children_cv)
```

- [ ] **Step 2: Run**

Run: `pytest braincell/cv/_lower_test.py::BuildGeoCrossBranchTest -v`
Expected: PASS (implementation from 3.5 should handle this).

- [ ] **Step 3: Commit**

```bash
git add braincell/cv/_lower_test.py
git commit -m "test(cv): cross-branch CV parent wiring"
```

---

## Phase 4: Mechanism lowering (paint + place)

### Task 4.1: Morpho + bounds + connectivity validators

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import (
    _validate_bounds,
    _validate_connectivity,
    _validate_morpho,
)


class ValidateMorphoTest(unittest.TestCase):
    def test_zero_length_branch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_morpho(
                Morphology.from_root(
                    Branch.from_lengths(lengths=[1e-12] * u.um, radii=[2.0, 2.0] * u.um, type="soma"),
                    name="soma",
                )
            )

    def test_nonpositive_radius_raises(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[0.0, 2.0] * u.um, type="soma")
        with self.assertRaises(ValueError):
            _validate_morpho(Morphology.from_root(soma, name="soma"))

    def test_valid_morpho_ok(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        _validate_morpho(Morphology.from_root(soma, name="soma"))   # does not raise


class ValidateBoundsTest(unittest.TestCase):
    def _morpho(self):
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        return Morphology.from_root(soma, name="soma")

    def test_valid_bounds(self) -> None:
        _validate_bounds((((0.0, 0.5), (0.5, 1.0)),), self._morpho())

    def test_gap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.4), (0.6, 1.0)),), self._morpho())

    def test_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.6), (0.5, 1.0)),), self._morpho())

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.1, 1.0),),), self._morpho())

    def test_missing_end_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.9),),), self._morpho())

    def test_empty_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds(((),), self._morpho())
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::ValidateMorphoTest braincell/cv/_lower_test.py::ValidateBoundsTest -v`
Expected: FAIL — imports missing.

- [ ] **Step 3: Implement**

```python
def _validate_morpho(morpho: Morphology) -> None:
    if not isinstance(morpho, Morphology):
        raise TypeError(f"Expected Morphology, got {type(morpho).__name__!s}.")
    for branch_id, branch in enumerate(morpho.branches):
        lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        if float(np.sum(lengths_um)) <= EPS_LEN_UM:
            raise ValueError(
                f"Branch {branch_id} (type={branch.type!r}) has total length "
                f"<= {EPS_LEN_UM} μm; morphology rejected."
            )
        radii_prox = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
        radii_dist = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
        if np.any(radii_prox <= 0.0) or np.any(radii_dist <= 0.0):
            raise ValueError(
                f"Branch {branch_id} (type={branch.type!r}) has non-positive radii; "
                "morphology rejected."
            )


def _validate_bounds(
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
    morpho: Morphology,
) -> None:
    if len(bounds_by_branch) != len(morpho.branches):
        raise ValueError(
            f"CV bounds length {len(bounds_by_branch)} does not match "
            f"branch count {len(morpho.branches)}."
        )
    for branch_id, branch_bounds in enumerate(bounds_by_branch):
        if len(branch_bounds) == 0:
            raise ValueError(f"Branch {branch_id} has no CV bounds.")
        prev_dist = 0.0
        for i, (prox, dist) in enumerate(branch_bounds):
            prox_f = float(prox)
            dist_f = float(dist)
            if not (0.0 - EPS_PARAM <= prox_f < dist_f - EPS_PARAM and dist_f <= 1.0 + EPS_PARAM):
                raise ValueError(
                    f"Branch {branch_id} CV {i} has invalid bounds "
                    f"(prox={prox_f}, dist={dist_f}); must satisfy 0 <= prox < dist <= 1."
                )
            if i == 0 and abs(prox_f - 0.0) > EPS_PARAM:
                raise ValueError(
                    f"Branch {branch_id} first CV must start at 0.0, got {prox_f}."
                )
            if i > 0 and abs(prox_f - prev_dist) > EPS_PARAM:
                raise ValueError(
                    f"Branch {branch_id} CV {i} prox={prox_f} does not meet "
                    f"previous dist={prev_dist} (overlap or gap)."
                )
            prev_dist = dist_f
        if abs(prev_dist - 1.0) > EPS_PARAM:
            raise ValueError(
                f"Branch {branch_id} last CV must end at 1.0, got {prev_dist}."
            )


def _validate_connectivity(
    geos: tuple[_GeoCV, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    morpho: Morphology,
) -> None:
    # 1. Every referenced id is a valid index.
    n = len(geos)
    for geo in geos:
        if geo.parent_cv is not None and not (0 <= geo.parent_cv < n):
            raise ValueError(
                f"CV {geo.id} has out-of-range parent_cv {geo.parent_cv}."
            )
        for child in geo.children_cv:
            if not (0 <= child < n):
                raise ValueError(
                    f"CV {geo.id} has out-of-range child_cv {child}."
                )
    # 2. Intra-branch chain.
    for branch_id, ids in cv_ids_by_branch.items():
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            if geos[right_id].parent_cv != left_id:
                raise ValueError(
                    f"CV {right_id} on branch {branch_id} expects parent {left_id}, "
                    f"got {geos[right_id].parent_cv}."
                )
    # 3. Root branch's first CV has no parent.
    root_ids = cv_ids_by_branch[morpho.root.branch.index]
    if geos[root_ids[0]].parent_cv is not None:
        raise ValueError(
            f"Root-branch first CV {root_ids[0]} must have parent_cv=None, "
            f"got {geos[root_ids[0]].parent_cv}."
        )
    # 4. DFS cycle detection.
    visited: set[int] = set()
    stack: list[int] = []
    for geo in geos:
        if geo.id in visited:
            continue
        path: set[int] = set()
        cursor = geo.id
        while cursor is not None and cursor not in visited:
            if cursor in path:
                raise ValueError(f"Cycle detected through CV {cursor}.")
            path.add(cursor)
            stack.append(cursor)
            cursor = geos[cursor].parent_cv
        visited.update(stack)
        stack.clear()
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): morpho/bounds/connectivity validators"
```

### Task 4.2: Coverage fraction helper + density application

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import _apply_density, _coverage_fraction


class CoverageFractionTest(unittest.TestCase):
    def _morpho(self):
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        return Morphology.from_root(soma, name="soma")

    def _geo(self, prox=0.0, dist=1.0):
        morpho = self._morpho()
        geos, _ = _build_geo(morpho, (((prox, dist),),))
        return morpho, geos[0]

    def test_full_overlap_fraction_one(self) -> None:
        morpho, geo = self._geo(prox=0.0, dist=1.0)
        f = _coverage_fraction(morpho, geo, ((0.0, 1.0),))
        self.assertAlmostEqual(f, 1.0)

    def test_half_overlap(self) -> None:
        morpho, geo = self._geo(prox=0.0, dist=1.0)
        f = _coverage_fraction(morpho, geo, ((0.0, 0.5),))
        self.assertAlmostEqual(f, 0.5, places=3)

    def test_zero_overlap(self) -> None:
        morpho, geo = self._geo(prox=0.0, dist=0.5)
        f = _coverage_fraction(morpho, geo, ((0.6, 1.0),))
        self.assertAlmostEqual(f, 0.0)


class ApplyDensityTest(unittest.TestCase):
    def test_channel_full_coverage_no_scaling(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(
            cable=_DEFAULT_CABLE, density_by_key={}, points=[],
        )
        _apply_density(bucket, ch, region_key=AllRegion(), fraction=1.0)
        self.assertEqual(len(bucket.density_by_key), 1)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 1.0)

    def test_channel_half_coverage_records_fraction(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(
            cable=_DEFAULT_CABLE, density_by_key={}, points=[],
        )
        _apply_density(bucket, ch, region_key=AllRegion(), fraction=0.5)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 0.5)

    def test_ion_ignores_coverage(self) -> None:
        ion = Ion("SodiumFixed")
        bucket = _MechBucket(
            cable=_DEFAULT_CABLE, density_by_key={}, points=[],
        )
        _apply_density(bucket, ion, region_key=AllRegion(), fraction=0.5)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 1.0)

    def test_same_key_replaces(self) -> None:
        c1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        c2 = Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(
            cable=_DEFAULT_CABLE, density_by_key={}, points=[],
        )
        region = AllRegion()
        _apply_density(bucket, c1, region_key=region, fraction=1.0)
        _apply_density(bucket, c2, region_key=region, fraction=1.0)
        self.assertEqual(len(bucket.density_by_key), 1)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(
            stored.params["g_max"],
            0.2 * (u.mS / u.cm ** 2),
        )
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py -v -k "CoverageFraction or ApplyDensity"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _coverage_fraction(
    morpho: Morphology,
    geo: _GeoCV,
    intervals: tuple[tuple[float, float], ...],
) -> float:
    if geo.lateral_area_um2 <= EPS_AREA_UM2:
        return 0.0
    branch = morpho.branches[geo.branch_id]
    overlap = 0.0
    for left, right in intervals:
        start = max(geo.prox, float(left))
        end = min(geo.dist, float(right))
        if end - start <= EPS_PARAM:
            continue
        overlap += _lateral_area_um2(
            _build_frusta(branch, prox=start, dist=end)
        )
    return max(0.0, min(1.0, overlap / geo.lateral_area_um2))


def _apply_density(
    bucket: _MechBucket,
    mechanism: Density,
    *,
    region_key: object,
    fraction: float,
) -> None:
    key = (id(region_key), mechanism.class_name)
    if mechanism.category != "channel":
        bucket.density_by_key[key] = mechanism
        return
    if fraction >= 1.0 - EPS_PARAM:
        bucket.density_by_key[key] = mechanism
    else:
        bucket.density_by_key[key] = mechanism.with_coverage(fraction)
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): coverage_fraction + apply_density with single repr"
```

### Task 4.3: Point-mechanism name resolution + collision raise

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing tests**

```python
from braincell.cv._lower import _apply_place, _resolve_point_name
from braincell.mech import CurrentClamp, StateProbe


class ResolvePointNameTest(unittest.TestCase):
    def test_state_probe_auto_name(self) -> None:
        probe = StateProbe(field="V")
        named = _resolve_point_name(probe, display_name="loc_0")
        self.assertEqual(named.name, "loc_0_V")

    def test_state_probe_keeps_explicit_name(self) -> None:
        probe = StateProbe(field="V", name="my_probe")
        named = _resolve_point_name(probe, display_name="loc_0")
        self.assertEqual(named.name, "my_probe")

    def test_clamp_untouched_when_no_auto_name(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        named = _resolve_point_name(clamp, display_name="loc_0")
        self.assertIs(named, clamp)


class ApplyPlaceCollisionTest(unittest.TestCase):
    def test_duplicate_name_raises(self) -> None:
        seen: set[str] = set()
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        probe_a = StateProbe(field="V", name="dup")
        probe_b = StateProbe(field="V", name="dup")
        _apply_place(bucket, probe_a, display_name="loc_0", seen_names=seen)
        with self.assertRaises(ValueError):
            _apply_place(bucket, probe_b, display_name="loc_1", seen_names=seen)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py -v -k "ResolvePointName or ApplyPlaceCollision"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from dataclasses import replace

from braincell.mech import CurrentProbe, MechanismProbe, StateProbe


def _resolve_point_name(mechanism: Point, *, display_name: str) -> Point:
    if isinstance(mechanism, StateProbe):
        if mechanism.name is not None:
            return mechanism
        return replace(mechanism, name=f"{display_name}_{mechanism.field}")
    if isinstance(mechanism, MechanismProbe):
        if mechanism.name is not None:
            return mechanism
        return replace(
            mechanism,
            name=f"{display_name}_{mechanism.mechanism}_{mechanism.field}",
        )
    if isinstance(mechanism, CurrentProbe):
        if mechanism.name is not None:
            return mechanism
        suffix = (
            f"{mechanism.mechanism}_current"
            if mechanism.mechanism is not None
            else f"{mechanism.ion}_current"
        )
        return replace(mechanism, name=f"{display_name}_{suffix}")
    return mechanism


def _apply_place(
    bucket: _MechBucket,
    mechanism: Point,
    *,
    display_name: str,
    seen_names: set[str],
) -> None:
    named = _resolve_point_name(mechanism, display_name=display_name)
    candidate_name = getattr(named, "name", None)
    if candidate_name is not None:
        if candidate_name in seen_names:
            raise ValueError(
                f"Duplicate point-mechanism name {candidate_name!r} from place "
                f"rule at {display_name!r}. Supply an explicit name= argument "
                "to disambiguate."
            )
        seen_names.add(candidate_name)
    bucket.points.append(named)
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): point-mechanism name uniqueness enforcement"
```

### Task 4.4: `_build_mech` — full paint/place application

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing test**

```python
from braincell.cv._lower import _build_mech


class BuildMechTest(unittest.TestCase):
    def test_paint_cable_and_channel_on_all_region(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        geos, ids = _build_geo(morpho, (((0.0, 1.0),),))
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        paint = (
            PaintRule(region=AllRegion(), mechanism=cable),
            PaintRule(region=AllRegion(), mechanism=ch),
        )
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho, geos, ids,
            paint_rules=paint, place_rules=(), cache=cache,
        )
        self.assertEqual(len(buckets), 1)
        self.assertEqual(
            buckets[0].cable.membrane_capacitance,
            2.0 * (u.uF / u.cm ** 2),
        )
        self.assertEqual(len(buckets[0].density_by_key), 1)

    def test_place_clamp_attaches_to_one_cv(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        geos, ids = _build_geo(morpho, (((0.0, 0.5), (0.5, 1.0)),))
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        place = (PlaceRule(
            locset=BranchPoints(branch_index=0, x=(0.25,)),
            mechanisms=(clamp,),
        ),)
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho, geos, ids,
            paint_rules=(), place_rules=place, cache=cache,
        )
        totals = [len(b.points) for b in buckets]
        self.assertEqual(totals, [1, 0])
```

Add `from braincell.filter import BranchPoints` near the top of the test file if not imported yet.

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::BuildMechTest -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _init_bucket() -> _MechBucket:
    return _MechBucket(
        cable=_DEFAULT_CABLE,
        density_by_key={},
        points=[],
    )


def _build_mech(
    morpho: Morphology,
    geos: tuple[_GeoCV, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    *,
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
    cache: _RegionCache,
) -> list[_MechBucket]:
    buckets = [_init_bucket() for _ in geos]

    for rule in paint_rules:
        intervals_by_branch = cache.intervals(rule.region)
        mechanism = rule.mechanism

        for branch_id, cv_ids in cv_ids_by_branch.items():
            intervals = intervals_by_branch.get(branch_id, ())
            if len(intervals) == 0:
                continue
            for cv_id in cv_ids:
                geo = geos[cv_id]
                bucket = buckets[cv_id]

                if isinstance(mechanism, CableProperty):
                    if not _interval_contains(
                        intervals, geo.midpoint, epsilon=EPS_PARAM
                    ):
                        continue
                    bucket.cable = mechanism
                    continue

                fraction = _coverage_fraction(morpho, geo, intervals)
                if fraction <= EPS_PARAM:
                    continue
                _apply_density(
                    bucket, mechanism,
                    region_key=rule.region, fraction=fraction,
                )

    seen_names: set[str] = set()
    for rule in place_rules:
        if rule.site != "mid":
            raise ValueError(
                f"Unsupported place site {rule.site!r}; only 'mid' is allowed."
            )
        for branch_id, x, display_name in cache.points(rule.locset):
            ids = cv_ids_by_branch.get(branch_id)
            if not ids:
                continue
            cv_id = _locate_cv_on_branch(ids, geos, x=x)
            for mechanism in rule.mechanisms:
                _apply_place(
                    buckets[cv_id], mechanism,
                    display_name=display_name, seen_names=seen_names,
                )

    return buckets


def _interval_contains(
    intervals: tuple[tuple[float, float], ...],
    x: float,
    *,
    epsilon: float,
) -> bool:
    for left, right in intervals:
        if left - epsilon <= x <= right + epsilon:
            return True
    return False
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_lower_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): _build_mech with coverage + name-unique place"
```

### Task 4.5: `_validate_names` — final post-build global sweep

**Files:**
- Modify: `braincell/cv/_lower.py`

- [ ] **Step 1: Add validator**

```python
def _validate_names(buckets: list[_MechBucket]) -> None:
    seen: set[str] = set()
    for bucket in buckets:
        for point in bucket.points:
            name = getattr(point, "name", None)
            if name is None:
                continue
            if name in seen:
                raise ValueError(
                    f"Duplicate point-mechanism name {name!r} across CVs."
                )
            seen.add(name)
```

(`_apply_place` already catches collisions live; this is belt-and-suspenders for unnamed paths that later grow names.)

- [ ] **Step 2: Commit**

```bash
git add braincell/cv/_lower.py
git commit -m "feat(cv): _validate_names final global check"
```

---

## Phase 5: `lower()` composer

### Task 5.1: Wire it all together

**Files:**
- Modify: `braincell/cv/_lower.py`
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Failing test**

```python
from braincell.cv._lower import lower


class LowerSmokeTest(unittest.TestCase):
    def test_single_branch_default_cable(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        from braincell.cv._policy import CVPerBranch
        cvs = lower(
            morpho,
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertEqual(len(cvs), 2)
        self.assertEqual(cvs[0].id, 0)
        self.assertEqual(cvs[0].branch_id, 0)
        self.assertEqual(cvs[1].parent_cv, 0)
        self.assertAlmostEqual(float(cvs[0].length.to_decimal(u.um)), 5.0)

    def test_rejects_invalid_policy_bounds(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")

        class BadPolicy(CVPolicy):
            def resolve_cv_bounds(self, morpho, *, paint_rules=None):
                return (((0.0, 0.5),),)  # missing 0.5..1.0

        from braincell.cv._policy import CVPolicy
        with self.assertRaises(ValueError):
            lower(
                morpho,
                policy=BadPolicy(),
                paint_rules=(),
                place_rules=(),
            )
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_lower_test.py::LowerSmokeTest -v`
Expected: FAIL — `lower` and `CV` assembly not wired yet.

- [ ] **Step 3: Implement `_assemble` + `lower`**

```python
from braincell.cv._cv import CV   # forward import resolved after Phase 6


def _assemble(geo: _GeoCV, bucket: _MechBucket) -> CV:
    cable = bucket.cable
    ra = cable.axial_resistivity
    ra_ohm_cm = float(np.asarray(ra.to_decimal(u.ohm * u.cm), dtype=float))
    return CV(
        id=geo.id,
        branch_id=geo.branch_id,
        branch_type=geo.branch_type,
        prox=geo.prox,
        dist=geo.dist,
        parent_cv=geo.parent_cv,
        children_cv=geo.children_cv,
        length=u.Quantity(geo.length_um, u.um),
        area=u.Quantity(geo.lateral_area_um2, u.um ** 2),
        cm=cable.membrane_capacitance,
        ra=cable.axial_resistivity,
        v=cable.resting_potential,
        temp=cable.temperature,
        r_axial=u.Quantity(ra_ohm_cm * geo.axial_factor_total_per_cm, u.ohm),
        r_axial_prox=u.Quantity(ra_ohm_cm * geo.axial_factor_prox_per_cm, u.ohm),
        r_axial_dist=u.Quantity(ra_ohm_cm * geo.axial_factor_dist_per_cm, u.ohm),
        radius_prox=u.Quantity(geo.r_prox_um, u.um),
        radius_mid=u.Quantity(geo.r_mid_um, u.um),
        radius_dist=u.Quantity(geo.r_dist_um, u.um),
        density_mech=tuple(bucket.density_by_key.values()),
        point_mech=tuple(bucket.points),
    )


def lower(
    morpho: Morphology,
    *,
    policy: "CVPolicy",
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
) -> tuple[CV, ...]:
    _validate_morpho(morpho)
    cache = _RegionCache(morpho)
    from braincell.cv._policy import CVPolicy
    if not isinstance(policy, CVPolicy):
        raise TypeError(
            f"lower(...) expects a CVPolicy, got {type(policy).__name__!s}."
        )
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

Add `lower` to `__all__`.

Note: `_assemble` imports `CV` from `_cv.py`. `_cv.py` will be created in Phase 6. If `_cv.py` doesn't exist yet, this task's `_lower_test.py` will fail on import; that's expected — Phase 6 follows immediately.

- [ ] **Step 4: Skip running until Phase 6 completes**

Note in the implementation plan that tests depending on `CV` are gated by Phase 6.

- [ ] **Step 5: Commit (wip — won't import until Phase 6)**

```bash
git add braincell/cv/_lower.py braincell/cv/_lower_test.py
git commit -m "feat(cv): lower() composer (forward-imports CV from _cv)"
```

---

## Phase 6: New `_cv.py` with `CV` dataclass + `build_cvs`

### Task 6.1: Backup and rewrite `_cv.py`

**Files:**
- Rewrite: `braincell/cv/_cv.py`
- Create: `braincell/cv/_cv_test.py`

- [ ] **Step 1: Failing test**

Create `braincell/cv/_cv_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import brainunit as u

from braincell import Branch, Morphology
from braincell.cv import CV, CVPerBranch
from braincell.cv._cv import build_cvs
from braincell.cv._lower import default_paint_rules
from braincell.filter import RegionMask


class CVShapeTest(unittest.TestCase):
    def test_cv_fields_are_frozen(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertEqual(len(cvs), 1)
        cv = cvs[0]
        with self.assertRaises(Exception):
            cv.id = 5  # type: ignore[misc]

    def test_cv_region_property(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertIsInstance(cvs[0].region, RegionMask)
        self.assertEqual(cvs[0].region.intervals, ((0, 0.0, 0.5),))

    def test_precomputed_radii(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 4.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        cv = cvs[0]
        self.assertAlmostEqual(float(cv.radius_prox.to_decimal(u.um)), 2.0)
        self.assertAlmostEqual(float(cv.radius_mid.to_decimal(u.um)), 3.0)
        self.assertAlmostEqual(float(cv.radius_dist.to_decimal(u.um)), 4.0)
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_cv_test.py -v`
Expected: FAIL — `CV` construction with new fields not implemented.

- [ ] **Step 3: Rewrite `braincell/cv/_cv.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (... license header ...)

"""User-facing control-volume records and build entry point."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainunit as u

from braincell.filter import RegionMask
from braincell.mech import Density, Point

if TYPE_CHECKING:
    from braincell.cv._policy import CVPolicy
    from braincell.cv._lower import PaintRule, PlaceRule
    from braincell.morph.morphology import Morphology


@dataclass(frozen=True)
class CV:
    """Immutable per-control-volume record exposed to users.

    Geometry, cable properties, and attached mechanisms are all frozen
    into this dataclass by :func:`build_cvs`. CVs carry no references
    back to their source morphology or rules — any post-build analysis
    must re-derive from morpho plus the CV's ``(branch_id, prox, dist)``
    range.
    """

    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length: u.Quantity
    area: u.Quantity
    cm: u.Quantity
    ra: u.Quantity
    v: u.Quantity
    temp: u.Quantity
    r_axial: u.Quantity
    r_axial_prox: u.Quantity
    r_axial_dist: u.Quantity
    radius_prox: u.Quantity
    radius_mid: u.Quantity
    radius_dist: u.Quantity
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]

    @property
    def region(self) -> RegionMask:
        return RegionMask(((self.branch_id, self.prox, self.dist),))


def build_cvs(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]" = (),
    place_rules: "tuple[PlaceRule, ...]" = (),
) -> tuple[CV, ...]:
    """Lower a morphology + policy + rules into a frozen ``tuple[CV, ...]``."""
    from braincell.cv._lower import lower
    return lower(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )
```

Note: `braincell/cv/__init__.py` currently exports `CV` from `_cv`; no change needed there.

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_cv_test.py braincell/cv/_lower_test.py braincell/cv/_policy_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_cv.py braincell/cv/_cv_test.py
git commit -m "feat(cv): new CV dataclass + build_cvs entry"
```

---

## Phase 7: `_debug.py`

### Task 7.1: `cv_to_branch`

**Files:**
- Create: `braincell/cv/_debug.py`
- Create: `braincell/cv/_debug_test.py`

- [ ] **Step 1: Failing test**

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import brainunit as u

from braincell import Branch, Morphology
from braincell.cv import CVPerBranch
from braincell.cv._cv import build_cvs
from braincell.cv._debug import cv_to_branch
from braincell.cv._lower import default_paint_rules


class CVToBranchTest(unittest.TestCase):
    def test_roundtrip_length_matches(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 4.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_cvs(
            morpho, policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(), place_rules=(),
        )
        branch = cv_to_branch(cvs[0], morpho)
        self.assertEqual(branch.type, "soma")
        self.assertAlmostEqual(
            float(branch.length.to_decimal(u.um)),
            float(cvs[0].length.to_decimal(u.um)),
            places=6,
        )
```

- [ ] **Step 2: Run, see fail**

Run: `pytest braincell/cv/_debug_test.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `braincell/cv/_debug.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Debug / visualization helpers that reconstruct geometry from a CV."""

import brainunit as u
import numpy as np

from braincell.cv._cv import CV
from braincell.cv._lower import _build_frusta
from braincell.morph.branch import Branch, branch_class_for_type
from braincell.morph.morphology import Morphology


def cv_to_branch(cv: CV, morpho: Morphology) -> Branch:
    """Reconstruct a standalone ``Branch`` for a CV range.

    Slices ``morpho.branches[cv.branch_id]`` from ``cv.prox`` to ``cv.dist``
    using the same frustum math as :func:`braincell.cv._lower._build_frusta`,
    then returns a typed subclass chosen via
    :func:`braincell.morph.branch.branch_class_for_type`.
    """
    source = morpho.branches[cv.branch_id]
    frusta = _build_frusta(source, prox=cv.prox, dist=cv.dist)
    lengths = np.asarray([p.length_um for p in frusta], dtype=float)
    r0 = np.asarray([p.r_prox_um for p in frusta], dtype=float)
    r1 = np.asarray([p.r_dist_um for p in frusta], dtype=float)

    has_points = all(
        p.point_prox_um is not None and p.point_dist_um is not None for p in frusta
    )
    branch_cls = branch_class_for_type(cv.branch_type)
    if has_points:
        first = frusta[0].point_prox_um
        points = [np.asarray(first, dtype=float)]
        for p in frusta:
            points.append(np.asarray(p.point_dist_um, dtype=float))
        radii = np.concatenate((r0[:1], r1), axis=0)
        return branch_cls.from_points(
            points=u.Quantity(np.asarray(points, dtype=float), u.um),
            radii=u.Quantity(radii, u.um),
        )
    return branch_cls.from_lengths(
        lengths=u.Quantity(lengths, u.um),
        radii_proximal=u.Quantity(r0, u.um),
        radii_distal=u.Quantity(r1, u.um),
    )
```

- [ ] **Step 4: Run, see pass**

Run: `pytest braincell/cv/_debug_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/cv/_debug.py braincell/cv/_debug_test.py
git commit -m "feat(cv): cv_to_branch debug reconstruction (replaces as_branch)"
```

---

## Phase 8: Migrate downstream consumers

### Task 8.1: Update `_multi_compartment/cell.py` imports

**Files:**
- Modify: `braincell/_multi_compartment/cell.py`

- [ ] **Step 1: Replace imports**

Replace lines 15–29 (current imports block) with:

```python
from braincell.cv import CV, CVPerBranch, CVPolicy
from braincell.cv._cv import build_cvs
from braincell.cv._lower import (
    PaintRule,
    PlaceRule,
    default_paint_rules,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
```

- [ ] **Step 2: Replace the body of `Cell.cvs` property**

Find:

```python
        cv_geo, cv_ids_by_branch = build_cv_geo(
            self._morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
        )
        cv_mech = init_cv_mech(len(cv_geo))
        apply_paint_rules(...)
        apply_place_rules(...)
        cvs = tuple(
            assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
        )
```

Replace with:

```python
        cvs = build_cvs(
            self._morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
            place_rules=self._place_rules,
        )
```

- [ ] **Step 3: Run multi-compartment tests**

Run: `pytest braincell/_multi_compartment/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add braincell/_multi_compartment/cell.py
git commit -m "refactor(multi_compartment): cell.py uses build_cvs"
```

### Task 8.2: Update `_multi_compartment/build.py` imports

**Files:**
- Modify: `braincell/_multi_compartment/build.py`

- [ ] **Step 1: Replace imports**

Replace lines 26–33 (imports block for `braincell.cv.*`) with:

```python
from braincell.cv._cv import build_cvs
```

- [ ] **Step 2: Replace the geo+mech chunk**

Find the block in `build(cell)`:

```python
    cv_geo, cv_ids_by_branch = build_cv_geo(...)
    cv_mech = init_cv_mech(len(cv_geo))
    apply_paint_rules(...)
    apply_place_rules(...)
    cvs = tuple(
        assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
    )
```

Replace with:

```python
    cvs = build_cvs(
        morpho,
        policy=cell.cv_policy,
        paint_rules=cell.paint_rules,
        place_rules=cell.place_rules,
    )
```

- [ ] **Step 3: Run**

Run: `pytest braincell/_multi_compartment/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add braincell/_multi_compartment/build.py
git commit -m "refactor(multi_compartment): build.py uses build_cvs"
```

### Task 8.3: Fix `compute/_point_tree.py` EPSILON import

**Files:**
- Modify: `braincell/compute/_point_tree.py`

- [ ] **Step 1: Check current usage**

Run:
```bash
grep -n "EPSILON\|from braincell.cv" braincell/compute/_point_tree.py
```
Expected: shows lines 21 (`CV`) and 22 (`EPSILON`) imports.

- [ ] **Step 2: Inline the constant**

Replace line 22:
```python
from braincell.cv._geo import EPSILON
```
with:
```python
_EPS_PARAM = 1e-9   # normalized-x tolerance (formerly imported from cv._geo)
```

Then replace every `EPSILON` usage in the file with `_EPS_PARAM`. Use grep to find uses:
```bash
grep -n "EPSILON" braincell/compute/_point_tree.py
```
Edit accordingly.

- [ ] **Step 3: Also update import to go through public API**

Replace line 21:
```python
from braincell.cv._cv import CV
```
with:
```python
from braincell.cv import CV
```

- [ ] **Step 4: Run compute tests**

Run: `pytest braincell/compute/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/compute/_point_tree.py
git commit -m "refactor(compute): inline EPSILON, use public CV import"
```

---

## Phase 9: Delete old modules, finalize `__init__.py`

### Task 9.1: Delete `_geo.py` and `_mech.py`

**Files:**
- Delete: `braincell/cv/_geo.py`
- Delete: `braincell/cv/_mech.py`

- [ ] **Step 1: Final grep for residual references**

```bash
grep -rn "from braincell.cv._geo\|from braincell.cv._mech\|cv._geo\|cv._mech" braincell/ examples/ docs/ 2>/dev/null
```
Expected: no hits in `braincell/` or `examples/`. Hits only in spec/plan docs (accepted).

- [ ] **Step 2: Delete**

```bash
git rm braincell/cv/_geo.py braincell/cv/_mech.py
```

- [ ] **Step 3: Run full module test**

Run: `pytest braincell/cv/ braincell/_multi_compartment/ braincell/compute/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(cv): remove legacy _geo.py and _mech.py"
```

### Task 9.2: Confirm `__init__.py` surface unchanged

**Files:**
- Verify: `braincell/cv/__init__.py`

- [ ] **Step 1: Diff against design**

```bash
cat braincell/cv/__init__.py
```
Expected: exports `CV`, `CompositeByTypePolicy`, `CVPerBranch`, `CVPolicy`, `CVPolicyByTypeRule`, `DLambda`, `MaxCVLen`. If any new export slipped in, remove.

- [ ] **Step 2: Update module docstring**

Replace the current docstring with a short note:
```python
"""Control-volume (CV) layer: geometry, mechanism rules, and policies.

The package resolves a :class:`braincell.morph.Morphology` together with a
:class:`CVPolicy` and paint / place rules into an immutable
``tuple[braincell.cv.CV, ...]`` via the internal ``build_cvs`` entry
point. Policies live in :mod:`braincell.cv._policy`; lowering lives in
:mod:`braincell.cv._lower`; the final data class lives in
:mod:`braincell.cv._cv`.
"""
```

- [ ] **Step 3: Commit (only if changed)**

```bash
git add braincell/cv/__init__.py
git commit -m "docs(cv): update __init__.py docstring"
```

---

## Phase 10: Property-based tests + full suite sweep

### Task 10.1: Property-based coverage invariants

**Files:**
- Modify: `braincell/cv/_lower_test.py`

- [ ] **Step 1: Add hypothesis tests**

Append to `_lower_test.py`:

```python
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False


@unittest.skipUnless(_HAS_HYPOTHESIS, "hypothesis not installed")
class LowerPropertyTest(unittest.TestCase):

    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_cv_lengths_sum_to_branch_total(self, cv_count: int) -> None:
        soma = Branch.from_lengths(
            lengths=[30.0] * u.um, radii=[3.0, 3.0] * u.um, type="soma"
        )
        morpho = Morphology.from_root(soma, name="soma")
        from braincell.cv._policy import CVPerBranch
        cvs = lower(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        total_um = sum(float(cv.length.to_decimal(u.um)) for cv in cvs)
        self.assertAlmostEqual(total_um, 30.0, places=4)

    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_coverage_sum_of_all_region_is_one(self, cv_count: int) -> None:
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um, radii=[2.0, 2.0] * u.um, type="soma"
        )
        morpho = Morphology.from_root(soma, name="soma")
        from braincell.cv._policy import CVPerBranch
        from braincell.cv._lower import _build_geo
        bounds = CVPerBranch(cv_per_branch=cv_count).resolve_cv_bounds(morpho)
        geos, _ = _build_geo(morpho, bounds)
        total_frac = sum(
            _coverage_fraction(morpho, geo, ((0.0, 1.0),)) for geo in geos
        )
        self.assertAlmostEqual(total_frac, 1.0, places=3)

    @given(cv_count=st.integers(min_value=1, max_value=6))
    @settings(max_examples=25, deadline=None)
    def test_each_cv_id_appears_once_as_child_or_root(self, cv_count: int) -> None:
        soma = Branch.from_lengths(
            lengths=[30.0] * u.um, radii=[3.0, 3.0] * u.um, type="soma"
        )
        dend = Branch.from_lengths(
            lengths=[20.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.d = dend
        from braincell.cv._policy import CVPerBranch
        cvs = lower(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        roots = [cv.id for cv in cvs if cv.parent_cv is None]
        children = [cid for cv in cvs for cid in cv.children_cv]
        self.assertEqual(sorted(roots + children), list(range(len(cvs))))
```

- [ ] **Step 2: Run**

Run: `pytest braincell/cv/_lower_test.py -v -k Property`
Expected: PASS (or skipped if hypothesis not installed).

- [ ] **Step 3: Commit**

```bash
git add braincell/cv/_lower_test.py
git commit -m "test(cv): property-based invariants for lower()"
```

### Task 10.2: Final green sweep + baseline comparison

**Files:**
- None (read-only)

- [ ] **Step 1: Run the full braincell suite**

Run: `pytest braincell/ -q`
Expected: PASS, count ≥ baseline from Task 0.1.

- [ ] **Step 2: Run the relevant subdirs with verbose output to confirm new tests discovered**

Run: `pytest braincell/cv/ -v --collect-only | tail -40`
Expected: includes `_cv_test.py`, `_lower_test.py`, `_policy_test.py`, `_debug_test.py`.

- [ ] **Step 3: Run pre-commit**

Run: `pre-commit run --all`
Expected: PASS.

- [ ] **Step 4: Final commit (if pre-commit made any whitespace / EOL fixes)**

```bash
git add -A
git diff --cached --quiet || git commit -m "chore: pre-commit whitespace fixes"
```

---

## Self-review checklist

Verified before handing off:

1. **Spec coverage:**
   - Bug #1 (swallowed scaling exceptions): Task 4.2 implements single `coverage_area_fraction` path with no try/except.
   - Bug #2 (mixed-dimension EPSILON): Task 1.1 + Task 3.1 split into `EPS_PARAM`, `EPS_LEN_UM`, `EPS_AREA_UM2`.
   - Bug #3 (zero-length segment boundary ownership): Task 3.2 drops interior zero-length segments; single ownership rule in `_locate_cv_on_branch` (Task 3.5).
   - Bug #4 (silent snap to `ids[-1]`): Task 3.5 `_locate_cv_on_branch` raises on no-owner instead of snapping.
   - Bug #5 (zero-length branches): Task 4.1 `_validate_morpho` raises.
   - Bug #6 (duplicate density paint): Task 2.4 `merge_paint_rules` last-wins.
   - Bug #7 (radii ≤ 0 → inf): Task 3.2 raises in `_build_frusta`; Task 4.1 raises in `_validate_morpho`.
   - Bug #8 (no post-build consistency check): Task 4.1 `_validate_connectivity`.
   - Bug #9 (sorted children): Task 3.5 `_build_geo` appends in traversal order, no sort.
   - Bug #10 (stale DLambda docstring): Task 1.1.
   - Bug #11 (place-rule name collisions): Tasks 4.3 + 4.5.
   - Bug #12 (“expects Morpho”): Task 4.1 error messages reference `Morphology`.
   - Architectural goal: public surface unchanged — Task 9.2.
   - Architectural goal: one functional lowering — Task 5.1.
   - Architectural goal: co-located tests — Tasks 2.*, 3.*, 4.*, 5.*, 6.*, 7.*, 10.1.
   - Migration goal: single `build_cvs(...)` used by `_multi_compartment` — Tasks 8.1, 8.2.
   - `compute/_point_tree.py` EPSILON import fix — Task 8.3.

2. **Placeholder scan:** no `TBD` / `TODO` / vague “handle errors”. Every step shows code or exact commands.

3. **Type consistency:**
   - `_Frustum.length_um`, `_GeoCV.length_um`, `_MechBucket` fields — used consistently from Task 3.1 onward.
   - `CV.length` is `Quantity` (μm), confirmed in Task 5.1 `_assemble` and Task 6.1 tests.
   - `build_cvs(morpho, *, policy, paint_rules, place_rules)` signature matches between Task 5.1, Task 6.1, Task 8.1, Task 8.2.
   - `_apply_density(bucket, mechanism, *, region_key, fraction)` consistent between Task 4.2 and Task 4.4 call site.
   - `_apply_place(bucket, mechanism, *, display_name, seen_names)` consistent between Task 4.3 and Task 4.4 call site.
   - `cv.radius_mid` (not `diam_mid`) consistent between spec, Task 6.1, Task 3.5, Task 5.1.

4. **Ordering dependency:** Phase 5 (`lower`) and Phase 6 (`CV`) have a forward reference. Phase 5 commits a deferred import; Phase 6 immediately follows. Tests gated correctly (Task 5.1 Step 4 notes that test runs after Phase 6).
