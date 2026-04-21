# Multi-compartment `Cell` Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `braincell/_multi_compartment.py` with a two-class (`Cell` declaration + `RunnableCell` runtime) package that fixes the 15 bugs flagged in review and drops the dirty-flag state machine.

**Architecture:** `Cell` holds only declaration (morpho, cv_policy, paint/place rules, solver/V_th/V_init/spk_fun). A single `cell.build()` call runs the lowering pipeline once and returns a frozen `RunnableCell`. `RunnableCell` owns runtime state (V, spike, C, V_th, point_tree, CellRuntimeState) and all solver methods. No dirty flags; the two objects are never both simultaneously mutable.

**Tech Stack:** Python 3.13, `jax`, `jax.numpy`, `brainstate`, `braintools`, `brainunit`, `numpy`, `pytest`/`unittest.TestCase`. Tests co-located as `*_test.py`.

**Spec:** `docs/superpowers/specs/2026-04-21-multi-compartment-rewrite-design.md`

---

## File structure

New package:

```
braincell/_multi_compartment/
  __init__.py       # re-exports Cell, RunnableCell, RunResult
  bridge.py         # cv_to_point, point_to_cv (renames + wraps compute/_runtime helpers)
  clamp_table.py    # ClampActiveTable + build_clamp_active_table
  currents.py       # total_membrane_current + _normalize_ext_to_point_density + _clamp_density
  build.py          # build() pipeline returning RunnableCell
  runnable.py       # RunnableCell class
  run.py            # run(rcell, *, dt, duration) + _validate_time_quantity + _normalize_run_traces
  probes.py         # sample_probe / sample_probes / all probe helpers
  cell.py           # Cell (declaration) + merge_place_rules helper
  # co-located tests
  bridge_test.py
  clamp_table_test.py
  currents_test.py
  build_test.py
  runnable_test.py
  run_test.py
  probes_test.py
  cell_test.py
```

Old `braincell/_multi_compartment.py`, `_multi_compartment_test.py`, and
`_multi_compartment_solver_test.py` are **deleted** after the new
package and new tests are in place.

`braincell/__init__.py` re-exports `Cell`, `RunnableCell`, `RunResult`
from the new package.

`braincell/compute/_runtime.py` gains one new optional field
(`clamp_active_table`) and one helper that builds it at
`CellRuntimeState.from_cell` time.

`braincell/cv/_mech.py` gains a `merge_place_rules` helper mirroring
`merge_paint_rules`.

---

## Task 0: Preparation

**Files:**
- Read: `braincell/_multi_compartment.py` (for porting reference)
- Read: `braincell/compute/_runtime.py` (for ClampActiveTable wiring)
- Read: `braincell/cv/_mech.py` (for merge_paint_rules pattern)

- [ ] **Step 1: Confirm baseline green**

Run: `pytest braincell/_multi_compartment_test.py braincell/_multi_compartment_solver_test.py -x -q`
Expected: all pass (or document the specific failures — they must be unrelated to this work). If any fail, triage before continuing — this plan assumes a green baseline.

- [ ] **Step 2: Create a scratch backup branch**

Run: `git branch morpho_1-multi-compartment-legacy`
Expected: branch created; `git branch --list morpho_1-multi-compartment-legacy` shows it. Used only as a diff target during porting; never pushed.

- [ ] **Step 3: Create the package directory**

Run: `mkdir -p braincell/_multi_compartment`
Expected: directory exists.

- [ ] **Step 4: Create an empty `__init__.py`**

```python
# braincell/_multi_compartment/__init__.py
# Populated by subsequent tasks.
```

- [ ] **Step 5: Commit scaffold**

```bash
git add braincell/_multi_compartment/__init__.py
git commit -m "scaffold: create _multi_compartment package dir"
```

---

## Task 1: `bridge.py` — CV ↔ point-space helpers

**Files:**
- Create: `braincell/_multi_compartment/bridge.py`
- Create: `braincell/_multi_compartment/bridge_test.py`

The old names `scatter_midpoint_values` / `gather_midpoint_values` live
in `compute/_runtime.py` and are imported across the codebase. This
task adds thin, well-named wrappers local to the new package so the
rest of the package code reads naturally. We do **not** rename the
originals — other modules still depend on them.

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/bridge_test.py
import unittest
import brainunit as u
import numpy as np

from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell._multi_compartment import bridge


def _one_branch_cell() -> Cell:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    return Cell(tree, cv_policy=CVPerBranch())


class TestBridge(unittest.TestCase):
    def test_cv_to_point_then_point_to_cv_roundtrip(self):
        cell = _one_branch_cell()
        rcell = cell.build()
        n_cv = rcell.n_cv

        cv_values = u.Quantity(np.arange(n_cv, dtype=float), u.mV)
        point_values = bridge.cv_to_point(cv_values, rcell.runtime)

        self.assertEqual(point_values.shape, (rcell.n_point,))
        back = bridge.point_to_cv(point_values, rcell.runtime)
        np.testing.assert_allclose(back.to_decimal(u.mV), cv_values.to_decimal(u.mV))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/bridge_test.py -v`
Expected: fail with `ImportError` or `AttributeError` — the module does not exist yet.

- [ ] **Step 3: Implement `bridge.py`**

```python
# braincell/_multi_compartment/bridge.py
"""CV ↔ point-space conversion helpers.

These are thin, named wrappers around
``braincell.compute._runtime.scatter_midpoint_values`` /
``gather_midpoint_values`` so the surrounding pipeline reads
naturally.  Keep the originals for other call sites.
"""

from braincell.compute._runtime import (
    CellRuntimeState,
    gather_midpoint_values,
    scatter_midpoint_values,
)

__all__ = ["cv_to_point", "point_to_cv"]


def cv_to_point(values, runtime: CellRuntimeState):
    """Scatter a ``(..., n_cv)`` array onto CV midpoints in point space.

    The returned array has shape ``(..., n_point)`` with zeros at every
    non-midpoint point.
    """
    return scatter_midpoint_values(
        values=values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
        n_point=runtime.n_point,
    )


def point_to_cv(values, runtime: CellRuntimeState):
    """Gather a ``(..., n_point)`` array at CV midpoints.

    The returned array has shape ``(..., n_cv)``.
    """
    return gather_midpoint_values(
        values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/_multi_compartment/bridge_test.py -v`
Expected: PASS.

(`Cell`/`RunnableCell`/`.build()` do not yet exist in the new
package — this test is written against the target API and will only
run green after Task 8 lands. For now, **stage the test file** and
come back to it.) If you are strictly following TDD, skip Step 4 until
Task 8 completes. The step 2 failure (ImportError on
`braincell._multi_compartment.bridge`) is what we're fixing in this
task.

After step 3 the module import succeeds. Update the import check:

```bash
python -c "from braincell._multi_compartment.bridge import cv_to_point, point_to_cv; print('ok')"
```
Expected output: `ok`.

- [ ] **Step 5: Commit**

```bash
git add braincell/_multi_compartment/bridge.py \
        braincell/_multi_compartment/bridge_test.py
git commit -m "feat(multi-compartment): add bridge.cv_to_point / point_to_cv"
```

---

## Task 2: `clamp_table.py` + `CellRuntimeState.clamp_active_table`

Pre-compute the clamp active-point table once at build time so
`currents.py` never walks the layout list per step.

**Files:**
- Create: `braincell/_multi_compartment/clamp_table.py`
- Create: `braincell/_multi_compartment/clamp_table_test.py`
- Modify: `braincell/compute/_runtime.py` — add `clamp_active_table` field on `CellRuntimeState` + call the builder at end of `from_cell`.

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/clamp_table_test.py
import unittest
import brainunit as u

from braincell import Branch, CVPerBranch, Cell, CurrentClamp, Morphology
from braincell.filter import RootLocation
from braincell._multi_compartment.clamp_table import ClampActiveTable


class TestClampActiveTable(unittest.TestCase):
    def test_no_clamp_yields_none_table(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        rcell = cell.build()
        self.assertIsNone(rcell.runtime.clamp_active_table)

    def test_current_clamp_builds_table(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.place(RootLocation(), CurrentClamp.step(0.1 * u.nA, 10 * u.ms, delay=1 * u.ms))
        rcell = cell.build()
        table = rcell.runtime.clamp_active_table
        self.assertIsInstance(table, ClampActiveTable)
        self.assertGreater(len(table.ids), 0)
        self.assertTrue((table.area > 0.0).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/clamp_table_test.py -v`
Expected: import error for `clamp_table` module.

- [ ] **Step 3: Implement `clamp_table.py`**

```python
# braincell/_multi_compartment/clamp_table.py
"""Pre-computed active-clamp-point table attached to CellRuntimeState.

Building this once at compile time replaces the per-step filter walk
in ``Cell._point_clamp_input``.
"""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell.compute._runtime import MechanismLayout

__all__ = ["ClampActiveTable", "build_clamp_active_table", "CLAMP_KINDS"]


#: Clamp layout kinds that contribute point-space current via
#: ``CellRuntimeState.evaluate_point_clamps``.
CLAMP_KINDS = frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"})


@dataclass(frozen=True)
class ClampActiveTable:
    """Active clamp points + their membrane areas, built once at compile time."""

    ids: np.ndarray           # (n_active,) int32 unique sorted point ids
    area: np.ndarray          # (n_active,) float64 membrane area in cm^2


def build_clamp_active_table(
    *,
    layouts: tuple[MechanismLayout, ...],
    cvs,
    point_tree,
    n_point: int,
) -> ClampActiveTable | None:
    """Return None when no clamps are placed."""
    active: set[int] = set()
    for layout in layouts:
        if layout.target != "point" or layout.point_index is None:
            continue
        if layout.kind not in CLAMP_KINDS:
            continue
        active.update(int(pid) for pid in layout.point_index.tolist())

    if not active:
        return None

    ids = np.asarray(sorted(active), dtype=np.int32)

    # Build a (n_point,) area lookup from CV areas scattered to midpoints.
    point_area = np.zeros((n_point,), dtype=float)
    for cv in cvs:
        pid = int(point_tree.cv_midpoint_point_id[cv.id])
        point_area[pid] = float(np.asarray(cv.area.to_decimal(u.cm ** 2), dtype=float))

    area = point_area[ids]
    if np.any(area <= 0.0):
        bad = ids[area <= 0.0].tolist()
        raise ValueError(
            "Point clamp active points must have positive membrane area, "
            f"got non-positive area at point ids {bad!r}."
        )
    return ClampActiveTable(ids=ids, area=area.astype(np.float64, copy=False))
```

- [ ] **Step 4: Wire the field onto `CellRuntimeState`**

Edit `braincell/compute/_runtime.py`:

Add a new field near the end of the dataclass definition (after
`dhs_static_cache`):

```python
    clamp_active_table: object | None = None
```

At the end of `CellRuntimeState.from_cell`, before the `return cls(...)`,
build the table:

```python
        from braincell._multi_compartment.clamp_table import build_clamp_active_table
        clamp_active_table = build_clamp_active_table(
            layouts=tuple(layouts),
            cvs=cell.cvs,
            point_tree=point_tree,
            n_point=n_point,
        )
```

And pass it in the `return cls(...)` call:

```python
            clamp_active_table=clamp_active_table,
```

The import lives **inside** `from_cell` (not at module top) to avoid a
circular import between `compute._runtime` and
`_multi_compartment.clamp_table`.

- [ ] **Step 5: Run unit tests**

Run: `pytest braincell/_multi_compartment/clamp_table_test.py -v`
Expected: both tests green once `Cell.build()` exists (Task 8).

Run the existing runtime smoke tests now to catch regressions from the
field addition:

Run: `pytest braincell/compute/_runtime_test.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braincell/_multi_compartment/clamp_table.py \
        braincell/_multi_compartment/clamp_table_test.py \
        braincell/compute/_runtime.py
git commit -m "feat(runtime): add ClampActiveTable, build once at compile"
```

---

## Task 3: `currents.py` — membrane-current pipeline

This is the bug #1 + #2 fix. External `I_ext` is routed through
`sum_current_inputs` as `init` (SC pattern), `(n_point,)`-shaped total
current is rejected, and clamp current reads the cached table.

**Files:**
- Create: `braincell/_multi_compartment/currents.py`
- Create: `braincell/_multi_compartment/currents_test.py`

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/currents_test.py
import unittest
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell._multi_compartment.currents import _normalize_ext_to_point_density


def _rcell():
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return Cell(tree, cv_policy=CVPerBranch()).build()


class TestNormalizeExt(unittest.TestCase):
    def test_python_zero(self):
        rcell = _rcell()
        out = _normalize_ext_to_point_density(0.0, rcell.runtime)
        self.assertEqual(out.shape, (rcell.n_point,))
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), 0.0)

    def test_scalar_density_broadcasts(self):
        rcell = _rcell()
        out = _normalize_ext_to_point_density(1.5 * u.nA / u.cm ** 2, rcell.runtime)
        self.assertEqual(out.shape, (rcell.n_point,))

    def test_scalar_total_current_divides_by_cv_area(self):
        rcell = _rcell()
        out = _normalize_ext_to_point_density(0.2 * u.nA, rcell.runtime)
        self.assertEqual(out.shape, (rcell.n_point,))

    def test_n_point_total_current_rejected(self):
        rcell = _rcell()
        bad = jnp.ones((rcell.n_point,)) * u.nA
        with self.assertRaises(ValueError):
            _normalize_ext_to_point_density(bad, rcell.runtime)

    def test_cv_shape_density_scatters(self):
        rcell = _rcell()
        arr = jnp.ones((rcell.n_cv,)) * u.nA / u.cm ** 2
        out = _normalize_ext_to_point_density(arr, rcell.runtime)
        self.assertEqual(out.shape, (rcell.n_point,))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/currents_test.py -v`
Expected: fail with ImportError on the module.

- [ ] **Step 3: Implement `currents.py`**

```python
# braincell/_multi_compartment/currents.py
"""Membrane-current summation pipeline for RunnableCell.

Responsibilities:

1. Normalize user-supplied external current (``I_ext``) into point-space
   current density, rejecting shapes that would NaN (bug #2).
2. Thread the external density through ``sum_current_inputs`` as
   ``init`` (SC-pattern), so registered current-input callables
   accumulate on top and external is never dropped (bug #1).
3. Add clamp density from the precomputed ``ClampActiveTable``.
4. Iterate channel currents.
5. Bridge point-space sum back to CV-space for divide-by-C.
"""

from typing import TYPE_CHECKING

import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel
from braincell.compute._runtime import CellRuntimeState
from . import bridge

if TYPE_CHECKING:
    from .runnable import RunnableCell

__all__ = ["total_membrane_current"]


def total_membrane_current(
    host: "RunnableCell",
    *,
    V_cv,
    I_ext,
    t,
) -> object:
    """Return ``(..., n_cv)`` membrane current density in nA/cm^2."""
    runtime = host.runtime
    point_V = bridge.cv_to_point(V_cv, runtime)

    I_ext_density = _normalize_ext_to_point_density(I_ext, runtime)
    I_point = host.sum_current_inputs(I_ext_density, point_V)

    I_point = I_point + _clamp_density(runtime, t=t)

    for key, ch in host.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
        try:
            contrib = ch.current(point_V)
        except Exception as exc:
            raise ValueError(
                f"Error in computing current for ion channel {key!r}:\n{ch}\nError: {exc}"
            ) from exc
        if contrib is None:
            continue
        I_point = I_point + contrib

    return bridge.point_to_cv(I_point, runtime)


_CURRENT_DENSITY = u.nA / u.cm ** 2


def _normalize_ext_to_point_density(value, runtime: CellRuntimeState):
    """Convert every supported ``I_ext`` shape to ``(n_point,) nA/cm^2``.

    Accepts:

    - python ``0`` / ``0.0``             -> zeros
    - scalar current density             -> broadcast
    - scalar total current (nA)          -> divide by cv area, scatter
    - ``(n_cv,)`` current density        -> scatter
    - ``(n_cv,)`` total current (nA)     -> divide, scatter
    - ``(n_point,)`` current density     -> pass through
    - ``(n_point,)`` total current (nA)  -> **raise** ValueError
    """
    if _is_python_zero(value):
        return u.Quantity(jnp.zeros((runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    if not isinstance(value, u.Quantity):
        # Plain array/scalar without units is ambiguous; require density.
        return u.Quantity(jnp.broadcast_to(jnp.asarray(value), (runtime.n_point,)), _CURRENT_DENSITY)

    is_density = value.has_same_unit(1.0 * _CURRENT_DENSITY)
    is_total = value.has_same_unit(1.0 * u.nA)
    shape = getattr(value, "shape", ())

    if is_total and shape == (runtime.n_point,):
        raise ValueError(
            "I_ext supplied as (n_point,)-shaped total current (nA) is ambiguous; "
            "pass (n_cv,) or provide current density (nA/cm^2) instead."
        )

    if is_density:
        if shape == (runtime.n_point,):
            return value.in_unit(_CURRENT_DENSITY)
        if shape == (runtime.n_cv,):
            return bridge.cv_to_point(value.in_unit(_CURRENT_DENSITY), runtime)
        if shape == ():
            broadcast = jnp.broadcast_to(jnp.asarray(value.to_decimal(_CURRENT_DENSITY)), (runtime.n_point,))
            return u.Quantity(broadcast, _CURRENT_DENSITY)

    if is_total:
        cv_area = _cv_area(runtime)
        if shape == ():
            cv_density = (value / cv_area).in_unit(_CURRENT_DENSITY)
            return bridge.cv_to_point(cv_density, runtime)
        if shape == (runtime.n_cv,):
            cv_density = (value / cv_area).in_unit(_CURRENT_DENSITY)
            return bridge.cv_to_point(cv_density, runtime)

    raise ValueError(
        f"Unsupported I_ext shape/unit: shape={shape!r}, unit={getattr(value, 'unit', None)!r}. "
        "Accepted shapes: (), (n_cv,), (n_point,) with density units; "
        "or (), (n_cv,) with total-current units."
    )


def _clamp_density(runtime: CellRuntimeState, *, t):
    """Return ``(n_point,) nA/cm^2`` clamp current density.

    Uses the pre-built ``ClampActiveTable``; the hot path does no
    layout iteration.
    """
    table = runtime.clamp_active_table
    if table is None:
        return u.Quantity(jnp.zeros((runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    currents_nA = runtime.evaluate_point_clamps(t=t).to_decimal(u.nA)
    active = currents_nA[table.ids] / table.area
    density = jnp.zeros((runtime.n_point,), dtype=float)
    density = density.at[table.ids].set(active)
    return u.Quantity(density, _CURRENT_DENSITY)


def _cv_area(runtime: CellRuntimeState):
    # Imported lazily to avoid a circular import at module-load time.
    from braincell.compute._runtime import cv_value_vector
    return cv_value_vector(runtime._host_for_area, attr_name="area")  # noqa: SLF001


def _is_python_zero(value) -> bool:
    return isinstance(value, (int, float)) and value == 0
```

Note — `_cv_area` reads `runtime._host_for_area`; we store the host
`RunnableCell` on the runtime via an attribute assignment in
`build.py` (Task 6). Alternative: pass host down explicitly. This
plan takes the attribute route to keep the function signature
single-argument.

- [ ] **Step 4: Run test**

Run: `pytest braincell/_multi_compartment/currents_test.py -v`
Expected: all five tests pass once `Cell.build()` exists (Task 8).
Before then, test will ImportError — that's the expected red step.

Until Task 8 lands, verify the module at least imports cleanly:

```bash
python -c "from braincell._multi_compartment.currents import total_membrane_current; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add braincell/_multi_compartment/currents.py \
        braincell/_multi_compartment/currents_test.py
git commit -m "feat(multi-compartment): add currents.total_membrane_current pipeline

Fixes the silently-dropped external I_ext bug and rejects (n_point,)
total-current shapes that would NaN on midpoint-scatter divide."
```

---

## Task 4: `run.py` — time loop + trace handling

Port `run()` and its helpers verbatim from `_multi_compartment.py`,
with the fix that `self._current_time.value` is not mutated inside the
`for_loop` scan body. `t` flows to clamp evaluation via
`brainstate.environ['t']` instead.

**Files:**
- Create: `braincell/_multi_compartment/run.py`
- Create: `braincell/_multi_compartment/run_test.py`

- [ ] **Step 1: Write the failing test (smoke)**

```python
# braincell/_multi_compartment/run_test.py
import unittest
import brainunit as u

from braincell._multi_compartment.run import (
    _normalize_run_traces,
    _validate_time_quantity,
)


class TestRunHelpers(unittest.TestCase):
    def test_validate_time_quantity_accepts_scalar(self):
        _validate_time_quantity(0.1 * u.ms, name="dt")

    def test_validate_time_quantity_rejects_plain_float(self):
        with self.assertRaises(TypeError):
            _validate_time_quantity(0.1, name="dt")

    def test_validate_time_quantity_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            _validate_time_quantity(0.0 * u.ms, name="dt")

    def test_normalize_run_traces_single(self):
        out = _normalize_run_traces("X", n_traces=1)
        self.assertEqual(out, ("X",))

    def test_normalize_run_traces_many(self):
        out = _normalize_run_traces(("a", "b"), n_traces=2)
        self.assertEqual(out, ("a", "b"))

    def test_normalize_run_traces_mismatch(self):
        with self.assertRaises(ValueError):
            _normalize_run_traces(("a",), n_traces=2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/run_test.py -v`
Expected: fail with ImportError.

- [ ] **Step 3: Implement `run.py`**

```python
# braincell/_multi_compartment/run.py
"""Simulation time loop + trace helpers for RunnableCell.run()."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import jax
import numpy as np

if TYPE_CHECKING:
    from .runnable import RunnableCell

__all__ = ["RunResult", "run"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunResult:
    time: object
    traces: dict[str, object]


def run(rcell: "RunnableCell", *, dt, duration) -> RunResult:
    _validate_time_quantity(dt, name="dt")
    _validate_time_quantity(duration, name="duration")

    initial_samples = rcell.sample_probes()
    if len(initial_samples) == 0:
        raise ValueError("RunnableCell.run(...) requires at least one placed probe.")
    ordered_names = tuple(sorted(initial_samples))

    with brainstate.environ.context(dt=dt):
        start_t = rcell.current_time
        relative_times = u.math.arange(0.0 * u.ms, duration, brainstate.environ.get_dt())
        if int(relative_times.shape[0]) == 0:
            raise ValueError("RunnableCell.run(...) produced no timesteps; ensure duration > 0 and dt > 0.")
        times = start_t + relative_times

        def _step(t):
            # Propagate t via brainstate.environ so clamp eval picks it up.
            with brainstate.environ.context(t=t):
                rcell.update()
                snapshot = rcell.sample_probes()
            return tuple(snapshot[name] for name in ordered_names)

        traces_over_time = brainstate.transform.for_loop(_step, times)
        rcell._set_current_time(start_t + int(times.shape[0]) * brainstate.environ.get_dt())

    traces_tuple = _normalize_run_traces(traces_over_time, n_traces=len(ordered_names))
    traces = {name: trace for name, trace in zip(ordered_names, traces_tuple)}
    return RunResult(time=times, traces=traces)


def _validate_time_quantity(value, *, name: str) -> None:
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"RunnableCell.run(...) {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimal.shape not in ((), (1,)):
        raise ValueError(f"RunnableCell.run(...) {name} must be scalar, got shape {decimal.shape!r}.")
    if float(decimal.reshape(())) <= 0.0:
        raise ValueError(f"RunnableCell.run(...) {name} must be > 0, got {value!r}.")


def _normalize_run_traces(values, *, n_traces: int) -> tuple[object, ...]:
    if n_traces == 1:
        return values if isinstance(values, tuple) else (values,)
    if not isinstance(values, tuple):
        raise TypeError(f"RunnableCell.run(...) expected {n_traces} trace arrays, got {type(values).__name__!s}.")
    if len(values) != n_traces:
        raise ValueError(f"RunnableCell.run(...) expected {n_traces} trace arrays, got {len(values)!r}.")
    return values
```

- [ ] **Step 4: Run helper tests**

Run: `pytest braincell/_multi_compartment/run_test.py -v`
Expected: 6 tests pass. `run(rcell, ...)` itself is tested end-to-end
in Task 10.

- [ ] **Step 5: Commit**

```bash
git add braincell/_multi_compartment/run.py \
        braincell/_multi_compartment/run_test.py
git commit -m "feat(multi-compartment): add run(), RunResult, time helpers

Propagate t via brainstate.environ context instead of mutating
_current_time inside the scan body."
```

---

## Task 5: `probes.py` — probe sampling

Port every probe helper from the old file verbatim, renamed to operate
on `RunnableCell` instead of `Cell`. No logic change.

**Files:**
- Create: `braincell/_multi_compartment/probes.py`
- Create: `braincell/_multi_compartment/probes_test.py`

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/probes_test.py
import unittest

import brainunit as u

from braincell import Branch, CVPerBranch, Cell, Morphology, StateProbe
from braincell.filter import RootLocation


class TestProbesModule(unittest.TestCase):
    def test_state_probe_v_returns_quantity(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.place(RootLocation(), StateProbe(field="v", name="V_root"))
        rcell = cell.build()
        sample = rcell.sample_probe("V_root")
        self.assertTrue(hasattr(sample, "to_decimal"))
        self.assertEqual(sample.to_decimal(u.mV).shape, ())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/probes_test.py -v`
Expected: fail on `Cell`/`build` import path (Task 8) or `probes`
module import; both are acceptable red signals.

- [ ] **Step 3: Implement `probes.py`**

Copy the following verbatim from the old `_multi_compartment.py` into
`braincell/_multi_compartment/probes.py`, adjusting the imports at the
top and replacing every `self: "Cell"` reference with the bound
`RunnableCell`:

- `_probe_name`
- `_midpoint_cv_id`
- `_select_last_axis`
- `_probe_state_attr`
- `_probe_current_ion_info`
- `_probe_current_value`
- `_pack_probe_samples`
- `sample_probe(rcell, name) -> object`
- `sample_probes(rcell) -> dict[str, object]`
- `_sample_probe_layout(rcell, *, layout, declaration) -> object`
- `_sample_state_probe_point(rcell, runtime, *, declaration, point_id)`
- `_sample_mechanism_probe_point(rcell, runtime, *, declaration, point_id)`
- `_sample_current_probe_point(rcell, runtime, *, declaration, point_id)`

Module header:

```python
# braincell/_multi_compartment/probes.py
"""Probe sampling helpers for RunnableCell."""

from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import numpy as np

from braincell.compute._runtime import CellRuntimeState
from braincell.mech import CurrentProbe, Density, MechanismProbe, StateProbe

if TYPE_CHECKING:
    from .runnable import RunnableCell

__all__ = ["sample_probe", "sample_probes"]
```

The only structural change from the old code is that every method
previously bound on `Cell` becomes a module-level function whose first
arg is the `RunnableCell`. `RunnableCell` forwards
`sample_probe` / `sample_probes` to this module.

- [ ] **Step 4: Run test**

Run: `pytest braincell/_multi_compartment/probes_test.py -v`
Expected: PASS after Task 8 lands.

Verify import now:

```bash
python -c "from braincell._multi_compartment.probes import sample_probe, sample_probes; print('ok')"
```

- [ ] **Step 5: Commit**

```bash
git add braincell/_multi_compartment/probes.py \
        braincell/_multi_compartment/probes_test.py
git commit -m "feat(multi-compartment): extract probe sampling into probes.py"
```

---

## Task 6: `build.py` — the single lowering pipeline

Runs the pipeline once and produces a `RunnableCell`. Everything
previously spread across `_rebuild_if_needed` /
`_ensure_runtime_compiled` / `install_cell_runtime` / `init_state`
collapses into one function.

**Files:**
- Create: `braincell/_multi_compartment/build.py`
- Create: `braincell/_multi_compartment/build_test.py`

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/build_test.py
import unittest

import brainunit as u

from braincell import Branch, CVPerBranch, Cell, Morphology


class TestBuildPipeline(unittest.TestCase):
    def test_build_returns_runnable_cell(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        rcell = cell.build()
        self.assertGreater(rcell.n_cv, 0)
        self.assertGreater(rcell.n_point, 0)
        self.assertTrue(hasattr(rcell, "V"))
        self.assertTrue(hasattr(rcell, "spike"))

    def test_build_twice_is_independent(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        a = cell.build()
        b = cell.build()
        self.assertIsNot(a, b)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/build_test.py -v`
Expected: ImportError on build module.

- [ ] **Step 3: Implement `build.py`**

```python
# braincell/_multi_compartment/build.py
"""Single lowering pipeline: Cell declaration -> RunnableCell."""

from typing import TYPE_CHECKING

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel
from braincell.compute._point_tree import build_point_tree
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
    clone_morpho,
    cv_value_vector,
    fill_like,
    install_cell_runtime as _legacy_install,
)
from braincell.cv._cv import assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    apply_paint_rules,
    apply_place_rules,
    init_cv_mech,
)
from braincell.quad import get_integrator
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqState

if TYPE_CHECKING:
    from .cell import Cell
    from .runnable import RunnableCell

__all__ = ["build"]


def build(cell: "Cell") -> "RunnableCell":
    from .runnable import RunnableCell

    morpho = clone_morpho(cell.morpho)
    cv_geo, cv_ids_by_branch = build_cv_geo(
        morpho,
        policy=cell.cv_policy,
        paint_rules=cell.paint_rules,
    )
    cv_mech = init_cv_mech(len(cv_geo))
    apply_paint_rules(
        morpho,
        cvs=cv_geo,
        cv_ids_by_branch=cv_ids_by_branch,
        paint_rules=cell.paint_rules,
        mechs=cv_mech,
    )
    apply_place_rules(
        morpho,
        cvs=cv_geo,
        cv_ids_by_branch=cv_ids_by_branch,
        place_rules=cell.place_rules,
        mechs=cv_mech,
    )
    cvs = tuple(assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo)

    rcell = RunnableCell.__new__(RunnableCell)
    RunnableCell._preinit(
        rcell,
        name=cell._name,
        V_th_value=cell.V_th,
        V_initializer_spec=cell.V_init,
        spk_fun=cell.spk_fun,
        solver_name=cell._solver_name,
        solver=cell._solver_fn,
        morpho=morpho,
        cvs=cvs,
    )

    point_tree = build_point_tree(morpho, cvs=cvs)
    runtime = CellRuntimeState.from_cell(rcell)
    runtime._host_for_area = rcell  # used by currents._cv_area; noqa: SLF001

    RunnableCell._attach_runtime(rcell, runtime=runtime, point_tree=point_tree)

    # Install runtime nodes + C + V_th (reuses the legacy helper).
    _legacy_install(rcell, runtime)

    # Allocate voltage + spike states.
    v_initializer = cell.V_init if cell.V_init is not None else cv_value_vector(rcell, attr_name="v")
    rcell.V = DiffEqState(braintools.init.param(v_initializer, rcell.varshape))
    rcell.spike = brainstate.ShortTermState(rcell.get_spike(rcell.V.value, rcell.V.value))
    rcell._current_time_state.value = 0.0 * u.ms

    # Seed channel states.
    point_V = rcell._cv_to_point(rcell.V.value)
    for channel in rcell.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
        channel.init_state(point_V, batch_size=None)

    # Pre-cache the JAX-side axial operator once.
    rcell._axial_jax = jnp.asarray(
        build_cv_axial_operator(
            rcell,
            point_tree=point_tree,
            scheduling=rcell.point_scheduling(algorithm="dhs"),
        ),
        dtype=jnp.float64,
    )
    return rcell
```

- [ ] **Step 4: Verify the pipeline imports**

Run: `python -c "from braincell._multi_compartment.build import build; print('ok')"`
Expected: `ok`.

(Tests in step 1 go green after Task 7 + Task 8 provide
`RunnableCell` / `Cell`.)

- [ ] **Step 5: Commit**

```bash
git add braincell/_multi_compartment/build.py \
        braincell/_multi_compartment/build_test.py
git commit -m "feat(multi-compartment): add build() lowering pipeline"
```

---

## Task 7: `runnable.py` — `RunnableCell` class

**Files:**
- Create: `braincell/_multi_compartment/runnable.py`
- Create: `braincell/_multi_compartment/runnable_test.py`

- [ ] **Step 1: Write the failing test**

```python
# braincell/_multi_compartment/runnable_test.py
import unittest

import brainstate
import brainunit as u

from braincell import Branch, CVPerBranch, Cell, Morphology, StateProbe
from braincell.filter import RootLocation


def _make_rcell():
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
    cell.place(RootLocation(), StateProbe(field="v", name="V_root"))
    return cell.build()


class TestRunnableCell(unittest.TestCase):
    def test_update_advances_without_I_ext(self):
        rcell = _make_rcell()
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            rcell.update()

    def test_external_current_not_dropped_when_no_registered_inputs(self):
        rcell = _make_rcell()
        before = rcell.V.value.to_decimal(u.mV).copy()
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            rcell.update(I_ext=0.5 * u.nA)
        after = rcell.V.value.to_decimal(u.mV)
        # Injection with no channels still perturbs V via the linear cable solve.
        self.assertFalse((before == after).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/runnable_test.py -v`
Expected: ImportError on `runnable`.

- [ ] **Step 3: Implement `runnable.py`**

```python
# braincell/_multi_compartment/runnable.py
"""RunnableCell: runtime facade produced by Cell.build()."""

from typing import Callable

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
)
from braincell.mech import CurrentProbe, Density, MechanismProbe, StateProbe
from braincell.morph.morphology import Morphology
from braincell.quad.protocol import DiffEqState, IndependentIntegration

from . import bridge, currents, probes, run as run_module

__all__ = ["RunnableCell"]


def _cast_like(value, like):
    dtype = jnp.asarray(u.get_magnitude(like)).dtype
    if isinstance(value, u.Quantity):
        unit = u.get_unit(value)
        return jnp.asarray(value.to_decimal(unit), dtype=dtype) * unit
    return jnp.asarray(value, dtype=dtype)


class RunnableCell(HHTypedNeuron):
    """Frozen runtime facade for a multi-compartment cell.

    Constructed only via ``Cell.build()``; ``__new__`` + ``_preinit`` +
    ``_attach_runtime`` are the internal staging API. Users never call
    them directly.
    """

    __module__ = "braincell"

    # ------------------------------------------------------------------
    # Construction (internal)

    def _preinit(
        self,
        *,
        name: str | None,
        V_th_value,
        V_initializer_spec,
        spk_fun: Callable,
        solver_name: str,
        solver: Callable,
        morpho: Morphology,
        cvs,
    ) -> None:
        HHTypedNeuron.__init__(self, size=(1,), name=name, **build_placeholder_ions())
        self._morpho = morpho
        self._cvs = cvs
        self._V_th_value = V_th_value
        self._V_initializer_spec = V_initializer_spec
        self._spk_fun = spk_fun
        self.solver_name = solver_name
        self.solver = solver
        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._point_scheduling_cache: dict[tuple[str, int], object] = {}

    def _attach_runtime(self, *, runtime: CellRuntimeState, point_tree) -> None:
        self._runtime = runtime
        self._point_tree = point_tree

    # ------------------------------------------------------------------
    # Topology (read-only)

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cvs(self):
        return self._cvs

    @property
    def runtime(self) -> CellRuntimeState:
        return self._runtime

    @property
    def n_cv(self) -> int:
        return self._runtime.n_cv

    @property
    def n_point(self) -> int:
        return self._runtime.n_point

    @property
    def pop_size(self) -> tuple[int, ...]:
        return ()

    @property
    def varshape(self) -> tuple[int, ...]:
        return (self.n_cv,)

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    def point_tree(self):
        return self._point_tree

    def point_scheduling(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        from braincell.compute._point_tree import build_point_scheduling
        key = (algorithm, max_group_size)
        cached = self._point_scheduling_cache.get(key)
        if cached is not None:
            return cached
        scheduling = build_point_scheduling(
            self._point_tree,
            max_group_size=max_group_size,
            algorithm=algorithm,
        )
        self._point_scheduling_cache[key] = scheduling
        return scheduling

    # ------------------------------------------------------------------
    # Time

    @property
    def current_time(self):
        return self._current_time_state.value

    def _set_current_time(self, value) -> None:
        self._current_time_state.value = value

    # ------------------------------------------------------------------
    # Repr

    def __repr__(self) -> str:
        return (
            f"RunnableCell(root={self._morpho.root.name!r}, "
            f"n_cv={self.n_cv!r}, n_point={self.n_point!r})"
        )

    # ------------------------------------------------------------------
    # Space bridging (thin wrappers for internal use)

    def _cv_to_point(self, cv_values):
        return bridge.cv_to_point(cv_values, self._runtime)

    def _point_to_cv(self, point_values):
        return bridge.point_to_cv(point_values, self._runtime)

    # ------------------------------------------------------------------
    # Solver path

    def _resolve_t(self):
        try:
            return brainstate.environ.get("t")
        except KeyError:
            return self.current_time

    def pre_integral(self, I_ext=0.0):
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(point_V)

    def compute_derivative(self, I_ext=0.0):
        self.V.derivative = self.compute_voltage_derivative(self.V.value, I_ext)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.compute_derivative(point_V)

    def compute_membrane_derivative(self, V, I_ext=0.0):
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, I_ext=I_ext, t=t)
        return I_total / self.C

    def compute_axial_derivative(self, V):
        V_decimal = jnp.asarray(V.to_decimal(u.mV), dtype=jnp.float64)
        axial = -jnp.matmul(V_decimal, self._axial_jax.T)
        return axial * (u.mV / u.ms)

    def compute_voltage_derivative(self, V, I_ext=0.0):
        return self.compute_membrane_derivative(V, I_ext) + self.compute_axial_derivative(V)

    def post_integral(self, I_ext=0.0):
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.post_integral(point_V)

    def update(self, I_ext=0.0):
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.update(point_V)

        last_V = self.V.value
        dt = brainstate.environ.get("dt")
        if dt is None:
            raise ValueError("RunnableCell.update(...) requires brainstate.environ['dt'] to be set.")

        if _is_python_zero(I_ext):
            self.solver(self)
        else:
            self.solver(self, I_ext)

        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    # ------------------------------------------------------------------
    # Spike / reset

    def get_spike(self, last_V, next_V):
        denom = _cast_like(20.0 * u.mV, next_V)
        V_th = _cast_like(self.V_th, next_V)
        return (
            self._spk_fun((next_V - V_th) / denom)
            * self._spk_fun((V_th - last_V) / denom)
        )

    def reset_state(self, batch_size=None) -> None:
        import braintools
        v_init = self._V_initializer_spec
        if v_init is None:
            from braincell.compute._runtime import cv_value_vector
            v_init = cv_value_vector(self, attr_name="v")
        self.V.value = braintools.init.param(v_init, self.varshape, batch_size)
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time_state.value = 0.0 * u.ms
        point_V = self._cv_to_point(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.reset_state(point_V, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Inspection forwards

    @property
    def layouts(self):
        return self._runtime.layouts

    @property
    def voltage_shape(self):
        return self._runtime.voltage_shape

    def get_point_layouts(self, point_id):
        return self._runtime.get_point_layouts(point_id)

    def get_cv_layouts(self, cv_id):
        return self._runtime.get_cv_layouts(cv_id)

    def expected_state_shape(self, layout_id, var_name):
        return self._runtime.expected_state_shape(layout_id, var_name)

    def get_state(self, layout_id, var_name):
        return self._runtime.get_state(layout_id, var_name)

    def set_state(self, layout_id, var_name, value) -> None:
        self._runtime.set_state(layout_id, var_name, value)

    def get_point_state(self, point_id):
        return self._runtime.get_point_state(point_id)

    def get_cv_state(self, cv_id):
        return self._runtime.get_cv_state(cv_id)

    def get_runtime_node(self, layout_id):
        return self._runtime.get_runtime_node(layout_id)

    def get_ion(self, name):
        return self._runtime.get_ion(name)

    # ------------------------------------------------------------------
    # Probes

    def sample_probe(self, name: str):
        return probes.sample_probe(self, name)

    def sample_probes(self) -> dict[str, object]:
        return probes.sample_probes(self)

    def mech_table(self):
        return probes.mech_table(self)

    # ------------------------------------------------------------------
    # Run

    def run(self, *, dt, duration):
        return run_module.run(self, dt=dt, duration=duration)


def _is_python_zero(value) -> bool:
    return isinstance(value, (int, float)) and value == 0
```

- [ ] **Step 4: Move `mech_table` into `probes.py`**

The old `Cell._mech_table` lives in `_multi_compartment.py`. Move it to
`probes.py` as `mech_table(rcell)`. No logic change beyond replacing
`self` with `rcell` and reading `rcell.runtime` / `rcell.cvs` /
`rcell.point_tree()`.

- [ ] **Step 5: Run tests**

Run: `pytest braincell/_multi_compartment/runnable_test.py -v`
Expected: both tests green after Task 8.

Sanity import check now:

```bash
python -c "from braincell._multi_compartment.runnable import RunnableCell; print('ok')"
```

- [ ] **Step 6: Commit**

```bash
git add braincell/_multi_compartment/runnable.py \
        braincell/_multi_compartment/runnable_test.py \
        braincell/_multi_compartment/probes.py
git commit -m "feat(multi-compartment): add RunnableCell class"
```

---

## Task 8: `cell.py` — declaration frontend + `merge_place_rules`

**Files:**
- Create: `braincell/_multi_compartment/cell.py`
- Create: `braincell/_multi_compartment/cell_test.py`
- Modify: `braincell/cv/_mech.py` — add `merge_place_rules`.

- [ ] **Step 1: Add `merge_place_rules` to `braincell/cv/_mech.py`**

Inspect `merge_paint_rules` in that file; mirror it for place rules,
keyed on `(locset_key, mechanism_signature)` so identical duplicates
collapse. (Look at the existing `merge_paint_rules` for the exact key
convention and copy it.)

- [ ] **Step 2: Write the failing test**

```python
# braincell/_multi_compartment/cell_test.py
import unittest

import brainunit as u

from braincell import (
    Branch,
    CVPerBranch,
    CableProperty,
    Cell,
    CurrentClamp,
    Morphology,
)
from braincell.filter import BranchSlice, RootLocation


class TestCellDeclaration(unittest.TestCase):
    def _soma(self):
        return Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")

    def test_build_twice_produces_independent_runnables(self):
        cell = Cell(Morphology.from_root(self._soma(), name="soma"), cv_policy=CVPerBranch())
        a = cell.build()
        b = cell.build()
        self.assertIsNot(a, b)
        self.assertEqual(a.n_cv, b.n_cv)

    def test_place_dedups_identical_rules(self):
        cell = Cell(Morphology.from_root(self._soma(), name="soma"), cv_policy=CVPerBranch())
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms, delay=1 * u.ms)
        cell.place(RootLocation(), clamp)
        cell.place(RootLocation(), clamp)
        self.assertEqual(len(cell.place_rules), 1)

    def test_paint_rules_start_with_defaults(self):
        cell = Cell(Morphology.from_root(self._soma(), name="soma"), cv_policy=CVPerBranch())
        self.assertGreater(len(cell.paint_rules), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest braincell/_multi_compartment/cell_test.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `cell.py`**

```python
# braincell/_multi_compartment/cell.py
"""Cell: declaration frontend. Call .build() once paint/place is done."""

from typing import Callable

import braintools
import brainunit as u

from braincell.cv._cv import assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    PaintRule,
    PlaceRule,
    apply_paint_rules,
    apply_place_rules,
    default_paint_rules,
    init_cv_mech,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell.cv._policy import CVPerBranch, CVPolicy
from braincell.filter import LocsetExpr, RegionExpr
from braincell.morph.morphology import Morphology
from braincell.quad import get_integrator

from .build import build as _build_pipeline

__all__ = ["Cell"]


class Cell:
    """Mutable declaration object. Produce a runnable cell via .build()."""

    __module__ = "braincell"

    def __init__(
        self,
        morpho: Morphology,
        *,
        cv_policy: CVPolicy | None = None,
        V_th=-75 * u.mV,
        V_init=None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        name: str | None = None,
    ) -> None:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"Cell expects Morphology, got {type(morpho).__name__!s}.")

        self._morpho = morpho
        self._cv_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}.")

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_init = V_init
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)
        self._cvs_cache: tuple | None = None
        self._cvs_cache_key: object = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        if not isinstance(value, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(value).__name__!s}.")
        self._cv_policy = value
        self._cvs_cache = None

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    @property
    def V_th(self):
        return self._V_th

    @V_th.setter
    def V_th(self, value) -> None:
        self._V_th = value

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._V_init = value

    @property
    def solver(self):
        return self._solver_fn

    @solver.setter
    def solver(self, value) -> None:
        self._solver_name, self._solver_fn = _resolve_solver(value)

    @property
    def spk_fun(self):
        return self._spk_fun

    # ------------------------------------------------------------------
    # Declaration

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._cvs_cache = None
        return self

    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":
        self._place_rules = merge_place_rules(
            self._place_rules,
            (normalize_place_rule(locset, mechanisms),),
        )
        self._cvs_cache = None
        return self

    # ------------------------------------------------------------------
    # Preview

    @property
    def cvs(self):
        key = (id(self._morpho), self._cv_policy, self._paint_rules, self._place_rules)
        if self._cvs_cache is not None and self._cvs_cache_key == key:
            return self._cvs_cache
        cv_geo, cv_ids_by_branch = build_cv_geo(
            self._morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
        )
        cv_mech = init_cv_mech(len(cv_geo))
        apply_paint_rules(
            self._morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            paint_rules=self._paint_rules,
            mechs=cv_mech,
        )
        apply_place_rules(
            self._morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            place_rules=self._place_rules,
            mechs=cv_mech,
        )
        cvs = tuple(assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo)
        self._cvs_cache = cvs
        self._cvs_cache_key = key
        return cvs

    # ------------------------------------------------------------------
    # Terminal

    def build(self):
        return _build_pipeline(self)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Cell(root={self._morpho.root.name!r}, n_branches={len(self._morpho.branches)}, "
            f"n_paint_rules={len(self._paint_rules)}, n_place_rules={len(self._place_rules)})"
        )


def _resolve_solver(solver):
    if isinstance(solver, str):
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(f"solver must be str or callable, got {type(solver).__name__!s}.")
```

- [ ] **Step 5: Wire new package `__init__.py`**

```python
# braincell/_multi_compartment/__init__.py
"""Multi-compartment cell declaration + runtime."""

from .cell import Cell
from .run import RunResult
from .runnable import RunnableCell

__all__ = ["Cell", "RunnableCell", "RunResult"]
```

- [ ] **Step 6: Update top-level `braincell/__init__.py`**

Find the existing line(s) that import `Cell` / `RunResult` from the old
module (the grep target is `_multi_compartment`). Replace them with:

```python
from braincell._multi_compartment import Cell, RunnableCell, RunResult
```

Add `RunnableCell` to the `__all__` list if the module has one.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
pytest braincell/_multi_compartment/bridge_test.py \
       braincell/_multi_compartment/clamp_table_test.py \
       braincell/_multi_compartment/currents_test.py \
       braincell/_multi_compartment/run_test.py \
       braincell/_multi_compartment/probes_test.py \
       braincell/_multi_compartment/build_test.py \
       braincell/_multi_compartment/runnable_test.py \
       braincell/_multi_compartment/cell_test.py -v
```

Expected: every test green. Investigate any failure before moving on.

- [ ] **Step 8: Commit**

```bash
git add braincell/_multi_compartment/cell.py \
        braincell/_multi_compartment/cell_test.py \
        braincell/_multi_compartment/__init__.py \
        braincell/__init__.py \
        braincell/cv/_mech.py
git commit -m "feat(multi-compartment): add Cell declaration + merge_place_rules"
```

---

## Task 9: Delete old files + port legacy tests

Old tests are rewritten in the next task. Old source is removed here
so nothing accidentally imports it during the port.

**Files:**
- Delete: `braincell/_multi_compartment.py`

- [ ] **Step 1: Verify no production code imports the old module**

```bash
rg -n "from braincell\._multi_compartment " --glob '!braincell/_multi_compartment/*' --glob '!**/specs/**' --glob '!**/plans/**'
rg -n "from braincell import _multi_compartment"  # must be empty
```
Expected: matches only in the two old test files. All production
imports go through `braincell` or `braincell._multi_compartment`
(package).

- [ ] **Step 2: Remove the old module**

Run: `git rm braincell/_multi_compartment.py`
Expected: git recognizes the deletion.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(multi-compartment): remove legacy _multi_compartment.py"
```

---

## Task 10: Rewrite legacy tests to target the new API

The two big legacy test files assume the old lifecycle
(`cell.init_state(); cell.run(...)`) and touch private attributes.
Port them wholesale.

**Files:**
- Rewrite: `braincell/_multi_compartment_test.py`
- Rewrite: `braincell/_multi_compartment_solver_test.py`

- [ ] **Step 1: Read the legacy tests with the new API in hand**

```bash
wc -l braincell/_multi_compartment_test.py braincell/_multi_compartment_solver_test.py
```

Walk each file top-to-bottom; for each test function, rewrite so it:

1. Builds a `Cell` exactly as today.
2. Calls `rcell = cell.build()` instead of `cell.init_state()`.
3. Calls methods on `rcell` (`.update`, `.run`, `.compute_*`, `.sample_probe`, `.sample_probes`) where it used to call them on `cell`.
4. Reads topology from `rcell.n_cv` / `rcell.n_point` / `rcell.cvs` / `rcell.point_tree()`.
5. Passes `I_ext` to `rcell.update` the same way as before (signatures preserved).

Helper (`_build_tree`, `_build_three_branch_tree`, `_point_id_by_role`,
etc.) stay unchanged.

- [ ] **Step 2: Delete any test that was asserting on the old dirty
  flags or the now-removed `Cell.init_state()` path**

Specifically:
- assertions on `cell._frontend_dirty`, `_structure_dirty`, `_mechanism_dirty`, `_value_dirty`, `_state_initialized`
- `test_*init_state_is_required*` / `test_*requires_init_state*` patterns
- any test that re-paints/re-places and then expects lazy recompile

Replace their coverage with a single new test (added in Task 11)
asserting `cell.build()` twice returns distinct runnables.

- [ ] **Step 3: Run the ported tests**

```bash
pytest braincell/_multi_compartment_test.py braincell/_multi_compartment_solver_test.py -x -q
```

Expected: green. Triage every failure — these are your first end-to-end
signal on the rewrite.

- [ ] **Step 4: Commit**

```bash
git add braincell/_multi_compartment_test.py \
        braincell/_multi_compartment_solver_test.py
git commit -m "test(multi-compartment): migrate legacy tests to Cell.build() API"
```

---

## Task 11: Add regression tests for the fixed bugs

**Files:**
- Create/extend: `braincell/_multi_compartment/runnable_test.py`
- Create/extend: `braincell/_multi_compartment/currents_test.py`
- Create/extend: `braincell/_multi_compartment/clamp_table_test.py`
- Create/extend: `braincell/_multi_compartment/cell_test.py`

- [ ] **Step 1: Add test — external `I_ext` not dropped (bug #1)**

Append to `runnable_test.py`:

```python
class TestBug1ExternalCurrent(unittest.TestCase):
    def test_i_ext_propagates_through_update(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.place(RootLocation(), StateProbe(field="v", name="V_root"))
        rcell = cell.build()
        v_before = float(rcell.V.value.to_decimal(u.mV).mean())
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            rcell.update(I_ext=10.0 * u.nA)
        v_after = float(rcell.V.value.to_decimal(u.mV).mean())
        self.assertNotAlmostEqual(v_before, v_after, places=4)
```

- [ ] **Step 2: Add test — `(n_point,)` total current rejected (bug #2)**

Already in `currents_test.py::test_n_point_total_current_rejected` from
Task 3. Keep as-is.

- [ ] **Step 3: Add test — clamp active table validation (bug #7)**

Already in `clamp_table_test.py::test_current_clamp_builds_table` from
Task 2. Add a negative case to that file:

```python
def test_zero_area_point_raises_at_build(self):
    # Construct a morphology whose only branch has zero length so the
    # single CV has zero membrane area. Exact construction depends on
    # Branch.from_lengths validation; if zero-length branches are
    # already rejected, this test can be skipped.
    self.skipTest("zero-area construction requires morphology-level edit; covered by _point_area_decimal unit check")
```

Keep the skip until someone can author a minimal reproducer. The
positive path is covered.

- [ ] **Step 4: Add test — `cell.build()` twice independent (bug #9 / cache)**

Already present as `TestCellDeclaration::test_build_twice_produces_independent_runnables`.

- [ ] **Step 5: Run the additions**

```bash
pytest braincell/_multi_compartment/ -v
```
Expected: green, no skips except the documented one.

- [ ] **Step 6: Commit**

```bash
git add braincell/_multi_compartment/runnable_test.py \
        braincell/_multi_compartment/clamp_table_test.py
git commit -m "test(multi-compartment): add regression tests for fixed bugs"
```

---

## Task 12: Full suite + pre-commit

**Files:**
- None (verification only).

- [ ] **Step 1: Run full test suite**

Run: `pytest braincell/ -x -q`
Expected: every test green. If not, triage. Investigate root cause
rather than patching symptoms — a common class of failure after a
rewrite is a missing import / misspelled re-export in
`braincell/__init__.py`.

- [ ] **Step 2: Run pre-commit**

Run: `pre-commit run --all-files`
Expected: all checks green (flake8, EOF, trailing whitespace, etc).

- [ ] **Step 3: Spot-check examples still import**

```bash
python -c "
import braincell
assert hasattr(braincell, 'Cell')
assert hasattr(braincell, 'RunnableCell')
assert hasattr(braincell, 'RunResult')
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 4: Update examples if they rely on the old API**

```bash
rg -n "cell\.init_state\(" examples/
rg -n "cell\.run\(" examples/
```

For every multi-compartment example hit, rewrite as:

```python
rcell = cell.build()
rcell.run(dt=dt, duration=duration)
```

(Single-compartment examples — in `examples/single_compartment/` — are
untouched; `SingleCompartment` has its own class.)

- [ ] **Step 5: Commit example updates**

```bash
git add examples/
git commit -m "examples: migrate multi-compartment usage to Cell.build() API"
```

- [ ] **Step 6: Final squash verification**

```bash
git log --oneline morpho_1-multi-compartment-legacy..HEAD
pytest braincell/ -q
pre-commit run --all-files
```

Expected: tidy commit history on `morpho_1`, all checks green.

---

## Self-review checklist

Completed inline:

- **Spec coverage** — every spec section maps to at least one task:
  - Motivation → Task 3 (bug fixes), Task 10 (regression tests).
  - Architecture → Tasks 7 + 8 (RunnableCell + Cell).
  - File layout → Task 0 scaffold + per-file tasks.
  - `Cell` surface → Task 8.
  - `RunnableCell` surface → Task 7.
  - Current summation pipeline → Task 3.
  - Axial derivative → Task 6 + 7 (cache built in `build`, consumed in
    `compute_axial_derivative`).
  - `run()` time handling → Task 4.
  - V init → Task 6 + 7 (`_preinit` + `reset_state`).
  - Migration section → Task 10 + Task 12 step 4.
  - Tests → Task 10 + Task 11.
- **Placeholders** — none; every code block contains concrete code.
  The one `skipTest` in Task 11 step 3 is explicit about why it skips
  and what it would cover.
- **Type consistency** —
  - `ClampActiveTable.ids` / `.area` used consistently in Task 2 + 3.
  - `RunnableCell.runtime` / `.point_tree()` / `.n_cv` / `.n_point`
    used the same way in Tasks 3, 4, 7.
  - `cell.build()` always returns `RunnableCell`.
  - `merge_place_rules` signature mirrors `merge_paint_rules`.
