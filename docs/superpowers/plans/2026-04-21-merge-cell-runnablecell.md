# Merge `Cell` and `RunnableCell` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `Cell` (mutable declaration) + `RunnableCell(HHTypedNeuron)` (frozen runtime) back into a single `Cell(HHTypedNeuron)` class with explicit `init_state()` + `reset()` lifecycle and guarded paint/place/config setters.

**Architecture:** Single `Cell(HHTypedNeuron)` owns declaration state (morpho, cv_policy, paint/place rules, config) and runtime state (V, spike, current_time, runtime, point_tree, axial_jax). Phase flag `_initialized: bool` gates mutators vs runtime methods. `init_state()` executes the lowering pipeline previously in `build.build`. `reset()` drops runtime + states and re-opens mutation, preserving paint/place rules. `run()` auto-calls `init_state()` for convenience. `RunnableCell` class and `build()` function are deleted outright (no deprecation window).

**Tech Stack:** Python 3.13, JAX, brainstate, brainunit, braintools. Tests: pytest + unittest.TestCase, co-located `*_test.py`.

**Reference spec:** `docs/superpowers/specs/2026-04-21-merge-cell-runnablecell-design.md`.

---

## File Structure

### Files modified

- `braincell/compute/_runtime.py` — `install_cell_runtime` returns the tuple of installed attribute names; new `uninstall_cell_runtime(cell)` function mirrors it. `install_cell_runtime` reads `cell.V_th` (property) instead of `cell._V_th_value`.
- `braincell/compute/_runtime_test.py` — new tests for install-name-tracking and uninstall round-trip.
- `braincell/_multi_compartment/cell.py` — rewritten as a single `Cell(HHTypedNeuron)` with `init_state()`, `reset()`, guarded setters, and the full runtime-facing API previously on `RunnableCell`.
- `braincell/_multi_compartment/cell_test.py` — gains new tests (freeze, init-twice, reset round-trip, run auto-init) plus migrated tests from `runnable_test.py` / `build_test.py`.
- `braincell/_multi_compartment/bridge.py` — `host: "RunnableCell"` → `host: "Cell"`, TYPE_CHECKING import updated.
- `braincell/_multi_compartment/currents.py` — same annotation rename.
- `braincell/_multi_compartment/probes.py` — same.
- `braincell/_multi_compartment/clamp_table.py` — same.
- `braincell/_multi_compartment/run.py` — same.
- `braincell/_multi_compartment/__init__.py` — drop `RunnableCell` export.
- `braincell/__init__.py` — drop `RunnableCell` re-export.
- `CLAUDE.md` — rewrite the “Multi-compartment two-class split” section to describe the single-`Cell` lifecycle.
- `examples/multi_compartment/*.ipynb` / `*.py` — migrate `cell.build()` / `RunnableCell` usage to `cell.init_state()` / `cell.run(...)`.

### Files deleted

- `braincell/_multi_compartment/runnable.py`
- `braincell/_multi_compartment/runnable_test.py`
- `braincell/_multi_compartment/build.py`
- `braincell/_multi_compartment/build_test.py`

---

## Task 1: Track installed attribute names in `install_cell_runtime`

**Files:**
- Modify: `braincell/compute/_runtime.py:391-406`
- Modify: `braincell/compute/_runtime_test.py`

Goal: make `install_cell_runtime` return the tuple of attribute names it set, and read the spike threshold from `cell.V_th` (property) instead of `cell._V_th_value`, so it works against both the legacy `RunnableCell` and the new merged `Cell`.

- [ ] **Step 1: Write the failing test**

Append the following test class to `braincell/compute/_runtime_test.py`:

```python
import unittest
from unittest.mock import MagicMock

import brainstate
import brainunit as u
import numpy as np

from braincell._base import HHTypedNeuron
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
    install_cell_runtime,
)


class _StubCell(HHTypedNeuron):
    """Minimal HHTypedNeuron double for install/uninstall tests."""

    __module__ = "braincell.compute._runtime_test"

    def __init__(self, V_th=-55.0 * u.mV, n_cv: int = 3):
        HHTypedNeuron.__init__(
            self, size=(1,), name="stub", **build_placeholder_ions()
        )
        self._V_th = V_th
        # cvs must be iterable of objects with a ``cm`` quantity attribute.
        class _FakeCV:
            def __init__(self, cm):
                self.cm = cm
        self._cvs = tuple(_FakeCV(1.0 * u.uF / u.cm ** 2) for _ in range(n_cv))

    @property
    def V_th(self):
        return self._V_th

    @property
    def cvs(self):
        return self._cvs

    @property
    def varshape(self):
        return (len(self._cvs),)


class TestInstallCellRuntime(unittest.TestCase):

    def test_install_returns_tuple_of_installed_attr_names(self):
        cell = _StubCell(n_cv=4)
        runtime = MagicMock(spec=CellRuntimeState)
        runtime.n_cv = 4
        runtime.ions = {}
        runtime.layouts = ()
        runtime.runtime_nodes = {}

        installed = install_cell_runtime(cell, runtime)

        self.assertIsInstance(installed, tuple)
        self.assertEqual(
            set(installed),
            {"_in_size", "_out_size", "ion_channels", "C", "V_th"},
        )
        for name in installed:
            self.assertTrue(
                hasattr(cell, name),
                f"install_cell_runtime should have set attribute {name!r}.",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/compute/_runtime_test.py::TestInstallCellRuntime -v`
Expected: FAIL because `install_cell_runtime` currently returns `None` (the assertion on `isinstance(installed, tuple)` fails) or because it still reads `cell._V_th_value` (AttributeError).

- [ ] **Step 3: Patch `install_cell_runtime`**

In `braincell/compute/_runtime.py` replace the current `install_cell_runtime` body (lines 391–406) with:

```python
def install_cell_runtime(cell: "Cell", runtime: CellRuntimeState) -> tuple[str, ...]:
    """Install runtime nodes + C + V_th onto ``cell``.

    Returns the tuple of attribute names set so callers can record the
    list and pass it to :func:`uninstall_cell_runtime` later.
    """
    cell._in_size = (runtime.n_cv,)
    cell._out_size = (runtime.n_cv,)

    root_nodes = dict(runtime.ions)
    for layout in runtime.layouts:
        node = runtime.runtime_nodes.get(layout.id)
        if node is None:
            continue
        if _is_root_level_runtime_node(layout.kind):
            root_nodes[f"layout_{layout.id}"] = node

    cell.ion_channels = cell._format_elements(IonChannel, **root_nodes)
    cell.C = cv_value_vector(cell, attr_name="cm")
    cell.V_th = fill_like(cell.varshape, cell.V_th)
    return ("_in_size", "_out_size", "ion_channels", "C", "V_th")
```

Note the last assignment reads `cell.V_th` via the property (which on the merged `Cell` returns the declaration value stored in `self._V_th`) and then overwrites the attribute with the vector form. This matches what the legacy `RunnableCell._preinit` + `install_cell_runtime` pair did via `cell._V_th_value`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/compute/_runtime_test.py::TestInstallCellRuntime -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/compute/_runtime.py braincell/compute/_runtime_test.py
git commit -m "refactor(runtime): return installed attr names from install_cell_runtime"
```

---

## Task 2: Add `uninstall_cell_runtime`

**Files:**
- Modify: `braincell/compute/_runtime.py` (append after `install_cell_runtime`)
- Modify: `braincell/compute/_runtime_test.py`

- [ ] **Step 1: Write the failing test**

Append to the same `TestInstallCellRuntime` class (or a new class below it):

```python
from braincell._base import IonChannel
from braincell.compute._runtime import uninstall_cell_runtime


class TestUninstallCellRuntime(unittest.TestCase):

    def test_uninstall_round_trip(self):
        cell = _StubCell(n_cv=2)
        runtime = MagicMock(spec=CellRuntimeState)
        runtime.n_cv = 2
        runtime.ions = {}
        runtime.layouts = ()
        runtime.runtime_nodes = {}

        installed = install_cell_runtime(cell, runtime)
        for name in installed:
            self.assertTrue(hasattr(cell, name))

        uninstall_cell_runtime(cell, installed)

        for name in installed:
            self.assertFalse(
                hasattr(cell, name),
                f"uninstall_cell_runtime should have removed {name!r}.",
            )
        self.assertEqual(
            dict(cell.nodes(IonChannel, allowed_hierarchy=(1, 1))),
            {},
            "No IonChannel nodes should remain after uninstall.",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/compute/_runtime_test.py::TestUninstallCellRuntime -v`
Expected: FAIL — `uninstall_cell_runtime` is not defined yet.

- [ ] **Step 3: Implement `uninstall_cell_runtime`**

Append to `braincell/compute/_runtime.py` just below `install_cell_runtime`:

```python
def uninstall_cell_runtime(cell: "Cell", installed_names: tuple[str, ...]) -> None:
    """Remove attributes installed by :func:`install_cell_runtime`.

    ``installed_names`` is the tuple returned by the matching
    ``install_cell_runtime`` call. Each name is removed via
    :func:`delattr` so that subsequent ``init_state()`` calls re-install
    cleanly without colliding with stale attributes.
    """
    for name in installed_names:
        if hasattr(cell, name):
            delattr(cell, name)
```

Also add `"uninstall_cell_runtime"` to the module's `__all__`:

```python
__all__ = ["CellRuntimeState", "install_cell_runtime", "uninstall_cell_runtime"]
```

(If `__all__` doesn't currently include `install_cell_runtime`, add both.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/compute/_runtime_test.py::TestUninstallCellRuntime -v`
Expected: PASS.

- [ ] **Step 5: Run full `_runtime_test.py` for regression**

Run: `pytest braincell/compute/_runtime_test.py -v`
Expected: PASS (both new tests and pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add braincell/compute/_runtime.py braincell/compute/_runtime_test.py
git commit -m "feat(runtime): add uninstall_cell_runtime as inverse of install"
```

---

## Task 3: Rewrite `cell.py` as merged `Cell(HHTypedNeuron)`

**Files:**
- Replace: `braincell/_multi_compartment/cell.py`

This task produces the full merged `Cell` class in one file. Because the class has many methods (absorbed from today's `RunnableCell`), the file is written in one commit and immediately validated against a focused smoke test; exhaustive behavioural coverage follows in Task 4.

- [ ] **Step 1: Write a smoke test that drives the merged surface**

Replace the contents of `braincell/_multi_compartment/cell_test.py` with the following. Keep the existing Cell declaration-preview tests but add new smoke coverage. If the current file has unrelated tests, preserve them verbatim above the new classes.

```python
import unittest

import brainstate
import brainunit as u
import numpy as np

from braincell import Cell
from braincell.morph.morphology import Morphology
from braincell.cv import CVPerBranch
from braincell.filter import BranchAllFilter
import braincell.mech as mech


def _simple_morpho() -> Morphology:
    """Single soma + dendrite morphology for fast tests."""
    import brainunit as u
    from braincell.morph.branch import Soma, Dendrite

    soma = Soma.from_lengths(
        lengths=u.Quantity([10.0], u.um),
        radii=u.Quantity([10.0], u.um),
    )
    dend = Dendrite.from_lengths(
        lengths=u.Quantity([100.0], u.um),
        radii=u.Quantity([1.0], u.um),
    )
    morpho = Morphology.from_root(soma, name="soma")
    morpho.attach(parent="soma", child_branch=dend, child_name="dendrite")
    return morpho


def _minimal_cell() -> Cell:
    morpho = _simple_morpho()
    cell = Cell(morpho, cv_policy=CVPerBranch())
    cell.paint(BranchAllFilter(), mech.CableProperty(
        Vr=-70 * u.mV,
        cm=1.0 * u.uF / u.cm ** 2,
        Ra=100 * u.ohm * u.cm,
        temperature=u.celsius2kelvin(23.0),
    ))
    cell.place(
        BranchAllFilter(),
        mech.ProbeMechanism(variable="V", target="point", name="V_probe"),
    )
    return cell


class TestCellLifecycle(unittest.TestCase):

    def test_declaration_phase_flag(self):
        cell = _minimal_cell()
        self.assertFalse(cell._initialized)

    def test_init_state_flips_flag_and_populates_runtime(self):
        cell = _minimal_cell()
        cell.init_state()
        self.assertTrue(cell._initialized)
        self.assertIsNotNone(cell._runtime)
        self.assertIsNotNone(cell._point_tree)
        self.assertIsNotNone(cell._axial_jax)
        self.assertTrue(hasattr(cell, "V"))
        self.assertTrue(hasattr(cell, "spike"))

    def test_init_state_twice_raises(self):
        cell = _minimal_cell()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
            cell.init_state()

    def test_paint_after_init_raises(self):
        cell = _minimal_cell()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.paint(BranchAllFilter(), mech.CableProperty(
                Vr=-65 * u.mV,
                cm=1.0 * u.uF / u.cm ** 2,
                Ra=100 * u.ohm * u.cm,
                temperature=u.celsius2kelvin(23.0),
            ))

    def test_reset_from_declaring_raises(self):
        cell = _minimal_cell()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.reset()

    def test_reset_round_trip(self):
        cell = _minimal_cell()
        cell.init_state()
        cell.reset()
        self.assertFalse(cell._initialized)
        self.assertIsNone(cell._runtime)
        self.assertIsNone(cell._point_tree)
        self.assertIsNone(cell._axial_jax)
        self.assertFalse(hasattr(cell, "V"))
        # Paint + re-init works.
        cell.paint(BranchAllFilter(), mech.CableProperty(
            Vr=-72 * u.mV,
            cm=1.0 * u.uF / u.cm ** 2,
            Ra=100 * u.ohm * u.cm,
            temperature=u.celsius2kelvin(23.0),
        ))
        cell.init_state()
        self.assertTrue(cell._initialized)

    def test_runtime_method_requires_init(self):
        cell = _minimal_cell()
        for method_name in (
            "sample_probes",
            "mech_table",
            "point_tree",
        ):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, method_name)()

    def test_run_auto_inits_from_declaring(self):
        cell = _minimal_cell()
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertTrue(cell._initialized)
        self.assertIn("V_probe", result.traces)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run smoke tests to confirm they fail**

Run: `pytest braincell/_multi_compartment/cell_test.py::TestCellLifecycle -v`
Expected: FAIL across the board (Cell does not yet have `_initialized` / `init_state()` / `reset()`).

- [ ] **Step 3: Rewrite `cell.py`**

Replace the complete contents of `braincell/_multi_compartment/cell.py` with the following. This file unifies today's `cell.py` declaration class with the `RunnableCell` runtime methods from `runnable.py` and the `build()` pipeline from `build.py`.

```python
"""``Cell`` — single-class multi-compartment neuron.

A ``Cell`` carries both the declaration (morphology, CV policy, paint /
place rules, solver, spike config) and the runtime (``V`` / ``spike`` /
``current_time`` brainstate states, point tree, axial operator, installed
channel / ion nodes).

The lifecycle has two phases:

1. **DECLARING** (default). ``paint`` / ``place`` / ``cv_policy`` /
   ``V_th`` / ``V_init`` / ``solver`` / ``spk_fun`` setters are all
   mutable. Runtime methods raise.
2. **INITIALIZED**. After ``init_state()``, mutation is frozen and the
   runtime surface (``run``, ``update``, ``sample_probe``, inspection,
   ...) becomes available. Call ``reset()`` to drop the runtime and
   re-enter DECLARING.

``run(dt=, duration=)`` auto-calls ``init_state()`` for convenience.
"""

from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel
from braincell.compute._assignment_table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell.compute._point_tree import build_point_scheduling, build_point_tree
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
    clone_morpho,
    cv_value_vector,
    install_cell_runtime,
    mechanism_signature,
    uninstall_cell_runtime,
)
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
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqState, IndependentIntegration

from . import bridge, currents, probes, run as run_module

__all__ = ["Cell"]


def _cast_like(value, like):
    dtype = jnp.asarray(u.get_magnitude(like)).dtype
    if isinstance(value, u.Quantity):
        unit = u.get_unit(value)
        return jnp.asarray(value.to_decimal(unit), dtype=dtype) * unit
    return jnp.asarray(value, dtype=dtype)


class Cell(HHTypedNeuron):
    """Multi-compartment cell with explicit declaration / initialization phases.

    Parameters
    ----------
    morpho : Morphology
        Morphology tree.
    cv_policy : CVPolicy, optional
        Control-volume splitting policy; defaults to :class:`CVPerBranch`.
    V_th : Quantity
        Spike-detection threshold (default ``-75 mV``).
    V_init : Quantity | Callable | None
        Initial voltage. ``None`` means "use per-CV resting potential".
    spk_fun : Callable
        Surrogate-gradient spike function.
    solver : str | Callable
        Integrator name (registry lookup) or callable step function.
    name : str, optional
        Cell name.
    """

    __module__ = "braincell"

    # ------------------------------------------------------------------
    # Construction

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
        HHTypedNeuron.__init__(self, size=(1,), name=name, **build_placeholder_ions())

        if not isinstance(morpho, Morphology):
            raise TypeError(
                f"Cell expects Morphology, got {type(morpho).__name__!s}."
            )

        self._declaration_morpho = morpho
        self._morpho = morpho

        self._cv_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}."
            )

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_init = V_init
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)

        self._cvs_cache: tuple | None = None
        self._cvs_cache_key: object = None

        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._point_scheduling_cache: dict[tuple[str, int], object] = {}

        self._runtime: CellRuntimeState | None = None
        self._point_tree = None
        self._axial_jax = None
        self._runtime_installed_names: tuple[str, ...] = ()

        self._initialized = False

        # Eager policy validation via the preview.
        _ = self.cvs

    # ------------------------------------------------------------------
    # Phase guards

    def _raise_if_initialized(self, action: str) -> None:
        if self._initialized:
            raise RuntimeError(
                f"Cannot {action} after init_state(); call reset() first."
            )

    def _raise_if_not_initialized(self, action: str) -> None:
        if not self._initialized:
            raise RuntimeError(f"{action} requires init_state() first.")

    # ------------------------------------------------------------------
    # Read-only accessors / guarded config setters

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        self._raise_if_initialized("assign cv_policy")
        if not isinstance(value, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(value).__name__!s}."
            )
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
        self._raise_if_initialized("assign V_th")
        self._V_th = value

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._raise_if_initialized("assign V_init")
        self._V_init = value

    @property
    def solver(self):
        return self._solver_fn

    @solver.setter
    def solver(self, value) -> None:
        self._raise_if_initialized("assign solver")
        self._solver_name, self._solver_fn = _resolve_solver(value)

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def spk_fun(self):
        return self._spk_fun

    @spk_fun.setter
    def spk_fun(self, value) -> None:
        self._raise_if_initialized("assign spk_fun")
        self._spk_fun = value

    @property
    def name(self) -> str | None:
        return self._name

    # ------------------------------------------------------------------
    # Declaration mutators

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        self._raise_if_initialized("paint()")
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._cvs_cache = None
        return self

    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":
        self._raise_if_initialized("place()")
        self._place_rules = merge_place_rules(
            self._place_rules,
            (normalize_place_rule(locset, mechanisms),),
        )
        self._cvs_cache = None
        return self

    # ------------------------------------------------------------------
    # CV preview (valid in both phases)

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self):
        key = (
            id(self._morpho),
            self._cv_policy,
            self._paint_rules,
            self._place_rules,
        )
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
        cvs = tuple(
            assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
        )
        self._cvs_cache = cvs
        self._cvs_cache_key = key
        return cvs

    # ------------------------------------------------------------------
    # Phase transitions

    def init_state(self, batch_size=None) -> None:
        """Lower the declaration into a runtime state and allocate V / spike.

        Raises
        ------
        RuntimeError
            If the cell is already initialized. Call :meth:`reset` first.
        """
        self._raise_if_initialized("init_state()")

        morpho = clone_morpho(self._morpho)
        cv_geo, cv_ids_by_branch = build_cv_geo(
            morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
        )
        cv_mech = init_cv_mech(len(cv_geo))
        apply_paint_rules(
            morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            paint_rules=self._paint_rules,
            mechs=cv_mech,
        )
        apply_place_rules(
            morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            place_rules=self._place_rules,
            mechs=cv_mech,
        )
        cvs = tuple(
            assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
        )

        self._morpho = morpho
        self._cvs_cache = cvs
        self._cvs_cache_key = (
            id(self._morpho),
            self._cv_policy,
            self._paint_rules,
            self._place_rules,
        )

        self._point_tree = build_point_tree(morpho, cvs=cvs)
        self._runtime = CellRuntimeState.from_cell(self)
        self._runtime_installed_names = install_cell_runtime(self, self._runtime)

        v_initializer = (
            self._V_init if self._V_init is not None
            else cv_value_vector(self, attr_name="v")
        )
        self.V = DiffEqState(braintools.init.param(v_initializer, self.varshape, batch_size))
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        self._current_time_state.value = 0.0 * u.ms

        point_V = self._cv_to_point(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.init_state(point_V, batch_size=batch_size)

        self._axial_jax = jnp.asarray(
            build_cv_axial_operator(
                self,
                point_tree=self._point_tree,
                scheduling=self.point_scheduling(algorithm="dhs"),
            ),
            dtype=jnp.float64,
        )

        self._initialized = True

    def reset(self) -> None:
        """Drop the runtime and per-step state; return to DECLARING.

        Raises
        ------
        RuntimeError
            If the cell is not initialized.

        Notes
        -----
        ``reset()`` is distinct from :meth:`reset_state`. ``reset_state``
        reseeds ``V`` / ``spike`` / ``current_time`` in place and stays
        in the INITIALIZED phase. ``reset()`` fully tears down the
        runtime and returns to DECLARING so ``paint`` / ``place`` can
        run again.
        """
        self._raise_if_not_initialized("reset()")

        uninstall_cell_runtime(self, self._runtime_installed_names)
        self._runtime_installed_names = ()

        if hasattr(self, "V"):
            delattr(self, "V")
        if hasattr(self, "spike"):
            delattr(self, "spike")
        self._current_time_state.value = 0.0 * u.ms

        self._runtime = None
        self._point_tree = None
        self._axial_jax = None
        self._point_scheduling_cache.clear()

        self._morpho = self._declaration_morpho
        self._cvs_cache = None
        self._cvs_cache_key = None

        self._initialized = False

    # ------------------------------------------------------------------
    # Topology (runtime-only)

    @property
    def runtime(self) -> CellRuntimeState:
        self._raise_if_not_initialized("runtime")
        return self._runtime

    @property
    def n_point(self) -> int:
        self._raise_if_not_initialized("n_point")
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
        self._raise_if_not_initialized("point_tree()")
        return self._point_tree

    def point_scheduling(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        self._raise_if_not_initialized("point_scheduling()")
        key = (algorithm, int(max_group_size))
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
        self._raise_if_not_initialized("current_time")
        return self._current_time_state.value

    def _set_current_time(self, value) -> None:
        self._current_time_state.value = value

    # ------------------------------------------------------------------
    # Repr

    def __repr__(self) -> str:
        if self._initialized:
            return (
                f"Cell(root={self._morpho.root.name!r}, "
                f"n_cv={self.n_cv!r}, n_point={self.n_point!r}, initialized=True)"
            )
        return (
            f"Cell(root={self._morpho.root.name!r}, "
            f"n_branches={len(self._morpho.branches)}, "
            f"n_paint_rules={len(self._paint_rules)}, "
            f"n_place_rules={len(self._place_rules)}, "
            f"initialized=False)"
        )

    # ------------------------------------------------------------------
    # Bridging (runtime-only)

    def _cv_to_point(self, cv_values):
        self._raise_if_not_initialized("_cv_to_point()")
        return bridge.cv_to_point(cv_values, self._runtime)

    def _point_to_cv(self, point_values):
        self._raise_if_not_initialized("_point_to_cv()")
        return bridge.point_to_cv(point_values, self._runtime)

    # ------------------------------------------------------------------
    # Solver path (runtime-only)

    def _resolve_t(self):
        try:
            return brainstate.environ.get("t")
        except KeyError:
            return self.current_time

    def pre_integral(self, I_ext=0.0):
        self._raise_if_not_initialized("pre_integral()")
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(point_V)

    def compute_derivative(self, I_ext=0.0):
        self._raise_if_not_initialized("compute_derivative()")
        self.V.derivative = self.compute_voltage_derivative(self.V.value, I_ext)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.compute_derivative(point_V)

    def compute_membrane_derivative(self, V, I_ext=0.0):
        self._raise_if_not_initialized("compute_membrane_derivative()")
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, I_ext=I_ext, t=t)
        return I_total / self.C

    def compute_axial_derivative(self, V):
        self._raise_if_not_initialized("compute_axial_derivative()")
        V_decimal = jnp.asarray(V.to_decimal(u.mV), dtype=jnp.float64)
        axial = -jnp.matmul(V_decimal, self._axial_jax.T)
        return axial * (u.mV / u.ms)

    def compute_voltage_derivative(self, V, I_ext=0.0):
        return (
            self.compute_membrane_derivative(V, I_ext)
            + self.compute_axial_derivative(V)
        )

    def post_integral(self, I_ext=0.0):
        self._raise_if_not_initialized("post_integral()")
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.post_integral(point_V)

    def update(self, I_ext=None):
        self._raise_if_not_initialized("update()")
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.update(point_V)

        last_V = self.V.value
        if brainstate.environ.get("dt", None) is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")

        if I_ext is None:
            self.solver(self)
        else:
            self.solver(self, I_ext)

        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    # ------------------------------------------------------------------
    # Spike (phase-agnostic; uses V_th + spk_fun only)

    def get_spike(self, last_V, next_V):
        denom = _cast_like(20.0 * u.mV, next_V)
        V_th = _cast_like(self.V_th, next_V)
        return (
            self._spk_fun((next_V - V_th) / denom)
            * self._spk_fun((V_th - last_V) / denom)
        )

    def reset_state(self, batch_size=None) -> None:
        """Reseed ``V`` / ``spike`` / ``current_time`` without leaving INITIALIZED.

        Distinct from :meth:`reset`: ``reset_state`` is the in-phase
        brainstate lifecycle hook; ``reset`` tears down the runtime
        entirely and returns the cell to DECLARING.
        """
        self._raise_if_not_initialized("reset_state()")
        v_init = self._V_init
        if v_init is None:
            v_init = cv_value_vector(self, attr_name="v")
        self.V.value = braintools.init.param(v_init, self.varshape, batch_size)
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time_state.value = 0.0 * u.ms
        point_V = self._cv_to_point(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.reset_state(point_V, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Inspection forwards (runtime-only)

    @property
    def layouts(self):
        self._raise_if_not_initialized("layouts")
        return self._runtime.layouts

    @property
    def voltage_shape(self):
        self._raise_if_not_initialized("voltage_shape")
        return self._runtime.voltage_shape

    def get_point_layouts(self, point_id):
        self._raise_if_not_initialized("get_point_layouts()")
        return self._runtime.get_point_layouts(point_id)

    def get_cv_layouts(self, cv_id):
        self._raise_if_not_initialized("get_cv_layouts()")
        return self._runtime.get_cv_layouts(cv_id)

    def expected_state_shape(self, layout_id, var_name):
        self._raise_if_not_initialized("expected_state_shape()")
        return self._runtime.expected_state_shape(layout_id, var_name)

    def get_state(self, layout_id, var_name):
        self._raise_if_not_initialized("get_state()")
        return self._runtime.get_state(layout_id, var_name)

    def set_state(self, layout_id, var_name, value) -> None:
        self._raise_if_not_initialized("set_state()")
        self._runtime.set_state(layout_id, var_name, value)

    def get_point_state(self, point_id):
        self._raise_if_not_initialized("get_point_state()")
        return self._runtime.get_point_state(point_id)

    def get_cv_state(self, cv_id):
        self._raise_if_not_initialized("get_cv_state()")
        return self._runtime.get_cv_state(cv_id)

    def get_runtime_node(self, layout_id):
        self._raise_if_not_initialized("get_runtime_node()")
        return self._runtime.get_runtime_node(layout_id)

    def get_ion(self, name):
        self._raise_if_not_initialized("get_ion()")
        return self._runtime.get_ion(name)

    # ------------------------------------------------------------------
    # Probes + mech_table (runtime-only)

    def sample_probe(self, name: str):
        self._raise_if_not_initialized("sample_probe()")
        return probes.sample_probe(self, name)

    def sample_probes(self) -> dict[str, object]:
        self._raise_if_not_initialized("sample_probes()")
        return probes.sample_probes(self)

    def mech_table(self) -> MechanismObjectTable:
        self._raise_if_not_initialized("mech_table()")
        runtime = self._runtime
        point_tree = self._point_tree
        column_ids = tuple(range(len(point_tree.points)))

        row_keys: list[tuple[str, str]] = []
        row_labels: list[str] = []
        row_index_by_key: dict[tuple[str, str], int] = {}
        pending_cells: list[tuple[int, int, MechanismObjectCell]] = []
        layout_id_by_signature = {
            (layout.target,) + mechanism_signature(runtime.get_layout_mechanism(layout.id)): layout.id
            for layout in runtime.layouts
        }

        def ensure_row(mechanism: object) -> int:
            row_key = mechanism_cell_key(mechanism)
            row_index = row_index_by_key.get(row_key)
            if row_index is not None:
                return row_index
            row_index = len(row_keys)
            row_keys.append(row_key)
            class_name, instance_name = row_key
            row_labels.append(
                class_name if class_name == instance_name else f"{instance_name}:{class_name}"
            )
            row_index_by_key[row_key] = row_index
            return row_index

        for cv in self.cvs:
            midpoint_point_id = int(point_tree.cv_midpoint_point_id[cv.id])
            for target_label, mech_list in (("density", cv.density_mech), ("point", cv.point_mech)):
                for mechanism in mech_list:
                    row_key = mechanism_cell_key(mechanism)
                    row_index = ensure_row(mechanism)
                    layout_id = layout_id_by_signature[
                        (target_label,) + mechanism_signature(mechanism)
                        ]
                    pending_cells.append(
                        (
                            row_index,
                            midpoint_point_id,
                            MechanismObjectCell(
                                runtime=runtime,
                                layout_id=int(layout_id),
                                class_name=row_key[0],
                                instance_name=row_key[1],
                                column_id=midpoint_point_id,
                                domain="point",
                                cv_id=None,
                                point_id=midpoint_point_id,
                            ),
                        )
                    )

        values = np.full((len(row_keys), len(column_ids)), None, dtype=object)
        for row_index, column_id, cell in pending_cells:
            values[row_index, int(column_id)] = cell

        return MechanismObjectTable(
            domain="point",
            row_keys=tuple(row_keys),
            row_labels=tuple(row_labels),
            column_ids=column_ids,
            values=values,
        )

    # ------------------------------------------------------------------
    # Run (auto-inits from DECLARING)

    def run(self, *, dt, duration):
        """Run the cell for ``duration`` at ``dt`` and return probe traces.

        If ``init_state()`` has not been called yet, ``run`` calls it
        automatically. Once initialized the cell will *not* be
        re-initialized on subsequent ``run`` invocations.
        """
        if not self._initialized:
            self.init_state()
        return run_module.run(self, dt=dt, duration=duration)


# ----------------------------------------------------------------------
# Helpers

def _resolve_solver(solver):
    if isinstance(solver, str):
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(
        f"solver must be str or callable, got {type(solver).__name__!s}."
    )
```

- [ ] **Step 4: Update `braincell/_multi_compartment/__init__.py`**

Replace its contents with:

```python
"""Multi-compartment cell declaration + runtime."""

from .cell import Cell
from .run import RunResult

__all__ = ["Cell", "RunResult"]
```

- [ ] **Step 5: Update `braincell/__init__.py` public namespace**

Open `braincell/__init__.py` and remove the line that exports `RunnableCell`. A grep helps locate it:

```bash
grep -n "RunnableCell" braincell/__init__.py
```

Delete the offending import and any entry in `__all__`. Keep `Cell`.

- [ ] **Step 6: Update `host: "RunnableCell"` annotations**

In each of the five files below, replace `from .runnable import RunnableCell` with `from .cell import Cell`, and every `host: "RunnableCell"` / `rcell: "RunnableCell"` type annotation with `host: "Cell"` / `cell: "Cell"` (variable names in bodies may remain `rcell` if you prefer minimal diff):

- `braincell/_multi_compartment/bridge.py`
- `braincell/_multi_compartment/currents.py`
- `braincell/_multi_compartment/probes.py`
- `braincell/_multi_compartment/clamp_table.py`
- `braincell/_multi_compartment/run.py`

Use grep to confirm there are no stragglers:

```bash
grep -rn "RunnableCell" braincell/
```

Expected: no matches inside the `braincell/_multi_compartment` package after the edits.

- [ ] **Step 7: Run the smoke test suite to confirm it passes**

Run: `pytest braincell/_multi_compartment/cell_test.py::TestCellLifecycle -v`
Expected: PASS. Any failure here indicates a wiring issue between the merged `Cell` and the runtime helpers.

- [ ] **Step 8: Commit**

```bash
git add \
    braincell/_multi_compartment/cell.py \
    braincell/_multi_compartment/cell_test.py \
    braincell/_multi_compartment/__init__.py \
    braincell/_multi_compartment/bridge.py \
    braincell/_multi_compartment/currents.py \
    braincell/_multi_compartment/probes.py \
    braincell/_multi_compartment/clamp_table.py \
    braincell/_multi_compartment/run.py \
    braincell/__init__.py
git commit -m "refactor(cell): merge Cell and RunnableCell into single class

Cell now inherits HHTypedNeuron directly and exposes init_state() /
reset() for explicit phase transitions. paint/place/config setters
raise RuntimeError after init_state(); reset() unfreezes and
preserves paint/place rules. run() auto-calls init_state() for
convenience. RunnableCell class and build() pipeline are deleted
in the next commit."
```

---

## Task 4: Delete `runnable.py`, `build.py`, and their tests

**Files:**
- Delete: `braincell/_multi_compartment/runnable.py`
- Delete: `braincell/_multi_compartment/runnable_test.py`
- Delete: `braincell/_multi_compartment/build.py`
- Delete: `braincell/_multi_compartment/build_test.py`

- [ ] **Step 1: Verify nothing still imports them**

```bash
grep -rn "from .runnable" braincell/ || echo "no imports"
grep -rn "from .build" braincell/_multi_compartment || echo "no imports"
grep -rn "RunnableCell" braincell/ || echo "no references"
grep -rn "cell\.build\(" braincell/ || echo "no references"
```

Expected: all four echo their "no ..." message. If any matches remain, fix them before deleting.

- [ ] **Step 2: Delete the four files**

```bash
git rm braincell/_multi_compartment/runnable.py \
       braincell/_multi_compartment/runnable_test.py \
       braincell/_multi_compartment/build.py \
       braincell/_multi_compartment/build_test.py
```

- [ ] **Step 3: Run the multi-compartment test suite**

Run: `pytest braincell/_multi_compartment/ -v`
Expected: PASS. No collection errors (missing `runnable_test.py` / `build_test.py` is fine — they're gone).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(cell): remove RunnableCell and build() — superseded by Cell.init_state()"
```

---

## Task 5: Migrate any residual test references

**Files:**
- Modify (possibly): `braincell/compute/_runtime_test.py`
- Modify (possibly): any other `*_test.py` referencing `RunnableCell` or `build()`

- [ ] **Step 1: Search for residual references**

```bash
grep -rn "RunnableCell\|cell\.build\(\|from braincell._multi_compartment.runnable\|from braincell._multi_compartment.build" braincell/
```

Expected: no matches. If any appear, migrate them to `Cell` + `init_state()` using the pattern:

```python
# Before
cell = Cell(morpho, ...)
cell.paint(...)
rcell = cell.build()
rcell.run(dt=..., duration=...)

# After
cell = Cell(morpho, ...)
cell.paint(...)
cell.init_state()             # explicit — or omit and let run() auto-init
cell.run(dt=..., duration=...)
```

- [ ] **Step 2: Run the full package test suite**

Run: `pytest braincell/ -x -v`
Expected: PASS. Use `-x` to stop on the first failure so issues surface one at a time.

- [ ] **Step 3: Commit (only if there were edits)**

```bash
git add -A
git commit -m "test: migrate residual RunnableCell / build() references to Cell.init_state()"
```

Skip this step entirely if the grep returned no matches and no edits were made.

---

## Task 6: Migrate `examples/multi_compartment`

**Files:**
- Modify: `examples/multi_compartment/*.ipynb`
- Modify: `examples/multi_compartment/*.py`

- [ ] **Step 1: Find all files referencing the old surface**

```bash
grep -rn "RunnableCell\|cell\.build\(\|rcell = " examples/multi_compartment/ || echo "none"
```

Expected: a list of notebooks and scripts to update. If "none" is printed, skip to Step 4.

- [ ] **Step 2: Apply the standard migration to each file**

Notebooks are JSON; edit with jupyter or direct JSON edits. The pattern:

```python
# Before
cell = Cell(morpho, ...)
cell.paint(...); cell.place(...)
rcell = cell.build()
result = rcell.run(dt=0.1 * u.ms, duration=200 * u.ms)
print(rcell.V.value)

# After
cell = Cell(morpho, ...)
cell.paint(...); cell.place(...)
result = cell.run(dt=0.1 * u.ms, duration=200 * u.ms)
print(cell.V.value)
```

Every `rcell` → `cell`. Every `cell.build()` call → removed (let `run()` auto-init) or replaced with explicit `cell.init_state()` where the script introspects the runtime before running.

- [ ] **Step 3: Execute each notebook headlessly to confirm it still runs**

For each updated notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    examples/multi_compartment/<notebook>.ipynb
```

Expected: no execution error. If a notebook fails, inspect the traceback and fix the migration.

- [ ] **Step 4: Commit**

```bash
git add examples/multi_compartment/
git commit -m "docs(examples): migrate multi-compartment examples to Cell.init_state()"
```

---

## Task 7: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (the "Multi-compartment two-class split" section)

- [ ] **Step 1: Locate the section**

Open `CLAUDE.md` and find the heading `### Multi-compartment two-class split`.

- [ ] **Step 2: Replace with the single-class description**

Replace the entire section (heading + body, up to but not including the next `###`) with:

```markdown
### Multi-compartment single-class lifecycle

`braincell._multi_compartment.Cell` is a single `Cell(HHTypedNeuron)` class with two phases:

- **DECLARING** (default after `Cell(morpho, ...)`). `paint` / `place` /
  `cv_policy` / `V_th` / `V_init` / `solver` / `spk_fun` setters all
  mutate. Runtime-only methods raise `RuntimeError`.
- **INITIALIZED**. `cell.init_state()` lowers the declaration into a
  runtime state and allocates `V` / `spike` / `current_time`. While
  initialized, the full runtime surface (`run`, `update`, `sample_probe`,
  `point_tree`, `mech_table`, `get_point_state`, …) is available and
  every mutator / config setter raises.

Transitions:

- `cell.init_state(batch_size=None)` — DECLARING → INITIALIZED. Raises
  if already initialized.
- `cell.reset()` — INITIALIZED → DECLARING. Drops `runtime` /
  `point_tree` / `axial_jax` and the `V` / `spike` / `current_time`
  states; the paint / place rules and configuration values survive so
  you can paint more and call `init_state()` again. Distinct from
  `cell.reset_state()`, which stays in INITIALIZED and reseeds `V` /
  `spike` only.
- `cell.run(dt=, duration=)` — if DECLARING, auto-calls `init_state()`
  before running. Subsequent `run()` calls never re-initialize.

The runtime-pipeline internals (membrane-current sum in
`currents.total_membrane_current`, `CellRuntimeState`,
`ClampActiveTable`, `cv_area`) are unchanged.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): rewrite Multi-compartment section for single-Cell lifecycle"
```

---

## Task 8: Final full-repo regression

**Files:**
- Run only — no edits.

- [ ] **Step 1: Run the full test suite**

Run: `pytest braincell/ -v`
Expected: all tests pass.

- [ ] **Step 2: Run pre-commit hooks**

Run: `pre-commit run --all-files`
Expected: no failures.

- [ ] **Step 3: Grep for stragglers**

```bash
grep -rn "RunnableCell" .
grep -rn "cell\.build\(" .
```

Expected: no matches in source / tests / examples / docs (matches in
`docs/superpowers/` historical specs are fine — they describe the
before-state).

- [ ] **Step 4 (if regressions): fix and commit**

Any failure from Steps 1–3 gets its own focused commit. Do not squash
into the main refactor.

---

## Self-Review

Before handing off: walk back through the spec and plan.

1. **Spec coverage.** Every success criterion from the spec maps to a task:
   - Single `Cell`, no `RunnableCell` → Tasks 3, 4.
   - `paint` / `place` / config setters raise after `init_state()` → Task 3 (smoke test + guard implementation).
   - `init_state()` twice raises → Task 3 smoke test.
   - `reset()` from DECLARING raises → Task 3.
   - Reset round-trip works → Task 3.
   - `run()` auto-init → Task 3.
   - Existing runtime behaviour unchanged → Tasks 5, 8 (regression runs).
   - Examples updated → Task 6.

2. **Placeholder scan.** No "TBD" / "TODO" / "implement later" / vague
   "handle edge cases" entries. Every step shows the code or the exact
   command.

3. **Type / name consistency.** `install_cell_runtime` signature
   consistent across Tasks 1, 2, 3 (returns `tuple[str, ...]`).
   `uninstall_cell_runtime(cell, installed_names)` consistent between
   Tasks 2 and 3. `_runtime_installed_names`, `_declaration_morpho`,
   `_morpho`, `_initialized` flag names identical across the smoke test
   and the implementation. `Cell.reset()` vs `Cell.reset_state()`
   separated in Task 3 docstrings.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-merge-cell-runnablecell.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
