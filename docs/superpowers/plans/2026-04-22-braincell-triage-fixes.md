# braincell Triage Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two Critical and four High-severity findings from the
2026-04-22 triage review (`docs/superpowers/specs/2026-04-22-braincell-triage-review-report.md`).
Preserve behavior outside the broken paths; each fix lands with a failing
regression test that passes afterwards.

**Architecture:** TDD, one finding per task. Every task is a self-contained
commit: failing test → minimal code change → test passes → commit. No
cross-task refactors. Existing assertion-based contract tests that encode the
old behavior are updated in the same commit that changes the behavior, so the
suite is green at every commit boundary.

**Tech Stack:** Python 3, JAX, brainstate, brainunit, pytest / unittest.

**Scope note:** Six findings below — CRIT-01, CRIT-02, HIGH-01, HIGH-02,
HIGH-03, HIGH-04. Medium/Low/Arch findings are out of scope for this plan.

**File map:**

| File | Why it changes |
|------|----------------|
| `braincell/_base.py` | CRIT-01 (line 1193-1196), CRIT-02 (line 781-807), HIGH-02 (line 924-926) |
| `braincell/_single_compartment/base.py` | HIGH-01 (line 229-234) |
| `braincell/quad/_exp_euler.py` | HIGH-03 (line 188-191) |
| `braincell/quad/_staggered.py` | HIGH-03 (line 135-138), HIGH-04 (line 510-525, 545-558) |
| `braincell/_base_test.py` | Grow from placeholder to cover CRIT-01, CRIT-02, HIGH-02 |
| `braincell/_single_compartment/base_test.py` | Extend `SingleCompartmentComputeDerivativeTest` for HIGH-01 |
| `braincell/quad/_exp_euler_test.py` | Add `ExpEulerGuardTest` for HIGH-03 |
| `braincell/quad/_staggered_test.py` | Update existing tests that assert `AssertionError` → `TypeError`/`ValueError` for HIGH-03, HIGH-04 |

---

## Task 1: CRIT-01 — `MixIons.update` passes wrong receiver

**Files:**
- Modify: `braincell/_base.py:1193-1196`
- Modify: `braincell/_base_test.py` (add `MixIonsUpdateReceiverTest`)

**Why it is a bug:** `brainstate.graph.nodes(node, *filters, ...)` takes the
graph root as the first positional argument. `MixIons.update` calls
`brainstate.graph.nodes(IonChannel, allowed_hierarchy=(1, 1))` — `IonChannel`
is interpreted as the *root*, not a type filter, so iteration skips every
channel attached to this `MixIons` instance. Compare `Ion.update`
(`_base.py:847`) which correctly passes `self`.

- [ ] **Step 1: Write the failing test**

Replace the placeholder content of `braincell/_base_test.py` with the
following (imports + one new test class). The test installs a recording
subclass of `Channel` under a real `MixIons`, calls `update`, and asserts
the channel's `update` method was invoked with `V` and per-ion `IonInfo`s.

```python
# braincell/_base_test.py
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# (license header unchanged)

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell import Channel, HHTypedNeuron, Ion, IonInfo, MixIons
from braincell.ion import CalciumFixed, PotassiumFixed, SodiumFixed


class _RecordingKCaChannel(Channel):
    """Records every call to ``update`` so the test can assert dispatch."""

    root_type = brainstate.mixin.JointTypes[PotassiumFixed, CalciumFixed]

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self.calls = []

    def update(self, V, K: IonInfo, Ca: IonInfo):
        self.calls.append((V, K, Ca))

    def init_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def reset_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def compute_derivative(self, V, K, Ca):  # pragma: no cover
        pass

    def current(self, V, K, Ca):  # pragma: no cover
        return 0.0 * u.nA / u.cm ** 2


class MixIonsUpdateReceiverTest(unittest.TestCase):
    """Regression for CRIT-01: MixIons.update iterated the wrong graph."""

    def test_update_reaches_child_channel(self) -> None:
        k = PotassiumFixed(size=1)
        ca = CalciumFixed(size=1)
        mix = MixIons(k, ca)
        rec = _RecordingKCaChannel(size=1)
        mix.add(kca=rec)

        V = jnp.zeros((1,)) * u.mV
        mix.update(V)

        self.assertEqual(len(rec.calls), 1, "child channel must see exactly one update")
        seen_V, seen_K, seen_Ca = rec.calls[0]
        self.assertIsInstance(seen_K, IonInfo)
        self.assertIsInstance(seen_Ca, IonInfo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_base_test.py::MixIonsUpdateReceiverTest -v`
Expected: `FAIL`. The assert-length check reports `0 != 1` because
`brainstate.graph.nodes(IonChannel, ...)` treats `IonChannel` as a graph
root and iterates nothing.

- [ ] **Step 3: Fix `MixIons.update`**

Replace `braincell/_base.py:1193-1196` with:

```python
    def update(self, V, *args, **kwargs):
        for key, node in brainstate.graph.nodes(self, IonChannel, allowed_hierarchy=(1, 1)).items():
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.update(V, *infos)
```

Only change: insert `self` as the first positional argument to
`brainstate.graph.nodes`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/_base_test.py::MixIonsUpdateReceiverTest -v`
Expected: `PASS`.

Run regression: `pytest braincell/_base_test.py -v`
Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add braincell/_base.py braincell/_base_test.py
git commit -m "fix(base): pass self to MixIons.update graph walk (CRIT-01)"
```

---

## Task 2: CRIT-02 — `Ion.current(include_external=True)` crashes on empty channels

**Files:**
- Modify: `braincell/_base.py:781-807`
- Modify: `braincell/_base_test.py` (append `IonCurrentExternalOnlyTest`)

**Why it is a bug:** When `nodes` is empty, `current` stays `None`. The
`include_external` loop then executes `current = current + node(V, ion_info)`
which raises `TypeError: unsupported operand type(s) for +: 'NoneType' and
'Quantity'`. Reachable whenever `MixIons.add` registers an external current on
an Ion that has no native channels of its own.

- [ ] **Step 1: Write the failing test**

Append to `braincell/_base_test.py`:

```python
class IonCurrentExternalOnlyTest(unittest.TestCase):
    """Regression for CRIT-02: Ion.current crashed with empty nodes."""

    def test_external_only_returns_sum_without_crashing(self) -> None:
        na = SodiumFixed(size=1)  # no native channels

        expected = 1.5 * u.nA / u.cm ** 2
        na.register_external_current(
            "probe",
            lambda V, ion_info: u.math.broadcast_to(expected, V.shape),
        )

        V = jnp.zeros((1,)) * u.mV
        out = na.current(V, include_external=True)

        self.assertTrue(
            u.math.allclose(
                out.to_decimal(u.nA / u.cm ** 2),
                expected.to_decimal(u.nA / u.cm ** 2),
                atol=1e-9,
            )
        )

    def test_external_only_without_request_returns_none(self) -> None:
        na = SodiumFixed(size=1)
        na.register_external_current(
            "probe",
            lambda V, ion_info: 1.0 * u.nA / u.cm ** 2,
        )
        V = jnp.zeros((1,)) * u.mV
        self.assertIsNone(na.current(V, include_external=False))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_base_test.py::IonCurrentExternalOnlyTest -v`
Expected: `test_external_only_returns_sum_without_crashing` FAILs with
`TypeError: unsupported operand type(s) for +: 'NoneType' ...`.

- [ ] **Step 3: Fix `Ion.current`**

Replace the body of `Ion.current` at `braincell/_base.py:781-807` with:

```python
    def current(self, V, include_external: bool = False):
        """
        Generate ion channel current.

        This method calculates the total current from all channels and optionally includes external currents.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            include_external (bool): If True, include external currents in the calculation. Default is False.

        Returns:
            array-like: The total current generated by all channels (and external currents if included).
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        ion_info = self.pack_info()
        current = None
        if len(nodes) > 0:
            for node in nodes:
                node: Channel
                new_current = node.current(V, ion_info)
                current = new_current if current is None else (current + new_current)
        if include_external and self._external_currents:
            for key, node in self._external_currents.items():
                node: Callable
                contrib = node(V, ion_info)
                current = contrib if current is None else (current + contrib)
        return current
```

Only behavioral change: the external-current accumulator reuses the same
"first contrib bootstraps `current`" pattern as the native loop, so the
`None + Quantity` path is impossible.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/_base_test.py::IonCurrentExternalOnlyTest -v`
Expected: `PASS`.

Guard against regressions in the non-empty path:

Run: `pytest braincell/_single_compartment/base_test.py -v`
Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add braincell/_base.py braincell/_base_test.py
git commit -m "fix(base): Ion.current external loop no longer crashes on empty channels (CRIT-02)"
```

---

## Task 3: HIGH-01 — `SingleCompartment.compute_derivative` loses exception chain

**Files:**
- Modify: `braincell/_single_compartment/base.py:229-234`
- Modify: `braincell/_single_compartment/base_test.py` (extend
  `test_bad_channel_current_is_wrapped_in_value_error`)

**Why it is a bug:** The `raise ValueError(...)` inside the `except Exception
as e` block is missing `from e`. The `__cause__` traceback link is lost,
making debugging harder and hiding the original exception type. The
multi-compartment counterpart (`_multi_compartment/currents.py:52`) uses
`from exc` correctly; the single-compartment version diverged.

- [ ] **Step 1: Extend the existing test to assert the cause chain**

Edit `braincell/_single_compartment/base_test.py` at the existing
`test_bad_channel_current_is_wrapped_in_value_error` (around line 455-461)
to additionally assert that `ctx.exception.__cause__` is the original
`RuntimeError`:

```python
    def test_bad_channel_current_is_wrapped_in_value_error(self) -> None:
        sc = SingleCompartment(size=1, bad=_BadChannel(size=1))
        sc.init_state()
        with self.assertRaises(ValueError) as ctx:
            sc.compute_derivative(0.0 * u.nA / u.cm ** 2)
        self.assertIn("bad", str(ctx.exception))
        self.assertIn("intentional bad current", str(ctx.exception))
        # Regression: HIGH-01 — preserve the original exception as __cause__.
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("intentional bad current", str(ctx.exception.__cause__))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_single_compartment/base_test.py::SingleCompartmentComputeDerivativeTest::test_bad_channel_current_is_wrapped_in_value_error -v`
Expected: `FAIL` — `__cause__` is `None` because the `raise` has no
`from e`.

- [ ] **Step 3: Fix the raise site**

In `braincell/_single_compartment/base.py:229-234`, change:

```python
            except Exception as e:
                raise ValueError(
                    f"Error in computing current for ion channel '{key}': \n"
                    f"{ch}\n"
                    f"Error: {e}"
                )
```

to:

```python
            except Exception as e:
                raise ValueError(
                    f"Error in computing current for ion channel '{key}': \n"
                    f"{ch}\n"
                    f"Error: {e}"
                ) from e
```

Only change: append `from e`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/_single_compartment/base_test.py::SingleCompartmentComputeDerivativeTest -v`
Expected: `PASS` (all tests in that class).

- [ ] **Step 5: Commit**

```bash
git add braincell/_single_compartment/base.py braincell/_single_compartment/base_test.py
git commit -m "fix(single_compartment): preserve exception chain in compute_derivative wrapper (HIGH-01)"
```

---

## Task 4: HIGH-02 — `Ion.add` accepts any object as "channel"

**Files:**
- Modify: `braincell/_base.py:924-926`
- Modify: `braincell/_base_test.py` (append `IonAddChannelValidationTest`)

**Why it is a bug:** `self._format_elements(object, **elements)` short-circuits
the isinstance check (`isinstance(x, object)` is always `True`). Compare
`Ion.__init__` (`_base.py:719`) which passes `Channel`. `check_hierarchies`
earlier on line 925 enforces `root_type`, but only objects that *have* a
`root_type` attribute — arbitrary non-Channel objects with a matching
`root_type` class-attr would slip through.

- [ ] **Step 1: Write the failing test**

Append to `braincell/_base_test.py`:

```python
class _FakeChannelLike:
    """Not a Channel subclass, but has a root_type attribute."""

    root_type = HHTypedNeuron


class IonAddChannelValidationTest(unittest.TestCase):
    """Regression for HIGH-02: Ion.add must reject non-Channel objects."""

    def test_add_rejects_non_channel_object_even_with_root_type(self) -> None:
        na = SodiumFixed(size=1)
        with self.assertRaises(TypeError):
            na.add(fake=_FakeChannelLike())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest braincell/_base_test.py::IonAddChannelValidationTest -v`
Expected: `FAIL` — no exception is raised because `object` accepts
`_FakeChannelLike()`.

- [ ] **Step 3: Fix `Ion.add`**

In `braincell/_base.py:924-926`, change:

```python
        self.check_hierarchies(type(self), **elements)
        self.channels.update(self._format_elements(object, **elements))
```

to:

```python
        self.check_hierarchies(type(self), **elements)
        self.channels.update(self._format_elements(Channel, **elements))
```

Only change: `object` → `Channel`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest braincell/_base_test.py::IonAddChannelValidationTest -v`
Expected: `PASS`.

Regression: `pytest braincell/_base_test.py braincell/_single_compartment/base_test.py -v`
Expected: `PASS` — the happy path (`sc = SingleCompartment(size=1,
bad=_BadChannel(size=1))` in the existing test file) continues to work
because `_BadChannel` *is* a `Channel`.

- [ ] **Step 5: Commit**

```bash
git add braincell/_base.py braincell/_base_test.py
git commit -m "fix(base): Ion.add validates elements as Channel not object (HIGH-02)"
```

---

## Task 5: HIGH-03 — Integrator arg validation via `assert`

**Files:**
- Modify: `braincell/quad/_exp_euler.py:188-191`
- Modify: `braincell/quad/_staggered.py:135-138`
- Modify: `braincell/quad/_staggered_test.py` (lines 167, 174 — update
  existing tests that currently assert `AssertionError`)
- Modify: `braincell/quad/_exp_euler_test.py` (append
  `ExpEulerStepGuardTest`)

**Why it is a bug:** `assert isinstance(target, HHTypedNeuron)` is stripped
under `python -O`. A wrong target then fails deeper with a less helpful
error. Docstring currently advertises `AssertionError`; we migrate to
`TypeError` and the existing tests that encode the old contract are updated
in the same commit so the suite stays green.

- [ ] **Step 1: Write the failing test for `exp_euler_step`**

Append to `braincell/quad/_exp_euler_test.py` (at end of file, no existing
guard test exists):

```python
class ExpEulerStepGuardTest(unittest.TestCase):
    """Regression for HIGH-03: exp_euler_step must raise TypeError, not
    AssertionError, so the contract survives ``python -O``."""

    def test_rejects_non_hh_typed_neuron(self):
        from braincell.quad import exp_euler_step

        with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
            with self.assertRaises(TypeError) as ctx:
                exp_euler_step(object())
        self.assertIn("HHTypedNeuron", str(ctx.exception))
```

The test needs `brainstate` and `u` imported at the top of the file — they
already are (lines 19-20).

- [ ] **Step 2: Update existing `staggered_step` guard tests**

In `braincell/quad/_staggered_test.py`, replace lines 161-176
(`StaggeredStepGuardTest`) with:

```python
class StaggeredStepGuardTest(unittest.TestCase):

    def test_rejects_plain_module(self):
        class Plain(brainstate.nn.Module):
            pass

        # HIGH-03: TypeError (not AssertionError) so -O preserves the contract.
        with self.assertRaises(TypeError):
            staggered_step(Plain())

    def test_error_message_mentions_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaises(TypeError) as ctx:
            staggered_step(Plain())
        self.assertIn(DiffEqModule.__name__, str(ctx.exception))
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `pytest braincell/quad/_exp_euler_test.py::ExpEulerStepGuardTest braincell/quad/_staggered_test.py::StaggeredStepGuardTest -v`
Expected: `FAIL` — current code raises `AssertionError`, tests expect
`TypeError`.

- [ ] **Step 4: Fix `exp_euler_step`**

In `braincell/quad/_exp_euler.py:188-191`, change:

```python
    assert isinstance(target, HHTypedNeuron), (
        f"The target should be a {HHTypedNeuron.__name__}. "
        f"But got {type(target)} instead."
    )
```

to:

```python
    if not isinstance(target, HHTypedNeuron):
        raise TypeError(
            f"The target should be a {HHTypedNeuron.__name__}. "
            f"But got {type(target)} instead."
        )
```

- [ ] **Step 5: Fix `staggered_step`**

In `braincell/quad/_staggered.py:135-138`, change:

```python
    assert isinstance(target, DiffEqModule), (
        f"The stagger integrator only support {DiffEqModule.__name__}, "
        f"but we got {type(target)} instead."
    )
```

to:

```python
    if not isinstance(target, DiffEqModule):
        raise TypeError(
            f"The stagger integrator only support {DiffEqModule.__name__}, "
            f"but we got {type(target)} instead."
        )
```

- [ ] **Step 6: Run tests to verify pass**

Run: `pytest braincell/quad/_exp_euler_test.py braincell/quad/_staggered_test.py -v`
Expected: `PASS` — both guard tests plus untouched body tests.

- [ ] **Step 7: Commit**

```bash
git add braincell/quad/_exp_euler.py braincell/quad/_staggered.py \
        braincell/quad/_exp_euler_test.py braincell/quad/_staggered_test.py
git commit -m "fix(quad): integrator arg validation raises TypeError not AssertionError (HIGH-03)"
```

---

## Task 6: HIGH-04 — DHS kernel contract checks via `assert`

**Files:**
- Modify: `braincell/quad/_staggered.py:510-525` (`_check_comp_triang`)
- Modify: `braincell/quad/_staggered.py:545-558` (`_check_comp_backsub`)
- Modify: `braincell/quad/_staggered_test.py` (lines 84-94, 109-117, 122-128
  — update existing tests that expect `AssertionError`)

**Why it is a bug:** Kernel contract checks (unit-lessness, rank, shape) use
`assert`. Under `python -O` they evaporate, and a wrong caller runs the
kernel on garbage inputs without any signal. Internal-only call-sites make
the blast radius smaller than HIGH-03, but contract checks are defensive for
a reason.

- [ ] **Step 1: Update existing assertion-based tests**

In `braincell/quad/_staggered_test.py`:

**Replace lines 84-94** (`test_kernel_contract_violation_on_wrong_rank`):

```python
    def test_kernel_contract_violation_on_wrong_rank(self):
        # ``diags`` must be 2D — passing a 1D array trips the contract check.
        # HIGH-04: raises ValueError (not AssertionError) under ``python -O``.
        with self.assertRaises(ValueError):
            comp_triang_raw(
                jnp.array([1.0]),
                jnp.array([[1.0]]),
                jnp.array([0.0]),
                jnp.array([0.0]),
                jnp.empty((0, 2), dtype=jnp.int32),
                np.array([0], dtype=np.int32),
            )
```

**Replace lines 109-117** (`test_kernel_contract_violation_on_dimful_diags`):

```python
    def test_kernel_contract_violation_on_dimful_diags(self):
        diags = jnp.array([[2.0]]) * u.mV
        solves = jnp.array([[1.0]]) * u.mV
        lowers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        uppers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        edges = jnp.empty((0, 2), dtype=jnp.int32)
        level_offsets = np.array([0], dtype=np.int32)
        with self.assertRaises(ValueError):
            comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets)
```

**Replace lines 122-128** (`test_kernel_contract_violation_on_shape_mismatch`):

```python
    def test_kernel_contract_violation_on_shape_mismatch(self):
        diags = jnp.array([[2.0, 3.0]])
        solves = jnp.array([[1.0, 1.0, 1.0]])  # mismatched second dim
        lowers = jnp.array([0.0, 0.0])
        backsub_indices = jnp.zeros((1, 2), dtype=jnp.int32)
        with self.assertRaises(ValueError):
            comp_backsub_raw(diags, solves, lowers, backsub_indices)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest braincell/quad/_staggered_test.py::CompTriangRawTest::test_kernel_contract_violation_on_wrong_rank braincell/quad/_staggered_test.py::CompTriangRawTest::test_kernel_contract_violation_on_dimful_diags braincell/quad/_staggered_test.py::CompBacksubRawTest::test_kernel_contract_violation_on_shape_mismatch -v`
Expected: `FAIL` — current code raises `AssertionError`, tests expect
`ValueError`.

- [ ] **Step 3: Fix `_check_comp_triang`**

Replace `braincell/quad/_staggered.py:510-525` with:

```python
def _check_comp_triang(diags, solves, lowers, uppers, edges):
    """Kernel contract check for the quantity-aware DHS forward pass."""
    if isinstance(edges, u.Quantity):
        raise ValueError("edges must be a plain array, not a Quantity")
    if diags.ndim != 2:
        raise ValueError(f"diags must be 2D, got ndim={diags.ndim}")
    if solves.ndim != 2:
        raise ValueError(f"solves must be 2D, got ndim={solves.ndim}")
    if lowers.ndim != 1:
        raise ValueError(f"lowers must be 1D, got ndim={lowers.ndim}")
    if uppers.ndim != 1:
        raise ValueError(f"uppers must be 1D, got ndim={uppers.ndim}")
    if isinstance(diags, u.Quantity) and not u.get_unit(diags).is_unitless:
        raise ValueError(f"diags must be unitless, got unit={u.get_unit(diags)}")
    if isinstance(lowers, u.Quantity) and not u.get_unit(lowers).is_unitless:
        raise ValueError(f"lowers must be unitless, got unit={u.get_unit(lowers)}")
    if isinstance(uppers, u.Quantity) and not u.get_unit(uppers).is_unitless:
        raise ValueError(f"uppers must be unitless, got unit={u.get_unit(uppers)}")
    if lowers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"lowers.shape[0]={lowers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if uppers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"uppers.shape[0]={uppers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (_, 2), got {edges.shape}")
```

- [ ] **Step 4: Fix `_check_comp_backsub`**

Replace `braincell/quad/_staggered.py:545-558` with:

```python
def _check_comp_backsub(diags, solves, lowers, backsub_indices):
    """Kernel contract check for quantity-aware recursive doubling."""
    if isinstance(backsub_indices, u.Quantity):
        raise ValueError("backsub_indices must be a plain array, not a Quantity")
    if diags.ndim != 2:
        raise ValueError(f"diags must be 2D, got ndim={diags.ndim}")
    if solves.ndim != 2:
        raise ValueError(f"solves must be 2D, got ndim={solves.ndim}")
    if lowers.ndim != 1:
        raise ValueError(f"lowers must be 1D, got ndim={lowers.ndim}")
    if isinstance(diags, u.Quantity) and not u.get_unit(diags).is_unitless:
        raise ValueError(f"diags must be unitless, got unit={u.get_unit(diags)}")
    if isinstance(lowers, u.Quantity) and not u.get_unit(lowers).is_unitless:
        raise ValueError(f"lowers must be unitless, got unit={u.get_unit(lowers)}")
    if diags.shape != solves.shape:
        raise ValueError(
            f"diags.shape={diags.shape} must equal solves.shape={solves.shape}"
        )
    if lowers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"lowers.shape[0]={lowers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if backsub_indices.ndim != 2:
        raise ValueError(f"backsub_indices must be 2D, got ndim={backsub_indices.ndim}")
    if backsub_indices.shape[1] != diags.shape[1]:
        raise ValueError(
            f"backsub_indices.shape[1]={backsub_indices.shape[1]} "
            f"must equal diags.shape[1]={diags.shape[1]}"
        )
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest braincell/quad/_staggered_test.py -v`
Expected: `PASS` — all tests in the file, including the untouched
unit-preservation tests.

- [ ] **Step 6: Commit**

```bash
git add braincell/quad/_staggered.py braincell/quad/_staggered_test.py
git commit -m "fix(quad): DHS kernel contract checks raise ValueError not AssertionError (HIGH-04)"
```

---

## Final sanity pass

- [ ] **Step 1: Run the whole quad + base + single-compartment suite**

Run: `pytest braincell/_base_test.py braincell/_single_compartment/ braincell/quad/ braincell/_multi_compartment/ -v`
Expected: `PASS`. If anything downstream of `_base.py` or `staggered.py`
breaks, investigate before moving on — the runtime integrates these paths
end-to-end.

- [ ] **Step 2: Grep for any remaining `assert isinstance(target,` in the
      quad package**

Run: `grep -n "assert isinstance(target," braincell/quad/*.py`
Expected: no matches. If matches appear, either they are outside this
plan's scope (document and leave) or a sibling integrator needs the same
treatment — in which case raise with the user before expanding scope.

- [ ] **Step 3: Confirm fixes against the review report**

Open `docs/superpowers/specs/2026-04-22-braincell-triage-review-report.md`,
tick off CRIT-01, CRIT-02, HIGH-01, HIGH-02, HIGH-03, HIGH-04 mentally.
No report edit is required — the report is a historical artifact.

---

## Self-review checklist (author-run)

**Spec coverage:** Six findings requested. Six tasks above. ✓

**Type consistency:** All test classes live in the existing test files.
`_RecordingKCaChannel` / `_FakeChannelLike` naming is consistent inside
`_base_test.py`. Exception-type contract for `staggered_step` / `exp_euler_step`
/ `_check_comp_triang` / `_check_comp_backsub` is consistently `TypeError` for
argument mis-type and `ValueError` for kernel input shape/unit violations.

**No placeholders:** Every task shows the full replacement code block, exact
`git commit` message, and exact `pytest` invocation. No "TODO" / "similar to".

**Behavior preserved?** Per finding:

| Task | Behavior preserved? | Justification |
|------|---------------------|----------------|
| CRIT-01 | no | Fix changes broken code path; silent-wrong → correct. |
| CRIT-02 | no | Fix replaces a `TypeError` crash with a correct sum. |
| HIGH-01 | no | `__cause__` is now set; otherwise identical surface. |
| HIGH-02 | no | Tighter validation at entry; objects that were previously silently accepted now raise. |
| HIGH-03 | no | `AssertionError` → `TypeError`. Callers `except AssertionError`-guarding must update — call out in commit message. |
| HIGH-04 | no | `AssertionError` → `ValueError`. Internal-only callers; low blast radius. |

All `no` entries are intentional — each finding is itself "fix behavior."
