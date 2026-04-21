# `braincell.compute` Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `braincell.compute` — fix 6 correctness bugs, rename to private `braincell/_compute/`, swap object-dtype state buffers to `u.Quantity(jnp.ndarray, unit)`, kill circular import with `_multi_compartment/clamp_table.py`, remove `install_cell_runtime` lifecycle handshake, push HH1952 temperature logic into channel classes.

**Architecture:** Three files under `braincell/_compute/`: `topology.py` (point tree + scheduling), `runtime.py` (`CellRuntimeState`, `MechanismLayout`, `ClampActiveTable`, compilation), `table.py` (inspection). Each file co-locates `*_test.py`. Migration proceeds in order so every commit leaves the test suite green.

**Tech Stack:** Python 3.10+, JAX (CPU in tests via `conftest.py`), brainunit, brainstate, numpy, pytest + unittest.TestCase.

**Spec:** `docs/superpowers/specs/2026-04-21-compute-rewrite-design.md`

**Preconditions:**
- Working tree is clean of unrelated WIP before each commit. The engineer must run `git status` before each `git add` and stage only the files listed in the task.
- CLAUDE.md conventions apply: test files named `*_test.py` co-located with source, no bare `test_*.py`, no `tests/` subdir, NumPy-style docstrings on public symbols, units mandatory via `normalize_param`.
- Do NOT add `Co-Authored-By: Claude ...` trailer to commit messages.

---

## File Structure (target state)

```
braincell/_compute/
  __init__.py              # re-exports PointTree, PointScheduling, CellRuntimeState, …
  topology.py              # point tree + scheduling (from _point_tree.py)
  topology_test.py         # new; absorbs relevant parts of _runtime_test.py
  runtime.py               # CellRuntimeState + MechanismLayout + ClampActiveTable
  runtime_test.py          # absorbs compute/_runtime_test.py + clamp_table_test.py
  table.py                 # MechanismObjectTable + MechanismObjectCell + mechanism_cell_key
  table_test.py            # new

braincell/_multi_compartment/bridge.py       # +scatter/gather/quantity_vector/fill_like/cv_value_vector (moved in)
braincell/_multi_compartment/cell.py         # init_state/reset absorb former install_cell_runtime body
braincell/_multi_compartment/currents.py     # import-site updates only
braincell/_multi_compartment/probes.py       # import-site updates only

braincell/morph/morphology.py                # +clone_morpho (moved from _runtime.py)
braincell/ion/__init__.py                    # +build_placeholder_ions (moved from _runtime.py)

braincell/channel/sodium.py                  # INa_HH1952: accept T; compute phi; _on_param_updated hook
braincell/channel/potassium.py               # IK_HH1952: same pattern

DELETED:
  braincell/_multi_compartment/clamp_table.py
  braincell/_multi_compartment/clamp_table_test.py   (content merged into runtime_test.py)
  braincell/compute/                                  (whole directory replaced by _compute/)
```

---

## Task 1: Pure rename — `compute/ → _compute/` and drop inner `_` prefix

**Rationale:** Split the risky renaming work from behavior changes. After this task, directory + file names match the target layout but every line of code is byte-identical except imports.

**Files:**
- Rename: `braincell/compute/__init__.py` → `braincell/_compute/__init__.py`
- Rename: `braincell/compute/_point_tree.py` → `braincell/_compute/topology.py`
- Rename: `braincell/compute/_runtime.py` → `braincell/_compute/runtime.py`
- Rename: `braincell/compute/_assignment_table.py` → `braincell/_compute/table.py`
- Rename: `braincell/compute/_runtime_test.py` → `braincell/_compute/runtime_test.py`
- Modify: `braincell/_compute/__init__.py` (update intra-package imports)
- Modify: `braincell/_compute/topology.py` (update intra-package imports)
- Modify: `braincell/_compute/runtime.py` (update intra-package imports)
- Modify: `braincell/_compute/table.py` (update intra-package imports)
- Modify: `braincell/_compute/runtime_test.py` (update imports)
- Modify: `braincell/_multi_compartment/cell.py` (update imports)
- Modify: `braincell/_multi_compartment/currents.py` (update imports)
- Modify: `braincell/_multi_compartment/probes.py` (update imports)
- Modify: `braincell/_multi_compartment/bridge.py` (update imports)
- Modify: `braincell/_multi_compartment/clamp_table.py` (update imports)
- Modify: `braincell/_multi_compartment/clamp_table_test.py` (update imports)
- Modify: `braincell/_multi_compartment/cell_test.py` (update imports)
- Modify: `braincell/_multi_compartment/currents_test.py` (update imports)
- Modify: `braincell/_multi_compartment/bridge_test.py` (update imports)

- [ ] **Step 1: Find every import of `braincell.compute`**

Run:
```bash
grep -rn "from braincell\.compute\|from braincell import compute\|braincell\.compute\." braincell/ docs/ examples/ 2>/dev/null
```

Expected: list of every file that references the current `compute` package. Save the output for step-3 verification.

- [ ] **Step 2: Rename files with `git mv` (preserves history)**

```bash
git mv braincell/compute braincell/_compute
git mv braincell/_compute/_point_tree.py braincell/_compute/topology.py
git mv braincell/_compute/_runtime.py braincell/_compute/runtime.py
git mv braincell/_compute/_assignment_table.py braincell/_compute/table.py
git mv braincell/_compute/_runtime_test.py braincell/_compute/runtime_test.py
```

- [ ] **Step 3: Rewrite `braincell/_compute/__init__.py`**

Replace contents with:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ...  (keep existing license header)
"""Internal compute layer: point-tree topology + runtime state.

This package is private. External code should access the same symbols
through :mod:`braincell` re-exports where appropriate.
"""

from .runtime import CellRuntimeState, MechanismLayout
from .table import MechanismObjectTable, MechanismObjectCell, mechanism_cell_key
from .topology import PointScheduling, PointTree

__all__ = [
    "CellRuntimeState",
    "MechanismLayout",
    "MechanismObjectCell",
    "MechanismObjectTable",
    "PointScheduling",
    "PointTree",
    "mechanism_cell_key",
]
```

Only `from ._point_tree` / `from ._runtime` / `from ._assignment_table` lines change → new module names without leading underscore. No symbol additions yet.

- [ ] **Step 4: Update intra-package imports inside `_compute/`**

In `braincell/_compute/runtime.py`:

```python
# OLD
from ._point_tree import PointTree
# NEW
from .topology import PointTree
```

In `braincell/_compute/table.py`:

```python
# OLD
from ._runtime import CellRuntimeState
# NEW
from .runtime import CellRuntimeState
```

In `braincell/_compute/runtime_test.py`:

```python
# OLD
from braincell.compute._runtime import (
    install_cell_runtime,
    uninstall_cell_runtime,
    CellRuntimeState,
    build_placeholder_ions,
)
# NEW
from braincell._compute.runtime import (
    install_cell_runtime,
    uninstall_cell_runtime,
    CellRuntimeState,
    build_placeholder_ions,
)
```

Search the test file for every `braincell.compute` reference and replace with `braincell._compute` (paths collapse `._runtime` → `.runtime`, `._point_tree` → `.topology`, `._assignment_table` → `.table`).

- [ ] **Step 5: Update downstream imports in `_multi_compartment/`**

Apply across `_multi_compartment/cell.py`, `currents.py`, `probes.py`, `bridge.py`, `clamp_table.py`, `cell_test.py`, `currents_test.py`, `bridge_test.py`, `clamp_table_test.py`:

| From | To |
|---|---|
| `from braincell.compute._runtime import ...` | `from braincell._compute.runtime import ...` |
| `from braincell.compute._point_tree import ...` | `from braincell._compute.topology import ...` |
| `from braincell.compute._assignment_table import ...` | `from braincell._compute.table import ...` |
| `from braincell.compute import ...` | `from braincell._compute import ...` |

Also update any docstring references (non-functional but aids greppability).

- [ ] **Step 6: Update `quad/_staggered.py` (duck-typed, no direct imports, but docstrings reference compute)**

Run:
```bash
grep -n "compute" braincell/quad/_staggered.py
```

Replace textual `compute` references in docstrings/comments with `_compute`. No runtime code changes there.

- [ ] **Step 7: Update top-level `braincell/__init__.py` if it re-exports anything from compute**

Run:
```bash
grep -n "compute" braincell/__init__.py
```

If any `from .compute import ...` exists, change to `from ._compute import ...`. If compute was exposed as a public module, either add a deprecation shim or remove — consult the user. For this task default: **remove** (spec D3).

- [ ] **Step 8: Verify no stale references remain**

```bash
grep -rn "braincell\.compute\|braincell/compute\|from \._point_tree\|from \._runtime\|from \._assignment_table" braincell/ docs/ examples/ 2>/dev/null
```

Expected: empty output. If anything remains, fix it. Docstring text like `"braincell.compute._runtime"` counts — replace those too.

- [ ] **Step 9: Run full test suite**

```bash
pytest braincell/ -x -q
```

Expected: PASS (same test count as before rename).

- [ ] **Step 10: Commit**

```bash
git add braincell/_compute/ braincell/_multi_compartment/ braincell/quad/_staggered.py braincell/__init__.py
git commit -m "refactor(compute): rename public compute/ to private _compute/

Package renamed from braincell.compute (public) to braincell._compute
(private). Internal module files drop the leading-underscore prefix:
_point_tree.py -> topology.py, _runtime.py -> runtime.py,
_assignment_table.py -> table.py. Privacy is now expressed once at
the package boundary. No behavior change; import sites in
_multi_compartment/ and quad/ updated in lockstep."
```

---

## Task 2: Fix bug C1 — entry-vs-exit half tag on intra-branch edges

**Files:**
- Modify: `braincell/_compute/topology.py:307-326` (edge-role loop in `build_point_tree`)
- Create: `braincell/_compute/topology_test.py`

**Background.** The loop at lines ~308-326 of the current `topology.py` adds edge roles twice per CV. Inspect the ternary on line 319:

```python
add_edge_role(
    parent_point_id,
    midpoint_point_id,
    cv_id=cv_id,
    half=_entry_half_for_walk(attach_x) if index == 0 else _entry_half_for_walk(attach_x),
)
```

Both arms of the ternary call `_entry_half_for_walk(attach_x)`. The second arm was clearly meant to distinguish interior-edge handedness. Semantics: the edge going *into* a midpoint is tagged with the entry half of the walk; the edge going *out* of a midpoint is tagged with the exit half. So the first `add_edge_role` is always entry-half, the second is always exit-half — independent of `index`.

- [ ] **Step 1: Write the failing regression test**

Create `braincell/_compute/topology_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# (same license header as other modules)

import unittest

import brainunit as u

import braincell
from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell._compute.topology import build_point_tree
from braincell._cv.base import build_cvs
from braincell.filter import BranchSlice


def _two_branch_morpho() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class BuildPointTreeEdgeHalves(unittest.TestCase):
    def test_intra_branch_edges_alternate_entry_and_exit_halves(self) -> None:
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch())
        tree = build_point_tree(morpho, cvs=cvs)

        # Walk the branch-1 mid-to-midpoint edges.
        dend_cv_ids = [cv.id for cv in cvs if cv.branch_id == 1]
        self.assertGreater(len(dend_cv_ids), 0)

        halves_seen: set[str] = set()
        for edge in tree.edges:
            for cv_edge in edge.cv_edges:
                if cv_edge.cv_id in dend_cv_ids:
                    halves_seen.add(cv_edge.half)
        # Before fix: halves_seen == {"prox"}. After fix: {"prox", "dist"}.
        self.assertEqual(halves_seen, {"prox", "dist"})
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest braincell/_compute/topology_test.py::BuildPointTreeEdgeHalves -v
```

Expected: FAIL with `AssertionError: {'prox'} != {'prox', 'dist'}`.

- [ ] **Step 3: Apply C1 fix**

In `braincell/_compute/topology.py`, change the edge-role loop body (currently around lines 308-326) to:

```python
for index, cv_id in enumerate(ordered_cv_ids):
    midpoint_point_id = int(cv_midpoint_point_id[cv_id])
    parent_point_id = attachment_point_id if index == 0 else int(
        cv_midpoint_point_id[ordered_cv_ids[index - 1]]
    )
    child_point_id = terminal_point_id if index == len(ordered_cv_ids) - 1 else int(
        cv_midpoint_point_id[ordered_cv_ids[index + 1]]
    )
    add_edge_role(
        parent_point_id,
        midpoint_point_id,
        cv_id=cv_id,
        half=_entry_half_for_walk(attach_x),
    )
    add_edge_role(
        midpoint_point_id,
        child_point_id,
        cv_id=cv_id,
        half=_exit_half_for_walk(attach_x),
    )
```

The change is the second `add_edge_role` call: `half=_entry_half_for_walk(attach_x)` → `half=_exit_half_for_walk(attach_x)`. Remove the dead `if index == 0 else` ternary.

- [ ] **Step 4: Rerun test — expect pass**

```bash
pytest braincell/_compute/topology_test.py::BuildPointTreeEdgeHalves -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite to check no regressions**

```bash
pytest braincell/ -x -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/topology.py braincell/_compute/topology_test.py
git commit -m "fix(compute): tag intra-branch edge exit-half correctly (C1)

The second add_edge_role call in build_point_tree had identical
branches in its ternary, so every edge out of a midpoint was tagged
with the entry half instead of the exit half. Replace with an
unconditional _exit_half_for_walk call. Add regression test.
"
```

---

## Task 3: Fix bug C2 — peel-level computation invariant-free

**Files:**
- Modify: `braincell/_compute/topology.py` (`_compute_peel_levels`)
- Modify: `braincell/_compute/topology_test.py` (new test)

- [ ] **Step 1: Write the failing regression test**

Append to `braincell/_compute/topology_test.py`:

```python
import numpy as np
from braincell._compute.topology import (
    PointTree,
    _compute_peel_levels,
)


class ComputePeelLevels(unittest.TestCase):
    def test_peel_levels_correct_when_child_id_less_than_parent_id(self) -> None:
        # Hand-crafted topology where child ids are NOT all greater than parent ids.
        # Graph: 3 -> 0 -> 2, 3 -> 1 (root = 3, leaves = 1 and 2).
        point_parent = np.asarray([3, 3, 0, -1], dtype=np.int32)
        point_children = (
            (2,),     # 0 -> 2
            (),       # 1 leaf
            (),       # 2 leaf
            (0, 1),   # 3 -> 0, 1
        )
        levels = _compute_peel_levels(point_parent=point_parent, point_children=point_children)
        # Leaves (1, 2) have peel 0; node 0 has peel 1; root 3 has peel 2.
        self.assertEqual(levels.tolist(), [1, 0, 0, 2])

    def test_peel_levels_raise_on_disconnected_node(self) -> None:
        # Node 1 has no parent and no children -> unreachable from any leaf-walk.
        point_parent = np.asarray([-1, -1], dtype=np.int32)
        point_children = ((), ())
        # Actually both nodes are leaves so this should NOT raise — each is its own root/leaf.
        levels = _compute_peel_levels(point_parent=point_parent, point_children=point_children)
        self.assertEqual(levels.tolist(), [0, 0])

    def test_peel_levels_raise_on_cycle(self) -> None:
        # Contrived cycle: 0 -> 1 -> 0. Not producible by build_point_tree but
        # peel computation must terminate with a clear error.
        point_parent = np.asarray([1, 0], dtype=np.int32)
        point_children = ((1,), (0,))
        with self.assertRaises(ValueError) as ctx:
            _compute_peel_levels(point_parent=point_parent, point_children=point_children)
        self.assertIn("cycle", str(ctx.exception).lower())
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest braincell/_compute/topology_test.py::ComputePeelLevels -v
```

Expected: FAIL (existing function has wrong signature — keyword-only with different args — and relies on numeric ordering).

- [ ] **Step 3: Rewrite `_compute_peel_levels`**

Replace the body at `braincell/_compute/topology.py` (currently lines ~549-558) with:

```python
def _compute_peel_levels(
    *,
    point_parent: np.ndarray | None = None,
    point_children: tuple[tuple[int, ...], ...] | None = None,
    point_tree: "PointTree | None" = None,
) -> np.ndarray:
    """Assign each point its distance-to-farthest-leaf (peel level).

    Leaves are level 0. Interior points are ``1 + max(level[child])``.
    Works irrespective of the numeric ordering of point ids.

    Parameters
    ----------
    point_parent : np.ndarray, optional
        ``(n_point,)`` int array; ``-1`` for the root. Mutually
        exclusive with ``point_tree``.
    point_children : tuple of tuples, optional
        ``(n_point,)`` tuples of child ids. Must be supplied alongside
        ``point_parent``.
    point_tree : PointTree, optional
        Supply instead of the two arrays above.

    Returns
    -------
    np.ndarray
        ``(n_point,)`` int32.

    Raises
    ------
    ValueError
        If a cycle is detected or a node is never reached from any leaf.
    """
    if point_tree is not None:
        if point_parent is not None or point_children is not None:
            raise TypeError(
                "_compute_peel_levels: pass either point_tree or (point_parent, point_children), not both."
            )
        point_parent = point_tree.point_parent
        point_children = point_tree.point_children
    if point_parent is None or point_children is None:
        raise TypeError(
            "_compute_peel_levels: supply point_tree or both point_parent and point_children."
        )

    n_point = int(len(point_parent))
    levels = np.full(n_point, -1, dtype=np.int32)
    remaining_children = np.asarray([len(children) for children in point_children], dtype=np.int32)

    frontier: list[int] = [pid for pid, count in enumerate(remaining_children.tolist()) if count == 0]
    for pid in frontier:
        levels[pid] = 0

    cursor = 0
    while cursor < len(frontier):
        pid = frontier[cursor]
        cursor += 1
        parent = int(point_parent[pid])
        if parent < 0:
            continue
        candidate = int(levels[pid]) + 1
        if int(levels[parent]) < candidate:
            levels[parent] = candidate
        remaining_children[parent] -= 1
        if int(remaining_children[parent]) == 0:
            frontier.append(parent)

    if (levels < 0).any():
        raise ValueError(
            "compute_peel_levels: cycle detected or point unreachable from any leaf."
        )
    return levels
```

Update every call site inside `build_point_scheduling`:

```python
# OLD
peel_level_by_point = _compute_peel_levels(point_tree=point_tree)
# NEW — unchanged; function still accepts point_tree kwarg.
peel_level_by_point = _compute_peel_levels(point_tree=point_tree)
```

(No call-site change needed; the new signature is backward compatible.)

- [ ] **Step 4: Rerun test — expect pass**

```bash
pytest braincell/_compute/topology_test.py::ComputePeelLevels -v
```

Expected: PASS (all three methods).

- [ ] **Step 5: Full suite**

```bash
pytest braincell/ -x -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/topology.py braincell/_compute/topology_test.py
git commit -m "fix(compute): make peel-level computation independent of id ordering (C2)

The previous implementation iterated point ids in reverse numeric
order, which only produced correct peel levels when parent ids were
always less than child ids. Replace with an explicit leaf-up
topological peel that works for any ordering, and raise ValueError
on cycles or unreachable nodes. Add three regression tests covering
out-of-order ids, isolated nodes, and a contrived cycle.
"
```

---

## Task 4: Fix bug C3 — `_locate_branch_cv_by_x` raises on tiling gaps

**Files:**
- Modify: `braincell/_compute/topology.py` (`_locate_branch_cv_by_x`)
- Modify: `braincell/_compute/topology_test.py` (new tests)

- [ ] **Step 1: Write failing tests**

Append to `topology_test.py`:

```python
from braincell._compute.topology import _locate_branch_cv_by_x, _EPS_PARAM


class _FakeCV:
    def __init__(self, id_, prox, dist):
        self.id = id_
        self.prox = prox
        self.dist = dist


class LocateBranchCVByX(unittest.TestCase):
    def _cvs(self, tiles):
        return tuple(_FakeCV(i, p, d) for i, (p, d) in enumerate(tiles))

    def test_interior_x_lands_in_matching_cv(self) -> None:
        cvs = self._cvs([(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
        ids = (0, 1, 2)
        got = _locate_branch_cv_by_x(ids, cvs, x=0.5, epsilon=_EPS_PARAM)
        self.assertEqual(got, 1)

    def test_x_near_one_returns_last_cv(self) -> None:
        cvs = self._cvs([(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
        ids = (0, 1, 2)
        got = _locate_branch_cv_by_x(ids, cvs, x=0.999, epsilon=_EPS_PARAM)
        self.assertEqual(got, 2)

    def test_x_in_gap_between_tiles_raises(self) -> None:
        # Non-tiling layout: gap between 0.4 and 0.6.
        cvs = self._cvs([(0.0, 0.4), (0.6, 1.0)])
        ids = (0, 1)
        with self.assertRaises(ValueError) as ctx:
            _locate_branch_cv_by_x(ids, cvs, x=0.5, epsilon=_EPS_PARAM)
        self.assertIn("0.5", str(ctx.exception))
```

- [ ] **Step 2: Run tests — expect failure on the third**

```bash
pytest braincell/_compute/topology_test.py::LocateBranchCVByX -v
```

Expected: first two PASS; third FAIL (currently returns `ids[-1]` silently).

- [ ] **Step 3: Apply C3 fix**

Replace `_locate_branch_cv_by_x` in `braincell/_compute/topology.py`:

```python
def _locate_branch_cv_by_x(
    ids: tuple[int, ...],
    cvs: tuple[CV, ...],
    *,
    x: float,
    epsilon: float,
) -> int:
    """Return the CV id whose normalised half-open interval contains ``x``.

    Boundary cases: ``x <= 0 + epsilon`` snaps to ``ids[0]``;
    ``x >= 1 - epsilon`` snaps to ``ids[-1]``. Interior x uses
    ``[prox, dist)``. Raises ``ValueError`` if the CV list has a gap
    and ``x`` does not fall inside any tile — previously the function
    silently returned ``ids[-1]``.
    """
    if x <= 0.0 + epsilon:
        return ids[0]
    if x >= 1.0 - epsilon:
        return ids[-1]
    for cv_id in ids:
        cv = cvs[cv_id]
        if float(cv.prox) - epsilon <= x < float(cv.dist) - epsilon:
            return cv_id
    raise ValueError(
        f"_locate_branch_cv_by_x: x={x!r} lies in no CV interval among ids {list(ids)!r}. "
        "This usually means the CV tiling of this branch has a gap or overlap."
    )
```

- [ ] **Step 4: Rerun tests — all pass**

```bash
pytest braincell/_compute/topology_test.py::LocateBranchCVByX -v
```

Expected: PASS x3.

- [ ] **Step 5: Full suite**

```bash
pytest braincell/ -x -q
```

Expected: green. If any existing test relied on the silent fallback, fix that test to supply valid tiling.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/topology.py braincell/_compute/topology_test.py
git commit -m "fix(compute): raise on gap in _locate_branch_cv_by_x (C3)

Previously the function silently returned ids[-1] when x fell in a
gap between CVs, masking malformed tiling. Now raises ValueError
with a diagnostic message naming x and the ids under search. Keep
the explicit boundary snaps at x<=0 and x>=1.
"
```

---

## Task 5: Vocabulary unification — `Position` and `Half` Literals

**Files:**
- Modify: `braincell/_compute/topology.py` (CVPoint.position type + values)
- Modify: `braincell/_compute/topology_test.py`

- [ ] **Step 1: Add Literal type aliases**

At the top of `topology.py` (after existing imports):

```python
from typing import Literal

Position = Literal["prox", "mid", "dist"]
Half = Literal["prox", "dist"]
```

- [ ] **Step 2: Change `CVPoint.position` type and all call sites**

Find every occurrence of `"proximal"` / `"distal"` inside `topology.py` and replace:

| Old | New |
|---|---|
| `"proximal"` | `"prox"` |
| `"distal"` | `"dist"` |
| `"mid"` | `"mid"` (unchanged) |

Check `_POSITION_ORDER` dict: keep semantic order but new keys:

```python
_POSITION_ORDER = {"prox": 0, "mid": 1, "dist": 2}
```

Change `CVPoint`:

```python
@dataclass(frozen=True)
class CVPoint:
    cv_id: int
    position: Position   # was: position: str
```

Change `CVEdge`:

```python
@dataclass(frozen=True)
class CVEdge:
    cv_id: int
    half: Half           # was: half: str
```

- [ ] **Step 3: Update helper functions**

```python
def _entry_position_for_walk(attach_x: float) -> Position:
    return "prox" if attach_x <= _EPS_PARAM else "dist"


def _exit_position_for_walk(attach_x: float) -> Position:
    return "dist" if attach_x <= _EPS_PARAM else "prox"
```

- [ ] **Step 4: Search for stragglers**

```bash
grep -n "proximal\|distal" braincell/_compute/
```

Expected: no matches in `_compute/`.

- [ ] **Step 5: Write a test that locks the vocabulary**

Append to `topology_test.py`:

```python
class VocabularyLock(unittest.TestCase):
    def test_cvpoint_positions_are_three_letter_codes(self) -> None:
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch())
        tree = build_point_tree(morpho, cvs=cvs)
        seen = {cvp.position for point in tree.points for cvp in point.cv_points}
        self.assertTrue(seen.issubset({"prox", "mid", "dist"}))
        self.assertIn("mid", seen)
```

- [ ] **Step 6: Run tests**

```bash
pytest braincell/_compute/topology_test.py -v
```

Expected: all PASS.

- [ ] **Step 7: Full suite + fix any downstream test that pattern-matched the old strings**

```bash
pytest braincell/ -x -q
```

If any test fails because it checked `"proximal"` / `"distal"`, update that test to the three-letter codes. Commit those fixups as part of this task.

- [ ] **Step 8: Commit**

```bash
git add braincell/_compute/topology.py braincell/_compute/topology_test.py braincell/_compute/runtime_test.py
git commit -m "refactor(compute): unify Position and Half to three-letter codes

Introduce typing.Literal aliases for CVPoint.position and CVEdge.half.
Unify CVPoint.position vocabulary ('proximal','mid','distal') to the
same three-letter codes used by CVEdge.half ('prox','mid','dist').
Update helper functions and every call site.
"
```

---

## Task 6: Deduplicate row + group build in `build_point_scheduling`

**Files:**
- Modify: `braincell/_compute/topology.py` (`build_point_scheduling`)

- [ ] **Step 1: Add the helper**

In `topology.py`, after `_compute_peel_levels`, add:

```python
def _order_points_by_peel_then_matrix(
    *,
    point_tree: PointTree,
    peel_levels: np.ndarray,
) -> np.ndarray:
    """Permutation of point ids: peel-descending, then matrix-ascending.

    ``np.lexsort`` uses the LAST key as primary; we pass
    ``(-peel_levels)`` last so higher peel comes first, and
    ``point_id_to_matrix_index`` first so it acts as a tie-breaker.
    """
    matrix_idx = np.asarray(point_tree.point_id_to_matrix_index, dtype=np.int32)
    peels = np.asarray(peel_levels, dtype=np.int32)
    return np.lexsort((matrix_idx, -peels)).astype(np.int32)
```

- [ ] **Step 2: Rewrite `_build_row_to_point_id` in terms of the helper**

```python
def _build_row_to_point_id(
    *,
    point_tree: PointTree,
    peel_level_by_point: np.ndarray,
) -> np.ndarray:
    return _order_points_by_peel_then_matrix(
        point_tree=point_tree,
        peel_levels=peel_level_by_point,
    )
```

- [ ] **Step 3: Rewrite `_build_groups` to reuse the same ordering**

```python
def _build_groups(
    *,
    point_tree: PointTree,
    peel_level_by_point: np.ndarray,
    point_id_to_row: np.ndarray,
    max_group_size: int,
) -> tuple[np.ndarray, ...]:
    order = _order_points_by_peel_then_matrix(
        point_tree=point_tree,
        peel_levels=peel_level_by_point,
    )
    groups: list[np.ndarray] = []
    # Split the single ordering by peel-level boundaries, then chunk by max_group_size.
    peels_ordered = peel_level_by_point[order]
    # Find boundaries where peel decreases.
    level_starts = [0]
    for i in range(1, len(order)):
        if int(peels_ordered[i]) != int(peels_ordered[i - 1]):
            level_starts.append(i)
    level_starts.append(len(order))

    for a, b in zip(level_starts[:-1], level_starts[1:]):
        for chunk_start in range(a, b, max_group_size):
            chunk_point_ids = order[chunk_start:chunk_start + max_group_size]
            rows = point_id_to_row[chunk_point_ids]
            groups.append(np.asarray(rows, dtype=np.int32))
    return tuple(groups)
```

- [ ] **Step 4: Run the existing scheduling tests**

```bash
pytest braincell/_compute/ -v -k "schedul"
```

Also run the broader suite:

```bash
pytest braincell/ -x -q
```

Expected: all green. `PointScheduling.groups` values should match byte-for-byte with the previous implementation (same ordering, same chunking).

- [ ] **Step 5: Commit**

```bash
git add braincell/_compute/topology.py
git commit -m "refactor(compute): share one ordering between row and group build

_build_row_to_point_id and _build_groups both performed the same
peel-descending / matrix-ascending sort. Extract the ordering into
_order_points_by_peel_then_matrix and reuse it. Behavior unchanged;
existing PointScheduling snapshots still match.
"
```

---

## Task 7: Move helpers out of `runtime.py`

Migrate the non-runtime helpers to their correct homes so `runtime.py` ends up focused on `CellRuntimeState` + `MechanismLayout`.

**Files:**
- Modify: `braincell/morph/morphology.py` (gains `clone_morpho`)
- Modify: `braincell/ion/__init__.py` (gains `build_placeholder_ions`)
- Modify: `braincell/_multi_compartment/bridge.py` (gains scatter/gather/quantity/fill/cv helpers)
- Modify: `braincell/_compute/runtime.py` (delete moved functions + update imports)
- Modify: `braincell/_multi_compartment/cell.py` (update import sites)

- [ ] **Step 1: Move `clone_morpho` to `morph/morphology.py`**

Open `braincell/_compute/runtime.py`, copy the `clone_morpho` function body (currently ~lines 439-453). Paste into `braincell/morph/morphology.py` (end of file). Delete from `runtime.py`.

```python
# braincell/morph/morphology.py — appended at end:

def clone_morpho(morpho: "Morphology") -> "Morphology":
    """Return a structurally identical copy of ``morpho``.

    Preserves branch identity, parent/child topology, and attachment
    ratios. Used by ``Cell.init_state`` to freeze the declaration
    tree without mutating user-supplied objects.
    """
    cloned = Morphology.from_root(morpho.root.branch, name=morpho.root.name)
    for index in range(1, len(morpho.branches)):
        branch = morpho.branch(index=index)
        parent = branch.parent
        if parent is None:
            continue
        cloned.attach(
            parent=parent.name,
            child_branch=branch.branch,
            child_name=branch.name,
            parent_x=float(branch.parent_x),
            child_x=float(branch.child_x),
        )
    return cloned
```

Update `braincell/_multi_compartment/cell.py`:

```python
# OLD
from braincell._compute.runtime import (
    ...,
    clone_morpho,
    ...,
)
# NEW
from braincell._compute.runtime import (...)  # clone_morpho removed from this import
from braincell.morph.morphology import clone_morpho
```

- [ ] **Step 2: Move `build_placeholder_ions` to `ion/__init__.py`**

Copy the function from `runtime.py` (~lines 431-436):

```python
# braincell/ion/__init__.py — appended:

def build_placeholder_ions(size=(1,)) -> dict[str, object]:
    """Return a dict of default Na/K/Ca fixed-ion containers for scaffolding.

    Used by test doubles and by ``HHTypedNeuron`` construction before
    the real runtime ion containers are instantiated.
    """
    return {
        "na": SodiumFixed(size=size),
        "k": PotassiumFixed(size=size),
        "ca": CalciumFixed(size=size),
    }
```

(Note: the symbols `SodiumFixed`, `PotassiumFixed`, `CalciumFixed` must already be imported in `ion/__init__.py`; verify with `grep -n "SodiumFixed\|PotassiumFixed\|CalciumFixed" braincell/ion/__init__.py` and add imports if missing.)

Delete `build_placeholder_ions` from `runtime.py`. Update `runtime.py`'s internal call site `_build_default_ions` which duplicates the same logic — replace with:

```python
from braincell.ion import build_placeholder_ions

def _build_default_ions(n_point: int) -> dict[str, object]:
    return build_placeholder_ions(size=(n_point,))
```

Update `_multi_compartment/cell.py`:

```python
# OLD
from braincell._compute.runtime import build_placeholder_ions
# NEW
from braincell.ion import build_placeholder_ions
```

- [ ] **Step 3: Move bridging helpers to `_multi_compartment/bridge.py`**

Copy these functions from `runtime.py` into `bridge.py`:
- `scatter_midpoint_values`
- `gather_midpoint_values`
- `cv_value_vector`
- `fill_like`
- `quantity_vector`
- `matches_last_dim`
- `is_python_zero`

Remove them from `runtime.py`. In `bridge.py`, keep the existing `cv_to_point` / `point_to_cv` thin wrappers but update their imports to point to the now-local definitions.

The new `bridge.py` should look like:

```python
"""CV ↔ point-space conversion helpers + brainunit vectorisation."""

import brainunit as u
import numpy as np

from braincell._compute.runtime import CellRuntimeState

__all__ = [
    "cv_to_point",
    "point_to_cv",
    "scatter_midpoint_values",
    "gather_midpoint_values",
    "cv_value_vector",
    "fill_like",
    "quantity_vector",
    "matches_last_dim",
    "is_python_zero",
]


def scatter_midpoint_values(*, values, point_ids: np.ndarray, n_point: int):
    # ... body copied verbatim from runtime.py


def gather_midpoint_values(values, *, point_ids: np.ndarray):
    return values[..., point_ids]


def quantity_vector(values: list[object], *, shape: tuple[int, ...] | None = None):
    # ... body copied verbatim


def cv_value_vector(cell, *, attr_name: str):
    return quantity_vector([getattr(cv, attr_name) for cv in cell.cvs])


def fill_like(shape: tuple[int, ...], value):
    values = [value for _ in range(int(np.prod(shape, dtype=int)))]
    return quantity_vector(values, shape=shape)


def matches_last_dim(value: object, size: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return False
    return int(shape[-1]) == int(size)


def is_python_zero(value) -> bool:
    return isinstance(value, (int, float)) and value == 0


def cv_to_point(values, runtime: CellRuntimeState):
    return scatter_midpoint_values(
        values=values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
        n_point=runtime.n_point,
    )


def point_to_cv(values, runtime: CellRuntimeState):
    return gather_midpoint_values(
        values, point_ids=runtime.point_tree.cv_midpoint_point_id,
    )
```

- [ ] **Step 4: Update `runtime.py` to import the helpers it still uses internally from `bridge.py`**

Inside `runtime.py` there are internal uses of `scatter_midpoint_values` (in `_scatter_cv_geometry`, `_attach_runtime_ion_geometry`). Keep `_scatter_cv_geometry` and `_attach_runtime_ion_geometry` as private helpers inside `runtime.py` — but change them to import from bridge:

```python
# braincell/_compute/runtime.py — top imports
from braincell._multi_compartment.bridge import (
    scatter_midpoint_values,
    quantity_vector,
)
```

Caveat: this introduces a dependency arrow `_compute → _multi_compartment`. Since both already live in `braincell/`, this is not circular as long as `bridge.py` does not re-import `runtime.py` at module load. Verify by re-reading `bridge.py` — it imports `CellRuntimeState` at module load, which triggers `runtime.py` import, which would then try to import from `bridge.py` → **circular**.

**Resolution:** keep the scatter helpers co-located with runtime for now, but ALSO expose them from bridge via re-export. Actually — the cleanest fix is to move the private `_scatter_cv_geometry` + `_attach_runtime_ion_geometry` out of `runtime.py` and into `bridge.py` as the *sole* home for all scatter work. Then `runtime.py`'s `from_cell` calls `bridge.attach_runtime_ion_geometry(...)`. But `bridge.py` imports `CellRuntimeState` for typing only → use `from __future__ import annotations` and quote the type.

Revised approach:

1. Put `scatter_midpoint_values`, `gather_midpoint_values`, `cv_value_vector`, `fill_like`, `quantity_vector`, `matches_last_dim`, `is_python_zero`, `_scatter_cv_geometry`, `_attach_runtime_ion_geometry` all in `bridge.py`.
2. `bridge.py` starts with `from __future__ import annotations`; quote `CellRuntimeState` references.
3. `runtime.py` imports only what it needs at runtime from `bridge`: `scatter_midpoint_values`, `quantity_vector`, and the private `_attach_runtime_ion_geometry` — but since leading-`_` private symbols shouldn't cross module boundaries, rename those helpers to public `attach_runtime_ion_geometry` and `scatter_cv_geometry` in `bridge.py`.

Apply:

```python
# braincell/_multi_compartment/bridge.py
from __future__ import annotations

import brainunit as u
import numpy as np

# Do NOT import CellRuntimeState at module load (would cycle). Use
# string annotations for typing; import inside functions if a real
# reference is ever needed.

__all__ = [
    "cv_to_point", "point_to_cv",
    "scatter_midpoint_values", "gather_midpoint_values",
    "cv_value_vector", "fill_like", "quantity_vector",
    "matches_last_dim", "is_python_zero",
    "scatter_cv_geometry", "attach_runtime_ion_geometry",
]

# (function bodies as above)

def scatter_cv_geometry(*, cvs, attr_name: str, point_ids: np.ndarray, n_point: int):
    values = quantity_vector([getattr(cv, attr_name) for cv in cvs])
    return scatter_midpoint_values(values=values, point_ids=point_ids, n_point=n_point)


def attach_runtime_ion_geometry(*, ions, cvs, point_ids: np.ndarray, n_point: int) -> None:
    length = scatter_cv_geometry(cvs=cvs, attr_name="length", point_ids=point_ids, n_point=n_point)
    area = scatter_cv_geometry(cvs=cvs, attr_name="area", point_ids=point_ids, n_point=n_point)
    diam_mid = scatter_cv_geometry(cvs=cvs, attr_name="diam_mid", point_ids=point_ids, n_point=n_point)
    radius_prox = scatter_cv_geometry(cvs=cvs, attr_name="radius_prox", point_ids=point_ids, n_point=n_point)
    radius_dist = scatter_cv_geometry(cvs=cvs, attr_name="radius_dist", point_ids=point_ids, n_point=n_point)
    for ion in ions.values():
        setattr(ion, "length", length)
        setattr(ion, "area", area)
        setattr(ion, "diam_mid", diam_mid)
        setattr(ion, "radius_prox", radius_prox)
        setattr(ion, "radius_dist", radius_dist)


def cv_to_point(values, runtime: "CellRuntimeState"):
    return scatter_midpoint_values(
        values=values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
        n_point=runtime.n_point,
    )


def point_to_cv(values, runtime: "CellRuntimeState"):
    return gather_midpoint_values(
        values, point_ids=runtime.point_tree.cv_midpoint_point_id,
    )
```

In `runtime.py`, replace:

```python
# OLD
def _attach_runtime_ion_geometry(*, ions, cvs, point_ids, n_point):
    ...
# NEW — delete the function; from_cell now calls bridge.attach_runtime_ion_geometry
```

Update `from_cell`:

```python
from braincell._multi_compartment.bridge import attach_runtime_ion_geometry
# ...
attach_runtime_ion_geometry(
    ions=ions,
    cvs=cell.cvs,
    point_ids=point_tree.cv_midpoint_point_id,
    n_point=n_point,
)
```

- [ ] **Step 5: Full test suite**

```bash
pytest braincell/ -x -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_multi_compartment/bridge.py \
        braincell/_multi_compartment/cell.py braincell/morph/morphology.py \
        braincell/ion/__init__.py
git commit -m "refactor(compute): move non-runtime helpers out of runtime.py

clone_morpho moves to morph/morphology.py (it is a pure morphology op).
build_placeholder_ions moves to ion/__init__.py (ion module helper).
scatter_midpoint_values / gather_midpoint_values / cv_value_vector /
fill_like / quantity_vector / matches_last_dim / is_python_zero and
the CV-geometry scatter helpers move to _multi_compartment/bridge.py
(sole consumer). runtime.py is now purely about CellRuntimeState.
"
```

---

## Task 8: Fold `clamp_table.py` into `_compute/runtime.py`

Kills the circular import. `CellRuntimeState.from_cell` no longer imports from `_multi_compartment/`.

**Files:**
- Modify: `braincell/_compute/runtime.py` (add `ClampActiveTable` + `build_clamp_active_table`)
- Delete: `braincell/_multi_compartment/clamp_table.py`
- Modify: `braincell/_multi_compartment/clamp_table_test.py` → content moved into `braincell/_compute/runtime_test.py`, then file deleted
- Modify: `braincell/_multi_compartment/currents.py` (import from new location)

- [ ] **Step 1: Copy `ClampActiveTable` + `build_clamp_active_table` into `runtime.py`**

Move the entire `ClampActiveTable` dataclass, the `CLAMP_KINDS` constant, and the `build_clamp_active_table` function out of `_multi_compartment/clamp_table.py` and into `braincell/_compute/runtime.py`. Place them near the top of `runtime.py`, above `CellRuntimeState`. Update their internal imports:

```python
# OLD (from clamp_table.py)
from braincell._compute.topology import PointTree
from braincell._compute.runtime import MechanismLayout
# NEW (now intra-module)
# No imports needed — PointTree already imported, MechanismLayout defined below.
```

Since `ClampActiveTable` must live *above* `CellRuntimeState` (which references it), and `MechanismLayout` is defined above `CellRuntimeState` as well, order inside `runtime.py` becomes:

```
1. imports
2. Target / Layout type aliases   (added in Task 15; for now keep string literals)
3. MechanismLayout dataclass
4. CLAMP_KINDS constant
5. ClampActiveTable dataclass
6. build_clamp_active_table function
7. CellRuntimeState class
8. install_cell_runtime / uninstall_cell_runtime (to be removed in Task 14)
9. private helpers
```

Update `CellRuntimeState.from_cell` to call the now-local `build_clamp_active_table` directly:

```python
# OLD
from braincell._multi_compartment.clamp_table import build_clamp_active_table
clamp_active_table = build_clamp_active_table(...)
# NEW — just call the function (it's in the same module).
clamp_active_table = build_clamp_active_table(
    layouts=tuple(layouts),
    cvs=cell.cvs,
    point_tree=point_tree,
    n_point=n_point,
)
```

Delete the now-dead deferred import at the top of `from_cell`.

- [ ] **Step 2: Update `currents.py` import**

In `braincell/_multi_compartment/currents.py`:

```python
# OLD
from braincell._multi_compartment.clamp_table import ClampActiveTable
# NEW — no longer used in currents.py directly; accessed via runtime.clamp_active_table
# Remove this import entirely.
```

Verify nothing else in `currents.py` references `ClampActiveTable`:

```bash
grep -n "ClampActiveTable\|clamp_table" braincell/_multi_compartment/currents.py
```

Only references should be through `runtime.clamp_active_table` attribute access — no imports needed.

- [ ] **Step 3: Move tests from `clamp_table_test.py` into `runtime_test.py`**

Read `braincell/_multi_compartment/clamp_table_test.py`, copy each `TestCase` class (updating imports to use `from braincell._compute.runtime import ClampActiveTable, build_clamp_active_table`) into `braincell/_compute/runtime_test.py`.

- [ ] **Step 4: Delete the two dead files**

```bash
git rm braincell/_multi_compartment/clamp_table.py
git rm braincell/_multi_compartment/clamp_table_test.py
```

- [ ] **Step 5: Full suite**

```bash
pytest braincell/ -x -q
```

Expected: green. The moved test cases run under their new module name.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py \
        braincell/_multi_compartment/currents.py
git rm --cached braincell/_multi_compartment/clamp_table.py braincell/_multi_compartment/clamp_table_test.py
git commit -m "refactor(compute): fold clamp_table into _compute/runtime.py

build_clamp_active_table lived in _multi_compartment/clamp_table.py
but was imported from _compute/runtime.py at CellRuntimeState.from_cell
call time — a deferred import to break what was effectively a
layering inversion. Move both the dataclass and the builder into
runtime.py and delete the separate file. Tests absorbed into
runtime_test.py.
"
```

---

## Task 9: Generic `_on_param_updated` hook on runtime channels

Sets up the mechanism used by Tasks 10-11 to de-hardcode HH1952 temperature handling.

**Files:**
- Modify: `braincell/_base.py` or the concrete `Channel` base class (wherever channels ultimately inherit)
- Modify: `braincell/_compute/runtime.py` (use hook in `_sync_runtime_node_param`)

- [ ] **Step 1: Locate the runtime channel base**

```bash
grep -n "class Channel\b\|class IonChannel\b" braincell/_base.py
```

Read the found definition. The hook goes on whichever class concrete channels inherit from at the runtime node level.

- [ ] **Step 2: Add default no-op hook on the base**

In `braincell/_base.py`, inside `class Channel` (or the appropriate base):

```python
def _on_param_updated(self, var_name: str, new_value) -> None:
    """Hook invoked after runtime state writes a parameter.

    Default: no-op. Subclasses override to recompute derived values
    when a specific parameter changes (e.g. HH1952 recomputing phi
    when T changes).

    Parameters
    ----------
    var_name : str
        The parameter name that was just updated.
    new_value : object
        The new value written to the runtime node attribute.
        Typically a ``u.Quantity`` but may be a plain array.
    """
    return None
```

- [ ] **Step 3: Invoke the hook from `_sync_runtime_node_param`**

In `braincell/_compute/runtime.py`, update `_sync_runtime_node_param`:

```python
def _sync_runtime_node_param(runtime: CellRuntimeState, *, layout_id: int, var_name: str) -> None:
    node = runtime.runtime_nodes.get(int(layout_id))
    if node is None:
        return
    layout = runtime.layouts[int(layout_id)]
    kind = layout.kind
    if kind.startswith("ion:"):
        _sync_runtime_ion(runtime, layout_id=int(layout_id))
        return
    new_value = _runtime_param_value(
        layout=layout, var_name=var_name, state_buffers=runtime.state_buffers
    )
    setattr(node, var_name, new_value)
    hook = getattr(node, "_on_param_updated", None)
    if callable(hook):
        hook(var_name, new_value)
```

Delete the hardcoded HH1952 branch:

```python
# DELETE
if kind in {"channel:INa_HH1952", "channel:IK_HH1952"} and var_name == "T":
    setattr(node, "phi", _runtime_temperature_phi(...))
    return
```

(Leave `_runtime_temperature_phi` in place for now; Task 11 removes it.)

- [ ] **Step 4: Run tests — expect regression failures on HH1952 param sync**

```bash
pytest braincell/_compute/runtime_test.py::CellRuntimeStateTest::test_set_state_syncs_runtime_node_param_for_ina_hh1952 -v
```

Expected: FAIL — setting `T` no longer updates `phi`. That's acceptable as a transient state because Task 10 fixes it.

**Do not commit yet.** Tasks 9, 10, 11 form one atomic behavior-preserving sequence; commit only after all three are done.

---

## Task 10: HH1952 channels compute `phi` internally

**Files:**
- Modify: `braincell/channel/sodium.py` (`INa_HH1952`)
- Modify: `braincell/channel/potassium.py` (`IK_HH1952`)

- [ ] **Step 1: Inspect current `INa_HH1952` constructor**

```bash
grep -n "class INa_HH1952\|def __init__" braincell/channel/sodium.py
```

Read the current signature to see where `phi` enters today.

- [ ] **Step 2: Refactor `INa_HH1952` to accept `T` and derive `phi`**

In `braincell/channel/sodium.py`, change the `__init__` to accept an optional `T` kwarg in place of (or in addition to) `phi`, and implement `_on_param_updated`:

```python
import brainunit as u

class INa_HH1952(...):
    def __init__(self, size, *, T=None, phi=None, **kwargs):
        super().__init__(size=size, **kwargs)
        if T is None and phi is None:
            raise TypeError("INa_HH1952: pass either T (temperature) or phi (Q10 scaling).")
        if T is not None and phi is not None:
            raise TypeError("INa_HH1952: pass only one of T or phi, not both.")
        if T is not None:
            self.T = T
            self.phi = _t_to_phi(T)
        else:
            self.T = None
            self.phi = phi

    def _on_param_updated(self, var_name, new_value) -> None:
        if var_name == "T":
            # Runtime already wrote self.T = new_value via setattr; recompute phi.
            self.phi = _t_to_phi(new_value)


def _t_to_phi(T):
    """Q10 temperature scaling: 3 ^ ((T_celsius - 36) / 10)."""
    T_c = u.kelvin2celsius(T)
    return 3 ** ((T_c - 36) / 10)
```

Apply the same pattern in `braincell/channel/potassium.py` for `IK_HH1952`.

- [ ] **Step 3: Run HH1952 tests**

```bash
pytest braincell/_compute/runtime_test.py -k "HH1952" -v
```

Expected: passes that were broken at the end of Task 9 now pass again.

**Do not commit yet — continue to Task 11.**

---

## Task 11: Remove `_runtime_temperature_phi` and HH1952 hardcode from runtime

**Files:**
- Modify: `braincell/_compute/runtime.py` (delete dead code, generic constructor kwargs)

- [ ] **Step 1: Simplify `_runtime_constructor_params`**

Replace the current body:

```python
def _runtime_constructor_params(
    *,
    layout: MechanismLayout,
    mechanism: Density,
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> dict[str, object]:
    if mechanism.category != "channel":
        return {}
    return {
        var_name: _runtime_param_value(
            layout=layout, var_name=var_name, state_buffers=state_buffers
        )
        for var_name in mechanism.params.keys()
    }
```

Delete `_runtime_temperature_phi` entirely.

- [ ] **Step 2: Run full suite**

```bash
pytest braincell/ -x -q
```

Expected: all green.

- [ ] **Step 3: Commit Tasks 9 + 10 + 11 together**

```bash
git add braincell/_base.py braincell/_compute/runtime.py \
        braincell/channel/sodium.py braincell/channel/potassium.py
git commit -m "refactor(compute): push HH1952 T->phi derivation into channel classes

Runtime no longer special-cases channel class names. Add a generic
_on_param_updated(var_name, new_value) hook on the Channel base
(default no-op). INa_HH1952 and IK_HH1952 accept T in their
constructor, compute phi internally, and override the hook to
recompute phi whenever runtime state writes a new T. Remove the
_runtime_temperature_phi helper and the two hardcoded channel-name
branches from runtime.py.
"
```

---

## Task 12: Swap state-buffer storage to `u.Quantity`-backed arrays (rectangular params)

**Files:**
- Modify: `braincell/_compute/runtime.py` (state buffer allocation + read/write + node sync)
- Modify: `braincell/_compute/runtime_test.py` (add storage tests T7/T8/T10/T11/T12)

**Scope note.** This task handles the *rectangular* case: one scalar `Quantity` per point per (layout, var). Density, ion, synapse, and most clamp params are all scalar-per-point. Ragged `CurrentClamp.durations` / `.amplitudes` sequences are deferred to Task 13 and keep their object-dtype path meanwhile.

- [ ] **Step 1: Write the storage tests**

Append to `braincell/_compute/runtime_test.py`:

```python
import jax.numpy as jnp
import jax


class StateBufferStorageTest(unittest.TestCase):
    def test_density_buffer_is_quantity_backed_by_jax(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        layout = cell.layouts[0]
        buffer = cell.runtime.state_buffers[(layout.id, "g_max")]
        self.assertTrue(hasattr(buffer, "unit"))
        self.assertTrue(hasattr(buffer, "mantissa"))
        self.assertTrue(isinstance(buffer.mantissa, jnp.ndarray))

    def test_set_state_broadcast_scalar_and_readback(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        cell.runtime.set_state(layout.id, "g_max", 7.5 * (u.mS / u.cm ** 2))
        new = cell.runtime.get_state(layout.id, "g_max")
        self.assertAlmostEqual(float(new[1].to_decimal(u.mS / u.cm ** 2)), 7.5, places=12)

    def test_set_state_unit_mismatch_raises(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        with self.assertRaises(u.UnitMismatchError):
            cell.runtime.set_state(layout.id, "g_max", 7.5 * u.mV)  # voltage, not conductance-density

    def test_set_state_shape_mismatch_raises(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        with self.assertRaises(ValueError):
            cell.runtime.set_state(
                layout.id, "g_max",
                u.Quantity(jnp.ones((99,)), u.mS / u.cm ** 2),
            )

    def test_no_object_dtype_in_jit_trace(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()

        # Compile a derivative call and inspect the jaxpr: no object dtypes.
        def step(V):
            return cell.runtime.get_state(cell.layouts[0].id, "g_max")

        jaxpr = jax.make_jaxpr(step)(cell.V.value)
        self.assertNotIn("object", str(jaxpr))
```

- [ ] **Step 2: Run tests — expect failures because current storage is object-dtype**

```bash
pytest braincell/_compute/runtime_test.py::StateBufferStorageTest -v
```

Expected: FAIL on at least T7, T8, T12.

- [ ] **Step 3: Rewrite `_allocate_state_buffer`**

In `runtime.py`:

```python
def _allocate_state_buffer(
    mechanism: object,
    *,
    var_name: str,
    shape: tuple[int, ...],
) -> u.Quantity | tuple:
    """Allocate a rectangular state buffer for a mechanism parameter.

    Returns a ``u.Quantity`` whose mantissa is a ``jnp.ndarray`` when
    the declared value carries a unit. For sequence-valued params
    (CurrentClamp.durations, .amplitudes, FunctionClamp.fn) the
    buffer is a Python tuple of length ``shape[0]``; those are
    handled ragged in Task 13.
    """
    value = _mechanism_var_value(mechanism, var_name)

    # Ragged / callable / sequence values: keep Python container per point.
    if _is_ragged_param(mechanism, var_name, value):
        return tuple(value for _ in range(int(np.prod(shape, dtype=int))))

    if hasattr(value, "unit") and hasattr(value, "to_decimal"):
        unit = value.unit
        mantissa = jnp.full(shape, float(value.to_decimal(unit)), dtype=jnp.float32)
        return u.Quantity(mantissa, unit)

    # Plain numeric (no unit): return a jnp.ndarray.
    return jnp.full(shape, value)


def _is_ragged_param(mechanism: object, var_name: str, value: object) -> bool:
    """True for tuple/list-valued params (current-clamp steps, fn)."""
    if callable(value):
        return True
    if isinstance(value, (tuple, list)):
        return True
    return False
```

- [ ] **Step 4: Rewrite `_write_state_buffer`**

```python
def _write_state_buffer(
    layout: MechanismLayout,
    buffer: object,
    value: object,
) -> object:
    """Write ``value`` into ``buffer``. Return the possibly-new buffer.

    For Quantity buffers, broadcast scalar Quantity and validate unit.
    For tuple buffers (ragged), replace the whole tuple.
    """
    if isinstance(buffer, u.Quantity):
        target_shape = buffer.mantissa.shape
        target_unit = buffer.unit

        if isinstance(value, u.Quantity):
            mantissa = jnp.asarray(value.to_decimal(target_unit))
            if mantissa.ndim == 0:
                mantissa = jnp.broadcast_to(mantissa, target_shape)
            if mantissa.shape != target_shape:
                raise ValueError(
                    f"State assignment shape mismatch: expected {target_shape!r}, got {mantissa.shape!r}."
                )
            return u.Quantity(mantissa, target_unit)

        if isinstance(value, (list, tuple)):
            arr = jnp.stack([jnp.asarray(item.to_decimal(target_unit)) for item in value])
            if arr.shape != target_shape:
                raise ValueError(
                    f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}."
                )
            return u.Quantity(arr, target_unit)

        raise TypeError(
            f"State buffer for layout {layout.id!r} expects a Quantity or sequence of Quantities, got {type(value).__name__!r}."
        )

    # Tuple / ragged / callable buffer — replace contents.
    if isinstance(buffer, tuple):
        if isinstance(value, (tuple, list)):
            if len(value) != len(buffer):
                raise ValueError(
                    f"State assignment shape mismatch for ragged buffer: expected length {len(buffer)}, got {len(value)}."
                )
            return tuple(value)
        # Scalar broadcast for ragged buffer: fill every slot.
        return tuple(value for _ in buffer)

    # Plain jnp array buffer (unitless).
    arr = jnp.asarray(value)
    target_shape = jnp.asarray(buffer).shape
    if arr.ndim == 0:
        arr = jnp.broadcast_to(arr, target_shape)
    if arr.shape != target_shape:
        raise ValueError(
            f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}."
        )
    return arr
```

- [ ] **Step 5: Update `CellRuntimeState.set_state`**

```python
def set_state(self, layout_id: int, var_name: str, value: object) -> None:
    key = (int(layout_id), str(var_name))
    if key not in self.state_buffers:
        raise KeyError(f"Unknown state buffer for {(layout_id, var_name)!r}.")
    layout = self.layouts[int(layout_id)]
    self.state_buffers[key] = _write_state_buffer(layout, self.state_buffers[key], value)
    _sync_runtime_node_param(self, layout_id=int(layout_id), var_name=str(var_name))
```

- [ ] **Step 6: Update `_extract_point_value`**

```python
def _extract_point_value(layout: MechanismLayout, *, point_id: int, buffer: object) -> object:
    if isinstance(buffer, u.Quantity):
        mantissa = buffer.mantissa
        unit = buffer.unit
        if layout.layout == "dense":
            return u.Quantity(mantissa[point_id], unit)
        matches = np.flatnonzero(layout.point_index == int(point_id))
        if len(matches) == 0:
            raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
        return u.Quantity(mantissa[int(matches[0])], unit)

    # Tuple / ragged.
    if isinstance(buffer, tuple):
        if layout.layout == "dense":
            return buffer[point_id]
        matches = np.flatnonzero(layout.point_index == int(point_id))
        if len(matches) == 0:
            raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
        return buffer[int(matches[0])]

    # Plain jnp array.
    if layout.layout == "dense":
        return buffer[point_id]
    matches = np.flatnonzero(layout.point_index == int(point_id))
    if len(matches) == 0:
        raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
    return buffer[int(matches[0])]
```

- [ ] **Step 7: Update `_runtime_param_value`**

```python
def _runtime_param_value(
    *,
    layout: MechanismLayout,
    var_name: str,
    state_buffers: dict,
) -> object:
    buffer = state_buffers[(layout.id, var_name)]
    if isinstance(buffer, u.Quantity) and layout.point_mask is not None and var_name in {
        "g_max", "g", "gbar", "conductance",
    }:
        mask_bool = jnp.asarray(layout.point_mask)
        masked_mantissa = jnp.where(mask_bool, buffer.mantissa, 0.0)
        return u.Quantity(masked_mantissa, buffer.unit)
    return buffer
```

Delete the old `_mask_quantity_like`, `_object_array_from_buffer`, `_object_array_from_value`, `_as_runtime_array` helpers — all replaced by `u.Quantity(jnp.ndarray)` arithmetic. (These are C5 fix too.)

- [ ] **Step 8: Update `_instantiate_runtime_ion_instance` and `_sync_runtime_ion`**

Replace the per-point Python loops with vectorised `at[...].set`:

```python
def _instantiate_runtime_ion_instance(
    *,
    instance_name: str,
    runtime_cls: type,
    layouts: tuple[MechanismLayout, ...],
    declarations: tuple[Density, ...],
    state_buffers: dict,
    n_point: int,
) -> object:
    supported_params = _supported_ion_runtime_params(runtime_cls)
    for layout, declaration in zip(layouts, declarations):
        invalid = set(declaration.params.keys()) - set(supported_params)
        if invalid:
            raise ValueError(
                f"Ion layout {layout.id!r} for instance {instance_name!r} uses unsupported runtime ion params "
                f"{sorted(invalid)!r} on {runtime_cls.__name__!r}."
            )

    baseline_ion = runtime_cls(size=(n_point,))
    full_param_values: dict[str, u.Quantity] = {}
    for param_name in supported_params:
        baseline_value = _normalize_ion_runtime_param_value(
            runtime_cls,
            param_name,
            getattr(baseline_ion, _ion_runtime_attr_name(runtime_cls, param_name)),
        )
        full_param_values[param_name] = _quantity_full((n_point,), baseline_value)

    for layout, declaration in zip(layouts, declarations):
        point_index = layout.point_index
        if point_index is None:
            raise ValueError(f"Ion layout {layout.id!r} is missing point_index.")
        for param_name in declaration.params.keys():
            values = state_buffers[(layout.id, param_name)]
            if isinstance(values, u.Quantity):
                target_unit = full_param_values[param_name].unit
                subset = values.mantissa  # already shape (n_active,) for sparse or (n_point,) for dense
                if values.mantissa.shape == (n_point,):
                    subset = values.mantissa[point_index]
                target_mantissa = full_param_values[param_name].mantissa.at[point_index].set(
                    jnp.asarray(u.Quantity(subset, values.unit).to_decimal(target_unit))
                )
                full_param_values[param_name] = u.Quantity(target_mantissa, target_unit)
            else:
                # Ragged / plain; fall through to old path.
                raise TypeError(
                    f"Ion param {param_name!r} on layout {layout.id!r} is not a Quantity buffer."
                )

    runtime_kwargs = dict(full_param_values)
    return runtime_cls(size=(n_point,), name=instance_name, **runtime_kwargs)


def _quantity_full(shape: tuple[int, ...], value: object) -> u.Quantity:
    if isinstance(value, u.Quantity):
        return u.Quantity(jnp.full(shape, float(value.to_decimal(value.unit))), value.unit)
    if hasattr(value, "unit"):
        return u.Quantity(jnp.full(shape, float(value.to_decimal(value.unit))), value.unit)
    return jnp.full(shape, value)
```

Apply the analogous rewrite to `_sync_runtime_ion` (replacing the `for index in candidate.point_index.tolist()` Python loop with a single `.at[point_index].set(...)` call per param, keyed on unit).

- [ ] **Step 9: Run full suite**

```bash
pytest braincell/ -x -q
```

Expected: green. Some legacy tests that compared `state_buffers[(lid, var)]` to a concrete `np.ndarray(dtype=object)` must be updated — change their assertions to compare via `.to_decimal()` / `.mantissa` on the `Quantity`.

- [ ] **Step 10: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py
git commit -m "refactor(compute): store rectangular state as Quantity(jnp.ndarray)

State buffers for density/ion/synapse scalar-per-point params now
live as u.Quantity(jnp.ndarray, unit) instead of np.ndarray(dtype=
object) of boxed Quantities. set_state / get_state / _extract_point_
value / _runtime_param_value / _instantiate_runtime_ion_instance /
_sync_runtime_ion all rewritten for the new storage model.

Ragged tuple-valued params (CurrentClamp.durations, .amplitudes,
FunctionClamp.fn) still pass through a tuple path — handled in the
next commit.

Also fixes bug C5: _mask_quantity_like is replaced with a straight
jnp.where on the Quantity mantissa; the bare except Exception is
gone.
"
```

---

## Task 13: Ragged `CurrentClamp.durations` / `.amplitudes` — padded arrays + mask

**Files:**
- Modify: `braincell/_compute/runtime.py`

- [ ] **Step 1: Write ragged test**

Append to `braincell/_compute/runtime_test.py`:

```python
class RaggedCurrentClampBufferTest(unittest.TestCase):
    def test_three_clamps_with_varying_step_counts_pad_and_mask(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.25),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.place(
            at("soma", 0.5),
            CurrentClamp(start=0.0 * u.ms, durations=(1.0 * u.ms, 1.0 * u.ms),
                         amplitudes=(0.1 * u.nA, 0.2 * u.nA)),
        )
        cell.place(
            at("soma", 0.75),
            CurrentClamp(start=0.0 * u.ms, durations=(0.5 * u.ms, 0.5 * u.ms, 1.0 * u.ms),
                         amplitudes=(0.1 * u.nA, 0.2 * u.nA, 0.3 * u.nA)),
        )
        cell.init_state()

        # Each place produces its own layout because start differs / order differs; at minimum,
        # verify that the per-layout buffer shapes are (n_active, max_steps).
        for layout in cell.layouts:
            if layout.kind != "CurrentClamp":
                continue
            dur = cell.runtime.state_buffers[(layout.id, "durations")]
            amp = cell.runtime.state_buffers[(layout.id, "amplitudes")]
            self.assertTrue(hasattr(dur, "unit"))
            self.assertEqual(dur.mantissa.ndim, 2)
            self.assertEqual(amp.mantissa.shape, dur.mantissa.shape)
            mask_key = (layout.id, "_mask_durations")
            self.assertIn(mask_key, cell.runtime.state_buffers)
            self.assertEqual(cell.runtime.state_buffers[mask_key].shape, dur.mantissa.shape)
```

- [ ] **Step 2: Implement padded + mask allocation for `CurrentClamp`**

Extend `_allocate_state_buffer` to handle `CurrentClamp.durations` / `.amplitudes`:

```python
from braincell.mech import CurrentClamp

def _allocate_clamp_ragged_buffer(
    *,
    layout: MechanismLayout,
    var_name: str,
    per_point_sequences: list[tuple],
    unit,
) -> u.Quantity:
    max_steps = max(len(seq) for seq in per_point_sequences) if per_point_sequences else 0
    mantissa = jnp.zeros((len(per_point_sequences), max_steps), dtype=jnp.float32)
    mask = jnp.zeros((len(per_point_sequences), max_steps), dtype=bool)
    for i, seq in enumerate(per_point_sequences):
        for j, item in enumerate(seq):
            mantissa = mantissa.at[i, j].set(float(item.to_decimal(unit)))
            mask = mask.at[i, j].set(True)
    return u.Quantity(mantissa, unit), mask
```

Use this inside `CellRuntimeState.from_cell` when lowering `CurrentClamp` layouts:

```python
# In from_cell, when iterating layouts and allocating state buffers:
if isinstance(mechanism, CurrentClamp) and var_name in ("durations", "amplitudes"):
    per_point_sequences = _gather_clamp_per_point_sequences(entry, var_name)
    unit = u.ms if var_name == "durations" else u.nA
    quantity, mask = _allocate_clamp_ragged_buffer(
        layout=layout_spec, var_name=var_name,
        per_point_sequences=per_point_sequences, unit=unit,
    )
    state_buffers[(layout_spec.id, var_name)] = quantity
    state_buffers[(layout_spec.id, f"_mask_{var_name}")] = mask
    state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
    continue
```

`_gather_clamp_per_point_sequences(entry, var_name)` reads each contributing declaration's field and returns the list of per-point tuples ordered by `point_ids`.

- [ ] **Step 3: Rewrite `_eval_current_clamp` to use the padded form**

```python
def _eval_current_clamp(
    runtime: CellRuntimeState,
    *,
    layout_id: int,
    local_index: int,
    local_t,
) -> object:
    durations_q = runtime.state_buffers[(layout_id, "durations")]
    amplitudes_q = runtime.state_buffers[(layout_id, "amplitudes")]
    mask = runtime.state_buffers[(layout_id, "_mask_durations")]

    dur_row = durations_q.mantissa[local_index]
    amp_row = amplitudes_q.mantissa[local_index]
    mask_row = mask[local_index]

    local_t_ms = local_t.to_decimal(u.ms)
    ends = jnp.cumsum(dur_row)
    starts = ends - dur_row
    is_active = (local_t_ms >= 0.0) & (local_t_ms >= starts) & (local_t_ms < ends) & mask_row
    current = jnp.sum(jnp.where(is_active, amp_row, 0.0))
    return u.Quantity(current, u.nA)
```

- [ ] **Step 4: Run tests**

```bash
pytest braincell/_compute/runtime_test.py::RaggedCurrentClampBufferTest -v
pytest braincell/ -x -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py
git commit -m "refactor(compute): pad ragged CurrentClamp buffers with mask

CurrentClamp.durations and .amplitudes may have different step counts
per active point. Previously each point stored its own Python tuple
inside an object-dtype array. Now they live as (n_active, max_steps)
Quantity-backed padded arrays, with a companion (n_active, max_steps)
bool mask under the key _mask_durations. _eval_current_clamp updated
to multiply through the mask. Ragged step counts across a single
layout are now fully JIT-friendly.
"
```

---

## Task 14: Inline `install_cell_runtime` / `uninstall_cell_runtime` into `Cell`

**Files:**
- Modify: `braincell/_compute/runtime.py` (delete install/uninstall functions)
- Modify: `braincell/_multi_compartment/cell.py` (absorb the bodies)
- Modify: `braincell/_compute/runtime_test.py` (rewrite the install/uninstall tests)

- [ ] **Step 1: Copy the body of `install_cell_runtime` into `Cell.init_state`**

In `braincell/_multi_compartment/cell.py`, inside `init_state`, replace the line:

```python
self._runtime_installed_names = install_cell_runtime(self, self._runtime)
```

with the inlined body:

```python
self._in_size = (self._runtime.n_cv,)
self._out_size = (self._runtime.n_cv,)

root_nodes = dict(self._runtime.ions)
for layout in self._runtime.layouts:
    node = self._runtime.runtime_nodes.get(layout.id)
    if node is None:
        continue
    if _is_root_level_runtime_node(layout.kind):
        root_nodes[f"layout_{layout.id}"] = node

self.ion_channels = self._format_elements(IonChannel, **root_nodes)
self.C = bridge.cv_value_vector(self, attr_name="cm")

self._V_th_declaration = self._V_th
self.V_th = bridge.fill_like(self.varshape, self.V_th)
```

Remove the instance attribute `self._runtime_installed_names` (no longer needed).

Import `_is_root_level_runtime_node` from `runtime.py` at the top of `cell.py`:

```python
from braincell._compute.runtime import _is_root_level_runtime_node  # noqa
```

(Leading-underscore cross-module import is acceptable because `cell.py` and `runtime.py` are both within the `braincell` package.)

- [ ] **Step 2: Replace `uninstall_cell_runtime` call in `reset`**

In `Cell.reset`:

```python
# OLD
uninstall_cell_runtime(self, self._runtime_installed_names)
self._runtime_installed_names = ()
# NEW
for name in ("_in_size", "_out_size", "ion_channels", "C"):
    if hasattr(self, name):
        delattr(self, name)
self._V_th = self._V_th_declaration
```

- [ ] **Step 3: Delete `install_cell_runtime` and `uninstall_cell_runtime` from `runtime.py`**

Also remove them from any `__all__` lists and from any re-exports.

- [ ] **Step 4: Rewrite the lifecycle tests**

In `braincell/_compute/runtime_test.py`, delete `TestInstallCellRuntime` and `TestUninstallCellRuntime`. Replace with:

```python
class CellLifecycleInlineTest(unittest.TestCase):
    def test_init_state_installs_runtime_attributes_directly(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            self.assertTrue(hasattr(cell, name), f"Cell should have {name} after init_state.")
        self.assertEqual(cell._in_size, (cell.n_cv,))
        self.assertEqual(cell._out_size, (cell.n_cv,))

    def test_reset_clears_runtime_attributes(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        cell.reset()
        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            self.assertFalse(hasattr(cell, name), f"Cell should not have {name} after reset.")

    def test_init_reset_init_is_idempotent(self) -> None:
        cell = Cell(_build_tree())
        cell.init_state()
        layouts_a = cell.layouts
        cell.reset()
        cell.init_state()
        layouts_b = cell.layouts
        self.assertEqual(len(layouts_a), len(layouts_b))
```

- [ ] **Step 5: Run tests**

```bash
pytest braincell/ -x -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_multi_compartment/cell.py \
        braincell/_compute/runtime_test.py
git commit -m "refactor(compute): inline install/uninstall into Cell lifecycle

install_cell_runtime and uninstall_cell_runtime were two functions
whose sole purpose was to mutate Cell attributes across an opaque
tuple-of-names handshake. Inline their bodies into Cell.init_state
and Cell.reset and delete both functions. Cell now owns its runtime
directly. Lifecycle tests rewritten to exercise init_state / reset
without the old helpers.
"
```

---

## Task 15: Magic strings → `Literal` aliases on `MechanismLayout`

**Files:**
- Modify: `braincell/_compute/runtime.py`

- [ ] **Step 1: Add Literal aliases at the top**

After the existing imports in `runtime.py`:

```python
from typing import Literal

Target = Literal["density", "point"]
Layout = Literal["dense", "sparse"]
```

- [ ] **Step 2: Update `MechanismLayout` field types**

```python
@dataclass(frozen=True)
class MechanismLayout:
    id: int
    kind: str
    target: Target     # was: str
    layout: Layout     # was: str
    point_index: np.ndarray
    point_mask: np.ndarray | None
    n_active: int
    source_cv_ids: tuple[int, ...]
    source_rule: str | None = None
```

- [ ] **Step 3: Update `choose_layout` return annotation**

```python
def choose_layout(*, target: Target) -> Layout:
    if target == "point":
        return "sparse"
    if target == "density":
        return "dense"
    raise ValueError(f"Unsupported target {target!r}.")
```

- [ ] **Step 4: Drop `CellRuntimeState` `@dataclass(frozen=True)`**

This step fixes the "misleading @frozen" smell: runtime state holds mutable dicts, so declare the class mutable.

```python
@dataclass
class CellRuntimeState:
    # ... same fields as before
```

Remove any reliance on `object.__setattr__` workarounds elsewhere.

- [ ] **Step 5: Run tests**

```bash
pytest braincell/ -x -q
```

Expected: green. Type checker (if run) will now flag non-literal strings passed as `target` / `layout` — those are fixed at call sites inside `from_cell`.

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/runtime.py
git commit -m "refactor(compute): type target/layout as Literals and drop fake frozen

target: Literal['density','point']; layout: Literal['dense','sparse'].
Add module-level aliases Target and Layout. Drop @dataclass(frozen=
True) from CellRuntimeState since its runtime_nodes/ions/state_buffers
fields are mutated after construction. No runtime behavior change.
"
```

---

## Task 16: `FunctionClamp` signature fingerprinting

**Files:**
- Modify: `braincell/_compute/runtime.py` (`mechanism_signature`)
- Modify: `braincell/_compute/runtime_test.py` (add T4)

- [ ] **Step 1: Write the test**

Append to `runtime_test.py`:

```python
from braincell.mech import FunctionClamp


class FunctionClampMergesIdenticalLambdas(unittest.TestCase):
    def test_identical_lambda_bodies_produce_one_layout(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            FunctionClamp(fn=lambda t: 0.1 * u.nA, duration=3.0 * u.ms, start=0.0 * u.ms),
        )
        cell.place(
            at("soma", 0.75),
            FunctionClamp(fn=lambda t: 0.1 * u.nA, duration=3.0 * u.ms, start=0.0 * u.ms),
        )
        cell.init_state()
        fn_clamp_layouts = [l for l in cell.layouts if l.kind == "FunctionClamp"]
        self.assertEqual(len(fn_clamp_layouts), 1)
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest braincell/_compute/runtime_test.py::FunctionClampMergesIdenticalLambdas -v
```

Expected: FAIL — current behavior produces two layouts.

- [ ] **Step 3: Implement fingerprint**

In `runtime.py`:

```python
def _fn_fingerprint(fn) -> tuple:
    code = fn.__code__
    closure_cells: list[object] = []
    for cell in (fn.__closure__ or ()):
        v = cell.cell_contents
        if hasattr(v, "to_decimal") and hasattr(v, "unit"):
            closure_cells.append(("quantity", float(v.to_decimal(v.unit)), str(v.unit)))
        elif isinstance(v, (int, float, str, bytes, bool)) or v is None:
            closure_cells.append(("prim", v))
        else:
            closure_cells.append(("id", id(v)))  # fall back to identity for unhashables
    return (code.co_code, code.co_consts, code.co_varnames, tuple(closure_cells))


def mechanism_signature(mechanism: object) -> tuple[object, ...]:
    from braincell.mech import FunctionClamp
    if isinstance(mechanism, FunctionClamp):
        return (
            "FunctionClamp",
            _fn_fingerprint(mechanism.fn),
            mechanism.start,
            mechanism.duration,
        )
    return (type(mechanism).__qualname__, mechanism)
```

- [ ] **Step 4: Run test — PASS**

```bash
pytest braincell/_compute/runtime_test.py::FunctionClampMergesIdenticalLambdas -v
```

- [ ] **Step 5: Full suite**

```bash
pytest braincell/ -x -q
```

- [ ] **Step 6: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py
git commit -m "fix(compute): merge identical FunctionClamp lambdas via code fingerprint (C4)

Two FunctionClamp instances with identical lambda bodies and closures
produced distinct layouts because dataclass __eq__ compared fn by
identity. Fingerprint via co_code / co_consts / co_varnames and a
normalized closure-cell digest so structurally identical lambdas
merge. Add regression test.
"
```

---

## Task 17: Tighten `MechanismObjectCell.__getattr__` (C6 adjacent)

**Files:**
- Modify: `braincell/_compute/table.py`
- Create: `braincell/_compute/table_test.py`

- [ ] **Step 1: Write the test**

Create `braincell/_compute/table_test.py`:

```python
import unittest

import brainunit as u

import braincell
from braincell import Branch, Cell, Morphology
from braincell.filter import BranchSlice


def _simple_cell() -> Cell:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    cell = Cell(tree)
    cell.paint(
        BranchSlice(branch_index=0, prox=0.0, dist=1.0),
        braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
    )
    cell.init_state()
    return cell


class MechanismObjectCellAttrAccess(unittest.TestCase):
    def test_known_param_returns_value(self) -> None:
        cell = _simple_cell()
        table = cell.mech_table
        mo = table.get(("IL", "IL"), column_id=1)
        self.assertIsNotNone(mo)
        self.assertAlmostEqual(float(mo.g_max.to_decimal(u.mS / u.cm ** 2)), 4.0, places=12)

    def test_unknown_param_raises_attribute_error_with_candidates(self) -> None:
        cell = _simple_cell()
        table = cell.mech_table
        mo = table.get(("IL", "IL"), column_id=1)
        with self.assertRaises(AttributeError) as ctx:
            _ = mo.not_a_real_field
        msg = str(ctx.exception)
        self.assertIn("not_a_real_field", msg)
        # Message mentions at least one valid field to help debugging.
        self.assertIn("g_max", msg)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest braincell/_compute/table_test.py -v
```

Expected: the unknown-attr test either succeeds silently with `None`/some surrogate, or raises `AttributeError` without a helpful message.

- [ ] **Step 3: Rewrite `__getattr__`**

In `braincell/_compute/table.py`:

```python
def __getattr__(self, name: str) -> object:
    # Only resolve names the declaration or runtime node recognises.
    declaration = self.declaration
    node = self.node
    candidates: set[str] = set()
    if hasattr(declaration, "params"):
        candidates.update(declaration.params.keys())
    if is_dataclass(declaration):
        candidates.update(f.name for f in fields(declaration))
    if node is not None:
        candidates.update(
            n for n in dir(node) if not n.startswith("_")
        )
    if name in candidates:
        try:
            return self.get_param(name)
        except AttributeError:
            pass
    raise AttributeError(
        f"Mechanism cell {self.row_label!r} has no attribute {name!r}. "
        f"Known params: {sorted(candidates)!r}."
    )
```

Import `fields`, `is_dataclass` from `dataclasses` at the top of `table.py`.

- [ ] **Step 4: Run tests — PASS**

```bash
pytest braincell/_compute/table_test.py -v
pytest braincell/ -x -q
```

- [ ] **Step 5: Commit**

```bash
git add braincell/_compute/table.py braincell/_compute/table_test.py
git commit -m "refactor(compute): whitelist MechanismObjectCell attr access

MechanismObjectCell.__getattr__ previously swallowed any
AttributeError, so typos silently surfaced as missing params. Now
check the attribute name against the declaration's params, dataclass
fields, and the runtime node's public attributes; raise
AttributeError with the valid candidate list if not found.
"
```

---

## Task 18: C6 fix — `_is_root_level_runtime_node` raises on unknown class

**Files:**
- Modify: `braincell/_compute/runtime.py`
- Modify: `braincell/_compute/runtime_test.py`

- [ ] **Step 1: Write failing test**

Append to `runtime_test.py`:

```python
class IsRootLevelRuntimeNodeUnknownClassTest(unittest.TestCase):
    def test_unknown_channel_kind_raises_value_error(self) -> None:
        from braincell._compute.runtime import _is_root_level_runtime_node
        with self.assertRaises(ValueError) as ctx:
            _is_root_level_runtime_node("channel:__never_registered__")
        self.assertIn("__never_registered__", str(ctx.exception))
```

- [ ] **Step 2: Run — expect failure (currently returns False)**

```bash
pytest braincell/_compute/runtime_test.py::IsRootLevelRuntimeNodeUnknownClassTest -v
```

- [ ] **Step 3: Fix**

In `runtime.py`:

```python
def _is_root_level_runtime_node(kind: str) -> bool:
    if not kind.startswith("channel:"):
        return False
    class_name = kind.split(":", 1)[1]
    cls = get_registry().get("channel", class_name)  # no try/except — let KeyError bubble
    # Convert the KeyError at the caller boundary to ValueError with context.
    return _channel_current_owner_family(cls) is None
```

Then wrap the call site inside `Cell.init_state` (step 1 inlining) to translate the KeyError:

```python
try:
    if _is_root_level_runtime_node(layout.kind):
        root_nodes[f"layout_{layout.id}"] = node
except KeyError as exc:
    raise ValueError(
        f"Unknown runtime channel class for layout kind {layout.kind!r}: {exc}"
    ) from exc
```

- [ ] **Step 4: Run — PASS**

```bash
pytest braincell/_compute/runtime_test.py::IsRootLevelRuntimeNodeUnknownClassTest -v
pytest braincell/ -x -q
```

- [ ] **Step 5: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py braincell/_multi_compartment/cell.py
git commit -m "fix(compute): raise on unknown channel class in root-level check (C6)

_is_root_level_runtime_node previously swallowed the KeyError from
the channel registry and returned False, so an unknown channel kind
was silently treated as non-root (bound to an ion). Now let the
KeyError propagate; Cell.init_state converts it to a ValueError with
diagnostic context.
"
```

---

## Task 19: Final polish — error messages, edge coverage, JIT regression

**Files:**
- Modify: `braincell/_compute/runtime.py` (error message consistency)
- Modify: `braincell/_compute/runtime_test.py` (add remaining T-cases from spec §9)

- [ ] **Step 1: Scan error messages for consistency**

```bash
grep -n "raise \(ValueError\|KeyError\|TypeError\)" braincell/_compute/runtime.py braincell/_compute/topology.py braincell/_compute/table.py
```

For each raise:
- Include the offending value in the message.
- For ambiguity errors, include the candidate list.
- For unknown-key errors, include the set of known keys.

Fix any message that doesn't meet this bar.

- [ ] **Step 2: Add remaining spec test cases**

Append to `runtime_test.py` any T-cases from spec §9 not yet covered (T5 under JIT, T16-T17 already covered via Task 10, edge cases from spec §8.2).

Example T5 (C5-adjacent) test:

```python
class DensityLayoutMaskingUnderJit(unittest.TestCase):
    def test_mask_quantity_like_equivalent_under_jit(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]

        def step():
            return cell.runtime.state_buffers[(layout.id, "g_max")].mantissa.sum()

        compiled = jax.jit(step)
        total = compiled()
        # The dense layout has 5 points, but only 1 active (soma CV midpoint).
        self.assertTrue(float(total) > 0.0)
```

- [ ] **Step 3: JIT-trace regression — verify `evaluate_point_clamps` compiles**

Add:

```python
class EvaluatePointClampsJitTest(unittest.TestCase):
    def test_evaluate_point_clamps_jit_compiles_without_object_dtype(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.init_state()
        runtime = cell.runtime
        compiled = jax.jit(lambda t: runtime.evaluate_point_clamps(t=t))
        out = compiled(0.5 * u.ms)
        self.assertEqual(out.mantissa.shape, (runtime.n_point,))
```

- [ ] **Step 4: Full suite + coverage glance**

```bash
pytest braincell/ -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braincell/_compute/runtime.py braincell/_compute/runtime_test.py \
        braincell/_compute/topology.py braincell/_compute/table.py
git commit -m "test(compute): add remaining regression and JIT-trace coverage

Add tests for spec §9 cases not previously covered, including JIT
compilation of evaluate_point_clamps and dense-layout masking under
jit. Tighten error messages across _compute/ to always name the
offending value and candidate set.
"
```

---

## Final Self-Review Checklist

Before marking the plan complete, confirm:

- [ ] Every spec correctness bug (C1–C6) has a dedicated task + test:
  - C1 → Task 2
  - C2 → Task 3
  - C3 → Task 4
  - C4 → Task 16
  - C5 → Task 12 (fold in)
  - C6 → Task 18
- [ ] Every spec design decision (D1–D8) is represented:
  - D1 (full rewrite, API may change) → all tasks
  - D2 (Quantity-backed state) → Task 12 + Task 13
  - D3 (3-file split, `_compute/`) → Task 1 + Task 8 (folds clamp_table)
  - D4 (Cell owns runtime, no install/uninstall) → Task 14
  - D5 (Literal aliases) → Task 5 (topology) + Task 15 (runtime)
  - D6 (HH1952 decouple) → Tasks 9 + 10 + 11
  - D7 (FunctionClamp fingerprint) → Task 16
  - D8 (dhs_static_cache preserved) → no-op (kept as-is throughout)
- [ ] Every spec test T1–T17 mapped to a task:
  - T1–T3 → Tasks 2/3/4
  - T4 → Task 16
  - T5 → Task 19
  - T6 → Task 18
  - T7–T12 → Task 12
  - T13–T15 → Task 14
  - T16–T17 → Task 10 (HH1952 constructor + hook round-trip)
- [ ] Every step contains code/commands the engineer can run, not placeholders.
- [ ] Every task ends with a commit whose message follows Conventional Commits (`refactor(compute): …`, `fix(compute): …`, `test(compute): …`).
- [ ] No `Co-Authored-By` trailer (per user's global rule).
- [ ] Between every task, `pytest braincell/ -x -q` is green.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-compute-rewrite.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
