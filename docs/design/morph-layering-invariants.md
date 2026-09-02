# Morph layering invariants

`braincell.morph` owns `Branch` and `Morphology`, the two objects every
other package builds on. It therefore sits at the bottom of the import
order — and yet twelve of its methods import *upward*, into
`braincell.io`, `braincell.vis`, `braincell.filter`, and the `braincell`
root package itself.

Every one of those twelve is a **function-body import**. That is not a
style choice and not leftover scaffolding: hoisting any of them to module
scope turns `import braincell` into a hard `ImportError`. This document
records why the edges exist, why they are deferred, and what the guard in
`braincell/morph/__init___test.py` pins.

## The load order

`braincell/__init__.py` builds the public surface in a fixed sequence.
`morph` is imported early, because `filter`, `io`, `vis`,
`_discretization`, and `_multi_compartment` all need `Branch` and
`Morphology` at *their* module scope. By the time `morph/branch.py` and
`morph/morphology.py` execute:

- `braincell` itself is in `sys.modules` but has bound almost nothing,
- `braincell.filter`, `braincell.io`, and `braincell.vis` have not been
  imported at all.

So at `morph` load time there is nothing upward to import. The only
`braincell` name `morph` may touch at module scope is
`braincell._misc`, which is a leaf.

## The twelve upward edges

| Site | Target | Method |
| --- | --- | --- |
| `branch.py:883` | `braincell` (root) | `Branch.vis2d` |
| `branch.py:953` | `braincell` (root) | `Branch.vis3d` |
| `branch.py:1017` | `braincell.io.checkpoint` | `Branch.save` |
| `branch.py:1058` | `braincell.io.checkpoint` | `Branch.load` |
| `morphology.py:369` | `braincell.io.swc` | `Morphology.from_swc` |
| `morphology.py:416` | `braincell.io.asc` | `Morphology.from_asc` |
| `morphology.py:483` | `braincell.io.neuromorpho` | `Morphology.from_neuromorpho` |
| `morphology.py:528` | `braincell.io.checkpoint` | `Morphology.save` |
| `morphology.py:565` | `braincell.io.checkpoint` | `Morphology.load` |
| `morphology.py:1242` | `braincell.vis.plot3d` | `Morphology.vis3d` |
| `morphology.py:1392` | `braincell.vis.plot2d` | `Morphology.vis2d` |
| `morphology.py:1453` | `braincell.filter` | `Morphology.filter` |

They fall into three groups, and each group is deliberate:

- **Readers and checkpoints** (`from_swc`, `from_asc`,
  `from_neuromorpho`, `save`, `load`). These are the *primary documented
  entry points* for building a morphology. A user who has a `.swc` file
  reaches for `Morphology.from_swc`, not for `braincell.io.SwcReader`.
  The deferred import is the correct implementation of that API, not a
  workaround to be cleaned up.
- **Visualization** (`vis2d`, `vis3d` on both classes). PR #144 removed
  the equivalent `Cell.vis_*` methods in favour of
  `braincell.vis.plot_cell_topology`, so the obvious symmetry argument is
  to remove these too and delete four edges. That was considered and
  declined: `examples/multi_compartment/vis.ipynb` documents them as an
  intentional "Convenience subset" in a comparison table against
  `braincell.vis`, and 86 references across five notebooks and the docs
  depend on them. `Cell` is a simulation object whose visualization is a
  side quest; `Morphology` is a geometry object whose visualization is a
  core use.
- **Selection** (`Morphology.filter`). Returns a `LocsetExpr` /
  `RegionExpr`, which is `braincell.filter`'s vocabulary. The alternative
  is to move the expression types into `morph`, which would make `morph`
  own a query language.

## Why hoisting breaks

Each of the five distinct targets was tested by moving the import to
module scope and running `import braincell` in a fresh interpreter. All
five fail, and the failure is not always where you would guess:

| Hoisted | Failure |
| --- | --- |
| `from braincell.io.checkpoint import save_branch` (in `branch.py`) | `ImportError: cannot import name 'Branch' from partially initialized module 'braincell.morph.branch'` |
| `from braincell.io.swc import SwcReadOptions, SwcReader` | `ImportError: cannot import name 'Branch' from partially initialized module 'braincell.morph.morphology'` |
| `from braincell.vis.plot3d import plot3d` | `ImportError: cannot import name 'RegionMask' from partially initialized module 'braincell.filter'` |
| `from braincell.filter import LocsetExpr, RegionExpr` | `ImportError: cannot import name 'LocsetExpr' from partially initialized module 'braincell.filter'` |
| `from braincell import Morphology` (in `branch.py`) | `ImportError: cannot import name 'Morphology' from 'braincell'` |

The `vis.plot3d` row is the instructive one: hoisting a *vis* import
reports a *filter* failure, because `vis` imports `filter` on the way up
and the cycle closes two packages away from the edit. A future editor who
hoists one of these will be reading a traceback that does not mention the
line they changed. That is the whole reason for the guard.

## The guard

`braincell/morph/__init___test.py` holds two AST checks over every
non-test module in the package:

- `UpwardImportsAreDeferredTest` — no module-scope import may name
  `braincell.io`, `braincell.vis`, `braincell.filter`, or the `braincell`
  root. This is the invariant; violating it is import-time fatal.
- `LoadTimeImportsTest` — pins the full set of module-scope `braincell`
  imports, so a *new* upward dependency has to be declared here rather
  than appearing silently.

`braincell/network/__init___test.py::PartialParentTest` guards the same
class of hazard for `braincell.network`, which is imported from the
middle of `braincell._multi_compartment`'s initialization. The two
packages are the only ones in the repository whose import order is load
bearing in this way.
