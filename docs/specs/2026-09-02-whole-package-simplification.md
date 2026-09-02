# Whole-package simplification sweep

Iteration 14, the last of the module-by-module `/simplify` sweep. The
previous thirteen each took one package; this one takes what only shows up
when you look at all of them at once: duplication that spans packages, the
shape of the import graph, and the coherence of the public surface.

Baseline on `21bf528`:

- `pytest braincell/ -q` → 2854 passed, 20 skipped, 417 subtests passed
  (182.29 s)
- 78 223 source lines across 165 files, plus 41 226 lines of co-located
  tests

| Package | src LOC | test LOC |
| --- | --- | --- |
| `channel` | 17 145 | 6 868 |
| `vis` | 11 339 | 6 804 |
| `io` | 7 322 | 4 868 |
| `_multi_compartment` | 6 364 | 2 586 |
| `ion` | 6 242 | 2 950 |
| `network` | 5 403 | 2 271 |
| `_compute` | 4 431 | 4 475 |
| `quad` | 4 371 | 1 754 |
| `morph` | 3 844 | 1 662 |
| `_discretization` | 3 331 | 1 753 |
| `filter` | 2 922 | 1 794 |
| `mech` | 2 503 | 1 452 |
| root modules | 2 494 | 998 |
| `_single_compartment` | 346 | 797 |
| `synapse` | 166 | 194 |

## What the scans found

Three mechanical scans, kept in `dev/` (gitignored), each run over the
whole package.

### Cross-package duplication is nearly gone

`dev/scan_dupes.py` normalizes every function body to its AST shape —
names, attributes, and literals erased — and groups the matches, reporting
only groups that span more than one top-level package. At a threshold of
two statements it finds **six groups in 78 000 lines**. Thirteen
iterations of intra-package deduplication left very little that crosses a
boundary.

Of the six, three are worth acting on and three are coincidence or too
small to pay for a helper. They are listed under Themes and Declined.

### `morph` is a leaf that reaches upward twelve times

`dev/scan_layering.py` builds the inter-package import graph. Six package
pairs import each other, and **`morph` is one half of four of them**:

| Pair | Why |
| --- | --- |
| `_compute` ↔ `_multi_compartment` | documented, deliberate |
| `_multi_compartment` ↔ `network` | documented in `docs/design/network/module-layout.md` |
| `<root>` ↔ `morph` | `Branch.vis2d` / `vis3d` build a `Morphology` |
| `filter` ↔ `morph` | `Morphology.filter` builds a `LocsetExpr` |
| `io` ↔ `morph` | `Morphology.from_swc` / `from_asc` / `from_neuromorpho` / checkpoints |
| `morph` ↔ `vis` | `Morphology.vis2d` / `vis3d` |

Every one of `morph`'s twelve upward edges is a **function-body import**,
placed there to break the cycle the method creates:

```
morph/branch.py:883      from braincell import Morphology
morph/branch.py:953      from braincell import Morphology
morph/branch.py:1017     from braincell.io.checkpoint import save_branch
morph/branch.py:1058     from braincell.io.checkpoint import load_branch
morph/morphology.py:369  from braincell.io.swc import SwcReadOptions, SwcReader
morph/morphology.py:416  from braincell.io.asc import AscReader
morph/morphology.py:483  from braincell.io.neuromorpho import load_neuromorpho
morph/morphology.py:528  from braincell.io.checkpoint import save_morpho
morph/morphology.py:565  from braincell.io.checkpoint import load_morpho
morph/morphology.py:1242 from braincell.vis.plot3d import plot3d
morph/morphology.py:1392 from braincell.vis.plot2d import plot2d
morph/morphology.py:1453 from braincell.filter import LocsetExpr, RegionExpr
```

These are load-bearing, not vestigial. Each of the five distinct targets
was tested by moving the import to module scope and running
`import braincell` in a fresh interpreter; all five fail, and the failure
is not always local:

| Hoisted | Failure |
| --- | --- |
| `io.checkpoint` in `branch.py` | `cannot import name 'Branch' from partially initialized module 'braincell.morph.branch'` |
| `io.swc` in `morphology.py` | `cannot import name 'Branch' from partially initialized module 'braincell.morph.morphology'` |
| `vis.plot3d` | `cannot import name 'RegionMask' from partially initialized module 'braincell.filter'` |
| `filter` | `cannot import name 'LocsetExpr' from partially initialized module 'braincell.filter'` |
| `braincell` root in `branch.py` | `cannot import name 'Morphology' from 'braincell'` |

The `vis.plot3d` row is the instructive one: hoisting a *vis* import
reports a *filter* failure, because `vis` imports `filter` on the way up
and the cycle closes two packages away from the edit. Nothing says so,
and nothing tests it. `network` has exactly this hazard and *does* guard
it, with the AST scan in
`network/__init___test.py::PartialParentTest`. `morph` gets the same
treatment.

### Seven of eleven `__all__` lists are unsorted, and six packages have no guard at all

`dev/scan_surface.py`:

| Package | names | sorted | `ReExportTests` guard |
| --- | --- | --- | --- |
| `braincell` | 76 | yes | yes |
| `channel` | 123 | **no** | yes |
| `filter` | 36 | **no** | **no** |
| `io` | 35 | **no** | **no** |
| `ion` | 28 | **no** | yes |
| `mech` | 30 | **no** | **no** |
| `morph` | 4 | **no** | **no** |
| `network` | 4 | yes | yes |
| `quad` | 31 | **no** | **no** |
| `synapse` | 2 | yes | **no** |
| `vis` | 29 | yes | yes |

Iteration 13 built `_testing.ReExportTests` and deliberately left
`require_sorted_all` opt-in, recording that flipping it "is a package-wide
convention change, not a root-module one" and that iteration 14 owns it.
This is that iteration.

## Themes

### One helper where two packages had the same body

**`_require_name`** is byte-identical in
`_multi_compartment/density_views.py:337` and
`network/recording.py:623`. `network/connection.py:54` already imports the
`recording` copy, so the name is already being shared — just not across the
package boundary. It moves to `_misc.py`.

**`_normalize_run_traces`** (`_multi_compartment/run.py:248`) and
**`_normalize_scan_samples`** (`network/engine.py:863`) differ only in a
parameter name, a message prefix, and a noun. Both wrap a `for_loop` /
`scan` output that collapses to a bare array when exactly one thing is
collected, and both raise on a count mismatch. This is precisely the shape
`_misc.validate_time_quantity` already handles — shared body, caller-supplied
`prefix` and `name` — so it becomes `_misc.normalize_loop_outputs`.

**One soma-plus-dendrite tree, built three times.**
`make_two_branch_morpho` (`_discretization/_testing.py:61`) and
`make_soma_dend_tree` (`vis/_testing.py:91`) have byte-identical bodies;
the `vis` docstring even reasons about being "deliberately distinct from
`make_length_only_tree`" without noticing it is identical to the
`_discretization` one. The scan did not connect the third,
`filter/_testing.py:53`, because that one is parameterized and uses
`tree.attach(...)` rather than `tree.soma.dend = ...`.

Building all three and comparing name, index, type, lengths, radii,
parent, `parent_x`, and `child_x` per branch shows the `filter` builder at
`dend_length=100.0` produces exactly the tree the other two hard-code —
`attach(parent="soma", child_name="dend", parent_x=1.0)` and
`tree.soma.dend = ...` are the same operation. So the parameterized one
is the general case, and it moves to `morph/_testing.py`, which all three
packages already depend on, following the precedent AGENTS.md records for
`io/_testing.FIXTURE_DIR`.

`filter` re-exports it unchanged. `vis` and `_discretization` keep their
zero-argument names as one-line delegations that pin `dend_length=100.0`,
because that is how ~110 call sites read and because the pinned length is
load-bearing there (it is what keeps the CV count small enough to assert
coverage fractions by hand). `_compute/_testing.py:_build_tree` already
establishes that pattern for exactly this reason.

### One convention, applied where it carries information

Nine of the eleven package `__all__` lists become ASCII-sorted, and every
public package gets a `ReExportTests` subclass. `channel` and `ion` build
`__all__` by concatenating their submodules', so they are wrapped in
`sorted(...)` rather than hand-ordered.

`mech` and `quad` are the two exceptions, and they are deliberate: both
group `__all__` by category under comment headings, and `quad`'s
Runge–Kutta block is ordered by accuracy rather than by name. Sorting
them would destroy information a reader uses. They get the guard with
`require_sorted_all = False` and a comment saying why, so membership and
uniqueness are still pinned.

### One invariant, written down and tested

`docs/design/morph-layering-invariants.md` records why `morph` sits below
`io`, `vis`, and `filter` in the import order, why its twelve upward edges
are deferred, and the measured failure for each one.
`morph/__init___test.py` pins it with two AST checks in the idiom of
`network/__init___test.py::PartialParentTest`:
`UpwardImportsAreDeferredTest` (no module-scope import may name
`braincell.io`, `.vis`, `.filter`, or the root) and `LoadTimeImportsTest`
(the full load-time `braincell` import set is declared, so a new edge has
to be written down).

The guard's detection logic was exercised against synthetic sources
rather than assumed: it flags the hoisted import in all seven placements
that would be fatal — module scope, class body, and `try:` body, for each
of the four upward targets — and passes the four legitimate spellings
(function body, `if TYPE_CHECKING:`, relative sibling, `braincell._misc`).
It could not be demonstrated by editing the real file, because once the
import is hoisted `import braincell` dies during pytest collection.

## Breaking changes

None intended. Sorting `__all__` is order-only and `import *` is
order-insensitive; the moved helpers are all private (leading underscore)
and every in-repo caller is updated here.

## Declined, with the check that overturned it

- **`braincell.Channel` / `Ion` / `Synapse` collide with
  `braincell.mech.Channel` / `Ion` / `Synapse`.** Confirmed they are
  genuinely different classes — the runtime base you subclass versus the
  declaration you hand to `paint()`. But this is a decision the project
  already made and wrote down: PR #143's body states that the two
  "remain different classes in disjoint namespaces" and that
  `mech.Synapse` "is deliberately not re-exported at top level". Verified
  that none of the three is re-exported at root, and that the four names
  `mech` *does* share with the root (`CableProperty`, `CurrentClamp`,
  `FunctionClamp`, `SineClamp`) are the same objects. Not re-litigated.
- **`get_registry` means two different functions in `mech` and `quad`.**
  Same situation: disjoint namespaces, neither re-exported at root.
- **`PointPlacement` looked like a dead root export** — nothing in `docs/`
  or `examples/` names it. It is the documented return type of the public
  `Cell.point_placements` property (`_multi_compartment/cell.py:1317`), so
  a caller needs the name to test against. Kept.
- **`local_position` in `filter/_sampling.py:61` and
  `mech/_context.py:87`.** Three lines each, and the message names the
  owning class. A shared helper would take a class name as an argument to
  produce a message two lines of code already produce.
- **`_stack_values` in `network/connection.py:837`,
  `network/recording.py:576`, and the same shape inside
  `_multi_compartment/density_views.py`.** Three variants, not three
  copies: axis 0 versus axis −1, `np.asarray` versus `u.math.stack`, and
  correspondingly `ndarray float64` versus `ArrayImpl float32`.
  Iteration 12 measured this and declined it; the third variant does not
  change the answer.
- **Extending `DocstringConformanceTests` past `channel` and `ion`.** It
  requires a `References` section with a `.. [n]` citation on every public
  symbol. That is a documentation project, not a simplification, and its
  `_NO_PRIMARY_SOURCE` allowlist would swallow most of the surface on the
  way.
- **Removing `Branch.vis2d` / `vis3d` and `Morphology.vis2d` / `vis3d`,**
  which is what PR #144 did to `Cell.vis_*` and which would delete four of
  `morph`'s ten upward edges. 86 references across five notebooks and the
  docs, and `examples/multi_compartment/vis.ipynb:32` documents them as an
  intentional "Convenience subset" in a comparison table against
  `braincell.vis`. Removing them is a large, hostile break for a layering
  nicety. The edges are documented and guarded instead.
- **The `io` / `vis` / `filter` entry points on `Morphology`**
  (`from_swc`, `from_asc`, `from_neuromorpho`, checkpoints, `filter`).
  These are the primary documented entry points; the deferred import is
  the correct implementation, not a workaround to clean up.
