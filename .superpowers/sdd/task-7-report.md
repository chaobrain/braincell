# Task 7 report — layering guard + `_compute/__init__.py` cleanup

Branch `worktree-compute-arch02`, worktree
`/mnt/d/codes/projects/braincell/.claude/worktrees/compute-arch02`.

## What landed

- **New** `braincell/_compute/_layering_test.py` (2026 Apache-2.0 header,
  NumPy-style docstrings) — 17 tests.
- **Modified** `braincell/_compute/__init__.py` — deleted the two re-exports and
  the `__all__` (8 lines). Module docstring untouched.

Nothing else changed. `git diff --stat` against the previous commit is one file,
`0 insertions / 8 deletions`, plus the new test file.

---

## Part 1 — the guard's design

### Structure

The file is a small static-analysis library followed by four `TestCase`s. The
library half is deliberately factored into pure functions that take *source
text* rather than file paths, so the walker can be unit-tested on synthetic
sources — that is what makes the "guard that cannot fail" failure mode
impossible to reach silently (see `WalkerTest` below).

| Symbol | Role |
| --- | --- |
| `ImportRecord` | frozen dataclass: `lineno`, `statement` (via `ast.unparse`), `paths` |
| `_is_type_checking_guard(test)` | recognises `TYPE_CHECKING` and `typing.TYPE_CHECKING` |
| `_iter_load_time_statements(body)` | statement walker; the core of the design |
| `_resolve_relative(level, module, package)` | `from . import x` → `braincell._compute.x` |
| `_import_paths(node, package)` | statement → candidate absolute dotted paths |
| `load_time_imports(source, package)` | source text → `tuple[ImportRecord, ...]` |
| `is_upward(path)` | path is `braincell._multi_compartment[.*]` |
| `sibling_of(path, known, package)` | path → sibling module name, or `None` |
| `is_test_module(filename)` | `*_test.py` or `_testing.py` |
| `_production_modules()` / `_scan_package()` / `_sibling_graph()` | package → graph |
| `_find_cycle(graph)` | DFS with an on-path set; returns the closed cycle |

### `ast`, not `importlib` — and why it matters here specifically

The property being asserted is *not observable at runtime*. The original
`_compute` ↔ `_multi_compartment` cycle was importable; it only broke under
particular import orders. An `importlib` probe would have returned "fine"
against exactly the code this guard exists to reject.

This is not a hypothetical. Negative control 1 below injects a real upward
import into `table.py`, and `python -c "import braincell"` **succeeds** — while
the static guard fails loudly. That single pair of outputs is the whole
argument for the design choice.

Secondary benefits: no import side effects (no JAX init, no registry
population, no ordering sensitivity in the test itself), and the whole package
parses in milliseconds.

### The three properties, and how each is encoded

1. **No upward imports** (`NoUpwardImportTest`) — the highest-value assertion.
   Scans every production module for a load-time import whose candidate paths
   include `braincell._multi_compartment` or anything below it. The failure
   message names `<module>.py:<lineno>` and reproduces the statement, then tells
   the contributor the two legitimate remedies (move the dependency down, or
   defer behind `if TYPE_CHECKING:`).

2. **Acyclic** (`AcyclicTest`) — DFS over the intra-package graph; the failure
   message renders the cycle as `bindings -> ions -> layouts -> state ->
   bindings`.

   I added a second assertion here, `test_graph_matches_the_declared_layering`,
   which pins the *entire* expected edge set:

   ```
   __init__   -> {}                     layouts    -> {}
   bridge     -> {}                     ions       -> {layouts}
   scheduling -> {}                     bindings   -> {ions, layouts}
   table      -> {state}                state      -> {bindings, bridge, layouts}
   ```

   Rationale: assertions 1–3 only reject *illegal* edges. A new legal-but-
   unintended edge (say `bridge -> layouts`) would slip through all three while
   quietly eroding the layering. Pinning the graph forces any new intra-package
   edge to be typed out in a diff, where a reviewer sees it. This matches the
   brief's stated graph exactly, with `__init__ -> {}` after Part 2.

3. **Layer order** (`LayerOrderTest`) — encoded as a rank over
   `_LAYER_ORDER = ("layouts", "ions", "bindings", "state")`: a module may
   import any predecessor and no successor. This single rule subsumes all three
   clauses in the brief (layouts ∌ {ions, bindings, state}; ions ∌ {bindings,
   state}; bindings ∌ state) and, unlike three hand-written pair checks,
   automatically covers a fifth layer if one is ever added.
   `test_every_layer_module_exists` guards against the rule silently becoming a
   no-op if a layer module is renamed.

### The two ways this test is easy to get wrong

**(a) `TYPE_CHECKING` false positives.** `_iter_load_time_statements` skips the
`body` of an `if` whose test is `TYPE_CHECKING` (bare `ast.Name`) or
`*.TYPE_CHECKING` (`ast.Attribute`), while still walking its `orelse` — an
`else:` branch does execute at import time. All four deliberate guards are
therefore invisible to the graph:

| Module | Guarded import | Kind |
| --- | --- | --- |
| `layouts.py:70` | `from .state import CellRuntimeState` | intra-package back-edge |
| `ions.py:66` | `from .state import CellRuntimeState` | intra-package back-edge |
| `bindings.py:90` | `from .state import CellRuntimeState` | intra-package back-edge |
| `state.py:100` | `from braincell._multi_compartment.cell import Cell` | upward |

`state.py`'s exemption is pinned explicitly by
`test_the_type_checking_cell_import_in_state_is_not_flagged`, which asserts both
that the import text is still present in the file *and* that the walker does not
report it. Without that, a future over-tightening of the walker (one that
stopped seeing any imports at all) would make assertion 1 pass for entirely the
wrong reason.

**(b) A guard that cannot fail.** Two independent defences:

- *Permanent*: `WalkerTest` — 11 tests over synthetic sources, covering both
  `TYPE_CHECKING` spellings, the `else:` branch, function- and method-body
  imports, a non-`TYPE_CHECKING` top-level `if` (whose body *is* walked), all
  four spellings of an upward import, non-upward imports, `from . import
  bridge`, attribute-vs-submodule disambiguation, and `_find_cycle` on both a
  cycle and a diamond. These fail if the walker degenerates.
- *One-off*: the two injected violations recorded below.

### Additional walker decisions

- **Only `tree.body` plus top-level `if`/`try`.** No bare `ast.walk` — it would
  descend into function and method bodies and flag deferred imports, of which
  the test files have several. `try:` blocks are walked because a module-level
  `try: import x / except ImportError:` *is* a load-time edge.
- **Candidate over-generation.** `from X import a, b` yields `X`, `X.a`, `X.b`,
  because syntax alone cannot say whether `a` is a submodule or an attribute.
  `sibling_of` then filters against the set of module names actually on disk, so
  `from .layouts import MechanismLayout` resolves to `layouts` and the
  non-existent `layouts.MechanismLayout` is discarded. This is what lets one
  code path handle `from . import bridge`, `from .layouts import X`, and
  `from braincell._compute.state import X` (the absolute intra-package form used
  in `bridge.py`) identically.
- **Test modules excluded**, by `is_test_module`. A test importing `Cell` is not
  an architecture violation.

---

## Negative controls — real, unedited output

### Control 0 (discarded): the naive injection breaks collection

My first attempt added `from braincell._multi_compartment.cell import Cell` to
`table.py`. That is a *hard* cycle: pytest cannot even collect the module.

```
ERROR collecting braincell/_compute/_layering_test.py
ImportError while importing test module '.../braincell/_compute/_layering_test.py'.
Traceback:
braincell/__init__.py:53: in <module>
    from ._multi_compartment import (
braincell/_multi_compartment/__init__.py:18: in <module>
    from .cell import Cell, MultiCompartment
braincell/_multi_compartment/cell.py:60: in <module>
    from braincell._compute.table import (
braincell/_compute/table.py:31: in <module>
    from braincell._multi_compartment.cell import Cell
E   ImportError: cannot import name 'Cell' from partially initialized module
    'braincell._multi_compartment.cell' (most likely due to a circular import)
```

This proves nothing about the guard — the interpreter caught it, not the test.
I discarded it and used an *importable* violation instead, which is the faithful
reproduction of the original bug.

### Control 1 — upward import (importable, so only the static guard sees it)

Injected into `braincell/_compute/table.py` after line 30:

```python
from braincell._multi_compartment import probes
```

The interpreter is happy:

```
$ python -c "import braincell; print('import braincell: OK (the violation is importable)')"
import braincell: OK (the violation is importable)
```

The guard is not:

```
$ python -m pytest braincell/_compute/_layering_test.py -q -p no:randomly
F................                                                 [100%]
=================================== FAILURES ===================================
____ NoUpwardImportTest.test_no_production_module_imports_multi_compartment ____

    def test_no_production_module_imports_multi_compartment(self) -> None:
        offences = []
        for name, records in _scan_package().items():
            for record in records:
                if any(is_upward(path) for path in record.paths):
                    offences.append(f"  {name}.py:{record.lineno}: {record.statement}")
>       self.assertEqual(
            [],
            offences,
            ...
        )
E       AssertionError: Lists differ: [] != ['  table.py:31: from braincell._multi_compartment import probes']
E
E       Second list contains 1 additional elements.
E       First extra element 0:
E       '  table.py:31: from braincell._multi_compartment import probes'
E
E       - []
E       + ['  table.py:31: from braincell._multi_compartment import probes'] : braincell._compute must not import braincell._multi_compartment at load time; _compute is the lower layer and such an import re-closes the package cycle the runtime split removed. Move the dependency down, or defer it behind `if TYPE_CHECKING:` if it is only needed for annotations. Offending imports:
E         table.py:31: from braincell._multi_compartment import probes

braincell/_compute/_layering_test.py:403: AssertionError
=========================== short test summary info ============================
FAILED braincell/_compute/_layering_test.py::NoUpwardImportTest::test_no_production_module_imports_multi_compartment
1 failed, 16 passed, 7 subtests passed in 5.17s
```

**This is the central result of the task**: a violation that `import braincell`
accepts, and that the guard rejects by file and line.

### Control 2 — layering violation

Injected into `braincell/_compute/layouts.py` after line 68 (`from . import
state`, chosen over `from .state import CellRuntimeState` for the same reason as
above — the name-import form dies at collection, the module-import form is
importable via CPython's `sys.modules` fallback, so the guard is what fails):

```
$ python -c "import braincell; print('import braincell: OK (the violation is importable)')"
import braincell: OK (the violation is importable)
```

Three tests fail — the acyclic guard, the pinned graph, and the layer-order
guard:

```
$ python -m pytest braincell/_compute/_layering_test.py -q -p no:randomly
..FF.F...........                                                 [100%]
=================================== FAILURES ===================================
_____________ AcyclicTest.test_graph_matches_the_declared_layering _____________
E       AssertionError: {...} != {...}
E         {'__init__': set(),
E          'bindings': {'layouts', 'ions'},
E          'bridge': set(),
E          'ions': {'layouts'},
E       -  'layouts': set(),
E       ?               ^^^
E
E       +  'layouts': {'state'},
E       ?             ++ +++ ^^
E
E          'scheduling': set(),
E          'state': {'layouts', 'bindings', 'bridge'},
E          'table': {'state'}}

____________ AcyclicTest.test_intra_package_import_graph_is_acyclic ____________
E       AssertionError: ('bindings', 'ions', 'layouts', 'state', 'bindings') is not None :
        braincell._compute modules must form an acyclic import graph, but found:
        bindings -> ions -> layouts -> state -> bindings. Break the cycle by moving
        the shared code into the lower module, or defer one edge behind `if TYPE_CHECKING:`.

_____________ LayerOrderTest.test_no_module_imports_a_later_layer ______________
E       AssertionError: Lists differ: [] != ['  layouts.py:69 imports state: from . import state']
E
E       Second list contains 1 additional elements.
E       First extra element 0:
E       '  layouts.py:69 imports state: from . import state'
E
E       - []
E       + ['  layouts.py:69 imports state: from . import state'] : braincell._compute layers must be imported in the order layouts -> ions -> bindings -> state; a module may import its predecessors and never its successors. Offending imports:
E         layouts.py:69 imports state: from . import state
```

### Both experiments reverted

Both files were restored from byte-for-byte backups taken before injection.
After `ruff format` was applied to the new test file, control 1 was re-run once
more to confirm the reformatted guard still fires (`1 failed, 16 passed`) and
then reverted again. Final state:

```
$ git status --short braincell/
 M braincell/_compute/__init__.py
?? braincell/_compute/_layering_test.py

$ git diff --stat -- braincell/
 braincell/_compute/__init__.py | 8 --------
 1 file changed, 8 deletions(-)
```

No trace of either experiment remains.

---

## Part 2 — `_compute/__init__.py`

### Zero consumers, verified independently

Repo-wide grep for every import form that could reach these names:

```
$ grep -rn "from braincell._compute import\|from braincell import _compute\|import braincell._compute" \
      --include=*.py --include=*.rst --include=*.md --include=*.ipynb .
braincell/_compute/bridge_test.py:31:               from braincell._compute import bridge
braincell/_multi_compartment/cell.py:102:           from braincell._compute import bridge
braincell/_multi_compartment/currents.py:36:        from braincell._compute import bridge
braincell/_multi_compartment/currents_test.py:85:   from braincell._compute import bridge
braincell/_multi_compartment/probes.py:36:          from braincell._compute import bridge
```

(plus matches inside `.superpowers/sdd/*.md`, which are this refactor's own
briefs and reports, not code.) Every single one imports the `bridge`
**submodule**. Not one imports `NodeScheduling` or `NodeTree` from `_compute`.

Cross-checked by symbol name as well:

- `NodeScheduling` — defined and `__all__`-exported in
  `braincell/_compute/scheduling.py`; the only other occurrences are within that
  same module. No importer outside it.
- `NodeTree` — its real home is `braincell._discretization.base`. Consumers
  (`braincell/vis/point_topology.py:32`, `_compute/layouts.py:52`,
  `_compute/scheduling.py:22`, `_compute/state.py:66`) all import it from
  `braincell._discretization.base` directly. Re-exporting it from `_compute` was
  a pure duplicate.

Confirmed: zero consumers. Nothing needed repointing.

### The edit

Deleted lines 35–41 in full (both imports and the `__all__`). **The module
docstring, including the module inventory at lines 16–33, is untouched** — the
diff is `0 insertions, 8 deletions`, all below the docstring.

Side benefit: `__init__` now has no intra-package edge at all, matching the
brief's target graph and making the package initialiser genuinely inert.

---

## Verification — real, unedited output

**1. `_compute` suite** — baseline 141, **+17 added → 158**:

```
$ python -m pytest braincell/_compute/ -q
158 passed
```

**N = 17.** Breakdown: `NoUpwardImportTest` 2, `AcyclicTest` 2,
`LayerOrderTest` 2, `WalkerTest` 11.

**2. Full suite** — baseline 2240 passed / 19 skipped, **+17 → 2257 / 19**:

```
$ python -m pytest braincell/ -q
2257 passed, 19 skipped
```

**3. Import cleanliness:**

```
$ python -c "import braincell; print('import braincell OK')"
import braincell OK
```

(the `An NVIDIA GPU may be present ... falling back to cpu` line is jaxlib's
standard banner on this machine, present on `main` too.)

**4. Negative controls** — above.

**5. ruff** (`/home/chaoming/.cache/pre-commit/repotpyvnr8t/py_env-python3/bin/ruff`):

```
$ ruff check braincell/_compute/
All checks passed!

$ ruff format --check braincell/_compute/
18 files already formatted
```

`ruff format` initially wanted four reflows in the new file (line-length 120
joins); I applied `ruff format` and re-ran both the formatter check and the
suite. Two blank lines separate every top-level definition.

---

## Conventions checklist

- Apache-2.0 header, year **2026**, at the very top of the new file.
- NumPy-style docstrings on the module and on every public function, dataclass
  and `TestCase`.
- `unittest.TestCase` under pytest, matching the rest of the repo.
- `_compute` siblings import each other relatively — unchanged by this task;
  the guard now enforces the *direction* of those imports.
- Precedent for the underscore-prefixed architectural test name:
  `braincell/channel/_docstring_test.py`, `_deprecation_test.py`,
  `braincell/ion/_docstring_test.py`. `_layering_test.py` matches `*_test.py`
  and is collected (confirmed: the `_compute` count rose by exactly 17).
- One commit, imperative message, no `Co-Authored-By` trailer.

---

## Concerns / notes for the next task

- `_layering_test.py` hard-codes the expected edge set in
  `test_graph_matches_the_declared_layering`. This is intentional (it surfaces
  new edges in review) but means **any** future intra-package import — legal or
  not — requires a one-line update to that dict. Task 8 should be aware if it
  touches `_compute` imports.
- The pinned graph also encodes `__init__ -> {}`. If a later task re-adds a
  re-export to `_compute/__init__.py`, that test will fail by design.

---

## Post-review fixes

A review of `_layering_test.py` found that `_iter_load_time_statements`
recursed into `ast.If` and `ast.Try` bodies but not into `ast.ClassDef`
bodies, even though a class body executes at import time exactly like an
`if`/`try` body does. The reviewer demonstrated a real evasion — an import
written inside a class body (`class _Upward: from
braincell._multi_compartment import probes`) went completely undetected, and
the guard still reported all tests passing. Compounding the hole, both the
module docstring ("What counts as an edge") and the `_iter_load_time_statements`
docstring stated that class-body imports are "deferred", which is factually
wrong and would have led a future maintainer to think the gap was
intentional.

### The fix

1. `_iter_load_time_statements` (`braincell/_compute/_layering_test.py`) now
   recurses into `ast.ClassDef.body`, in addition to `ast.If`/`ast.Try`.
2. While fixing the class-body hole, I also considered module-level `with`,
   `for`, and `while` bodies, which share the identical root cause (their
   bodies run at import time; only function/method bodies defer). I chose to
   descend into all three (`ast.With`/`ast.AsyncWith`, `ast.For`/`ast.AsyncFor`,
   `ast.While`) rather than leave them as a second, now-documented gap right
   next to the one just closed — the walker already had the recursive
   machinery, so the additional branches are three `elif` clauses, not new
   design. `for`/`while` also recurse into `node.orelse`, matching the
   language semantics (`for`/`else` and `while`/`else` bodies both execute at
   module load time under normal, no-`break` control flow).
3. Corrected both docstrings (module-level "What counts as an edge" section
   and the `_iter_load_time_statements` docstring) to state accurately that:
   function/method bodies are deferred and excluded; `class`, `with`, `for`,
   and `while` bodies execute at import time and are included; `if
   TYPE_CHECKING:` bodies are excluded because they never execute at runtime.

No assertion, the pinned edge-set dict, or any other part of the file was
touched — this was scoped entirely to `_iter_load_time_statements` and its
two docstrings.

### Verification — real, unedited output

**1. Negative control — the class-body evasion that motivated this fix.**
Injected into `braincell/_compute/table.py` immediately after the existing
`from .state import CellRuntimeState` line:

```python
class _Upward:
    from braincell._multi_compartment import probes
```

```
$ rtk proxy python -m pytest braincell/_compute/_layering_test.py -q
F................                                                 [100%]
=================================== FAILURES ===================================
____ NoUpwardImportTest.test_no_production_module_imports_multi_compartment ____
braincell/_compute/_layering_test.py:417: in test_no_production_module_imports_multi_compartment
    self.assertEqual(
E   AssertionError: Lists differ: [] != ['  table.py:34: from braincell._multi_compartment import probes']
E   
E   Second list contains 1 additional elements.
E   First extra element 0:
E   '  table.py:34: from braincell._multi_compartment import probes'
E   
E   - []
E   + ['  table.py:34: from braincell._multi_compartment import probes'] : braincell._compute must not import braincell._multi_compartment at load time; _compute is the lower layer and such an import re-closes the package cycle the runtime split removed. Move the dependency down, or defer it behind `if TYPE_CHECKING:` if it is only needed for annotations. Offending imports:
E     table.py:34: from braincell._multi_compartment import probes
1 failed, 16 passed, 7 subtests passed in 5.86s
```

The guard now names `table.py` and the exact evading import, and fails as
required. `table.py` was then restored byte-for-byte from a pre-injection
backup:

```
$ git status --short braincell/_compute/table.py
$ git diff --stat -- braincell/_compute/table.py
```

Both empty — no trace of the injection remained.

**2. Clean tree, the layering test alone:**

```
$ rtk proxy python -m pytest braincell/_compute/_layering_test.py -q
.................                                                 [100%]
17 passed, 7 subtests passed in 5.00s
```

Same 17 tests as before the fix — the walker changed, no assertions were
added.

**3. `_compute` package:**

```
$ rtk proxy python -m pytest braincell/_compute/ -q
................................................................. [ 41%]
........................................................................ [ 86%]
.....................                                                    [100%]
158 passed, 7 subtests passed in 30.21s
```

**4. Full suite:**

```
$ rtk proxy python -m pytest braincell/ -q
...
2257 passed, 19 skipped, 52 warnings, 296 subtests passed in 499.30s (0:08:19)
```

**5. False-positive direction — the five genuine `if TYPE_CHECKING:` imports
stay clean.** Scanned `layouts.py`, `ions.py`, `bindings.py`, `state.py`, and
`bridge.py` directly through `_scan_package()`/`is_upward()` (the same
functions the guard's tests use) and confirmed none of the `TYPE_CHECKING`-
guarded imports appear in the load-time record list at all, let alone flagged
upward — the walker still skips `if TYPE_CHECKING:` bodies exactly as before:

```
layouts flagged upward: []
ions flagged upward: []
bindings flagged upward: []
state flagged upward: []
bridge flagged upward: []
```

(`bridge.py`'s `TYPE_CHECKING` import is the one at line 32 referenced in the
task; the guarded imports for all five modules are simply absent from each
module's load-time-statement list, confirming the `if TYPE_CHECKING:` skip is
unaffected by the `ClassDef`/`With`/`For`/`While` additions.)

**6. ruff** (`/home/chaoming/.cache/pre-commit/repotpyvnr8t/py_env-python3/bin/ruff`):

```
$ rtk proxy ruff check braincell/_compute/
All checks passed!

$ rtk proxy ruff format --check braincell/_compute/
18 files already formatted
```

**7. Working tree clean of experiments:**

```
$ git status --short
 M braincell/_compute/_layering_test.py
?? .superpowers/sdd/...   (pre-existing untracked scratch, not touched by this fix)
```

Only `_layering_test.py` is modified; `git diff` shows exactly the walker
change and the two docstring corrections, nothing else.
