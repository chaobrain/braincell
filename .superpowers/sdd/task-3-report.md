# Task 3 report — Extract `ions.py` from `_compute/runtime.py`

Branch `worktree-compute-arch02`, worktree
`/mnt/d/codes/projects/braincell/.claude/worktrees/compute-arch02`.
Base commit: `e98f2f4` (*Extract layouts.py from _compute/runtime.py*).

## What moved

All 14 symbols named in the brief, in source order, from
`braincell/_compute/runtime.py` into `braincell/_compute/ions.py`. Every span was
lifted by AST (`node.lineno` .. `node.end_lineno`, extended over decorators)
rather than by hand, so no body was retyped.

| Symbol | Old line in `runtime.py` |
|---|---|
| `_build_runtime_ions` | 777–833 |
| `_build_default_ions` | 836–839 |
| `_collect_runtime_ion_instances` | 842–889 |
| `_build_ion_alias_map` | 892–918 |
| `_runtime_ion_species_key` | 921–930 |
| `_runtime_ion_family` | 933–942 |
| `_supported_ion_runtime_params` | 945–957 |
| `_ion_runtime_attr_name` | 960–963 |
| `_normalize_ion_runtime_param_value` | 966–976 |
| `_instantiate_runtime_ion_instance` | 979–1036 |
| `_restore_shaped_species_initializers` | 1039–1051 |
| `_ion_param_broadcast` | 1054–1085 |
| `_ion_param_scatter` | 1088–1169 |
| `_sync_runtime_ion` | 2114–2160 |

The start lines match the brief's table exactly, which confirms the brief's
numbers were current as of `HEAD`.

`_restore_shaped_species_initializers` moved even though the old shim's import
list named only the other 13 — its sole caller,
`_instantiate_runtime_ion_instance`, is now in `ions.py`, so leaving it behind
would have created an upward `ions -> runtime` edge. It is included in
`runtime.py`'s new back-import block, so `runtime.py`'s attribute surface grew
by one name rather than losing one.

`runtime.py`: 2218 -> 1774 lines. `ions.py`: 36 -> 494 lines.

## The `_constant_quantity_value` question — settled

The brief asked me to verify rather than assume. **It really is called.**
`_normalize_ion_runtime_param_value` ended with, verbatim (old `runtime.py`
lines 973–976):

```python
    constant_quantity = _constant_quantity_value(value)
    if constant_quantity is not None:
        return constant_quantity
    return value
```

So `from .layouts import ... _constant_quantity_value` is a real call site in
`ions.py`, not a speculative import. It is the second of the two downward edges
into the leaf module (the other being `MechanismLayout`, used as a parameter
annotation by `_build_runtime_ions`, `_collect_runtime_ion_instances`, and
`_instantiate_runtime_ion_instance`).

## Imports added to `ions.py`

Derived by walking the moved bodies' free names with `ast`, not by copying
`runtime.py`'s block:

```python
import inspect
from typing import TYPE_CHECKING

import brainunit as u
import numpy as np

from braincell import ion as runtime_ion
from braincell.ion import build_placeholder_ions
from braincell.ion._base import DynamicNernstIon, FixedIon, InitNernstIon, KineticIon
from braincell.mech import Density, get_registry
from .layouts import MechanismLayout, _constant_quantity_value

if TYPE_CHECKING:
    from .state import CellRuntimeState
```

Who needs what:

- `inspect` — `_supported_ion_runtime_params` only.
- `u` (`brainunit`) — `_normalize_ion_runtime_param_value`,
  `_restore_shaped_species_initializers`, `_ion_param_broadcast`,
  `_ion_param_scatter`.
- `np` — the broadcast/scatter pair, plus `_build_runtime_ions`'s annotation.
- `runtime_ion` — `_runtime_ion_species_key` (`runtime_ion.Sodium`, `.Potassium`,
  `.Calcium`, `.NonSpecific`).
- `build_placeholder_ions` — the two real call sites inside
  `_build_default_ions`. Not a pass-through re-export; it moved and still works.
- `DynamicNernstIon` — `_ion_runtime_attr_name`,
  `_normalize_ion_runtime_param_value`. `KineticIon`, `InitNernstIon`,
  `FixedIon` — `_runtime_ion_family`.
- `Density` — `_collect_runtime_ion_instances`, `_sync_runtime_ion`, and
  `_instantiate_runtime_ion_instance`'s annotation.
- `get_registry` — `_collect_runtime_ion_instances`.
- `MechanismLayout`, `_constant_quantity_value` — see above.

`from __future__ import annotations` is the first statement after the module
docstring, and the `CellRuntimeState` reference from `_sync_runtime_ion`'s
signature is annotation-only, resolved through `TYPE_CHECKING` pointed at
`.state` (not `.runtime`), so the line never needs revisiting when `runtime.py`
is deleted in task 5. This mirrors `layouts.py` exactly.

No `__all__` was added: all 14 names are underscore-prefixed, so the module has
no public surface to declare. (`layouts.py` declares an `__all__` because it
genuinely exports seven public names.)

## Imports changed in `runtime.py`

**Deleted, because nothing there uses them any more:**

- `import inspect` — its only user in the whole module was
  `_supported_ion_runtime_params`. Confirmed no remaining textual or AST
  reference in the stripped file.
- `DynamicNernstIon`, `FixedIon`, `InitNernstIon` from
  `braincell.ion._base`. The statement was trimmed to
  `from braincell.ion._base import KineticIon`; `KineticIon` survives because
  `_configure_runtime_subsolvers` still does
  `isinstance(node, (Markov, KineticIon))`. I grepped the rest of the package
  for anyone importing those three names *from `_compute.runtime`* and found
  no one — the only other references live under `braincell/ion/` itself.

**Deliberately kept although now formally unused inside the module:**

- `_constant_quantity_value` in the `from .layouts import (...)  # noqa: F401`
  block. It flipped from used to unused as a direct result of this task, but
  that block exists to preserve `runtime.py`'s attribute surface for the two
  remaining extractions, and the brief is explicit that this task deletes
  nothing. Removing it would shrink the surface and would have shown up as a
  `lost` symbol in verification step 4.
- `build_placeholder_ions`, which is listed in `runtime.py`'s `__all__`. Its one
  real call site left with `_build_default_ions`, but it is a declared re-export,
  so it stays.

**Added:** a `from .ions import (...)  # noqa: F401` block naming all 14 symbols,
placed between the `.bridge` and `.layouts` blocks, following task 2's
precedent. Two of the 14 (`_build_runtime_ions`, `_sync_runtime_ion`) are still
genuinely called from `runtime.py`; the other twelve are pure surface
preservation for `runtime_test.py`, which imports by module path. The whole
block disappears when `runtime.py` does.

## Docstrings

`ions.py`'s docstring no longer claims to be a re-export shim or to be "part of
the ARCH-02 logical partition". It now describes the four things the module
actually holds — runtime ion instantiation, parameter normalization,
broadcast/scatter buffer algebra, and per-layout synchronization — and states
the dependency rule (down into `layouts`, out to public `braincell.ion` /
`braincell.mech`, nothing from `runtime`).

`runtime.py`'s docstring dropped "and the ion-construction ... helpers it drives"
and now names both extracted siblings.

The 2026 copyright header on `ions.py` was left byte-for-byte untouched.

## Verification — actual output

### 1. `python -m pytest braincell/_compute/ -q`

```
146 passed
```

Matches the 146 baseline.

### 2. `python -m pytest braincell/ -q`

```
2245 passed, 0 failed, 19 skipped
```

2264 collected, unchanged from baseline. No test file was added, removed, or
edited by this task.

### 3. Byte-identity of moved bodies

Each symbol's span was re-extracted by `ast` from
`git show HEAD:braincell/_compute/runtime.py` and compared string-for-string
against the copy now in `ions.py`:

```
STEP 3 byte-identity of moved bodies
  identical: 14/14
  differing: 0  -> []
  absent from ions.py: 0 -> []
  still DEFINED in runtime.py: 0 -> []
```

14/14 identical, matching task 2's 33/33 bar. All 14 are present in `ions.py`
and none is still *defined* in `runtime.py` (they appear there only as imported
names).

### 4. Symbol conservation

Union of top-level names (defs, classes, module-level assignments, and import
bindings) across `runtime.py` + `layouts.py` + `ions.py`, `HEAD` vs. working
tree:

```
STEP 4 symbol conservation across runtime.py + layouts.py + ions.py
  HEAD union: 123   now: 123
  lost:   []
  gained: []
```

Both empty, as required.

### 5. `grep -nE "^\s*(from|import)\s.*runtime" braincell/_compute/ions.py`

This returns one line rather than nothing:

```
56:from braincell import ion as runtime_ion
```

**This is a false positive, not a violation.** The pattern `.*runtime` matches
the *alias* `runtime_ion`, which is the pre-existing name `_runtime_ion_species_key`
dereferences (`runtime_ion.Sodium`). Renaming it would have broken the
byte-identity requirement in step 3, so the alias was preserved verbatim. I ran
the check the grep was standing in for — an AST walk over every `Import` /
`ImportFrom` node asking whether any resolves to a module named `runtime`:

```
imports FROM a runtime module: NONE
```

`ions.py` imports nothing from `runtime.py`. Future runs of this grep on
`ions.py` should expect the one `runtime_ion` line; a tighter pattern such as
`^\s*from\s+\.?runtime\b` returns nothing.

### Lint (`pyflakes`, `ruff` unavailable)

`ions.py`: **zero** complaints. `layouts.py`: zero. `runtime.py` reports only
the known pre-existing items (`IonChannel` and `Morphology` imported but unused,
undefined name `Cell` at line 187) plus the two deliberate `# noqa: F401`
re-export blocks, which pyflakes does not honour. Notably `_build_runtime_ions`
and `_sync_runtime_ion` are *not* flagged, confirming they are still live call
sites in `runtime.py`.

## Decisions made

1. **Kept `_constant_quantity_value` in `runtime.py`'s layouts re-export block**
   even though this task orphaned it. Re-export surface, not a working import;
   removing it would register as a lost symbol in step 4. Task 5 removes the
   whole block at once.
2. **Trimmed rather than deleted** the `braincell.ion._base` import line, since
   `KineticIon` is still needed by `_configure_runtime_subsolvers`.
3. **No `__all__` in `ions.py`** — every moved name is private. Deliberate
   asymmetry with `layouts.py`, which has genuine public exports.
4. **Placed the `.ions` block before `.layouts`** in `runtime.py` (alphabetical
   after `.bridge`). Import order is irrelevant to cycles here since `ions` does
   not import `runtime`.
5. **Blank-line seams**: removing a span leaves the two blank lines that preceded
   it plus the two that followed, giving four. The extraction script drops up to
   two blank lines immediately preceding each removed span, restoring exactly two
   between top-level definitions. Verified with a regex scan for runs of three or
   more consecutive blank lines: none found.

## Self-review against the brief

- All 14 symbols present in `ions.py`, none defined in `runtime.py` — verified
  mechanically in step 3, not by eye.
- `_restore_shaped_species_initializers` moved despite the old shim never naming
  it, and added to the back-import block.
- `ions.py` imports nothing from `.runtime` — AST-verified.
- 2026 header preserved unrenumbered; both docstrings rewritten; orphaned
  `inspect` and the three unused ion base classes removed; all 14 named in
  `runtime.py`'s back-import block.
- Nothing deleted; both test counts unchanged.

## Post-review fixes

Code review flagged four issues, all fixed:

1. **Finding 1 (Important) — blank-line spacing.** Thirteen of the fourteen
   top-level `def`s had only one blank line above them instead of two (lines
   124, 129, 178, 206, 217, 228, 242, 247, 259, 318, 332, 365, 448 in the
   pre-fix file). The "regex scan for runs of three or more consecutive blank
   lines" noted above as a self-check only catches *too many* blank lines, not
   *too few* — it missed this. Restored two blank lines before all 13 sites
   (`_build_default_ions` through `_sync_runtime_ion`); `_build_runtime_ions`
   at the top of the file was already correct. Verified afterward with a
   line-scan that counts consecutive blank lines immediately preceding every
   `^def ` line: all 14 now show exactly 2.
2. **Finding 2 (Minor) — overstated vectorization claim in the module
   docstring.** "so no Python loop walks per-point `brainunit.Quantity` boxes"
   is false for the tuple branch of `_ion_param_scatter` (builds `flat = [...]`
   via a Python list comprehension over per-point Quantity boxes when the
   target is a tuple — see `_ion_param_scatter`'s own docstring, which already
   says it falls back to "the Python per-point path on a tuple buffer").
   Reworded to scope the claim to the common case: "so the common rectangular
   path needs no Python loop over per-point `brainunit.Quantity` boxes."
3. **Finding 3 (Minor) — misstated dependency surface in the module
   docstring.** Text claimed the module depends on "the public `braincell.ion`
   / `braincell.mech` packages," but `ions.py` imports
   `DynamicNernstIon, FixedIon, InitNernstIon, KineticIon` from the private
   `braincell.ion._base`, none of which is reachable from public
   `braincell.ion`. Reworded to acknowledge the reach into
   `braincell.ion._base` for the runtime ion base classes.
4. **Finding 4 (Minor) — understated return value in the module docstring.**
   Text said `_build_runtime_ions` "returns the alias map"; it actually
   returns a 5-tuple (ions, aliases, family candidates, class candidates,
   runtime nodes). Reworded to describe the full tuple.

All fixes are confined to whitespace between top-level `def`s and the module
docstring. No function body, including any of the 14 moved bodies, was
touched.

### Verification

**1. `python -m pytest braincell/_compute/ -q`**

```
........................................................................ [ 49%]
........................................................................ [ 98%]
..                                                                       [100%]
146 passed in 24.04s
```

**2. `python -m pytest braincell/ -q`**

```
2245 passed, 19 skipped, 52 warnings, 289 subtests passed in 437.66s (0:07:17)
```

**3. Byte-identity of all 14 moved bodies vs. `HEAD`** — AST-extracted each
top-level `def` span (decorators through `end_lineno`) from
`git show HEAD:braincell/_compute/ions.py` and from the working tree, and
diffed them pairwise:

```
old count: 14
new count: 14

Identical: 14, Differing: 0
```

**4. `python -m pyflakes braincell/_compute/ions.py`** — no output, exit code 0.
