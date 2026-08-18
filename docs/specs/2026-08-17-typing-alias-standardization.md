# Standardize type annotations on `braincell/_typing.py` aliases

## Problem

`braincell/_typing.py` is meant to be the single source of truth for the
project's shared type annotations, but only six modules imported from it. Every
other module respelled the raw type expression inline:

| Raw expression | Occurrences | Files |
|---|---|---|
| `Union[brainstate.typing.ArrayLike, Callable]` | 418 | 11 |
| `Union[brainstate.typing.ArrayLike, Callable, None]` | 46 | 4 |
| `brainstate.typing.ArrayLike` (bare) | 117 | 8 |
| `brainstate.typing.Size` | 258 | 17 |

Roughly 840 hand-written annotations that should be four short names. Widening
what an initializer accepts would have meant editing 418 sites, and a single
`__init__` signature in `braincell/channel/potassium.py` repeated
`Union[brainstate.typing.ArrayLike, Callable]` seven times. The alias that did
exist (`Initializer`) was silently bypassed, so `_typing.py` no longer described
the codebase.

## Scope

- Add `ArrayLike`, `Size`, and `PyTree` aliases to `_typing.py`; redefine
  `Initializer` in terms of `ArrayLike` so the three stay consistent by
  construction.
- Replace every raw occurrence of the four expressions above **in annotation
  position**.
- Rename the `VectorFiled` typo to `VectorField` (internal only — not re-exported
  from `braincell/__init__.py`, so no public API break).
- Route `braincell/vis/scene.py` through `_typing` instead of its own
  `from brainstate.typing import ArrayLike`.
- Record the convention in AGENTS.md under `## Critical Conventions`.

### Out of scope

These raw types are *not* semantically the alias and stay as they are:

- `Hashable` at `braincell/_base_ion.py:418` and in `braincell/vis/layout/_cache.py`
  — generic hashables, not `SectionName`.
- The 60+ `tuple[str, ...]` across `_compute/runtime.py`, `io/neuromorpho/query.py`,
  `mech/_registry.py`, `channel/_base.py` — branch names, species, and columns,
  not state `Path`s. The only true `Path` uses (`quad/_exp_euler.py:372,382`)
  already used the alias.
- `Optional[str]` on the ubiquitous `name=` parameter.

## Key decision: annotations only, not prose

The first pass rewrote docstrings too (`size : brainstate.typing.Size` →
`size : Size`, 180 occurrences). That was reverted.

`_typing.py` carries a leading underscore precisely because its path is not part
of the supported public API (AGENTS.md, "Note on package naming"), and
`braincell/__init__.py` does not re-export these names. A reader of a public
NumPy-doc `Parameters` block who sees `size : Size` has no importable name to
follow and Sphinx has nothing to cross-reference. Prose therefore keeps the
fully-qualified `brainstate.typing.Size`, which resolves.

Three modules (`channel/_base.py`, `channel/leaky.py`, `quad/_exp_euler.py`)
carried bare `ArrayLike`/`Size` in their docstrings before this change; their
prose is left exactly as it was.

## Implementation notes

Ordered literal substitution — the `Union[...]` forms must be rewritten before
the bare `brainstate.typing.ArrayLike` form, or the longer patterns stop
matching:

1. `Union[brainstate.typing.ArrayLike, Callable, None]` → `Optional[Initializer]`
2. `Union[brainstate.typing.ArrayLike, Callable]` → `Initializer`
3. `brainstate.typing.ArrayLike` → `ArrayLike`
4. `brainstate.typing.Size` → `Size`

There is no PEP-604 (`ArrayLike | Callable`) spelling anywhere and no multi-line
`Union[` continuation, so literal replacement is safe.

The substitution makes `Union`, `Callable`, and `import brainstate` dead in most
touched modules; `braincell/channel/leaky.py` is the exception and keeps `Union`
for `size: Union[int, Sequence[int]]`. Dead imports were removed only where the
name was live before the change and dead after — verified by differential
analysis, not by inspection.

## Verification

- Zero residual raw forms; zero `VectorFiled`.
- `python -m compileall braincell/` and `import braincell` clean.
- **Differential lint.** `pyproject.toml` puts both `F401` and `F821` in
  `[tool.ruff] lint.ignore`, so `pre-commit` cannot catch a dead or missing
  import — the obvious safety net for the import cleanup does not exist. Use
  `pyflakes` over the tree and diff its output against the merge-base instead;
  the requirement is **zero introduced findings**, not zero findings.
- **Annotation equivalence.** The aliases are definitionally identical to what
  they replace, so `typing.get_type_hints()` over a representative set of classes
  must hash identically before and after.
- `pytest braincell/` green.
- Diff review: every changed line is an annotation, an import, or part of the
  `_typing.py` / AGENTS.md additions.
