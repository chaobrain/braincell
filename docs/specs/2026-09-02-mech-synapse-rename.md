# Rename `braincell.mech.SynapseSpec` to `braincell.mech.Synapse`

## Motivation

`braincell.mech.SynapseSpec` is the immutable, registry-keyed declaration a user
places on a cell:

```python
cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
```

The `Spec` suffix carries no information. Every type in `braincell.mech` is a
declaration — that is the package's entire purpose, stated in its own docstring
("The `braincell.mech` package is purely declarative"). No sibling carries the
suffix: `Channel`, `Ion`, `Density`, `Junction`, `CableProperty`, `CurrentClamp`,
`SineClamp`, `FunctionClamp`, `StateProbe`, `MechanismProbe`, `CurrentProbe`.
`SynapseSpec` was the lone exception, so the suffix read as a distinction where
none existed.

The canonical spelling becomes `mech.Synapse`.

## History: this reclaims a name that already meant this class

Until commit `6c3ae89` (PR #141, "refactor(mech): fix two silent bugs, delete dead
API, cache Params hashing"), `braincell/mech/_point.py` carried both spellings:

```python
class SynapseSpec(Point): ...

class Synapse(SynapseSpec):
    """Deprecated spelling of :class:`SynapseSpec`."""
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "braincell.mech.Synapse is deprecated; use braincell.mech.SynapseSpec.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

PR #141 deleted that shim as dead API. This change reverses the direction of the
former deprecation: `Synapse` becomes the primary and only name, and `SynapseSpec`
is retired.

Because the shim is already gone from `main`, the name `Synapse` is free in
`braincell.mech` and no intermediate state ever has both spellings live.

## Decisions

**No compatibility alias.** `SynapseSpec` is removed outright; `mech.SynapseSpec`
raises `AttributeError`. The package is at `0.1.0` and the surrounding series of
refactors (#137–#142) has landed breaking changes freely. A shim would also
recreate exactly the dead API that #141 just deleted, in mirror image.

**`mech.Synapse` and `braincell.Synapse` are different classes, and that is
tolerated.** `braincell._base_channel.Synapse` is the *runtime* base class,
re-exported as top-level `braincell.Synapse` and subclassed by
`braincell/synapse/markov.py` (`ExpSyn`, `Exp2Syn`, `AMPA`, `GABAa`, `NMDA`).
`mech.Synapse` is the declaration that *names* one of those runtime classes by
registry key. They never appear in the same expression, and the namespaces stay
disjoint — `mech.Synapse` is deliberately **not** re-exported at top level, just as
`SynapseSpec` never was.

The codebase already had a convention for the two files that must mention both, and
it is preserved verbatim:

| Import site | Alias |
|---|---|
| runtime class, in `_compute/state.py`, `_multi_compartment/{cell,currents}.py` | `Synapse as RuntimeSynapse` |
| declaration, in `_compute/{bindings,layouts,state}.py`, `_multi_compartment/cell.py` | `Synapse as SynapsePlacement` |

Only `_compute/state.py` and `_multi_compartment/cell.py` strictly need the
`SynapsePlacement` alias; `_compute/bindings.py` and `_compute/layouts.py` keep it
for consistency across `_compute`, and because the aliased spelling is what every
downstream line in those files already reads.

The three files that import the declaration and never mention the runtime class —
`_compute/table.py`, `_multi_compartment/synapses.py`, `network/engine.py` — import
plain `Synapse`.

**Private identifiers named after the old class are renamed too**, so no stale echo
survives: `_synapse_spec_origins` → `_synapse_origins` and
`_split_synapse_spec_rows` → `_split_synapse_rows`.

**Historical spec filenames are not renamed.** `2026-08-21-synapse-spec-view-runtime.md`
keeps its slug — the date-prefixed filename is a chronological record per
`AGENTS.md` §8, and renaming it would rewrite history rather than document it. Only
body text referring to the live API changes.

## Scope

166 occurrences across 36 files, all renamed:

- **Definition** — `braincell/mech/_point.py` (class, `__all__`, module docstring
  bullet, docstring example, two `require_str` owner strings, `__repr__`).
- **Export** — `braincell/mech/__init__.py` (import block, `__all__`, docstring);
  `braincell/mech/_base.py` cross-reference. `braincell/__init__.py` is untouched —
  it imports only `CableProperty`, `CurrentClamp`, `FunctionClamp`, `SineClamp`
  from `.mech`.
- **Consumers** — 7 modules across `_compute/`, `_multi_compartment/`, `network/`,
  including two `raise TypeError` messages and the `Network.connect` docstring in
  `network/engine.py`.
- **Tests** — 10 files plus the `braincell/network/_testing.py` helper. The test
  class in `mech/_point_test.py` was already named `SynapseTest`.
- **Docs** — `docs/apis/braincell.mech.rst`, `docs/design/interface-map.md`,
  `docs/design/network/{api,architecture}.md`, and body text in the two historical
  specs.
- **Examples** — 6 notebooks (source cells and Chinese-prose markdown; no stored
  `__repr__` output contains the name, so no outputs are invalidated) and 3 scripts.

## Verification

1. `grep -rn "SynapseSpec\|synapse_spec" braincell/ docs/ examples/` → zero hits.
2. `from braincell.mech import Synapse; repr(Synapse("ExpSyn"))` →
   `Synapse(synapse_type='ExpSyn', params=Params({}), name=None)`.
3. `hasattr(braincell.mech, "SynapseSpec")` is `False`; `braincell.Synapse is not
   braincell.mech.Synapse`.
4. `pytest braincell/` → same pass/skip counts as the `b1e7d40` baseline; this
   change adds and removes no tests.
5. `ruff check` clean; every `examples/**/*.ipynb` still parses as JSON.
