# Channel/Ion API Migration + Doc Refresh — Design

Date: 2026-05-25

## Problem

The classes exported by `braincell.channel` and `braincell.ion` were renamed
across PRs #80 and #93 (dropping the `I` prefix, adding model/region suffixes
like `_MA2020_GoC`), but the API documentation was never updated:

- `docs/apis/braincell.channel.rst` lists **44** classes, **41 of which no
  longer exist** (e.g. `INa_HH1952`, `ICav12_Ma2020`, `Ih_HM1992`,
  `CalciumChannel`). The package now exports **109** channel classes.
- `docs/apis/braincell.ion.rst` lists **8** classes, all still valid, but **18
  current exports are undocumented** (`CalciumInitNernst`, `Cdp*`, `Toy*`,
  `PotassiumInitNernst`, `SodiumInitNernst`, `build_placeholder_ions`).
- `braincell/channel/__init__.py` contains broken dead code: a misspelled
  `__get_attr__()` (never invoked; references `Warning.warn()`) that was an
  abandoned attempt at a backward-compat shim.

All 774 channel/ion tests currently pass — nothing is broken at runtime. The
old names appear nowhere in the codebase, examples, or tests; the rename is
fully complete internally.

## Goals

1. External code using the **clean 1:1 renamed** old names keeps working via a
   `DeprecationWarning`-emitting alias in `braincell.channel`.
2. The API docs (`docs/apis/braincell.channel.rst`, `docs/apis/braincell.ion.rst`)
   list only the current ("latest") exported names.
3. Remove the broken `__get_attr__` dead code.

## Non-goals

- No changes to channel/ion implementation classes.
- No backward-compat aliases for ambiguous renames (one old name → multiple
  region variants) or for removed classes (old base classes, `_ss`/`_markov`/
  `_Rsg` channels). These resolve to a normal `AttributeError`.
- No changes to `braincell.ion` runtime behavior (no ion names were removed —
  only doc completion is needed there).

## Part 1 — Deprecation aliases in `braincell.channel`

Replace the broken `__get_attr__()` in `braincell/channel/__init__.py` with a
real PEP-562 module-level `__getattr__`, backed by an explicit alias map of the
**19 clean 1:1 renames** (old name = new name with leading `I` dropped, where
the stripped name is an exact current export):

```python
_DEPRECATED_ALIASES = {
    "INa_HH1952": "Na_HH1952",
    "INa_Ba2002": "Na_Ba2002",
    "INa_TM1991": "Na_TM1991",
    "IK_HH1952": "K_HH1952",
    "IK_TM1991": "K_TM1991",
    "IK_Leak": "K_Leak",
    "IKDR_Ba2002": "KDR_Ba2002",
    "IKNI_Ya1989": "KNI_Ya1989",
    "IKA1_HM1992": "KA1_HM1992",
    "IKA2_HM1992": "KA2_HM1992",
    "IKK2A_HM1992": "KK2A_HM1992",
    "IKK2B_HM1992": "KK2B_HM1992",
    "ICaN_IS2008": "CaN_IS2008",
    "ICaL_IS2008": "CaL_IS2008",
    "ICaT_HM1992": "CaT_HM1992",
    "ICaT_HP1992": "CaT_HP1992",
    "ICaHT_HM1992": "CaHT_HM1992",
    "ICaHT_Re1993": "CaHT_Re1993",
    "IAHP_De1994": "AHP_De1994",
}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"braincell.channel.{name} is deprecated and will be removed; "
            f"use braincell.channel.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Notes:
- `import warnings` added at the top of the module.
- Aliases are **not** added to `__all__`, so `from braincell.channel import *`
  and the Sphinx autosummary ignore them.
- PEP-562 `__getattr__` only fires when normal attribute lookup fails, so it
  does not interfere with the existing `from .calcium import *` etc.
- Excluded (resolve to `AttributeError`): ambiguous `_Ma2020` names
  (`ICav12_Ma2020`, `ICav13_Ma2020`, `ICav23_Ma2020`, `ICav31_Ma2020`,
  `ICaGrc_Ma2020`, `IKM_Grc_Ma2020`, `IKca1_1_Ma2020`, `IKca2_2_Ma2020`,
  `IKca3_1_Ma2020`, `IKv11_Ak2007`, `IKv34_Ma2020`, `IKv43_Ma2020`,
  `Ih1_Ma2020`, `Ih2_Ma2020`, `Ih_HM1992`); removed classes
  (`IKA_p4q_ss`, `IKK2_pq_ss`, `IK_p4_markov`, `INa_Rsg`, `INa_p3q_markov`,
  `CalciumChannel`, `PotassiumChannel`, `SodiumChannel`).

## Part 2 — Rewrite `docs/apis/braincell.channel.rst`

Drop all 41 dead entries. List the **109 current exports**, grouped to match
the source submodules. Counts per group:

| Group | Submodule | Count |
|-------|-----------|-------|
| Calcium Channels | `calcium.py` | 31 |
| Hyperpolarization-Activated Channels | `hyperpolarization_activated.py` | 8 |
| Leakage Channels | `leaky.py` | 2 (`IL`, `LeakageChannel`) |
| Potassium Channels | `potassium.py` | 39 |
| Potassium-Calcium Channels | `potassium_calcium.py` | 15 |
| Sodium Channels | `sodium.py` | 14 |

Each group is an `autosummary` block (same directives as the existing file:
`:toctree: generated/`, `:nosignatures:`, `:template: classtemplate.rst`).
Old base classes `CalciumChannel`/`PotassiumChannel`/`SodiumChannel` are gone
(channels now subclass the non-exported `HH` base); only `LeakageChannel`
remains as a documented base.

`K_Kv_test` and `Cav3p1Test_PC24` are included (faithful to the current
`__all__`).

## Part 3 — Complete `docs/apis/braincell.ion.rst`

Rewrite to list all **26 exports**, grouped:

| Group | Members |
|-------|---------|
| Calcium Ions & Dynamics | `Calcium`, `CalciumFixed`, `CalciumDetailed`, `CalciumFirstOrder`, `CalciumInitNernst`, `Cdp*` (6), `Toy*` (5) — 19 total |
| Potassium Ions | `Potassium`, `PotassiumFixed`, `PotassiumInitNernst` |
| Sodium Ions | `Sodium`, `SodiumFixed`, `SodiumInitNernst` |
| Helpers | `build_placeholder_ions` (function — uses `autofunction` rather than `autosummary` classtemplate) |

## Verification

1. New test (e.g. `braincell/channel/_deprecation_test.py`):
   - `with pytest.warns(DeprecationWarning): assert braincell.channel.INa_HH1952 is braincell.channel.Na_HH1952` (sample a few aliases).
   - A removed/ambiguous name (e.g. `INa_Rsg`, `ICav12_Ma2020`) raises `AttributeError`.
   - Deprecated names are absent from `braincell.channel.__all__`.
2. `python -c "import braincell.channel, braincell.ion"` runs clean.
3. Existing 774 channel/ion tests still pass.
4. Sphinx doc build emits no autosummary "failed to import" warnings for the
   newly listed names (every listed name resolves as a real export).

## Source-of-truth lists (current exports)

Channel `__all__` per submodule and ion `__all__` per submodule are captured at
design time and used verbatim to populate the rst files. The implementation
step regenerates them from the live package to avoid transcription drift.
