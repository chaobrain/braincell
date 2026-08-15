# Docstring and reference upgrade for `braincell.ion` and `braincell.channel`

Status: proposed
Branch: `worktree-ion-channel-docstrings`
Base: `ca9ec8c` (main)

## Goal

Bring every public symbol in `braincell/ion/` and `braincell/channel/` up to the
NumPy-doc standard mandated by `AGENTS.md`, and give every literature-derived
channel and ion a complete, web-verified bibliographic citation.

This is a documentation-only change. No class is renamed, no signature changes,
no behavior changes.

## Background: the current state

A survey of the two packages at `ca9ec8c` found:

| Package | Source modules | Public symbols | Lines |
|---|---|---|---|
| `braincell/channel/` | 8 | 119 (118 classes + `ghk_flux`) | 6,013 |
| `braincell/ion/` | 5 | 36 classes | 3,698 |

Across both packages there are **two `References` sections and three numbered
citations** in total — one in `channel/hyperpolarization_activated.py`, one in
`ion/calcium.py` — plus a third citation block that exists only inside a
commented-out class. There is one `Examples` section.

Most channel classes carry a single summary line and nothing else. The current
state of `KA1_HM1992` is representative: eight constructor parameters, four rate
functions full of hard-coded literature constants, and this docstring:

```python
r"""Huguenard & McCormick 1992 IA1 potassium current."""
```

**The bibliography currently lives in the class names, not in the docstrings.**
Names encode a two-letter author key plus a year — `KDR_Ba2002`, `Kv4p3_MA2024_PC`,
`Nav1p6_MA2020_GoC`. A reader who wants the source paper has to decode the key
and guess. Fifteen distinct keys cover 122 of the 154 public classes:

| Key | Public classes | Modules |
|---|---|---|
| `MA2020` | 32 | channel: calcium, hyperpolarization_activated, potassium, potassium_calcium, potassium_sodium, sodium |
| `MA2024` | 19 | channel: calcium, hyperpolarization_activated, potassium, potassium_calcium, sodium |
| `SU2015` | 16 | channel: calcium, hyperpolarization_activated, potassium, potassium_calcium, sodium |
| `MA2025` | 16 | channel: calcium, hyperpolarization_activated, potassium, potassium_calcium, sodium |
| `RI2021` | 15 | channel: calcium, hyperpolarization_activated, potassium, potassium_calcium, sodium |
| `HM1992` | 7 | channel: calcium, hyperpolarization_activated, potassium |
| `ZH2019` | 5 | channel: calcium, hyperpolarization_activated, potassium, sodium |
| `IS2008` | 2 | channel: calcium |
| `Ba2002` | 2 | channel: potassium, sodium |
| `TM1991` | 2 | channel: potassium, sodium |
| `HH1952` | 2 | channel: potassium, sodium |
| `HP1992` | 1 | channel: calcium |
| `Re1993` | 1 | channel: calcium |
| `Ya1989` | 1 | channel: potassium |
| `De1994` | 1 | channel: potassium_calcium |

The remaining 32 public classes carry no key: the template layer (`Gate`,
`Transition`, `HH`, `OhmicHH`, `Markov`, `ghk_flux`, `KineticIon`, the ion
mixins, the reaction-network dataclasses), the ion containers (`Calcium`,
`CalciumFixed`, `CalciumDetailed`, …), the leak channels, and two apparent
scratch classes.

Counts in this table are derived from `__all__` membership via AST parsing and
supersede any figure obtained by grepping class names, which over-counts.
Phase 1 re-derives the exact key-to-class mapping programmatically rather than
trusting this table.

### The citation key names the model, not always the kinetics

103 of the 122 cited classes were imported from NMODL. Their current docstrings
say so — `"""Template-based import of ``Kv4p3_MA20_GoC.mod``."""` — and the repo
ships the 98 source `.mod` files under
`examples/neuron_compare/Cerebellum_mod/{BC,DCN,GoC,GrC,IO,PC,SC}/`. Eighteen of
them carry an explicit author attribution in their header, and it frequently is
not the author the class name implies. `Kv4p3_MA20_GoC.mod`, keyed `MA2020`,
opens:

```
TITLE Cerebellum Granule Cell Model
COMMENT
        KA channel
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT
```

The equations are D'Angelo, Nieus & Fontana's; Masoli et al. (2020) reused them
in the Golgi cell assembly that BrainCell imported from. Both facts are true and
a reader needs both.

This is systematic across the cerebellar imports rather than a handful of
mistakes, so it is handled structurally by the two-level reference format in
decision 7, not by 103 individual discrepancy notes under the mismatch policy.
The mismatch policy still governs genuine one-off errors.

`examples/neuron_compare/Cerebellum_mod/README.md` additionally pins each suffix
to a cell type — BC basket, DCN deep cerebellar nuclei, GoC Golgi, GrC granule,
IO inferior olive, PC Purkinje, SC stellate — and records per-mechanism import
deviations (`TABLE` removal, `derivimplicit`→`cnexp`, NMODL default-precision
rewrites). Those deviations belong in `Notes`.

## Scope

**In scope.** Every name in the `__all__` of each source module in
`braincell/ion/` and `braincell/channel/`, including the public template bases
in both `_base.py` files. That is 155 symbols.

**Out of scope.**

- Underscore-private helpers: `_Specs`, `_Species`, `_Conserve`, `_Flux` in
  `ion/_base.py`.
- Module-level docstrings in the two `__init__.py` files.
- Any rename, signature change, or behavior change.
- `Examples` sections. Individual channels are exercised by the module-level
  documentation and tutorials; 155 doctested example blocks would be a large
  maintenance surface for little gain.
- `braincell.mech`, `braincell.synapse`, and every other package.

## Decisions

Settled during brainstorming, recorded so implementation does not relitigate
them:

1. **Depth**: full NumPy-doc — summary, extended summary, Parameters, Notes,
   References — with `Examples` omitted.
2. **Verification standard**: every citation is web-verified against the real
   literature record. No citation ships on recall alone.
3. **Bibliography layout**: the complete entry is repeated in every docstring
   that cites it, so `help(cls)` and the rendered API page are self-contained;
   `docs/design/ion-channel-bibliography.md` holds the single canonical list
   that those entries are copied from.
4. **Coverage**: public API plus public template bases; private helpers excluded.
5. **Mismatch policy**: where verification shows the code's equations came from
   a different paper than the class name implies, the docstring cites the paper
   the equations actually came from, a `Notes` line explains the discrepancy,
   and the design doc logs it as an open question. Nothing is renamed.
6. **Enforcement**: ship a references-presence guard test, not a
   Parameters-versus-signature guard, to avoid brittleness when signatures change.
7. **Two-level references for imported mechanisms**: where a class was imported
   from an NMODL file whose kinetics predate the model named in the class name,
   `References` carries both entries — `.. [1]` the origin of the equations,
   `.. [2]` the model BrainCell imported from — and `Notes` names the source
   `.mod` file. Where the two coincide, a single entry is used.

## Phase 1 — Verified bibliography

**Artifact**: `docs/design/ion-channel-bibliography.md`, a durable design note
per `AGENTS.md` rule 8.

**Structure**: one section per citation key, each containing

- the full record: authors, title, journal, volume, issue, page range, year, DOI;
- the exact `.. [1]` reST block to be pasted into docstrings, so the docstring
  text is copied rather than retyped;
- the list of classes citing the key, generated from the AST;
- an attribution note recording what was checked and how confident the result is.

**Step 0 — harvest in-repo provenance first.** Before any web search, read the
`TITLE`/`COMMENT` header of all 98 `.mod` files under
`examples/neuron_compare/Cerebellum_mod/*/{channel,ion}/` and record every author,
title, and revision date found. This is free, authoritative for what BrainCell
actually imported, and is what surfaces cases like `Kv4p3_MA20_GoC`. Web
verification then confirms and completes what the headers assert, rather than
starting from a guess.

**Verification protocol.** A key is verified only when both of the following hold:

1. *Record check.* The bibliographic fields are confirmed against the publisher
   record or an equivalent authoritative index. A DOI that resolves to the
   stated title, authors, and year is sufficient.
2. *Attribution check.* The paper is confirmed to actually describe the current
   in question, by comparing the kinetics the paper reports against the
   hard-coded constants in the citing class. A paper by the right authors in the
   right year that does not contain the channel is a failed attribution, not a
   pass.

The attribution check is what makes this phase worth doing separately. It is
also where `IS2008` is most at risk: the only in-repo evidence for it is the
prose fragment "Strowbridge 2008", and the two classes citing it are calcium
channels whose provenance has not been confirmed.

**Failure handling.** A key that fails either check is recorded in the
bibliography's "Unresolved attributions" section with the evidence gathered so
far, and its classes get a `Notes` line stating that the source is unconfirmed
rather than a fabricated citation. Shipping an unverified citation formatted to
look verified is worse than shipping an explicit gap.

**Deliverable**: one commit adding `docs/design/ion-channel-bibliography.md`.

## Phase 2 — Docstrings

### Template

Raw strings (`r"""`) throughout, since `Notes` carries LaTeX. Sections follow the
`AGENTS.md` canonical order with `Examples` omitted.

- **Short summary** — one line naming the current and its source.
- **Extended summary** — 2–4 sentences: the biophysical role of the current, the
  preparation or cell type it was characterized in, the model it belongs to, and
  its gating structure.
- **Parameters** — every `__init__` argument, with its default and unit stated in
  the description. Units go in prose, not the type field, because the type is
  almost always `brainstate.typing.ArrayLike or Callable`.
- **See Also** — only where it earns its place: sibling variants such as
  `KA1_HM1992`/`KA2_HM1992`, or a channel and the ion it binds to. Not
  boilerplate across all 155.
- **Notes** — the gating equations as `.. math::`, plus the conventions needed to
  reproduce them: that `V` enters as `(V - V_sh)` in mV, that τ is in ms, and how
  `q10`/`temp_ref` scale each gate. Class attributes (`root_type`, `gates`) are
  described here rather than in a separate `Attributes` section, keeping to the
  nine canonical sections.
- **References** — the full entry, byte-identical to the bibliography.

**Equations are transcribed from the code, not from the paper.** The docstring
documents the implementation as it stands. Any place where the two disagree is a
Phase 1 finding handled under the mismatch policy.

### Worked example

Replacing the current one-liner in `braincell/channel/potassium.py`:

```python
@register_channel("KA1_HM1992")
class KA1_HM1992(OhmicHH):
    r"""Huguenard & McCormick (1992) fast transient A-type K+ current (IA1).

    Rapidly activating and inactivating outward potassium current of thalamic
    relay neurons, responsible for delaying the onset of firing after a
    hyperpolarizing step. This is the faster of the two A-type components in
    the Huguenard & McCormick thalamocortical relay cell model; see
    :class:`KA2_HM1992` for the slower component. Gating follows the
    Hodgkin-Huxley form :math:`g = \bar{g} p^4 q`, with a fourth-power
    activation gate ``p`` and a single inactivation gate ``q``.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the channel.
    g_max : brainstate.typing.ArrayLike or Callable
        Maximal conductance density. Defaults to ``30 mS/cm^2``.
    temp : brainstate.typing.ArrayLike
        Absolute simulation temperature. Defaults to 36 degrees Celsius.
    q10_p : brainstate.typing.ArrayLike or Callable
        Temperature coefficient of the activation gate. Defaults to ``1.0``.
    temp_ref_p : brainstate.typing.ArrayLike
        Reference temperature at which the ``p`` rates were measured.
        Defaults to 36 degrees Celsius.
    q10_q : brainstate.typing.ArrayLike or Callable
        Temperature coefficient of the inactivation gate. Defaults to ``1.0``.
    temp_ref_q : brainstate.typing.ArrayLike
        Reference temperature at which the ``q`` rates were measured.
        Defaults to 36 degrees Celsius.
    V_sh : brainstate.typing.ArrayLike or Callable
        Voltage shift applied to every rate expression. Defaults to ``0 mV``.
    name : str, optional
        Instance name.

    See Also
    --------
    KA2_HM1992 : Slower A-type component of the same model.

    Notes
    -----
    Writing :math:`v = (V - V_{sh})/\mathrm{mV}`, activation is

    .. math::
        p_\infty(v) = \frac{1}{1 + \exp(-(v + 60)/8.5)}

    .. math::
        \tau_p(v) = \frac{1}{\exp((v + 35.8)/19.7)
                   + \exp(-(v + 79.7)/12.7)} + 0.37

    and inactivation is

    .. math::
        q_\infty(v) = \frac{1}{1 + \exp((v + 78)/6)}

    .. math::
        \tau_q(v) = \begin{cases}
            \left[\exp((v + 46)/5)
                + \exp(-(v + 238)/37.5)\right]^{-1} & v < -63 \\
            19 & v \ge -63
        \end{cases}

    All time constants are in milliseconds. Each gate is scaled independently
    by :math:`q_{10}^{(T - T_{ref})/10}`, so ``q10_p``/``temp_ref_p`` and
    ``q10_q``/``temp_ref_q`` act only on their own gate. The class binds to
    :class:`braincell.ion.Potassium` through ``root_type``.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """
```

### Commit order

One commit per source module, smallest first so the template is exercised on
cheap modules before the large ones:

| # | Module | Public symbols |
|---|---|---|
| 1 | `channel/potassium_sodium.py` | 1 |
| 2 | `channel/leaky.py` | 2 |
| 3 | `ion/nonspecific.py` | 2 |
| 4 | `ion/potassium.py` | 3 |
| 5 | `ion/sodium.py` | 3 |
| 6 | `channel/_base.py` | 6 |
| 7 | `channel/hyperpolarization_activated.py` | 8 |
| 8 | `ion/_base.py` | 9 |
| 9 | `channel/sodium.py` | 14 |
| 10 | `channel/potassium_calcium.py` | 15 |
| 11 | `ion/calcium.py` | 19 |
| 12 | `channel/calcium.py` | 35 |
| 13 | `channel/potassium.py` | 38 |

## Phase 3 — Conformance guard

**Files**: `braincell/channel/_docstring_test.py` and
`braincell/ion/_docstring_test.py`, co-located per `AGENTS.md` rule 10. The
leading underscore is cosmetic; the `*_test.py` suffix is what pytest collects.

**Assertions**, for every in-scope symbol:

1. It has a non-empty docstring whose first line is a summary ending in a period.
2. It has a `References` section containing at least one `.. [n]` entry, **unless**
   it appears in the test module's explicit `_NO_PRIMARY_SOURCE` allowlist.
3. Every name in `_NO_PRIMARY_SOURCE` still exists and is still public, so the
   allowlist cannot rot into a list of deleted symbols.

References are therefore required by default. The citation-key pattern is not
itself an assertion — it is only the heuristic used when populating the
allowlist, since a key-named class can never legitimately belong on it.

**Why the allowlist.** 32 public classes have no citation key, and requiring a
`References` section from all of them would be wrong — `Gate`, `Transition`, and
`CalciumFixed` have no primary literature source. But some of them genuinely do:
`ghk_flux` implements Goldman–Hodgkin–Katz, `CalciumDetailed` derives from
Destexhe, and `KineticIon` follows NEURON's NMODL `KINETIC` semantics. An
allowlist forces each of the 32 to be a conscious decision rather than a silent
exemption, and forces the same decision when a new class is added.

**Parsing.** `numpydoc` is not a project dependency and this change does not add
one. The test uses a ~20-line section splitter over `inspect.getdoc` output.

**Deliverable**: one commit adding both test modules.

**Ordering refinement.** The implementation plan below lands the guard *before*
the module docstring tasks rather than after, scoped by a `_COVERED_MODULES`
tuple that each module task extends by one entry. This keeps the TDD cycle
(extend tuple → test fails → write docstrings → test passes → commit) without
ever leaving the guard red across the whole package, which was the objection to
building the guard first.

## Edge cases

- **`Cav3p1Test_PC24`** (`channel/calcium.py`) uses the suffix `PC24`, a
  two-digit year, so key-matching over four-digit years alone misses it. Under
  the references-required-by-default rule this is not a silent exemption — the
  class would fail the guard until documented — but the Phase 1 key inventory
  must still match the two-digit form or the class is dropped from the
  bibliography mapping. Treated as `MA2024`-derived pending confirmation.
- **`K_Kv_test`** (`channel/potassium.py`) and **`Cav3p1Test_PC24`** appear to be
  scratch or fixture classes that are nonetheless public. They get docstrings
  that say so plainly. Whether they should be public at all is logged as an open
  question, not resolved here.
- **Deprecated aliases** in `channel/__init__.py` (`INa_HH1952` → `Na_HH1952`,
  and 18 others) resolve through `__getattr__` and inherit the target's
  docstring. No separate documentation; the guard iterates `__all__`, which
  excludes them.
- **Subclass chains** such as `Nav1p6_MA2024_PC(Nav1p6_MA2020_GoC)` inherit a
  docstring if none is written. The guard uses `cls.__dict__.get("__doc__")`,
  not `cls.__doc__`, so an inherited docstring does not count as documented —
  otherwise the four `Nav1p6_*` subclasses would pass while citing the wrong
  paper for their cell type.
- **`u.celsius2kelvin(36.0)` defaults** must be described as "36 degrees Celsius",
  not as a kelvin number, or the docs will not match how a user writes the value.
- **Piecewise and numerically-stabilized rate functions** (`_linoid_stable`,
  `_x_over_one_minus_exp_neg_stable`) differ from the paper's closed form near
  singularities. `Notes` documents the stabilization where it applies rather than
  presenting the naive expression.

## Verification

- `pytest braincell/channel/ braincell/ion/` passes at every commit. Since the
  change is documentation-only, any failure means an edit escaped a docstring.
- `pytest braincell/` passes before the branch is proposed for merge.
- The new `_docstring_test.py` modules pass, and are confirmed to actually
  collect (`pytest --collect-only`) — a file that collects nothing is the failure
  mode `AGENTS.md` rule 10 exists to prevent.
- `pre-commit run --all` passes.
- Sphinx builds `docs/apis/braincell.channel.rst` and `docs/apis/braincell.ion.rst`
  without new warnings, confirming the `.. math::` and `.. [n]` markup renders.
  Duplicate citation labels across docstrings are expected and handled by
  numpydoc's per-docstring label mangling; a build that warns about them means
  the markup is wrong.

## Open questions

Logged for later decision; none blocks implementation.

- Should `K_Kv_test` and `Cav3p1Test_PC24` remain in the public `__all__`?
- Any attribution that fails Phase 1 verification is appended here with its
  evidence, per the mismatch policy.

## Risks

- **`IS2008` may not verify.** Handled by the failure protocol: an explicit
  "source unconfirmed" note, never a fabricated citation.
- **The Masoli-family papers (`MA2020`, `MA2024`, `MA2025`) cover 67 classes
  between them.** A wrong attribution there is the highest-impact error in the
  change, which is why the attribution check compares reported kinetics against
  the code's constants rather than matching on author and year alone.
- **Scale.** 155 symbols is a large diff. Mitigated by per-module commits and by
  the fact that the change cannot alter behavior.

---

# Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all 155 public symbols in `braincell/ion/` and `braincell/channel/`
complete NumPy-doc docstrings with web-verified references, guarded by a
conformance test.

**Architecture:** Three stages. A verified bibliography is built first from
in-repo `.mod` provenance plus literature verification, and written to
`docs/design/ion-channel-bibliography.md`. A conformance guard then lands with an
empty coverage tuple. Thirteen module tasks each add one module to that tuple,
watch the guard fail, write the docstrings, and watch it pass.

**Tech Stack:** Python 3, pytest + `unittest.TestCase`, `brainstate`,
`brainunit`, `braintools`. Sphinx with `numpydoc`-style parsing for rendering.
No new runtime or test dependency is added.

## Global constraints

- Documentation only. No renames, no signature changes, no behavior changes. The
  existing suite must pass unchanged at every commit.
- All work on branch `worktree-ion-channel-docstrings`. Never commit to `main`.
- Test modules use the `*_test.py` suffix, co-located with the code under test.
  Never `test_*.py`, never a `tests/` directory.
- Shared test-only helpers live in a leading-underscore module so pytest does not
  collect them.
- Physical quantities always carry explicit `brainunit` units. Docstrings state
  defaults in the unit a user would write, so `u.celsius2kelvin(36.0)` is
  documented as "36 degrees Celsius", not as a kelvin number.
- Docstrings are raw strings (`r"""`) wherever `Notes` carries LaTeX.
- Sections follow the AGENTS.md canonical order. `Examples` is omitted throughout.
- No new dependency. `numpydoc` is not installed and must not be added.
- Maintain JAX >= 0.8.0 compatibility (no JAX API is touched by this change).

## File structure

| File | Status | Responsibility |
|---|---|---|
| `docs/design/ion-channel-bibliography.md` | create | Canonical verified bibliography; every docstring citation is copied from here |
| `braincell/_testing.py` | create | Test-only NumPy-doc section splitter shared by both guard modules |
| `braincell/channel/_docstring_test.py` | create | Conformance guard for `braincell.channel` |
| `braincell/ion/_docstring_test.py` | create | Conformance guard for `braincell.ion` |
| `braincell/channel/*.py` (8 files) | modify | Docstrings only |
| `braincell/ion/*.py` (5 files) | modify | Docstrings only |

## Docstring templates

Seven structural cases cover all 155 symbols. Each module task names which
template applies to which symbol. **The citation text in these templates shows
format only — the actual entry text is copied verbatim from
`docs/design/ion-channel-bibliography.md` produced by Tasks 1–3.**

### T1 — Literature channel, `OhmicHH` or `HH`, single source

Applies where the class name's key is also the origin of the equations
(`KA1_HM1992`, `K_HH1952`, `Na_Ba2002`, `KNI_Ya1989`, …).

```python
@register_channel("KA1_HM1992")
class KA1_HM1992(OhmicHH):
    r"""Huguenard & McCormick (1992) fast transient A-type K+ current (IA1).

    Rapidly activating and inactivating outward potassium current of thalamic
    relay neurons, responsible for delaying the onset of firing after a
    hyperpolarizing step. This is the faster of the two A-type components in
    the Huguenard & McCormick thalamocortical relay cell model; see
    :class:`KA2_HM1992` for the slower component. Gating follows the
    Hodgkin-Huxley form :math:`g = \bar{g} p^4 q`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the channel.
    g_max : brainstate.typing.ArrayLike or Callable
        Maximal conductance density. Defaults to ``30 mS/cm^2``.
    temp : brainstate.typing.ArrayLike
        Absolute simulation temperature. Defaults to 36 degrees Celsius.
    q10_p : brainstate.typing.ArrayLike or Callable
        Temperature coefficient of the activation gate. Defaults to ``1.0``.
    temp_ref_p : brainstate.typing.ArrayLike
        Reference temperature at which the ``p`` rates were measured.
        Defaults to 36 degrees Celsius.
    q10_q : brainstate.typing.ArrayLike or Callable
        Temperature coefficient of the inactivation gate. Defaults to ``1.0``.
    temp_ref_q : brainstate.typing.ArrayLike
        Reference temperature at which the ``q`` rates were measured.
        Defaults to 36 degrees Celsius.
    V_sh : brainstate.typing.ArrayLike or Callable
        Voltage shift applied to every rate expression. Defaults to ``0 mV``.
    name : str, optional
        Instance name.

    See Also
    --------
    KA2_HM1992 : Slower A-type component of the same model.

    Notes
    -----
    Writing :math:`v = (V - V_{sh})/\mathrm{mV}`, activation is

    .. math::
        p_\infty(v) = \frac{1}{1 + \exp(-(v + 60)/8.5)}

    .. math::
        \tau_p(v) = \frac{1}{\exp((v + 35.8)/19.7)
                   + \exp(-(v + 79.7)/12.7)} + 0.37

    and inactivation is

    .. math::
        q_\infty(v) = \frac{1}{1 + \exp((v + 78)/6)}

    .. math::
        \tau_q(v) = \begin{cases}
            \left[\exp((v + 46)/5)
                + \exp(-(v + 238)/37.5)\right]^{-1} & v < -63 \\
            19 & v \ge -63
        \end{cases}

    All time constants are in milliseconds. Each gate is scaled independently
    by :math:`q_{10}^{(T - T_{ref})/10}`, so ``q10_p``/``temp_ref_p`` and
    ``q10_q``/``temp_ref_q`` act only on their own gate. The class binds to
    :class:`braincell.ion.Potassium` through ``root_type``.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of the
           currents involved in rhythmic oscillations in thalamic relay
           neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """
```

### T2 — NMODL import with two-level provenance

Applies wherever the bibliography records that the `.mod` kinetics predate the
model named in the class name.

```python
@register_channel("Kv4p3_MA2020_GoC")
class Kv4p3_MA2020_GoC(OhmicHH):
    r"""A-type (Kv4.3) potassium current of the Masoli et al. (2020) Golgi cell.

    Fast transient outward potassium current, gated as
    :math:`g = \bar{g} a^3 b` with independent activation ``a`` and
    inactivation ``b``. The kinetics originate in the D'Angelo et al.
    cerebellar granule cell model and were reused unchanged in the Masoli
    et al. (2020) Golgi cell, which is the assembly BrainCell imported.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the channel.
    g_max : brainstate.typing.ArrayLike or Callable
        Maximal conductance density. Defaults to ``3.2 mS/cm^2``.
    temp : brainstate.typing.ArrayLike
        Absolute simulation temperature. Defaults to 23 degrees Celsius.
    V_sh : brainstate.typing.ArrayLike or Callable
        Voltage shift applied to every rate expression. Defaults to ``0 mV``.
    name : str, optional
        Instance name.

    Notes
    -----
    Imported from ``Kv4p3_MA20_GoC.mod`` under
    ``examples/neuron_compare/Cerebellum_mod/GoC/channel/``. Two deviations
    from that source are deliberate and documented in
    ``examples/neuron_compare/Cerebellum_mod/README.md``: the NMODL ``TABLE``
    over ``[-100, 30] mV`` is not reproduced, so rates are evaluated from the
    continuous formulas at every step and no longer clamp outside that range;
    and the gate ODEs are integrated as ``cnexp`` rather than
    ``derivimplicit``, which is exact here because the two gates are
    independent.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Fontana, A., et al. (2001). Theta-frequency
           bursting and resonance in cerebellar granule cells. Journal of
           Neuroscience, 21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Tognolina, M., Narang, U., et al. (2020). Single neuron
           optimization as a basis for accurate biophysical modeling.
           Frontiers in Cellular Neuroscience, 14, 517.
           doi:10.3389/fncel.2020.00517
    """
```

### T3 — Parameter-variant subclass

Applies to the 30-odd classes whose body is only `__module__` plus overrides
(`Cav1p2_MA2025_BC`, `Nav1p6_MA2024_PC`, `Kca3p1_MA2024_PC`, the `*_Frozen`
variants, …). These must define their own docstring — an inherited one fails the
guard and would cite the wrong cell type.

```python
@register_channel("Cav1p2_MA2025_BC")
class Cav1p2_MA2025_BC(Cav1p2_MA2020_GoC):
    r"""Cav1.2 L-type calcium current of the Masoli et al. (2025) basket cell.

    Parameter variant of :class:`Cav1p2_MA2020_GoC`: the gating structure and
    rate expressions are identical, and only the basket-cell parameter values
    differ. See the base class for the full kinetics.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the channel.
    g_max : brainstate.typing.ArrayLike or Callable
        Maximal conductance density. Defaults to the basket-cell value.
    V_sh : brainstate.typing.ArrayLike or Callable
        Voltage shift applied to every rate expression. Defaults to ``0 mV``.
    temp : brainstate.typing.ArrayLike
        Absolute simulation temperature. Defaults to 22 degrees Celsius.
    q10 : brainstate.typing.ArrayLike or Callable
        Temperature coefficient shared by all gates. Defaults to ``1.0``.
    temp_ref : brainstate.typing.ArrayLike
        Reference temperature for ``q10``. Defaults to 22 degrees Celsius.
    name : str, optional
        Instance name.

    See Also
    --------
    Cav1p2_MA2020_GoC : Golgi cell variant this class derives from.

    Notes
    -----
    Imported from ``Cav1p2_MA25_BC.mod`` under
    ``examples/neuron_compare/Cerebellum_mod/BC/channel/``. Rate refresh was
    moved from the NMODL ``BREAKPOINT`` into the derivative evaluation so that
    ``inf``/``tau`` are current before each ``cnexp`` state update.

    References
    ----------
    .. [1] <origin-of-kinetics entry from the bibliography>
    .. [2] <Masoli et al. (2025) basket cell entry from the bibliography>
    """
```

### T4 — `Markov` channel

Applies to `Nav1p6_MA2020_GoC`, `Nav1p1_MA2025_BC`, `Nav_MA2020_GrC`,
`NaFHF_MA2020_GrC`, `Kca2p2_MA2020_GoC`, `Kca1p1_MA2020_GoC`. `Notes` documents
the state graph and the `dependent_state`, not per-gate equations.

```python
@register_channel("Nav1p6_MA2020_GoC")
class Nav1p6_MA2020_GoC(Markov):
    r"""Nav1.6 sodium current of the Masoli et al. (2020) Golgi cell.

    Thirteen-state Markov sodium channel with five closed states, six
    inactivated states, one open state, and one blocked state. Current flows
    only through the open state ``O``.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the channel.
    temp : brainstate.typing.ArrayLike
        Absolute simulation temperature. Defaults to 22 degrees Celsius.
    g_max : brainstate.typing.ArrayLike or Callable
        Maximal conductance density. Defaults to ``16 mS/cm^2``.
    name : str, optional
        Instance name.
    solver : str, optional
        Solver used when this channel is integrated independently.
    substeps : int, optional
        Number of substeps run inside one parent update.

    Notes
    -----
    The transition graph is declared in ``pairs`` as 17 reversible edges over
    the states ``C1``-``C5``, ``I1``-``I6``, ``O``, and ``B``. ``I6`` is the
    ``dependent_state``: its occupancy is recovered from the conservation
    relation rather than integrated, so only twelve states carry ODEs.

    Forward and backward rates are exponential in voltage,
    :math:`f(V) = A \exp(V/x)\,\phi`, with the temperature factor
    :math:`\phi = 3^{(T - 22^\circ\mathrm{C})/10}`. Allosteric coupling uses
    :math:`\mathrm{alfac} = (O_{on}/C_{on})^{1/4}` and
    :math:`\mathrm{btfac} = (O_{off}/C_{off})^{1/4}`. The current is
    :math:`I = \bar{g}\,[O]\,(E_{Na} - V)`.

    References
    ----------
    .. [1] <origin-of-kinetics entry from the bibliography>
    .. [2] <Masoli et al. (2020) entry from the bibliography>
    """
```

### T5 — `KineticIon` calcium pool

Applies to the 14 keyed classes in `ion/calcium.py` (`CdpStC_MA2020_GoC`,
`CdpCAM_MA2024_PC`, the `Toy*_SU2015_DCN` family, …). `Notes` documents the
species table, factors, and conservation, and preserves the existing
unit-conversion comments.

```python
@register_ion("CdpStC_MA2020_GoC")
class CdpStC_MA2020_GoC(Calcium, KineticIon):
    r"""Calcium pool with pump, buffers, and calmodulin of the Masoli (2020) Golgi cell.

    Reaction-network calcium dynamics combining a plasma-membrane pump, two
    generic buffers, the BTC and DMNPE indicator dyes, parvalbumin, and the
    full calmodulin binding cascade. ``Ci`` is the cytosolic free calcium
    concentration and is what the ion protocol exposes to channels.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the ion container.
    temp : brainstate.typing.ArrayLike
        Absolute temperature used by the Nernst equation.
    Co : brainstate.typing.ArrayLike, optional
        Extracellular calcium concentration override.
    valence : brainstate.typing.ArrayLike, optional
        Ionic valence override. Defaults to ``2``.
    solver : str, optional
        Solver used when this ion is integrated independently.
    substeps : int, optional
        Number of substeps run inside one parent update.
    name : str, optional
        Instance name.

    Notes
    -----
    ``Ci`` corresponds to the NMODL calcium pool ``ca``/``cai``. The reversible
    kinetic scheme is preserved as 20 explicit reactions, plus the original
    current-driven source and the single pump conservation relation, so
    ``uses_total_current`` is ``True``.

    Three factors mediate the visible-to-amount mapping: ``cyto`` for cytosolic
    species, ``pump_area`` for the surface pump, and ``cam_unit`` for the
    calmodulin states. The source NMODL uses ``COMPARTMENT (1e10)*parea``
    because ``pump`` and ``pumpca`` are stored visibly in ``mol/cm2`` and
    NEURON needs an explicit area conversion; BrainCell factors already provide
    that mapping, so the extra ``1e10`` is intentionally not carried over.

    Imported from ``CdpStC_MA20_GoC.mod`` under
    ``examples/neuron_compare/Cerebellum_mod/GoC/ion/``.

    References
    ----------
    .. [1] <origin-of-kinetics entry from the bibliography>
    .. [2] <Masoli et al. (2020) entry from the bibliography>
    """
```

### T6 — Ion container

Applies to `Calcium`, `CalciumFixed`, `CalciumInitNernst`, `CalciumFirstOrder`,
`Potassium`, `PotassiumFixed`, `PotassiumInitNernst`, `Sodium`, `SodiumFixed`,
`SodiumInitNernst`, `NonSpecific`, `NonSpecificFixed`. These carry no primary
literature source and go in the guard's `_NO_PRIMARY_SOURCE` allowlist —
`CalciumDetailed` is the exception and cites Destexhe.

```python
class PotassiumFixed(Potassium):
    r"""Potassium container with a fixed reversal potential.

    Holds a constant intracellular concentration and a constant reversal
    potential, so no potassium dynamics are integrated. Use it when channels
    need a potassium reversal potential but the concentration is not itself a
    state variable.

    Parameters
    ----------
    size : brainstate.typing.Size
        Population and compartment shape of the ion container.
    E : brainstate.typing.ArrayLike or Callable
        Reversal potential. Defaults to ``-95 mV``.
    C : brainstate.typing.ArrayLike or Callable
        Intracellular concentration. Defaults to ``0.0407 mM``.
    name : str, optional
        Instance name.

    See Also
    --------
    Potassium : Abstract potassium interface.
    PotassiumInitNernst : Variant whose reversal potential is initialized from
        the Nernst equation.
    """
```

### T7 — Template base, dataclass, or function

Applies to `Gate`, `Transition`, `HH`, `OhmicHH`, `Markov`, `ghk_flux`,
`Factor`, `Species`, `Reaction`, `Source`, `Conserve`, `FixedIon`,
`InitNernstIon`, `DynamicNernstIon`, `KineticIon`. **All fifteen already have
substantive docstrings.** The work is to bring them to canonical section order
and add references to the three that have one:

- `ghk_flux` — Goldman (1943) and Hodgkin & Katz (1949).
- `KineticIon` — Hines & Carnevale's NEURON/NMODL `KINETIC` semantics, which its
  existing `Notes` already invokes by name.
- `CalciumDetailed` (in `ion/calcium.py`, not a template base) — Destexhe et al.

The remaining twelve go in `_NO_PRIMARY_SOURCE`. Do not rewrite their existing
prose where it is already correct; `Gate`'s parameter documentation and
`OhmicHH`'s 40-line docstring are good and only need section-order checks.

---

## Task 1: Provenance harvest and bibliography skeleton

**Files:**
- Create: `docs/design/ion-channel-bibliography.md`

**Interfaces:**
- Produces: `docs/design/ion-channel-bibliography.md` with a complete key→symbol
  inventory and one empty section per citation key. Tasks 2 and 3 fill the
  sections; Tasks 5–17 copy citation text out of them.

- [ ] **Step 1: Generate the authoritative key→symbol inventory**

Run from the worktree root:

```bash
python -c "
import ast, glob, re, collections
KEY = re.compile(r'(?:[A-Za-z]{2}(?:19|20)\d{2}|[A-Z]{2}\d{2}\b)')
rows = collections.defaultdict(list)
for f in sorted(glob.glob('braincell/channel/*.py') + glob.glob('braincell/ion/*.py')):
    if f.endswith('_test.py') or f.endswith('__init__.py'):
        continue
    tree = ast.parse(open(f).read())
    names = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '__all__':
            names = [e.value for e in n.value.elts]
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names:
            k = KEY.findall(n.name)
            rows[k[-1] if k else 'NO_KEY'].append(f'{f}::{n.name}')
for k in sorted(rows, key=lambda k: -len(rows[k])):
    print(f'## {k}  ({len(rows[k])} symbols)')
    for r in rows[k]:
        print(f'   - {r}')
"
```

Expected: 16 key buckets plus `NO_KEY`, totalling 155 symbols.

- [ ] **Step 2: Harvest every `.mod` header**

```bash
for f in $(find examples/neuron_compare/Cerebellum_mod -name '*.mod' \( -path '*channel*' -o -path '*ion*' \) | sort); do
  echo "=== $f"
  sed -n '1,25p' "$f" | grep -iE 'TITLE|COMMENT|Author|Ref|revis|[0-9]{4}' || true
done
```

Expected: 98 files listed; roughly 18 carry an explicit `Author:` line.

- [ ] **Step 3: Write the skeleton**

Create `docs/design/ion-channel-bibliography.md` with `# Ion and channel
bibliography` as its H1, a short preamble stating that docstring `References`
entries are copied verbatim from this file, then one `## <KEY>` section per
bucket containing: the symbol list from Step 1, a `### Provenance evidence`
block with the raw `.mod` header text from Step 2, and empty `### Verified
record` and `### Attribution` blocks for Tasks 2–3. Add an
`## Unresolved attributions` section at the end.

- [ ] **Step 4: Commit**

```bash
git add docs/design/ion-channel-bibliography.md
git commit -m "Add ion/channel bibliography skeleton with mod-file provenance"
```

---

## Task 2: Verify the classical and thalamic citations

**Files:**
- Modify: `docs/design/ion-channel-bibliography.md`

**Interfaces:**
- Consumes: the skeleton from Task 1.
- Produces: filled `### Verified record` and `### Attribution` blocks for
  `HH1952`, `TM1991`, `Ba2002`, `HM1992`, `Ya1989`, `Re1993`, `HP1992`,
  `De1994`, `IS2008` — 19 symbols.

- [ ] **Step 1: Verify each key against the literature**

For each of the nine keys, establish authors, title, journal, volume, issue,
pages, year, and DOI, then confirm the paper actually contains the current
implemented by its symbols. Symbols per key are in the Task 1 inventory. Starting
hypotheses, all of which must be confirmed or corrected, not assumed:

| Key | Hypothesis | Symbols |
|---|---|---|
| `HH1952` | Hodgkin & Huxley (1952), J Physiol | `K_HH1952`, `Na_HH1952` |
| `TM1991` | Traub & Miles (1991), *Neuronal Networks of the Hippocampus* | `K_TM1991`, `Na_TM1991` |
| `Ba2002` | Bazhenov et al. (2002), J Neurosci | `KDR_Ba2002`, `Na_Ba2002` |
| `HM1992` | Huguenard & McCormick (1992), J Neurophysiol | 7 symbols across 3 modules |
| `Ya1989` | Yamada, Koch & Adams (1989), book chapter | `KNI_Ya1989` |
| `Re1993` | Reuveni et al. (1993), J Neurosci | `CaHT_Re1993` |
| `HP1992` | Huguenard & Prince (1992), J Neurosci | `CaT_HP1992` |
| `De1994` | Destexhe et al. (1994) | `AHP_De1994` |
| `IS2008` | unknown; only in-repo evidence is the prose "Strowbridge 2008" | `CaN_IS2008`, `CaL_IS2008` |

`HM1992` is already cited in `channel/hyperpolarization_activated.py:78-80` and
`De1994`-adjacent entries in `ion/calcium.py:227-233`; check those existing
strings and correct them if verification disagrees.

- [ ] **Step 2: Record failures honestly**

Any key that cannot be confirmed goes in `## Unresolved attributions` with the
evidence gathered. `IS2008` is the most likely candidate. Do not invent a
citation to fill the gap.

- [ ] **Step 3: Commit**

```bash
git add docs/design/ion-channel-bibliography.md
git commit -m "Verify classical and thalamic channel citations"
```

---

## Task 3: Verify the cerebellar model citations

**Files:**
- Modify: `docs/design/ion-channel-bibliography.md`

**Interfaces:**
- Consumes: Task 1 skeleton, Task 2 conventions for entry formatting.
- Produces: filled records for `MA2020`, `MA2024`, `MA2025`, `RI2021`, `SU2015`,
  `ZH2019`, `PC24` — 103 symbols, plus origin-of-kinetics entries for every
  `.mod` whose header names a different author.

- [ ] **Step 1: Verify the seven model papers**

Cell-type mapping is fixed by `examples/neuron_compare/Cerebellum_mod/README.md`:

| Key | Cell type | Hypothesis to confirm |
|---|---|---|
| `MA2020` | Golgi (GoC) and granule (GrC) | Masoli et al. (2020) |
| `MA2024` | Purkinje (PC) | Masoli et al. (2024) |
| `MA2025` | basket (BC) | Masoli et al. (2025) |
| `RI2021` | stellate (SC) | Rizza et al. (2021) |
| `SU2015` | deep cerebellar nuclei (DCN) | Sudhakar et al. (2015) |
| `ZH2019` | inferior olive (IO) | Zang et al. (2019) |
| `PC24` | Purkinje, one symbol | same as `MA2024` |

**Verification outcome — four of these hypotheses were wrong.** Recorded here
so the refuted guesses are not mistaken for findings; the authoritative records
are in `docs/design/ion-channel-bibliography.md`.

- `ZH2019` is **Zhang & Santaniello**, PNAS 116(27), 13592-13601 — not Zang &
  De Schutter. ModelDB accession 257028 names "Xu Zhang" as submitter, matching
  the porter credit in the `.mod` headers.
- `RI2021` is Rizza et al., **Scientific Reports** 11(1), 3873 — not
  Communications Biology.
- `SU2015` is Sudhakar, **Torben-Nielsen & De Schutter**, PLOS Computational
  Biology 11(12), e1004641. The author list guessed above belongs to a
  different, 2017 paper.
- `MA2020` needs **two** papers, not one: PLOS Computational Biology 16(12),
  e1007937 for the Golgi cell and Communications Biology 3(1), 222 for the
  granule cell. No single paper covers both cell types.

Step 0's `.mod` harvest also needs a wider window than the first pass used.
Reading only the first 25 lines, and grepping only for keywords like `Author:`
or `Ref:`, missed the DCN attribution "Translated from GENESIS by Johannes
Luthman and Volker Steuber" — which contains none of those keywords and is the
evidence that resolved `SU2015`.

- [ ] **Step 2: Resolve every origin-of-kinetics reference**

For each `.mod` header from Task 1 Step 2 that names an author other than the
model authors, verify that origin paper too and add it as its own bibliography
entry. `Kv4p3_MA20_GoC.mod` (Author: E. D'Angelo, T. Nieus, A. Fontana) is the
known case; find the rest. Record, per symbol, whether it needs one citation or
the two-level pair from template T2.

- [ ] **Step 3: Record the import deviations**

Copy the per-mechanism deviations from
`examples/neuron_compare/Cerebellum_mod/README.md` — `TABLE` removal and its
former clamp range, `derivimplicit`→`cnexp` substitutions, rate-refresh
relocation, NMODL default-precision rewrites — into each key's section, so
module tasks can put them in `Notes` without re-reading that file.

- [ ] **Step 4: Commit**

```bash
git add docs/design/ion-channel-bibliography.md
git commit -m "Verify cerebellar model citations and mod-file provenance"
```

---

## Task 4: Conformance guard

**Files:**
- Create: `braincell/_testing.py`
- Create: `braincell/channel/_docstring_test.py`
- Create: `braincell/ion/_docstring_test.py`

**Interfaces:**
- Produces: `braincell._testing.public_symbols(module)`,
  `own_docstring(obj) -> str | None`, `sections(doc) -> dict[str, str]`,
  `has_citation(doc) -> bool`, and the `DocstringConformanceTests` mixin
  carrying the four assertions. Each guard module exposes a `_COVERED_MODULES`
  tuple that Tasks 5–17 extend by exactly one entry, and a
  `_NO_PRIMARY_SOURCE` frozenset.

The assertions live in the shared mixin rather than being copied into both
guard modules. Each package's `*_test.py` stays co-located per AGENTS.md rule 10
but contains only its coverage tuple, its allowlist, and the class that binds
them to the mixin.

- [ ] **Step 1: Write the shared helper**

Create `braincell/_testing.py`:

```python
"""Test-only helpers for the docstring conformance guards.

Not a test module: the leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import inspect
import re
from types import ModuleType
from typing import Iterator

_SECTION = re.compile(r"^(?P<title>[A-Z][A-Za-z ]{2,})\n(?P<rule>-{3,})[ \t]*$", re.MULTILINE)
_CITATION = re.compile(r"^\s*\.\.\s+\[\d+\]\s+\S", re.MULTILINE)


def public_symbols(module: ModuleType) -> Iterator[tuple[str, object]]:
    """Yield ``(name, obj)`` for every entry in ``module.__all__``."""
    for name in getattr(module, "__all__", ()):
        yield name, getattr(module, name)


def own_docstring(obj) -> str | None:
    """Return the docstring defined on ``obj`` itself, never an inherited one."""
    doc = obj.__dict__.get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        return inspect.cleandoc(doc)
    return None


def sections(doc: str) -> dict[str, str]:
    """Split a NumPy-doc docstring into ``{section title: section body}``."""
    found = {}
    matches = list(_SECTION.finditer(doc))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        found[match.group("title").rstrip()] = doc[match.end():end]
    return found


def has_citation(doc: str) -> bool:
    """True when ``doc`` has a References section holding a ``.. [n]`` entry."""
    body = sections(doc).get("References")
    return bool(body) and bool(_CITATION.search(body))


class DocstringConformanceTests:
    """Assertions shared by the per-package docstring guards.

    Mix into a :class:`unittest.TestCase` subclass that sets
    ``covered_modules`` and ``no_primary_source``. This class is deliberately
    not a ``TestCase`` itself, so it is never collected on its own.
    """

    covered_modules: tuple[ModuleType, ...] = ()
    no_primary_source: frozenset[str] = frozenset()

    def _symbols(self):
        for module in self.covered_modules:
            for name, obj in public_symbols(module):
                yield module.__name__, name, obj

    def test_every_public_symbol_defines_its_own_docstring(self):
        missing = [
            f"{mod}.{name}"
            for mod, name, obj in self._symbols()
            if own_docstring(obj) is None
        ]
        self.assertEqual(missing, [], f"undocumented public symbols: {missing}")

    def test_summary_is_a_single_sentence_line(self):
        bad = []
        for mod, name, obj in self._symbols():
            doc = own_docstring(obj)
            if doc is None:
                continue
            summary = doc.splitlines()[0].strip()
            if not summary.endswith("."):
                bad.append(f"{mod}.{name}: {summary!r}")
        self.assertEqual(bad, [], f"summary must be one sentence ending in '.': {bad}")

    def test_every_public_symbol_cites_a_reference(self):
        uncited = []
        for mod, name, obj in self._symbols():
            if name in self.no_primary_source:
                continue
            doc = own_docstring(obj)
            if doc is None or not has_citation(doc):
                uncited.append(f"{mod}.{name}")
        self.assertEqual(uncited, [], f"missing References with '.. [n]': {uncited}")

    def test_no_primary_source_allowlist_has_no_dead_entries(self):
        if not self.covered_modules:
            self.skipTest("no modules covered yet")
        live = {name for _, name, _ in self._symbols()}
        dead = sorted(n for n in self.no_primary_source if n not in live)
        self.assertEqual(dead, [], f"allowlist names no longer public: {dead}")
```

- [ ] **Step 2: Write the channel guard with an empty coverage tuple**

Create `braincell/channel/_docstring_test.py`:

```python
"""Docstring conformance guard for :mod:`braincell.channel`."""

import unittest

from braincell._testing import DocstringConformanceTests

# Extended by one module per docstring task. A module is listed only once
# every one of its public symbols satisfies the shared assertions.
_COVERED_MODULES = ()

# Public symbols with no primary literature source. Membership must be a
# deliberate decision: a new channel that lands undocumented fails instead of
# silently inheriting an exemption.
_NO_PRIMARY_SOURCE = frozenset()


class ChannelDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES
    no_primary_source = _NO_PRIMARY_SOURCE


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Write the ion guard**

Create `braincell/ion/_docstring_test.py` with the same shape: module docstring
naming `braincell.ion`, its own `_COVERED_MODULES` and `_NO_PRIMARY_SOURCE`, and
`class IonDocstringTest(DocstringConformanceTests, unittest.TestCase)` binding
them. Only the coverage data differs between the two files; the assertions are
inherited from the mixin.

- [ ] **Step 4: Verify both modules collect and pass vacuously**

```bash
pytest braincell/channel/_docstring_test.py braincell/ion/_docstring_test.py -v
```

Expected: 8 tests collected (4 per module) — 6 pass vacuously against the empty
coverage tuple and 2 skip with "no modules covered yet". A result of "no tests
ran" means the filename or class name is wrong; fix it before continuing, since
a guard that collects nothing is the exact failure AGENTS.md rule 10 exists to
prevent. Also confirm `DocstringConformanceTests` is not itself collected.

- [ ] **Step 5: Commit**

```bash
git add braincell/_testing.py braincell/channel/_docstring_test.py braincell/ion/_docstring_test.py
git commit -m "Add docstring conformance guards with empty coverage"
```

---

## Tasks 5-17: Module docstrings

Thirteen tasks, one per module, in this order. Each follows the identical
five-step cycle below.

| Task | Module | Symbols | Templates | Allowlist additions |
|---|---|---|---|---|
| 5 | `channel/potassium_sodium.py` | 1 | T3 | — |
| 6 | `channel/leaky.py` | 2 | T7 | `LeakageChannel`, `IL` |
| 7 | `ion/nonspecific.py` | 2 | T6 | `NonSpecific`, `NonSpecificFixed` |
| 8 | `ion/potassium.py` | 3 | T6 | `Potassium`, `PotassiumFixed`, `PotassiumInitNernst` |
| 9 | `ion/sodium.py` | 3 | T6 | `Sodium`, `SodiumFixed`, `SodiumInitNernst` |
| 10 | `channel/_base.py` | 6 | T7 | `Gate`, `Transition`, `HH`, `OhmicHH`, `Markov` (not `ghk_flux`) |
| 11 | `channel/hyperpolarization_activated.py` | 8 | T1, T2, T3 | — |
| 12 | `ion/_base.py` | 9 | T7 | all but `KineticIon` |
| 13 | `channel/sodium.py` | 14 | T1, T2, T3, T4 | — |
| 14 | `channel/potassium_calcium.py` | 15 | T2, T3, T4 | — |
| 15 | `ion/calcium.py` | 19 | T5, T6 | `Calcium`, `CalciumFixed`, `CalciumInitNernst`, `CalciumFirstOrder` (not `CalciumDetailed`) |
| 16 | `channel/calcium.py` | 35 | T1, T2, T3 | — |
| 17 | `channel/potassium.py` | 38 | T1, T2, T3 | `K_Leak`, `K_Kv_test` |

**The five-step cycle, run once per task.** Substitute the module and the
guard file for that package (`braincell/channel/_docstring_test.py` for channel
modules, `braincell/ion/_docstring_test.py` for ion modules).

- [ ] **Step 1: Extend coverage and the allowlist**

Both `_COVERED_MODULES` and `_NO_PRIMARY_SOURCE` are **cumulative**: append this
task's module and allowlist entries to whatever earlier tasks already put there.
Never replace either collection with a fresh literal — that silently drops a
completed module out of coverage. For Task 5, with Task 6 (`leaky`) already
landed, that is:

```python
from braincell.channel import leaky, potassium_sodium

_COVERED_MODULES = (leaky, potassium_sodium)

# potassium_sodium adds no allowlist entries; leaky's stay.
_NO_PRIMARY_SOURCE = frozenset({
    "LeakageChannel",
    "IL",
})
```

- [ ] **Step 2: Run the guard and record what it does**

```bash
pytest braincell/channel/_docstring_test.py -v
```

For a module with **keyed symbols** (any symbol needing a citation), expect
FAIL on `test_every_public_symbol_cites_a_reference` listing that module's
uncited symbols.

For a module whose symbols are **all allowlisted** the run legitimately passes
here, and that is not evidence the docstrings are adequate. The guard checks
only four things — own docstring present, summary ends in a period, citation
present unless allowlisted, allowlist not dead — so a one-line docstring
satisfies it. Task 6 hit exactly this: both `leaky` symbols passed all four
assertions before any documentation work. Record the actual result and
continue to Step 3 regardless; the guard is a floor, not the acceptance
criterion. Section completeness is the task reviewer's job.

- [ ] **Step 3: Write the docstrings**

Apply the template named in the table to each public symbol. Rules that apply to
every module:

- Transcribe every equation from the code as written, including
  `u.math.where` piecewise branches and the `_linoid_stable` /
  `_x_over_one_minus_exp_neg_stable` guards, not from the paper's closed form.
- Copy `References` text verbatim from `docs/design/ion-channel-bibliography.md`.
  Use the two-level T2 form wherever Task 3 recorded a distinct origin.
- Document every `__init__` parameter, with defaults in user-facing units.
- Name the source `.mod` file and its import deviations in `Notes` for imported
  mechanisms.
- Subclasses get their own docstring; never rely on inheritance.
- For any symbol whose attribution landed in `## Unresolved attributions`, state
  in `Notes` that the source is unconfirmed and add the symbol to
  `_NO_PRIMARY_SOURCE` with an inline comment saying why.

- [ ] **Step 4: Run the guard and the module's own tests**

```bash
pytest braincell/channel/_docstring_test.py braincell/channel/potassium_sodium_test.py -v
```

Expected: all PASS. Modules without a sibling `*_test.py`
(`channel/leaky.py`, `channel/potassium_sodium.py`, `ion/nonspecific.py`) run the
guard plus `pytest braincell/channel/` for that package.

- [ ] **Step 5: Commit**

```bash
git add braincell/channel/potassium_sodium.py braincell/channel/_docstring_test.py
git commit -m "Document braincell.channel.potassium_sodium public API"
```

---

## Task 18: Full verification

**Files:** none modified unless a check fails.

- [ ] **Step 1: Confirm complete coverage**

```bash
python -c "
import braincell.channel as c, braincell.ion as i
from braincell.channel import _docstring_test as ct
from braincell.ion import _docstring_test as it
print('channel modules covered:', len(ct._COVERED_MODULES), 'expect 8')
print('ion modules covered:', len(it._COVERED_MODULES), 'expect 5')
"
```

Expected: `8` and `5`.

- [ ] **Step 2: Run the full suite**

```bash
pytest braincell/ -q
```

Expected: all pass, no new failures against the `ca9ec8c` baseline.

- [ ] **Step 3: Run pre-commit**

```bash
pre-commit run --all
```

Expected: pass.

- [ ] **Step 4: Build the docs and check for new warnings**

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html 2>&1 | tail -40
```

Expected: no warnings referencing `braincell.channel` or `braincell.ion`
docstrings. Duplicate citation labels across docstrings are handled by
per-docstring label mangling; a warning about them means the markup is wrong.

- [ ] **Step 5: Confirm the bibliography and the code agree**

```bash
python -c "
import ast, glob, re
cited = set()
for f in glob.glob('braincell/channel/*.py') + glob.glob('braincell/ion/*.py'):
    if f.endswith('_test.py'):
        continue
    for m in re.finditer(r'^\s*\.\.\s+\[\d+\]\s+(.+)$', open(f).read(), re.MULTILINE):
        cited.add(m.group(1).strip())
bib = open('docs/design/ion-channel-bibliography.md').read()
missing = sorted(c for c in cited if c.split('(')[0].strip() not in bib)
print('citation first-lines absent from bibliography:', missing)
"
```

Expected: an empty list. Any entry printed means a docstring citation was typed
rather than copied.

- [ ] **Step 6: Commit any fixes**

```bash
git add -A
git commit -m "Fix documentation build and bibliography consistency issues"
```

## Plan self-review

Checked against the spec:

- Spec Phase 1 → Tasks 1–3. Phase 2 → Tasks 5–17. Phase 3 → Task 4, moved
  earlier per the ordering refinement recorded in Phase 3.
- All six original decisions plus decision 7 are reflected: full NumPy-doc
  without `Examples` (templates T1–T7), web verification (Tasks 2–3), duplicated
  entries plus central doc (Task 1 + Task 18 Step 5), public API plus template
  bases (T7, Tasks 10 and 12), mismatch policy (Task 2 Step 2 and the Step 3
  rule), references-only guard (Task 4 Step 2), two-level references (T2, Task 3
  Step 2).
- Every spec edge case has a home: `Cav3p1Test_PC24` and `K_Kv_test` in Task 17,
  deprecated aliases excluded because the guard iterates `__all__`, subclass
  chains via `own_docstring`, celsius defaults in Global Constraints, stabilized
  rate functions in the Task 5–17 Step 3 rules.
- Names used consistently throughout: `_COVERED_MODULES`, `_NO_PRIMARY_SOURCE`,
  `public_symbols`, `own_docstring`, `sections`, `has_citation`.
- Symbol counts in the Task 5–17 table sum to 155.
