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

## Phase 1 — Verified bibliography

**Artifact**: `docs/design/ion-channel-bibliography.md`, a durable design note
per `AGENTS.md` rule 8.

**Structure**: one section per citation key, each containing

- the full record: authors, title, journal, volume, issue, page range, year, DOI;
- the exact `.. [1]` reST block to be pasted into docstrings, so the docstring
  text is copied rather than retyped;
- the list of classes citing the key, generated from the AST;
- an attribution note recording what was checked and how confident the result is.

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
