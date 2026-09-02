# `braincell.channel` simplification

Iteration 6 of the module-by-module simplification sweep. Written before any
code was edited; the plan below is what the implementation follows, and the
"Verification" section is filled in from real command output afterwards.

## Scope and method

Target: the whole `braincell/channel/` package — eight production modules and
the eight co-located `*_test.py` files.

The package is a *documented catalogue*. Measured before starting:

| | total lines | docstrings | comments | blank | code |
|---|---|---|---|---|---|
| production (8 files) | 17 006 | 11 670 | 285 | 2 044 | 2 227 |
| tests (8 files) | 6 991 | 34 | 187 | 1 091 | 5 669 |

So the docstrings are the package — 11 670 lines of NumPy-doc carrying
literature references, NMODL provenance, and the reasoning behind constants.
**No docstring is deleted or shortened by this change.** A class that becomes a
subclass keeps its full docstring verbatim; the docstring moves only when the
code it documents moves.

That table also sets the priorities. `potassium.py` is 5 501 lines of which 311
are code; there is no meaningful "shrink `potassium.py`" work. The code volume
is concentrated in `_base.py` (497), `sodium.py` (502), `calcium.py` (773) and,
above all, in the tests (5 669).

Four independent reviews were run over the package (reuse, simplification,
efficiency, altitude). Every load-bearing claim from those reviews was
re-verified here by execution — see "What the reviews got wrong" for the ones
that did not survive.

### Numerical safety net

Before touching anything, a fingerprint harness recorded, for all **113**
concrete channel classes: constructor success, MRO, post-`reset_state` state
values, `current`, `conductance_factor`, `state_values`, every state derivative
after a deterministic off-steady-state perturbation, and the traced-equation
count of `current` and `compute_derivative`. Baseline: 113/113 fingerprinted,
318 non-zero derivative entries, **8 826** traced equations in total.

The stated goal for every production change below is **bit-identical output**.
The fingerprint comparison is what proves it, and any deviation is reported in
"Numerical effect" rather than waved away.

## Bugs fixed

1. **`README.md` quick-start does not run.** Lines 99 and 102 call
   `mech.Channel("INa_Ba2002", ...)` and `mech.Channel("ICaL_IS2008", ...)`.
   Those names are not in the mechanism registry — only the 20 module-level
   `__getattr__` aliases ever carried the old spellings, and the registry was
   never part of that shim:

   ```
   >>> get_registry().get("channel", "INa_Ba2002")
   KeyError: No 'channel' mechanism registered as 'INa_Ba2002'.
             Did you mean ['Na_Ba2002', 'KDR_Ba2002']?
   ```

   Fixed to `"Na_Ba2002"` / `"CaL_IS2008"`.

2. **`sodium_test.py` sets 64-bit precision process-wide at import.**
   `brainstate.environ.set(precision=64)` at module scope changes float
   precision for every test module collected after it, so results depend on
   collection order. Scoped to the tests that need it.

3. **Three vacuous tests.** `calcium_test.py`'s three
   `test_current_uses_frozen_voltage` methods have the body
   `ch = X(size=1); self.assertTrue(callable(ch.current))`. A bound method is
   always callable; the test cannot fail and never checks the frozen voltage
   its name promises. Given the real assertion (the one
   `Cav3p1FrozenMA20GoCTest.test_current_matches_unfrozen_value` already uses).

4. **`Markov.compute_derivative` accumulates a derivative that is discarded.**
   The dependent state's derivative is built up across every transition pair
   and then never read, because only independent states are written back.

5. **`is_disabled` raised `TracerBoolConversionError` under `jit`.** Found by
   the fingerprint harness after the fast path landed, not by the test suite.
   The predicate tested a *concrete* flag with `u.math.any(...)`, but under an
   open trace every jax operation returns a tracer regardless of whether its
   inputs are concrete, so `bool()` on the result raised. That broke
   `current()` under `jit`/`for_loop` -- the execution model AGENTS.md
   mandates -- for all eight gating-current channels (`Kv1p1_*` x5,
   `Kv3p3_MA2024_PC`, `Nav1p1_*` x2). The test suite never caught it because
   nothing traced those channels' `current`. Fixed by running the test in
   numpy, which stages nothing; two regression tests were added first, at the
   predicate level and at the channel level.

## Breaking changes

Every one of these deletes the old spelling outright — no deprecation shim, no
alias, no `warnings.warn` bridge. All in-repo callers are updated in this same
change.

1. **`braincell.channel._DEPRECATED_ALIASES` and the module `__getattr__` are
   deleted.** The 20 pre-normalisation names (`INa_HH1952`, `IKDR_Ba2002`,
   `ICaN_IS2008`, …) no longer resolve; `braincell.channel.INa_HH1952` raises
   `AttributeError`. Verified repo-wide first: every remaining occurrence of
   the 20 names is prose (provenance sentences in docstrings, planning notes in
   `docs/design/`, `TODO.md`) or a tutorial cell that *defines its own*
   `class ICaT_HP1992(...)`. The only code that read the shim was its own three
   tests. `IL` is untouched — that is a real current name, not an old alias.

2. **`Markov` subclasses must declare `dependent_state`.** The implicit
   fallback to "the last state discovered while scanning `pairs`" and its
   `DeprecationWarning` are gone; omitting the declaration is now a
   `ValueError` at class-creation time. All 18 shipped Markov channels already
   declare it (verified by walking `Markov.__subclasses__()` transitively), so
   nothing in the catalogue changes. The old regression test that asserted "no
   shipped channel relies on the fallback" is deleted — the new check *is* that
   guarantee, enforced earlier and unconditionally.

3. **Dead constructor parameters removed.** `V_sh` on `Cav1p2_MA2020_GoC`,
   `Cav1p3_MA2020_GoC`, `Cav3p1_MA2020_GoC` and `Cav3p1_MA2024_PC_Frozen`, and
   `BBiD` on `Kv2p2_0010_MA2020_GrC`. Each is documented in its own docstring
   as "accepted, stored, and never read". All default to a neutral value
   (`0 mV`, `10.0`), and no method or class-level lambda reads them, so removal
   changes no number — it only stops the constructor from silently swallowing a
   value the caller believes is doing something. The `Notes` paragraph
   explaining the mod-file provenance stays.

4. **Class re-parenting.** Several published channels become subclasses of the
   channel they were copy-pasted from. This is a breaking change only in the
   `__mro__` sense — every public name, constructor signature, default and
   registry key is preserved, and the fingerprint proves the numbers are
   unchanged. Any code doing `isinstance`/`issubclass` against the old flat
   hierarchy sees more `True` answers than before, never fewer.

5. **`K_Leak.current` reports a different unit spelling.** `mS*mV/cm^2`
   becomes `10.0^-2 * A/m^2`. The mantissas are bit-identical and the
   dimensions are the same, so arithmetic and `to_decimal` are unaffected —
   but code comparing `u.get_unit(...)` by `repr`, or printing the current,
   sees the new spelling. It is the spelling 99 of the 113 channels already
   use.

6. **`sodium_test.py` no longer sets 64-bit precision process-wide.**
   `sodium_test.py`'s module-scope `brainstate.environ.set(precision=64)` is
   replaced by a module-scoped autouse fixture. Any test module that was
   silently inheriting 64-bit precision because it happened to be collected
   after `sodium_test.py` now runs at the project default; the full suite is
   green either way.

## Altitude

### `GhkHH`: the missing third current template

`HH` is gating-only and `OhmicHH` adds an ohmic driving force, which 63
concrete classes take unchanged. The 12 constant-field (GHK) calcium channels
had no such template, so they carried **6 hand-written `current()` bodies**
that vary along four orthogonal axes:

| classes | GHK helper | voltage shift | frozen | scale |
|---|---|---|---|---|
| `Cav2p1_{MA2024_PC, MA2025_BC, RI2021_SC}` | `ghk_flux` | yes | no | `g_max` |
| `Cav2p1_*_Frozen` (3) | `_cav3p1_nmodl_ghk_flux` | yes | yes | `g_max` |
| `Cav3p1_{MA2020_GoC, MA2024_PC}` | `_cav3p1_nmodl_ghk_flux` | no | no | `g_max` |
| `Cav3p1_*_Frozen` (2) | `_cav3p1_nmodl_ghk_flux` | no | yes | `g_max` |
| `Cav3p3_{MA2024_PC, RI2021_SC}` | `_cav3p3_nmodl_ghk_flux` | yes | no | `g_scale * perm` |
| `Cav3p3_MA2024_PC_Frozen` | `_cav3p3_nmodl_ghk_flux` | yes | yes | `g_scale * perm` |

Rows 3 and 4 are the same cell of that table written twice — the tell that this
is a product of independent axes, not six distinct behaviours.

`GhkHH(HH)` lands in `_base.py` with the four axes as declarations:

```python
class GhkHH(HH):
    ghk = staticmethod(ghk_flux)
    freeze_drive_gradient: ClassVar[bool] = False

    def permeability(self):
        return self.g_max

    def current(self, V, ion):
        drive_V = freeze_gradient(V) if type(self).freeze_drive_gradient else V
        drive = type(self).ghk(
            V=self._shifted_voltage(drive_V), ci=ion.Ci, co=ion.Co,
            z=self.z, temp=self.temp,
        )
        return -self.permeability() * self.conductance_factor(V, ion) * drive
```

`HH._shifted_voltage` gains a default of `return V`, so the classes that do not
shift declare nothing and the four that do keep their existing two-line
override. The expression tree is identical to every one of the six bodies it
replaces, so the result is bit-identical.

**The freeze is applied to `V` before the shift, never after.** Forward values
agree either way, but `stop_gradient(V) - V_sh` leaves a live gradient path to
`V_sh` that `stop_gradient(V - V_sh)` would kill. The template preserves the
existing order.

### `OhmicMarkov`: the missing sibling of `OhmicHH`

`Markov` is documented as the sibling of `HH`, but `HH` got `OhmicHH` and
`Markov` never got its counterpart — so 16 of the 18 Markov classes reach the
identical ohmic law through three hand-written `current()` bodies, and `Markov`
consequently has no `reversal_potential()` hook at all.

The ohmic current expression is extracted from `OhmicHH` into a private
`_OhmicCurrent` mixin holding `reversal_potential` and `current`. Then
`OhmicHH = _OhmicCurrent + HH` (unchanged behaviour) and `OhmicMarkov =
_OhmicCurrent + Markov`, where `conductance_factor` sums the states named by a
new `open_states` declaration. Left-to-right summation in declaration order
reproduces the existing parenthesisation exactly.

The two `Nav1p1_*` classes add a gating current on top; they keep their own
`current()` and take the conductive term from `super().current(...)`.

### Fixed reversal potentials stop being five copies of one override

Five channels overrode `reversal_potential` with the identical two-line body
`return self.E`. Four of them (`HCN_HM1992`, `HCN1_MA2025_BC`,
`HCN_SU2015_DCN`, `HCN_ZH2019_IO`) do so for one structural reason: their
`root_type` is `HHTypedNeuron`, so `current` is called with no ions at all and
`ions[0].E` would raise `IndexError`. That is a property of the *template*, not
of the four classes, so it moves into the template:

```python
return ions[0].E if ions else self.E
```

`ions` is a Python tuple, so the branch resolves at trace time and costs
nothing. The four overrides are deleted. The fifth, `CaL_SU2015_DCN`, keeps
its override: it *does* receive a calcium ion and deliberately ignores its
reversal potential, which is a real per-class decision and not the same fact.

An earlier draft added a `_FixedReversal` mixin for this. Generalising the
template is the better altitude — one line instead of a class, no new name in
the MRO, and a channel that forgets to set `self.E` now gets an
`AttributeError` naming the attribute instead of an `IndexError` on a tuple.

This is also cheaper: the five `Kca1p1_*` classes currently call
`state_values()`, which reconstructs the dependent state (`conserve` minus a
nine-term sum) purely to throw it away. `OhmicMarkov.conductance_factor` reads
the independent states directly and only falls back to `state_values()` if a
declared open state *is* the dependent one — which no shipped scheme does.

### Freezing becomes a declaration, not a copy-pasted class

Three `*_Frozen` calcium classes are standalone `HH` subclasses that re-declare
an entire published channel — constructor, defaults, gates, and every rate
method — so that one line of `current()` can differ. Verified by MRO-resolved
method comparison:

```
Cav3p1_MA2024_PC_Frozen vs Cav3p1_MA2024_PC : differing = ['current']
Cav3p3_MA2024_PC_Frozen vs Cav3p3_MA2024_PC : differing = ['current']
Cav2p1_MA2024_PC_Frozen vs Cav2p1_MA2024_PC : differing = ['current']
```

`current` is the *only* difference in all three cases. The fourth frozen class,
`Cav3p1_MA2020_GoC_Frozen`, already does it correctly as a three-line subclass —
and `calcium.py`'s own docstring names the defect: *"The two frozen variants
therefore compute the same numbers while sharing no code, and a change to
`Cav3p1_MA2020_GoC` propagates to one of them and not to the other."*

With `GhkHH` in place all four collapse to the pure-alias form the rest of the
catalogue already uses — `class Cav3p1_MA2024_PC_Frozen(Cav3p1_MA2024_PC):
freeze_drive_gradient = True` — carrying their docstrings unchanged.

`Cav2p1_MA2024_PC_Frozen` additionally declares `ghk = staticmethod(
_cav3p1_nmodl_ghk_flux)`, because it deliberately uses different physical
constants from its unfrozen parent. That divergence is real and published —
measured at up to **5.4e-4** relative over V ∈ [-100, 60] mV — so it must
survive re-parenting as an explicit declaration rather than be inherited away.

### Registry keys stop being a second source of truth by convention alone

`@register_channel("Name")` repeats the class name at 112 sites in this package
(138 repo-wide) and there is no case where the two differ. Making the argument
optional is a `braincell/mech` change and belongs to a later iteration; what
lands here is the guard that closes the drift hole today — a package-scope test
asserting that every channel registered from `braincell.channel` is registered
under its own `__name__`.

## Shared declaration tables and reuse

### `_base.py`: one steady-state solver instead of two

`Markov._solve_steady_state_jax` (75 lines) and `_solve_steady_state_host`
(79 lines) are line-for-line parallel; 48 lines are identical, including a
13-line rate-resolution loop that is identical down to its comment. Every
difference is mechanical (`jnp` vs `np`, functional scatter vs in-place, one
`jax.device_get`).

The backend-independent prologue — flattening `conserve` and resolving every
transition rate — is hoisted into one shared helper. Each solver keeps only its
own generator assembly and validation.

The `try/except (ConcretizationTypeError, TracerArrayConversionError)`
dispatch in `_solve_steady_state` **stays**, and the shared helper is called
only after the host path's existing `jax.device_get(template)` guard, so the
abort point does not move. See "What the reviews got wrong" — the case for
replacing the dispatch did not survive measurement.

### Sibling channels that were copy-pasted become subclasses

Verified pairwise by comparing constructor signatures, defaults, `gates`,
`root_type`, and the source of every method resolved through the MRO:

| becomes a subclass of | entire difference |
|---|---|
| `CaHT_HM1992(CaT_HM1992)` | `V_sh` default (`25.0` vs `-3.0` mV) |
| `KA2_HM1992(KA1_HM1992)` | `g_max` default + `f_p_inf` |
| `KK2B_HM1992(KK2A_HM1992)` | `f_q_tau` |
| `sKdr_SU2015_DCN(fKdr_SU2015_DCN)` | `f_m_inf`, `f_m_tau` |
| `NaFHF_MA2020_GrC(Nav_MA2020_GrC)` | 2 constants, 16 extra rate methods, longer `pairs` |
| `Cav3p1_MA2024_PC_Frozen(Cav3p1_MA2024_PC)` | `freeze_drive_gradient` |
| `Cav3p3_MA2024_PC_Frozen(Cav3p3_MA2024_PC)` | `freeze_drive_gradient` |
| `Cav2p1_*_Frozen(_FrozenCav3p1Ghk, Cav2p1_*)` | `freeze_drive_gradient` + `ghk` |
| `K_Leak(OhmicHH)` | `gates = ()`, `g_max`/`E` defaults |

Every retained method is moved verbatim, so each is bit-identical. The 51
members `NaFHF_MA2020_GrC` now inherits were confirmed byte-identical to their
`Nav_MA2020_GrC` counterparts by AST comparison before deletion, not by eye.

Three candidates on the plan did **not** survive measurement and were solved a
different way instead:

- **`Kv3p3_MA2024_PC(Kv1p1_MA2025_BC)`** -- rejected on altitude. Kv3.3 is a
  different protein from Kv1.1 and eight rate constants differ, so the
  inheritance would assert a relationship that is not there. What the two
  actually share is the *gating-current current law*, which moved to an
  `_ExpRateGateCurrent` mixin that both mix in alongside `HH`.
- **`HCN2_MA2020_GoC(HCN1_MA2020_GoC)`** -- same objection: two isoforms,
  neither a specialisation of the other. Both now derive from a private
  `_FastSlowHCN(HH)` template carrying the gates, the constructor and all six
  shared formulas; each isoform declares its eleven kinetic constants (moved
  from `__init__` to class attributes, since neither `.mod` exposes them as
  `RANGE` variables) and its own `r(V)`.
- **`Cav3p1Test_PC24(Cav3p1_MA2020_GoC)`** -- not viable. `g_max` carries
  different *dimensions* in the two classes (`S/cm^2` vs `cm/s`) and all four
  rate functions differ.

`K_Leak` is the one re-parenting that is not bit-identical, and only in unit
spelling: `OhmicHH.current` multiplies one more factor than the two-factor body
it replaces, so `array * Unit` becomes `array * Quantity` and brainunit
canonicalises `mS*mV/cm^2` to `10.0^-2 * A/m^2`. The mantissas are bit-identical
(see "Numerical effect"), and the new spelling is the one 99 of the 113 channels
already report.

`NaFHF_MA2020_GrC` re-declares `pairs` in full rather than appending to its
parent's tuple. `pairs` order determines `_resolved_state_names` order, which
drives both the generator-matrix column permutation in the steady-state solve
and the per-pair accumulation order in `compute_derivative`; appending would be
a last-ulp change to a published model.

### `braincell/channel/_testing.py`

`braincell/channel` is the only tested package in the repo with no
package-local `_testing.py`, and it pays for it: `_V` is defined six times,
`_DENSITY_UNIT` six times, `_k_info` four times, `_ca_info` three, `_na_info`
twice — 21 helper definitions backing 700+ call sites.

The copies have **drifted**, and two of them are physically wrong:

| helper | copies | divergence |
|---|---|---|
| `_V` | 6 | byte-identical, and byte-identical to `braincell.ion._testing.V` |
| `_DENSITY_UNIT` | 6 | byte-identical |
| `_k_info` | 4 | three use `Ci = 0.04 mM`, one uses `140.0 mM` |
| `_na_info` | 2 | `0.04 mM` vs `10.0 mM` |
| `_ca_info` | 3 | three different `Ci` defaults; one carries a dead `e_mV` alias |

`0.04 mM` is the resting *calcium* concentration; as an intracellular sodium or
potassium concentration it is unphysical and was copy-pasted in. The new
`_testing.py` exports `voltage` (re-exported from `braincell.ion._testing.V`,
following the `vis/_testing.py` → `io/_testing.py` precedent AGENTS.md
documents, and spelled out because every test body binds a local `V`),
`DENSITY_UNIT`, `k_info`, `na_info`, `ca_info` and `nonspecific_info` with the
concentrations as keyword arguments.

**No existing test's numbers change.** The two modules that relied on an
outlier value keep it, as a two-line documented wrapper naming the reason:
`potassium_calcium_test._k_info` pins `Ci = 140 mM` and
`potassium_sodium_test._na_info` pins `Ci = 10 mM`. The point of consolidating
is that the outlier is now a visible, justified override instead of a silent
per-file default.

`_testing.py` also carries `assert_channels_agree(case, expected, actual, V,
*ions, states=(...))` — reset both channels, compare every named state, its
derivative, and the current density. That is the entire body of the
"cell-type variant matches its sibling" test, which this package had written
out by hand 24 times.

### Test bodies become tables

The four largest test files are 4 900 lines of code, much of it one assertion
body copy-pasted across sibling variant classes. The idiom for fixing it
already exists in the package —
`potassium_calcium_test.py`'s `KcaInheritedCellVariantTest` drives a
`VARIANTS = ((name, cls, base), ...)` tuple through `subTest`, and
`potassium_test.py` already has `_P4HHMixin` / `_P4QHHMixin` / `_PQHHMixin`.

Applied to `potassium_test.py`'s 17 single-purpose variant classes (21 test
bodies over five shapes) and `hyperpolarization_activated_test.py`'s three HCN1
variant classes, which become `_HCN1VariantTests` plus a
`_HCN1DerivedVariantTests` subclass carrying the one assertion that only the
two derived variants can make. `sodium_test.py` already had `_HHNaMixin`,
`_Nav1p6Mixin` and `_Nav1p1Mixin`; its two remaining standalone classes
(`NavMA20GrCTest`, `NaFHFMA20GrCTest`) are left alone because the two channels
now differ structurally, not just in constants.

`potassium_test.py` loses 480 net lines. The rewrite was done mechanically:
each candidate body was parsed, matched against the exact expected statement
sequence, and skipped rather than guessed at if anything failed to match — four
of the 25 `test_matches_*` bodies did not match and were left untouched.

## Efficiency

**Every win here is a tracing win, and none is claimed as a runtime win.** That
distinction was tested, not assumed. XLA's CSE and algebraic simplifier remove
the duplicates before execution: `Nav1p6_MA2020_GoC.compute_derivative` emits 22
`exp` in its jaxpr and **5** `exponential` in the optimized HLO, and
hand-deduplicating the rate table down to 8 jaxpr `exp` still gives **5**. For
the structural rewrites below the optimized-HLO op inventory is byte-identical
(`multiply` 111/111, `add` 43/43). A 1000-step interleaved runtime A/B gave
+11.1%, +1.0%, -10.4% across three channels — noise, and nothing is claimed
from it. There is also no host-device transfer on any hot path; all
`jax.device_get` use is confined to the reset-time steady-state solve.

What that buys is model-build and compile latency, which is what a user of this
package actually waits on.

| # | change | measured |
|---|---|---|
| 1 | `Markov.make_integration` short-circuits the `for_loop` when `substeps == 1` | trace **1187 -> 964 ms** across all 18 Markov classes (**-18.7%**), every class improved 15.0-23.2% |
| 2 | `q10_factor` results memoised per instance instead of recomputed per rate call | `Kca1p1_MA2020_GoC` 24.17 -> 18.68 ms (-22.7%), `Kca2p2_RI2021_SC` 10.40 -> 6.50 ms (-37.5%); ~41 ms over the 10 Kca classes, ~18 ms over the 92 HH channels via `gate_phi` |
| 3 | gating-current `where` skipped when `gateCurrent` is concrete and uniformly zero | `current()` trace **3.13 -> 1.22 ms (-61%)**, jaxpr **25 -> 5 eqns (-80%)** |
| 4 | `Markov.compute_derivative` binds `states[src] * forward` once instead of twice | trace 283 -> 261 ms across 18 classes (-7.8%); jaxpr `Nav1p6` 339 -> 283 (-17%), `NaFHF` 408 -> 339 |
| 5 | `Markov.compute_derivative` stops accumulating the dependent state's derivative | ~26 traced ops built and discarded per step for a 13-pair `Kca1p1_*` |
| 6 | accumulators start from the first term instead of `_state_zero()` | `_state_zero()` costs 66.8 us eager; 188 calls across the Markov catalogue ~ 12.6 ms |
| 7 | `OhmicMarkov.conductance_factor` reads independent open states directly | drops the dependent-state reconstruction (`conserve` minus a nine-term sum) from every `Kca1p1_*` `current()` |
| 8 | `HH.conductance_factor` starts the product from the first gate, not `1.0` | one wasted `mul` per HH channel per trace, 92 across the catalogue |
| 9 | `_INVERSE_TIME_DIM` hoisted to module scope; `_as_gate_quantity` builds `expected` lazily | `u.get_dim(1 / u.ms)` costs 14.9 us vs 0.02 us cached, x318 per full-catalogue trace ~ 4.8 ms |

Change 3 preserves the documented contract that "``gateCurrent`` may be an
array and the choice stays traceable": the fast path is taken only when the
flag is concrete *and* uniformly zero, otherwise the `u.math.where` runs
unchanged.

Changes 2 and 6 need care about staleness. `braincell._compute.bindings`
rebinds channel parameters at runtime via `setattr` and then calls
`_on_param_updated`. `Nav1p6_MA2020_GoC` already precomputes `self.phi` in
`__init__` and would therefore go stale if `temp` were written at runtime — a
latent bug nobody has hit. The memo added here is keyed on the *identity* of
the values it was computed from, so a rebind invalidates it automatically; that
is the same self-healing cache shape used for the radial-shell geometry factors
in `braincell/ion/_base.py`.

Baseline for comparison: **8 826** traced equations across the 113 classes;
per-model trace / first-`jit` / steady-state step timings recorded for 19
representative channels.

## Numerical effect

The fingerprint was re-run after every structural change and compared against
the baseline. Over the 113 classes and the six numeric fields per class
(`states`, `current`, `conductance_factor`, `state_values`,
`current_perturbed`, `derivatives`):

**Every numeric value is bit-identical.** The comparison reports 21 changed
entries, and all 21 are accounted for:

| entries | what changed | why |
|---|---|---|
| 19 | `conductance_factor` newly present on the `Kca2p2_*`, `Kca1p1_*`, `Nav1p6_*`, `Nav1p1_*`, `Nav_MA2020_GrC`, `NaFHF_MA2020_GrC` classes | plain `Markov` had no such method; `OhmicMarkov` adds it. Nothing changed value — the field went from absent to present |
| 2 | `K_Leak.current` and `K_Leak.current_perturbed` unit spelling, `mS * mV / cm^2` → `10.0^-2 * A / m^2` | `K_Leak` moved to `OhmicHH`, adding a third factor to the product; the mantissas are bit-identical (`-0.09999999403953552`, `-0.17499999701976776`, `-0.3499999940395355` before and after) |

No `current`, `current_perturbed`, `states`, `state_values` or `derivatives`
value changed on any of the 113 classes.

The unit-spelling change is a real, if cosmetic, breaking change and is listed
as one. It also *reduces* the catalogue's spelling divergence: the baseline
reported currents in four different spellings (99 x `10.0^-2 * A/m^2`,
9 x `10.0^1 * A/m^2`, 2 x `mA/cm^2`, 2 x `mS*mV/cm^2`), and `K_Leak` moves from
a two-member outlier to the 99-member majority. `IL` is now the last
`mS*mV/cm^2` channel; normalising all four spellings is recorded for the
whole-package iteration.

## Deliberately not changed

Each of these was investigated and rejected; the reason is recorded so the next
sweep does not re-litigate it.

1. **The three GHK helpers are not merged.** `ghk_flux`,
   `_cav3p1_nmodl_ghk_flux` and `_cav3p3_nmodl_ghk_flux` use different Faraday
   and gas constants and different Kelvin offsets, taken verbatim from
   `Cav3p1_MA20_GoC.mod` and `Cav3p3_RI21_SC.mod`. Measured divergence over
   V ∈ [-100, 60] mV: shared vs cav3p1 **5.36e-4**, shared vs cav3p3
   **1.60e-3**, cav3p1 vs cav3p3 **2.14e-3**. The constants are load-bearing
   and the classes are validated against NEURON. What `GhkHH` fixes is that the
   *choice* of helper is now a declaration instead of a copy-pasted call.

   Separately: `_cav3p3_nmodl_ghk_flux` writes the flux in the reciprocal
   algebraic direction (`co - ci·e^{+w}`, negated). Substituting its own
   constants into the Goldman form agrees to **8.7e-16** relative — it is the
   same equation transcribed in its mod file's idiom, not a different one. That
   is exactly why it must not be folded into the shared body: the merge would
   buy three lines and cost bit-for-bit invariance.

2. **`CaHVA_SU2015_DCN` / `CaLVA_SU2015_DCN` do not join `GhkHH`.** They inline
   a GHK block with hardcoded `4.47814e6` / `-23.20764929` constants and,
   unlike all three helpers, **no singularity guard** — `drive = … / (1.0 - A)`
   is a bare 0/0 at V = 0. Adding the guard is a behaviour change to a
   published model, so it is a separate decision from this refactor. Recorded
   as a known defect.

   The conversion was implemented and then **reverted**, because it changed the
   reported unit. `GhkHH.current` multiplies three operands where these two
   bodies multiply the drive by a bare `u.mA / u.cm**2` at the end;
   `array * Unit` attaches the unit verbatim while `array * Quantity` goes
   through unit arithmetic and canonicalises, so `mA/cm^2` became
   `10.0^1 * A/m^2` with bit-identical mantissas. These are the only two
   channels in the catalogue reporting `mA/cm^2`, and silently re-spelling a
   published model's output unit is not a simplification. The duplication is
   removed a different way: the shared constant-field block is extracted into
   `_su2015_dcn_ghk_drive(V, ci, co, temp)`, which returns a bare magnitude, so
   each class keeps its own one-line `current` and its own `array * Unit`.

3. **The declarative parameter table.** `self.X = braintools.init.param(X,
   self.varshape, allow_none=False)` appears **241 times** in this package,
   character-identical modulo the attribute name, and 414 times repo-wide.
   `Synapse.__init__` in `braincell/_base_channel.py` already contains exactly
   that loop. Lifting it onto `IonChannel` would remove ~175 lines here and
   ~300 repo-wide — but `IonChannel` is a root module and `braincell/ion`
   carries 169 of the remaining sites, so this belongs to the root-module and
   whole-package iterations, not to a per-package one.

4. **`Markov` gains no temperature declaration.** `phi` is hand-applied at ~70
   Markov rate sites (34 in `Nav1p6_MA2020_GoC` alone), which `Gate`-style
   `q10` / `temp_ref` on `Markov` would collapse into one multiplication inside
   `_transition_rate`. But only the sites that write `phi` as the *last* factor
   migrate bit-identically (60 of 70); `Kca2p2_*` puts it mid-expression and
   `Nav_MA2020_GrC` / `NaFHF_MA2020_GrC` put it first. A partial migration
   leaves the mechanism half-adopted, which is worse than either end state.

5. **`register_channel`'s name argument stays mandatory.** Making it default to
   `cls.__name__` is a `braincell/mech` change touching 138 sites across three
   packages. Deferred to the whole-package iteration; the drift guard added
   here is the interim.

6. **`OhmicHH` gains no `conductance_scale` hook.** Four classes insert a single
   extra factor into the ohmic product. A hook would add one traced multiply to
   all 63 classes that use `OhmicHH.current` unchanged, and overriding
   `conductance_factor` instead reassociates the product. Neither trade is
   worth eight lines.

7. **`LeakageChannel` is not re-based on `OhmicHH`.** `OhmicHH` with
   `gates = ()` is bit-identical to `IL` (verified: `x * 1.0` is exact in
   IEEE-754), so the leak hierarchy is arguably redundant. But
   `LeakageChannel` is a public, documented extension point whose contract is
   "`current` raises until you implement it", it is imported by
   `docs/tutorials/channel.ipynb`, and `leaky_test.py` pins that contract.
   What lands here is the subset with no contract change: five of
   `LeakageChannel`'s six lifecycle overrides merely repeat `IonChannel`'s own
   no-ops and are deleted, leaving `root_type` and the one `compute_derivative`
   no-op that is the class's actual content. `K_Leak` is a separate case and
   *does* move to `OhmicHH` with `gates = ()`: it was never a `LeakageChannel`
   (its `root_type` is `Potassium` and its lifecycle methods take `(V, K)`,
   not `(V)`), so it duplicated the no-ops without being able to inherit them.
   `IL` stays on `LeakageChannel`.

8. **The 65 inline Boltzmann expressions stay inline.** They resolve to 47
   distinct expressions, varying in sign convention and in whether the slope
   divides or multiplies. A `_boltzmann(V, v_half, k)` helper would turn each
   site into an argument list that no longer visually matches the `.. math::`
   block directly above it — and for a transcribed `.mod` file, that literal
   correspondence *is* the provenance evidence.

9. **`potassium.py`'s `_sigm` is neither promoted nor deleted.** It has three
   call sites and is named in five docstrings; removing it means editing those
   docstrings for a two-line function, and promoting it runs into item 8.

10. **`_rate_ion_count`'s `*args` branch is not dead.** It has real users:
    `_base_test.py`'s `test_gate_method_with_varargs_receives_every_ion` and
    eight methods in
    `examples/single_compartment/SC07_Straital_beta_oscillation_2011.py`.
    Removing it would silently change the documented semantics of
    `f_x_inf(self, V, *ions)` from "receive every ion" to "receive none".

11. **`calcium_test.py`'s `_cav3p1_nmodl_ghk_flux` is not deduplicated against
    the production helper.** It inlines `9.6485e4`, `8.3145` and `0.04 K` as
    literals precisely so that
    `test_current_uses_nmodl_constants_not_generic_ghk` and
    `test_current_matches_ghk_formula` are independent oracles. Importing the
    production constants would make both tautological.

12. **`__init__.py` keeps its explicit import block.** Replacing 143 lines of
    `from .calcium import (...)` with star-imports is exactly equivalent —
    every submodule defines `__all__` — but costs IDE navigation, mypy
    resolution and Sphinx discovery across a 130-name public API, and the
    existing `ChannelReExportTest` already closes the drift hole. What does
    change is that `__all__` is built from the names actually imported rather
    than from a second concatenation of the submodules' `__all__`, so one of
    the two drift guards becomes true by construction.

13. **The rate tables are not hand-CSE'd.** 1 987 of the 7 129 traced equations
    the catalogue emits (28%) are exact duplicates — 179 of 408 in
    `NaFHF_MA2020_GrC` alone. A hand-written shared-exponential rewrite of
    `Nav1p6_MA2020_GoC` cut its jaxpr 339 -> 274 (-19%) and its trace time
    16.86 -> 14.52 ms (-14%) with bit-identical derivatives — but the optimized
    HLO kept **5** `exponential` either way and compile time did not move
    (127.9 -> 133.7 ms, noise). The price is rewriting rate tables that are
    deliberately transcribed line-by-line from published `.mod` sources, which
    is exactly the literal correspondence item 8 argues is worth keeping. The
    nine changes in "Efficiency" get most of the same trace-time benefit from
    ~20 lines in `_base.py`.

14. **The dimension checks stay on the per-evaluation path.** `_as_markov_rate`
    and `_check_derivative` look like a cost paid every step forever. They are
    not: under this repo's mandated `brainstate.transform` execution model the
    body is traced once. Instrumented over `for_loop` rollouts of 50 and 500
    steps, the call counts were identical (`_check_derivative` 5,
    `_as_gate_quantity` 8, `_rate_ion_count` 10) — they are Python-level
    operations on unit metadata that emit no XLA ops. Moving them earlier would
    lose the ability to catch a dimensioned `phi` or `conserve`, which are read
    off the instance and cannot be checked at class-creation time.

## What the reviews got wrong

- One review reported that `Cav2p1_MA2024_PC_Frozen` could simply inherit its
  parent's `current()`. It cannot: the frozen class calls
  `_cav3p1_nmodl_ghk_flux` where the unfrozen one calls the shared `ghk_flux`,
  a deliberate and documented difference measured at up to 5.4e-4 relative.
  Acting on it would have silently changed a published model by 0.05%. The
  helper choice is carried through as an explicit class-level declaration
  instead.

- One review recommended keeping `_DEPRECATED_ALIASES` on the grounds that it
  maps Python attribute names, a different namespace from the mech registry's
  aliases. That is true but does not save it: the shim has zero users, and its
  half-coverage is precisely what let `README.md` ship a broken example for the
  registry path while the attribute path kept working.

- One review's first diff of the frozen classes compared only methods defined
  in each class *body*, which made the parents look like they defined nothing.
  Re-run with MRO resolution, the result is the much stronger claim used above:
  `current` is the sole difference in all three cases.

- One review argued the `try/except` dispatch in `_solve_steady_state`
  evaluates every rate method twice under `jit` — once staged into a jaxpr that
  is thrown away when `device_get` raises, then again for real — and should be
  replaced with an explicit `is_traced_value` check. **Measured false.**
  Instrumenting `_transition_rate` shows a traced reset makes exactly **34**
  calls for `Nav1p6_MA2020_GoC`, the same as a concrete reset: the host path
  aborts at `jax.device_get(template)` *before* the rate loop begins. The
  dispatch is also not a loser on speed — the host solve is **15.5 ms vs
  39.0 ms** for the JAX path on `Kca1p1`, 2.4x faster, which is the reason it
  exists. This was on the way into the implementation plan before the
  measurement arrived; the plan now keeps the dispatch and dedupes only the
  shared prologue, and the shared helper is invoked *after* the existing
  `device_get` guard so the abort point does not move.

  The residual objection is real but small and is left alone: a
  `ConcretizationTypeError` raised from inside a user's own rate function
  (`if V > 0: ...`) is still swallowed and the whole solve retried on the JAX
  path, where it will raise again.

## Tests

- New `braincell/channel/_testing.py` holding the shared fixtures
  (`voltage`, `DENSITY_UNIT`, four `*_info` builders) and
  `assert_channels_agree`.
- New tests for `GhkHH` and `OhmicMarkov` in `_base_test.py`: axis-by-axis
  coverage of `ghk`, `_shifted_voltage`, `freeze_drive_gradient` and
  `permeability`, and of `open_states` including the dependent-open-state
  fallback that no shipped scheme exercises.
- New test that `Markov.__init_subclass__` rejects a subclass declaring `pairs`
  without `dependent_state`.
- New package-scope guard in `__init___test.py` that every channel registered
  from this package uses its own class name as the registry key.
- Deleted: the three `_DEPRECATED_ALIASES` tests, the `dependent_state`
  fallback tests, the "no shipped channel relies on the fallback" test (now
  enforced structurally), and one `_base_test.py` clip test that is a
  byte-identical prefix of the test immediately below it.
- Rewritten: three vacuous `test_current_uses_frozen_voltage` bodies, replaced
  by `_FrozenVariantTests`, which asserts what the names promised — that the
  gradient through the GHK drive is actually zero, via `jax.grad`.
- New: two regression tests for the `is_disabled` tracing bug, one at the
  predicate level (`make_jaxpr` around a concrete flag) and one at the channel
  level (`jax.make_jaxpr(Kv1p1_MA2025_BC.current)`).
- New: `_HCN1VariantTests` / `_HCN1DerivedVariantTests`, replacing three
  hand-written HCN1 variant test classes.

## Docs

- `README.md`: the two broken `mech.Channel(...)` names.
- `docs/apis/braincell.channel.rst`: a new "Channel Templates" section listing
  `Gate`, `Transition`, `HH`, `OhmicHH`, `GhkHH`, `Markov`, `OhmicMarkov`,
  `ghk_flux`, `q10_factor` and `freeze_gradient`, plus the four `*_Frozen`
  classes that were in `__all__` but undocumented.
- `docs/design/cell.md`: two more `Channel("INa_HH1952")` occurrences of the
  same broken-name bug as the README.

## Verification

### Size

Line counts are a poor proxy in a package that is 70% docstring, so both are
reported. "code" excludes docstrings, comments and blank lines.

| | total | docstring | code |
|---|---|---|---|
| production, before | 17 006 | 11 741 | 4 022 |
| production, after | 17 145 | 12 078 | **3 772** (−250, −6.2%) |
| tests, before | 7 019 | 34 | 5 665 |
| tests, after | 6 879 | 88 | **5 492** (−173, −3.1%) |

Docstrings grew by 391 lines: no existing docstring was deleted or shortened,
and the new templates (`GhkHH`, `OhmicMarkov`, `_FastSlowHCN`,
`_ExpRateGateCurrent`, `_testing.py`) are documented to the same standard as
the catalogue they serve.

### Numerics

```
$ PYTHONPATH=$PWD python /tmp/chan_fingerprint.py /tmp/chan_final.json
wrote /tmp/chan_final.json: 113 classes, 113 fully fingerprinted

$ python /tmp/chan_diff.py /tmp/chan_before.json /tmp/chan_final.json
NUMERIC DIFFS: 21
   K_Leak current
     before: ['mS * mV / cm^2', [-0.09999999403953552, -0.17499999701976776, -0.3499999940395355]]
     after : ['10.0^-2 * A / m^2', [-0.09999999403953552, -0.17499999701976776, -0.3499999940395355]]
   K_Leak current_perturbed
     before: ['mS * mV / cm^2', [-0.09999999403953552, -0.17499999701976776, -0.3499999940395355]]
     after : ['10.0^-2 * A / m^2', [-0.09999999403953552, -0.17499999701976776, -0.3499999940395355]]
   (19 further entries: conductance_factor, absent before / present after)
traced eqns: 8826 -> 7535  (-14.63%)
```

### Tracing

Traced equations over `current` + `compute_derivative` for all 113 classes:
**8 826 → 7 535 (−14.63%)**.

Wall-clock, 19 representative channels:

| | before | after | |
|---|---|---|---|
| jaxpr equations | 1 467 | 1 278 | −12.9% |
| trace | 150.63 ms | 89.86 ms | −40.4% |
| first `jit` | 803.08 ms | 618.36 ms | −23.0% |
| steady-state step | 1.84 ms | 1.52 ms | −17.5% |

**The step-time column is not claimed as a win.** It is single-run
microbenchmark noise: `Na_HH1952`, which this change does not touch at all,
moved 0.223 → 0.061 ms in the same run. Only the equation counts are
deterministic; the trace and first-`jit` columns track them and are
directionally trustworthy. Runtime is unchanged, as the "Efficiency" section
argues it must be.

### Tests

```
$ pytest braincell/channel -q
684 passed, 20 subtests passed

$ pytest braincell/ -q
2751 passed, 15 skipped

$ pre-commit run --files <every changed file>
check for added large files..............................................Passed
check python ast.........................................................Passed
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
```

`pytest braincell/channel` was 682 before this change and is 684 after: the
consolidations are net-neutral on test count (a mixin runs the same assertions
against the same classes), and the two added tests are the `is_disabled`
tracing regressions.

### Docs

`docs/apis/braincell.channel.rst` gained a "Channel Templates" section and the
four `*_Frozen` classes it had been missing; a check that every name in
`braincell.channel.__all__` appears in the `rst` now reports nothing missing.
