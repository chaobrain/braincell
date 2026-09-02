# `braincell/ion` simplification

Iteration 5 of the module-by-module simplification sweep. Target:
`braincell/ion` (9,490 lines across 11 files, of which `calcium.py` is
4,453 and `_base.py` is 1,183).

Breaking changes are allowed. Every in-repo caller is updated in the same
PR. No deprecation shims, no back-compat aliases, no `warnings.warn`
bridges — the old spelling is deleted outright.

## Scope and method

Four parallel review passes (reuse, simplification, efficiency, altitude)
over the package, then independent re-verification of every claim I acted
on. Re-verification mattered: one high-value reuse claim — that the five
`Cdp*` **species** tables compose the way their **reaction** tables do —
is false, and acting on it unchecked would have silently changed
`CdpCAM_MA2024_PC`'s calmodulin scaling. See "What the reviews got wrong".

Unlike `braincell/mech` (iteration 4), `braincell/ion` is **not** a leaf
package: `calcium.py` already imports `braincell._base_ion`,
`braincell._base_neuron`, `braincell._typing`, and `braincell.mech`, and
`_base.py` imports `braincell.quad.protocol`. The `_typing.py` aliases and
the rest of the shared infrastructure are therefore available here, and
`braincell/ion/_base.py` is the natural home for shared ion-side logic.

## Bugs fixed

### `_Flux.compute` swallowed genuine `TypeError`s from source callbacks

`_base.py:1155-1165` called each `Source.flux` with a `total_current=`
keyword and, on `TypeError`, retried without it:

```python
try:
    contrib = source.flux(self.owner, V, species_values, total_current=total_current)
except TypeError:
    contrib = source.flux(self.owner, V, species_values)
```

The probe exists for exactly one declaration. There are five `Source(...)`
constructions in the repository (`calcium.py:863, 999, 1194, 2073, 2773`);
four already accept `total_current=None`, and only `calcium.py:865`
(`flux=lambda self, V, x: self.ci_source`) does not.

The cost is not the wasted call. A real `TypeError` raised *inside* any
flux body — a unit mismatch, a bad index, a `None` where an array was
expected — is indistinguishable from a signature mismatch, so it is
swallowed, the callback is re-invoked with a different signature, and the
user sees a confusing second failure from a different line. This runs on
the per-step derivative path.

Fixed by giving `calcium.py:865` the four-argument signature every other
source already has and deleting the `try`/`except` entirely, leaving one
unconditional call.

### `CdpCAM_MA2024_PC` docstring claimed six helpers were delegated that are simply inherited

`calcium.py:3161-3164` stated that `vrat`, `parea`, `dsq`, `dsqvol`,
`_require_diam_arc_mean`, `_ci_source_flux`, and `_as_initializer`
"explicitly delegate to `CdpStC_MA2020_GoC` rather than redefining the
same logic". Six of the seven are inherited from `_RadialShellGeometry`
and involve no delegation at all; only `_ci_source_flux` was a genuine
cross-class call, and this change removes that too. Docstring corrected.

### 19 corrupted parameter descriptions in `calcium.py`

A mechanical replacement of the `Initializer` type alias with its
expansion hit prose as well as annotations, leaving 19 NumPy-doc
descriptions that read:

```
Ci_initializer : array-like or callable, optional
    Union[brainstate.typing.ArrayLike, Callable] for the ``Ci`` species.
```

The sentence should begin `Initializer for the ...`. `calcium.py` is the
only package module affected. These render as gibberish in Sphinx.

### `# -*- coding: utf-8 -*-` below the licence header is inert

`calcium.py:16` carries the PEP 263 magic comment on line 16. PEP 263
only honours it on line 1 or 2, so the line does nothing, and AGENTS.md's
placement rule ("only a shebang and a PEP 263 encoding line may sit above
the header") forbids it there. Deleted. The other five files in the
package have it correctly on line 1 and are left alone.

### `build_placeholder_ions` docstring contradicted by a test

`ion/__init__.py:60-69` claims the function is "used by test doubles and
by `HHTypedNeuron` construction". No test calls it, and
`braincell/_multi_compartment/cell_test.py:1053` is named
`test_build_placeholder_ions_not_called_in_init` and patches it with
`side_effect=AssertionError(...)` — asserting the opposite. The docstring
also says "Na/K/Ca" while the body returns four containers including
`"no"`. Rewritten to name the real caller, and given the missing
`Returns` section.

## Breaking changes

Each of these deletes a spelling outright. Every in-repo caller is updated
in this PR.

### `_diffeq_species` class variables are deleted

The five `Cdp*` kinetic classes each declared a `_diffeq_species` tuple
(`calcium.py:1711, 2086, 2786, 3389, 3970` — 105 lines) restating a value
`_Specs` already derives. Verified at runtime against the worktree tree:

```
CdpStC_CAMOnly_MA2020_GoC      equal=True  n=10
CdpStC_NoCAM_MA2020_GoC        equal=True  n=14
CdpStC_MA2020_GoC              equal=True  n=23
CdpCAM_MA2024_PC               equal=True  n=27
CdpCR_MA2020_GrC               equal=True  n=21
```

Two fields that must agree, with no guard: adding a species and
forgetting the tuple silently makes the `species_initializers` validation
reject a legitimate override. Replaced by a derived
`KineticIon.diffeq_species` property reading
`_Specs.for_type(type(self)).diffeq_names`.

`calcium_test.py:1289`, which asserted the two thin `CdpStC_*` subclasses
inherit the parent's tuple, is rewritten against the property.

### `_Species.algebraic_state` is deleted

`_base.py:1031-1052`. Its only call site is the `else` branch of
`_Conserve.writeback`, whose own docstring says "the allocation branch
below is dead in normal flow", and `_base_test.py:535` asserts it is
never reached. `writeback` collapses to the unconditional assignment.

Two `_base_test.py` tests that drove the deleted branch directly
(`test_late_allocation_still_matches_its_siblings`,
`test_late_allocation_stays_plain_when_its_siblings_are_plain`) are
deleted with it; the invariant they protected is now structural.

### `Calcium.root_type` is deleted

`calcium.py:120` restated `Ion.root_type = HHTypedNeuron`, already set for
every ion at `_base_ion.py:706`. `Sodium`, `Potassium`, and `NonSpecific`
correctly say nothing. Verified `Calcium.root_type is Ion.root_type` is
`True` before the change. Removing it also removes `calcium.py`'s only
import from the private `braincell._base_neuron`.

### `_RadialShellGeometry._as_initializer` moves to `KineticIon`

`_base.py:815`. It normalizes a species initializer and touches no
geometry field; it was reachable only because the `Cdp*` classes happen to
mix in both. Moved to `KineticIon`, where every kinetic ion can use it.
`_base_test.py`'s `_as_initializer` tests move with it.

### Five duplicated `*_initializer` attributes become views (not deletions)

`calcium.py:764, 898, 1045` (`self.BC_initializer = BC_initializer`) and
`calcium.py:1252, 1447` (`self.PumpBound_initializer = ...`) each store a
second copy of a value that has already gone into
`species_initializers={...}`, which is the dict `_Species._species_value`
actually reads. The two copies do not move together: writing through
`species_initializers` leaves the attribute stale, and vice versa. This is
exactly the hazard `KineticIon.Ci_initializer` was introduced to fix — its
docstring says so — and the fix was applied to `Ci` and not to its
siblings.

**These attributes are not dead, and an earlier draft of this spec was
wrong to say so.** A text search finds only the five assignments, because
the read is through a *computed* attribute name:
`_compute/ions.py:310` does
`getattr(baseline_ion, _ion_runtime_attr_name(runtime_cls, param_name))`,
where `param_name` comes from reflecting over `cls.__init__`'s signature
(`_supported_ion_runtime_params`, `ions.py:238`). Deleting them made
`bindings_test.py::test_cell_schedule_applies_to_markov_channels_and_kinetic_ions`
fail with `'ToyCaBindingKinetic_SU2015_DCN' object has no attribute
'BC_initializer'`. Neither the reviews nor my own grep could see that
read; the full-suite run is what caught it.

There is a real cross-module contract here, now written down: **a kinetic
ion must expose every constructor parameter as a readable attribute of
the same name.** So the five are converted to views rather than removed.
`_base.py` gains `species_initializer_view(name)`, a property factory over
`species_initializers`, and `KineticIon.Ci_initializer` is re-expressed
through it, so one mechanism now serves all six.

### `_Specs.for_type` no longer coerces bare tuples

`_base.py:898-918` ran five parallel
`X if isinstance(X, Klass) else Klass(*X)` branches. Across every
`KineticIon` subclass in the package there are 288 spec declarations and
zero non-dataclass ones, and no subclass exists outside `braincell/ion`.
A mis-shaped declaration now fails at the dataclass constructor with a
named field rather than at an unpacking `TypeError`.

## Altitude: logic moved to where it belongs

### `_resolve_species_initializers` becomes a template method

Five copies (`calcium.py:1773, 2178, 2905, 3522, 4073`) whose 12-line
preamble and 2-line trailer are byte-identical; only the `defaults` dict
literal varies. `KineticIon` gains the concrete method — owning the
validation, the error message, the `.update`, and the `_as_initializer`
map — delegating to an abstract `_default_species_initializers`. Each leaf
keeps only its dict.

### The parvalbumin equilibrium quintet gets a mixin

`_kdc`, `_kdm`, `_ss_pv_free`, `_ss_pv_ca`, `_ss_pv_mg` are byte-identical
across `CdpStC_NoCAM_MA2020_GoC`, `CdpStC_MA2020_GoC`, and
`CdpCAM_MA2024_PC` (verified by source comparison). Hoisted to a
`_ParvalbuminEquilibrium` mixin in `calcium.py` — not into `_base.py`,
because a competitive Ca/Mg buffer is calcium chemistry, not shell
geometry.

`_ss_pv_free`/`_ss_pv_ca`/`_ss_pv_mg` each recomputed `_kdc()` and
`_kdm()`; the mixin computes them once and shares them. This is
setup-path work (once per ion construction), so it is a readability fix,
not a measurable speedup — see "Efficiency".

### `_ci_source_flux` moves to `_RadialShellGeometry`

Two verbatim copies (`calcium.py:2230, 2966`) plus two classes reaching
across the class graph for an implementation they do not inherit:

```python
calcium.py:3614:  return CdpStC_MA2020_GoC._ci_source_flux(self, total_current)
calcium.py:4112:  return CdpStC_MA2020_GoC._ci_source_flux(self, total_current)
```

`CdpCAM_MA2024_PC` and `CdpCR_MA2020_GrC` are not subclasses of
`CdpStC_MA2020_GoC`; editing that sibling silently changed two other
models. The body uses only `self.dsqvol` and `self._require_diam_arc_mean()`,
both already members of `_RadialShellGeometry`. All four classes mix that
in, so both copies and both cross-class stubs disappear.

### The Nernst read block is written once

`_base.py:388-395`, `:509-517`, `:713-721` were three verbatim copies of
`_nernst(Ci=_unwrap(self.Ci), Co=_unwrap(self.Co), temp=_unwrap(self.temp),
valence=_unwrap(self.valence))`. `_nernst` gains an owner-taking form so
the four keyword arguments cannot drift out of alignment in one copy only.

### `_RadialShellGeometry` hooks use `super()`

`_base.py:879, 884` called `KineticIon._ion_init_state_hook(self, ...)` by
explicit class name. `KineticIon` is the MRO successor of
`_RadialShellGeometry` for all five users, so `super()` is equivalent and
drops the hard-coded coupling.

## Shared declaration tables

The five `Cdp*` reaction tables total 508 lines. Runtime-verified
composition (structural comparison of `lhs`/`rhs` and the source text of
every `forward`/`backward` lambda, not textual diffing):

```
stc == nocam + camonly  : True   (20 == 8 + 12)
cdpcam[-12:] == camonly : True
first 6 shared across nocam/stc/cdpcam/cdpcr : True
```

The species tables share four blocks, verified the same way:

```
buffer head (10: Ci, mg, Buff1, Buff1_ca, Buff2, Buff2_ca,
             BTC, BTC_ca, DMNPE, DMNPE_ca)  shared by nocam/stc/cdpcam/cdpcr : True
PV triple   (PV, PV_ca, PV_mg)              shared by nocam/stc/cdpcam       : True
pump pair   (pump, pumpca)                  shared by all four               : True
CAM block   (9)  stc[13:22] == camonly[1:]                                   : True
```

Hoisted to module-level constants in `calcium.py` and composed, using the
idiom the file already uses at `calcium.py:860-861` and `:3386-3387`.

## What the reviews got wrong

Two claims did not survive re-verification, and are **not** acted on:

1. **"The species tables compose like the reactions do."** They do not.
   `stc == nocam + camonly[1:]` is `False`, and
   `cdpcam[-9:] == camonly[1:]` is `False`. `CdpCAM_MA2024_PC`'s
   calmodulin species carry `factor="cyto"` where
   `CdpStC_CAMOnly_MA2020_GoC`'s carry `factor="cam_unit"`. The names and
   initializers match; the factor does not. The CAM block is therefore
   shared through a `_cam_species(factor)` helper, not a shared constant.
   Blind composition here would have silently rescaled a published model.

2. **"`CdpLVA_SU2015_DCN` is a rename-only clone of `CdpHVA_SU2015_DCN`,
   collapse it."** Correct as an observation, wrong as a prescription:
   `kCal`/`tauCal`/`caliBase` are the NMODL `RANGE` names the port exists
   to preserve, and `examples/convert_mod` validates against them. Left
   alone. A second reviewer independently reached the same conclusion.

## Efficiency

The kinetic-ion runtime re-derives run-constant geometry 39-97 times per
`compute_derivative` call. Measured evaluation counts for one call:

| class | `dsqvol` | `dsq` | `vrat` | `parea` |
|---|---|---|---|---|
| `CdpCAM_MA2024_PC` | 97 | 97 | 97 | 8 |
| `CdpStC_MA2020_GoC` | 81 | 81 | 81 | 8 |
| `CdpCR_MA2020_GrC` | 75 | 75 | 75 | 8 |
| `CdpStC_CAMOnly_MA2020_GoC` | 44 | 44 | 44 | 0 |
| `CdpStC_NoCAM_MA2020_GoC` | 39 | 39 | 39 | 8 |

Three changes address it:

- `_RadialShellGeometry` computes `vrat`/`dsq`/`dsqvol`/`parea` once in
  its lifecycle hooks and the properties become cache readers. Safe
  because `diam_arc_mean` is written exactly once per ion, by
  `_compute/bridge.py:302`, before `init_state`, and both hooks reseed.
- `_Species` memoizes `factor_value` per name. A `_Species` never
  outlives one derivative evaluation, and `Factor`'s own docstring
  already declares the factor "constant during one integration step".
- `Factor("cam_unit", ...)` built a full `dsqvol` (8 array ops) and threw
  the value away, keeping only `ones_like`'s shape. Replaced with a
  scalar `1.0 * u.um**2`, which broadcasts identically.

**This is a tracing win, not a simulation-speed win, and is reported as
such.** XLA already common-subexpression-eliminates and fuses the
duplicates away, so the compiled program does the same work either way.

Emitted jaxpr equations for one `compute_derivative`:

| model | before | after | |
|---|---:|---:|---|
| `CdpCAM_MA2024_PC` | 699 | 497 | −28.9% |
| `CdpStC_MA2020_GoC` | 605 | 417 | −31.1% |
| `CdpCR_MA2020_GrC` | 550 | 392 | −28.7% |
| `CdpStC_CAMOnly_MA2020_GoC` | 340 | 234 | −31.2% |
| `CdpStC_NoCAM_MA2020_GoC` / `_MA2025_BC` / `_RI2021_SC` | 278 | 192 | −30.9% |
| `ToyDiamFactorKinetic_SU2015_DCN` | 39 | 34 | −12.8% |

Python tracing time for that same call, `size=512`, median of 5:

| model | before | after | |
|---|---:|---:|---|
| `CdpCAM_MA2024_PC` | 64.6 ms | 46.3 ms | −28% |
| `CdpStC_MA2020_GoC` | 61.2 ms | 40.4 ms | −34% |
| `CdpCR_MA2020_GrC` | 48.8 ms | 36.7 ms | −25% |
| `CdpStC_CAMOnly_MA2020_GoC` | 32.7 ms | 22.6 ms | −31% |
| `CdpStC_NoCAM_MA2020_GoC` | 27.1 ms | 18.2 ms | −33% |

Two things measured and found **unchanged**, reported because they are
what a reader would assume improved:

- **First `jit` call (trace + XLA compile)**: 814→879, 713→722, 579→566,
  374→353, 358→348 ms. Compilation dominates and does not shrink; the
  differences are within run-to-run variance in both directions.
- **Steady-state step time**: unchanged. Three alternating before/after
  rounds, `size=512`, median of 40 steps each — `CdpCAM_MA2024_PC`
  1.779/1.943/1.889 before vs 1.780/1.799/1.820 after;
  `CdpStC_CAMOnly_MA2020_GoC` 0.737/0.730/0.733 vs 0.752/0.735/0.773 ms.
  The spread within each arm is larger than the gap between them.

### Numerical effect

Every model's `compute_derivative` output, resolved species map, geometry
factors, and `E` are **bit-identical** before and after, across all 12
kinetic and 4 dynamic calcium models.

The one difference is after a full `backward_euler` step: 35 of the
species values differ, by at most `3.47e-07` relative — 2.9 float32 ULP.
This is floating-point reassociation, expected when common
subexpressions are hoisted, and it enters through the Newton solve rather
than through the model equations, whose outputs are unchanged.

## Tests

- **New `braincell/ion/_testing.py`** — the sanctioned private-helper
  location. Holds:
  - `V()`, an identical 2-line helper copied into `calcium_test.py:57`,
    `potassium_test.py:31`, and `sodium_test.py:31`;
  - `make_shell_ion(cls, **kwargs)`, replacing six near-identical
    `_make_ion` copies at `calcium_test.py:854, 1042, 1110, 1268, 1339,
    1507`, each standing in for the compartment layer that writes
    `diam_arc_mean`/`diam_mid`;
  - `KineticPumpContractTests`, for the two whole test methods that were
    byte-identical across four test classes (`calcium_test.py:927-951`,
    `1180-1204`, `1411-1435`, `1586-1610`);
  - `FixedIonContractTests`, a nine-test suite parameterized by
    `(ION_CLASS, FAMILY_CLASS, DEFAULT_E, DEFAULT_CI, DEFAULT_CO,
    DEFAULT_VALENCE)`, replacing the near-duplicate defaults / varshape /
    callable-broadcast / `pack_info` / container suites in
    `potassium_test.py` and `sodium_test.py`.

  A contract mixin is the point here: a new ion species now inherits the
  shared suite by declaring six attributes, rather than by copying
  methods that then drift. `NonSpecificFixed` picks up nine tests it
  never had this way, and the mixin is what surfaced that
  `NonSpecificFixed(size=1, E=None)` raising `ValueError` was untested for
  every species except through hand-written copies.
- **New `braincell/ion/nonspecific_test.py`** — AGENTS.md rule 10
  violation: `nonspecific.py` has no sibling test file. Its only current
  coverage is a docstring-conformance sweep in `__init___test.py` and two
  *channel* tests that touch the type object. `NonSpecificFixed.__init__`
  — including the `ValueError` its own `Raises` section documents — has no
  direct test anywhere.

## Docs

`docs/apis/braincell.ion.rst` documents every calcium, potassium, and
sodium class but omits `NonSpecific` and `NonSpecificFixed`, both of which
are in `braincell.ion.__all__`. A "Non-Specific Ions" section is added.

## Deliberately not changed

Recorded rather than done, with the reason:

1. **`_cached_total_current` is an ad-hoc `setattr`/`hasattr`/`delattr`
   protocol** spanning `_multi_compartment/cell.py:3150-3169`,
   `ion/_base.py:541-549`, and `ion/_base.py:765-771` (the last two an
   identical 8-line block). The fix — `uses_total_current` plus
   `cache_total_current`/`clear_total_current_cache`/`resolve_total_current`
   on `Ion` — lands in `braincell/_base_ion.py` and
   `braincell/_multi_compartment/`, which are iterations 13 and 11.
2. **Two duplicate four-way `issubclass` species chains**
   (`_compute/ions.py:214-223`, `_compute/bindings.py:983-994`) while the
   class attribute encoding exactly that fact — `ion_symbol`, declared on
   all four family bases — is read nowhere. `ion_symbol.lower()` yields
   the exact keys both functions return. Deferred to iteration 8
   (`_compute`) rather than reaching into that package now; `ion_symbol`
   is kept precisely so iteration 8 can make it load-bearing.
3. **`_runtime_ion_family`, `_ion_runtime_attr_name`, the hardcoded
   `excluded = {"solver", "substeps", "species_initializers"}`, the
   `cainull` fallback in a generic module, and the
   `hasattr(ion, "_update_reversal")` probe** — all in
   `_compute/ions.py`. Same reason: iteration 8.
4. **`KineticIon._step_solver` detects `excluded_paths` support by
   string-matching a `TypeError` message** (`_base.py:733-738`). The real
   fix declares the capability on the integrator protocol in
   `braincell/quad/protocol.py`; cross-package, iteration 14.
5. **A shared `_CdpBufferedShellIon` base for the four big kinetic
   classes**, collapsing their 23-parameter constructors and the 169
   `self.X = braintools.init.param(X, self.varshape, allow_none=False)`
   lines in this file. Real, and larger than everything above combined —
   but it changes five published models' MRO at once. The declaration
   tables are shared in this PR; the constructor is a separate change
   that deserves its own before/after numerical comparison.
6. **Hoisting the four identical `*Fixed` constructors and the three
   byte-identical `*InitNernst` constructors.** `Ion.__init__` precedes
   both mixins in the MRO (`PotassiumFixed -> Potassium -> Ion -> ... ->
   FixedIon`), so a bare hoist is shadowed and the bases must be
   reordered — and `_compute/ions.py:239` introspects `cls.__init__` for
   runtime parameters. The per-leaf signatures also carry the numpydoc
   `Parameters` blocks. A real trade, not a free win.
7. **`build_placeholder_ions` builds a `NonSpecificFixed` that its only
   consumer discards** (`_compute/ions.py:107` iterates `("na","k","ca")`).
   Changing the returned dict is a public-API break whose payoff is one
   avoided construction, and the real fix is in the consumer. Docstring
   corrected here; behaviour left alone.
8. **`_unwrap` has three more copies outside this package**
   (`_base_ion.py:443-451`, `_multi_compartment/probes.py:314,325`). A
   shared `unwrap_state` in `_misc.py` is a cross-package change:
   iteration 14.

## Verification

```
before:  pytest braincell/ion -q -> 232 passed, 1 warning, 7 subtests passed in 57.20s
after:   pytest braincell/ion -q -> 248 passed, 1 warning, 8 subtests passed in 61.36s
before:  pytest braincell/     -q -> 2724 passed, 15 skipped
after:   pytest braincell/     -q -> 2740 passed, 15 skipped, 402 subtests passed in 216.39s
pre-commit run --files <11 changed files> -> all hooks Passed
```

The `+16` in `braincell/ion` reconciles exactly:

| change | delta |
|---|---:|
| `_base_test.py`: two late-allocation tests deleted, one precondition test added | −1 |
| `potassium_test.py`: 8 tests + a 2-test class replaced by the 9-test contract mixin | −1 |
| `sodium_test.py`: same | −1 |
| `calcium_test.py`: 8 duplicated pump tests removed, 8 inherited from the mixin | 0 |
| `nonspecific_test.py`: new file | +19 |
| **total** | **+16** |

The full-suite `+16` is the same 16 — no test outside `braincell/ion`
changed count.

Beyond the suite, this change edits five published model declarations, so
it is also checked numerically: a fingerprint harness records every
kinetic and dynamic calcium model's resolved species map, geometry
factors, `E`, per-species derivatives, and post-step state, and compares
the two trees. Results in "Numerical effect" above.
