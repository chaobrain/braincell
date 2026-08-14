# Channel Gate/Markov Template Hardening

## Problem

`braincell.channel` is unified on a declarative template layer: 92 `HH` subclasses declare
`gates = (Gate(...), ...)` and 18 `Markov` subclasses declare `pairs = (Transition(...), ...)`,
all sharing one lifecycle implementation in `braincell/channel/_base.py`. The abstraction is the
right one, but it is a reflection-driven mini-DSL with no definition-time validation and several
internal inconsistencies.

Two defects were reproduced against the pre-change tree:

1. A mistyped gate name (`Gate("m")` with the method written as `f_n_inf`) raises nothing at class
   definition nor at `init_state()`. It surfaces only at `reset_state()`.
2. `init_state()` performs `setattr(self, gate.name, DiffEqState(...))` with no guard. A gate whose
   name matches a constructor parameter **silently replaces that parameter** with a `DiffEqState`.

Five further inconsistencies:

3. `compute_derivative()` hard-codes `/ u.ms`, so `f_*_tau` must return a bare number implicitly
   denominated in ms and `alpha`/`beta` implicitly in 1/ms. This contradicts the project-wide
   "units are mandatory" rule and is checked nowhere; implementations carry united parameters and
   strip them at the gate boundary (`self.tau_max.to_decimal(u.ms)`, `self.Ra / (1/u.mV/u.ms)`).
4. `Gate` metadata binds instance parameters through `lambda self: self.q10`, used 75 times.
   Class-level metadata reaching into instance state: unpicklable and opaque to tooling.
5. `Markov` projects states into `[0, 1]` before evaluating kinetics; `HH` does not clip at all.
6. `Markov.dependent_state` defaults to "last name discovered while scanning `pairs`", so reordering
   `pairs` — a semantically neutral edit — silently changes which state is eliminated.
7. `self.g_max * self.conductance_factor(...) * (E - V)` is repeated verbatim in 63 classes: the
   template captured gating but not the ohmic driving force.

## Constraints

- Numeric output of all existing channels must be **bit-for-bit unchanged**. NEURON `.mod`
  comparison in `dev/mod_validate/` depends on it.
- `HH` / `Markov` / `Gate` are subclassed almost entirely inside `braincell/channel/`. Two places
  outside it also subclass them and must be migrated with the catalogue:
  `examples/single_compartment/SC07_Straital_beta_oscillation_2011.py` and
  `docs/tutorials/channel.ipynb`. Being a public API, `Gate`'s existing callable metadata form has
  to keep working regardless.
- Existing public constructor signatures and channel class names must not change.

## Decisions

- **Gate clipping is opt-in.** `Gate(clip=False)` by default; `Markov.clip_states` defaults to
  `True`, preserving today's behaviour. NEURON does not clip HH gates either, so defaulting HH to
  clipping would diverge from the reference implementation.
- **Ohmic current gets its own base class.** `OhmicHH(HH)` carries the ohmic `current()`;
  GHK/permeability channels keep inheriting `HH` and implementing `current()` themselves. This
  keeps `HH` responsible for gating only.
- **Unit handling accepts both forms.** Bare returns are interpreted through a per-gate
  `time_unit` (default `u.ms`); united returns are used directly. Every existing channel returns
  bare values, so the default path is numerically identical.
- **`dependent_state` gains an explicit value everywhere in-repo**, with a `DeprecationWarning` on
  the implicit fallback rather than an immediate hard error.

## Invariants established by pre-change probes

- The proposed definition-time checks pass for all 93 existing `HH` subclasses. Only the abstract
  `HH` base is rejected (empty `gates`), so validation must skip classes declaring no gates.
- 8 `Markov` classes rely on the implicit `dependent_state`; all resolve to `"I6"`. Pinning it on
  the 3 root classes in `sodium.py` preserves behaviour exactly (5 subclasses inherit it).
- No existing gate or Markov state name collides with an instance attribute, so the collision guard
  cannot break current code.
- `MixIons` builds ion arguments as `tuple(self._get_ion(i).pack_info() for i in root_type.__args__)`
  (`braincell/_base_ion.py:543`). Ion order is therefore the declaration order of the joint root
  type, which makes prefix-slicing of gate-method arguments well defined.

## Implementation

Seven independently revertable commits, each carrying its own tests:

| Commit | Scope |
| --- | --- |
| C1 | `__init_subclass__` validation for `HH`/`Markov`; cache resolved `gates`/`pairs`; name-collision guard in `init_state`; drop the class-holding `lru_cache` |
| C2 | `Gate.time_unit`; dimension-dispatched `tau` / `alpha` / `beta`; derivative dimension assertion |
| C3 | Route HH gate calls through `_call_rate`; delete the 178-line forwarding block in `potassium_sodium.py` |
| C4 | `Gate.clip`; `Markov.clip_states` |
| C5 | String parameter references in `Gate`; migrate 75 lambdas |
| C6 | Explicit `dependent_state` in `sodium.py`; deprecation warning on the implicit path |
| C7 | `OhmicHH`; migrate 63 ohmic channels; export `_base` symbols from the package |

## Verification

A numeric-invariance harness records, for every constructible gate-template channel, the
`reset_state` values, per-state `compute_derivative` outputs and `current()` at seven clamp
voltages, plus a 20-step explicit-Euler trajectory. The baseline is captured before any edit and
re-checked after every commit; all seven commits must report bit-for-bit equality.

> **Harness pitfall, hit during implementation.** Running the harness as
> `python /tmp/harness.py` puts `/tmp` on `sys.path`, not the working directory, so `import
> braincell` resolved to an installed copy in `site-packages` and the comparison was vacuous — it
> reported "identical" while never loading the edited tree. Always pin `PYTHONPATH` to the checkout
> under test. This is recorded in `docs/design/channel-template-invariants.md` too.

Results:

| Check | Outcome |
| --- | --- |
| Numeric invariance, 99 channels reached by the harness | bit-for-bit identical after every commit |
| `pytest braincell/channel/` | 659 passed (from 615; +44 new cases) |
| `pytest braincell/` | 2186 passed, 19 skipped |
| `pytest examples/neuron_compare/cable/tests` | 51 passed |
| `pytest examples/neuron_compare/channel_no_conc` | 152 passed, and 152 again run alone |
| `pre-commit run --all-files` | passed |

The catalogue holds 110 constructible gate-template channels (92 `HH` + 18
`Markov`); the harness reached 99 of them, the rest needing ion state a bare
`IonInfo` cannot supply. An earlier draft of this table claimed a second
independent re-check over "111 channels" — that harness is untracked and the
number does not correspond to anything reproducible, so the row is gone rather
than restated.

Line count across the catalogue: −226 / +128 from the `OhmicHH` migration, plus 178 lines deleted
from `potassium_sodium.py`.

The author's checkout also ran `dev/mod_validate/channel_validate_sweep_test.py` (9 passed) against
NEURON `.mod` references. **That is not a reproducible gate:** `.gitignore:246` ignores `/dev/`, so
the directory exists on no other machine and is absent from worktrees. The tracked NEURON harness is
`examples/convert_mod/mod_validate/`; treat the `dev/` run as corroboration, not evidence.

Still outstanding: the CI matrix (JAX 0.8.0 floor / 0.10.0 / latest). See also the end of
"Corrections found at review".

## Corrections found at review

### First round

Three claims above were wrong as first written and are corrected in place: the catalogue holds 92
`HH` and 18 `Markov` subclasses (not 93/19); `potassium_sodium.py` lost 178 lines (not 177); and the
"closed refactor surface" constraint was false — `SC07` and `docs/tutorials/channel.ipynb` subclass
`HH`/`Markov` outside `braincell/channel/`.

Two implementation gaps found and closed:

- **C2's derivative dimension assertion had not been written.** `_as_time`/`_as_rate` checked only
  the rate methods' returns. `phi` is read off the instance and checked nowhere, so a dimensioned
  `phi` silently produced a `mV / ms` derivative. `_check_derivative` now runs on every gate.
- **`OhmicHH.conductance_attr` was scope creep** beyond C7 and had no consumer in the catalogue. It
  is removed; `current()` reads `self.g_max`, and a channel naming it differently overrides
  `current()`.

### Second round

The first round's own response was reviewed, and two of its judgements did not hold.

- **C2 was closed on the HH side only.** `Markov.compute_derivative` still divided the accumulated
  derivative by `u.ms` with no check, and both steady-state solvers stripped units with
  `u.get_magnitude`. All three were applied blind, so the exact failure the first round claimed to
  close — a dimensioned `phi` reaching the derivative through a rate expression — was still live on
  the Markov path, and would additionally have flattened a dimensioned rate into the generator
  matrix silently. Every Markov rate now goes through `Markov._transition_rate`, and
  `_check_derivative` is generic over its label and guards the Markov derivative too. This is the
  gap that mattered; the rest of this section is bookkeeping.
- **Deferring the notebook was wrong.** The first round called `docs/tutorials/channel.ipynb` "a
  docs-freshness item rather than a break" in the same breath as adding the constraint that says it
  "must be migrated with the catalogue". Sphinx does not execute notebooks, so the stored output —
  including a `Gate` repr predating `time_unit`, `clip` and the string metadata form — is what gets
  published. It is migrated now, with the two affected outputs regenerated by running the cells.

Three further things the second round corrected:

- The `Gate.clip` / `Markov.clip_states` pair is *not* symmetric in what it projects, despite what
  commit `8bf7cb9`'s subject line says. Only the Markov side clips the values fed to the kinetics.
  That asymmetry is deliberate — clipping an HH gate's derivative input weakens the very term that
  pulls it back into range — but it was undocumented, and is now argued for in
  `docs/design/channel-template-invariants.md`.
- `TODO.md` and `changelog.md` were untouched, which CONTRIBUTING.md asks for on a change that adds
  a public class, two public `Gate` fields and a `DeprecationWarning`.
- `AGENTS.md` rule 8's clause requiring a `docs/` note to be cited from the module docstring was
  removed **on the maintainer's explicit instruction**, not to dodge a check this branch would have
  failed. Recorded here because two independent reviews read the deletion as the latter, which is
  the natural reading from the diff alone.

Still outstanding after this round: the implicit `dependent_state` fallback remains behind its
`DeprecationWarning` rather than being removed, and the CI matrix has not been run.
