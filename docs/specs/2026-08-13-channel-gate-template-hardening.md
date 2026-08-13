# Channel Gate/Markov Template Hardening

## Problem

`braincell.channel` is unified on a declarative template layer: 93 `HH` subclasses declare
`gates = (Gate(...), ...)` and 19 `Markov` subclasses declare `pairs = (Transition(...), ...)`,
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
- `HH` / `Markov` / `Gate` are subclassed only inside `braincell/channel/`, so the refactor surface
  is closed.
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
| C3 | Route HH gate calls through `_call_rate`; delete the 177-line forwarding block in `potassium_sodium.py` |
| C4 | `Gate.clip`; `Markov.clip_states` |
| C5 | String parameter references in `Gate`; migrate 75 lambdas |
| C6 | Explicit `dependent_state` in `sodium.py`; deprecation warning on the implicit path |
| C7 | `OhmicHH`; migrate 63 ohmic channels; export `_base` symbols from the package |

## Verification

A numeric-invariance harness records, for all 99 constructible gate-template channels, the
`reset_state` values, per-state `compute_derivative` outputs and `current()` at seven clamp
voltages, plus a 20-step explicit-Euler trajectory. The baseline is captured before any edit and
re-checked after every commit; all seven commits must report bit-for-bit equality.

Alongside it: `pytest braincell/channel/` (615 cases), `pytest braincell/`,
`dev/mod_validate/channel_validate_sweep_test.py`, the `SC07` example, `pre-commit run --all`, and
the CI matrix (JAX 0.8.0 floor / 0.10.0 / latest).
