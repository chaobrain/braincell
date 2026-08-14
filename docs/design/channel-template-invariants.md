# Channel Template Invariants

`braincell/channel/_base.py` provides three declarative templates that the whole channel catalogue
is built on. This note records the conventions they enforce, so a new channel can be written — and
reviewed — without reverse-engineering the base class.

| Template | Use for | Declares |
| --- | --- | --- |
| `HH` | gating only; current is non-ohmic (GHK flux, permeability scaling) | `gates` |
| `OhmicHH` | gating plus `g * gates * (E - V)` | `gates` |
| `Markov` | a conserved probability pool over discrete states | `pairs`, `dependent_state` |

## Everything declared is validated when the class is created

`HH.__init_subclass__` and `Markov.__init_subclass__` resolve `gates` / `pairs` once, cache them on
the class, and reject:

- gate or state names that are not valid identifiers, and duplicate gate names;
- a gate that defines neither `f_<g>_inf` + `f_<g>_tau` nor `f_<g>_alpha` + `f_<g>_beta`, or that
  defines both;
- a `dependent_state` that is not one of the declared states;
- a transition naming a rate method that does not exist;
- fewer than two Markov states.

A class declaring no gates or pairs is skipped, so abstract intermediates stay definable. This is
why an invalid template cannot be written as a module-level `class` statement in a test file — it
aborts collection. Tests that assert on rejection build the class inside `assertRaises`; see
`_make_hh` / `_make_markov` in `braincell/channel/_base_test.py`.

## Gate and state names must not collide with parameters

`init_state` binds one `DiffEqState` per gate / independent state, by name. `_bind_state` refuses to
overwrite an attribute that is not already a `DiffEqState`, so a gate named `g_max` is an error
rather than a silent replacement of the conductance parameter. Re-running `init_state` is
legitimate and replaces the existing state.

## Rate methods declare the ions they read

A gate or transition rate receives `V` plus **only the ion arguments its signature declares**, taken
as a prefix of the channel's ions. `MixIons` builds those arguments in the declaration order of
`root_type.__args__`, so on `JointTypes[Potassium, Sodium, NonSpecific]`:

```python
def f_m_inf(self, V, K):          # receives potassium only
def f_p_alpha(self, V, K, Ca):    # receives potassium and calcium
def f_z_inf(self, V, *ions):      # receives every ion
```

A method asking for more ions than the channel has raises `TypeError`. Because the binding is
positional, a gate that reads only the *second* ion must still accept the first.

## Rate units

Rate methods may return either a bare value or a united quantity:

| Method | Bare value means | United value must have |
| --- | --- | --- |
| `f_<g>_tau` | multiples of `Gate.time_unit` (default `u.ms`) | a time dimension |
| `f_<g>_alpha`, `f_<g>_beta` | reciprocal `Gate.time_unit` | an inverse-time dimension |

Anything else raises an error naming the gate and the rate. Prefer returning united quantities in
new channels; the bare form exists because the catalogue was written against NEURON `.mod` sources
that are implicitly in milliseconds.

Those two checks only police what the rate methods return. The derivative also picks up `phi`, which
is read off the instance and is validated nowhere at construction, so it is checked at the point of
use: every gate derivative must come out with an inverse-time dimension. Without that check a
dimensioned `phi` — say `phi = 2 * u.mV` — yields a `mV / ms` derivative that the integrator carries
without complaint.

Markov transition rates are always bare and implicitly per-millisecond — `compute_derivative`
divides by `u.ms` once at the end.

## Gate metadata binds by name

`phi`, `q10` and `temp_ref` are class-level but usually refer to a constructor parameter. Name it
with a string; a callable is still accepted for anything that is not a plain attribute lookup:

```python
gates = (Gate("p", power=3, q10="q10", temp_ref="temp_ref"),)
```

## Clipping is opt-in for HH and on for Markov

`Gate.clip` defaults to `False`. NEURON does not clip HH gates, and this catalogue is validated
against those mechanisms, so clipping by default would introduce divergence exactly where a model
is hardest to debug. Enable it per gate when a solver is expected to overshoot and an odd `power`
would otherwise produce a negative conductance.

`Markov.clip_states` defaults to `True`: a probability pool that leaves the simplex makes the
kinetics meaningless, whereas an out-of-range HH gate is merely inaccurate.

Both project only the value fed to the conductance product or the kinetics. Stored state is never
rewritten.

## `dependent_state` is effectively required

One Markov state is eliminated and reconstructed as `conserve - sum(others)`. Leaving
`dependent_state` unset falls back to "the last state discovered while scanning `pairs`", which
makes a reordering of `pairs` silently change which state is eliminated. That fallback warns and is
slated for removal;
`ChannelTemplateTest.test_shipped_markov_channels_declare_dependent_state` locks the catalogue
against regressing.

## Numeric-invariance expectation

Changes to this layer are expected to leave every shipped channel bit-for-bit identical unless the
change is explicitly about numerics. The check is behavioural, not structural: construct every
concrete channel, record `reset_state`, per-state `compute_derivative` and `current()` across a
range of clamp voltages plus a short explicit-Euler trajectory, and compare against the same
recording from the pre-change tree. When running such a harness from outside the source tree, set
`PYTHONPATH` to the checkout under test — an installed copy of `braincell` in `site-packages` will
otherwise shadow it and the comparison becomes vacuous.
