# Extending braincell

`braincell`'s registries are open: you can add your own ion channels, ion
species, synapses, and numerical integrators, and they become usable by name
just like the built-ins.

## Adding an ion channel

Concrete channels live in {mod}`braincell.channel` and **self-register** at
import time with the `@register_channel` decorator from {mod}`braincell.mech`.
A new channel subclasses the appropriate base ({class}`braincell.IonChannel` /
{class}`braincell.Channel`), implements its current and gating dynamics, and
registers a name:

```python
import braincell
from braincell.mech import register_channel

@register_channel("MyNa")
class MyNa(braincell.Channel):
    # define states, derivatives, and current here
    ...
```

Once imported, the registered name is what string-based declarations resolve to:

```python
import braincell.mech as mech
mech.Channel("MyNa", g_max=0.1 * u.S / u.cm**2)
```

Use the existing channels in `braincell/channel/` (sodium, potassium, calcium,
…) as templates — they show the expected state declaration, parameter
normalization, and docstring style.

## Adding an ion species

Ion species register the same way with `@register_ion`, subclassing the
relevant ion base ({class}`braincell.Ion`). Model the reversal potential and any
concentration dynamics, following the patterns in `braincell/ion/`.

## Adding a synapse

Synaptic point processes register with `@register_synapse` and are declared via
{class}`braincell.mech.Synapse`. See `braincell/synapse/` for the Markov-model
examples (AMPA, GABAa, NMDA).

## Adding an integrator

Solvers live in {mod}`braincell.quad` and register with
`@register_integrator`, after which they are selectable by name through the
`solver=` argument of any cell:

```python
from braincell.quad import register_integrator

@register_integrator("my_solver")
def my_solver_step(...):
    # advance the state by one dt
    ...
```

```python
cell = braincell.SingleCompartment(size, solver="my_solver")
```

The {doc}`../integration/advanced` guide covers the integrator protocol
({class}`braincell.DiffEqState`, {class}`braincell.DiffEqModule`) in depth.

## Testing your extension

Add a co-located `*_test.py` next to your new module (see {doc}`testing`) that:

- constructs the mechanism with united parameters,
- checks the dynamics against a known reference (an analytic limit, a published
  trace, or NEURON), and
- exercises any edge cases.

## See also

- {doc}`../concepts/ions_channels` — how ions and channels relate.
- {doc}`../concepts/integration` — the solver concept.
- {doc}`project_layout` — where each kind of mechanism lives.
