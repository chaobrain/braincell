# Mechanisms

A *mechanism* is anything you install on a cell that affects its dynamics: an
ion channel, an ion species, a passive cable property, a current clamp, a
synapse, or a probe. In `braincell`, mechanisms are **purely declarative** —
they live in {mod}`braincell.mech` and describe *what* to install without
touching JAX, time, or runtime state.

```python
import braincell.mech as mech
```

## Two families: density and point

Every declaration inherits from the {class}`~braincell.mech.Mechanism` marker
base and falls into one of two families:

```{list-table}
:header-rows: 1
:widths: 22 30 48

* - Family
  - Base class
  - Distributed how?
* - **Density**
  - {class}`~braincell.mech.Density`
  - spread over a **region** of cable (a quantity *per unit area*). Painted.
* - **Point**
  - {class}`~braincell.mech.Point`
  - attached at a single **location**. Placed.
```

This distinction drives the two verbs you use to decorate a cell:

```text
   cell.paint(region,  density_mechanism)   # distribute over cable
   cell.place(locset,  point_mechanism)      # attach at points
```

The `region` and `locset` arguments are selection expressions from
{mod}`braincell.filter` — see {doc}`regions_locsets`.

## Density mechanisms (paint these)

```{list-table}
:header-rows: 1
:widths: 30 70

* - Declaration
  - Describes
* - {class}`~braincell.mech.Channel`
  - an ion channel, distributed with a maximal conductance density `g_max`.
* - {class}`~braincell.mech.Ion`
  - an ion species (its dynamics / reversal potential model).
* - {class}`~braincell.CableProperty`
  - passive cable: resting potential, membrane capacitance, axial resistivity,
    temperature.
```

`Channel` and `Ion` name the concrete implementation either by class **or by
string**, then take its parameters as keywords:

```python
import brainunit as u
import braincell.mech as mech

# by string name
mech.Channel("Na_Ba2002", g_max=0.12 * u.S / u.cm**2)

# passive cable
mech.CableProperty(
    resting_potential=-65. * u.mV,
    membrane_capacitance=1.0 * u.uF / u.cm**2,
    axial_resistivity=100. * u.ohm * u.cm,
)
```

The string names correspond to the concrete classes in
{mod}`braincell.channel` and {mod}`braincell.ion`, which self-register at import
time (see {doc}`ions_channels`).

## Point mechanisms (place these)

```{list-table}
:header-rows: 1
:widths: 30 70

* - Declaration
  - Describes
* - {class}`~braincell.CurrentClamp`
  - injected current; `CurrentClamp.step(amp, duration=, delay=)` for a step.
* - {class}`~braincell.SineClamp` / {class}`~braincell.FunctionClamp`
  - sinusoidal or arbitrary-function current injection.
* - {class}`~braincell.mech.Synapse`
  - a synaptic point process.
* - {class}`~braincell.mech.Junction`
  - a gap junction.
* - {class}`~braincell.mech.StateProbe` / {class}`~braincell.mech.MechanismProbe` / {class}`~braincell.mech.CurrentProbe`
  - **recording** probes — what you `place` to read out a simulation.
```

```python
from braincell.filter import RootLocation
import braincell.mech as mech
import brainunit as u

# inject a step current at the soma
cell.place(RootLocation(0.5),
           mech.CurrentClamp.step(0.2 * u.nA, duration=50 * u.ms, delay=10 * u.ms))

# record membrane voltage there
cell.place(RootLocation(0.5), mech.StateProbe("V"))
```

```{important}
A multi-compartment run needs **at least one placed probe** — the probes define
what `Cell.run(...)` returns. Without one you will get
`ValueError: Cell.run(...) requires at least one placed probe.`
```

## Why declarations are separate from the runtime

Because a mechanism is plain, hashable data, `braincell` can:

- **deduplicate** identical paints across regions (via
  {class}`~braincell.mech.Params`, an order-insensitive frozen mapping);
- **inspect and diff** a model before it runs;
- **compile** the whole decorated cell into a single differentiable kernel.

This is the declaration layer of the {doc}`architecture`.

## See also

- {doc}`ions_channels` — the channels and ions you name in `Channel`/`Ion`.
- {doc}`regions_locsets` — the `paint`/`place` targets.
- {doc}`../apis/braincell.mech` — full mechanism API reference.
