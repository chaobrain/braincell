# Cells

A *cell* is the top-level object you simulate. `braincell` offers two cell
classes for two modeling regimes, both descended from the common base
{class}`braincell.HHTypedNeuron`.

## Single-compartment vs. multi-compartment

```{list-table}
:header-rows: 1
:widths: 24 38 38

* -
  - {class}`~braincell.SingleCompartment`
  - {class}`~braincell.Cell`
* - **Geometry**
  - none — one isopotential point
  - a full {class}`~braincell.Morphology`
* - **Spatial structure**
  - voltage is uniform
  - voltage spreads along dendrites/axons
* - **How you build it**
  - subclass and add ions/channels in `__init__`
  - instantiate, then `paint` and `place` declarations
* - **Best for**
  - channel prototyping, point-neuron networks, fitting
  - dendritic computation, distributed channels, synaptic integration
```

Because both share the same ion, channel, and integrator machinery, the
biophysics you learn in one transfers directly to the other.

## Single-compartment cells

A {class}`~braincell.SingleCompartment` treats the neuron as a single
well-mixed compartment. You define one by subclassing and attaching mechanisms
imperatively:

```python
import brainunit as u
import braincell

class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='exp_euler'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,
                                       g_max=0.03 * (u.mS / u.cm ** 2))
```

- `size` is a **batch dimension**: `HH(100)` simulates 100 independent neurons.
- `solver` names the integrator (see {doc}`integration`).
- Channels attach to the ion whose reversal potential drives them (see
  {doc}`ions_channels`).

You then `init_state()` and step the model with `brainstate`. The full runnable
loop is in {doc}`../getting_started/first_steps`.

## Multi-compartment cells

A {class}`~braincell.Cell` wraps a morphology and is decorated declaratively.
The pattern is **paint** (distribute a density mechanism over a region) and
**place** (attach a point mechanism at a location):

```python
import brainunit as u
import braincell
import braincell.mech as mech
from braincell.filter import AllRegion, RootLocation, branch_in

# 1. geometry
morpho = braincell.Morphology.from_swc("neuron.swc")
cell = braincell.Cell(morpho)

# 2. passive cable everywhere
cell.paint(AllRegion(), mech.CableProperty(
    resting_potential=-65. * u.mV,
    membrane_capacitance=1.0 * u.uF / u.cm**2,
    axial_resistivity=100. * u.ohm * u.cm,
))

# 3. channels onto regions
cell.paint(AllRegion(), mech.Channel("IL", g_max=0.0003 * u.S / u.cm**2, E=-70. * u.mV))
cell.paint(branch_in("type", "soma"), mech.Channel("Na_Ba2002", g_max=0.12 * u.S / u.cm**2))

# 4. stimulus at the soma
cell.place(RootLocation(0.5),
           mech.CurrentClamp.step(0.2 * u.nA, duration=50 * u.ms, delay=10 * u.ms))
```

`paint` targets a **region** (a set of cable); `place` targets a **locset** (a
set of points). Those selection expressions come from {mod}`braincell.filter` —
see {doc}`regions_locsets`. The mechanisms themselves are covered in
{doc}`mechanisms`.

```{note}
The multi-compartment build-and-run pipeline (`Cell.run`, probe sampling) is
under active development. The declarative `paint`/`place` front end shown here
is stable; the worked multi-compartment notebooks in
{doc}`../multi_compartment/index` track the current runnable workflow.
```

## The shared base: `HHTypedNeuron`

Both classes inherit from {class}`braincell.HHTypedNeuron`, a
Hodgkin–Huxley-style neuron that owns the integrator and the ion/channel
registry. You rarely instantiate it directly, but it is the reason the two cell
types feel so similar.

## See also

- {doc}`morphology` — building the geometry a `Cell` wraps.
- {doc}`discretization` — how a `Cell`'s geometry becomes control volumes.
- {doc}`../single_compartment/index` and {doc}`../multi_compartment/index` —
  hands-on tutorials.
