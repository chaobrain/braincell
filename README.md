

# Biologically Detailed Brain Cell Modeling in BrainX

<p align="center">
  	<img alt="Header image of BrainCell." src="https://brainx.chaobrain.com/images/braincell.webp" width=50%>
</p>



<p align="center">
	<a href="https://pypi.org/project/braincell/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braincell"></a>
	<a href="https://github.com/chaobrain/braincell/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://brainx.chaobrain.com/braincell/'>
        <img src='https://readthedocs.org/projects/braincell/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://badge.fury.io/py/braincell"><img alt="PyPI version" src="https://badge.fury.io/py/braincell.svg"></a>
    <a href="https://github.com/chaobrain/braincell/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braincell/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.14969987"><img src="https://zenodo.org/badge/825447742.svg" alt="DOI"></a>
</p>



[braincell](https://github.com/chaobrain/braincell) provides a unified interface for biophysically detailed brain cell models — from single-compartment Hodgkin-Huxley neurons to fully morphological multi-compartment cells with realistic dendrites and axons.
It is built on top of [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate), offering highly parallelized, differentiable simulation of biologically realistic neural dynamics.



## Features

- **Point neurons** — `braincell.Cell.point(...)`, a `Cell` with one isopotential compartment, with a rich library of ion channels (Na, K, Ca, HCN, K-Ca) and numerical integrators.
- **Multi-compartment cells** — `braincell.Cell` on a morphology, with declarative mechanism painting (`cell.paint`) and point-process placement (`cell.place`) onto morphological regions.
- **Morphology system** — `braincell.morph`: immutable `Branch` geometry, typed subclasses (`Soma`, `Dendrite`, `Axon`, `BasalDendrite`, `ApicalDendrite`), and the mutable `Morphology` tree.
- **IO readers** — SWC, Neurolucida ASC and NeuroML2 morphology readers; a full [NeuroMorpho.Org](https://neuromorpho.org) client with search, download, and local caching.
- **Visualization** — `Morphology.vis2d` / `vis3d` and the plots in [`braintools.visualize.morphology`](https://github.com/chaobrain/braintools): 2D tree layouts (matplotlib) and 3D rendering (PyVista / Plotly), color-by-values, morphometry plots, topology graphs and movie export. braincell objects convert with `Morphology.to_vis_data()`, `Cell.topology_data()` and `NodeTree.to_topology_data()`, and every plot accepts a `Morphology` directly.
- **Integrator registry** — `braincell.quad`: explicit (Euler, RK2/3/4), implicit, exponential-Euler, and staggered cable solve, all selectable by name.
- **Declarative mechanisms** — `braincell.mech`: `Channel`, `Ion`, `CableProperty`, `CurrentClamp`, `Synapse`, `Junction` specs for the `Cell` frontend.
- **CLI** — `braincell-neuromorpho` command for searching and downloading from NeuroMorpho.Org.



## Quick start

### Point neuron

Declare a thalamocortical relay neuron as a point neuron, then stimulate and run it:

```python
import braincell
import brainunit as u
from braincell import mech


def HTC(size, solver: str = 'ind_exp_euler'):
    cell = braincell.Cell.point(size, V_init=-65. * u.mV, V_th=20. * u.mV, solver=solver)
    cell.paint(
        mech.Ion('SodiumFixed', E=50. * u.mV),
        mech.Channel('Na_Ba2002', V_sh=-30 * u.mV),

        mech.Ion('PotassiumFixed', E=-90. * u.mV),
        mech.Channel('K_Leak', g_max=0.01 * (u.mS / u.cm ** 2)),
        mech.Channel('KDR_Ba2002', V_sh=-30. * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)),

        mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um),
        mech.Channel('CaL_IS2008', g_max=0.5 * (u.mS / u.cm ** 2)),
        mech.Channel('CaT_HM1992', g_max=2.1 * (u.mS / u.cm ** 2)),

        mech.Channel('AHP_De1994', g_max=0.3 * (u.mS / u.cm ** 2)),  # uses K and Ca
        mech.Channel('HCN_HM1992', g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV),
        mech.Channel('IL', g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV),
    )
    return cell


cell = HTC(10)
cell.place(mech.CurrentClamp(delay=10. * u.ms, durations=100. * u.ms, amplitudes=0.01 * u.nA))
cell.record('v', braincell.observe.state('V'))
result = cell.run(dt=0.025 * u.ms, duration=200. * u.ms)
result.samples['v'].values.shape  # (8000, 10)
```

### Multi-compartment cell

Build a morphological neuron from an SWC file and paint ion channels onto it declaratively:

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion, RootLocation, branch_in

# Load morphology from SWC
morpho = braincell.Morphology.from_swc("path/to/neuron.swc")

# Declare and simulate a multi-compartment cell
cell = braincell.Cell(morpho)

# Paint passive cable properties everywhere
cell.paint(AllRegion(), mech.CableProperty(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * u.uF / u.cm**2,
    axial_resistivity=100.0 * u.ohm * u.cm,
))

# Paint ion channels onto specific regions
cell.paint(AllRegion(), mech.Channel("IL", g_max=0.0003 * u.siemens / u.cm**2, E=-70 * u.mV))
cell.paint(branch_in("type", "soma"), mech.Channel("Na_Ba2002", g_max=0.12 * u.siemens / u.cm**2))
cell.paint(
    branch_in("type", ("dendrite", "basal_dendrite", "apical_dendrite")),
    mech.Channel("CaL_IS2008", g_max=0.002 * u.siemens / u.cm**2),
)

# Inject current at the soma
cell.place(RootLocation(0.5), mech.CurrentClamp(delay=10 * u.ms, durations=50 * u.ms, amplitudes=0.2 * u.nA))
```

### Morphology and visualization

```python
import braincell
import braintools.visualize.morphology as bvm

# Load from NeuroMorpho.Org by neuron id (downloaded and cached locally)
morpho = braincell.Morphology.from_neuromorpho(12345)

# 2-D layout plot (same as bvm.plot2d(morpho))
morpho.vis2d()

# 3-D rendering
morpho.vis3d()

# Sholl profile, or the control-volume topology of a cell
bvm.plot_sholl(morpho)
cell = braincell.Cell(morpho)
bvm.plot_topology_graph(cell.topology_data("cv"))
```



## Installation

```bash
pip install braincell --upgrade
```

Optional dependency groups:

| Extra | What it installs |
|-------|-----------------|
| `braincell[vis]` | `braintools[morphology]`: networkx, pyvista, plotly (visualization backends) |
| `braincell[io]` | requests (NeuroMorpho.Org client) |
| `braincell[all]` | all of the above |
| `braincell[cpu]` | jax[cpu] |
| `braincell[cuda12]` | jax[cuda12] |

For example, to install with visualization and IO support:

```bash
pip install "braincell[vis,io]" --upgrade
```

Alternatively, install `BrainX` to get `braincell` together with the rest of the brain modeling ecosystem:

```bash
pip install BrainX -U
```


## Documentation

The official documentation is hosted on Read the Docs: [https://brainx.chaobrain.com/braincell](https://brainx.chaobrain.com/braincell)



## See also the ecosystem

BrainCell is one part of our brain modeling ecosystem: https://brainx.chaobrain.com/
