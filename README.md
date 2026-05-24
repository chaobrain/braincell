

# Biologically Detailed Brain Cell Modeling in JAX

<p align="center">
  	<img alt="Header image of BrainCell." src="https://brainx.chaobrain.com/images/braincell.webp" width=50%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/braincell/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braincell"></a>
	<a href="https://github.com/chaobrain/braincell/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://braincell.readthedocs.io/?badge=latest'>
        <img src='https://readthedocs.org/projects/braincell/badge/?version=latest' alt='Documentation Status' />
    </a>  	
    <a href="https://badge.fury.io/py/braincell"><img alt="PyPI version" src="https://badge.fury.io/py/braincell.svg"></a>
    <a href="https://github.com/chaobrain/braincell/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braincell/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.14969987"><img src="https://zenodo.org/badge/825447742.svg" alt="DOI"></a>
</p>



[braincell](https://github.com/chaobrain/braincell) provides a unified interface for biophysically detailed brain cell models — from single-compartment Hodgkin-Huxley neurons to fully morphological multi-compartment cells with realistic dendrites and axons.
It is built on top of [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate), offering highly parallelized, differentiable simulation of biologically realistic neural dynamics.



## Features

- **Single-compartment neurons** — `braincell.SingleCompartment` with a rich library of ion channels (Na, K, Ca, HCN, K-Ca) and numerical integrators.
- **Multi-compartment cells** — `braincell.Cell` with declarative mechanism painting (`cell.paint`) and point-process placement (`cell.place`) onto morphological regions.
- **Morphology system** — `braincell.morph`: immutable `Branch` geometry, typed subclasses (`Soma`, `Dendrite`, `Axon`, `BasalDendrite`, `ApicalDendrite`), and the mutable `Morphology` tree.
- **IO readers** — SWC, ASC, and NeuroML2 file readers; a full [NeuroMorpho.Org](https://neuromorpho.org) client with search, download, and local caching.
- **Visualization** — `braincell.vis`: 2D tree layouts (matplotlib) and 3D rendering (PyVista / Plotly), color-by-values, morphometry plots, and movie export.
- **Integrator registry** — `braincell.quad`: explicit (Euler, RK2/3/4), implicit, exponential-Euler, staggered cable solve, and diffrax-backed adaptive solvers, all selectable by name.
- **Declarative mechanisms** — `braincell.mech`: `Channel`, `Ion`, `CableProperty`, `CurrentClamp`, `Synapse`, `Junction` specs for the `Cell` frontend.
- **CLI** — `braincell-neuromorpho` command for searching and downloading from NeuroMorpho.Org.



## Quick start

### Single-compartment neuron

```python
import braincell
import brainstate
import braintools
import brainunit as u

class HTC(braincell.SingleCompartment):
    def __init__(self, size, solver: str = 'ind_exp_euler'):
        super().__init__(size, V_initializer=braintools.init.Constant(-65. * u.mV), V_th=20. * u.mV, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_Ba2002(size, V_sh=-30 * u.mV))

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))
        self.k.add(IDR=braincell.channel.IKDR_Ba2002(size, V_sh=-30. * u.mV, phi=0.25))

        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaL=braincell.channel.ICaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaT=braincell.channel.ICaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.IAHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))

        self.Ih = braincell.channel.Ih_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)
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
cell.paint(AllRegion(), mech.Channel("IL", g_max=0.0003 * u.S / u.cm**2, E=-70 * u.mV))
cell.paint(branch_in("type", "soma"), mech.Channel("INa_Ba2002", g_max=0.12 * u.S / u.cm**2))
cell.paint(
    branch_in("type", ("dendrite", "basal_dendrite", "apical_dendrite")),
    mech.Channel("ICaL_IS2008", g_max=0.002 * u.S / u.cm**2),
)

# Inject current at the soma
cell.place(RootLocation(0.5), mech.CurrentClamp.step(0.2 * u.nA, duration=50 * u.ms, delay=10 * u.ms))
```

### Morphology and visualization

```python
import braincell
import braincell.vis as vis

# Load from NeuroMorpho.Org (cached locally)
morpho = braincell.load_neuromorpho("cnic_001")

# 2-D layout plot
vis.plot2d(morpho)

# 3-D rendering
vis.plot3d(morpho)
```



## Installation

```bash
pip install braincell --upgrade
```

Optional dependency groups:

| Extra | What it installs |
|-------|-----------------|
| `braincell[vis]` | matplotlib, pyvista, plotly (visualization backends) |
| `braincell[io]` | requests (NeuroMorpho.Org client) |
| `braincell[quad]` | diffrax (adaptive ODE solvers) |
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

If you are a Windows user and need a WSL-based development setup with `NEURON 8.2.6` and `nrnivmodl`, see [develop_doc/windows_wsl_neuron_setup.md](develop_doc/windows_wsl_neuron_setup.md).



## Documentation

The official documentation is hosted on Read the Docs: [https://braincell.readthedocs.io](https://braincell.readthedocs.io)



## See also the ecosystem

BrainCell is one part of our brain modeling ecosystem: https://brainmodeling.readthedocs.io/
