

# Biologically Detailed Brain Cell Modeling in JAX

<p align="center">
  	<img alt="Header image of BrainCell." src="https://raw.githubusercontent.com/chaobrain/braincell/main/docs/_static/braincell.png" width=50%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/braincell/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braincell"></a>
	<a href="https://github.com/chaobrain/braincell/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://braincell.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/braincell/badge/?version=latest' alt='Documentation Status' />
    </a>  	
    <a href="https://badge.fury.io/py/braincell"><img alt="PyPI version" src="https://badge.fury.io/py/braincell.svg"></a>
    <a href="https://github.com/chaobrain/braincell/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braincell/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.14969987"><img src="https://zenodo.org/badge/825447742.svg" alt="DOI"></a>
</p>



[braincell](https://github.com/chaobrain/braincell) provides a unified interface for modeling single-compartment and multi-compartment Hodgkin-Huxley-styled neuron models. 
It is built on top of [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate), offering a highly parallelized and efficient simulation 
of biophysically detailed brain cell models.






## Quick start


Here is an example to model the **single-compartment** thalamus neuron model by using the interface of `braincell.neuron.SingleCompartment`:

```python
import braincell
import brainstate
import brainunit as u

class HTC(braincell.neuron.SingleCompartment):
    def __init__(self, size, solver: str = 'exp_euler'):
        super().__init__(size, V_initializer=brainstate.init.Constant(-65. * u.mV), V_th=20. * u.mV, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_Ba2002(size, V_sh=-30 * u.mV))

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))
        self.k.add(IDR=braincell.channel.IKDR_Ba2002(size, V_sh=-30. * u.mV, phi=0.25))

        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaL=braincell.channel.ICaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaN=braincell.channel.ICaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaT=braincell.channel.ICaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaHT=braincell.channel.ICaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.IAHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))

        self.Ih = braincell.channel.Ih_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)

```


Here is an example to model the **multi-compartment** neuron model by using the interface of `braincell.neuron.MultiCompartment`:


```python
import braincell
import brainstate
import brainunit as u

# TODO

class HTC(braincell.neuron.MultiCompartment):
    def __init__(self, size, solver: str = 'exp_euler'):
        super().__init__(size, V_initializer=brainstate.init.Constant(-65. * u.mV), V_th=20. * u.mV, solver=solver)


```




## Installation

You can install ``braincell`` via pip:

```bash
pip install braincell --upgrade
```

## Documentation

The official documentation is hosted on Read the Docs: [https://braincell.readthedocs.io](https://braincell.readthedocs.io)



## See also the brain modeling ecosystem

We are building the BDP ecosystem: https://brainmodeling.readthedocs.io/

