Biologically Detailed Brain Cell Modeling
=========================================


``braincell`` provides a unified interface for modeling single-compartment and multi-compartment Hodgkin-Huxley-styled neuron models.
It is built on top of `JAX <https://github.com/jax-ml/jax>`_ and `brainstate <https://github.com/chaobrain/brainstate>`_, offering a highly parallelized and efficient simulation of biophysically detailed brain cell models.

Features
--------

- **Biophysical State Precision**: ``BrainCell`` enables biophysically accurate modeling of neural dynamics across scales, from ion channel gating to network-wide population activity.
- **Stiff Dynamics Optimization**: Features specialized solvers optimized for stiff neural systems, efficiently handling rapid biophysical transitions.
- **JAX-based**: Leverages JAX for high-performance computing, automatic differentiation, and compilation to XLA (CPU/GPU/TPU).


----

Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braincell[cpu]

    .. tab-item:: GPU (CUDA)

       .. code-block:: bash

          pip install -U braincell[cuda12]

          # or
          pip install -U braincell[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braincell[tpu]





Quick Start
-----------

Here is an example to model a **single-compartment** thalamus neuron model:

.. code-block:: python

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
           self.ca.add(ICaN=braincell.channel.ICaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
           self.ca.add(ICaT=braincell.channel.ICaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
           self.ca.add(ICaHT=braincell.channel.ICaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

           self.kca = braincell.MixIons(self.k, self.ca)
           self.kca.add(IAHP=braincell.channel.IAHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))

           self.Ih = braincell.channel.Ih_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
           self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)


.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorial/channel-en.ipynb
   tutorial/channel-zh.ipynb
   tutorial/ion-en.ipynb
   tutorial/ion-zh.ipynb
   tutorial/cell-en.ipynb
   tutorial/cell-zh.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials
   :hidden:

   advanced_tutorial/rationale-en.ipynb
   advanced_tutorial/rationale-zh.ipynb
   advanced_tutorial/differential_equation-en.ipynb
   advanced_tutorial/differential_equation-zh.ipynb
   advanced_tutorial/examples.rst
   advanced_tutorial/more-en.ipynb
   advanced_tutorial/more-zh.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Documentation

   apis/braincell.rst
   apis/morphology.rst
   apis/braincell.neuron.rst
   apis/braincell.synapse.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst
   apis/integration.rst
   apis/changelog.md


Ecosystem
---------

BrainCell is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.