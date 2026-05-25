Biologically Detailed Brain Cell Modeling
=========================================

``braincell`` is a Python library for **biophysically detailed neuron modeling**,
from single-compartment Hodgkin–Huxley cells to fully morphological
multi-compartment reconstructions with realistic dendrites and axons.

It is built on `JAX <https://github.com/jax-ml/jax>`_ and
`brainstate <https://github.com/chaobrain/brainstate>`_, so every model is
**differentiable**, **JIT-compiled**, and runs unchanged on **CPU, GPU, and TPU**.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 🚀 Get Started
      :link: getting_started/installation
      :link-type: doc

      Install ``braincell`` and simulate your first neuron in a few minutes.

   .. grid-item-card:: 💡 Concepts
      :link: concepts/index
      :link-type: doc

      The mental model behind cells, mechanisms, morphology, discretization,
      and integration.

   .. grid-item-card:: 🧪 Tutorials & Examples
      :link: single_compartment/index
      :link-type: doc

      Step-by-step notebooks for single- and multi-compartment models.

   .. grid-item-card:: 📐 Numerical Integration
      :link: integration/index
      :link-type: doc

      Choose, compose, and write the solvers that advance your equations.

   .. grid-item-card:: 📂 File Formats & IO
      :link: file_formats/index
      :link-type: doc

      Load morphologies from SWC, ASC, NeuroML2, and NeuroMorpho.Org.

   .. grid-item-card:: 🔧 API Reference
      :link: apis/braincell
      :link-type: doc

      The complete public API: classes, functions, channels, and ions.


What is ``braincell``?
----------------------

``braincell`` lets you describe a neuron the way a neuroscientist thinks about
one — geometry, ion channels, ion species, and stimuli — and then turns that
description into a fast, differentiable simulation. Its design goals are:

- **Biophysical precision.** Model neural dynamics across scales, from single
  ion-channel gating to morphologically detailed dendritic computation.
- **Stiff-system performance.** A registry of numerical solvers
  (:mod:`braincell.quad`) — explicit, implicit, exponential-Euler, and staggered
  cable solvers — handles the stiff dynamics of biophysical neurons efficiently.
- **Differentiability.** Because everything runs on JAX, gradients flow through
  whole simulations, enabling gradient-based fitting and optimization.
- **One canonical API.** Mechanisms are *declared* with explicit physical units
  and *painted* or *placed* onto a cell, keeping declaration cleanly separated
  from the runtime that executes it.


Feature highlights
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - What you get
   * - **Single-compartment neurons**
     - :class:`braincell.SingleCompartment` with a large library of Na, K, Ca,
       HCN, and K-Ca channels and selectable integrators.
   * - **Multi-compartment cells**
     - :class:`braincell.Cell` with declarative mechanism painting
       (``cell.paint``) and point-process placement (``cell.place``) onto
       morphological regions.
   * - **Morphology system**
     - :class:`braincell.Morphology` with an immutable :class:`~braincell.Branch`
       geometry and typed sections (:class:`~braincell.Soma`,
       :class:`~braincell.Dendrite`, :class:`~braincell.Axon`, …).
   * - **File-format readers**
     - SWC, Neurolucida ASC, and NeuroML2 readers, plus a full
       `NeuroMorpho.Org <https://neuromorpho.org>`_ client with caching.
   * - **Visualization**
     - :mod:`braincell.vis`: 2-D tree layouts (matplotlib) and 3-D rendering
       (PyVista / Plotly), color-by-values, and morphometry plots.
   * - **Numerical integration**
     - :mod:`braincell.quad`: a registry of solvers selectable by name.
   * - **Declarative mechanisms**
     - :mod:`braincell.mech`: ``Channel``, ``Ion``, ``CableProperty``,
       ``CurrentClamp``, ``Synapse``, and probe specifications.


A first taste
-------------

A single-compartment thalamic relay (HTC) neuron, assembled from ion species
and channels:

.. code-block:: python

   import braincell
   import braintools
   import brainunit as u

   class HTC(braincell.SingleCompartment):
       def __init__(self, size, solver: str = 'ind_exp_euler'):
           super().__init__(size, V_initializer=braintools.init.Constant(-65. * u.mV),
                            V_th=20. * u.mV, solver=solver)

           self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
           self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))

           self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
           self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV))

           self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
           self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))

           self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
           self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)

See :doc:`getting_started/first_steps` to run this model and plot a spike train,
or :doc:`concepts/index` to understand each piece.


How this documentation is organized
-----------------------------------

Following the structure that works well for libraries like
`Arbor <https://docs.arbor-sim.org>`_, the docs are layered by what you are
trying to do:

- **Get Started** — install the package and run a first model. Start here if
  you are new.
- **Concepts** — the design and vocabulary of ``braincell``: cells, mechanisms,
  morphology, discretization, and integration. Read this to build a mental
  model before diving deep.
- **Modeling guides** — task-focused tutorials and runnable examples for
  :doc:`single-compartment <single_compartment/index>` and
  :doc:`multi-compartment <multi_compartment/index>` models, and for
  :doc:`numerical integration <integration/index>`.
- **File Formats & IO** — how to load and save morphologies.
- **Developer Guide** — project layout, testing, and how to extend the library.
- **API Reference** — the exhaustive, generated reference for every public
  symbol.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   getting_started/installation
   getting_started/overview
   getting_started/first_steps

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Concepts

   concepts/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modeling Guides

   single_compartment/index
   multi_compartment/index
   integration/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: File Formats & IO

   file_formats/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Guide

   developer/index
   troubleshooting

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   apis/braincell.rst
   apis/morphology.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst
   apis/braincell.synapse.rst
   apis/braincell.mech.rst
   apis/filter.rst
   apis/io.rst
   apis/integration.rst
   apis/vis.rst
   apis/changelog.md


Ecosystem
---------

``braincell`` is one part of the `BrainX brain-modeling ecosystem
<https://brainx.chaobrain.com/>`_. If you use it in your research, please cite
it via its `Zenodo DOI <https://doi.org/10.5281/zenodo.14969987>`_.
