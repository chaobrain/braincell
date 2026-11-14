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
      :link: concepts/architecture
      :link-type: doc

      The mental model behind cells, mechanisms, morphology, discretization,
      and integration.

   .. grid-item-card:: 🧪 Tutorials & Examples
      :link: tutorials/index
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
   * - **Point neurons**
     - :meth:`braincell.Cell.point`, a cell with one isopotential compartment,
       with a large library of Na, K, Ca, HCN, and K-Ca channels and
       selectable integrators.
   * - **Multi-compartment cells**
     - :class:`braincell.Cell` on a morphology, with declarative mechanism
       painting (``cell.paint``) and point-process placement (``cell.place``)
       onto morphological regions.
   * - **Morphology system**
     - :class:`braincell.Morphology` with an immutable :class:`~braincell.Branch`
       geometry and typed sections (:class:`~braincell.Soma`,
       :class:`~braincell.Dendrite`, :class:`~braincell.Axon`, …).
   * - **File-format readers**
     - SWC, Neurolucida ASC and NeuroML2 readers, plus a full
       `NeuroMorpho.Org <https://neuromorpho.org>`_ client with caching.
   * - **Visualization**
     - ``Morphology.vis2d`` / ``vis3d`` and adapters for
       ``braintools.visualize.morphology``: 2-D tree layouts (matplotlib),
       3-D rendering (PyVista / Plotly), color-by-values, morphometry and
       topology plots.
   * - **Numerical integration**
     - :mod:`braincell.quad`: a registry of solvers selectable by name.
   * - **Declarative mechanisms**
     - :mod:`braincell.mech`: ``Channel``, ``Ion``, ``CableProperty``,
       ``CurrentClamp``, and ``Synapse``; recordings are declared with
       ``Cell.record`` and ``braincell.observe``.


A first taste
-------------

A thalamic relay (HTC) point neuron, assembled from ion species and channels:

.. code-block:: python

   import braincell
   import brainunit as u
   from braincell import mech

   def HTC(size, solver: str = 'ind_exp_euler'):
       cell = braincell.Cell.point(size, V_init=-65. * u.mV, V_th=20. * u.mV, solver=solver)
       cell.paint(
           mech.Ion('SodiumFixed', E=50. * u.mV),
           mech.Channel('Na_Ba2002', V_sh=-30 * u.mV),
           mech.Ion('PotassiumFixed', E=-90. * u.mV),
           mech.Channel('KDR_Ba2002', V_sh=-30. * u.mV),
           mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um),
           mech.Channel('CaL_Li2017', g_max=0.5 * (u.mS / u.cm ** 2)),
           mech.Channel('HCN_HM1992', g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV),
           mech.Channel('IL', g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV),
       )
       return cell

See :doc:`getting_started/first_steps` to run a point neuron and plot a spike train,
or :doc:`concepts/architecture` to understand each piece.


How this documentation is organized
-----------------------------------

The docs are layered by what you are trying to do:

- **Get Started** — install the package and run a first model. Start here if
  you are new.
- **Concepts** — the design and vocabulary of ``braincell``: cells, mechanisms,
  morphology, discretization, and integration. Read this to build a mental
  model before diving deep.
- **Modeling guides** — task-focused :doc:`tutorials <tutorials/index>` and
  runnable :doc:`examples <examples/index>` for single- and multi-compartment
  models, and for :doc:`numerical integration <integration/index>`.
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

   concepts/architecture
   concepts/units
   concepts/cells
   concepts/morphology
   concepts/mechanisms
   concepts/ions_channels
   concepts/regions_locsets
   concepts/discretization
   concepts/integration

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modeling Guides

   tutorials/index
   examples/index
   integration/index
   file_formats/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Guide

   developer/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   apis/changelog.md
   apis/braincell.rst
   apis/morphology.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst
   apis/braincell.synapse.rst
   apis/braincell.mech.rst
   apis/filter.rst
   apis/io.rst
   apis/network.rst
   apis/braincell.plasticity.rst
   apis/reduction.rst
   apis/trainable.rst
   apis/integration.rst
   apis/vis.rst


Ecosystem
---------

``braincell`` is one part of the `BrainX brain-modeling ecosystem
<https://brainx.chaobrain.com/>`_. If you use it in your research, please cite
it via its `Zenodo DOI <https://doi.org/10.5281/zenodo.14969987>`_.
