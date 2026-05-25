Single-Compartment Modeling
===========================

A single-compartment neuron treats the cell as one isopotential point — the
simplest useful model, ideal for prototyping channel dynamics, building
point-neuron networks, and gradient-based fitting. These models are built by
subclassing :class:`braincell.SingleCompartment` and attaching ion species and
channels.

New to the concepts? See :doc:`../concepts/cells` and
:doc:`../concepts/ions_channels` first, or run a complete model end-to-end in
:doc:`../getting_started/first_steps`.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/channel
   tutorial/ion
   tutorial/cell

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index
