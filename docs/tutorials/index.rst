Modeling Tutorials
==================

A single, progressive path through ``braincell`` — from a one-point Hodgkin–Huxley
cell to a morphologically detailed multi-compartment neuron. Each notebook builds
on the previous one, so working through them in order is the fastest way to learn
the library.

New to the concepts? Read :doc:`../concepts/cells`,
:doc:`../concepts/ions_channels`, :doc:`../concepts/morphology`, and
:doc:`../concepts/mechanisms` first for the mental model, or jump straight to a
complete runnable model in :doc:`../getting_started/first_steps`.

The tutorials fall into three groups:

- **Foundations** — the cell hierarchy and the ion/channel mechanisms shared by
  every model (single- and multi-compartment alike).
- **Morphology & space** — geometry, region/locset selection, and declarative
  mechanism placement.
- **Multi-compartment cells** — discretization into control volumes, running a
  simulation, and visualizing the result.

.. toctree::
   :maxdepth: 1
   :caption: Foundations

   overview
   single_compartment
   channel
   ion

.. toctree::
   :maxdepth: 1
   :caption: Morphology & Space

   morphology
   filter
   mech

.. toctree::
   :maxdepth: 1
   :caption: Multi-Compartment Cells

   cell
   vis

Looking for complete, runnable models? See the :doc:`../examples/index`.
