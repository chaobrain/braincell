Concepts
========

This section explains the **ideas and vocabulary** behind ``braincell``. It is
not a tutorial — it answers *"what is this, and why does it work this way?"* so
that the tutorials and API reference make sense.

If you are new, read :doc:`architecture` first for the big picture, then
:doc:`units` (because units are mandatory everywhere), and then dip into the
topic you need.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      How declaration, discretization, runtime, and integration fit together.

   .. grid-item-card:: Units
      :link: units
      :link-type: doc

      Why every quantity carries a physical unit, and how to work with them.

   .. grid-item-card:: Cells
      :link: cells
      :link-type: doc

      Single-compartment vs. multi-compartment cells.

   .. grid-item-card:: Morphology
      :link: morphology
      :link-type: doc

      Branches, types, and the morphology tree.

   .. grid-item-card:: Mechanisms
      :link: mechanisms
      :link-type: doc

      The declarative layer: paint densities, place point processes.

   .. grid-item-card:: Ions & Channels
      :link: ions_channels
      :link-type: doc

      Ion species, the channels that carry their currents, and ``MixIons``.

   .. grid-item-card:: Regions & Locsets
      :link: regions_locsets
      :link-type: doc

      Selecting *where* on a cell a mechanism goes.

   .. grid-item-card:: Discretization
      :link: discretization
      :link-type: doc

      Control volumes and the policies that create them.

   .. grid-item-card:: Integration
      :link: integration
      :link-type: doc

      How equations are advanced in time.


.. toctree::
   :hidden:
   :maxdepth: 1

   architecture
   units
   cells
   morphology
   mechanisms
   ions_channels
   regions_locsets
   discretization
   integration
