Multi-Compartment Modeling
==========================

Multi-compartment cells capture *dendritic geometry* — voltage attenuation along
branches, channels distributed by region, and synaptic integration in space.
A :class:`braincell.Cell` wraps a :class:`~braincell.Morphology` and is decorated
declaratively: you **paint** density mechanisms onto regions and **place** point
mechanisms at locations.

New to these ideas? Read :doc:`../concepts/cells`, :doc:`../concepts/morphology`,
:doc:`../concepts/mechanisms`, and :doc:`../concepts/regions_locsets` for the
mental model, then work through the notebooks below in order.

.. toctree::
   :maxdepth: 1

   morphology
   filter
   mech
   cell
   vis
   channel
   ion
