File Formats & IO
=================

Real morphologies come from reconstructions, not hand-typed coordinates.
``braincell`` reads the common neuronal-morphology formats and integrates with
`NeuroMorpho.Org <https://neuromorpho.org>`_, the largest public repository of
reconstructed neurons. It can also save and reload its own morphologies.

All readers and the NeuroMorpho client live in :mod:`braincell.io`; the most
common entry points are the ``Morphology.from_*`` constructors.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Format
     - Loader
     - Notes
   * - :doc:`SWC <swc>`
     - ``Morphology.from_swc``
     - the de-facto standard; simple point + parent table
   * - :doc:`Neurolucida ASC <asc>`
     - ``Morphology.from_asc``
     - MicroBrightField / Neurolucida ASCII
   * - :doc:`NeuroML2 <neuroml2>`
     - ``NeuroMlReader``
     - XML-based standard for detailed models
   * - :doc:`NeuroMorpho.Org <neuromorpho>`
     - ``Morphology.from_neuromorpho`` / ``io.load_neuromorpho``
     - download + cache directly from the repository
   * - :doc:`Checkpointing <checkpointing>`
     - ``io.save_morpho`` / ``io.load_morpho``
     - persist a ``braincell`` morphology to disk

.. toctree::
   :hidden:
   :maxdepth: 1

   overview
   swc
   asc
   neuroml2
   neuromorpho
   checkpointing
