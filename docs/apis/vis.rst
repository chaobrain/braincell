Visualization
=============

braincell has no plotting module of its own. Morphology and topology plots
live in ``braintools.visualize.morphology`` (install the backends with
``pip install braincell[vis]``, which pulls in ``braintools[morphology]``).
Those plots read neutral, frozen data rather than braincell objects; braincell
converts its objects with three methods, and every plot also accepts a
:class:`~braincell.Morphology` or a ``NodeTree`` directly.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - braincell
     - What it returns
   * - :meth:`Morphology.to_vis_data() <braincell.Morphology.to_vis_data>`
     - ``MorphologyData``: one ``BranchData`` per branch, in micrometres,
       cached until the morphology is edited.
   * - :meth:`Cell.topology_data(level, value=, region=, locset=) <braincell.Cell.topology_data>`
     - ``TopologyGraph`` at ``level="node"``, ``"cv"`` or ``"branch"``,
       highlighted by a region or locset, or carrying values (``"V"``,
       ion, channel or layout fields, or arrays; a ``(T, n)`` array gives
       one frame per row).
   * - ``cell.node_tree.to_topology_data()``
     - ``TopologyGraph`` of the solver nodes (``level="node"``).
   * - :meth:`Morphology.vis2d() <braincell.Morphology.vis2d>`,
       :meth:`Morphology.vis3d() <braincell.Morphology.vis3d>`,
       ``Branch.vis2d()``, ``Branch.vis3d()``
     - Convenience wrappers over ``plot2d`` and ``plot3d``; ``show=True``
       (the default) calls ``matplotlib.pyplot.show()``.

.. code-block:: python

    import braincell as bc
    import braintools.visualize.morphology as bvm

    morpho = bc.Morphology.from_swc("neuron.swc")
    morpho.vis2d(layout="stem")                    # bvm.plot2d(morpho, layout="stem")
    bvm.plot_sholl(morpho)

    cell = bc.Cell(morpho, cv_policy=bc.CVPerBranch(2))
    bvm.plot_topology_graph(cell.topology_data("cv", region=bc.filter.BranchSlice(1, 0.0, 0.5)))
    cell.init_state()
    bvm.plot_topology_graph(cell.topology_data(value="V"), cmap="viridis")

Evaluated region and locset masks of a morphology (``morpho.select(expr)``)
can be passed to ``plot2d`` and ``plot3d`` as ``region=`` and ``locset=``;
their branch indices are the positions in :attr:`Morphology.branches
<braincell.Morphology.branches>`, the same as in ``to_vis_data()``.

The full plotting API (layouts, value spaces, themes, backends, movies and
export) is documented with ``braintools.visualize.morphology``.
