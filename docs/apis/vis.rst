``braincell.vis`` module
========================

.. currentmodule:: braincell.vis
.. automodule:: braincell.vis

``braincell.vis`` is the visualization layer of BrainCell. It turns a
:class:`~braincell.Morphology` (or any higher-level object that carries
one) into static plots through matplotlib, interactive 3D through PyVista
or Plotly, and publication-quality exports. The module is deliberately
split into three layers:

1. **Scene builders** (``scene2d`` / ``scene3d``) translate a morphology
   plus an overlay spec into backend-agnostic primitive tuples.
2. **Backends** (``backend_matplotlib``, ``backend_pyvista``,
   ``backend_plotly``) render those primitives. Each backend advertises
   its supported scene kinds via a capability set.
3. **High-level entry points** (``plot2d``, ``plot3d``, ``plot_movie``,
   ``plot_traces``, morphometry plots) are the user-facing surface.

Optional dependencies (``matplotlib``, ``pyvista``, ``plotly``) are
imported lazily inside the backend that uses them so the base install
stays small.


Top-level plot entry points
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    plot2d
    plot3d
    plot_movie
    plot_traces


Morphometry and topology plots
------------------------------

These take a :class:`~braincell.Morphology` (or, for
``plot_point_topology``, a :class:`~braincell.NodeTree`) and know nothing
about a cell's runtime.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    plot_dendrogram
    plot_topology
    plot_point_topology
    plot_sholl
    plot_branch_order_histogram


Cell topology plots
-------------------

``plot_cell_topology`` is the cell-aware counterpart of the plots above.
It takes a :class:`~braincell.Cell` rather than a bare morphology, which
is what lets it resolve ``region`` / ``locset`` selections against the
cell's control volumes and colour nodes by runtime state. Pick the
granularity with ``level``:

``level="node"`` (the default)
    One node per runtime point — the level the solver works at, and the
    only one that can show per-point state. Requires an initialized cell.
``level="cv"``
    One node per control volume.
``level="branch"``
    One node per morphology branch. Topology only.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    plot_cell_topology


Comparison helpers
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    compare_morphologies
    compare_values


Interactivity hooks
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    VisHooks
    PickInfo


Styling, themes and configuration
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    theme
    publication_theme
    configure_defaults
    get_defaults
    reset_defaults
    set_defaults
    save_figure

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    VisDefaults
    PublicationTheme


Scene primitives and overlays
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    OverlaySpec
    ValueSpec


Layout engine
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    LayoutConfig
    LayoutCache


Publication constants
----------------------

.. py:data:: PUBLICATION_BRANCH_TYPE_COLORS

   Publication-ready branch-type colour palette (RGB tuples keyed by branch
   type). High-contrast, print-friendly, and colour-blind safe; mirrors the
   keys of the default palette so the two presets can be diffed side by side.

.. py:data:: PUBLICATION_RC_PARAMS

   Matplotlib ``rcParams`` applied when the publication theme is active.
   Tuned for LaTeX-style output (serif font, thicker lines, no grid, tight
   margins) and 300 dpi raster export.
