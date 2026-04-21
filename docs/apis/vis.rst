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

.. autosummary::
   :toctree: generated/
   :nosignatures:

    plot_dendrogram
    plot_topology
    plot_sholl
    plot_branch_order_histogram


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
