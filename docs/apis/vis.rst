``braincell.vis`` module
========================

.. currentmodule:: braincell.vis
.. automodule:: braincell.vis

``braincell.vis`` is the visualization layer of BrainCell. It turns a
:class:`~braincell.Morphology` into static plots through matplotlib,
interactive 3D through PyVista or Plotly, and publication-quality exports.
Pass ``cell.morpho`` to morphology plots; use ``plot_cell_topology(cell, ...)``
to resolve cell-level selections and runtime fields. The module is
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


Start with a Cell
-----------------

Build a morphology, inspect its CV topology, then initialize the cell to
colour runtime nodes by membrane voltage:

.. code-block:: python

    import braincell as bc
    import brainunit as u
    from braincell import vis

    soma = bc.Branch.from_points(
        points=[[0., 0., 0.], [10., 0., 0.]] * u.um,
        radii=[5., 5.] * u.um,
        type="soma",
    )
    morpho = bc.Morphology.from_root(soma, name="soma")
    cell = bc.Cell(morpho, cv_policy=bc.CVPerBranch(2))
    ax = vis.plot2d(cell.morpho)
    vis.plot_cell_topology(cell, level="cv", layout="kamada_kawai")
    cell.init_state()
    voltage_ax = vis.plot_cell_topology(
        cell, value="V", layout="kamada_kawai",
    )
    vis.save_figure(voltage_ax, "voltage.png")

Plotting reads current state without advancing the simulation. Replot to
show new state; reinitializing the cell is a separate model operation.
``Morphology.vis2d()`` and ``Branch.vis2d()`` are convenience wrappers.
Their ``show=True`` default calls ``matplotlib.pyplot.show()``; the function
entry points return figures or axes for explicit display and customization.
PyVista may create a notebook viewer according to its notebook settings.


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
    One node per runtime point, including CV midpoints and boundary points.
    Requires an initialized cell. CV membrane values such as ``"V"`` are
    shown at midpoints; other points receive NaN. Explicit point arrays can
    colour every point.
``level="cv"``
    One node per control volume. Structure and selections can be inspected
    before initialization; value rendering requires initialized state.
``level="branch"``
    One node per morphology branch. Supports region coverage; rejects
    locset, value, and value-colormap options.

Cell plots accept region/locset expressions or evaluated masks. Morphology
plots take evaluated masks. ``value=`` is mutually exclusive with
``region=`` and ``locset=`` in Cell topology plots. Region and locset
highlights at node level select the owning CV's midpoint.

Morphology ``values=`` arrays describe branches, geometric segments, or
centerline points. Cell ``value=`` arrays describe CVs or runtime nodes.
These spaces are distinct even when their array lengths happen to match.

Use ``value="V"``, ``("ion", ion_name, field)``,
``("channel", class_name, field)``, or ``("layout_id", layout_id, field)``
to select fields. Non-singleton population axes require selecting a member
explicitly and passing its one-dimensional array.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    plot_cell_topology


Return values and display
-------------------------

Matplotlib plots return ``Axes``; comparison helpers return a figure and a
tuple of axes. ``plot_traces`` returns ``TracesResult`` with ``figure``,
``morpho_axes`` and ``trace_axes``. ``plot_movie`` returns ``MovieResult``
with ``animation``, ``frames`` and ``output_path``; keep this object alive
while displaying an animation.

Plotly returns a ``Figure``. PyVista returns a ``Plotter`` outside notebooks,
and may return a viewer or an HTML display object inside a notebook.
``return_plotter=True`` requests the raw plotter; use ``notebook=False``
as well to avoid notebook-viewer creation. The morphology convenience
methods return their backend result even when ``return_plotter=False``.

``save_figure`` accepts Matplotlib Axes/Figure, Plotly Figure, or PyVista
Plotter. Pass ``traces.figure`` for a trace result. Output directories must
exist. Plotly static-image export needs an image engine; PyVista vector
export depends on its ``save_graphic`` capability. Animation output uses
``plot_movie(out=...)`` and the relevant writer dependencies.


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
