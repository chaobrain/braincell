``braincell.mech`` module
=========================

.. currentmodule:: braincell.mech
.. automodule:: braincell.mech

``braincell.mech`` is the purely declarative mechanism layer for
:class:`braincell.Cell`. It describes *what* to install on a cell without
touching runtime state, JAX, or ``brainstate``. Every declaration inherits
from the :class:`Mechanism` marker base class and splits into two families:

- **Density mechanisms** are distributed over a region of a cell
  (:class:`Density` and its concrete subclasses :class:`Channel` for ion
  channels and :class:`Ion` for ion species).
- **Point mechanisms** are attached to a single location (:class:`Point` and
  its subclasses, including :class:`SynapseSpec`, :class:`Junction`, and the probe
  declarations).

The passive cable property and the stimulus clamps
(:class:`~braincell.CableProperty`, :class:`~braincell.CurrentClamp`,
:class:`~braincell.SineClamp`, :class:`~braincell.FunctionClamp`) are also
declared here but are re-exported at the top level; see
:doc:`braincell` for their reference entries.


Base
----

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Mechanism
    Params


Density Mechanisms
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Density
    Channel
    Ion


Spatial Parameters
------------------

Cable-property fields and density-mechanism parameters may be callables that
accept a :class:`CVContext`. Cable-property callables are resolved during
discretization; channel and ion callables are resolved once per active CV by
``Cell.init_state()``. They are not called during every simulation step.

.. code-block:: python

    import brainunit as u
    from braincell.filter import AllRegion
    from braincell.mech import Channel

    def na_gmax(context):
        distance = context.path_distance_from_soma.to_decimal(u.um)
        return (0.02 + 0.00008 * distance) * (u.mS / u.cm ** 2)

    cell.paint(
        AllRegion(),
        Channel("Na_HH1952", name="na_distance", g_max=na_gmax),
    )
    cell.init_state()

``path_distance_from_soma`` is zero for every soma CV and starts at zero at
each first-order neurite attachment. ``path_distance_to_root`` retains the
path through the root/soma branch. Callable results must be scalar and must
use a consistent unit type across all selected CVs.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CVContext


Point Mechanisms
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Point
    Junction
    SynapseSpec


Probes
------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    StateProbe
    MechanismProbe
    CurrentProbe
    ProbeMechanism
