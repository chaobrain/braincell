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
  its subclasses, including :class:`Synapse`, :class:`Junction`, and the probe
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


Point Mechanisms
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Point
    Junction
    Synapse


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
