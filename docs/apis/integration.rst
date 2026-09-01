``braincell.quad`` module
=========================

.. currentmodule:: braincell.quad
.. automodule:: braincell.quad



``braincell.quad`` provides a mechanism to define coupled ordinary differential equations (ODEs)
and solve them using various numerical integration methods.
The integration methods are categorized into exponential integrators, Runge-Kutta methods,
and implicit methods.


Defining Coupled ODEs
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    DiffEqState
    DiffEqSingleState
    DiffEqGroupState
    DiffEqModule
    IndependentIntegration


Hidden-State Classes by Host
----------------------------

A :class:`~braincell.SingleCompartment` has no spatial axis, so its hidden
states are plain :class:`DiffEqSingleState` instances. A :class:`~braincell.Cell`
is spatial — the trailing axis of every hidden state enumerates
compartments or points — so its states are :class:`DiffEqGroupState`,
which is a :class:`brainstate.HiddenGroupState`.

Channel, ion, and synapse code is shared by both hosts, so the class
cannot be chosen at the creation site. ``DiffEqState`` is a marker mixin,
not a concrete class — ``DiffEqState(...)`` now raises ``TypeError``.
Write ``state(...)`` instead in a custom mechanism's ``init_state`` and
the right class is selected for whichever host owns it. ``state_grouping``
is the host-side scope that makes that choice; only a model that is
itself a host needs to call it.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    state
    hidden_state
    state_grouping


Integrator Registry
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    IntegratorEntry
    IntegratorRegistry
    get_integrator
    get_registry
    register_integrator
    all_integrators


Exponential Integrators
------------------------

.. autosummary::
   :toctree: generated/

    exp_euler_step
    ind_exp_euler_step


Runge-Kutta Integrators
-----------------------

.. autosummary::
   :toctree: generated/

    euler_step
    midpoint_step
    rk2_step
    heun2_step
    ralston2_step
    rk3_step
    heun3_step
    ssprk3_step
    ralston3_step
    rk4_step
    ralston4_step


Implicit Integrators
--------------------

.. autosummary::
   :toctree: generated/

    backward_euler_step
    implicit_euler_step


Other Integrators
-----------------

.. autosummary::
   :toctree: generated/

    staggered_step
