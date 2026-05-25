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
    DiffEqModule
    IndependentIntegration


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
    exp_exp_euler_step
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
    implicit_exp_euler_step
    implicit_rk4_step


Other Integrators
-----------------

.. autosummary::
   :toctree: generated/

    cn_exp_euler_step
    cn_rk4_step
    splitting_step
    staggered_step
