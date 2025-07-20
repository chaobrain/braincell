Numerical Integration Methods
=============================

.. currentmodule:: braincell 
.. automodule:: braincell 


Overview
--------

.. autosummary::
   :toctree: generated/

    get_integrator


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


Diffrax Explicit Integrators
-----------------------------

.. autosummary::
   :toctree: generated

    diffrax_euler_step
    diffrax_heun_step
    diffrax_midpoint_step
    diffrax_ralston_step
    diffrax_bosh3_step
    diffrax_tsit5_step
    diffrax_dopri5_step
    diffrax_dopri8_step


Diffrax Implicit Integrators
-----------------------------

.. autosummary::
   :toctree: generated

    diffrax_bwd_euler_step
    diffrax_kvaerno3_step
    diffrax_kvaerno4_step
    diffrax_kvaerno5_step
