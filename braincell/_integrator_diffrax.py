# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import importlib.util

import brainunit as u

from ._integrator_util import apply_standard_solver_step, VectorFiled, Y0, T, DT
from ._protocol import DiffEqModule

__all__ = [
    # runge-kutta methods
    'diffrax_euler_step',
    'diffrax_heun_step',
    'diffrax_midpoint_step',
    'diffrax_ralston_step',
    'diffrax_bosh3_step',
    'diffrax_tsit5_step',
    'diffrax_dopri5_step',
    'diffrax_dopri8_step',

    # implicit methods

]

diffrax_installed = importlib.util.find_spec('diffrax') is not None
if not diffrax_installed:
    class Diffrax:
        def __getattr__(self, item):
            raise ModuleNotFoundError(
                'diffrax is not installed. Please install diffrax to use this feature.'
            )


    diffrax = Diffrax()

else:
    import diffrax


def _explicit_solver(solver, fn: VectorFiled, y0: Y0, t0: T, dt: DT, args=()):
    dt = u.Quantity(dt)
    t0 = u.get_magnitude(u.Quantity(t0).to_decimal(dt.unit))
    y1, _, _, state, _ = solver.step(
        diffrax.ODETerm(lambda t, y, args_: fn(t, y, *args_)[0]),
        t0,
        t0 + dt,
        y0,
        args,
        (False, y0),
        made_jump=False
    )
    return y1, {}


def _diffrax_explicit_solver(
    solver,
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    apply_standard_solver_step(
        functools.partial(_explicit_solver, solver),
        target,
        t,
        dt,
        *args,
        merging_method='stack'
    )


def diffrax_euler_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Euler method
    from the diffrax library: `diffrax.Euler <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler>`_.

    This function serves as a wrapper that applies the explicit Euler solver to the given target module.
    It is intended for use in time-stepping routines where the state of the system is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Euler solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Euler(), target, t, dt, *args)


def diffrax_heun_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Heun method
    from the diffrax library: `diffrax.Heun <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Heun>`_.

    This function serves as a wrapper that applies the explicit Heun solver (also known as the improved Euler method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Heun solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Heun(), target, t, dt, *args)


def diffrax_midpoint_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Midpoint method
    from the diffrax library: `diffrax.Midpoint <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Midpoint>`_.

    This function serves as a wrapper that applies the explicit Midpoint solver (a second-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Midpoint solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Midpoint(), target, t, dt, *args)


def diffrax_ralston_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Ralston method
    from the diffrax library: `diffrax.Ralston <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Ralston>`_.

    This function serves as a wrapper that applies the explicit Ralston solver (a second-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Ralston solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Ralston(), target, t, dt, *args)


def diffrax_bosh3_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Bosh3 method
    from the diffrax library: `diffrax.Bosh3 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Bosh3>`_.

    This function serves as a wrapper that applies the explicit Bosh3 solver (a third-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Bosh3 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Bosh3(), target, t, dt, *args)


def diffrax_tsit5_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Tsit5 method
    from the diffrax library: `diffrax.Tsit5 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5>`_.

    This function serves as a wrapper that applies the explicit Tsit5 solver (a fifth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Tsit5 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Tsit5(), target, t, dt, *args)


def diffrax_dopri5_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Dopri5 method
    from the diffrax library: `diffrax.Dopri5 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri5>`_.

    This function serves as a wrapper that applies the explicit Dormand-Prince 5(4) solver (a fifth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Dopri5 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Dopri5(), target, t, dt, *args)


def diffrax_dopri8_step(target: DiffEqModule, t: T, dt: DT, *args):
    """
    Advances the state of a differential equation module by one integration step using the Dopri8 method
    from the diffrax library: `diffrax.Dopri8 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri8>`_.

    This function serves as a wrapper that applies the explicit Dormand-Prince 8(5,3) solver (an eighth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        t (u.Quantity[u.second]): The current simulation time, represented as a quantity with units of seconds.
        dt (u.Quantity[u.second]): The numerical time step of the integration step.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Returns:
        None: The function updates the state of the target module in place.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Dopri8 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    _diffrax_explicit_solver(diffrax.Dopri8(), target, t, dt, *args)


def diffrax_bwd_euler_step(target: DiffEqModule, t: T, dt: DT, *args):
    pass
