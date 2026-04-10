# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell._misc import set_module_as
from braincell._typing import VectorFiled, Y0, T, DT
from ._protocol import DiffEqModule
from ._registry import register_integrator
from ._util import apply_standard_solver_step

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
    'diffrax_bwd_euler_step',
    'diffrax_kvaerno3_step',
    'diffrax_kvaerno4_step',
    'diffrax_kvaerno5_step',
]

# ``find_spec`` only checks whether the package is importable; it does not
# actually import diffrax (or any of its sizable dependency tree such as
# equinox, jaxtyping, and sympy). The real import is deferred until the
# first time a ``diffrax_*_step`` function is invoked — see ``__getattr__``
# below.
diffrax_installed = importlib.util.find_spec('diffrax') is not None


def __getattr__(name):
    """Lazily import :mod:`diffrax` on first attribute access.

    This module is imported eagerly by :mod:`braincell.quad.__init__` so its
    ``diffrax_*_step`` functions are visible and (if diffrax is installed)
    registered in the integrator registry. Importing the actual ``diffrax``
    package, however, is expensive — it transitively pulls in equinox,
    jaxtyping, sympy, and friends — so we defer it until the first call to
    one of the diffrax-backed step functions.

    PEP 562 module ``__getattr__`` is invoked when ``diffrax`` is referenced
    inside a step body and not yet present in this module's globals. We
    import diffrax once, stash it in ``globals()`` so subsequent lookups
    take the fast attribute path, and return it.

    Raises
    ------
    ModuleNotFoundError
        If diffrax is not installed and the user calls a diffrax-backed
        step function.
    """
    if name == 'diffrax':
        import diffrax as _diffrax
        globals()['diffrax'] = _diffrax
        return _diffrax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_diffrax(*args, **kwargs):
    """Conditional decorator that only registers when diffrax is installed.

    When ``diffrax`` is missing, the decorated function still exists (so
    ``from braincell.quad import diffrax_*_step`` keeps working) but it is
    not added to the integrator registry. Calling it raises a clean
    ``ModuleNotFoundError`` from the lazy import in :func:`__getattr__`.
    """
    if diffrax_installed:
        return register_integrator(*args, **kwargs)

    def _identity(func):
        return func

    return _identity


def _explicit_solver(solver, fn: VectorFiled, y0: Y0, t0: T, dt: DT, args=()):
    dt = u.Quantity(dt)
    t0 = u.get_magnitude(u.Quantity(t0).to_decimal(dt.unit))
    dt = dt.mantissa
    dt = jnp.asarray(dt)
    t0 = jnp.asarray(t0)
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
        merging='stack'
    )


@_register_diffrax(
    "diffrax_euler",
    category="diffrax",
    order=1,
    description="diffrax.Euler explicit Euler step.",
)
@set_module_as('braincell')
def diffrax_euler_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Euler method
    from the diffrax library: `diffrax.Euler <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler>`_.

    This function serves as a wrapper that applies the explicit Euler solver to the given target module.
    It is intended for use in time-stepping routines where the state of the system is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Euler solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Euler(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_heun",
    category="diffrax",
    order=2,
    description="diffrax.Heun improved-Euler step.",
)
@set_module_as('braincell')
def diffrax_heun_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Heun method
    from the diffrax library: `diffrax.Heun <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Heun>`_.

    This function serves as a wrapper that applies the explicit Heun solver (also known as the improved Euler method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Heun solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Heun(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_midpoint",
    category="diffrax",
    order=2,
    description="diffrax.Midpoint second-order Runge-Kutta step.",
)
@set_module_as('braincell')
def diffrax_midpoint_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Midpoint method
    from the diffrax library: `diffrax.Midpoint <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Midpoint>`_.

    This function serves as a wrapper that applies the explicit Midpoint solver (a second-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Midpoint solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Midpoint(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_ralston",
    category="diffrax",
    order=2,
    description="diffrax.Ralston second-order Runge-Kutta step.",
)
@set_module_as('braincell')
def diffrax_ralston_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Ralston method
    from the diffrax library: `diffrax.Ralston <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Ralston>`_.

    This function serves as a wrapper that applies the explicit Ralston solver (a second-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Ralston solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Ralston(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_bosh3",
    category="diffrax",
    order=3,
    description="diffrax.Bosh3 third-order Runge-Kutta step.",
)
@set_module_as('braincell')
def diffrax_bosh3_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Bosh3 method
    from the diffrax library: `diffrax.Bosh3 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Bosh3>`_.

    This function serves as a wrapper that applies the explicit Bosh3 solver (a third-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.

    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Bosh3 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Bosh3(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_tsit5",
    category="diffrax",
    order=5,
    description="diffrax.Tsit5 fifth-order Runge-Kutta step.",
)
@set_module_as('braincell')
def diffrax_tsit5_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Tsit5 method
    from the diffrax library: `diffrax.Tsit5 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5>`_.

    This function serves as a wrapper that applies the explicit Tsit5 solver (a fifth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Tsit5 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Tsit5(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_dopri5",
    category="diffrax",
    order=5,
    description="diffrax.Dopri5 Dormand-Prince 5(4) step.",
)
@set_module_as('braincell')
def diffrax_dopri5_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Dopri5 method
    from the diffrax library: `diffrax.Dopri5 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri5>`_.

    This function serves as a wrapper that applies the explicit Dormand-Prince 5(4) solver (a fifth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Dopri5 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Dopri5(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_dopri8",
    category="diffrax",
    order=8,
    description="diffrax.Dopri8 Dormand-Prince 8(5,3) step.",
)
@set_module_as('braincell')
def diffrax_dopri8_step(target: DiffEqModule, *args):
    """
    Advances the state of a differential equation module by one integration step using the Dopri8 method
    from the diffrax library: `diffrax.Dopri8 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri8>`_.

    This function serves as a wrapper that applies the explicit Dormand-Prince 8(5,3) solver (an eighth-order Runge-Kutta method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver, such as step size or solver-specific options.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Dopri8 solver for numerical integration.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Dopri8(), target, t, dt, *args)


def _implicit_solver(solver, fn: VectorFiled, y0: Y0, t0: T, dt: DT, args=()):
    dt = u.Quantity(dt)
    t0 = u.Quantity(t0).to_decimal(dt.unit)
    dt = u.get_magnitude(dt)
    y1 = solver.step(
        diffrax.ODETerm(lambda t, y, args_: fn(t, y, *args_)[0]),
        t0,
        t0 + dt,
        y0,
        args,
        None,
        made_jump=False
    )[0]
    return y1, {}


def _diffrax_implicit_solver(solver, target: DiffEqModule, t: T, dt: DT, *args):
    apply_standard_solver_step(
        functools.partial(_implicit_solver, solver),
        target, t, dt, *args, merging='stack'
    )


@_register_diffrax(
    "diffrax_bwd_euler",
    category="diffrax",
    order=1,
    description="diffrax.ImplicitEuler backward Euler step.",
)
@set_module_as('braincell')
def diffrax_bwd_euler_step(target: DiffEqModule, *args, tol=1e-5):
    """
    Advances the state of a differential equation module by one integration step using the implicit
    Backward Euler method from the diffrax library:
    `diffrax.ImplicitEuler <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.ImplicitEuler>`_.

    This function serves as a wrapper that applies the implicit Backward Euler solver to the given
    target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place. The root-finding tolerance for the implicit step can be controlled via the
    `tol` parameter.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver.
        tol (float, optional): Tolerance for the root-finding algorithm used in the implicit step.
            Defaults to 1e-5.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.ImplicitEuler solver for numerical integration.
        - The root-finding algorithm used is diffrax.VeryChord, with both relative and absolute
          tolerances set to `tol`.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(
        diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=tol, atol=tol)),
        target, t, dt, *args
    )


@_register_diffrax(
    "diffrax_kvaerno3",
    category="diffrax",
    order=3,
    description="diffrax.Kvaerno3 implicit third-order step.",
)
@set_module_as('braincell')
def diffrax_kvaerno3_step(target: DiffEqModule, *args, tol=1e-5):
    """
    Advances the state of a differential equation module by one integration step using the Kvaerno3 method
    from the diffrax library: `diffrax.Kvaerno3 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno3>`_.

    This function serves as a wrapper that applies the implicit Kvaerno3 solver (a third-order method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place. The root-finding tolerance for the implicit step can be controlled via the `tol` parameter.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver.
        tol (float, optional): Tolerance for the root-finding algorithm used in the implicit step.
            Defaults to 1e-5.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Kvaerno3 solver for numerical integration.
        - The root-finding algorithm used is diffrax.VeryChord, with both relative and absolute
          tolerances set to `tol`.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(
        diffrax.Kvaerno3(root_finder=diffrax.VeryChord(rtol=tol, atol=tol)),
        target, t, dt, *args
    )


@_register_diffrax(
    "diffrax_kvaerno4",
    category="diffrax",
    order=4,
    description="diffrax.Kvaerno4 implicit fourth-order step.",
)
@set_module_as('braincell')
def diffrax_kvaerno4_step(target: DiffEqModule, *args, tol=1e-5):
    """
    Advances the state of a differential equation module by one integration step using the Kvaerno4 method
    from the diffrax library: `diffrax.Kvaerno4 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno4>`_.

    This function serves as a wrapper that applies the implicit Kvaerno4 solver (a fourth-order method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place. The root-finding tolerance for the implicit step can be controlled via the `tol` parameter.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver.
        tol (float, optional): Tolerance for the root-finding algorithm used in the implicit step.
            Defaults to 1e-5.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Kvaerno4 solver for numerical integration.
        - The root-finding algorithm used is diffrax.VeryChord, with both relative and absolute
          tolerances set to `tol`.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(
        diffrax.Kvaerno4(root_finder=diffrax.VeryChord(rtol=tol, atol=tol)),
        target, t, dt, *args
    )


@_register_diffrax(
    "diffrax_kvaerno5",
    category="diffrax",
    order=5,
    description="diffrax.Kvaerno5 implicit fifth-order step.",
)
@set_module_as('braincell')
def diffrax_kvaerno5_step(target: DiffEqModule, *args, tol=1e-5):
    """
    Advances the state of a differential equation module by one integration step using the Kvaerno5 method
    from the diffrax library: `diffrax.Kvaerno5 <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno5>`_.

    This function serves as a wrapper that applies the implicit Kvaerno5 solver (a fifth-order method)
    to the given target module. It is intended for use in time-stepping routines where the state of the system
    is updated in-place. The root-finding tolerance for the implicit step can be controlled via the `tol` parameter.

    Args:
        target (DiffEqModule): The differential equation module whose state will be advanced.
        *args: Additional arguments to be passed to the solver.
        tol (float, optional): Tolerance for the root-finding algorithm used in the implicit step.
            Defaults to 1e-5.


    Raises:
        ModuleNotFoundError: If the diffrax library is not installed.

    Notes:
        - This function relies on the diffrax.Kvaerno5 solver for numerical integration.
        - The root-finding algorithm used is diffrax.VeryChord, with both relative and absolute
          tolerances set to `tol`.
        - It is part of a suite of step functions that provide different integration methods.
        - The function is designed to be compatible with the braincell integration framework.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(
        diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=tol, atol=tol)),
        target, t, dt, *args
    )
