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
@set_module_as('braincell.quad')
def diffrax_euler_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's explicit Euler solver.

    Wraps `diffrax.Euler
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler>`_
    so it can be driven from the same ``brainstate.environ`` context as
    the native braincell integrators. The first call triggers a one-time
    import of :mod:`diffrax` (see :func:`__getattr__`); subsequent calls
    take the fast path.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s
        ``compute_derivative`` and ``pre/post_integral`` hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    euler_step : Native (no-diffrax) explicit Euler implementation.
    diffrax_heun_step, diffrax_midpoint_step, diffrax_ralston_step :
        Other low-order diffrax-backed schemes.

    Notes
    -----
    The current time and step size are read from the active
    :mod:`brainstate.environ` context. The braincell state vector is
    stacked along the last axis before being handed to diffrax.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Euler(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_heun",
    category="diffrax",
    order=2,
    description="diffrax.Heun improved-Euler step.",
)
@set_module_as('braincell.quad')
def diffrax_heun_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Heun (improved Euler) solver.

    Wraps `diffrax.Heun
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Heun>`_,
    a two-stage second-order explicit Runge-Kutta method.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    heun2_step : Native (no-diffrax) Heun implementation.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Heun(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_midpoint",
    category="diffrax",
    order=2,
    description="diffrax.Midpoint second-order Runge-Kutta step.",
)
@set_module_as('braincell.quad')
def diffrax_midpoint_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's explicit midpoint solver.

    Wraps `diffrax.Midpoint
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Midpoint>`_,
    a two-stage second-order explicit Runge-Kutta method.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    midpoint_step : Native (no-diffrax) midpoint implementation.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Midpoint(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_ralston",
    category="diffrax",
    order=2,
    description="diffrax.Ralston second-order Runge-Kutta step.",
)
@set_module_as('braincell.quad')
def diffrax_ralston_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Ralston second-order solver.

    Wraps `diffrax.Ralston
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Ralston>`_,
    the second-order explicit Runge-Kutta method that minimises the
    leading-order truncation error coefficient.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    ralston2_step : Native (no-diffrax) Ralston implementation.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Ralston(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_bosh3",
    category="diffrax",
    order=3,
    description="diffrax.Bosh3 third-order Runge-Kutta step.",
)
@set_module_as('braincell.quad')
def diffrax_bosh3_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Bogacki-Shampine 3(2) solver.

    Wraps `diffrax.Bosh3
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Bosh3>`_,
    the third-order Bogacki-Shampine explicit Runge-Kutta method (the
    fixed-step ``BS23`` solver familiar from MATLAB's ``ode23``). Note
    that braincell drives diffrax with a fixed step, so the embedded
    error estimator that the original BS23 uses for adaptive stepping is
    discarded here.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    rk3_step, ssprk3_step : Native third-order RK alternatives.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Bosh3(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_tsit5",
    category="diffrax",
    order=5,
    description="diffrax.Tsit5 fifth-order Runge-Kutta step.",
)
@set_module_as('braincell.quad')
def diffrax_tsit5_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Tsitouras 5(4) solver.

    Wraps `diffrax.Tsit5
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5>`_,
    a fifth-order explicit Runge-Kutta method tuned to minimise the
    leading-order error coefficient. Tsit5 is the default explicit
    solver in many modern ODE libraries (e.g. Julia's
    ``DifferentialEquations.jl``).

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_dopri5_step : Dormand-Prince 5(4), an alternative fifth-order
        explicit method.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Tsit5(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_dopri5",
    category="diffrax",
    order=5,
    description="diffrax.Dopri5 Dormand-Prince 5(4) step.",
)
@set_module_as('braincell.quad')
def diffrax_dopri5_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Dormand-Prince 5(4) solver.

    Wraps `diffrax.Dopri5
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri5>`_,
    the classical explicit Dormand-Prince fifth-order Runge-Kutta method
    used by SciPy's ``RK45`` and MATLAB's ``ode45``.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_tsit5_step, diffrax_dopri8_step : Other diffrax-backed
        higher-order explicit Runge-Kutta solvers.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(diffrax.Dopri5(), target, t, dt, *args)


@_register_diffrax(
    "diffrax_dopri8",
    category="diffrax",
    order=8,
    description="diffrax.Dopri8 Dormand-Prince 8(5,3) step.",
)
@set_module_as('braincell.quad')
def diffrax_dopri8_step(target: DiffEqModule, *args):
    """Advance one step with diffrax's Dormand-Prince 8(5,3) solver.

    Wraps `diffrax.Dopri8
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri8>`_,
    a high-order explicit Runge-Kutta method (Prince-Dormand 8(7)) for
    smooth, non-stiff problems where extreme accuracy is needed.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_dopri5_step, diffrax_tsit5_step : Cheaper fifth-order
        alternatives.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
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
@set_module_as('braincell.quad')
def diffrax_bwd_euler_step(target: DiffEqModule, *args, tol=1e-5):
    """Advance one step with diffrax's implicit (backward) Euler solver.

    Wraps `diffrax.ImplicitEuler
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.ImplicitEuler>`_,
    an :math:`L`-stable first-order implicit method. The implicit
    residual is solved by a ``diffrax.VeryChord`` chord iteration with
    matching relative and absolute tolerances.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.
    tol : float, optional
        Combined relative and absolute tolerance for the chord
        root-finder. Defaults to ``1e-5``.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    backward_euler_step : Native (no-diffrax) linearised backward Euler.
    implicit_euler_step : Native Newton-iteration backward Euler.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
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
@set_module_as('braincell.quad')
def diffrax_kvaerno3_step(target: DiffEqModule, *args, tol=1e-5):
    """Advance one step with diffrax's Kvaerno 3 ESDIRK solver.

    Wraps `diffrax.Kvaerno3
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno3>`_,
    a third-order, :math:`L`-stable explicit-singly-diagonally-implicit
    Runge-Kutta (ESDIRK) method designed for stiff systems. Implicit
    stages are solved by a ``diffrax.VeryChord`` chord iteration.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.
    tol : float, optional
        Combined relative and absolute tolerance for the chord
        root-finder. Defaults to ``1e-5``.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_kvaerno4_step, diffrax_kvaerno5_step : Higher-order Kvaerno
        variants from the same family.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
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
@set_module_as('braincell.quad')
def diffrax_kvaerno4_step(target: DiffEqModule, *args, tol=1e-5):
    """Advance one step with diffrax's Kvaerno 4 ESDIRK solver.

    Wraps `diffrax.Kvaerno4
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno4>`_,
    a fourth-order, :math:`L`-stable ESDIRK method for stiff systems.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.
    tol : float, optional
        Combined relative and absolute tolerance for the chord
        root-finder. Defaults to ``1e-5``.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_kvaerno3_step, diffrax_kvaerno5_step : Other Kvaerno variants.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
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
@set_module_as('braincell.quad')
def diffrax_kvaerno5_step(target: DiffEqModule, *args, tol=1e-5):
    """Advance one step with diffrax's Kvaerno 5 ESDIRK solver.

    Wraps `diffrax.Kvaerno5
    <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno5>`_,
    a fifth-order, :math:`L`-stable ESDIRK method for stiff systems.

    Parameters
    ----------
    target : DiffEqModule
        Module whose differential states will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.
    tol : float, optional
        Combined relative and absolute tolerance for the chord
        root-finder. Defaults to ``1e-5``.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`diffrax` is not installed.

    See Also
    --------
    diffrax_kvaerno3_step, diffrax_kvaerno4_step : Lower-order Kvaerno
        variants.
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _diffrax_explicit_solver(
        diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=tol, atol=tol)),
        target, t, dt, *args
    )
