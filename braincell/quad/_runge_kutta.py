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

from dataclasses import dataclass
from typing import Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braincell._misc import cast_like as _cast_like, set_module_as
from braincell._typing import T, DT
from .protocol import DiffEqState, DiffEqModule
from ._registry import register_integrator

__all__ = [
    'euler_step',
    'midpoint_step',
    'rk2_step',
    'heun2_step',
    'ralston2_step',
    'rk3_step',
    'heun3_step',
    'ssprk3_step',
    'ralston3_step',
    'rk4_step',
    'ralston4_step',
]


@dataclass(frozen=True)
class ButcherTableau:
    """Butcher tableau for an explicit or diagonally-implicit Runge-Kutta method.

    A Butcher tableau encodes the coefficients of an :math:`s`-stage
    Runge-Kutta scheme. For a stage value :math:`k_i`,

    .. math::

        k_i = f\\!\\left(t_n + c_i \\Delta t,\\
            y_n + \\Delta t \\sum_{j=1}^{s} a_{ij} k_j\\right),
        \\qquad
        y_{n+1} = y_n + \\Delta t \\sum_{i=1}^{s} b_i k_i.

    Attributes
    ----------
    A : Sequence[Sequence[float]]
        Lower-triangular stage-coupling matrix :math:`a_{ij}`. The first row
        is conventionally empty for explicit methods.
    B : Sequence[float]
        Stage weights :math:`b_i` used to combine the :math:`k_i` into the
        final update.
    C : Sequence[float]
        Time offsets :math:`c_i` at which each stage derivative is evaluated.
    """

    A: Sequence[Sequence]  # The A matrix in the Butcher tableau.
    B: Sequence  # The B vector in the Butcher tableau.
    C: Sequence  # The C vector in the Butcher tableau.


def _array_dtype(value) -> jnp.dtype:
    return jnp.asarray(u.get_magnitude(value)).dtype


def _cast_scalar_like(value, like):
    return jnp.asarray(value, dtype=_array_dtype(like))

def _rk_update(
    coeff: Sequence,
    st: brainstate.State,
    y0: brainstate.typing.PyTree,
    dt: DT,
    *ks
):
    assert len(coeff) == len(ks), 'The number of coefficients must be equal to the number of ks.'

    def _step(y0_, *k_):
        kds = [_cast_scalar_like(c_, k_leaf) * k_leaf for c_, k_leaf in zip(coeff, k_)]
        update = kds[0]
        for kd in kds[1:]:
            update += kd
        return y0_ + update * _cast_like(dt, y0_)

    st.value = jax.tree.map(_step, y0, *ks, is_leaf=u.math.is_quantity)


def _general_rk_step(
    tableau: ButcherTableau,
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    # before one-step integration
    target.pre_integral(*args)

    # Runge-Kutta stages
    ks = []

    # k1: first derivative step
    assert len(tableau.A[0]) == 0, f'The first row of A must be empty. Got {tableau.A[0]}'
    with brainstate.environ.context(t=t), brainstate.StateTraceStack() as trace:
        # compute derivative
        target.compute_derivative(*args)

        # collection of states, initial values, and derivatives
        states = []  # states
        k1hs = []  # k1hs: k1 holder
        y0 = []  # initial values
        for st, val, writen in zip(trace.states, trace.original_state_values, trace.been_writen):
            if isinstance(st, DiffEqState):
                assert writen, f'State {st} must be written.'
                y0.append(val)
                states.append(st)
                k1hs.append(st.derivative)
            else:
                if writen:
                    raise ValueError(f'State {st} is not for integral.')
        ks.append(k1hs)

    time_like = y0[0] if y0 else dt
    t_like = _cast_like(t, time_like)
    dt_like = _cast_like(dt, time_like)

    # intermediate steps
    for i in range(1, len(tableau.C)):
        with brainstate.environ.context(
            t=t_like + _cast_scalar_like(tableau.C[i], time_like) * dt_like
        ), brainstate.check_state_value_tree():
            for st, y0_, *ks_ in zip(states, y0, *ks):
                _rk_update(tableau.A[i], st, y0_, dt, *ks_)
            target.compute_derivative(*args)
            ks.append([st.derivative for st in states])

    # final step
    with brainstate.check_state_value_tree():
        # update states with derivatives
        for st, y0_, *ks_ in zip(states, y0, *ks):
            _rk_update(tableau.B, st, y0_, dt, *ks_)

    # after one-step integration
    target.post_integral(*args)


euler_tableau = ButcherTableau(
    A=((),),
    B=(1.0,),
    C=(0.0,),
)
midpoint_tableau = ButcherTableau(
    A=[(),
       (0.5,)],
    B=(0.0, 1.0),
    C=(0.0, 0.5),
)
rk2_tableau = ButcherTableau(
    A=[(),
       (2 / 3,)],
    B=(1 / 4, 3 / 4),
    C=(0.0, 2 / 3),
)
heun2_tableau = ButcherTableau(
    A=[(),
       (1.,)],
    B=[0.5, 0.5],
    C=[0, 1],
)
ralston2_tableau = ButcherTableau(
    A=[(),
       (2 / 3,)],
    B=[0.25, 0.75],
    C=[0, 2 / 3],
)
rk3_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (-1, 2)],
    B=[1 / 6, 2 / 3, 1 / 6],
    C=[0, 0.5, 1],
)
heun3_tableau = ButcherTableau(
    A=[(),
       (1 / 3,),
       (0, 2 / 3)],
    B=[0.25, 0, 0.75],
    C=[0, 1 / 3, 2 / 3],
)
ralston3_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (0, 0.75)],
    B=[2 / 9, 1 / 3, 4 / 9],
    C=[0, 0.5, 0.75],
)
ssprk3_tableau = ButcherTableau(
    A=[(),
       (1,),
       (0.25, 0.25)],
    B=[1 / 6, 1 / 6, 2 / 3],
    C=[0, 1, 0.5],
)
rk4_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (0., 0.5),
       (0., 0., 1)],
    B=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    C=[0, 0.5, 0.5, 1],
)
ralston4_tableau = ButcherTableau(
    A=[(),
       (.4,),
       (.29697761, .15875964),
       (.21810040, -3.05096516, 3.83286476)],
    B=[.17476028, -.55148066, 1.20553560, .17118478],
    C=[0, .4, .45573725, 1],
)


@register_integrator(
    "euler",
    aliases=("explicit",),
    category="explicit",
    order=1,
    description="Forward (explicit) Euler method.",
)
@set_module_as('braincell.quad')
def euler_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with the explicit (forward) Euler method.

    Forward Euler is the simplest explicit Runge-Kutta scheme. For a system

    .. math::

        \frac{dy}{dt} = f(t, y),

    the update reads

    .. math::

        y_{n+1} = y_n + \Delta t \, f(t_n, y_n).

    The local truncation error is :math:`O(\Delta t^2)` and the global
    error is :math:`O(\Delta t)` (first-order accurate). The method is
    only conditionally stable; for stiff problems prefer
    :func:`backward_euler_step`, :func:`exp_euler_step`, or one of the
    implicit Runge-Kutta variants.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance. Its
        :meth:`pre_integral`, :meth:`compute_derivative`, and
        :meth:`post_integral` hooks are called by the underlying
        Runge-Kutta driver.
    *args
        Extra positional arguments forwarded to ``target``'s pre/derivative/post
        hooks (typically the input currents for this step).

    Returns
    -------
    None
        The state held inside *target* (every :class:`DiffEqState`) is
        updated in place.

    See Also
    --------
    midpoint_step, rk2_step, heun2_step : Second-order explicit RK schemes.
    rk4_step : Classical fourth-order Runge-Kutta.
    backward_euler_step : Implicit (backward) Euler counterpart.

    Notes
    -----
    The corresponding Butcher tableau (stored as ``euler_tableau``) is

    .. math::

        \begin{array}{c|c}
        0 & 0 \\
        \hline
          & 1
        \end{array}.

    The current time ``t`` and step size ``dt`` are read from the active
    :mod:`brainstate.environ` context.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import euler_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     euler_step(my_neuron)                       # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(euler_tableau, target, t, dt, *args)


@register_integrator(
    "midpoint",
    category="explicit",
    order=2,
    description="Explicit midpoint (modified Euler) method.",
)
@set_module_as('braincell.quad')
def midpoint_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with the explicit midpoint (modified Euler) method.

    The midpoint method is a two-stage, second-order explicit Runge-Kutta
    scheme:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{\Delta t}{2},\
            y_n + \tfrac{\Delta t}{2}\, k_1\right), \\
        y_{n+1} &= y_n + \Delta t \, k_2.

    Local truncation error is :math:`O(\Delta t^3)` and global error is
    :math:`O(\Delta t^2)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    euler_step : First-order explicit Euler.
    rk2_step, heun2_step, ralston2_step : Other second-order Runge-Kutta
        schemes.

    Notes
    -----
    Butcher tableau (``midpoint_tableau``):

    .. math::

        \begin{array}{c|cc}
        0           & 0           & 0 \\
        \tfrac{1}{2} & \tfrac{1}{2} & 0 \\
        \hline
                    & 0           & 1
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import midpoint_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     midpoint_step(my_neuron)                    # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(midpoint_tableau, target, t, dt, *args)


@register_integrator(
    "rk2",
    category="explicit",
    order=2,
    description="Generic second-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def rk2_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with a generic second-order Runge-Kutta method.

    The two-stage scheme used here is

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{2}{3}\Delta t,\
            y_n + \tfrac{2}{3}\Delta t \, k_1\right), \\
        y_{n+1} &= y_n + \Delta t \left(\tfrac{1}{4} k_1 + \tfrac{3}{4} k_2\right).

    It coincides with Ralston's two-stage method, which is the second-order
    explicit Runge-Kutta method that minimises the leading-order truncation
    error coefficient. Local truncation error is :math:`O(\Delta t^3)` and
    global error is :math:`O(\Delta t^2)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    midpoint_step, heun2_step, ralston2_step : Alternative second-order
        explicit Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``rk2_tableau``):

    .. math::

        \begin{array}{c|cc}
        0           & 0           & 0 \\
        \tfrac{2}{3} & \tfrac{2}{3} & 0 \\
        \hline
                    & \tfrac{1}{4} & \tfrac{3}{4}
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import rk2_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     rk2_step(my_neuron)                         # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk2_tableau, target, t, dt, *args)


@register_integrator(
    "heun2",
    category="explicit",
    order=2,
    description="Heun's second-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def heun2_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with Heun's second-order Runge-Kutta method.

    Heun's method (also known as the explicit trapezoidal rule or improved
    Euler) is a two-stage, second-order explicit scheme:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \Delta t,\, y_n + \Delta t \, k_1\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{2}\left(k_1 + k_2\right).

    Local truncation error is :math:`O(\Delta t^3)`; global error is
    :math:`O(\Delta t^2)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    midpoint_step, rk2_step, ralston2_step : Other second-order explicit
        Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``heun2_tableau``):

    .. math::

        \begin{array}{c|cc}
        0 & 0           & 0 \\
        1 & 1           & 0 \\
        \hline
          & \tfrac{1}{2} & \tfrac{1}{2}
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import heun2_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     heun2_step(my_neuron)                       # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(heun2_tableau, target, t, dt, *args)


@register_integrator(
    "ralston2",
    category="explicit",
    order=2,
    description="Ralston's second-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def ralston2_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with Ralston's second-order Runge-Kutta method.

    Ralston's second-order method is the two-stage explicit Runge-Kutta
    scheme that minimises the leading-order truncation error coefficient
    among second-order RK2 variants [1]_:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{2}{3}\Delta t,\
            y_n + \tfrac{2}{3}\Delta t \, k_1\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{4}\left(k_1 + 3 k_2\right).

    It is therefore identical to :func:`rk2_step` and useful when an
    error-optimal RK2 scheme is desired without committing to higher
    cost. Local truncation error is :math:`O(\Delta t^3)`; global error
    is :math:`O(\Delta t^2)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    rk2_step, midpoint_step, heun2_step : Alternative second-order
        explicit Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``ralston2_tableau``):

    .. math::

        \begin{array}{c|cc}
        0           & 0           & 0 \\
        \tfrac{2}{3} & \tfrac{2}{3} & 0 \\
        \hline
                    & \tfrac{1}{4} & \tfrac{3}{4}
        \end{array}

    References
    ----------
    .. [1] Ralston, A. (1962). "Runge-Kutta methods with minimum error
           bounds." *Mathematics of Computation*, 16(80), 431-437.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import ralston2_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     ralston2_step(my_neuron)                    # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston2_tableau, target, t, dt, *args)


@register_integrator(
    "rk3",
    category="explicit",
    order=3,
    description="Classical third-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def rk3_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with the classical third-order Runge-Kutta method.

    The classical (Kutta's) three-stage RK3 scheme is

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{1}{2}\Delta t,\
            y_n + \tfrac{1}{2}\Delta t \, k_1\right), \\
        k_3 &= f\!\left(t_n + \Delta t,\
            y_n - \Delta t \, k_1 + 2 \Delta t \, k_2\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{6}\left(k_1 + 4 k_2 + k_3\right).

    Local truncation error is :math:`O(\Delta t^4)`; global error is
    :math:`O(\Delta t^3)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    heun3_step, ralston3_step, ssprk3_step : Alternative third-order
        explicit Runge-Kutta variants.
    rk4_step : Fourth-order classical Runge-Kutta.

    Notes
    -----
    Butcher tableau (``rk3_tableau``):

    .. math::

        \begin{array}{c|ccc}
        0           & 0           & 0 & 0 \\
        \tfrac{1}{2} & \tfrac{1}{2} & 0 & 0 \\
        1           & -1          & 2 & 0 \\
        \hline
                    & \tfrac{1}{6} & \tfrac{2}{3} & \tfrac{1}{6}
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import rk3_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     rk3_step(my_neuron)                         # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk3_tableau, target, t, dt, *args)


@register_integrator(
    "heun3",
    category="explicit",
    order=3,
    description="Heun's third-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def heun3_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with Heun's third-order Runge-Kutta method.

    Heun's three-stage third-order Runge-Kutta scheme:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{1}{3}\Delta t,\
            y_n + \tfrac{1}{3}\Delta t \, k_1\right), \\
        k_3 &= f\!\left(t_n + \tfrac{2}{3}\Delta t,\
            y_n + \tfrac{2}{3}\Delta t \, k_2\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{4}\left(k_1 + 3 k_3\right).

    Local truncation error is :math:`O(\Delta t^4)`; global error is
    :math:`O(\Delta t^3)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    rk3_step, ralston3_step, ssprk3_step : Alternative third-order
        explicit Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``heun3_tableau``):

    .. math::

        \begin{array}{c|ccc}
        0           & 0           & 0           & 0 \\
        \tfrac{1}{3} & \tfrac{1}{3} & 0           & 0 \\
        \tfrac{2}{3} & 0           & \tfrac{2}{3} & 0 \\
        \hline
                    & \tfrac{1}{4} & 0           & \tfrac{3}{4}
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import heun3_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     heun3_step(my_neuron)                       # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(heun3_tableau, target, t, dt, *args)


@register_integrator(
    "ssprk3",
    category="explicit",
    order=3,
    description="Strong-stability-preserving third-order Runge-Kutta (SSPRK3).",
)
@set_module_as('braincell.quad')
def ssprk3_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with the Strong-Stability-Preserving RK3 method.

    The Shu-Osher third-order strong-stability-preserving Runge-Kutta
    scheme (SSPRK3) is a convex combination of forward Euler steps [1]_:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \Delta t,\, y_n + \Delta t \, k_1\right), \\
        k_3 &= f\!\left(t_n + \tfrac{1}{2}\Delta t,\
            y_n + \tfrac{1}{4}\Delta t \, k_1 + \tfrac{1}{4}\Delta t \, k_2\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{6}\left(k_1 + k_2 + 4 k_3\right).

    Because every stage can be written as a convex combination of forward
    Euler updates, SSPRK3 inherits the monotonicity (TVD) properties of
    forward Euler under the same CFL number. This makes it the explicit
    method of choice for problems with discontinuities or sharp gradients
    where preserving positivity matters. Local truncation error is
    :math:`O(\Delta t^4)`; global error is :math:`O(\Delta t^3)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    rk3_step, heun3_step, ralston3_step : Alternative third-order
        explicit Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``ssprk3_tableau``):

    .. math::

        \begin{array}{c|ccc}
        0           & 0           & 0           & 0 \\
        1           & 1           & 0           & 0 \\
        \tfrac{1}{2} & \tfrac{1}{4} & \tfrac{1}{4} & 0 \\
        \hline
                    & \tfrac{1}{6} & \tfrac{1}{6} & \tfrac{2}{3}
        \end{array}

    References
    ----------
    .. [1] Shu, C.-W. and Osher, S. (1988). "Efficient implementation of
           essentially non-oscillatory shock-capturing schemes."
           *Journal of Computational Physics*, 77(2), 439-471.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import ssprk3_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     ssprk3_step(my_neuron)                      # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(ssprk3_tableau, target, t, dt, *args)


@register_integrator(
    "ralston3",
    category="explicit",
    order=3,
    description="Ralston's third-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def ralston3_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with Ralston's third-order Runge-Kutta method.

    Ralston's three-stage third-order RK method is the third-order
    explicit Runge-Kutta scheme that minimises the leading-order
    truncation error coefficient [1]_:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{1}{2}\Delta t,\
            y_n + \tfrac{1}{2}\Delta t \, k_1\right), \\
        k_3 &= f\!\left(t_n + \tfrac{3}{4}\Delta t,\
            y_n + \tfrac{3}{4}\Delta t \, k_2\right), \\
        y_{n+1} &= y_n + \Delta t \left(\tfrac{2}{9} k_1
            + \tfrac{1}{3} k_2 + \tfrac{4}{9} k_3\right).

    Local truncation error is :math:`O(\Delta t^4)`; global error is
    :math:`O(\Delta t^3)`.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    rk3_step, heun3_step, ssprk3_step : Alternative third-order
        explicit Runge-Kutta variants.

    Notes
    -----
    Butcher tableau (``ralston3_tableau``):

    .. math::

        \begin{array}{c|ccc}
        0           & 0           & 0           & 0 \\
        \tfrac{1}{2} & \tfrac{1}{2} & 0           & 0 \\
        \tfrac{3}{4} & 0           & \tfrac{3}{4} & 0 \\
        \hline
                    & \tfrac{2}{9} & \tfrac{1}{3} & \tfrac{4}{9}
        \end{array}

    References
    ----------
    .. [1] Ralston, A. (1962). "Runge-Kutta methods with minimum error
           bounds." *Mathematics of Computation*, 16(80), 431-437.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import ralston3_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     ralston3_step(my_neuron)                    # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston3_tableau, target, t, dt, *args)


@register_integrator(
    "rk4",
    category="explicit",
    order=4,
    description="Classical four-stage fourth-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def rk4_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with the classical fourth-order Runge-Kutta method.

    The classical RK4 scheme is the four-stage, fourth-order explicit
    Runge-Kutta method:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f\!\left(t_n + \tfrac{\Delta t}{2},\
            y_n + \tfrac{\Delta t}{2} k_1\right), \\
        k_3 &= f\!\left(t_n + \tfrac{\Delta t}{2},\
            y_n + \tfrac{\Delta t}{2} k_2\right), \\
        k_4 &= f\!\left(t_n + \Delta t,\, y_n + \Delta t \, k_3\right), \\
        y_{n+1} &= y_n + \tfrac{\Delta t}{6}\left(k_1 + 2 k_2 + 2 k_3 + k_4\right).

    Local truncation error is :math:`O(\Delta t^5)`; global error is
    :math:`O(\Delta t^4)`. RK4 is the canonical fixed-step ODE solver and
    is a sensible default whenever the right-hand side is smooth and the
    problem is not stiff.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    ralston4_step : An error-optimal four-stage fourth-order variant.
    rk3_step : Lower-order three-stage scheme with cheaper steps.

    Notes
    -----
    Butcher tableau (``rk4_tableau``):

    .. math::

        \begin{array}{c|cccc}
        0           & 0           & 0           & 0 & 0 \\
        \tfrac{1}{2} & \tfrac{1}{2} & 0           & 0 & 0 \\
        \tfrac{1}{2} & 0           & \tfrac{1}{2} & 0 & 0 \\
        1           & 0           & 0           & 1 & 0 \\
        \hline
                    & \tfrac{1}{6} & \tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{6}
        \end{array}

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import rk4_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     rk4_step(my_neuron)                         # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk4_tableau, target, t, dt, *args)


@register_integrator(
    "ralston4",
    category="explicit",
    order=4,
    description="Ralston's fourth-order Runge-Kutta method.",
)
@set_module_as('braincell.quad')
def ralston4_step(
    target: DiffEqModule,
    *args,
):
    r"""Advance one step with Ralston's fourth-order Runge-Kutta method.

    Ralston's four-stage fourth-order RK method uses non-rational
    coefficients chosen to minimise the leading-order truncation error
    coefficient relative to classical RK4 [1]_:

    .. math::

        k_1 &= f(t_n, y_n), \\
        k_2 &= f(t_n + 0.4\,\Delta t,\
                  y_n + 0.4\,\Delta t \, k_1), \\
        k_3 &= f(t_n + 0.45573725\,\Delta t,\
                  y_n + 0.29697761\,\Delta t \, k_1
                      + 0.15875964\,\Delta t \, k_2), \\
        k_4 &= f(t_n + \Delta t,\
                  y_n + 0.21810040\,\Delta t \, k_1
                      - 3.05096516\,\Delta t \, k_2
                      + 3.83286476\,\Delta t \, k_3), \\
        y_{n+1} &= y_n + \Delta t \,\bigl(
                  0.17476028\, k_1
                - 0.55148066\, k_2
                + 1.20553560\, k_3
                + 0.17118478\, k_4 \bigr).

    Local truncation error is :math:`O(\Delta t^5)`; global error is
    :math:`O(\Delta t^4)`. The negative weight on :math:`k_2` makes this
    scheme slightly more sensitive to floating-point cancellation than
    classical :func:`rk4_step`, but the leading error constant is smaller.

    Parameters
    ----------
    target : DiffEqModule
        Differential-equation module to advance.
    *args
        Extra positional arguments forwarded to ``target``'s integration
        hooks.

    Returns
    -------
    None
        Updates *target*'s state in place.

    See Also
    --------
    rk4_step : Classical fourth-order Runge-Kutta with rational weights.

    Notes
    -----
    The Butcher tableau is stored as ``ralston4_tableau``.

    References
    ----------
    .. [1] Ralston, A. (1962). "Runge-Kutta methods with minimum error
           bounds." *Mathematics of Computation*, 16(80), 431-437.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import ralston4_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.01 * u.ms):
        ...     ralston4_step(my_neuron)                    # doctest: +SKIP
    """
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston4_tableau, target, t, dt, *args)
