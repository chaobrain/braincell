# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Explicit Runge-Kutta steps built on the :mod:`braintools.quad` kernels.

braincell keeps no Butcher tables of its own. :func:`module_step_from_ode`
turns any braintools ODE stepper ``step(f, y, t)`` into a module step
``step(target, *args) -> None`` that advances every :class:`DiffEqState`
written by ``target.compute_derivative`` in place.
"""

import threading
from typing import Callable

import brainstate
import braintools.quad as btq

from braincell._misc import cast_like as _cast_like
from braincell.quad._util import environ_time, split_diffeq_states
from braincell.quad.protocol import DiffEqModule, DiffEqState

__all__ = [
    'module_step_from_ode',
    'euler_step',
    'midpoint_step',
    'heun2_step',
    'ralston2_step',
    'rk3_step',
    'heun3_step',
    'ssprk3_step',
    'ralston3_step',
    'rk4_step',
    'ralston4_step',
]

_CACHE: dict[Callable, Callable] = {}
_CACHE_LOCK = threading.Lock()


def _independent_state_ids(target: DiffEqModule) -> set[int]:
    """Ids of the :class:`DiffEqState` leaves owned by ``IndependentIntegration`` submodules."""
    all_states, integrated, _ = split_diffeq_states(target)
    integrated_ids = {id(st) for st in integrated.values()}
    return {id(st) for st in all_states.values() if isinstance(st, DiffEqState) and id(st) not in integrated_ids}


def _discover(target: DiffEqModule, t, args):
    """Evaluate the first stage and collect the integrated states.

    Every :class:`DiffEqState` written by ``compute_derivative`` is integrated;
    writing any other state is an error, as for the previous braincell kernels.
    A :class:`DiffEqState` that is only read must belong to an
    :class:`~braincell.quad.IndependentIntegration` submodule (a Markov channel
    or a kinetic ion): the host advances it through ``ind_update``, and the
    stages see it as a constant. Any other unwritten one is an error.
    """
    with brainstate.environ.context(t=t), brainstate.StateTraceStack() as trace:
        target.compute_derivative(*args)
        states, y0, k1, unwritten = [], [], [], []
        for st, val, written in zip(trace.states, trace.original_state_values, trace.been_writen):
            if isinstance(st, DiffEqState):
                if not written:
                    unwritten.append(st)
                    continue
                states.append(st)
                y0.append(val)
                k1.append(st.derivative)
            elif written:
                raise ValueError(f'State {st} is not for integral.')
    if unwritten:
        # Only traversed when needed: the common model writes every state it reads.
        independent = _independent_state_ids(target)
        for st in unwritten:
            if id(st) not in independent:
                raise ValueError(f'State {st} must be written by compute_derivative().')
    return states, tuple(y0), tuple(k1)


def _make_module_step(ode_step: Callable) -> Callable:
    def step(target: DiffEqModule, *args) -> None:
        t, dt = environ_time(target)
        target.pre_integral(*args)
        states, y0, k1 = _discover(target, t, args)
        time_like = y0[0] if y0 else dt
        t0 = _cast_like(t, time_like)

        def f(y, t_stage):
            # braintools' explicit steppers evaluate the first stage at the very
            # ``y`` they were given; reuse it instead of a second evaluation.
            if y is y0:
                return k1
            with brainstate.environ.context(t=t_stage), brainstate.check_state_value_tree():
                for st, value in zip(states, y):
                    st.value = value
                target.compute_derivative(*args)
                return tuple(st.derivative for st in states)

        y1 = ode_step(f, y0, t0)
        with brainstate.check_state_value_tree():
            for st, value in zip(states, y1):
                st.value = value
        target.post_integral(*args)

    step.__name__ = getattr(ode_step, '__name__', 'ode_step')
    step.__qualname__ = step.__name__
    step.__module__ = 'braincell.quad'
    step.__wrapped_ode_step__ = ode_step
    step.__doc__ = (
        f'Module step wrapping :func:`{getattr(ode_step, "__module__", "")}.{step.__name__}`.\n\n'
        'Called as ``step(target, *args)``; see :func:`braincell.quad.module_step_from_ode`.'
    )
    return step


def module_step_from_ode(ode_step: Callable) -> Callable:
    """Wrap a braintools ODE stepper as a braincell module step.

    The returned ``step(target, *args) -> None`` advances ``target`` by one
    step of ``brainstate.environ.get_dt()``:

    1. calls ``target.pre_integral(*args)``;
    2. evaluates ``target.compute_derivative(*args)`` once at the current
       time; every :class:`DiffEqState` it writes is integrated (writing any
       other state raises ``ValueError``);
    3. calls ``ode_step(f, y0, t)``, where ``f(y, t_stage)`` writes ``y``
       into those states, sets ``t_stage`` as the environment time and
       returns their derivatives;
    4. writes the result back and calls ``target.post_integral(*args)``.

    Parameters
    ----------
    ode_step : callable
        A stepper of the braintools ``'ode'`` kind,
        ``ode_step(f, y, t) -> y_next``, that accepts a tuple of states
        (every built-in except ``ode_expeuler_step``, which takes a single
        array).

    Returns
    -------
    callable
        The module step. The same object is returned for the same
        ``ode_step``, so ``module_step_from_ode(ode_rk4_step) is rk4_step``.

    Examples
    --------

    .. code-block:: python

        >>> import braintools
        >>> from braincell.quad import module_step_from_ode, rk4_step
        >>> module_step_from_ode(braintools.quad.ode_rk4_step) is rk4_step
        True
    """
    if not callable(ode_step):
        raise TypeError(f'ode_step must be callable, got {type(ode_step).__name__}.')
    with _CACHE_LOCK:
        step = _CACHE.get(ode_step)
        if step is None:
            step = _CACHE[ode_step] = _make_module_step(ode_step)
    return step


module_step_from_ode.__module__ = 'braincell.quad'


def _named(name: str, ode_step: Callable, doc: str) -> Callable:
    step = module_step_from_ode(ode_step)
    # The braincell names stay the public identity of these steps.
    step.__name__ = step.__qualname__ = name
    step.__doc__ = doc
    return step


_DOC = """Advance ``target`` by one step of {method}.

Wraps :func:`braintools.quad.{ode}` (order {order}) with
:func:`module_step_from_ode`.

Parameters
----------
target : DiffEqModule
    Differential-equation module to advance. Its ``pre_integral``,
    ``compute_derivative`` and ``post_integral`` hooks are called.
*args
    Extra positional arguments forwarded to the hooks (typically the input
    current for this step).

Returns
-------
None
    Every :class:`DiffEqState` written by ``compute_derivative`` is updated in
    place. ``t`` and ``dt`` are read from :mod:`brainstate.environ`.
"""

_STEPS = (
    ('euler_step', btq.ode_euler_step, 'the explicit (forward) Euler method', 1),
    ('midpoint_step', btq.ode_midpoint_step, 'the explicit midpoint method', 2),
    ('heun2_step', btq.ode_rk2_step, "Heun's second-order method", 2),
    ('ralston2_step', btq.ode_ralston2_step, "Ralston's second-order method", 2),
    ('rk3_step', btq.ode_rk3_step, "Kutta's third-order method", 3),
    ('heun3_step', btq.ode_heun_step, "Heun's third-order method", 3),
    ('ssprk3_step', btq.ode_ssprk33_step, 'the strong-stability-preserving RK(3,3) method', 3),
    ('ralston3_step', btq.ode_ralston3_step, "Ralston's third-order method", 3),
    ('rk4_step', btq.ode_rk4_step, 'the classical fourth-order Runge-Kutta method', 4),
    ('ralston4_step', btq.ode_ralston4_step, "Ralston's fourth-order method", 4),
)

(
    euler_step,
    midpoint_step,
    heun2_step,
    ralston2_step,
    rk3_step,
    heun3_step,
    ssprk3_step,
    ralston3_step,
    rk4_step,
    ralston4_step,
) = (
    _named(name, ode, _DOC.format(method=method, ode=ode.__name__, order=order)) for name, ode, method, order in _STEPS
)
