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

from ._integrator_protocol import DiffEqState, DiffEqModule
from ._misc import set_module_as
from ._typing import T, DT

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
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    A: Sequence[Sequence]  # The A matrix in the Butcher tableau.
    B: Sequence  # The B vector in the Butcher tableau.
    C: Sequence  # The C vector in the Butcher tableau.


def _rk_update(
    coeff: Sequence,
    st: brainstate.State,
    y0: brainstate.typing.PyTree,
    dt: DT,
    *ks
):
    assert len(coeff) == len(ks), 'The number of coefficients must be equal to the number of ks.'

    def _step(y0_, *k_):
        kds = [c_ * k_ for c_, k_ in zip(coeff, k_)]
        update = kds[0]
        for kd in kds[1:]:
            update += kd
        return y0_ + update * dt

    st.value = jax.tree.map(_step, y0, *ks, is_leaf=u.math.is_quantity)


@set_module_as('braincell')
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
    with brainstate.environ.context(t=t + tableau.C[0] * dt), brainstate.StateTraceStack() as trace:
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

    # intermediate steps
    for i in range(1, len(tableau.C)):
        with brainstate.environ.context(t=t + tableau.C[i] * dt), brainstate.check_state_value_tree():
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


@set_module_as('braincell')
def euler_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the Euler method for solving differential equations.

    This function applies the Euler method, which is the simplest explicit method
    for numerical integration of ordinary differential equations.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the Euler method
    approximates the solution using the following equation:

    $$
    y_{n+1} = y_n + \Delta t \cdot f(t_n, y_n)
    $$

    Where:
    - $y_n$ is the current state
    - $t_n$ is the current time
    - $\Delta t$ is the time step
    - $f(t_n, y_n)$ is the right-hand side of the differential equation

    The local truncation error of the Euler method is $O(\Delta t^2)$, and the global
    truncation error is $O(\Delta t)$.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    The Euler method uses the simplest Butcher tableau:

    $$
    \begin{array}{c|c}
    0 & 0 \\
    \hline
    & 1
    \end{array}
    $$

    This tableau is defined elsewhere in the module as `euler_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(euler_tableau, target, t, dt, *args)


@set_module_as('braincell')
def midpoint_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the midpoint method for solving differential equations.

    This function applies the midpoint method, which is a second-order Runge-Kutta method
    that provides improved accuracy over the Euler method.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the midpoint method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} k_1) \\
    y_{n+1} &= y_n + \Delta t \cdot k_2
    \end{align*}
    $$

    Where:
    - $y_n$ is the current state
    - $t_n$ is the current time
    - $\Delta t$ is the time step

    The local truncation error of the midpoint method is $O(\Delta t^3)$, and the global
    truncation error is $O(\Delta t^2)$.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    The midpoint method uses the following Butcher tableau:

    $$
    \begin{array}{c|cc}
    0 & 0 & 0 \\
    \frac{1}{2} & \frac{1}{2} & 0 \\
    \hline
    & 0 & 1
    \end{array}
    $$

    This tableau is defined elsewhere in the module as `midpoint_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(midpoint_tableau, target, t, dt, *args)


@set_module_as('braincell')
def rk2_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the second-order Runge-Kutta method for solving differential equations.

    This function applies the second-order Runge-Kutta method, which is an explicit
    integration scheme that provides improved accuracy over the Euler method.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the second-order Runge-Kutta method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{2}{3}\Delta t, y_n + \frac{2}{3}\Delta t \cdot k_1) \\
    y_{n+1} &= y_n + \Delta t \cdot (\frac{1}{4}k_1 + \frac{3}{4}k_2)
    \end{align*}
    $$

    Where:
    - $y_n$ is the current state
    - $t_n$ is the current time
    - $\Delta t$ is the time step

    The local truncation error of this method is $O(\Delta t^3)$, and the global
    truncation error is $O(\Delta t^2)$.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    ----
    This second-order Runge-Kutta method uses the following Butcher tableau:

    $$
    \begin{array}{c|cc}
    0 & 0 & 0 \\
    \frac{2}{3} & \frac{2}{3} & 0 \\
    \hline
    & \frac{1}{4} & \frac{3}{4}
    \end{array}
    $$

    This tableau is defined elsewhere in the module as `rk2_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk2_tableau, target, t, dt, *args)


@set_module_as('braincell')
def heun2_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of Heun's second-order Runge-Kutta method for solving differential equations.

    This function applies Heun's second-order Runge-Kutta method, which is an explicit
    integration scheme that provides improved accuracy over the Euler method.

    Mathematical Description
    ------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, Heun's second-order method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \Delta t, y_n + \Delta t \cdot k_1)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{2}(k_1 + k_2)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    This method has a local truncation error of $O(\Delta t^3)$ and a global truncation error of $O(\Delta t^2)$.

    Parameters
    ----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Returns
    -------
    None
        This function updates the state of the target in-place and does not return a value.

    Note
    ----
    This method uses the Butcher tableau specific to Heun's second-order method,
    which is defined elsewhere in the module as `heun2_tableau`. The Butcher tableau for this method is:

    $$
    \begin{array}{c|cc}
    0 & 0 & 0 \\
    1 & 1 & 0 \\
    \hline
    & \frac{1}{2} & \frac{1}{2}
    \end{array}
    $$
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(heun2_tableau, target, t, dt, *args)


@set_module_as('braincell')
def ralston2_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of Ralston's second-order Runge-Kutta method for solving differential equations.

    This function applies Ralston's second-order Runge-Kutta method, which is an explicit
    integration scheme designed to minimize the truncation error for a given step size.

    Mathematical Description
    ------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, Ralston's second-order method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{2}{3}\Delta t, y_n + \frac{2}{3}\Delta t \cdot k_1)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{4}(k_1 + 3k_2)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    This method has a local truncation error of $O(\Delta t^3)$ and a global truncation error of $O(\Delta t^2)$.

    Parameters
    ----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Returns
    -------
    None
        This function updates the state of the target in-place and does not return a value.

    Note
    ----
    This method uses the Butcher tableau specific to Ralston's second-order method,
    which is defined elsewhere in the module as `ralston2_tableau`. The Butcher tableau for this method is:

    $$
    \begin{array}{c|cc}
    0 & 0 & 0 \\
    \frac{2}{3} & \frac{2}{3} & 0 \\
    \hline
    & \frac{1}{4} & \frac{3}{4}
    \end{array}
    $$
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston2_tableau, target, t, dt, *args)


@set_module_as('braincell')
def rk3_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the third-order Runge-Kutta method for solving differential equations.

    This function applies the third-order Runge-Kutta method, which is an explicit
    integration scheme that provides improved accuracy over lower-order methods.

    Mathematical Description
    ------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the third-order Runge-Kutta method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{1}{2}\Delta t, y_n + \frac{1}{2}\Delta t \cdot k_1) \\
    k_3 &= f(t_n + \Delta t, y_n - \Delta t \cdot k_1 + 2\Delta t \cdot k_2)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 4k_2 + k_3)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    ----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to the third-order Runge-Kutta method,
    which is defined elsewhere in the module as `rk3_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk3_tableau, target, t, dt, *args)


@set_module_as('braincell')
def heun3_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of Heun's third-order Runge-Kutta method for solving differential equations.

    This function applies Heun's third-order Runge-Kutta method, which is an explicit
    integration scheme that provides improved accuracy over lower-order methods.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, Heun's third-order method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{1}{3}\Delta t, y_n + \frac{1}{3}\Delta t \cdot k_1) \\
    k_3 &= f(t_n + \frac{2}{3}\Delta t, y_n + \frac{2}{3}\Delta t \cdot k_2)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{4}(k_1 + 3k_3)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to Heun's third-order method,
    which is defined elsewhere in the module as `heun3_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(heun3_tableau, target, t, dt, *args)


@set_module_as('braincell')
def ssprk3_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the Strong Stability Preserving Runge-Kutta 3rd order (SSPRK3) method for solving differential equations.

    This function applies the SSPRK3 method, which is designed to maintain strong stability properties
    for certain classes of differential equations, particularly those with discontinuities or sharp gradients.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the SSPRK3 method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \Delta t, y_n + \Delta t \cdot k_1) \\
    k_3 &= f(t_n + \frac{1}{2}\Delta t, y_n + \frac{1}{4}\Delta t \cdot k_1 + \frac{1}{4}\Delta t \cdot k_2)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + k_2 + 4k_3)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to the SSPRK3 method,
    which is defined elsewhere in the module as `ssprk3_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(ssprk3_tableau, target, t, dt, *args)


@set_module_as('braincell')
def ralston3_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of Ralston's third-order Runge-Kutta method for solving differential equations.

    This function applies Ralston's third-order Runge-Kutta method, which is an explicit
    integration scheme designed to minimize the truncation error for a given step size.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, Ralston's third-order method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{1}{2}\Delta t, y_n + \frac{1}{2}\Delta t \cdot k_1) \\
    k_3 &= f(t_n + \frac{3}{4}\Delta t, y_n + \frac{3}{4}\Delta t \cdot k_2)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \Delta t \cdot (\frac{2}{9}k_1 + \frac{1}{3}k_2 + \frac{4}{9}k_3)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to Ralston's third-order method,
    which is defined elsewhere in the module as `ralston3_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston3_tableau, target, t, dt, *args)


@set_module_as('braincell')
def rk4_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of the fourth-order Runge-Kutta method (RK4) for solving differential equations.

    This function applies the classical RK4 method to numerically integrate a system of 
    differential equations. RK4 is a widely used method that provides a good balance 
    between accuracy and computational cost.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the RK4 method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} k_1) \\
    k_3 &= f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} k_2) \\
    k_4 &= f(t_n + \Delta t, y_n + \Delta t k_3)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to the classical fourth-order Runge-Kutta method,
    which is defined elsewhere in the module as `rk4_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(rk4_tableau, target, t, dt, *args)


@set_module_as('braincell')
def ralston4_step(
    target: DiffEqModule,
    *args,
):
    r"""
    Perform a single step of Ralston's fourth-order Runge-Kutta method for solving differential equations.

    This function applies Ralston's fourth-order Runge-Kutta method, which is an explicit
    integration scheme designed to minimize the truncation error for a given step size.

    Mathematical Description
    -------------------------
    For a differential equation of the form $\frac{dy}{dt} = f(t, y)$, the Ralston's fourth-order method
    approximates the solution using the following steps:

    $$
    \begin{align*}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + 0.4\Delta t, y_n + 0.4\Delta t \cdot k_1) \\
    k_3 &= f(t_n + 0.45573725\Delta t, y_n + 0.29697761\Delta t \cdot k_1 + 0.15875964\Delta t \cdot k_2) \\
    k_4 &= f(t_n + \Delta t, y_n + 0.21810040\Delta t \cdot k_1 - 3.05096516\Delta t \cdot k_2 + 3.83286476\Delta t \cdot k_3)
    \end{align*}
    $$

    The final step is:

    $$
    y_{n+1} = y_n + \Delta t \cdot (0.17476028 \cdot k_1 - 0.55148066 \cdot k_2 + 1.20553560 \cdot k_3 + 0.17118478 \cdot k_4)
    $$

    Where $\Delta t$ is the time step, and $t_n$ and $y_n$ are the time and state at the n-th step.

    Parameters
    -----------
    target : DiffEqModule
        The differential equation module that defines the system to be integrated.
    *args
        Additional arguments to be passed to the target's methods.

    Note
    -----
    This method uses the Butcher tableau specific to Ralston's fourth-order method,
    which is defined elsewhere in the module as `ralston4_tableau`.
    """
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')
    _general_rk_step(ralston4_tableau, target, t, dt, *args)
