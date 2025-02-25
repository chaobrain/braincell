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

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import brainstate
import brainunit as u
import jax

from ._misc import set_module_as
from ._protocol import DiffEqState, DiffEqModule

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
    *ks
):
    assert len(coeff) == len(ks), 'The number of coefficients must be equal to the number of ks.'

    def _step(y0_, *k_):
        kds = [c_ * k_ for c_, k_ in zip(coeff, k_)]
        update = kds[0]
        for kd in kds[1:]:
            update += kd
        return y0_ + update * brainstate.environ.get_dt()

    st.value = jax.tree.map(_step, y0, *ks, is_leaf=u.math.is_quantity)


@set_module_as('braincell')
def _general_rk_step(
    tableau: ButcherTableau,
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    dt = brainstate.environ.get_dt()

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
                _rk_update(tableau.A[i], st, y0_, *ks_)
            target.compute_derivative(*args)
            ks.append([st.derivative for st in states])

    # final step
    with brainstate.check_state_value_tree():
        # update states with derivatives
        for st, y0_, *ks_ in zip(states, y0, *ks):
            _rk_update(tableau.B, st, y0_, *ks_)

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
    t: u.Quantity[u.second],
    *args,
):
    """
    The Euler step for the differential equations.
    """
    _general_rk_step(euler_tableau, target, t, *args)


@set_module_as('braincell')
def midpoint_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The midpoint step for the differential equations.
    """
    _general_rk_step(midpoint_tableau, target, t, *args)


@set_module_as('braincell')
def rk2_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The second-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(rk2_tableau, target, t, *args)


@set_module_as('braincell')
def heun2_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Heun's second-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(heun2_tableau, target, t, *args)


@set_module_as('braincell')
def ralston2_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Ralston's second-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(ralston2_tableau, target, t, *args)


@set_module_as('braincell')
def rk3_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The third-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(rk3_tableau, target, t, *args)


@set_module_as('braincell')
def heun3_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Heun's third-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(heun3_tableau, target, t, *args)


@set_module_as('braincell')
def ssprk3_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Strong Stability Preserving Runge-Kutta 3rd order step for the differential equations.
    """
    _general_rk_step(ssprk3_tableau, target, t, *args)


@set_module_as('braincell')
def ralston3_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Ralston's third-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(ralston3_tableau, target, t, *args)


@set_module_as('braincell')
def rk4_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The fourth-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(rk4_tableau, target, t, *args)


@set_module_as('braincell')
def ralston4_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
):
    """
    The Ralston's fourth-order Runge-Kutta step for the differential equations.
    """
    _general_rk_step(ralston4_tableau, target, t, *args)
