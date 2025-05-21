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
from typing import Callable

import brainunit as u

from ._integrator_util import apply_standard_solver_step
from ._protocol import DiffEqModule

__all__ = [
    'diffrax_euler_step',
    'diffrax_heun_step',
    'diffrax_midpoint_step',
    'diffrax_ralston_step',
    'diffrax_bosh3_step',
    'diffrax_tsit5_step',
    'diffrax_dopri5_step',
    'diffrax_dopri8_step',
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


def _explicit_solver(solver, fn: Callable, y0, t0, dt, args=()):
    t0 = u.Quantity(t0)
    dt = u.Quantity(dt).to_decimal(t0.unit)
    t0 = t0.magnitude
    y1, _, _, state, _ = solver.step(
        diffrax.ODETerm(lambda t, y, args_: fn(t, y, *args_)[0]),
        t0, t0 + dt, y0, args, (False, y0), made_jump=False
    )
    return y1, {}


def _diffrax_explicit_solver(
    solver,
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    apply_standard_solver_step(
        functools.partial(_explicit_solver, solver),
        target,
        t,
        *args,
        merging_method='stack'
    )


def diffrax_euler_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Euler(), target, t, *args)


def diffrax_heun_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Heun(), target, t, *args)


def diffrax_midpoint_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Midpoint(), target, t, *args)


def diffrax_ralston_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Ralston(), target, t, *args)


def diffrax_bosh3_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Bosh3(), target, t, *args)


def diffrax_tsit5_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Tsit5(), target, t, *args)


def diffrax_dopri5_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Dopri5(), target, t, *args)


def diffrax_dopri8_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    _diffrax_explicit_solver(diffrax.Dopri8(), target, t, *args)
