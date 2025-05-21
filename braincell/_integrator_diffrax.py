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

import importlib.util
from typing import Optional, Tuple, Callable, Dict, Any

import brainstate
import brainunit as u
import jax

from ._integrator_util import _check_diffeq_state_derivative
from ._protocol import DiffEqState, DiffEqModule

diffrax_installed = importlib.util.find_spec('diffrax') is not None
if diffrax_installed:
    import diffrax as dfx

    diffrax_solvers = {
        # explicit RK
        'euler': dfx.Euler,
        'revheun': dfx.ReversibleHeun,
        'heun': dfx.Heun,
        'midpoint': dfx.Midpoint,
        'ralston': dfx.Ralston,
        'bosh3': dfx.Bosh3,
        'tsit5': dfx.Tsit5,
        'dopri5': dfx.Dopri5,
        'dopri8': dfx.Dopri8,

        # implicit RK
        'ieuler': dfx.ImplicitEuler,
        'kvaerno3': dfx.Kvaerno3,
        'kvaerno4': dfx.Kvaerno4,
        'kvaerno5': dfx.Kvaerno5,
    }

__all__ = [
    'diffrax_solver_step',
]



def diffrax_solver_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    pass

