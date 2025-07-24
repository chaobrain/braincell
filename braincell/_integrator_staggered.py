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

import brainstate
import brainunit as u

from ._integrator_exp_euler import ind_exp_euler_step
from ._integrator_voltage_solver import dhs_voltage_step
from ._misc import set_module_as
from ._integrator_protocol import DiffEqState, DiffEqModule

__all__ = [
    'staggered_step',
]


@set_module_as('braincell')
def staggered_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    dt: u.Quantity[u.second],
    *args
):
    from ._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment), (
        f"The stagger integrator only support {MultiCompartment.__name__}, "
        f"but we got {type(target)} instead."
    )

    dhs_voltage_step(target, t, dt, *args)

    # # excluded_paths
    # all_states = brainstate.graph.states(target)
    # diffeq_states, _ = all_states.split(DiffEqState, ...)
    # excluded_paths = [('V',)]
    # for key in diffeq_states.keys():
    #     if 'INa_Rsg' in key:
    #         excluded_paths.append(key)

    # update markov
    for _ in range(2):
        target.update_state(*args)
        target.pre_integral(*args)

    # ind_exp_euler for non-v and non-markov
    ind_exp_euler_step(target, t, dt, *args, excluded_paths=[('V',)])
