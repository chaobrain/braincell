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

from ._integrator_exp_euler import ind_exp_euler_step
from ._integrator_protocol import DiffEqModule
from ._integrator_voltage_solver import dhs_voltage_step
from ._misc import set_module_as

__all__ = [
    'staggered_step',
]


@set_module_as('braincell')
def staggered_step(
    target: DiffEqModule,
    *args
):
    from ._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment), (
        f"The stagger integrator only support {MultiCompartment.__name__}, "
        f"but we got {type(target)} instead."
        f"The stagger integrator only support {MultiCompartment.__name__}, "
        f"but we got {type(target)} instead."
    )
    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')

    # voltage integration
    dhs_voltage_step(target, t, dt, *args)

    # ind_exp_euler for ion channels
    ind_exp_euler_step(target, t, dt, *args, excluded_paths=[('V',)])
