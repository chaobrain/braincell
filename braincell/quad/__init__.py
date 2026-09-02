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

"""Numerical integrators for BrainCell.

Exposes :func:`get_integrator` plus every canonical solver name
registered in :mod:`braincell.quad._registry`.

.. note::

    This ``__init__.py`` deliberately performs **side-effect imports** of
    every ``_<backend>.py`` module. Each backend self-registers into the
    integrator registry at import time via ``@register_integrator``. Do
    not lazily defer these imports — unresolved solver names would
    surface as ``ValueError: unknown integrator`` from
    :func:`get_integrator` at first call instead of at import time.
"""

# Importing the step modules below has the side effect of populating the
# global registry via @register_integrator decorators on each *_step function.
from ._backward_euler import backward_euler_step
from ._exp_euler import (
    exp_euler_step,
    ind_exp_euler_step,
)
from ._implicit import implicit_euler_step
from .protocol import (
    DiffEqGroupState,
    DiffEqModule,
    DiffEqSingleState,
    DiffEqState,
    IndependentIntegration,
    state,
    hidden_state,
    state_grouping,
)
from ._registry import (
    IntegratorEntry,
    IntegratorRegistry,
    all_integrators,
    get_integrator,
    get_registry,
    register_integrator,
)
from ._runge_kutta import (
    euler_step,
    heun2_step,
    heun3_step,
    midpoint_step,
    ralston2_step,
    ralston3_step,
    ralston4_step,
    rk2_step,
    rk3_step,
    rk4_step,
    ssprk3_step,
)
from ._staggered import dhs_voltage_step, staggered_step

# Grouped by method family rather than sorted: the Runge-Kutta block is in
# order of accuracy, which alphabetical order would scramble.
# ``__init___test`` checks membership and uniqueness but deliberately does
# not require ASCII order.
__all__ = [
    # registry
    'get_integrator',
    'register_integrator',
    'get_registry',
    'IntegratorEntry',
    'IntegratorRegistry',
    'all_integrators',
    # implicit backward Euler
    'backward_euler_step',
    # exponential Euler
    'exp_euler_step',
    'ind_exp_euler_step',
    # runge-kutta methods
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
    # staggered
    'staggered_step',
    'dhs_voltage_step',
    # implicit methods
    'implicit_euler_step',
    # protocol
    'DiffEqState',
    'DiffEqSingleState',
    'DiffEqGroupState',
    'DiffEqModule',
    'IndependentIntegration',
    'state',
    'hidden_state',
    'state_grouping',
]
