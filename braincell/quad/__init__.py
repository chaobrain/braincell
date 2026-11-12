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

Exposes :func:`get_integrator`, :func:`integrator_info` and
:func:`list_integrators`, which resolve names through the
:mod:`braintools.quad` registry, plus braincell's step functions and the
stateful integration protocol.

.. note::

    This ``__init__.py`` deliberately performs **side-effect imports** of
    every ``_<backend>.py`` module. Each backend registers its steps in the
    ``'module'`` kind of :mod:`braintools.quad` at import time via
    ``@register_module_step``. Do not lazily defer these imports; unresolved
    solver names would surface as ``ValueError: Unknown integrator`` from
    :func:`get_integrator` at first call instead of at import time.
"""

# Importing the step modules below registers braincell's module steps in the
# braintools.quad registry via @register_module_step.
from braincell.quad._backward_euler import backward_euler_step
from braincell.quad._exp_euler import exp_euler_step, ind_exp_euler_step
from braincell.quad._implicit import crank_nicolson_step
from braincell.quad.protocol import (
    DiffEqGroupState,
    DiffEqModule,
    DiffEqSingleState,
    DiffEqState,
    IndependentIntegration,
    state,
    hidden_state,
    state_grouping,
)
from braincell.quad._registry import (
    get_integrator,
    integrator_info,
    list_integrators,
)
from braincell.quad._explicit import (
    module_step_from_ode,
    euler_step,
    heun2_step,
    heun3_step,
    midpoint_step,
    ralston2_step,
    ralston3_step,
    ralston4_step,
    rk3_step,
    rk4_step,
    ssprk3_step,
)
from braincell.quad._staggered import dhs_voltage_step, staggered_step
from braincell._deprecation import removed_attributes as _removed_attributes

# Grouped by method family rather than sorted: the Runge-Kutta block is in
# order of accuracy, which alphabetical order would scramble.
# ``__init___test`` checks membership and uniqueness but deliberately does
# not require ASCII order.
__all__ = [
    # name resolution
    'get_integrator',
    'integrator_info',
    'list_integrators',
    'module_step_from_ode',
    # implicit backward Euler
    'backward_euler_step',
    # exponential Euler
    'exp_euler_step',
    'ind_exp_euler_step',
    # runge-kutta methods
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
    # staggered
    'staggered_step',
    'dhs_voltage_step',
    # implicit methods
    'crank_nicolson_step',
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

#: Removed names and their replacements, raised as ``AttributeError``
#: (``ImportError`` on a ``from braincell.quad import ...`` line).
_REMOVED_ATTRIBUTES = {
    'rk2_step': (
        "it was Ralston's second-order method; use "
        "braincell.quad.ralston2_step (same numbers), or heun2_step for Heun's method."
    ),
    'get_registry': "use braincell.quad.integrator_info(name) and braincell.quad.list_integrators().",
    'all_integrators': (
        "use {name: braincell.quad.get_integrator(name) for name in braincell.quad.list_integrators()}."
    ),
    'register_integrator': (
        "register module steps with braintools.quad.register_method(name, fn, kind='module', ...); "
        "pure-function ODE steppers registered with kind='ode' are wrapped automatically."
    ),
    'IntegratorRegistry': "use braincell.quad.integrator_info(name) and braincell.quad.list_integrators().",
    'IntegratorEntry': "use braincell.quad.integrator_info(name), which returns a braintools.quad.MethodInfo.",
    'implicit_euler_step': (
        "it solved the trapezoidal residual with a Newton loop that stopped after one iteration; "
        "use braincell.quad.crank_nicolson_step (converged trapezoidal rule, results change on "
        "nonlinear models) or braincell.quad.backward_euler_step (first-order implicit Euler)."
    ),
}

__getattr__ = _removed_attributes(
    __name__,
    {name: f'braincell.quad.{name} was removed; {hint}' for name, hint in _REMOVED_ATTRIBUTES.items()},
)
