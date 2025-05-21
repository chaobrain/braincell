# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Callable

from ._integrator_diffrax import *
from ._integrator_exp_euler import *
from ._integrator_runge_kutta import *

__all__ = [
    'get_integrator',

    # exponential Euler
    'exp_euler_step',

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

    # diffrax methods
    'diffrax_euler_step',
    'diffrax_heun_step',
    'diffrax_midpoint_step',
    'diffrax_ralston_step',
    'diffrax_bosh3_step',
    'diffrax_tsit5_step',
    'diffrax_dopri5_step',
    'diffrax_dopri8_step',
]

all_integrators = {k.replace('_step', ''): v for k, v in locals().items() if k.endswith('_step')}


def get_integrator(method: str | Callable) -> Callable:
    """
    Get the integrator function by name or return the provided callable.

    This function retrieves the appropriate integrator function based on the input.
    If a string is provided, it looks up the corresponding integrator in the
    `all_integrators` dictionary. If a callable is provided, it returns that callable directly.

    Args:
        method (str | Callable): The numerical integrator name as a string or a callable function.
            If a string, it should be one of the keys in the `all_integrators` dictionary.
            If a callable, it should be a valid integrator function.

    Returns:
        Callable: The integrator function corresponding to the input method.

    Raises:
        ValueError: If the input method is neither a valid string key in `all_integrators`
            nor a callable function.

    Examples::
        >>> get_integrator('euler')
        <function euler_step at ...>
        >>> get_integrator(custom_integrator_function)
        <function custom_integrator_function at ...>
    """
    if isinstance(method, str):
        return all_integrators[method]
    elif callable(method):
        return method
    else:
        raise ValueError(f"Invalid integrator method: {method}")
