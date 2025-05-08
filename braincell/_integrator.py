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

from __future__ import annotations

from typing import Callable

from ._integrator_exp_euler import *
from ._integrator_implicit import *
from ._integrator_runge_kutta import *


__all__ = [
    'get_integrator',
]

all_integrators = {
    # exponential Euler
    'exp_euler': exp_euler_step,

    # explicit Runge-Kutta methods
    'euler': euler_step,
    'midpoint': midpoint_step,
    'rk2': rk2_step,
    'heun2': heun2_step,
    'ralston2': ralston2_step,
    'rk3': rk3_step,
    'heun3': heun3_step,
    'ssprk3': ssprk3_step,
    'ralston3': ralston3_step,
    'rk4': rk4_step,
    'ralston4': ralston4_step,

    # splitting methods
    'implicit_euler': implicit_euler_step,
    'splitting': splitting_step,
    'cn_rk4': cn_rk4_step,
    'cn_exp_euler': cn_exp_euler_step,
    'implicit_rk4': implicit_rk4_step,
    'implicit_exp_euler': implicit_exp_euler_step,
    'exp_exp_euler': exp_exp_euler_step,
}


def get_integrator(
    method: str | Callable
) -> Callable:
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
