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

from ._integrators_exp_euler import *
from ._integrators_runge_kutta import *

__all__ = [
    'get_integrator',
]

all_integrators = {
    'exp_euler': exp_euler_step,
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
}


def get_integrator(name: str) -> Callable:
    """
    Get the integrator function by name.

    Args:
      name: The name of the integrator.

    Returns:
      The integrator function.
    """
    return all_integrators[name]
