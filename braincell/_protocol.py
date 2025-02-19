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

import brainstate
from brainstate._state import record_state_value_write

__all__ = [
    'DiffEqState',
    'DiffEqModule',
]


class DiffEqState(brainstate.ShortTermState):
    """
    A state that integrates the state of the system to the integral of the state.

    Attributes
    ----------
    derivative: The derivative of the differential equation state.
    diffusion: The diffusion of the differential equation state.

    """

    __module__ = 'braincell'

    # derivative of this state
    derivative: brainstate.typing.PyTree
    diffusion: brainstate.typing.PyTree

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derivative = None
        self._diffusion = None

    @property
    def derivative(self):
        """
        The derivative of the state.

        It is used to compute the derivative of the ODE system,
        or the drift of the SDE system.
        """
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        record_state_value_write(self)
        self._derivative = value

    @property
    def diffusion(self):
        """
        The diffusion of the state.

        It is used to compute the diffusion of the SDE system.
        If it is None, the system is considered as an ODE system.
        """
        return self._diffusion

    @diffusion.setter
    def diffusion(self, value):
        record_state_value_write(self)
        self._diffusion = value


class DiffEqModule(brainstate.mixin.Mixin):
    """
    The module for defining the differential equations.
    """
    __module__ = 'braincell'

    def pre_integral(self, *args, **kwargs):
        pass

    def compute_derivative(self, *args, **kwargs):
        raise NotImplementedError

    def post_integral(self, *args, **kwargs):
        pass
