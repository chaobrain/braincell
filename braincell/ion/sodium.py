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

from typing import Union, Callable, Optional

import brainstate
import brainunit as u

from braincell._base import Ion

__all__ = [
    'Sodium',
    'SodiumFixed',
]


class Sodium(Ion):
    """
    Base class for modeling Sodium ion.

    This class serves as a foundation for creating specific sodium ion models
    in neuronal simulations. It inherits from the Ion base class and provides
    a starting point for implementing various sodium dynamics.

    Note:
        This is an abstract base class and should be subclassed to implement
        specific sodium ion models with defined dynamics and properties.
    """
    __module__ = 'braincell.ion'


class SodiumFixed(Sodium):
    """
    Fixed Sodium dynamics.

    This calcium model has no dynamics. It holds fixed reversal
    potential :math:`E` and concentration :math:`C`.
    """
    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable] = 50. * u.mV,
        C: Union[brainstate.typing.ArrayLike, Callable] = 0.0400811 * u.mM,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self.E = brainstate.init.param(E, self.varshape, allow_none=False)
        self.C = brainstate.init.param(C, self.varshape, allow_none=False)
