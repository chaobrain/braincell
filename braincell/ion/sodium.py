# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
import braintools
import brainunit as u

from braincell._base import Ion
from braincell.mech import register_ion
from braincell.ion._template import FixedIon, InitNernstIon

__all__ = [
    'Sodium',
    'SodiumFixed',
    'SodiumInitNernst',
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
    ion_symbol = 'Na'
    default_Ci = 10. * u.mM
    default_Co = 140. * u.mM
    default_valence = 1


@register_ion("SodiumFixed")
class SodiumFixed(Sodium, FixedIon):
    """
    Fixed Sodium dynamics.

    This calcium model has no dynamics. It holds fixed reversal
    potential :math:`E` and concentration :math:`C`.
    """
    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable, None] = 50. * u.mV,
        Ci: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        valence: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("SodiumInitNernst")
class SodiumInitNernst(Sodium, InitNernstIon):
    """Fixed ``Ci/Co`` sodium model with ``E`` initialized from Nernst."""

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        Ci: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        valence: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_nernst_ion(Ci=Ci, Co=Co, temp=temp, valence=valence)
