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

"""Nonspecific current-owner ion placeholders."""

from typing import Callable, Optional, Union

import brainstate
import brainunit as u

from braincell._base import Ion
from braincell.ion._base import FixedIon
from braincell.mech import register_ion

__all__ = [
    "NonSpecific",
    "NonSpecificFixed",
]


class NonSpecific(Ion):
    """Base class for NEURON-style nonspecific current ownership.

    ``NonSpecific`` is a placeholder ion family used when an NMODL
    mechanism declares a written nonspecific current with ``USEION``. It
    exists so BrainCell can bind and probe that current separately from
    ordinary ionic currents while preserving the usual ion/current
    container interfaces.

    Notes
    -----
    This class does not represent a real chemical species and does not
    define concentration dynamics. It is intended for mechanisms such as
    NEURON ``USEION no WRITE ino`` declarations, where ``no`` is a
    current owner name rather than a physiologically conserved ion pool.

    Attributes
    ----------
    ion_symbol : str
        Symbol used for runtime family lookup.
    default_Ci : brainunit.Quantity
        Placeholder intracellular concentration.
    default_Co : brainunit.Quantity
        Placeholder extracellular concentration.
    default_valence : int
        Placeholder charge valence.
    """

    __module__ = "braincell.ion"
    ion_symbol = "no"
    default_Ci = 1.0 * u.mM
    default_Co = 1.0 * u.mM
    default_valence = 1


@register_ion("NonSpecificFixed")
class NonSpecificFixed(NonSpecific, FixedIon):
    """Fixed nonspecific current-owner placeholder.

    Parameters
    ----------
    size : brainstate.typing.Size
        Runtime variable shape for this placeholder ion.
    E : array-like or callable or None, optional
        Fixed reversal potential used only by channels that choose to
        read ``No.E``. Defaults to ``0 mV``.
    Ci : array-like or callable or None, optional
        Placeholder intracellular concentration. Defaults to
        :attr:`NonSpecific.default_Ci`.
    Co : array-like or callable or None, optional
        Placeholder extracellular concentration. Defaults to
        :attr:`NonSpecific.default_Co`.
    valence : array-like or callable or None, optional
        Placeholder valence. Defaults to
        :attr:`NonSpecific.default_valence`.
    name : str or None, optional
        Runtime ion instance name.
    **channels
        Optional channels added directly to the placeholder ion.

    Notes
    -----
    This class is intentionally minimal. It lets a channel expose a
    nonspecific current component through ``Ion.current(...)`` without
    changing the global membrane-current API. It should not be used as a
    substitute for sodium, potassium, calcium, or other explicit ion
    species when those concentrations carry model semantics.
    """

    __module__ = "braincell.ion"

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable, None] = 0.0 * u.mV,
        Ci: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        valence: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)
