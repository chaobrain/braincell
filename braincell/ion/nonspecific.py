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

from typing import Optional

import brainunit as u

from braincell._base import Ion
from braincell._typing import Initializer, Size
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

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Ion`.
    name : str or None, optional
        Runtime instance name. Defaults to ``None``, in which case the
        instance is unnamed. Forwarded unchanged to :class:`Ion`.
    **channels
        Channel instances to attach to this placeholder ion, forwarded
        unchanged to :class:`Ion`.

    See Also
    --------
    NonSpecificFixed : Fixed-parameter placeholder ion built on this
        base.
    braincell.channel.Kv1p5_MA2020_GrC : Shipped channel that declares a
        nonspecific current owner via ``current_owner_types``.

    Notes
    -----
    This class does not represent a real chemical species and does not
    define concentration dynamics. It is intended for mechanisms such as
    NEURON ``USEION no WRITE ino`` declarations, where ``no`` is a
    current owner name rather than a physiologically conserved ion pool.

    ``default_Ci``, ``default_Co``, and ``default_valence`` are arbitrary
    placeholder values, chosen only so that the ordinary ``Ion``
    interfaces (concentration lookups, Nernst-derived reversal
    potentials, etc.) resolve without error. They are not measured or
    physiological values, and no Nernst potential computed from them is
    biologically meaningful.

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
        Placeholder intracellular concentration. Defaults to ``None``,
        which falls back to :attr:`NonSpecific.default_Ci` inside
        :meth:`FixedIon._init_fixed_ion`.
    Co : array-like or callable or None, optional
        Placeholder extracellular concentration. Defaults to ``None``,
        which falls back to :attr:`NonSpecific.default_Co` inside
        :meth:`FixedIon._init_fixed_ion`.
    valence : array-like or callable or None, optional
        Placeholder valence. Defaults to ``None``, which falls back to
        :attr:`NonSpecific.default_valence` inside
        :meth:`FixedIon._init_fixed_ion`.
    name : str or None, optional
        Runtime ion instance name.
    **channels
        Optional channels added directly to the placeholder ion.

    Raises
    ------
    ValueError
        If ``E`` is explicitly passed as ``None``.
        :meth:`FixedIon._init_fixed_ion` requires an explicit fixed
        reversal potential and does not fall back to a class default
        for ``E``.

    See Also
    --------
    NonSpecific : Base placeholder ion family this class fixes.

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
        size: Size,
        E: Optional[Initializer] = 0.0 * u.mV,
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)
