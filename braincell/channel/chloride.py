# -*- coding: utf-8 -*-
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


"""Chloride channels: the passive chloride leak."""

from typing import Optional

import braintools
import brainunit as u

from braincell._typing import Initializer, Size
from braincell.channel._base import OhmicHH
from braincell.ion import Chloride
from braincell.mech import register_channel

__all__ = [
    "Cl_Leak",
]


@register_channel("Cl_Leak")
class Cl_Leak(OhmicHH):
    r"""Chloride leak current.

    A voltage-independent, always-open chloride conductance:

    .. math::

        I = g_{max} \, (E_{Cl} - V)

    with ``I`` positive inward, as for every braincell channel. The
    reversal potential is not a parameter: it is taken from the chloride
    ion this channel is attached to, so ``E_Cl`` follows whatever the ion
    computes (fixed, Nernst, or a concentration pool such as
    :class:`~braincell.ion.Cldif_JE2011`). The current is the chloride
    ion's, so it enters that ion's ``current(V)`` and drives a
    concentration pool. There are no gating variables and no state to
    integrate.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Leak conductance density. Defaults to ``0.1 mS/cm2``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.channel.K_Leak : The potassium counterpart.
    braincell.channel.IL : Ion-independent leak with its own fixed
        reversal potential parameter.
    braincell.ion.Chloride : Ion family supplying ``E_Cl``.

    Notes
    -----
    This class has **no primary literature source**: the current law is
    Ohm's law and the ``0.1 mS/cm2`` default is a placeholder that any
    caller overrides, so the class ships without a ``References``
    section.

    ``root_type`` is :class:`~braincell.ion.Chloride`, so the channel
    must be attached to a chloride ion; a Cell that paints it without
    declaring one raises ``KeyError`` at ``init_state``.

    A tonic (extrasynaptic) GABA_A current of total conductance
    :math:`g_t` whose bicarbonate share is :math:`P`,

    .. math::

        I = g_t \left[(1 - P)(E_{Cl} - V) + P (E_{HCO_3} - V)\right],

    is this channel with ``g_max`` :math:`= (1 - P)\, g_t` plus
    :class:`~braincell.channel.IL` with ``g_max`` :math:`= P\, g_t` and
    ``E`` :math:`= E_{HCO_3}`; only the chloride part enters the
    chloride ion. For example, :math:`g_t = 0.1` mS/cm², :math:`P = 0.18`,
    :math:`E_{HCO_3} = -12.892` mV (16 / 26 mM at 35 °C) and
    :math:`E_{Cl} = -87.222` mV give -1.3843 µA/cm² at -60 mV.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell as bc
        >>> cell = bc.Cell.point(1, length=20.0 * u.um, radius=0.5 * u.um)
        >>> cell.paint(bc.mech.Ion("ChlorideInitNernst", name="cl"))
        >>> cell.paint(bc.mech.Channel("Cl_Leak", g_max=0.1 * u.mS / u.cm**2))
    """

    __module__ = "braincell.channel"
    root_type = Chloride
    #: No gating variables: the empty tuple makes ``conductance_factor``
    #: return 1 and turns the whole lifecycle into the template's no-ops.
    gates = ()

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.1 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
