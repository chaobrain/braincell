# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import brainstate
import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel._template import Passive
from braincell.ion import Potassium
from braincell.mech import register_channel

__all__ = [
    "IK_Leak_v2",
]


@register_channel("IK_Leak_v2")
class IK_Leak_v2(Passive):
    """Template-based prototype of :class:`braincell.channel.potassium.IK_Leak`."""

    __module__ = "braincell.channel"
    root_type = Potassium

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[int, brainstate.typing.ArrayLike, Callable] = 0.005 * (u.mS * u.cm ** -2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)
