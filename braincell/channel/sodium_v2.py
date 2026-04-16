# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Union

import brainstate
import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel._template import Gate
from braincell.channel._template import HH
from braincell.ion import Sodium
from braincell.mech import register_channel

__all__ = [
    "INa_HH1952_v2",
]


@register_channel("INa_HH1952_v2")
class INa_HH1952_v2(HH):
    """Template-based prototype of :class:`braincell.channel.sodium.INa_HH1952`."""

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("p", power=3),
        Gate("q", power=1),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120. * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -45. * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 5
        return 1. / u.math.exprel(-temp / 10)

    def f_p_beta(self, V, Na: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 4.0 * u.math.exp(-(V + 20) / 18)

    def f_q_alpha(self, V, Na: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.07 * u.math.exp(-(V + 20) / 20.)

    def f_q_beta(self, V, Na: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1 / (1 + u.math.exp(-(V - 10) / 10))

    def current(self, V, Na: IonInfo):
        return self.g_max * self.conductance_factor(V, Na) * (Na.E - V)
