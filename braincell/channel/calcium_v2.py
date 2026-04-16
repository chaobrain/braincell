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
from braincell.channel._template import ghk_flux
from braincell.ion import Calcium
from braincell.mech import register_channel

__all__ = [
    "ICav31_Ma2020_v2",
]


@register_channel("ICav31_Ma2020_v2")
class ICav31_Ma2020_v2(HH):
    """Template-based prototype of :class:`braincell.channel.calcium.ICav31_Ma2020`."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10= 3, temp_ref=u.celsius2kelvin(37.0)),
        Gate("q", power=1, q10= 3, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        p_max: Union[brainstate.typing.ArrayLike, Callable] = 2.5e-4 * (u.cm / u.second),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)

        self.p_max = braintools.init.param(p_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

        self.v0_m_inf = -52 * u.mV
        self.v0_h_inf = -72 * u.mV
        self.k_m_inf = -5 * u.mV
        self.k_h_inf = 7 * u.mV

        self.C_tau_m = 1
        self.A_tau_m = 1.0
        self.v0_tau_m1 = -40 * u.mV
        self.v0_tau_m2 = -102 * u.mV
        self.k_tau_m1 = 9 * u.mV
        self.k_tau_m2 = -18 * u.mV

        self.C_tau_h = 15
        self.A_tau_h = 1.0
        self.v0_tau_h1 = -32 * u.mV
        self.k_tau_h1 = 7 * u.mV
        self.z = 2
        self.Co = 2.0 * u.mM

    def f_p_inf(self, V, Ca: IonInfo):
        return 1.0 / (1 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V, Ca: IonInfo):
        return 1.0 / (1 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V, Ca: IonInfo):
        return u.math.where(
            V <= -90 * u.mV,
            1.,
            (
                self.C_tau_m
                + self.A_tau_m / (
                    u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1)
                    + u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
                )
            ),
        )

    def f_q_tau(self, V, Ca: IonInfo):
        return self.C_tau_h + self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1)

    def current(self, V, Ca: IonInfo):
        return -self.p_max * self.conductance_factor(V, Ca) * ghk_flux(
            V=V,
            ci=Ca.C,
            co=self.Co,
            z=self.z,
            T=self.temp,
        )
