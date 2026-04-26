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


"""Voltage-dependent calcium channels built directly on templates."""

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braincell._base import HHTypedNeuron, IonInfo
from braincell.channel._base import Gate, HH, ghk_flux
from braincell.ion import Calcium
from braincell.mech import register_channel

__all__ = [
    "CaN_IS2008",
    "CaT_HM1992",
    "CaT_HP1992",
    "CaHT_HM1992",
    "CaHT_Re1993",
    "CaL_IS2008",
    "Cav1p2_MA2020",
    "Cav1p3_MA2020",
    "Cav3p1_MA2020",
    "CaHVA_MA2020_GoC",
    "CaHVA_MA2020_GrC",
    "Cav2p3_MA2020_GoC",
    "Ca_ZH2019_IO",
]


@register_channel("CaN_IS2008")
class CaN_IS2008(HH):
    r"""Inoue & Strowbridge 2008 calcium-activated non-selective cation current."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * u.mV,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        modulation = Ca.Ci / (Ca.Ci + 0.2 * u.mM)
        return self.g_max * modulation * self.conductance_factor(V, Ca) * (self.E - V)

    def f_p_inf(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 43.0) / 5.2))

    def f_p_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 2.7 / (u.math.exp(-(V + 55.0) / 15.0) + u.math.exp((V + 55.0) / 15.0)) + 1.6


@register_channel("CaT_HM1992")
class CaT_HM1992(HH):
    r"""Huguenard & McCormick 1992 low-threshold T-type calcium current."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 3.55,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -3.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_p = braintools.init.param(q10_p, self.varshape, allow_none=False)
        self.temp_ref_p = braintools.init.param(temp_ref_p, self.varshape, allow_none=False)
        self.q10_q = braintools.init.param(q10_q, self.varshape, allow_none=False)
        self.temp_ref_q = braintools.init.param(temp_ref_q, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_p_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 59.0) / 6.2))

    def f_p_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp(-(V + 132.0) / 16.7) + u.math.exp((V + 16.8) / 18.2)) + 0.612

    def f_q_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 83.0) / 4.0))

    def f_q_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V >= -80.0,
            u.math.exp(-(V + 22.0) / 10.5) + 28.0,
            u.math.exp((V + 467.0) / 66.6),
        )


@register_channel("CaT_HP1992")
class CaT_HP1992(HH):
    r"""Huguenard & Prince 1992 T-type calcium current for reticular nucleus."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.75 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 5.0,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -3.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_p = braintools.init.param(q10_p, self.varshape, allow_none=False)
        self.temp_ref_p = braintools.init.param(temp_ref_p, self.varshape, allow_none=False)
        self.q10_q = braintools.init.param(q10_q, self.varshape, allow_none=False)
        self.temp_ref_q = braintools.init.param(temp_ref_q, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_p_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 52.0) / 7.4))

    def f_p_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 3.0 + 1.0 / (u.math.exp((V + 27.0) / 10.0) + u.math.exp(-(V + 102.0) / 15.0))

    def f_q_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 5.0))

    def f_q_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 85.0 + 1.0 / (u.math.exp((V + 48.0) / 4.0) + u.math.exp(-(V + 407.0) / 50.0))


@register_channel("CaHT_HM1992")
class CaHT_HM1992(HH):
    r"""Huguenard & McCormick 1992 high-threshold calcium current."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 3.55,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 25.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_p = braintools.init.param(q10_p, self.varshape, allow_none=False)
        self.temp_ref_p = braintools.init.param(temp_ref_p, self.varshape, allow_none=False)
        self.q10_q = braintools.init.param(q10_q, self.varshape, allow_none=False)
        self.temp_ref_q = braintools.init.param(temp_ref_q, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_p_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 59.0) / 6.2))

    def f_p_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp(-(V + 132.0) / 16.7) + u.math.exp((V + 16.8) / 18.2)) + 0.612

    def f_q_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 83.0) / 4.0))

    def f_q_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V >= -80.0,
            u.math.exp(-(V + 22.0) / 10.5) + 28.0,
            u.math.exp((V + 467.0) / 66.6),
        )


@register_channel("CaHT_Re1993")
class CaHT_Re1993(HH):
    r"""Reuveni 1993 high-threshold calcium current."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 2.3,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 2.3,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_p = braintools.init.param(q10_p, self.varshape, allow_none=False)
        self.temp_ref_p = braintools.init.param(temp_ref_p, self.varshape, allow_none=False)
        self.q10_q = braintools.init.param(q10_q, self.varshape, allow_none=False)
        self.temp_ref_q = braintools.init.param(temp_ref_q, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_p_alpha(self, V, *unused):
        _ = unused
        temp = (-V + self.V_sh).to_decimal(u.mV)
        delta = -27.0 + temp
        return 0.055 * delta / (u.math.exp(delta / 3.8) - 1.0)

    def f_p_beta(self, V, *unused):
        _ = unused
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.94 * u.math.exp((-75.0 + temp) / 17.0)

    def f_q_alpha(self, V, *unused):
        _ = unused
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.000457 * u.math.exp((-13.0 + temp) / 50.0)

    def f_q_beta(self, V, *unused):
        _ = unused
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.0065 / (u.math.exp((-15.0 + temp) / 28.0) + 1.0)


@register_channel("CaL_IS2008")
class CaL_IS2008(HH):
    r"""Inoue & Strowbridge 2008 L-type calcium current."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 3.55,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(24.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_p = braintools.init.param(q10_p, self.varshape, allow_none=False)
        self.temp_ref_p = braintools.init.param(temp_ref_p, self.varshape, allow_none=False)
        self.q10_q = braintools.init.param(q10_q, self.varshape, allow_none=False)
        self.temp_ref_q = braintools.init.param(temp_ref_q, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_p_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 10.0) / 4.0))

    def f_p_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.4 + 0.7 / (u.math.exp(-(V + 5.0) / 15.0) + u.math.exp((V + 5.0) / 15.0))

    def f_q_inf(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 25.0) / 2.0))

    def f_q_tau(self, V, *unused):
        _ = unused
        V = (V - self.V_sh).to_decimal(u.mV)
        return 300.0 + 100.0 / (u.math.exp((V + 40.0) / 9.5) + u.math.exp(-(V + 40.0) / 9.5))


@register_channel("Cav1p2_MA2020")
class Cav1p2_MA2020(HH):
    r"""Evans/Beining Cav1.2 calcium current with calcium-dependent inactivation."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
        Gate("h", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
        Gate("n", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.kf = 0.0005
        self.VDI = 0.17

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_m_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 8.9) / -6.7))

    def f_h_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.VDI / (1.0 + u.math.exp((V + 55.0) / 8.0)) + (1.0 - self.VDI)

    def f_n_inf(self, V, Ca: IonInfo, *unused):
        _ = (V, unused)
        return self.kf / (self.kf + Ca.Ci / u.mM)

    def f_m_tau(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        m_alpha = 39800.0 * (V + 8.124) / (u.math.exp((V + 8.124) / 9.005) - 1.0)
        m_beta = 990.0 * u.math.exp(V / 31.4)
        return 1.0 / (m_alpha + m_beta)

    def f_h_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 44.3

    def f_n_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 0.5


@register_channel("Cav1p3_MA2020")
class Cav1p3_MA2020(HH):
    r"""Evans/Beining Cav1.3 calcium current with calcium-dependent inactivation."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
        Gate("h", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
        Gate("n", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.mS / u.cm ** 2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.kf = 0.0005
        self.VDI = 1.0

    def current(self, V, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_m_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (u.math.exp((V - (-40.0)) / -5.0) + 1.0)

    def f_h_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.VDI / (u.math.exp((V - (-37.0)) / 5.0) + 1.0) + (1.0 - self.VDI)

    def f_n_inf(self, V, Ca: IonInfo, *unused):
        _ = (V, unused)
        return self.kf / (self.kf + Ca.Ci / u.mM)

    def f_m_tau(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        m_alpha = 39800.0 * 15.005 / u.math.exprel((V + 67.24) / 15.005)
        m_beta = 3500.0 * u.math.exp(V / 31.4)
        return 1.0 / (m_alpha + m_beta)

    def f_h_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 44.3

    def f_n_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 0.5


@register_channel("Cav3p1_MA2020")
class Cav3p1_MA2020(HH):
    r"""Purkinje cell Cav3.1 low-threshold calcium current with GHK drive."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
        Gate("q", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.5e-4 * (u.cm / u.second),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(37.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.v0_m_inf = -52.0 * u.mV
        self.v0_h_inf = -72.0 * u.mV
        self.k_m_inf = -5.0 * u.mV
        self.k_h_inf = 7.0 * u.mV
        self.C_tau_m = 1.0
        self.A_tau_m = 1.0
        self.v0_tau_m1 = -40.0 * u.mV
        self.v0_tau_m2 = -102.0 * u.mV
        self.k_tau_m1 = 9.0 * u.mV
        self.k_tau_m2 = -18.0 * u.mV
        self.C_tau_h = 15.0
        self.A_tau_h = 1.0
        self.v0_tau_h1 = -32.0 * u.mV
        self.k_tau_h1 = 7.0 * u.mV
        self.z = 2

    def current(self, V, Ca: IonInfo):
        drive = ghk_flux(V=V, ci=Ca.Ci, co=Ca.Co, z=self.z, T=self.temp)
        return -self.g_max * self.conductance_factor(V, Ca) * drive

    def f_p_inf(self, V, *unused):
        _ = unused
        return 1.0 / (1.0 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V, *unused):
        _ = unused
        return 1.0 / (1.0 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V, *unused):
        _ = unused
        return u.math.where(
            V <= -90.0 * u.mV,
            1.0,
            self.C_tau_m
            + self.A_tau_m
            / (
                u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1)
                + u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
            ),
        )

    def f_q_tau(self, V, *unused):
        _ = unused
        return self.C_tau_h + self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1)

@register_channel("CaHVA_MA2020_GoC")
class CaHVA_MA2020_GoC(HH):
    """Template-based import of ``CaHVA_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("s", power=2, q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
        Gate("u", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.46 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_s = 0.04944
        self.Kalpha_s = 15.87301587302
        self.V0alpha_s = -29.06
        self.Abeta_s = 0.08298
        self.Kbeta_s = -25.641
        self.V0beta_s = -18.66
        self.Aalpha_u = 0.0013
        self.Kalpha_u = -18.183
        self.V0alpha_u = -48.0
        self.Abeta_u = 0.0013
        self.Kbeta_u = 83.33
        self.V0beta_u = -48.0

    def current(self, V, Ca: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_s_alpha(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_s * u.math.exp((V - self.V0alpha_s) / self.Kalpha_s)

    def f_s_beta(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_s * u.math.exp((V - self.V0beta_s) / self.Kbeta_s)

    def f_u_alpha(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_u * u.math.exp((V - self.V0alpha_u) / self.Kalpha_u)

    def f_u_beta(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_u * u.math.exp((V - self.V0beta_u) / self.Kbeta_u)

@register_channel("CaHVA_MA2020_GrC")
class CaHVA_MA2020_GrC(HH):
    """Template-based import of ``CaHVA_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("s", power=2, q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
        Gate("u", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.46 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_s = 0.04944
        self.Kalpha_s = 15.87301587302
        self.V0alpha_s = -29.06
        self.Abeta_s = 0.08298
        self.Kbeta_s = -25.641
        self.V0beta_s = -18.66
        self.Aalpha_u = 0.0013
        self.Kalpha_u = -18.183
        self.V0alpha_u = -48.0
        self.Abeta_u = 0.0013
        self.Kbeta_u = 83.33
        self.V0beta_u = -48.0

    def current(self, V, Ca: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_s_alpha(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_s * u.math.exp((V - self.V0alpha_s) / self.Kalpha_s)

    def f_s_beta(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_s * u.math.exp((V - self.V0beta_s) / self.Kbeta_s)

    def f_u_alpha(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_u * u.math.exp((V - self.V0alpha_u) / self.Kalpha_u)

    def f_u_beta(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_u * u.math.exp((V - self.V0beta_u) / self.Kbeta_u)

@register_channel("Cav2p3_MA2020_GoC")
class Cav2p3_MA2020_GoC(HH):
    """Template-based import of ``Cav2p3_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(34.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Ca) * (Ca.E - V)

    def f_m_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 48.5) / -3.0))

    def f_h_inf(self, V, Ca: IonInfo, *unused):
        _ = (Ca, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 53.0) / 1.0))

    def f_m_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 50.0

    def f_h_tau(self, V, Ca: IonInfo, *unused):
        _ = (V, Ca, unused)
        return 5.0

@register_channel("Ca_ZH2019_IO")
class Ca_ZH2019_IO(HH):
    """Template-based import of ``Ca_ZH2019_IO.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.4 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = 120.0 * u.mV,
        mMidV: Union[brainstate.typing.ArrayLike, Callable] = -61.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.mMidV = braintools.init.param(mMidV, self.varshape, allow_none=False)

    def current(self, V):
        return self.g_max * self.f_m_inf(V) * self.h.value * (self.E - V)

    def f_m_inf(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        m_mid = self.mMidV.to_decimal(u.mV)
        term = 1.0 + u.math.exp((m_mid - V) / 4.2)
        return 1.0 / (term * term * term)

    def f_h_inf(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 85.5) / 8.6))

    def f_h_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 40.0 + 30.0 * (
            1.0 / (1.0 + u.math.exp((V + 84.0) / 7.3))
        ) * u.math.exp((V + 160.0) / 30.0)

