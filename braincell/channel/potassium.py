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


"""Voltage-dependent potassium channels built directly on templates."""

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braincell._base import Channel, IonInfo
from braincell.channel._base import Gate, HH
from braincell.ion import Potassium
from braincell.mech import register_channel

__all__ = [
    "KDR_Ba2002",
    "K_TM1991",
    "K_HH1952",
    "KA1_HM1992",
    "KA2_HM1992",
    "KK2A_HM1992",
    "KK2B_HM1992",
    "KNI_Ya1989",
    "K_Leak",
    "K_Kv_test",
    "fKdr_SU2015_DCN",
    "sKdr_SU2015_DCN",
    "KM_RI2021_SC",
    "Kir2p3_MA2025_BC",
    "Kir2p3_MA2024_PC",
    "Kir2p3_RI2021_SC",
    "Kv1p1_MA2025_BC",
    "Kv1p1_MA2024_PC",
    "Kv1p1_RI2021_SC",
    "Kv1p5_MA2024_PC",
    "Kv3p3_MA2024_PC",
    "Kv3p4_MA2025_BC",
    "Kv3p4_MA2024_PC",
    "Kv3p4_RI2021_SC",
    "Kv4p3_MA2025_BC",
    "Kv4p3_MA2024_PC",
    "Kv4p3_RI2021_SC",
    "KM_MA2020_GoC",
    "Kv1p1_MA2020_GoC",
    "Kv3p4_MA2020_GoC",
    "Kv4p3_MA2020_GoC",
    "KM_MA2020_GrC",
    "Kir2p3_MA2020_GrC",
    "Kv1p1_MA2020_GrC",
    "Kv2p2_0010_MA2020_GrC",
    "Kv3p4_MA2020_GrC",
    "Kv4p3_MA2020_GrC",
    "Kdr_ZH2019_IO",
]


def _sigm(x, y):
    return 1.0 / (u.math.exp(x / y) + 1.0)


def _linoid_stable(x, y):
    ratio = x / y
    return u.math.where(
        u.math.abs(ratio) < 1e-6,
        y * (1.0 - ratio / 2.0),
        x / (u.math.exp(ratio) - 1.0),
    )


def _x_over_one_minus_exp_neg_stable(x):
    return u.math.where(
        u.math.abs(x) < 1e-6,
        1.0 + x / 2.0,
        x / (1.0 - u.math.exp(-x)),
    )


@register_channel("KDR_Ba2002")
class KDR_Ba2002(HH):
    r"""Bazhenov 2002 delayed-rectifier potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -50.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_alpha(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 15.0
        return 0.032 * 5.0 / u.math.exprel(-temp / 5.0)

    def f_p_beta(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.5 * u.math.exp(-(temp - 10.0) / 40.0)


@register_channel("K_TM1991")
class K_TM1991(HH):
    r"""Traub and Miles 1991 delayed-rectifier potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -60.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_alpha(self, V, K: IonInfo):
        temp = 15.0 + (self.V_sh - V).to_decimal(u.mV)
        return 0.032 * 5.0 / u.math.exprel(temp / 5.0)

    def f_p_beta(self, V, K: IonInfo):
        temp = (self.V_sh - V).to_decimal(u.mV)
        return 0.5 * u.math.exp((10.0 + temp) / 40.0)


@register_channel("K_HH1952")
class K_HH1952(HH):
    r"""Hodgkin-Huxley 1952 potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = -45.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_alpha(self, V, K: IonInfo):
        temp = -((V - self.V_sh).to_decimal(u.mV) + 10.0) / 10.0
        return 0.1 / u.math.exprel(temp)

    def f_p_beta(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.125 * u.math.exp(-(temp + 20.0) / 80.0)


@register_channel("KA1_HM1992")
class KA1_HM1992(HH):
    r"""Huguenard & McCormick 1992 IA1 potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 30.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
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

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 60.0) / 8.5))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (
            u.math.exp((temp + 35.8) / 19.7)
            + u.math.exp(-(temp + 79.7) / 12.7)
        ) + 0.37

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 78.0) / 6.0))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -63.0,
            1.0 / (
                u.math.exp((temp + 46.0) / 5.0)
                + u.math.exp(-(temp + 238.0) / 37.5)
            ),
            19.0,
        )


@register_channel("KA2_HM1992")
class KA2_HM1992(HH):
    r"""Huguenard & McCormick 1992 IA2 potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 20.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
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

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 36.0) / 20.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (
            u.math.exp((temp + 35.8) / 19.7)
            + u.math.exp(-(temp + 79.7) / 12.7)
        ) + 0.37

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 78.0) / 6.0))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -63.0,
            1.0 / (
                u.math.exp((temp + 46.0) / 5.0)
                + u.math.exp(-(temp + 238.0) / 37.5)
            ),
            19.0,
        )


@register_channel("KK2A_HM1992")
class KK2A_HM1992(HH):
    r"""Huguenard & McCormick 1992 IK2a potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
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

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 43.0) / 17.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (
            u.math.exp((temp - 81.0) / 25.6)
            + u.math.exp(-(temp + 132.0) / 18.0)
        ) + 9.9

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 58.0) / 10.6))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (
            u.math.exp((temp - 1329.0) / 200.0)
            + u.math.exp(-(temp + 130.0) / 7.1)
        ) + 120.0


@register_channel("KK2B_HM1992")
class KK2B_HM1992(HH):
    r"""Huguenard & McCormick 1992 IK2b potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", q10=lambda self: self.q10_p, temp_ref=lambda self: self.temp_ref_p),
        Gate("q", q10=lambda self: self.q10_q, temp_ref=lambda self: self.temp_ref_q),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_p: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_p: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10_q: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref_q: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
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

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 43.0) / 17.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (
            u.math.exp((temp - 81.0) / 25.6)
            + u.math.exp(-(temp + 132.0) / 18.0)
        ) + 9.9

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 58.0) / 10.6))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -70.0,
            1.0 / (
                u.math.exp((temp - 1329.0) / 200.0)
                + u.math.exp(-(temp + 130.0) / 7.1)
            ),
            8.9,
        )


@register_channel("KNI_Ya1989")
class KNI_Ya1989(HH):
    r"""Yamada 1989 slow non-inactivating potassium current."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.004 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        tau_max: Union[brainstate.typing.ArrayLike, Callable] = 4e3 * u.ms,
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.tau_max = braintools.init.param(tau_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 35.0) / 10.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) + 35.0
        tau_max = self.tau_max.to_decimal(u.ms)
        return tau_max / (
            3.3 * u.math.exp(temp / 20.0) + u.math.exp(-temp / 20.0)
        )


@register_channel("K_Leak")
class K_Leak(Channel):
    """Potassium leak current."""

    __module__ = "braincell.channel"
    root_type = Potassium

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.005 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def init_state(self, V, K: IonInfo, batch_size: int = None):
        _ = (V, K, batch_size)

    def reset_state(self, V, K: IonInfo, batch_size: int = None):
        _ = (V, K, batch_size)

    def compute_derivative(self, V, K: IonInfo):
        pass

    def current(self, V, K: IonInfo):
        return self.g_max * (K.E - V)


@register_channel("K_Kv_test")
class K_Kv_test(HH):
    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("n", q10=lambda self: self.Q10_n, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.siemens / (u.cm ** 2)),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(25.0),
        Ra: Union[brainstate.typing.ArrayLike, Callable] = 0.02 * (1 / u.mV / u.ms),
        Rb: Union[brainstate.typing.ArrayLike, Callable] = 0.006 * (1 / u.mV / u.ms),
        q: Union[brainstate.typing.ArrayLike, Callable] = 9.0 * u.mV,
        v12: Union[brainstate.typing.ArrayLike, Callable] = 25.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Ra = braintools.init.param(Ra, self.varshape, allow_none=False)
        self.Rb = braintools.init.param(Rb, self.varshape, allow_none=False)
        self.q = braintools.init.param(q, self.varshape, allow_none=False)
        self.v12 = braintools.init.param(v12, self.varshape, allow_none=False)
        self.temp_ref = u.celsius2kelvin(23.0)
        self.Q10_n = 1.0

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_n_inf(self, V, K: IonInfo):
        V = (V - self.V_sh) / u.mV
        q = self.q.to_decimal(u.mV) if hasattr(self.q, "to_decimal") else self.q
        v12 = self.v12.to_decimal(u.mV) if hasattr(self.v12, "to_decimal") else self.v12
        return 1.0 / (1.0 + u.math.exp(-(V - v12) / q))

    def f_n_tau(self, V, K: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        Ra = self.Ra / (1 / u.mV / u.ms)
        Rb = self.Rb / (1 / u.mV / u.ms)
        q = self.q.to_decimal(u.mV) if hasattr(self.q, "to_decimal") else self.q
        v12 = self.v12.to_decimal(u.mV) if hasattr(self.v12, "to_decimal") else self.v12
        denom = (
            Ra * (V - v12) / (1.0 - u.math.exp(-(V - v12) / q))
            + (-Rb) * (V - v12) / (1.0 - u.math.exp(-(V - v12) / (-q)))
        )
        return 1.0 / denom

@register_channel("fKdr_SU2015_DCN")
class fKdr_SU2015_DCN(HH):
    """Template-based import of ``fKdr_SU2015_DCN.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("m", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 40.0) / -7.8))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return (
            13.9
            / (
                u.math.exp((V + 40.0) / 12.0)
                + u.math.exp((V + 40.0) / -13.0)
            )
            + 0.1
        ) / self.qdeltat

@register_channel("sKdr_SU2015_DCN")
class sKdr_SU2015_DCN(HH):
    """Template-based import of ``sKdr_SU2015_DCN.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("m", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 50.0) / -9.1))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return (
            14.95
            / (
                u.math.exp((V + 50.0) / 21.74)
                + u.math.exp((V + 50.0) / -13.91)
            )
            + 0.05
        ) / self.qdeltat

@register_channel("KM_RI2021_SC")
class KM_RI2021_SC(HH):
    """Template-based import of ``KM_RI2021_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_n = 0.0033
        self.Kalpha_n = 40.0 * u.mV
        self.V0alpha_n = -30.0 * u.mV
        self.Abeta_n = 0.0033
        self.Kbeta_n = -20.0 * u.mV
        self.V0beta_n = -30.0 * u.mV
        self.V0_ninf = -35.0 * u.mV
        self.B_ninf = 6.0 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp(
            (V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV)
        )

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp(
            (V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV)
        )

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))

@register_channel("Kir2p3_MA2025_BC")
class Kir2p3_MA2025_BC(HH):
    """Template-based import of ``Kir2p3_MA2025_BC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_d = 0.13289
        self.Kalpha_d = -24.3902 * u.mV
        self.V0alpha_d = -83.94 * u.mV
        self.Abeta_d = 0.16994
        self.Kbeta_d = 35.714 * u.mV
        self.V0beta_d = -83.94 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )

@register_channel("Kir2p3_MA2024_PC")
class Kir2p3_MA2024_PC(HH):
    """Template-based import of ``Kir2p3_MA2024_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_d = 0.13289
        self.Kalpha_d = -24.3902 * u.mV
        self.V0alpha_d = -83.94 * u.mV
        self.Abeta_d = 0.16994
        self.Kbeta_d = 35.714 * u.mV
        self.V0beta_d = -83.94 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )

@register_channel("Kir2p3_RI2021_SC")
class Kir2p3_RI2021_SC(HH):
    """Template-based import of ``Kir2p3_RI2021_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_d = 0.13289
        self.Kalpha_d = -24.3902 * u.mV
        self.V0alpha_d = -83.94 * u.mV
        self.Abeta_d = 0.16994
        self.Kbeta_d = 35.714 * u.mV
        self.V0beta_d = -83.94 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )

@register_channel("Kv1p1_MA2025_BC")
class Kv1p1_MA2025_BC(HH):
    """Template-based import of ``Kv1p1_MA2025_BC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.12889
        self.cva = 45.0 * u.mV
        self.cka = -33.90877 * u.mV
        self.cb = 0.12889
        self.cvb = 45.0 * u.mV
        self.ckb = 12.42101 * u.mV
        self.zn = 2.7978
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(-(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV))

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(-(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV))

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)

@register_channel("Kv1p1_MA2024_PC")
class Kv1p1_MA2024_PC(HH):
    """Template-based import of ``Kv1p1_MA2024_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.12889
        self.cva = 45.0 * u.mV
        self.cka = -33.90877 * u.mV
        self.cb = 0.12889
        self.cvb = 45.0 * u.mV
        self.ckb = 12.42101 * u.mV
        self.zn = 2.7978
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(-(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV))

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(-(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV))

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)

@register_channel("Kv1p1_RI2021_SC")
class Kv1p1_RI2021_SC(HH):
    """Template-based import of ``Kv1p1_RI2021_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.12889
        self.cva = 45.0 * u.mV
        self.cka = -33.90877 * u.mV
        self.cb = 0.12889
        self.cvb = 45.0 * u.mV
        self.ckb = 12.42101 * u.mV
        self.zn = 2.7978
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv1p5_MA2024_PC")
class Kv1p5_MA2024_PC(HH):
    """Template-based import of the active K-current path in ``Kv1p5_MA24_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3),
        Gate("n"),
        Gate("u"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.13195e-3 * (u.siemens / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(37.0),
        Tauact: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        Tauinactf: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        Tauinacts: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Tauact = braintools.init.param(Tauact, self.varshape, allow_none=False)
        self.Tauinactf = braintools.init.param(Tauinactf, self.varshape, allow_none=False)
        self.Tauinacts = braintools.init.param(Tauinacts, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self._voltage_factor(V) * self.conductance_factor(V, K) * (K.E - V)

    def _q10(self):
        return 2.2 ** (((self.temp - u.celsius2kelvin(37.0)) / u.kelvin) / 10.0)

    def _voltage_factor(self, V):
        V = V.to_decimal(u.mV)
        return 0.1 + 1.0 / (1.0 + u.math.exp(-(V - 15.0) / 13.0))

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 30.3) / 9.6))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        alpha = self._q10() * 0.65 / (
            u.math.exp(-(V + 10.0) / 8.5) + u.math.exp(-(V - 30.0) / 59.0)
        )
        beta = self._q10() * 0.65 / (2.5 + u.math.exp((V + 82.0) / 17.0))
        return 1.0 / (alpha + beta) / 3.0 * self.Tauact

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 0.25 + 1.0 / (1.35 + u.math.exp((V + 7.0) / 14.0))

    def f_n_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        alpha = self._q10() * 0.001 / (2.4 + 10.9 * u.math.exp(-(V + 90.0) / 78.0))
        beta = self._q10() * 0.001 * u.math.exp((V - 168.0) / 16.0)
        return 1.0 / (alpha + beta) / 3.0 * self.Tauinactf

    def f_u_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 0.1 + 1.0 / (1.1 + u.math.exp((V + 7.0) / 14.0))

    def f_u_tau(self, V, K: IonInfo):
        return 6800.0 * self.Tauinacts


@register_channel("Kv3p3_MA2024_PC")
class Kv3p3_MA2024_PC(HH):
    """Template-based import of ``Kv3p3_MA24_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.005 * (u.siemens / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.22
        self.cva = 16.0 * u.mV
        self.cka = -26.5 * u.mV
        self.cb = 0.22
        self.cvb = 16.0 * u.mV
        self.ckb = 26.5 * u.mV
        self.zn = 1.9196
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv3p4_MA2025_BC")
class Kv3p4_MA2025_BC(HH):
    """Template-based import of ``Kv3p4_MA2025_BC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.junction_potential = 11.0 * u.mV
        self.mivh = -24.0
        self.mik = 15.4
        self.mty0 = 0.00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1
        self.hiy0 = 0.31
        self.hiA = 0.69
        self.hivh = -5.802
        self.hik = 11.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        mtau_func = u.math.where(
            V < -35.0,
            (3.4225e-5 + 0.00498 * u.math.exp(V / 28.29)) * 3.0,
            self.mty0
            + 1.0
            / (
                u.math.exp((V + self.mtvh1) / self.mtk1)
                + u.math.exp((V + self.mtvh2) / self.mtk2)
            ),
        )
        return 1000.0 * mtau_func

    def f_h_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func

@register_channel("Kv3p4_MA2024_PC")
class Kv3p4_MA2024_PC(HH):
    """Template-based import of ``Kv3p4_MA2024_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.junction_potential = 11.0 * u.mV
        self.mivh = -24.0
        self.mik = 15.4
        self.mty0 = 0.00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1
        self.hiy0 = 0.31
        self.hiA = 0.69
        self.hivh = -5.802
        self.hik = 11.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        mtau_func = u.math.where(
            V < -35.0,
            (3.4225e-5 + 0.00498 * u.math.exp(V / 28.29)) * 3.0,
            self.mty0
            + 1.0
            / (
                u.math.exp((V + self.mtvh1) / self.mtk1)
                + u.math.exp((V + self.mtvh2) / self.mtk2)
            ),
        )
        return 1000.0 * mtau_func

    def f_h_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func

@register_channel("Kv3p4_RI2021_SC")
class Kv3p4_RI2021_SC(HH):
    """Template-based import of ``Kv3p4_RI2021_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.junction_potential = 11.0 * u.mV
        self.mivh = -24.0
        self.mik = 15.4
        self.mty0 = 0.00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1
        self.hiy0 = 0.31
        self.hiA = 0.69
        self.hivh = -5.802
        self.hik = 11.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        mtau_func = u.math.where(
            V < -35.0,
            (3.4225e-5 + 0.00498 * u.math.exp(V / 28.29)) * 3.0,
            self.mty0
            + 1.0
            / (
                u.math.exp((V + self.mtvh1) / self.mtk1)
                + u.math.exp((V + self.mtvh2) / self.mtk2)
            ),
        )
        return 1000.0 * mtau_func

    def f_h_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func

@register_channel("Kv4p3_MA2025_BC")
class Kv4p3_MA2025_BC(HH):
    """Template-based import of ``Kv4p3_MA2025_BC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.3271
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.4718 * u.mV
        self.V0beta_a = -18.2791 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.332 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV)))

    def f_a_tau(self, V, K: IonInfo):
        return 1.0 / (self._a_alpha(V, K) + self._a_beta(V, K))

    def f_b_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV)))

    def f_b_tau(self, V, K: IonInfo):
        return 1.0 / (self._b_alpha(V, K) + self._b_beta(V, K))

@register_channel("Kv4p3_MA2024_PC")
class Kv4p3_MA2024_PC(HH):
    """Template-based import of ``Kv4p3_MA2024_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.3271
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.4718 * u.mV
        self.V0beta_a = -18.2791 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.332 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV)))

    def f_a_tau(self, V, K: IonInfo):
        return 1.0 / (self._a_alpha(V, K) + self._a_beta(V, K))

    def f_b_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV)))

    def f_b_tau(self, V, K: IonInfo):
        return 1.0 / (self._b_alpha(V, K) + self._b_beta(V, K))

@register_channel("Kv4p3_RI2021_SC")
class Kv4p3_RI2021_SC(HH):
    """Template-based import of ``Kv4p3_RI2021_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.3271 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.4718 * u.mV
        self.V0beta_a = -18.2791 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.332 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo):
        return 1.0 / (self._a_alpha(V, K) + self._a_beta(V, K))

    def f_b_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo):
        return 1.0 / (self._b_alpha(V, K) + self._b_beta(V, K))

@register_channel("KM_MA2020_GoC")
class KM_MA2020_GoC(HH):
    """Template-based import of ``KM_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_n = 0.0033
        self.Kalpha_n = 40.0 * u.mV
        self.V0alpha_n = -30.0 * u.mV
        self.Abeta_n = 0.0033
        self.Kbeta_n = -20.0 * u.mV
        self.V0beta_n = -30.0 * u.mV
        self.V0_ninf = -35.0 * u.mV
        self.B_ninf = 6.0 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp(
            (V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV)
        )

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp(
            (V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV)
        )

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))

@register_channel("Kv1p1_MA2020_GoC")
class Kv1p1_MA2020_GoC(HH):
    """Template-based import of ``Kv1p1_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.12889
        self.cva = 45.0 * u.mV
        self.cka = -33.90877 * u.mV
        self.cb = 0.12889
        self.cvb = 45.0 * u.mV
        self.ckb = 12.42101 * u.mV
        self.zn = 2.7978
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)

@register_channel("Kv3p4_MA2020_GoC")
class Kv3p4_MA2020_GoC(HH):
    """Template-based import of ``Kv3p4_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.junction_potential = 11.0 * u.mV
        self.mivh = -24.0
        self.mik = 15.4
        self.mty0 = 0.00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1
        self.hiy0 = 0.31
        self.hiA = 0.69
        self.hivh = -5.802
        self.hik = 11.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        mtau_func = u.math.where(
            V < -35.0,
            (3.4225e-5 + 0.00498 * u.math.exp(V / 28.29)) * 3.0,
            self.mty0
            + 1.0
            / (
                u.math.exp((V + self.mtvh1) / self.mtk1)
                + u.math.exp((V + self.mtvh2) / self.mtk2)
            ),
        )
        return 1000.0 * mtau_func

    def f_h_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func

@register_channel("Kv4p3_MA2020_GoC")
class Kv4p3_MA2020_GoC(HH):
    """Template-based import of ``Kv4p3_MA2020_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.3271 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.4718 * u.mV
        self.V0beta_a = -18.2791 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.332 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo):
        return 1.0 / (self._a_alpha(V, K) + self._a_beta(V, K))

    def f_b_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo):
        return 1.0 / (self._b_alpha(V, K) + self._b_beta(V, K))

@register_channel("KM_MA2020_GrC")
class KM_MA2020_GrC(HH):
    """Template-based import of ``KM_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_n = 0.0033
        self.Kalpha_n = 40.0 * u.mV
        self.V0alpha_n = -30.0 * u.mV
        self.Abeta_n = 0.0033
        self.Kbeta_n = -20.0 * u.mV
        self.V0beta_n = -30.0 * u.mV
        self.V0_ninf = -35.0 * u.mV
        self.B_ninf = 6.0 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp(
            (V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV)
        )

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp(
            (V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV)
        )

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))

@register_channel("Kir2p3_MA2020_GrC")
class Kir2p3_MA2020_GrC(HH):
    """Template-based import of ``Kir2p3_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_d = 0.13289
        self.Kalpha_d = -24.3902 * u.mV
        self.V0alpha_d = -83.94 * u.mV
        self.Abeta_d = 0.16994
        self.Kbeta_d = 35.714 * u.mV
        self.V0beta_d = -83.94 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )

@register_channel("Kv1p1_MA2020_GrC")
class Kv1p1_MA2020_GrC(HH):
    """Template-based import of ``Kv1p1_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.gunit = 16.0e-9 * u.mS
        self.ca = 0.12889
        self.cva = 45.0 * u.mV
        self.cka = -33.90877 * u.mV
        self.cb = 0.12889
        self.cvb = 45.0 * u.mV
        self.ckb = 12.42101 * u.mV
        self.zn = 2.7978
        self.e0 = 1.60217646e-19 * u.coulomb

    def f_n_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)

@register_channel("Kv2p2_0010_MA2020_GrC")
class Kv2p2_0010_MA2020_GrC(HH):
    """Template-based import of ``Kv2p2_0010_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m"),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        BBiD: Union[brainstate.typing.ArrayLike, Callable] = 10.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.BBiD = braintools.init.param(BBiD, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - 5.0) / -12.0))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 130.0 / (1.0 + u.math.exp((V + 46.56) / -44.14))

    def f_h_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 16.3) / 4.8))

    def f_h_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 10000.0 / (1.0 + u.math.exp((V + 46.56) / -44.14))

@register_channel("Kv3p4_MA2020_GrC")
class Kv3p4_MA2020_GrC(HH):
    """Template-based import of ``Kv3p4_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.junction_potential = 11.0 * u.mV
        self.mivh = -24.0
        self.mik = 15.4
        self.mty0 = 0.00012851
        self.mtvh1 = 100.7
        self.mtk1 = 12.9
        self.mtvh2 = -56.0
        self.mtk2 = -23.1
        self.hiy0 = 0.31
        self.hiA = 0.69
        self.hivh = -5.802
        self.hik = 11.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        mtau_func = u.math.where(
            V < -35.0,
            (3.4225e-5 + 0.00498 * u.math.exp(V / 28.29)) * 3.0,
            self.mty0
            + 1.0
            / (
                u.math.exp((V + self.mtvh1) / self.mtk1)
                + u.math.exp((V + self.mtvh2) / self.mtk2)
            ),
        )
        return 1000.0 * mtau_func

    def f_h_inf(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo):
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func

@register_channel("Kv4p3_MA2020_GrC")
class Kv4p3_MA2020_GrC(HH):
    """Template-based import of ``Kv4p3_MA2020_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm ** 2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_a = 0.8147
        self.Kalpha_a = -23.3271 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.4718 * u.mV
        self.V0beta_a = -18.2791 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.332 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo):
        return 1.0 / (self._a_alpha(V, K) + self._a_beta(V, K))

    def f_b_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo):
        return 1.0 / (self._b_alpha(V, K) + self._b_beta(V, K))

@register_channel("Kdr_ZH2019_IO")
class Kdr_ZH2019_IO(HH):
    """Template-based import of ``Kdr_ZH2019_IO.mod``."""

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 18.0 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        x = (V + 41.0) / 10.0
        return 10.0 * _x_over_one_minus_exp_neg_stable(x)

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return 12.5 * u.math.exp(-(V + 51.0) / 80.0)

    def f_n_inf(self, V, K: IonInfo):
        alpha = self._n_alpha(V)
        beta = self._n_beta(V)
        return alpha / (alpha + beta)

    def f_n_tau(self, V, K: IonInfo):
        alpha = self._n_alpha(V)
        beta = self._n_beta(V)
        return 10.0 / (alpha + beta)
