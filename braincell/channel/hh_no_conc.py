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


from __future__ import annotations

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braincell._base import HHTypedNeuron, IonInfo
from braincell.channel._template import Gate, HH
from braincell.ion import Calcium, Potassium, Sodium
from braincell.mech import register_channel

__all__ = [
    "HCN1_MA25_BC",
    "HCN1_MA24_PC",
    "HCN1_RI21_SC",
    "HCN1_MA20_GoC",
    "HCN2_MA20_GoC",
    "HCN_SU15_DCN",
    "NaF_SU15_DCN",
    "NaP_SU15_DCN",
    "fKdr_SU15_DCN",
    "sKdr_SU15_DCN",
    "CaHVA_MA20_GoC",
    "Cav2p3_MA20_GoC",
    "CaHVA_MA20_GrC",
    "KM_MA20_GoC",
    "KM_MA20_GrC",
    "KM_RI21_SC",
    "Kir2p3_MA25_BC",
    "Kir2p3_MA24_PC",
    "Kir2p3_MA20_GrC",
    "Kir2p3_RI21_SC",
    "Kv1p1_MA25_BC",
    "Kv1p1_MA24_PC",
    "Kv1p1_RI21_SC",
    "Kv1p1_MA20_GoC",
    "Kv1p1_MA20_GrC",
    "Kv2p2_0010_MA20_GrC",
    "Kv3p4_MA25_BC",
    "Kv3p4_MA24_PC",
    "Kv3p4_RI21_SC",
    "Kv3p4_MA20_GoC",
    "Kv3p4_MA20_GrC",
    "Kv4p3_MA25_BC",
    "Kv4p3_MA24_PC",
    "Kv4p3_RI21_SC",
    "Kv4p3_MA20_GoC",
    "Kv4p3_MA20_GrC",
    "HCN_ZH19_IO",
    "Na_ZH19_IO",
    "Kdr_ZH19_IO",
    "Ca_ZH19_IO",
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


@register_channel("HCN1_MA25_BC")
class HCN1_MA25_BC(HH):
    """Template-based import of ``HCN1_MA25_BC.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -34.4 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.ratetau = 1.0
        self.ljp = 9.3 * u.mV
        self.v_inf_half_noljp = -90.3 * u.mV
        self.v_inf_k = 9.67 * u.mV
        self.v_tau_const = 0.0018
        self.v_tau_half1_noljp = -68.0 * u.mV
        self.v_tau_half2_noljp = -68.0 * u.mV
        self.v_tau_k1 = -22.0 * u.mV
        self.v_tau_k2 = 7.14 * u.mV

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_h_inf(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half1 = (self.v_tau_half1_noljp - self.ljp).to_decimal(u.mV)
        v_half2 = (self.v_tau_half2_noljp - self.ljp).to_decimal(u.mV)
        v_k1 = self.v_tau_k1.to_decimal(u.mV)
        v_k2 = self.v_tau_k2.to_decimal(u.mV)
        return self.ratetau / (
            self.v_tau_const
            * (
                u.math.exp((V - v_half1) / v_k1)
                + u.math.exp((V - v_half2) / v_k2)
            )
        )


@register_channel("HCN1_MA24_PC")
class HCN1_MA24_PC(HH):
    """Template-based import of ``HCN1_MA24_PC.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -34.4 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.ratetau = 1.0
        self.ljp = 9.3 * u.mV
        self.v_inf_half_noljp = -90.3 * u.mV
        self.v_inf_k = 9.67 * u.mV
        self.v_tau_const = 0.0018
        self.v_tau_half1_noljp = -68.0 * u.mV
        self.v_tau_half2_noljp = -68.0 * u.mV
        self.v_tau_k1 = -22.0 * u.mV
        self.v_tau_k2 = 7.14 * u.mV

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_h_inf(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half1 = (self.v_tau_half1_noljp - self.ljp).to_decimal(u.mV)
        v_half2 = (self.v_tau_half2_noljp - self.ljp).to_decimal(u.mV)
        v_k1 = self.v_tau_k1.to_decimal(u.mV)
        v_k2 = self.v_tau_k2.to_decimal(u.mV)
        return self.ratetau / (
            self.v_tau_const
            * (
                u.math.exp((V - v_half1) / v_k1)
                + u.math.exp((V - v_half2) / v_k2)
            )
        )


@register_channel("HCN1_RI21_SC")
class HCN1_RI21_SC(HH):
    """Template-based import of ``HCN1_RI21_SC.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -34.4 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.ratetau = 1.0
        self.ljp = 9.3 * u.mV
        self.v_inf_half_noljp = -90.3 * u.mV
        self.v_inf_k = 9.67 * u.mV
        self.v_tau_const = 0.0018
        self.v_tau_half1_noljp = -68.0 * u.mV
        self.v_tau_half2_noljp = -68.0 * u.mV
        self.v_tau_k1 = -22.0 * u.mV
        self.v_tau_k2 = 7.14 * u.mV

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_h_inf(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V, *unused):
        V = V.to_decimal(u.mV)
        v_half1 = (self.v_tau_half1_noljp - self.ljp).to_decimal(u.mV)
        v_half2 = (self.v_tau_half2_noljp - self.ljp).to_decimal(u.mV)
        v_k1 = self.v_tau_k1.to_decimal(u.mV)
        v_k2 = self.v_tau_k2.to_decimal(u.mV)
        return self.ratetau / (
            self.v_tau_const
            * (
                u.math.exp((V - v_half1) / v_k1)
                + u.math.exp((V - v_half2) / v_k2)
            )
        )


@register_channel("HCN_SU15_DCN")
class HCN_SU15_DCN(HH):
    """Template-based import of ``HCN_SU15_DCN.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("m", power=2),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -45.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_m_inf(self, V, *unused):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 5.0))

    def f_m_tau(self, V, *unused):
        _ = (V, unused)
        return 400.0 / self.qdeltat


@register_channel("NaF_SU15_DCN")
class NaF_SU15_DCN(HH):
    """Template-based import of ``NaF_SU15_DCN.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def current(self, V, Na: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Na) * (Na.E - V)

    def f_m_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 45.0) / -7.3))

    def f_m_tau(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return (
            5.83
            / (
                u.math.exp((V - 6.4) / -9.0)
                + u.math.exp((V + 97.0) / 17.0)
            )
            + 0.025
        ) / self.qdeltat

    def f_h_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 42.0) / 5.9))

    def f_h_tau(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return (
            16.67
            / (
                u.math.exp((V - 8.3) / -29.0)
                + u.math.exp((V + 66.0) / 9.0)
            )
            + 0.2
        ) / self.qdeltat


@register_channel("NaP_SU15_DCN")
class NaP_SU15_DCN(HH):
    """Template-based import of ``NaP_SU15_DCN.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def current(self, V, Na: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Na) * (Na.E - V)

    def f_m_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 70.0) / -4.1))

    def f_m_tau(self, V, Na: IonInfo, *unused):
        _ = (V, Na, unused)
        return 50.0 / self.qdeltat

    def f_h_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 4.0))

    def f_h_tau(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        V = V.to_decimal(u.mV)
        return (1750.0 / (1.0 + u.math.exp((V + 65.0) / -8.0)) + 250.0) / self.qdeltat


@register_channel("fKdr_SU15_DCN")
class fKdr_SU15_DCN(HH):
    """Template-based import of ``fKdr_SU15_DCN.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 40.0) / -7.8))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return (
            13.9
            / (
                u.math.exp((V + 40.0) / 12.0)
                + u.math.exp((V + 40.0) / -13.0)
            )
            + 0.1
        ) / self.qdeltat


@register_channel("sKdr_SU15_DCN")
class sKdr_SU15_DCN(HH):
    """Template-based import of ``sKdr_SU15_DCN.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 50.0) / -9.1))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return (
            14.95
            / (
                u.math.exp((V + 50.0) / 21.74)
                + u.math.exp((V + 50.0) / -13.91)
            )
            + 0.05
        ) / self.qdeltat


@register_channel("KM_RI21_SC")
class KM_RI21_SC(HH):
    """Template-based import of ``KM_RI21_SC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
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

    def f_n_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kir2p3_MA25_BC")
class Kir2p3_MA25_BC(HH):
    """Template-based import of ``Kir2p3_MA25_BC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )


@register_channel("Kir2p3_MA24_PC")
class Kir2p3_MA24_PC(HH):
    """Template-based import of ``Kir2p3_MA24_PC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )


@register_channel("Kir2p3_RI21_SC")
class Kir2p3_RI21_SC(HH):
    """Template-based import of ``Kir2p3_RI21_SC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )


@register_channel("Kv1p1_MA25_BC")
class Kv1p1_MA25_BC(HH):
    """Template-based import of ``Kv1p1_MA25_BC.mod``."""

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

    def f_n_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(-(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV))

    def f_n_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(-(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV))

    def current(self, V, K: IonInfo, *unused):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv1p1_MA24_PC")
class Kv1p1_MA24_PC(HH):
    """Template-based import of ``Kv1p1_MA24_PC.mod``."""

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

    def f_n_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(-(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV))

    def f_n_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(-(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV))

    def current(self, V, K: IonInfo, *unused):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv1p1_RI21_SC")
class Kv1p1_RI21_SC(HH):
    """Template-based import of ``Kv1p1_RI21_SC.mod``."""

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

    def f_n_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo, *unused):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv3p4_MA25_BC")
class Kv3p4_MA25_BC(HH):
    """Template-based import of ``Kv3p4_MA25_BC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
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

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func


@register_channel("Kv3p4_MA24_PC")
class Kv3p4_MA24_PC(HH):
    """Template-based import of ``Kv3p4_MA24_PC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
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

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func


@register_channel("Kv3p4_RI21_SC")
class Kv3p4_RI21_SC(HH):
    """Template-based import of ``Kv3p4_RI21_SC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
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

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA25_BC")
class Kv4p3_MA25_BC(HH):
    """Template-based import of ``Kv4p3_MA25_BC.mod``."""

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
        self.Kalpha_a = -23.32708
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175 * u.mV
        self.V0beta_a = -18.27914 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.33209 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV)))

    def f_a_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._a_alpha(V, K, *unused) + self._a_beta(V, K, *unused))

    def f_b_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV)))

    def f_b_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._b_alpha(V, K, *unused) + self._b_beta(V, K, *unused))


@register_channel("Kv4p3_MA24_PC")
class Kv4p3_MA24_PC(HH):
    """Template-based import of ``Kv4p3_MA24_PC.mod``."""

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
        self.Kalpha_a = -23.32708
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175 * u.mV
        self.V0beta_a = -18.27914 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.33209 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV)))

    def f_a_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._a_alpha(V, K, *unused) + self._a_beta(V, K, *unused))

    def f_b_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV)))

    def f_b_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._b_alpha(V, K, *unused) + self._b_beta(V, K, *unused))


@register_channel("Kv4p3_RI21_SC")
class Kv4p3_RI21_SC(HH):
    """Template-based import of ``Kv4p3_RI21_SC.mod``."""

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
        self.Kalpha_a = -23.32708 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175 * u.mV
        self.V0beta_a = -18.27914 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.33209 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._a_alpha(V, K, *unused) + self._a_beta(V, K, *unused))

    def f_b_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._b_alpha(V, K, *unused) + self._b_beta(V, K, *unused))


@register_channel("HCN1_MA20_GoC")
class HCN1_MA20_GoC(HH):
    """Template-based import of ``HCN1_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (
        Gate("o_fast", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
        Gate("o_slow", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.05 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -20.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Q10_diff = 1.5
        self.Ehalf = -72.49
        self.c = 0.11305
        self.rA = 0.002096
        self.rB = 0.97596
        self.tCf = 0.01371
        self.tDf = -3.368
        self.tEf = 2.302585092
        self.tCs = 0.01451
        self.tDs = -4.056
        self.tEs = 2.302585092

    def current(self, V):
        o = self.o_fast.value + self.o_slow.value
        return self._gbar_phi() * self.g_max * o * (self.E - V)

    def _gbar_phi(self):
        temp_ref = u.celsius2kelvin(23.0)
        return self.Q10_diff ** (((self.temp - temp_ref) / u.kelvin) / 10.0)

    def o_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.Ehalf) * self.c))

    def r(self, V):
        V = V.to_decimal(u.mV)
        return self.rA * V + self.rB

    def f_o_fast_inf(self, V, *unused):
        _ = unused
        return self.r(V) * self.o_inf(V)

    def f_o_slow_inf(self, V, *unused):
        _ = unused
        return (1.0 - self.r(V)) * self.o_inf(V)

    def f_o_fast_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCf * V) - self.tDf) * self.tEf)

    def f_o_slow_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCs * V) - self.tDs) * self.tEs)


@register_channel("HCN2_MA20_GoC")
class HCN2_MA20_GoC(HH):
    """Template-based import of ``HCN2_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (
        Gate("o_fast", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
        Gate("o_slow", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.08 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -20.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Q10_diff = 1.5
        self.Ehalf = -81.95
        self.c = 0.1661
        self.rA = -0.0227
        self.rB = -1.4694
        self.tCf = 0.0269
        self.tDf = -5.6111
        self.tEf = 2.3026
        self.tCs = 0.0152
        self.tDs = -5.2944
        self.tEs = 2.3026

    def current(self, V):
        o = self.o_fast.value + self.o_slow.value
        return self._gbar_phi() * self.g_max * o * (self.E - V)

    def _gbar_phi(self):
        temp_ref = u.celsius2kelvin(23.0)
        return self.Q10_diff ** (((self.temp - temp_ref) / u.kelvin) / 10.0)

    def o_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - self.Ehalf) * self.c))

    def r(self, V):
        V = V.to_decimal(u.mV)
        return u.math.where(
            V >= -64.70,
            0.0,
            u.math.where(
                V <= -108.70,
                1.0,
                self.rA * V + self.rB,
            ),
        )

    def f_o_fast_inf(self, V, *unused):
        _ = unused
        return self.r(V) * self.o_inf(V)

    def f_o_slow_inf(self, V, *unused):
        _ = unused
        return (1.0 - self.r(V)) * self.o_inf(V)

    def f_o_fast_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCf * V) - self.tDf) * self.tEf)

    def f_o_slow_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCs * V) - self.tDs) * self.tEs)


@register_channel("CaHVA_MA20_GoC")
class CaHVA_MA20_GoC(HH):
    """Template-based import of ``CaHVA_MA20_GoC.mod``."""

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


@register_channel("KM_MA20_GoC")
class KM_MA20_GoC(HH):
    """Template-based import of ``KM_MA20_GoC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
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

    def f_n_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kv1p1_MA20_GoC")
class Kv1p1_MA20_GoC(HH):
    """Template-based import of ``Kv1p1_MA20_GoC.mod``."""

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

    def f_n_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo, *unused):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv3p4_MA20_GoC")
class Kv3p4_MA20_GoC(HH):
    """Template-based import of ``Kv3p4_MA20_GoC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
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

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA20_GoC")
class Kv4p3_MA20_GoC(HH):
    """Template-based import of ``Kv4p3_MA20_GoC.mod``."""

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
        self.Kalpha_a = -23.32708 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175 * u.mV
        self.V0beta_a = -18.27914 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.33209 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._a_alpha(V, K, *unused) + self._a_beta(V, K, *unused))

    def f_b_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._b_alpha(V, K, *unused) + self._b_beta(V, K, *unused))


@register_channel("KM_MA20_GrC")
class KM_MA20_GrC(HH):
    """Template-based import of ``KM_MA20_GrC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
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

    def f_n_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV))
        )

    def f_n_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kir2p3_MA20_GrC")
class Kir2p3_MA20_GrC(HH):
    """Template-based import of ``Kir2p3_MA20_GrC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_d_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp(
            (V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV)
        )

    def f_d_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp(
            (V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV)
        )


@register_channel("Kv1p1_MA20_GrC")
class Kv1p1_MA20_GrC(HH):
    """Template-based import of ``Kv1p1_MA20_GrC.mod``."""

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

    def f_n_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.ca * u.math.exp(
            -(V + self.cva.to_decimal(u.mV)) / self.cka.to_decimal(u.mV)
        )

    def f_n_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.cb * u.math.exp(
            -(V + self.cvb.to_decimal(u.mV)) / self.ckb.to_decimal(u.mV)
        )

    def current(self, V, K: IonInfo, *unused):
        conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
        phi = self.gate_phi(self._iter_gates()[0])
        n = self.n.value
        alpha = self.f_n_alpha(V, K)
        beta = self.f_n_beta(V, K)
        ngate_flip = phi * (alpha * (1.0 - n) - beta * n) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * 4.0 * self.zn * ngate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Kv2p2_0010_MA20_GrC")
class Kv2p2_0010_MA20_GrC(HH):
    """Template-based import of ``Kv2p2_0010_MA20_GrC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - 5.0) / -12.0))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 130.0 / (1.0 + u.math.exp((V + 46.56) / -44.14))

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 16.3) / 4.8))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 10000.0 / (1.0 + u.math.exp((V + 46.56) / -44.14))


@register_channel("Kv3p4_MA20_GrC")
class Kv3p4_MA20_GrC(HH):
    """Template-based import of ``Kv3p4_MA20_GrC.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _shifted_voltage(self, V):
        return (V + self.junction_potential).to_decimal(u.mV)

    def f_m_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.mivh) / self.mik))

    def f_m_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
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

    def f_h_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        return self.hiy0 + self.hiA / (1.0 + u.math.exp((V - self.hivh) / self.hik))

    def f_h_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = self._shifted_voltage(V)
        htau_func = u.math.where(
            V > 0.0,
            0.0012 + 0.0023 * u.math.exp(-0.141 * V),
            1.2202e-05 + 0.012 * u.math.exp(-((V + 56.3) / 49.6) ** 2),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA20_GrC")
class Kv4p3_MA20_GrC(HH):
    """Template-based import of ``Kv4p3_MA20_GrC.mod``."""

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
        self.Kalpha_a = -23.32708 * u.mV
        self.V0alpha_a = -9.17203 * u.mV
        self.Abeta_a = 0.1655
        self.Kbeta_a = 19.47175 * u.mV
        self.V0beta_a = -18.27914 * u.mV
        self.Aalpha_b = 0.0368
        self.Kalpha_b = 12.8433 * u.mV
        self.V0alpha_b = -111.33209 * u.mV
        self.Abeta_b = 0.0345
        self.Kbeta_b = -8.90123 * u.mV
        self.V0beta_b = -49.9537 * u.mV
        self.V0_ainf = -38.0 * u.mV
        self.K_ainf = -17.0 * u.mV
        self.V0_binf = -78.8 * u.mV
        self.K_binf = 8.4 * u.mV

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _a_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(
            V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV)
        )

    def _a_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp(
            (V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV)
        )

    def _b_alpha(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Aalpha_b * _sigm(
            V - self.V0alpha_b.to_decimal(u.mV),
            self.Kalpha_b.to_decimal(u.mV),
        )

    def _b_beta(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return self.Abeta_b * _sigm(
            V - self.V0beta_b.to_decimal(u.mV),
            self.Kbeta_b.to_decimal(u.mV),
        )

    def f_a_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_ainf.to_decimal(u.mV)) / self.K_ainf.to_decimal(u.mV))
        )

    def f_a_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._a_alpha(V, K, *unused) + self._a_beta(V, K, *unused))

    def f_b_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        V = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp((V - self.V0_binf.to_decimal(u.mV)) / self.K_binf.to_decimal(u.mV))
        )

    def f_b_tau(self, V, K: IonInfo, *unused):
        return 1.0 / (self._b_alpha(V, K, *unused) + self._b_beta(V, K, *unused))


@register_channel("CaHVA_MA20_GrC")
class CaHVA_MA20_GrC(HH):
    """Template-based import of ``CaHVA_MA20_GrC.mod``."""

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


@register_channel("Cav2p3_MA20_GoC")
class Cav2p3_MA20_GoC(HH):
    """Template-based import of ``Cav2p3_MA20_GoC.mod``."""

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


@register_channel("HCN_ZH19_IO")
class HCN_ZH19_IO(HH):
    """Template-based import of ``HCN_ZH19_IO.mod``."""

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("q"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.15 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -43.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_q_inf(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 75.0) / 5.5))

    def f_q_tau(self, V, *unused):
        _ = unused
        V = V.to_decimal(u.mV)
        return 1.0 / (
            u.math.exp(-0.086 * V - 14.6) + u.math.exp(0.07 * V - 1.87)
        )


@register_channel("Na_ZH19_IO")
class Na_ZH19_IO(HH):
    """Template-based import of ``Na_ZH19_IO.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 70.0 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def current(self, V, Na: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, Na) * (Na.E - V)

    def _m_alpha(self, V):
        V = V.to_decimal(u.mV)
        x = (V + 41.0) / 10.0
        return _x_over_one_minus_exp_neg_stable(x)

    def _m_beta(self, V):
        V = V.to_decimal(u.mV)
        return 9.0 * u.math.exp(-(V + 66.0) / 20.0)

    def _h_alpha(self, V):
        V = V.to_decimal(u.mV)
        return 5.0 * u.math.exp(-(V + 60.0) / 15.0)

    def _h_beta(self, V):
        V = V.to_decimal(u.mV)
        x = (V + 50.0) / 10.0
        return 10.0 * _x_over_one_minus_exp_neg_stable(x)

    def f_m_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        alpha = self._m_alpha(V)
        beta = self._m_beta(V)
        return alpha / (alpha + beta)

    def f_m_tau(self, V, Na: IonInfo, *unused):
        _ = (V, Na, unused)
        return 0.001

    def f_h_inf(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        alpha = self._h_alpha(V)
        beta = self._h_beta(V)
        return alpha / (alpha + beta)

    def f_h_tau(self, V, Na: IonInfo, *unused):
        _ = (Na, unused)
        alpha = self._h_alpha(V)
        beta = self._h_beta(V)
        return 250.0 / (alpha + beta)


@register_channel("Kdr_ZH19_IO")
class Kdr_ZH19_IO(HH):
    """Template-based import of ``Kdr_ZH19_IO.mod``."""

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

    def current(self, V, K: IonInfo, *unused):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        x = (V + 41.0) / 10.0
        return 10.0 * _x_over_one_minus_exp_neg_stable(x)

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return 12.5 * u.math.exp(-(V + 51.0) / 80.0)

    def f_n_inf(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        alpha = self._n_alpha(V)
        beta = self._n_beta(V)
        return alpha / (alpha + beta)

    def f_n_tau(self, V, K: IonInfo, *unused):
        _ = (K, unused)
        alpha = self._n_alpha(V)
        beta = self._n_beta(V)
        return 10.0 / (alpha + beta)


@register_channel("Ca_ZH19_IO")
class Ca_ZH19_IO(HH):
    """Template-based import of ``Ca_ZH19_IO.mod``."""

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
