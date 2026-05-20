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


"""
This module implements hyperpolarization-activated cation channel.
"""

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braincell._base import HHTypedNeuron
from braincell.channel._base import Gate, HH
from braincell.mech import register_channel

__all__ = [
    'HCN_HM1992',
    'HCN1_MA2025_BC',
    'HCN1_MA2024_PC',
    'HCN1_RI2021_SC',
    'HCN1_MA2020_GoC',
    'HCN2_MA2020_GoC',
    'HCN_SU2015_DCN',
    'HCN_ZH2019_IO',
]


@register_channel("HCN_HM1992")
class HCN_HM1992(HH):
    r"""
    The hyperpolarization-activated cation current model propsoed by (Huguenard & McCormick, 1992) [1]_.

    The hyperpolarization-activated cation current model is adopted from
    (Huguenard, et, al., 1992) [1]_. Its dynamics is given by:

    .. math::

        \begin{aligned}
        I_h &= g_{\mathrm{max}} p \\
        \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p} \\
        p_{\infty} &=\frac{1}{1+\exp ((V+75) / 5.5)} \\
        \tau_{p} &=\frac{1}{\exp (-0.086 V-14.59)+\exp (0.0701 V-1.87)}
        \end{aligned}

    where :math:`\phi=1` is a temperature-dependent factor.

    Parameters
    ----------
    g_max : float
      The maximal conductance density (:math:`mS/cm^2`).
    E : float
      The reversal potential (mV).
    temp : float
      Absolute temperature used by the template temperature interface.
    q10 : float
      Q10 scaling factor for gate kinetics.
    temp_ref : float
      Reference temperature for the Q10 formula.

    References
    ----------
    .. [1] Huguenard, John R., and David A. McCormick. "Simulation of the currents
           involved in rhythmic oscillations in thalamic relay neuron." Journal
           of neurophysiology 68, no. 4 (1992): 1373-1383.

    """
    __module__ = 'braincell.channel'

    root_type = HHTypedNeuron
    gates = (
        Gate("p", q10=lambda self: self.q10, temp_ref=lambda self: self.temp_ref),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10. * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = 43. * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)

    def current(self, V):
        return self.g_max * self.conductance_factor(V) * (self.E - V)

    def f_p_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1. / (1. + u.math.exp((V + 75.) / 5.5))

    def f_p_tau(self, V):
        V = V.to_decimal(u.mV)
        return 1. / (u.math.exp(-0.086 * V - 14.59) + u.math.exp(0.0701 * V - 1.87))


# class Ih_De1996(Channel):
#   r"""
#   The hyperpolarization-activated cation current model propsoed by (Destexhe, et al., 1996) [1]_.
#
#   The full kinetic schema was
#
#   .. math::
#
#      \begin{gathered}
#      C \underset{\beta(V)}{\stackrel{\alpha(V)}{\rightleftarrows}} O \\
#      P_{0}+2 \mathrm{Ca}^{2+} \underset{k_{2}}{\stackrel{k_{1}}{\rightleftarrows}} P_{1} \\
#      O+P_{1} \underset{k_{4}}{\rightleftarrows} O_{\mathrm{L}}
#      \end{gathered}
#
#   where the first reaction represents the voltage-dependent transitions of :math:`I_h` channel
#   between closed (C) and open (O) forms, with :math:`\alpha` and :math:`\beta` as transition rates.
#   The second reaction represents the biding of intracellular :math:`\mathrm{Ca^{2+}}` ion to a
#   regulating factor (:math:`P_0` for unbound and :math:`P_1` for bound) with four binding sites for
#   calcium and rates of :math:`k_1 = 2.5e^7\, mM^{-4} \, ms^{-1}` and :math:`k_2=4e-4 \, ms^{-1}`
#   (half-activation of 0.002 mM :math:`Ca^{2+}`). The calcium-bound form :math:`P_1` associates
#   with the open form of the channel, leading to a locked open form :math:`O_L`, with rates of
#   :math:`k_3=0.1 \, ms^{-1}` and :math:`k_4 = 0.001 \, ms^{-1}`.
#
#   The current is the proportional to the relative concentration of open channel
#
#   .. math::
#
#      I_h = g_h (O+g_{inc}O_L) (V - E_h)
#
#   with a maximal conductance of :math:`\bar{g}_{\mathrm{h}}=0.02 \mathrm{mS} / \mathrm{cm}^{2}`
#   and a reversal potential of :math:`E_{\mathrm{h}}=-40 \mathrm{mV}`. Because of the factor
#   :math:`g_{\text {inc }}=2`, the conductance of the calcium-bound open state of
#   :math:`I_{\mathrm{h}}` channel is twice that of the unbound open state. This produces an
#   augmentation of conductance after the binding of :math:`\mathrm{Ca}^{2+}`, as observed in
#   sino-atrial cells (Hagiwara and Irisawa 1989).
#
#   The rates of :math:`\alpha` and :math:`\beta` are:
#
#   .. math::
#
#      & \alpha = m_{\infty} / \tau_m \\
#      & \beta = (1-m_{\infty}) / \tau_m \\
#      & m_{\infty} = 1/(1+\exp((V+75-V_{sh})/5.5)) \\
#      & \tau_m = (5.3 + 267/(\exp((V+71.5-V_{sh})/14.2) + \exp(-(V+89-V_{sh})/11.6)))
#
#   and the temperature regulating factor :math:`\phi=2^{(T-24)/10}`.
#
#   References
#   ----------
#   .. [1] Destexhe, Alain, et al. "Ionic mechanisms underlying synchronized
#          oscillations and propagating waves in a model of ferret thalamic
#          slices." Journal of neurophysiology 76.3 (1996): 2049-2070.
#   """
#
#   root_type = Calcium
#
#   def __init__(
#       self,
#       size: brainstate.typing.Size,
#       E: Union[brainstate.typing.ArrayLike, Callable] = -40. * u.mV,
#       k2: Union[brainstate.typing.ArrayLike, Callable] = 4e-4,
#       k4: Union[brainstate.typing.ArrayLike, Callable] = 1e-3,
#       V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
#       g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.02 * (u.mS / u.cm ** 2),
#       g_inc: Union[brainstate.typing.ArrayLike, Callable] = 2.,
#       Ca_half: Union[brainstate.typing.ArrayLike, Callable] = 2e-3,
#       T: brainstate.typing.ArrayLike = 36.,
#       T_base: brainstate.typing.ArrayLike = 3.,
#       phi: Union[brainstate.typing.ArrayLike, Callable] = None,
#       name: Optional[str] = None,
#       mode: Optional[brainstate.mixin.Mode] = None,
#   ):
#     super().__init__(
#       size,
#       name=name,
#       mode=mode
#     )
#
#     # parameters
#     self.T = braintools.init.param(T, self.varshape, allow_none=False)
#     self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
#     if phi is None:
#       self.phi = self.T_base ** ((self.T - 24.) / 10)
#     else:
#       self.phi = braintools.init.param(phi, self.varshape, allow_none=False)
#     self.E = braintools.init.param(E, self.varshape, allow_none=False)
#     self.k2 = braintools.init.param(k2, self.varshape, allow_none=False)
#     self.Ca_half = braintools.init.param(Ca_half, self.varshape, allow_none=False)
#     self.k1 = self.k2 / self.Ca_half ** 4
#     self.k4 = braintools.init.param(k4, self.varshape, allow_none=False)
#     self.k3 = self.k4 / 0.01
#     self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
#     self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
#     self.g_inc = braintools.init.param(g_inc, self.varshape, allow_none=False)
#
#   def dO(self, O, t, OL, V):
#     inf = self.f_inf(V)
#     tau = self.f_tau(V)
#     alpha = inf / tau
#     beta = (1 - inf) / tau
#     return alpha * (1 - O - OL) - beta * O
#
#   def dOL(self, OL, t, O, P1):
#     return self.k3 * P1 * O - self.k4 * OL
#
#   def dP1(self, P1, t, C_Ca):
#     return self.k1 * C_Ca ** 4 * (1 - P1) - self.k2 * P1
#
#   def update_state(self, V, Ca: IonInfo):
#     self.O.value, self.OL.value, self.P1.value = self.integral(
#       self.O.value, self.OL.value, self.P1.value, brainstate.environ.get('t'), V=V,
#     )
#
#   def current(self, V, Ca: IonInfo):
#     return self.g_max * (self.O.value + self.g_inc * self.OL.value) * (self.E - V)
#
#   def init_state(self, V, Ca, batch_size=None):
#     self.O = DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size))
#     self.OL = DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size))
#     self.P1 = DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size))
#
#   def reset_state(self, V, Ca: IonInfo, batch_size=None):
#     varshape = self.varshape if (batch_size is None) else ((batch_size,) + self.varshape)
#     k1 = self.k1 * Ca.C ** 4
#     self.P1.value = u.math.broadcast_arrays(k1 / (k1 + self.k2), varshape)
#     inf = self.f_inf(V)
#     tau = self.f_tau(V)
#     alpha = inf / tau
#     beta = (1 - inf) / tau
#     self.O.value = alpha / (alpha + alpha * self.k3 * self.P1 / self.k4 + beta)
#     self.OL.value = self.k3 * self.P1.value * self.O.value / self.k4
#
#   def f_inf(self, V):
#     V = V.to_decimal(u.mV)
#     return 1 / (1 + u.math.exp((V + 75 - self.V_sh) / 5.5))
#
#   def f_tau(self, V):

#     V = V.to_decimal(u.mV)
#     return (20. + 1000 / (u.math.exp((V + 71.5 - self.V_sh) / 14.2) +
#                           u.math.exp(-(V + 89 - self.V_sh) / 11.6))) / self.phi

@register_channel("HCN1_MA2025_BC")
class HCN1_MA2025_BC(HH):
    """Template-based import of ``HCN1_MA2025_BC.mod``."""

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

    def f_h_inf(self, V):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V):
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

@register_channel("HCN1_MA2024_PC")
class HCN1_MA2024_PC(HH):
    """Template-based import of ``HCN1_MA2024_PC.mod``."""

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

    def f_h_inf(self, V):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V):
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

@register_channel("HCN1_RI2021_SC")
class HCN1_RI2021_SC(HH):
    """Template-based import of ``HCN1_RI2021_SC.mod``."""

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

    def f_h_inf(self, V):
        V = V.to_decimal(u.mV)
        v_half = (self.v_inf_half_noljp - self.ljp).to_decimal(u.mV)
        v_k = self.v_inf_k.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V - v_half) / v_k))

    def f_h_tau(self, V):
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

@register_channel("HCN1_MA2020_GoC")
class HCN1_MA2020_GoC(HH):
    """Template-based import of ``HCN1_MA2020_GoC.mod``."""

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

    def f_o_fast_inf(self, V):
        return self.r(V) * self.o_inf(V)

    def f_o_slow_inf(self, V):
        return (1.0 - self.r(V)) * self.o_inf(V)

    def f_o_fast_tau(self, V):
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCf * V) - self.tDf) * self.tEf)

    def f_o_slow_tau(self, V):
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCs * V) - self.tDs) * self.tEs)

@register_channel("HCN2_MA2020_GoC")
class HCN2_MA2020_GoC(HH):
    """Template-based import of ``HCN2_MA2020_GoC.mod``."""

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

    def f_o_fast_inf(self, V):
        return self.r(V) * self.o_inf(V)

    def f_o_slow_inf(self, V):
        return (1.0 - self.r(V)) * self.o_inf(V)

    def f_o_fast_tau(self, V):
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCf * V) - self.tDf) * self.tEf)

    def f_o_slow_tau(self, V):
        V = V.to_decimal(u.mV)
        return u.math.exp(((self.tCs * V) - self.tDs) * self.tEs)

@register_channel("HCN_SU2015_DCN")
class HCN_SU2015_DCN(HH):
    """Template-based import of ``HCN_SU2015_DCN.mod``."""

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

    def f_m_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 5.0))

    def f_m_tau(self, V):
        return 400.0 / self.qdeltat

@register_channel("HCN_ZH2019_IO")
class HCN_ZH2019_IO(HH):
    """Template-based import of ``HCN_ZH2019_IO.mod``."""

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

    def f_q_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 75.0) / 5.5))

    def f_q_tau(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (
            u.math.exp(-0.086 * V - 14.6) + u.math.exp(0.07 * V - 1.87)
        )

