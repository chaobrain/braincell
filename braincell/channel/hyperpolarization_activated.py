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
from braincell.channel._base import Gate, HH, OhmicHH
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
class HCN_HM1992(OhmicHH):
    r"""Hyperpolarization-activated current of Huguenard & McCormick (1992).

    Reproduces the thalamic relay-neuron :math:`I_h` model of
    (Huguenard & McCormick, 1992) [1]_: a single Boltzmann-gated pore
    driven by an ohmic force against a fixed reversal potential.
    Dynamics:

    .. math::

        \begin{aligned}
        I_h &= g_{\mathrm{max}} \, p \, (E - V) \\
        \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p} \\
        p_{\infty} &= \frac{1}{1 + \exp((V + 75) / 5.5)} \\
        \tau_p &= \frac{1}{\exp(-0.086 V - 14.59) +
                  \exp(0.0701 V - 1.87)}
        \end{aligned}

    where :math:`\phi = q_{10}^{(T - T_{\mathrm{ref}}) / 10}` and the
    default ``q10 = 1.0`` makes :math:`\phi \equiv 1` for any ``temp``.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``43.0 mV``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the ``p`` gate, default ``1.0``.
    temp_ref : array-like, optional
        Reference temperature for the Q10 formula, default 36
        degrees Celsius.
    name : str, optional
        Optional channel name.

    Notes
    -----
    This class overrides :meth:`reversal_potential` to return
    ``self.E`` instead of an ion's reversal potential, so the current
    is driven against the fixed ``E`` parameter, not an ion state --
    the correct closed form is ``g_max * p * (E - V)``, not
    ``g_max * p`` alone.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = 'braincell.channel'

    root_type = HHTypedNeuron
    gates = (Gate("p", q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
        E: Union[brainstate.typing.ArrayLike, Callable] = 43.0 * u.mV,
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

    def reversal_potential(self, V, *ions):
        return self.E

    def f_p_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 75.0) / 5.5))

    def f_p_tau(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (u.math.exp(-0.086 * V - 14.59) + u.math.exp(0.0701 * V - 1.87))


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
#   and the temperature regulating factor :math:`\phi=2^{(temp-24)/10}`.
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
#       temp: brainstate.typing.ArrayLike = 36.,
#       q10_base: brainstate.typing.ArrayLike = 3.,
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
#     self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
#     self.q10_base = braintools.init.param(q10_base, self.varshape, allow_none=False)
#     if phi is None:
#       self.phi = self.q10_base ** ((self.temp - 24.) / 10)
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
#     self.O = state(braintools.init.param(u.math.zeros, self.varshape, batch_size))
#     self.OL = state(braintools.init.param(u.math.zeros, self.varshape, batch_size))
#     self.P1 = state(braintools.init.param(u.math.zeros, self.varshape, batch_size))
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
class HCN1_MA2025_BC(OhmicHH):
    r"""HCN1 h-current imported for the cerebellar basket cell model.

    Ports the single ``h`` gate NEURON mechanism
    ``HCN1_MA2025_BC.mod`` used in the basket-cell deposit of
    (Masoli et al., 2025) [3]_. The Boltzmann activation curve and
    biexponential time constant are the same functional forms used
    across the ``HCN1_MA2024_PC`` / ``HCN1_RI2021_SC`` siblings; only
    the model citation differs.

    .. math::

        \begin{aligned}
        h_\infty &= \frac{1}{1 + \exp\left(\dfrac{V - V_{1/2}}
                    {k}\right)} \\
        \tau_h &= \frac{\mathrm{ratetau}}{c \left[
                   \exp\left(\dfrac{V - V_{\tau 1}}{k_1}\right) +
                   \exp\left(\dfrac{V - V_{\tau 2}}{k_2}\right)
                   \right]}
        \end{aligned}

    where :math:`V_{1/2} = V_{\infty,\mathrm{noljp}} - V_{\mathrm{ljp}}
    = -99.6\ \mathrm{mV}`, :math:`k = 9.67\ \mathrm{mV}`,
    :math:`V_{\tau 1} = V_{\tau 2} = V_{\tau,\mathrm{noljp}} -
    V_{\mathrm{ljp}} = -77.3\ \mathrm{mV}`, :math:`k_1 = -22.0\
    \mathrm{mV}`, :math:`k_2 = 7.14\ \mathrm{mV}`, :math:`c =
    0.0018\ \mathrm{ms^{-1}}`, and :math:`\mathrm{ratetau} = 1.0`.
    These six constants and ``ratetau`` are fixed internal values set
    in ``__init__``; they are not exposed as parameters.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.1 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-34.4 mV``.
    temp : array-like, optional
        Absolute temperature driving the ``h`` gate's Q10 factor
        (``q10=3.0``, ``temp_ref=37`` degrees Celsius), default 23
        degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    HCN1_MA2024_PC : Same kinetics ported for a Purkinje-cell model.
    HCN1_RI2021_SC : Same kinetics ported for a stellate-cell model.

    Notes
    -----
    Ported from ``HCN1_MA2025_BC.mod``. This class overrides
    :meth:`reversal_potential` to return ``self.E`` instead of an
    ion's reversal potential.

    ``HCN1_MA25_BC.mod`` inherits its comment verbatim from the
    ``HCN1_MA24_PC.mod`` Purkinje-cell port, including the note
    "We call it HCN1 as PC express only HCN1" -- a claim about
    Purkinje cells, not this basket-cell channel, and not repeated
    here as though it were. The default ``temp = 23`` degrees
    Celsius is likewise carried over unchanged: Angelo et al. (2007)
    [1]_ did not report a recording temperature, so 23 degrees
    Celsius is the porter's assumption, not a value from that paper.

    References
    ----------
    .. [1] Angelo, K., London, M., Christensen, S. R., & Hausser, M.
           (2007). Local and global effects of Ih distribution in
           dendrites of mammalian neurons. The Journal of
           Neuroscience, 27(32), 8643-8653.
           doi:10.1523/JNEUROSCI.5284-06.2007
    .. [2] Santoro, B., Chen, S., Luthi, A., Pavlidis, P.,
           Shumyatsky, G. P., Tibbs, G. R., & Siegelbaum, S. A.
           (2000). Molecular and functional heterogeneity of
           hyperpolarization-activated pacemaker channels in the
           mouse CNS. The Journal of Neuroscience, 20(14), 5264-5275.
           doi:10.1523/JNEUROSCI.20-14-05264.2000
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm**2),
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

    def reversal_potential(self, V, *ions):
        return self.E

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
        return self.ratetau / (self.v_tau_const * (u.math.exp((V - v_half1) / v_k1) + u.math.exp((V - v_half2) / v_k2)))


@register_channel("HCN1_MA2024_PC")
class HCN1_MA2024_PC(OhmicHH):
    r"""HCN1 h-current imported for the human Purkinje cell model.

    Ports the single ``h`` gate NEURON mechanism
    ``HCN1_MA2024_PC.mod`` used in the Purkinje-cell deposit of
    (Masoli et al., 2024) [3]_. The Boltzmann activation curve and
    biexponential time constant are the same functional forms used
    across the ``HCN1_MA2025_BC`` / ``HCN1_RI2021_SC`` siblings; only
    the model citation differs.

    .. math::

        \begin{aligned}
        h_\infty &= \frac{1}{1 + \exp\left(\dfrac{V - V_{1/2}}
                    {k}\right)} \\
        \tau_h &= \frac{\mathrm{ratetau}}{c \left[
                   \exp\left(\dfrac{V - V_{\tau 1}}{k_1}\right) +
                   \exp\left(\dfrac{V - V_{\tau 2}}{k_2}\right)
                   \right]}
        \end{aligned}

    where :math:`V_{1/2} = V_{\infty,\mathrm{noljp}} - V_{\mathrm{ljp}}
    = -99.6\ \mathrm{mV}`, :math:`k = 9.67\ \mathrm{mV}`,
    :math:`V_{\tau 1} = V_{\tau 2} = V_{\tau,\mathrm{noljp}} -
    V_{\mathrm{ljp}} = -77.3\ \mathrm{mV}`, :math:`k_1 = -22.0\
    \mathrm{mV}`, :math:`k_2 = 7.14\ \mathrm{mV}`, :math:`c =
    0.0018\ \mathrm{ms^{-1}}`, and :math:`\mathrm{ratetau} = 1.0`.
    These six constants and ``ratetau`` are fixed internal values set
    in ``__init__``; they are not exposed as parameters.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.1 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-34.4 mV``.
    temp : array-like, optional
        Absolute temperature driving the ``h`` gate's Q10 factor
        (``q10=3.0``, ``temp_ref=37`` degrees Celsius), default 23
        degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    HCN1_MA2025_BC : Same kinetics ported for a basket-cell model.
    HCN1_RI2021_SC : Same kinetics ported for a stellate-cell model.

    Notes
    -----
    Ported from ``HCN1_MA2024_PC.mod``. This class overrides
    :meth:`reversal_potential` to return ``self.E`` instead of an
    ion's reversal potential.

    ``HCN1_MA24_PC.mod`` carries the comment "We call it HCN1 as PC
    express only HCN1 Santoro et al. 2000" -- correctly describing
    this Purkinje-cell channel, and the origin of that "PC" claim is
    the subunit-identity paper [2]_, not the kinetics paper [1]_. The
    default ``temp = 23`` degrees Celsius is carried over unchanged
    from the ``.mod`` file: Angelo et al. (2007) [1]_ did not report
    a recording temperature, so 23 degrees Celsius is the porter's
    assumption, not a value from that paper.

    References
    ----------
    .. [1] Angelo, K., London, M., Christensen, S. R., & Hausser, M.
           (2007). Local and global effects of Ih distribution in
           dendrites of mammalian neurons. The Journal of
           Neuroscience, 27(32), 8643-8653.
           doi:10.1523/JNEUROSCI.5284-06.2007
    .. [2] Santoro, B., Chen, S., Luthi, A., Pavlidis, P.,
           Shumyatsky, G. P., Tibbs, G. R., & Siegelbaum, S. A.
           (2000). Molecular and functional heterogeneity of
           hyperpolarization-activated pacemaker channels in the
           mouse CNS. The Journal of Neuroscience, 20(14), 5264-5275.
           doi:10.1523/JNEUROSCI.20-14-05264.2000
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm**2),
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

    def reversal_potential(self, V, *ions):
        return self.E

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
        return self.ratetau / (self.v_tau_const * (u.math.exp((V - v_half1) / v_k1) + u.math.exp((V - v_half2) / v_k2)))


@register_channel("HCN1_RI2021_SC")
class HCN1_RI2021_SC(OhmicHH):
    r"""HCN1 h-current imported for the cerebellar stellate cell model.

    Ports the single ``h`` gate NEURON mechanism
    ``HCN1_RI2021_SC.mod`` used in the stellate-cell deposit of
    (Rizza et al., 2021) [3]_. The Boltzmann activation curve and
    biexponential time constant are the same functional forms used
    across the ``HCN1_MA2025_BC`` / ``HCN1_MA2024_PC`` siblings; only
    the model citation differs.

    .. math::

        \begin{aligned}
        h_\infty &= \frac{1}{1 + \exp\left(\dfrac{V - V_{1/2}}
                    {k}\right)} \\
        \tau_h &= \frac{\mathrm{ratetau}}{c \left[
                   \exp\left(\dfrac{V - V_{\tau 1}}{k_1}\right) +
                   \exp\left(\dfrac{V - V_{\tau 2}}{k_2}\right)
                   \right]}
        \end{aligned}

    where :math:`V_{1/2} = V_{\infty,\mathrm{noljp}} - V_{\mathrm{ljp}}
    = -99.6\ \mathrm{mV}`, :math:`k = 9.67\ \mathrm{mV}`,
    :math:`V_{\tau 1} = V_{\tau 2} = V_{\tau,\mathrm{noljp}} -
    V_{\mathrm{ljp}} = -77.3\ \mathrm{mV}`, :math:`k_1 = -22.0\
    \mathrm{mV}`, :math:`k_2 = 7.14\ \mathrm{mV}`, :math:`c =
    0.0018\ \mathrm{ms^{-1}}`, and :math:`\mathrm{ratetau} = 1.0`.
    These six constants and ``ratetau`` are fixed internal values set
    in ``__init__``; they are not exposed as parameters.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.1 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-34.4 mV``.
    temp : array-like, optional
        Absolute temperature driving the ``h`` gate's Q10 factor
        (``q10=3.0``, ``temp_ref=37`` degrees Celsius), default 23
        degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    HCN1_MA2025_BC : Same kinetics ported for a basket-cell model.
    HCN1_MA2024_PC : Same kinetics ported for a Purkinje-cell model.

    Notes
    -----
    Ported from ``HCN1_RI2021_SC.mod``. This class overrides
    :meth:`reversal_potential` to return ``self.E`` instead of an
    ion's reversal potential.

    ``HCN1_RI21_SC.mod`` carries the same inherited comment as the
    basket-cell port, "We call it HCN1 as PC express only HCN1
    Santoro et al. 2000" -- a claim about Purkinje cells, not this
    stellate-cell channel, and not repeated here as though it were.
    The default ``temp = 23`` degrees Celsius is carried over
    unchanged from the ``.mod`` file: Angelo et al. (2007) [1]_ did
    not report a recording temperature, so 23 degrees Celsius is the
    porter's assumption, not a value from that paper.

    References
    ----------
    .. [1] Angelo, K., London, M., Christensen, S. R., & Hausser, M.
           (2007). Local and global effects of Ih distribution in
           dendrites of mammalian neurons. The Journal of
           Neuroscience, 27(32), 8643-8653.
           doi:10.1523/JNEUROSCI.5284-06.2007
    .. [2] Santoro, B., Chen, S., Luthi, A., Pavlidis, P.,
           Shumyatsky, G. P., Tibbs, G. R., & Siegelbaum, S. A.
           (2000). Molecular and functional heterogeneity of
           hyperpolarization-activated pacemaker channels in the
           mouse CNS. The Journal of Neuroscience, 20(14), 5264-5275.
           doi:10.1523/JNEUROSCI.20-14-05264.2000
    .. [3] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm**2),
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

    def reversal_potential(self, V, *ions):
        return self.E

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
        return self.ratetau / (self.v_tau_const * (u.math.exp((V - v_half1) / v_k1) + u.math.exp((V - v_half2) / v_k2)))


@register_channel("HCN1_MA2020_GoC")
class HCN1_MA2020_GoC(HH):
    r"""HCN1 fast/slow h-current imported for the Golgi cell model.

    Ports the two-gate NEURON mechanism ``HCN1_MA2020_GoC.mod`` used
    in the Golgi-cell deposit of (Masoli et al., 2020) [3]_. Two
    independent open-state gates, ``o_fast`` and ``o_slow``, share
    one Boltzmann steady state split by a linear mixing fraction
    ``r(V)``:

    .. math::

        \begin{aligned}
        I_h &= \phi_Q \, g_{\mathrm{max}} \,
               (o_{\mathrm{fast}} + o_{\mathrm{slow}}) \, (E - V) \\
        o_\infty(V) &= \frac{1}{1 + \exp((V - E_{1/2}) \, c)} \\
        r(V) &= r_A V + r_B \\
        o_{\mathrm{fast},\infty} &= r(V) \, o_\infty(V) \\
        o_{\mathrm{slow},\infty} &= (1 - r(V)) \, o_\infty(V) \\
        \tau_{\mathrm{fast}} &= \exp((t_{Cf} V - t_{Df}) \, t_{Ef}) \\
        \tau_{\mathrm{slow}} &= \exp((t_{Cs} V - t_{Ds}) \, t_{Es})
        \end{aligned}

    with :math:`\phi_Q = Q_{10}^{(T - 23^{\circ}\mathrm{C}) / 10}`,
    :math:`E_{1/2} = -72.49\ \mathrm{mV}`,
    :math:`c = 0.11305\ \mathrm{mV^{-1}}`,
    :math:`r_A = 0.002096\ \mathrm{mV^{-1}}`, :math:`r_B = 0.97596`,
    :math:`t_{Cf} = 0.01371\ \mathrm{mV^{-1}}`, :math:`t_{Df} =
    -3.368`, :math:`t_{Ef} = 2.30259`, :math:`t_{Cs} =
    0.01451\ \mathrm{mV^{-1}}`, :math:`t_{Ds} = -4.056`,
    :math:`t_{Es} = 2.30259`, and :math:`Q_{10} = 1.5`. ``r(V)`` is
    not clamped for this isoform, unlike ``HCN2_MA2020_GoC``. Both
    gates additionally carry ``q10=3.0``, ``temp_ref=23`` degrees
    Celsius on their own state relaxation, independent of the
    :math:`\phi_Q` conductance prefactor above. All eleven constants
    are fixed internal values set in ``__init__``; they are not
    exposed as parameters.

    This class subclasses :class:`HH` directly, not
    :class:`OhmicHH`, because :meth:`current` sums two gate values
    before applying the driving force rather than taking a single
    ``power``-weighted product.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.05 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-20.0 mV``.
    temp : array-like, optional
        Absolute temperature driving both the gate Q10 factors and
        :math:`\phi_Q`, default 22 degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    HCN2_MA2020_GoC : Companion isoform with a clamped ``r(V)``.

    Notes
    -----
    Ported from ``HCN1_MA2020_GoC.mod``. The former NEURON ``TABLE``
    tabulated ``o_fast_inf``, ``o_slow_inf``, ``tau_f`` and ``tau_s``
    over ``[-100, 30] mV`` and clamped outside that range; BrainCell
    evaluates the continuous formulas above at every call instead, so
    values outside ``[-100, 30] mV`` are expected to diverge from the
    original NEURON boundary-clamped output.

    ``.mod`` writes the time-constant exponent constants ``tEf`` and
    ``tEs`` as ``2.302585092``; NEURON's compiled default rounds this
    to ``2.30259``, and BrainCell follows the compiled value rather
    than the ``.mod`` source text.

    ``HCN1_MA20_GoC.mod`` credits its kinetics data to "Santoro et
    al. J Neurosci. 2000" [2]_; the Boltzmann/exponential functional
    forms trace to the cerebellar Golgi cell model of Solinas et al.
    (2007) [1]_.

    References
    ----------
    .. [1] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [2] Santoro, B., Chen, S., Luthi, A., Pavlidis, P.,
           Shumyatsky, G. P., Tibbs, G. R., & Siegelbaum, S. A.
           (2000). Molecular and functional heterogeneity of
           hyperpolarization-activated pacemaker channels in the
           mouse CNS. The Journal of Neuroscience, 20(14), 5264-5275.
           doi:10.1523/JNEUROSCI.20-14-05264.2000
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (
        Gate("o_fast", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
        Gate("o_slow", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.05 * (u.mS / u.cm**2),
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
        self.tEf = 2.30259
        self.tCs = 0.01451
        self.tDs = -4.056
        self.tEs = 2.30259

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
    r"""HCN2 fast/slow h-current imported for the Golgi cell model.

    Ports the two-gate NEURON mechanism ``HCN2_MA2020_GoC.mod`` used
    in the Golgi-cell deposit of (Masoli et al., 2020) [3]_. Two
    independent open-state gates, ``o_fast`` and ``o_slow``, share
    one Boltzmann steady state split by a mixing fraction ``r(V)``
    that is linear only inside a clamped voltage window:

    .. math::

        \begin{aligned}
        I_h &= \phi_Q \, g_{\mathrm{max}} \,
               (o_{\mathrm{fast}} + o_{\mathrm{slow}}) \, (E - V) \\
        o_\infty(V) &= \frac{1}{1 + \exp((V - E_{1/2}) \, c)} \\
        r(V) &= \begin{cases}
                  0 & V \geq -64.70\ \mathrm{mV} \\
                  1 & V \leq -108.70\ \mathrm{mV} \\
                  r_A V + r_B & \text{otherwise}
                \end{cases} \\
        o_{\mathrm{fast},\infty} &= r(V) \, o_\infty(V) \\
        o_{\mathrm{slow},\infty} &= (1 - r(V)) \, o_\infty(V) \\
        \tau_{\mathrm{fast}} &= \exp((t_{Cf} V - t_{Df}) \, t_{Ef}) \\
        \tau_{\mathrm{slow}} &= \exp((t_{Cs} V - t_{Ds}) \, t_{Es})
        \end{aligned}

    with :math:`\phi_Q = Q_{10}^{(T - 23^{\circ}\mathrm{C}) / 10}`,
    :math:`E_{1/2} = -81.95\ \mathrm{mV}`,
    :math:`c = 0.1661\ \mathrm{mV^{-1}}`,
    :math:`r_A = -0.0227\ \mathrm{mV^{-1}}`, :math:`r_B = -1.4694`,
    :math:`t_{Cf} = 0.0269\ \mathrm{mV^{-1}}`, :math:`t_{Df} =
    -5.6111`, :math:`t_{Ef} = 2.3026`, :math:`t_{Cs} =
    0.0152\ \mathrm{mV^{-1}}`, :math:`t_{Ds} = -5.2944`,
    :math:`t_{Es} = 2.3026`, and :math:`Q_{10} = 1.5`. Both gates
    additionally carry ``q10=3.0``, ``temp_ref=23`` degrees Celsius
    on their own state relaxation, independent of the
    :math:`\phi_Q` conductance prefactor above. All eleven constants
    are fixed internal values set in ``__init__``; they are not
    exposed as parameters.

    This class subclasses :class:`HH` directly, not
    :class:`OhmicHH`, because :meth:`current` sums two gate values
    before applying the driving force rather than taking a single
    ``power``-weighted product.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.08 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-20.0 mV``.
    temp : array-like, optional
        Absolute temperature driving both the gate Q10 factors and
        :math:`\phi_Q`, default 22 degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    HCN1_MA2020_GoC : Companion isoform with an unclamped ``r(V)``.

    Notes
    -----
    Ported from ``HCN2_MA2020_GoC.mod``. The former NEURON ``TABLE``
    tabulated ``o_fast_inf``, ``o_slow_inf``, ``tau_f`` and ``tau_s``
    over ``[-100, 30] mV`` and clamped outside that range; BrainCell
    evaluates the continuous formulas above at every call instead, so
    values outside ``[-100, 30] mV`` are expected to diverge from the
    original NEURON boundary-clamped output. The explicit ``r(V)``
    clamp to ``{0, 1}`` above ``-64.70 mV`` / below ``-108.70 mV`` is
    reproduced directly in :meth:`r`, independent of the removed
    ``TABLE`` boundary clamp.

    ``HCN2_MA20_GoC.mod`` credits its kinetics data to "Santoro et
    al. J Neurosci. 2000" [2]_; the Boltzmann/exponential functional
    forms trace to the cerebellar Golgi cell model of Solinas et al.
    (2007) [1]_.

    References
    ----------
    .. [1] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [2] Santoro, B., Chen, S., Luthi, A., Pavlidis, P.,
           Shumyatsky, G. P., Tibbs, G. R., & Siegelbaum, S. A.
           (2000). Molecular and functional heterogeneity of
           hyperpolarization-activated pacemaker channels in the
           mouse CNS. The Journal of Neuroscience, 20(14), 5264-5275.
           doi:10.1523/JNEUROSCI.20-14-05264.2000
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (
        Gate("o_fast", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
        Gate("o_slow", q10=3.0, temp_ref=u.celsius2kelvin(23.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.08 * (u.mS / u.cm**2),
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
class HCN_SU2015_DCN(OhmicHH):
    r"""H-current imported for the deep cerebellar nucleus (DCN) model.

    Ports the single ``m`` gate (power 2) NEURON mechanism
    ``HCN_SU2015_DCN.mod`` used in the deep-cerebellar-nucleus
    deposit of (Sudhakar et al., 2015) [2]_.

    .. math::

        \begin{aligned}
        I_h &= g_{\mathrm{max}} \, m^2 \, (E - V) \\
        m_\infty &= \frac{1}{1 + \exp((V + 80) / 5)} \\
        \tau_m &= \frac{400}{\mathrm{qdeltat}}
        \end{aligned}

    :math:`\tau_m` is a voltage-independent constant, not a function
    of :math:`V`; ``qdeltat`` is a fixed internal value (default
    ``1.0``) set in ``__init__`` and is not exposed as a parameter.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.01 mS/cm2``.
    E : array-like or callable, optional
        Reversal potential, default ``-45.0 mV``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    This class overrides :meth:`reversal_potential` to return
    ``self.E`` instead of an ion's reversal potential. The ``m``
    gate carries no ``q10``/``phi``, so :meth:`HH.gate_phi` resolves
    to the default ``1.0``.

    Ported from ``HCN_SU2015_DCN.mod``. Its kinetics belong to the
    deep cerebellar nucleus model of Steuber et al. (2011) [1]_,
    translated from GENESIS to NEURON by Luthman et al. (2011), and
    used in Sudhakar et al. (2015) [2]_. ``HCN`` is one of only four
    mechanism names (with ``CaLVA``, ``NaP`` and ``SK``) that occur
    as strings in the text of [2]_; the paper does not, however,
    print the Boltzmann or time constants above for any mechanism --
    those are established only by direct comparison against the
    ``.mod`` source, not by a per-mechanism statement in either
    paper.

    :meth:`f_m_tau` returns the constant ``400.0 / qdeltat`` (400 ms
    by default), a genuine, finite, voltage-independent time
    constant -- the gate is **not** instantaneous. The former NEURON
    ``TABLE`` directive tabulated only ``minf`` over ``[-150, 100]
    mV``; ``taum`` was never tabulated because it does not depend on
    voltage, not because it is zero or absent.

    References
    ----------
    .. [1] Steuber, V., Schultheiss, N. W., Silver, R. A., De
           Schutter, E., & Jaeger, D. (2011). Determinants of
           synaptic integration and heterogeneity in rebound firing
           explored with data-driven models of deep cerebellar
           nucleus cells. Journal of Computational Neuroscience,
           30(3), 633-658.
           doi:10.1007/s10827-010-0282-z
    .. [2] Sudhakar, S. K., Torben-Nielsen, B., & De Schutter, E.
           (2015). Cerebellar nuclear neurons use time and rate
           coding to transmit Purkinje neuron pauses. PLOS
           Computational Biology, 11(12), e1004641.
           doi:10.1371/journal.pcbi.1004641
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("m", power=2),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -45.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def reversal_potential(self, V, *ions):
        return self.E

    def f_m_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 5.0))

    def f_m_tau(self, V):
        return 400.0 / self.qdeltat


@register_channel("HCN_ZH2019_IO")
class HCN_ZH2019_IO(OhmicHH):
    r"""Inferior olive somatic h-current, imported from ``HCN_ZH19_IO.mod``.

    A single-gate hyperpolarization-activated current for the
    single-compartment inferior olive somatic model of Zhang &
    Santaniello (2019) [2]_. The gating kinetics originate in the
    inferior olive compartmental model of Schweighofer, Doya, & Kawato
    (1999) [1]_ and reached this class through the NEURON port of
    Torben-Nielsen, Segev, & Yarom (2012) (see Notes).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximum conductance, default ``0.15 mS/cm2``.
    E : array-like or callable, optional
        Fixed reversal potential used in place of an ion-derived
        driving force, default ``-43.0 mV``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    The current is

    .. math::

        I_h = g_{max} \, q \, (E - V)

    with the single gate ``q`` following

    .. math::

        q_\infty(V) = \frac{1}{1 + \exp\left(\dfrac{V + 75\
        \text{mV}}{5.5\ \text{mV}}\right)}

    .. math::

        \tau_q(V) = \frac{1}{\exp(-0.086\,V/\text{mV} - 14.6)
        + \exp(0.07\,V/\text{mV} - 1.87)}\ \text{ms}

    ``q`` carries no ``phi``/``q10`` in its
    :class:`~braincell.channel._base.Gate`
    declaration, so :meth:`~braincell.channel._base.HH.gate_phi` resolves
    to the default ``1.0`` and neither rate method is temperature-scaled.

    This mechanism is ported from ``IO/channel/HCN_ZH19_IO.mod``, whose
    header credits "Somatic h channel from Schweighofer et al., 1999"
    and porter "Xu Zhang @ UConn, 6-22-2018". The inferior olive neurons
    modelled by Zhang & Santaniello (2019) [2]_ are **single-compartment**
    (``nseg = 1``); this class must not be described as part of a
    multi-compartment inferior olive mechanism.

    In the source `.mod` file the kinetics originate with Schweighofer,
    Doya, & Kawato (1999) [1]_ and reached ``HCN_ZH19_IO.mod`` through
    the intermediate NEURON port of Torben-Nielsen, Segev, & Yarom
    (2012), credited in the header as "B. Torben-Nielsen @ HUJI" on the
    sibling `Na`/`Kdr`/`Ca` files of the same deposit.

    **Import deviation.** In the upstream `.mod` file the ``rates(v)``
    call that refreshes ``q_inf``/``tau_q`` lives inside ``BREAKPOINT``;
    in this deposit's ``HCN_ZH19_IO.mod`` it was relocated into
    ``DERIVATIVE states``, so the rates are refreshed *before* the
    ``cnexp`` state update rather than after it. This is a semantic
    change, not a cosmetic one -- the only other difference from the
    upstream file is the ``SUFFIX`` rename, and the ``COMMENT`` header
    is otherwise untouched.

    References
    ----------
    .. [1] Schweighofer, N., Doya, K., & Kawato, M. (1999).
           Electrophysiological properties of inferior olive neurons: A
           compartmental model. Journal of Neurophysiology, 82(2),
           804-817.
           doi:10.1152/jn.1999.82.2.804
    .. [2] Zhang, X., & Santaniello, S. (2019). Role of cerebellar
           GABAergic dysfunctions in the origins of essential tremor.
           Proceedings of the National Academy of Sciences of the
           United States of America, 116(27), 13592-13601.
           doi:10.1073/pnas.1817689116
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("q"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.15 * (u.mS / u.cm**2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -43.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)

    def reversal_potential(self, V, *ions):
        return self.E

    def f_q_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 75.0) / 5.5))

    def f_q_tau(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (u.math.exp(-0.086 * V - 14.6) + u.math.exp(0.07 * V - 1.87))
