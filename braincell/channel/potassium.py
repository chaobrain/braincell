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
from braincell.channel._base import Gate, HH, OhmicHH
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
class KDR_Ba2002(OhmicHH):
    r"""Bazhenov 2002 delayed-rectifier potassium current.

    The fast delayed-rectifier potassium current :math:`I_K` of the
    thalamocortical sleep-oscillation model of (Bazhenov et al., 2002)
    [1]_, with :math:`p^4` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.032 \times 5}
                    {\mathrm{exprel}(-(V' - 15) / 5)} \\
        \beta_p &= 0.5 \exp(-(V' - 10) / 40)
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}`,
    :math:`\mathrm{exprel}(x) = (e^{x} - 1)/x`, and both rates are in
    :math:`\mathrm{ms}^{-1}`. Away from :math:`V' = 15` the activation
    rate is exactly the linoid
    :math:`0.032 (15 - V') / (\exp((15 - V')/5) - 1)`; ``exprel`` is
    used only to remove that expression's removable singularity, where
    the code returns :math:`0.16\ \mathrm{ms}^{-1}` rather than
    ``0/0``. The gate integrates
    :math:`\dot{p} = \phi (\alpha_p (1 - p) - \beta_p p)` with
    :math:`\phi` from :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``10.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 36 degrees Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both rates, default ``-50.0 mV``
        (see Notes).
    name : str, optional
        Optional channel name.

    See Also
    --------
    K_TM1991 : The same two rate functions with different ``V_sh`` and
        ``q10`` defaults (see Notes).
    K_HH1952 : Classical squid-axon delayed rectifier, also :math:`p^4`
        but with a different rate parameterisation.

    Notes
    -----
    The rate functions above are algebraically identical to
    :class:`K_TM1991`'s -- the Traub & Miles (1991) :math:`\alpha_n` /
    :math:`\beta_n` pair, written here in the mirrored sign convention.
    Expanding both classes term by term gives the same two
    expressions, so the kinetics are not restated separately in
    :class:`K_TM1991`. The two classes differ only in their shipped
    defaults: ``V_sh = -50.0 mV`` and ``q10 = 3.0`` here against
    ``-60.0 mV`` and ``1.0`` there. Their ``g_max`` defaults are the
    same, ``10.0 mS/cm2``.

    That default matches the value the paper's "Intrinsic currents:
    thalamus" section gives for thalamocortical relay cells,
    ``g_K = 10 mS/cm^2``, which is a stronger fingerprint for this
    attribution than the rate equations alone. The same section states
    that the model uses "a fast potassium current, I_K (Traub and
    Miles, 1991)", so the authors themselves attribute these kinetics
    to that book.

    **The 2002 paper does not print these rate expressions.** It
    defers them to Bazhenov, Timofeev, Steriade & Sejnowski (1998),
    J Neurophysiol 79(5), 2730-2748. This docstring therefore records
    only that the current is the one *used in* Bazhenov et al. (2002)
    [1]_; in particular the ``V_sh = -50.0 mV`` shift could not be
    traced to any equation printed in that paper.

    With the shipped defaults ``temp`` equals ``temp_ref``, so the Q10
    factor is unity and ``q10 = 3.0`` bites only when ``temp`` is
    changed.

    References
    ----------
    .. [1] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
           (2002). Model of thalamocortical slow-wave sleep
           oscillations and transitions to activated states. The
           Journal of Neuroscience, 22(19), 8691-8704.
           doi:10.1523/JNEUROSCI.22-19-08691.2002
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("p", power=4, q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
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

    def f_p_alpha(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 15.0
        return 0.032 * 5.0 / u.math.exprel(-temp / 5.0)

    def f_p_beta(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.5 * u.math.exp(-(temp - 10.0) / 40.0)


@register_channel("K_TM1991")
class K_TM1991(OhmicHH):
    r"""Traub and Miles 1991 delayed-rectifier potassium current.

    The delayed-rectifier potassium current of the hippocampal
    pyramidal cell model of (Traub & Miles, 1991) [1]_, with
    :math:`p^4` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.032 \times 5}
                    {\mathrm{exprel}((15 - V') / 5)} \\
        \beta_p &= 0.5 \exp((10 - V') / 40)
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}`,
    :math:`\mathrm{exprel}(x) = (e^{x} - 1)/x`, and both rates are in
    :math:`\mathrm{ms}^{-1}`. Away from :math:`V' = 15` the activation
    rate is exactly the published linoid
    :math:`0.032 (15 - V') / (\exp((15 - V')/5) - 1)`; ``exprel`` is
    used only to remove that expression's removable singularity, where
    the code returns :math:`0.16\ \mathrm{ms}^{-1}` rather than
    ``0/0``. The gate integrates
    :math:`\dot{p} = \phi (\alpha_p (1 - p) - \beta_p p)` with
    :math:`\phi` from :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``10.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``,
        i.e. no temperature correction at any temperature (see Notes).
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 36 degrees Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both rates, default ``-60.0 mV``
        -- **not** ``-63.0 mV`` (see Notes).
    name : str, optional
        Optional channel name.

    See Also
    --------
    KDR_Ba2002 : The same two rate functions with ``V_sh = -50.0 mV``
        and ``q10 = 3.0``, as used by Bazhenov et al. (2002).
    braincell.channel.sodium.Na_TM1991 : Sodium counterpart from the
        same source mechanism, which ships ``V_sh = -63.0 mV``.

    Notes
    -----
    Compared against ``HH2.mod`` from ModelDB accession 3670,
    Destexhe's NEURON implementation, whose header reads "Equations
    modified by Traub, for Hippocampal Pyramidal cells, in: Traub &
    Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991".
    With ``v2 = v - vtraub`` in the mod file and
    :math:`V' = (V - V_{sh})/\mathrm{mV}` here, the two rate functions
    above and the :math:`p^4` gating match the mod file term for term.

    **The shift default is -60 mV, and the two BrainCell ``TM1991``
    classes do not agree with each other.**
    :class:`braincell.channel.sodium.Na_TM1991` ships
    ``V_sh = -63.0 mV`` while this class ships ``-60.0 mV``, although
    both derive from the same mechanism; the 3 mV divergence is a
    BrainCell choice, not something inherited from the source. Any
    sentence about "the Traub & Miles -63 mV shift" is wrong for this
    class. ``HH2.mod``'s own ``PARAMETER`` block ships a third value,
    ``vtraub = -55 mV``; the rate equations are unaffected either way,
    since the shift enters only through ``v2``/:math:`V'`.

    The ``g_max`` default coincides with ``HH2.mod``'s
    ``gkbar = 0.01 mho/cm^2`` (= ``10 mS/cm2``), but it is documented
    here as a BrainCell default rather than as a value printed in the
    book.

    ``HH2.mod`` applies ``tadj = 3^((celsius - 36)/10)``, which is
    unity at 36 degrees Celsius; the shipped ``q10 = 1.0`` with
    ``temp_ref`` at 36 degrees Celsius agrees there, but stays unity at
    every other temperature as well, so raising ``temp`` does not speed
    this gate unless ``q10`` is also changed.

    References
    ----------
    .. [1] Traub, R. D., & Miles, R. (1991). Neuronal networks of the
           hippocampus. Cambridge University Press.
           doi:10.1017/CBO9780511895401
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("p", power=4, q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
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

    def f_p_alpha(self, V, K: IonInfo):
        temp = 15.0 + (self.V_sh - V).to_decimal(u.mV)
        return 0.032 * 5.0 / u.math.exprel(temp / 5.0)

    def f_p_beta(self, V, K: IonInfo):
        temp = (self.V_sh - V).to_decimal(u.mV)
        return 0.5 * u.math.exp((10.0 + temp) / 40.0)


@register_channel("K_HH1952")
class K_HH1952(OhmicHH):
    r"""Hodgkin-Huxley 1952 delayed-rectifier potassium current.

    The squid giant axon potassium current :math:`I_K` of (Hodgkin &
    Huxley, 1952) [1]_, with :math:`p^4` HH gating and an ohmic
    driving force:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.1}{\mathrm{exprel}(-(V' + 10) / 10)} \\
        \beta_p &= 0.125 \exp(-(V' + 20) / 80)
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}`,
    :math:`\mathrm{exprel}(x) = (e^{x} - 1)/x`, and both rates are in
    :math:`\mathrm{ms}^{-1}`. Away from :math:`V' = -10` the
    activation rate is exactly
    :math:`0.01 (V' + 10) / (1 - \exp(-(V' + 10)/10))`; ``exprel``
    only removes that expression's removable singularity, where the
    code returns :math:`0.1\ \mathrm{ms}^{-1}` rather than ``0/0``.
    The gate integrates
    :math:`\dot{p} = \phi (\alpha_p (1 - p) - \beta_p p)` with
    :math:`\phi` from :meth:`~braincell.channel._base.HH.gate_phi`.

    With the default ``V_sh = -45.0 mV`` -- which places rest at
    -65 mV in the modern absolute-potential convention -- these expand
    to the published rates
    :math:`\alpha_n = 0.01 (V + 55)/(1 - \exp(-(V + 55)/10))` and
    :math:`\beta_n = 0.125 \exp(-(V + 65)/80)`, with :math:`n^4`
    gating.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``10.0 mS/cm2``,
        which is **not** Hodgkin & Huxley's own value (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 36 degrees Celsius
        (see Notes).
    V_sh : array-like or callable, optional
        Threshold shift applied to both rates, default ``-45.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.channel.sodium.Na_HH1952 : Sodium counterpart of the
        same model, whose ``g_max`` default does match the paper.
    K_TM1991 : Traub & Miles reparameterisation of the same
        :math:`p^4` delayed rectifier.

    Notes
    -----
    Every rate constant was expanded by hand and compared with the
    classical Hodgkin-Huxley rate equations; the implementation
    reproduces them exactly. ``exprel`` changes no value: it removes
    the removable singularity at the linoid's midpoint only.

    **The default conductance is not the paper's.** Hodgkin & Huxley
    give :math:`\bar{g}_K = 36\ \mathrm{mS/cm^2}`; this class ships
    ``10.0 mS/cm2``. Document it as a BrainCell default, not as the
    published value.

    **The default temperature is not the paper's either.** The rates
    above were measured at 6.3 degrees Celsius, whereas ``temp`` and
    ``temp_ref`` both default to 36 degrees Celsius, which makes the
    Q10 correction a no-op as shipped. ``q10 = 3.0`` is Hodgkin &
    Huxley's own factor-of-three-per-ten-degrees, but the shipped
    defaults do not reproduce the paper's 6.3 degrees Celsius
    behaviour.

    References
    ----------
    .. [1] Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative
           description of membrane current and its application to
           conduction and excitation in nerve. The Journal of
           Physiology, 117(4), 500-544.
           doi:10.1113/jphysiol.1952.sp004764
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("p", power=4, q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
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

    def f_p_alpha(self, V, K: IonInfo):
        temp = -((V - self.V_sh).to_decimal(u.mV) + 10.0) / 10.0
        return 0.1 / u.math.exprel(temp)

    def f_p_beta(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.125 * u.math.exp(-(temp + 20.0) / 80.0)


@register_channel("KA1_HM1992")
class KA1_HM1992(OhmicHH):
    r"""Huguenard & McCormick 1992 A-type potassium current (IA1).

    The first of the two components into which (Huguenard &
    McCormick, 1992) [1]_ splits the rapidly inactivating transient
    potassium current :math:`I_A` of thalamic relay neurons, with
    :math:`p^4 q` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 60) / 8.5)} \\
        \tau_p &= \frac{1}
                  {\exp((V' + 35.8) / 19.7) + \exp(-(V' + 79.7) / 12.7)}
                  + 0.37 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 78) / 6)} \\
        \tau_q &= \begin{cases}
                  \left[\exp((V' + 46) / 5)
                  + \exp(-(V' + 238) / 37.5)\right]^{-1} & V' < -63 \\
                  19 & V' \geq -63
                  \end{cases}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and the time
    constants are in milliseconds, further scaled per gate by
    :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``30.0 mS/cm2``
        (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 36 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default ``1.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 36 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KA2_HM1992 : The other :math:`I_A` component of the same model;
        it differs from this class only in ``p_inf`` and ``g_max``
        (see Notes).
    KK2A_HM1992 : Slowly inactivating :math:`I_{K2}` component of the
        same model.
    KK2B_HM1992 : The other :math:`I_{K2}` component.

    Notes
    -----
    The paper's published abstract enumerates exactly four currents,
    which map onto BrainCell as: :math:`I_T` ->
    :class:`~braincell.channel.CaT_HM1992`; :math:`I_A` ->
    :class:`KA1_HM1992` / :class:`KA2_HM1992`; :math:`I_{K2}` ->
    :class:`KK2A_HM1992` / :class:`KK2B_HM1992`; and :math:`I_h` ->
    :class:`~braincell.channel.HCN_HM1992`.

    **One divergence between the shipped pair and the paper's own
    description, recorded rather than corrected.** The abstract says
    :math:`I_A` "was modeled by assuming two components with different
    time constants of inactivation". In BrainCell the two components'
    inactivation is *identical*: :class:`KA1_HM1992` and
    :class:`KA2_HM1992` carry the same ``q_inf``, the same piecewise
    ``tau_q`` and even the same ``tau_p``, and differ only in the
    activation midpoint and slope (-60 / 8.5 here, -36 / 20 there)
    and in the default conductance (30 against 20 mS/cm2). The
    :math:`I_{K2}` pair does differ in inactivation, as the paper
    describes for that current.

    The ``tau_q`` branches do not meet: at :math:`V' = -63` the lower
    branch evaluates to about 23.4 ms while the class returns the
    constant 19 ms. The discontinuity is in the source
    parameterisation, not introduced here; the code applies the
    expression strictly below -63 and the constant at and above it.

    Both Q10 factors default to ``1.0``, so with the shipped defaults
    neither gate is temperature-scaled at any temperature; each gate
    has its own ``q10``/``temp_ref`` pair and is scaled independently.
    The ``g_max`` default is a BrainCell value: the attribution for
    this key was established from the paper's current inventory, not
    by comparing conductance densities.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 30.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 60.0) / 8.5))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp((temp + 35.8) / 19.7) + u.math.exp(-(temp + 79.7) / 12.7)) + 0.37

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 78.0) / 6.0))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -63.0,
            1.0 / (u.math.exp((temp + 46.0) / 5.0) + u.math.exp(-(temp + 238.0) / 37.5)),
            19.0,
        )


@register_channel("KA2_HM1992")
class KA2_HM1992(OhmicHH):
    r"""Huguenard & McCormick 1992 A-type potassium current (IA2).

    The second of the two components into which (Huguenard &
    McCormick, 1992) [1]_ splits the rapidly inactivating transient
    potassium current :math:`I_A` of thalamic relay neurons, with
    :math:`p^4 q` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 36) / 20)} \\
        \tau_p &= \frac{1}
                  {\exp((V' + 35.8) / 19.7) + \exp(-(V' + 79.7) / 12.7)}
                  + 0.37 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 78) / 6)} \\
        \tau_q &= \begin{cases}
                  \left[\exp((V' + 46) / 5)
                  + \exp(-(V' + 238) / 37.5)\right]^{-1} & V' < -63 \\
                  19 & V' \geq -63
                  \end{cases}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and the time
    constants are in milliseconds, further scaled per gate by
    :meth:`~braincell.channel._base.HH.gate_phi`. Only
    :math:`p_\infty` differs from :class:`KA1_HM1992`: its activation
    curve is 24 mV more depolarized and considerably shallower.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``20.0 mS/cm2``
        (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 36 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default ``1.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 36 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KA1_HM1992 : The other :math:`I_A` component of the same model,
        with a lower activation midpoint and a steeper slope.
    KK2A_HM1992 : Slowly inactivating :math:`I_{K2}` component of the
        same model.
    KK2B_HM1992 : The other :math:`I_{K2}` component.

    Notes
    -----
    The paper's published abstract enumerates exactly four currents,
    which map onto BrainCell as: :math:`I_T` ->
    :class:`~braincell.channel.CaT_HM1992`; :math:`I_A` ->
    :class:`KA1_HM1992` / :class:`KA2_HM1992`; :math:`I_{K2}` ->
    :class:`KK2A_HM1992` / :class:`KK2B_HM1992`; and :math:`I_h` ->
    :class:`~braincell.channel.HCN_HM1992`.

    **One divergence between the shipped pair and the paper's own
    description, recorded rather than corrected.** The abstract says
    :math:`I_A` "was modeled by assuming two components with different
    time constants of inactivation". In BrainCell the two components'
    inactivation is *identical*: this class and :class:`KA1_HM1992`
    carry the same ``q_inf``, the same piecewise ``tau_q`` and even
    the same ``tau_p``, and differ only in the activation midpoint and
    slope and in the default conductance. The :math:`I_{K2}` pair does
    differ in inactivation, as the paper describes for that current.

    The ``tau_q`` branches do not meet: at :math:`V' = -63` the lower
    branch evaluates to about 23.4 ms while the class returns the
    constant 19 ms. The discontinuity is in the source
    parameterisation, not introduced here.

    Both Q10 factors default to ``1.0``, so with the shipped defaults
    neither gate is temperature-scaled at any temperature. The
    ``g_max`` default is a BrainCell value: the attribution for this
    key was established from the paper's current inventory, not by
    comparing conductance densities.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", power=4, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 20.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 36.0) / 20.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp((temp + 35.8) / 19.7) + u.math.exp(-(temp + 79.7) / 12.7)) + 0.37

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 78.0) / 6.0))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -63.0,
            1.0 / (u.math.exp((temp + 46.0) / 5.0) + u.math.exp(-(temp + 238.0) / 37.5)),
            19.0,
        )


@register_channel("KK2A_HM1992")
class KK2A_HM1992(OhmicHH):
    r"""Huguenard & McCormick 1992 slow potassium current (IK2a).

    The first of the two components into which (Huguenard &
    McCormick, 1992) [1]_ splits the slowly inactivating potassium
    current :math:`I_{K2}` of thalamic relay neurons, with :math:`p q`
    HH gating (both gates to the first power) and an ohmic driving
    force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 43) / 17)} \\
        \tau_p &= \frac{1}
                  {\exp((V' - 81) / 25.6) + \exp(-(V' + 132) / 18)}
                  + 9.9 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 58) / 10.6)} \\
        \tau_q &= \frac{1}
                  {\exp((V' - 1329) / 200) + \exp(-(V' + 130) / 7.1)}
                  + 120
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and the time
    constants are in milliseconds, further scaled per gate by
    :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``10.0 mS/cm2``
        (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 36 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default ``1.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 36 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KK2B_HM1992 : The other :math:`I_{K2}` component of the same
        model; it shares every rate function except ``tau_q``.
    KA1_HM1992 : Transient :math:`I_A` component of the same model.
    KA2_HM1992 : The other :math:`I_A` component.

    Notes
    -----
    The paper's published abstract enumerates exactly four currents,
    which map onto BrainCell as: :math:`I_T` ->
    :class:`~braincell.channel.CaT_HM1992`; :math:`I_A` ->
    :class:`KA1_HM1992` / :class:`KA2_HM1992`; :math:`I_{K2}` ->
    :class:`KK2A_HM1992` / :class:`KK2B_HM1992`; and :math:`I_h` ->
    :class:`~braincell.channel.HCN_HM1992`. The abstract states that
    :math:`I_{K2}` "was also modeled by assuming two components", and
    this pair does implement that split in the inactivation time
    constant: ``tau_q`` here is the continuous expression above, while
    :class:`KK2B_HM1992` evaluates the same bracket only below
    :math:`V' = -70` and returns a constant 8.9 ms above it.

    Both Q10 factors default to ``1.0``, so with the shipped defaults
    neither gate is temperature-scaled at any temperature; each gate
    has its own ``q10``/``temp_ref`` pair and is scaled independently.
    The ``g_max`` default is a BrainCell value: the attribution for
    this key was established from the paper's current inventory, not
    by comparing conductance densities.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 43.0) / 17.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp((temp - 81.0) / 25.6) + u.math.exp(-(temp + 132.0) / 18.0)) + 9.9

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 58.0) / 10.6))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp((temp - 1329.0) / 200.0) + u.math.exp(-(temp + 130.0) / 7.1)) + 120.0


@register_channel("KK2B_HM1992")
class KK2B_HM1992(OhmicHH):
    r"""Huguenard & McCormick 1992 slow potassium current (IK2b).

    The second of the two components into which (Huguenard &
    McCormick, 1992) [1]_ splits the slowly inactivating potassium
    current :math:`I_{K2}` of thalamic relay neurons, with :math:`p q`
    HH gating (both gates to the first power) and an ohmic driving
    force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 43) / 17)} \\
        \tau_p &= \frac{1}
                  {\exp((V' - 81) / 25.6) + \exp(-(V' + 132) / 18)}
                  + 9.9 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 58) / 10.6)} \\
        \tau_q &= \begin{cases}
                  \left[\exp((V' - 1329) / 200)
                  + \exp(-(V' + 130) / 7.1)\right]^{-1} & V' < -70 \\
                  8.9 & V' \geq -70
                  \end{cases}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and the time
    constants are in milliseconds, further scaled per gate by
    :meth:`~braincell.channel._base.HH.gate_phi`. Only :math:`\tau_q`
    differs from :class:`KK2A_HM1992`, which uses the same bracket at
    every voltage and adds a constant 120 ms to it.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``10.0 mS/cm2``
        (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 36 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default ``1.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 36 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KK2A_HM1992 : The other :math:`I_{K2}` component of the same
        model; it shares every rate function except ``tau_q``.
    KA1_HM1992 : Transient :math:`I_A` component of the same model.
    KA2_HM1992 : The other :math:`I_A` component.

    Notes
    -----
    The paper's published abstract enumerates exactly four currents,
    which map onto BrainCell as: :math:`I_T` ->
    :class:`~braincell.channel.CaT_HM1992`; :math:`I_A` ->
    :class:`KA1_HM1992` / :class:`KA2_HM1992`; :math:`I_{K2}` ->
    :class:`KK2A_HM1992` / :class:`KK2B_HM1992`; and :math:`I_h` ->
    :class:`~braincell.channel.HCN_HM1992`. The abstract states that
    :math:`I_{K2}` "was also modeled by assuming two components", and
    the inactivation time constant is where this pair differs.

    **The two ``tau_q`` branches are far apart at the switch.** Just
    below :math:`V' = -70` the expression evaluates to roughly 885 ms,
    while at and above -70 the class returns 8.9 ms -- a jump of two
    orders of magnitude at a single point. This is the shipped
    parameterisation, transcribed here rather than smoothed; the code
    applies the expression strictly below -70 and the constant at and
    above it.

    Both Q10 factors default to ``1.0``, so with the shipped defaults
    neither gate is temperature-scaled at any temperature. The
    ``g_max`` default is a BrainCell value: the attribution for this
    key was established from the paper's current inventory, not by
    comparing conductance densities.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("p", q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 43.0) / 17.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp((temp - 81.0) / 25.6) + u.math.exp(-(temp + 132.0) / 18.0)) + 9.9

    def f_q_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((temp + 58.0) / 10.6))

    def f_q_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            temp < -70.0,
            1.0 / (u.math.exp((temp - 1329.0) / 200.0) + u.math.exp(-(temp + 130.0) / 7.1)),
            8.9,
        )


@register_channel("KNI_Ya1989")
class KNI_Ya1989(OhmicHH):
    r"""Yamada 1989 slow non-inactivating potassium current.

    The muscarine-sensitive M current :math:`I_M` of (Yamada, Koch, &
    Adams, 1989) [1]_ -- a slow, non-inactivating potassium current
    with a single activation gate to the first power and an ohmic
    driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 35) / 10)} \\
        \tau_p &= \frac{\tau_{max}}
                  {3.3 \exp((V' + 35) / 20) + \exp(-(V' + 35) / 20)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\tau_{max}` is the ``tau_max`` parameter read in
    milliseconds. :math:`\tau_p` is further scaled by
    :meth:`~braincell.channel._base.HH.gate_phi`. There is no
    inactivation gate: the conductance is :math:`g_{max} p`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.004 mS/cm2``
        (see Notes).
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``,
        i.e. no temperature correction (see Notes).
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 36 degrees Celsius.
    tau_max : array-like or callable, optional
        Peak time constant scaling :math:`\tau_p`. Defaults to
        ``4000.0 ms`` (see Notes).
    V_sh : array-like or callable, optional
        Threshold shift applied to both rates, default ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KK2A_HM1992 : Slowly inactivating thalamic potassium current, for
        contrast: this class does not inactivate at all.

    Notes
    -----
    Compared against ``IM.mod`` from ModelDB accession 3817, whose
    header reads "Model taken from Yamada, W.M., Koch, C. and Adams,
    P.R. Multiple channels and calcium dynamics. In: Methods in
    Neuronal Modeling, edited by C. Koch and I. Segev, MIT press,
    1989, p 97-134." The mod file computes exactly the two functions
    above and a single, non-inactivating, linear-in-:math:`m`
    potassium conductance; this class implements them with one
    :class:`~braincell.channel._base.Gate` of power 1. The current is
    the M current of the bullfrog sympathetic ganglion B-type cell,
    which is the chapter's subject.

    **Two shipped defaults diverge from that reference
    implementation.** ``IM.mod`` ships ``taumax = 1000 ms`` and
    ``gkbar = 1e-6 mho/cm^2`` (= ``0.001 mS/cm2``), whereas this class
    defaults to ``tau_max = 4000.0 ms`` and
    ``g_max = 0.004 mS/cm2``. Both are BrainCell defaults and neither
    is a value from the chapter.

    **The temperature handling diverges too.** ``IM.mod`` assumes
    Q10 = 2.3 referenced to 36 degrees Celsius; this class defaults to
    ``q10 = 1.0``, so as shipped there is no temperature correction at
    any temperature.

    Sources disagree on the chapter's final page: the machine-readable
    bibliographic record gives 97-133, used in the entry below, while
    ``IM.mod``'s header gives 97-134. The discrepancy is recorded
    rather than adjudicated.

    References
    ----------
    .. [1] Yamada, W. M., Koch, C., & Adams, P. R. (1989). Multiple
           channels and calcium dynamics. In C. Koch & I. Segev
           (Eds.), Methods in neuronal modeling: From synapses to
           networks (pp. 97-133). MIT Press.
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("p", q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.004 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp + 35.0) / 10.0))

    def f_p_tau(self, V, K: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) + 35.0
        tau_max = self.tau_max.to_decimal(u.ms)
        return tau_max / (3.3 * u.math.exp(temp / 20.0) + u.math.exp(-temp / 20.0))


@register_channel("K_Leak")
class K_Leak(Channel):
    r"""Potassium leak current.

    A voltage-independent, always-open potassium conductance:

    .. math::

        I = g_{max} \, (E_K - V)

    Unlike :class:`~braincell.channel.IL`, the reversal potential is
    not a parameter of this class -- it is taken from the potassium
    ion object this channel is attached to, so ``E_K`` follows whatever
    the ion computes (fixed, Nernst, or concentration-driven). There
    are no gating variables and no state to integrate:
    :meth:`init_state`, :meth:`reset_state` and
    :meth:`compute_derivative` are all no-ops.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Leak conductance density. Defaults to ``0.005 mS/cm2``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.channel.IL : Ion-independent leak with its own fixed
        reversal potential parameter.
    braincell.ion.Potassium : Ion object supplying ``E_K``.

    Notes
    -----
    This class has **no primary literature source**, and that is a
    determination rather than an omission: the current law is
    textbook Ohm's law and the ``0.005 mS/cm2`` default is a
    conventional placeholder that any caller overrides. No source
    model was identified for it, so the class deliberately ships
    without a ``References`` section.

    ``root_type`` is :class:`~braincell.ion.Potassium`, so the channel
    must be attached to a potassium ion; the ion, not the channel,
    owns the driving force.
    """

    __module__ = "braincell.channel"
    root_type = Potassium

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.005 * (u.mS / u.cm**2),
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
class K_Kv_test(OhmicHH):
    r"""Scratch Kv fixture built on the generic vtrap rate form.

    A single-gate potassium channel whose activation uses the generic
    NEURON ``vtrap`` alpha/beta idiom, with the Boltzmann midpoint,
    slope and the two rate scales exposed as parameters:

    .. math::

        \begin{aligned}
        n_\infty &= \frac{1}{1 + \exp(-(V' - v_{12}) / q)} \\
        \alpha_n &= \frac{R_a (V' - v_{12})}
                    {1 - \exp(-(V' - v_{12}) / q)} \\
        \beta_n &= \frac{R_b (V' - v_{12})}
                   {\exp((V' - v_{12}) / q) - 1} \\
        \tau_n &= \frac{1}{\alpha_n + \beta_n}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and :math:`v_{12}`
    and :math:`q` are read in millivolts. The gate is declared in the
    ``inf``/``tau`` form, so :math:`\alpha_n` and :math:`\beta_n`
    above are assembled inside :meth:`f_n_tau` rather than exposed as
    rate methods; the code writes :math:`\beta_n` equivalently as
    :math:`-R_b (V' - v_{12}) / (1 - \exp((V' - v_{12}) / q))`.

    **This class is a fixture, not a model of a published
    mechanism** -- see Notes before using it for anything but
    exercising the template layer.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.0 S/cm2``, i.e.
        an instance carries no current until it is set.
    V_sh : array-like or callable, optional
        Threshold shift applied to the rates, default ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature, default 25 degrees Celsius. Inert as
        shipped (see Notes).
    Ra : array-like or callable, optional
        Forward rate scale. Defaults to ``0.02 / (mV ms)``.
    Rb : array-like or callable, optional
        Backward rate scale. Defaults to ``0.006 / (mV ms)``.
    q : array-like or callable, optional
        Slope factor of the activation curve. Defaults to ``9.0 mV``.
    v12 : array-like or callable, optional
        Half-activation voltage. Defaults to ``25.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    K_HH1952 : Delayed rectifier with a verified literature source,
        for a channel of this shape that is a model of something.

    Notes
    -----
    This class has **no primary literature source**, and that is a
    determination rather than an omission -- though a weaker one than
    for the other citation-free symbols in this package. Three things
    mark it as a scratch or template fixture: its name, its zero
    default conductance, and its unit Q10. Its rate form is the
    generic ``vtrap`` alpha/beta idiom that recurs across dozens of
    unrelated NEURON ``kv.mod`` files and identifies no particular
    one. **No specific source was pursued**, which is not the same as
    establishing that none exists; if a later task identifies one,
    that is an addition rather than a contradiction of anything
    recorded here. The class therefore ships without a ``References``
    section.

    ``temp_ref`` (23 degrees Celsius) and ``Q10_n`` (``1.0``) are
    assigned in :meth:`__init__` as fixed attributes rather than
    exposed as constructor parameters. ``temp`` enters the gate only
    through :math:`Q_{10}^{(T - T_{ref})/10}`, and with a unit
    ``Q10_n`` that factor is 1 at every temperature, so changing
    ``temp`` has no effect on this channel.

    **Numerical caveat.** Unlike the sibling classes in this module,
    which route their linoids through ``exprel`` or an explicit
    ``where`` guard, :meth:`f_n_tau` evaluates the two quotients
    directly. At exactly :math:`V' = v_{12}` both are ``0/0`` and the
    method returns ``nan``, although the limit is finite --
    :math:`1 / (q (R_a + R_b))`, about 4.3 ms with the shipped
    defaults. Away from that single point the expression is
    well-behaved.
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10="Q10_n", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.siemens / (u.cm**2)),
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
        denom = Ra * (V - v12) / (1.0 - u.math.exp(-(V - v12) / q)) + (-Rb) * (V - v12) / (
            1.0 - u.math.exp(-(V - v12) / (-q))
        )
        return 1.0 / denom


@register_channel("fKdr_SU2015_DCN")
class fKdr_SU2015_DCN(OhmicHH):
    r"""Fast delayed rectifier of the DCN model (Sudhakar 2015).

    The fast delayed-rectifier potassium current of the deep
    cerebellar nucleus (DCN) neuron model used by (Sudhakar et al.,
    2015) [2]_, with :math:`m^4` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp((V + 40) / -7.8)} \\
        \tau_m &= \left(\frac{13.9}
                  {\exp((V + 40) / 12) + \exp((V + 40) / -13)}
                  + 0.1\right) \Big/ q_{\Delta t}
        \end{aligned}

    where :math:`V` is in millivolts -- this class applies no voltage
    shift -- and :math:`\tau_m` is in milliseconds. The reversal
    potential comes from the potassium ion object, not from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.01 mS/cm2``,
        which is exactly the source mechanism's
        ``gbar = 1e-5 siemens/cm2``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    sKdr_SU2015_DCN : Slow delayed rectifier of the same DCN model,
        with the same gating shape and slower, more depolarized
        kinetics.
    braincell.channel.NaF_SU2015_DCN : Fast sodium current of the same
        model.

    Notes
    -----
    Ported from ``DCN/channel/fKdr_SU15_DCN.mod``, whose ``COMMENT``
    reads "Translated from GENESIS by Johannes Luthman and Volker
    Steuber." Accordingly, the kinetics originate in the GENESIS deep
    cerebellar nucleus model of Steuber, Schultheiss, Silver, De
    Schutter & Jaeger (2011) [1]_, were translated from GENESIS to
    NEURON by Luthman, Hoebeek, Maex, Davey, Adams, De Zeeuw &
    Steuber (2011), and were used in that translated form by Sudhakar
    et al. (2015) [2]_. The 2011 translation paper is named in this
    prose only, per house style, and not as a numbered reference.

    **What this docstring does not claim.** No paper in that chain
    was shown to print the Boltzmann and time-constant constants
    above; the code-side check ran from the ``.mod`` file to this
    class only. The string "fKdr" does not occur anywhere in the
    Sudhakar et al. (2015) text, so this records that the mechanism is
    part of the model published as [2]_, not that the paper names or
    describes it.

    ``qdeltat`` is set to ``1.0`` in :meth:`__init__` as a fixed
    attribute rather than exposed as a constructor parameter; it
    mirrors the mod file's ``GLOBAL qdeltat`` and divides
    :math:`\tau_m` as shown.

    **Import deviation.** The original mechanism's NMODL ``TABLE``
    over ``[-150, 100] mV``, covering ``minf`` and ``taum``, is not
    reproduced: both expressions are evaluated per call. NEURON
    clamped tabulated values to the boundary outside that window, so
    any BrainCell-versus-NEURON divergence below -150 mV or above
    100 mV is expected rather than a port error.

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
    root_type = Potassium
    gates = (Gate("m", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 40.0) / -7.8))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return (13.9 / (u.math.exp((V + 40.0) / 12.0) + u.math.exp((V + 40.0) / -13.0)) + 0.1) / self.qdeltat


@register_channel("sKdr_SU2015_DCN")
class sKdr_SU2015_DCN(OhmicHH):
    r"""Slow delayed rectifier of the DCN model (Sudhakar 2015).

    The slow delayed-rectifier potassium current of the deep
    cerebellar nucleus (DCN) neuron model used by (Sudhakar et al.,
    2015) [2]_, with :math:`m^4` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp((V + 50) / -9.1)} \\
        \tau_m &= \left(\frac{14.95}
                  {\exp((V + 50) / 21.74) + \exp((V + 50) / -13.91)}
                  + 0.05\right) \Big/ q_{\Delta t}
        \end{aligned}

    where :math:`V` is in millivolts -- this class applies no voltage
    shift -- and :math:`\tau_m` is in milliseconds. The reversal
    potential comes from the potassium ion object, not from the class.
    Compared with :class:`fKdr_SU2015_DCN`, activation is 10 mV more
    hyperpolarized and shallower.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.01 mS/cm2``,
        which is exactly the source mechanism's
        ``gbar = 1e-5 siemens/cm2``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    fKdr_SU2015_DCN : Fast delayed rectifier of the same DCN model,
        with the same gating shape.
    braincell.channel.NaF_SU2015_DCN : Fast sodium current of the same
        model.

    Notes
    -----
    Ported from ``DCN/channel/sKdr_SU15_DCN.mod``, whose ``COMMENT``
    reads "Translated from GENESIS by Johannes Luthman and Volker
    Steuber." Accordingly, the kinetics originate in the GENESIS deep
    cerebellar nucleus model of Steuber, Schultheiss, Silver, De
    Schutter & Jaeger (2011) [1]_, were translated from GENESIS to
    NEURON by Luthman, Hoebeek, Maex, Davey, Adams, De Zeeuw &
    Steuber (2011), and were used in that translated form by Sudhakar
    et al. (2015) [2]_. The 2011 translation paper is named in this
    prose only, per house style, and not as a numbered reference.

    **What this docstring does not claim.** No paper in that chain
    was shown to print the Boltzmann and time-constant constants
    above; the code-side check ran from the ``.mod`` file to this
    class only. The string "sKdr" does not occur anywhere in the
    Sudhakar et al. (2015) text, so this records that the mechanism is
    part of the model published as [2]_, not that the paper names or
    describes it.

    ``qdeltat`` is set to ``1.0`` in :meth:`__init__` as a fixed
    attribute rather than exposed as a constructor parameter; it
    mirrors the mod file's ``GLOBAL qdeltat`` and divides
    :math:`\tau_m` as shown.

    **Import deviation.** The original mechanism's NMODL ``TABLE``
    over ``[-150, 100] mV``, covering ``minf`` and ``taum``, is not
    reproduced: both expressions are evaluated per call. NEURON
    clamped tabulated values to the boundary outside that window, so
    any BrainCell-versus-NEURON divergence below -150 mV or above
    100 mV is expected rather than a port error.

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
    root_type = Potassium
    gates = (Gate("m", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def f_m_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 50.0) / -9.1))

    def f_m_tau(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return (14.95 / (u.math.exp((V + 50.0) / 21.74) + u.math.exp((V + 50.0) / -13.91)) + 0.05) / self.qdeltat


@register_channel("KM_RI2021_SC")
class KM_RI2021_SC(OhmicHH):
    r"""M-type potassium current of the stellate cell model.

    Slow, non-inactivating M-type potassium current imported from the
    cerebellar stellate cell model of Rizza et al. (2021) [2]_. A
    single first-order ``n`` gate of power 1 drives an ohmic current:

    .. math::

        \begin{aligned}
        n_\infty &= \frac{1}{1 + \exp(-(V + 35) / 6)} \\
        \alpha_n &= 0.0033 \, \exp((V + 30) / 40) \\
        \beta_n &= 0.0033 \, \exp(-(V + 30) / 20) \\
        \tau_n &= \frac{1}{\alpha_n + \beta_n}
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and :math:`\tau_n` is in milliseconds. This class applies no
    voltage shift, and the reversal potential comes from the potassium
    ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.25 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.00025 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KM_MA2020_GoC : Golgi-cell port of the same mechanism.
    KM_MA2020_GrC : Granule-cell port of the same mechanism.
    Kir2p3_RI2021_SC : Inward rectifier of the same stellate model,
        sharing the same origin paper.

    Notes
    -----
    Ported from ``SC/channel/KM_RI21_SC.mod``. That file, the Golgi
    port ``GoC/channel/KM_MA20_GoC.mod`` and the granule port
    ``GrC/channel/KM_MA20_GrC.mod`` are byte-identical apart from
    their ``SUFFIX`` line, and so are the three BrainCell classes: the
    rate constants above are shared verbatim with
    :class:`KM_MA2020_GoC` and :class:`KM_MA2020_GrC`. What differs is
    only the deposit each was imported from, and therefore the model
    paper cited below.

    The mechanism does not use the steady state implied by its own
    rates. Its ``n_inf = a_n/(a_n + b_n)`` line is commented out in
    the ``.mod`` source and replaced by the explicit Boltzmann shown
    above, so :math:`n_\infty` and :math:`\tau_n` are independent
    expressions here.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`(n_\infty - n)/\tau_n` term by
    :math:`\phi = 3^{(T - 22)/10}` (about 2.41 at the default 30
    degrees Celsius). The ``.mod`` file instead multiplies ``Q10``
    into ``alp_n`` and ``bet_n``, which divides its ``tau_n`` by the
    same factor. The two forms are algebraically identical, but it
    means :meth:`f_n_tau` returns the q10-free time constant rather
    than the mechanism's ``tau_n``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries ``Author: A. Fontana`` and
    ``CoAuthor: T.Nieus``. That credit line is copy-pasted verbatim
    across every cell-type port of this mechanism and names people
    unrelated to the stellate-cell key, so it is not treated as a
    citation here. The kinetics originate in the cerebellar granule
    cell model of D'Angelo et al. (2001) [1]_; the stellate-cell
    paper [2]_ names the model this parameterisation was imported
    from, not the origin of the equations.

    **Conductance default.** ``0.25 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 30] mV``, covering ``n_inf`` and ``tau_n``, is not
    reproduced: both expressions are evaluated per call. NEURON
    clamped tabulated values to the boundary outside that window, so
    any BrainCell-versus-NEURON divergence below -100 mV or above
    30 mV is expected rather than a port error. The integration
    method was also changed from ``derivimplicit`` to ``cnexp``;
    with one independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm**2),
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

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp((V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV))

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp((V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV))

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV)))

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kir2p3_MA2025_BC")
class Kir2p3_MA2025_BC(OhmicHH):
    r"""Kir2.3 inward-rectifier current of the basket cell model.

    Hyperpolarization-activated inward-rectifier potassium current
    imported from the cerebellar basket cell model of Masoli et al.
    (2025) [2]_. A single first-order ``d`` gate of power 1 drives an
    ohmic current, with the gate written in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_d &= 0.13289 \, \exp(-(V + 83.94) / 24.3902) \\
        \beta_d &= 0.16994 \, \exp((V + 83.94) / 35.714)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. The template forms
    :math:`d_\infty = \alpha_d / (\alpha_d + \beta_d)` and
    :math:`\tau_d = 1 / (\alpha_d + \beta_d)` from these; half
    activation falls near -87.5 mV, and :math:`d_\infty` rises towards
    1 as the membrane hyperpolarizes. This class applies no voltage
    shift, and the reversal potential comes from the potassium ion
    object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.9 mS/cm2``, which
        is exactly the source mechanism's ``gkbar = 0.0009 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kir2p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kir2p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kir2p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``BC/channel/Kir2p3_MA25_BC.mod``. That file and the
    granule, Purkinje and stellate ports are byte-identical apart from
    their ``SUFFIX`` line and, in the Purkinje port only, the
    mechanism-local ``celsius`` default. The four BrainCell classes
    are likewise identical, so the rate constants above are shared
    verbatim with :class:`Kir2p3_MA2020_GrC`,
    :class:`Kir2p3_MA2024_PC` and :class:`Kir2p3_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **The rectification lives in the gate, not in the current.** The
    current expression is the plain ohmic
    ``g_max * d * (E_K - V)`` supplied by :class:`OhmicHH`; there is
    no Mg2+ or polyamine block term anywhere in the mechanism. The
    inward-rectifier behaviour comes entirely from :math:`d_\infty`
    increasing as the membrane hyperpolarizes.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 20 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`\alpha_d (1 - d) - \beta_d d` term by
    :math:`\phi = 3^{(T - 20)/10}`, which is exactly 3 at the default
    30 degrees Celsius. The ``.mod`` file instead multiplies ``Q10``
    into ``alp_d`` and ``bet_d``. The two forms are algebraically
    identical, but it means :meth:`f_d_alpha` and :meth:`f_d_beta`
    return the q10-free rates rather than the mechanism's
    ``alpha_d``/``beta_d``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` ``COMMENT`` carries a reference string that is the
    published title of D'Angelo et al. (2001) truncated mid-subtitle,
    plus the porting note "Suffix from Ubc_Kir to Kir2_3". Neither is
    treated as a citation here. The kinetics originate in the
    cerebellar granule cell model of D'Angelo et al. (2001) [1]_; the
    basket-cell paper [2]_ names the model this parameterisation was
    imported from, not the origin of the equations.

    **Conductance default.** ``0.9 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 100] mV``, covering ``d_inf`` and ``tau_d``, is not
    reproduced: both expressions are evaluated per call. NEURON used
    the boundary value outside that window, so any
    BrainCell-versus-NEURON divergence below -100 mV or above 100 mV
    is expected rather than a port error. The integration method was
    also changed from ``derivimplicit`` to ``cnexp``; with one
    independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm**2),
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

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp((V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV))

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp((V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV))


@register_channel("Kir2p3_MA2024_PC")
class Kir2p3_MA2024_PC(OhmicHH):
    r"""Kir2.3 inward-rectifier current of the Purkinje cell model.

    Hyperpolarization-activated inward-rectifier potassium current
    imported from the human Purkinje cell model of Masoli et al.
    (2024) [2]_. A single first-order ``d`` gate of power 1 drives an
    ohmic current, with the gate written in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_d &= 0.13289 \, \exp(-(V + 83.94) / 24.3902) \\
        \beta_d &= 0.16994 \, \exp((V + 83.94) / 35.714)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. The template forms
    :math:`d_\infty = \alpha_d / (\alpha_d + \beta_d)` and
    :math:`\tau_d = 1 / (\alpha_d + \beta_d)` from these; half
    activation falls near -87.5 mV, and :math:`d_\infty` rises towards
    1 as the membrane hyperpolarizes. This class applies no voltage
    shift, and the reversal potential comes from the potassium ion
    object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.9 mS/cm2``, which
        is exactly the source mechanism's ``gkbar = 0.0009 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. See the note below on the Purkinje port's
        divergent mechanism-local ``celsius`` default.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kir2p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kir2p3_MA2025_BC : Basket-cell port of the same mechanism.
    Kir2p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``PC/channel/Kir2p3_MA24_PC.mod``. That file and the
    granule, basket and stellate ports are byte-identical apart from
    their ``SUFFIX`` line and one ``PARAMETER`` default described
    below. The four BrainCell classes are likewise identical, so the
    rate constants above are shared verbatim with
    :class:`Kir2p3_MA2020_GrC`, :class:`Kir2p3_MA2025_BC` and
    :class:`Kir2p3_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    **Recorded divergence: the mechanism-local temperature default.**
    ``PC/channel/Kir2p3_MA24_PC.mod`` writes ``celsius = 10 (degC)``
    in its ``PARAMETER`` block where the granule, basket and stellate
    ports all write ``celsius = 30 (degC)``. BrainCell's ``temp``
    default is 30 degrees Celsius in all four classes, so this class
    alone does not reproduce its own source file's number. In NEURON
    ``celsius`` is a simulator-wide global that the host model
    normally sets, which is why a mechanism-local default carries
    little weight; the difference is recorded here rather than
    resolved, and the code is left unchanged. At 30 degrees Celsius
    the gate's q10 factor is 3; at 10 degrees Celsius it would be
    1/3.

    **The rectification lives in the gate, not in the current.** The
    current expression is the plain ohmic
    ``g_max * d * (E_K - V)`` supplied by :class:`OhmicHH`; there is
    no Mg2+ or polyamine block term anywhere in the mechanism. The
    inward-rectifier behaviour comes entirely from :math:`d_\infty`
    increasing as the membrane hyperpolarizes.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 20 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`\alpha_d (1 - d) - \beta_d d` term by
    :math:`\phi = 3^{(T - 20)/10}`. The ``.mod`` file instead
    multiplies ``Q10`` into ``alp_d`` and ``bet_d``. The two forms are
    algebraically identical, but it means :meth:`f_d_alpha` and
    :meth:`f_d_beta` return the q10-free rates rather than the
    mechanism's ``alpha_d``/``beta_d``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` ``COMMENT`` carries a reference string that is the
    published title of D'Angelo et al. (2001) truncated mid-subtitle.
    It is not treated as a citation here. The kinetics originate in
    the cerebellar granule cell model of D'Angelo et al. (2001) [1]_;
    the Purkinje-cell paper [2]_ names the model this
    parameterisation was imported from, not the origin of the
    equations.

    **Conductance default.** ``0.9 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 100] mV``, covering ``d_inf`` and ``tau_d``, is not
    reproduced: both expressions are evaluated per call. NEURON used
    the boundary value outside that window, so any
    BrainCell-versus-NEURON divergence below -100 mV or above 100 mV
    is expected rather than a port error. The integration method was
    also changed from ``derivimplicit`` to ``cnexp``; with one
    independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm**2),
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

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp((V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV))

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp((V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV))


@register_channel("Kir2p3_RI2021_SC")
class Kir2p3_RI2021_SC(OhmicHH):
    r"""Kir2.3 inward-rectifier current of the stellate cell model.

    Hyperpolarization-activated inward-rectifier potassium current
    imported from the cerebellar stellate cell model of Rizza et al.
    (2021) [2]_. A single first-order ``d`` gate of power 1 drives an
    ohmic current, with the gate written in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_d &= 0.13289 \, \exp(-(V + 83.94) / 24.3902) \\
        \beta_d &= 0.16994 \, \exp((V + 83.94) / 35.714)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. The template forms
    :math:`d_\infty = \alpha_d / (\alpha_d + \beta_d)` and
    :math:`\tau_d = 1 / (\alpha_d + \beta_d)` from these; half
    activation falls near -87.5 mV, and :math:`d_\infty` rises towards
    1 as the membrane hyperpolarizes. This class applies no voltage
    shift, and the reversal potential comes from the potassium ion
    object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.9 mS/cm2``, which
        is exactly the source mechanism's ``gkbar = 0.0009 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kir2p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kir2p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kir2p3_MA2025_BC : Basket-cell port of the same mechanism.
    KM_RI2021_SC : M-type current of the same stellate model, sharing
        the same origin paper.

    Notes
    -----
    Ported from ``SC/channel/Kir2p3_RI21_SC.mod``. That file and the
    granule, Purkinje and basket ports are byte-identical apart from
    their ``SUFFIX`` line and, in the Purkinje port only, the
    mechanism-local ``celsius`` default. The four BrainCell classes
    are likewise identical, so the rate constants above are shared
    verbatim with :class:`Kir2p3_MA2020_GrC`,
    :class:`Kir2p3_MA2024_PC` and :class:`Kir2p3_MA2025_BC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **The rectification lives in the gate, not in the current.** The
    current expression is the plain ohmic
    ``g_max * d * (E_K - V)`` supplied by :class:`OhmicHH`; there is
    no Mg2+ or polyamine block term anywhere in the mechanism. The
    inward-rectifier behaviour comes entirely from :math:`d_\infty`
    increasing as the membrane hyperpolarizes.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 20 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`\alpha_d (1 - d) - \beta_d d` term by
    :math:`\phi = 3^{(T - 20)/10}`, which is exactly 3 at the default
    30 degrees Celsius. The ``.mod`` file instead multiplies ``Q10``
    into ``alp_d`` and ``bet_d``. The two forms are algebraically
    identical, but it means :meth:`f_d_alpha` and :meth:`f_d_beta`
    return the q10-free rates rather than the mechanism's
    ``alpha_d``/``beta_d``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` ``COMMENT`` carries a reference string that is the
    published title of D'Angelo et al. (2001) truncated mid-subtitle,
    plus the porting note "Suffix from Ubc_Kir to Kir2_3". Neither is
    treated as a citation here. The kinetics originate in the
    cerebellar granule cell model of D'Angelo et al. (2001) [1]_; the
    stellate-cell paper [2]_ names the model this parameterisation was
    imported from, not the origin of the equations.

    **Conductance default.** ``0.9 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 100] mV``, covering ``d_inf`` and ``tau_d``, is not
    reproduced: both expressions are evaluated per call. NEURON used
    the boundary value outside that window, so any
    BrainCell-versus-NEURON divergence below -100 mV or above 100 mV
    is expected rather than a port error. The integration method was
    also changed from ``derivimplicit`` to ``cnexp``; with one
    independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm**2),
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

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp((V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV))

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp((V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV))


@register_channel("Kv1p1_MA2025_BC")
class Kv1p1_MA2025_BC(HH):
    r"""Kv1.1 low-threshold potassium current of the basket cell model.

    Non-inactivating, low-threshold potassium current carried by Kv1.1
    subunits, imported from the cerebellar basket cell model of Masoli
    et al. (2025) [3]_. Gating is a single ``n`` gate of power 4 in
    alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.12889 \, \exp((V + 45) / 33.90877) \\
        \beta_n &= 0.12889 \, \exp(-(V + 45) / 12.42101)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -45 mV, and the :math:`n^4` conductance reaches half its
    maximum near -29.9 mV -- consistent with the
    ``Vhalf = -28.8 +- 2.3 mV`` that the ``.mod`` header quotes from
    the Zerr et al. (1998) recordings [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``, which
        is exactly the source mechanism's ``gbar = 0.004 S/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv1p1_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv1p1_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv1p1_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``BC/channel/Kv1p1_MA25_BC.mod``. That file and the
    Golgi, granule, Purkinje and stellate ports are identical apart
    from their ``SUFFIX`` line and one line of indentation, and the
    five BrainCell classes are identical too: the rate constants and
    the gating-current constants above are shared verbatim with
    :class:`Kv1p1_MA2020_GoC`, :class:`Kv1p1_MA2020_GrC`,
    :class:`Kv1p1_MA2024_PC` and :class:`Kv1p1_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 2.7978` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    divides its ``taun`` by the same factor. The two forms are
    algebraically identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau(V)` data from human Kv1.1
    expressed in Xenopus oocytes by Zerr et al. (1998) [1]_, and
    names the RIKEN implementation published by Akemann et al. (2009)
    [2]_ as its model reference. The basket-cell paper [3]_ names the
    model this parameterisation was imported from, not the origin of
    the equations.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic
           ataxia mutations in Kv1.1 alter potassium channel function
           by dominant negative effects or haploinsufficiency. The
           Journal of Neuroscience, 18(8), 2842-2848.
           doi:10.1523/JNEUROSCI.18-08-02842.1998
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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
    r"""Kv1.1 low-threshold potassium current of the Purkinje model.

    Non-inactivating, low-threshold potassium current carried by Kv1.1
    subunits, imported from the human Purkinje cell model of Masoli
    et al. (2024) [3]_. Gating is a single ``n`` gate of power 4 in
    alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.12889 \, \exp((V + 45) / 33.90877) \\
        \beta_n &= 0.12889 \, \exp(-(V + 45) / 12.42101)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -45 mV, and the :math:`n^4` conductance reaches half its
    maximum near -29.9 mV -- consistent with the
    ``Vhalf = -28.8 +- 2.3 mV`` that the ``.mod`` header quotes from
    the Zerr et al. (1998) recordings [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``, which
        is exactly the source mechanism's ``gbar = 0.004 S/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv1p1_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv1p1_MA2025_BC : Basket-cell port of the same mechanism.
    Kv1p1_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``PC/channel/Kv1p1_MA24_PC.mod``. That file and the
    Golgi, granule, basket and stellate ports are identical apart
    from their ``SUFFIX`` line and one line of indentation, and the
    five BrainCell classes are identical too: the rate constants and
    the gating-current constants above are shared verbatim with
    :class:`Kv1p1_MA2020_GoC`, :class:`Kv1p1_MA2020_GrC`,
    :class:`Kv1p1_MA2025_BC` and :class:`Kv1p1_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 2.7978` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    divides its ``taun`` by the same factor. The two forms are
    algebraically identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau(V)` data from human Kv1.1
    expressed in Xenopus oocytes by Zerr et al. (1998) [1]_, and
    names the RIKEN implementation published by Akemann et al. (2009)
    [2]_ as its model reference. The Purkinje-cell paper [3]_ names
    the model this parameterisation was imported from, not the origin
    of the equations.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic
           ataxia mutations in Kv1.1 alter potassium channel function
           by dominant negative effects or haploinsufficiency. The
           Journal of Neuroscience, 18(8), 2842-2848.
           doi:10.1523/JNEUROSCI.18-08-02842.1998
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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
    r"""Kv1.1 low-threshold potassium current of the stellate model.

    Non-inactivating, low-threshold potassium current carried by Kv1.1
    subunits, imported from the cerebellar stellate cell model of
    Rizza et al. (2021) [3]_. Gating is a single ``n`` gate of power 4
    in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.12889 \, \exp((V + 45) / 33.90877) \\
        \beta_n &= 0.12889 \, \exp(-(V + 45) / 12.42101)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -45 mV, and the :math:`n^4` conductance reaches half its
    maximum near -29.9 mV -- consistent with the
    ``Vhalf = -28.8 +- 2.3 mV`` that the ``.mod`` header quotes from
    the Zerr et al. (1998) recordings [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``, which
        is exactly the source mechanism's ``gbar = 0.004 S/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv1p1_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv1p1_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv1p1_MA2025_BC : Basket-cell port of the same mechanism.

    Notes
    -----
    Ported from ``SC/channel/Kv1p1_RI21_SC.mod``. That file and the
    Golgi, granule, Purkinje and basket ports are identical apart
    from their ``SUFFIX`` line and one line of indentation, and the
    five BrainCell classes are identical too: the rate constants and
    the gating-current constants above are shared verbatim with
    :class:`Kv1p1_MA2020_GoC`, :class:`Kv1p1_MA2020_GrC`,
    :class:`Kv1p1_MA2024_PC` and :class:`Kv1p1_MA2025_BC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 2.7978` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    divides its ``taun`` by the same factor. The two forms are
    algebraically identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau(V)` data from human Kv1.1
    expressed in Xenopus oocytes by Zerr et al. (1998) [1]_, and
    names the RIKEN implementation published by Akemann et al. (2009)
    [2]_ as its model reference. The stellate-cell paper [3]_ names
    the model this parameterisation was imported from, not the origin
    of the equations.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic
           ataxia mutations in Kv1.1 alter potassium channel function
           by dominant negative effects or haploinsufficiency. The
           Journal of Neuroscience, 18(8), 2842-2848.
           doi:10.1523/JNEUROSCI.18-08-02842.1998
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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


@register_channel("Kv1p5_MA2024_PC")
class Kv1p5_MA2024_PC(HH):
    r"""Kv1.5 ultrarapid delayed-rectifier current (IKur), K path only.

    Hodgkin-Huxley model of the cardiac ultrarapid delayed rectifier
    IKur, fitted to human atrial myocyte recordings by Feng et al.
    (1998) [1]_ and imported into BrainCell from the human Purkinje
    cell model of Masoli et al. (2024) [2]_. Three gates -- ``m``
    (power 3), ``n`` and ``u`` -- combine with a voltage-dependent
    conductance factor, so :meth:`current` returns

    .. math::

        g_{\max} \left(0.1 + \frac{1}{1 + \exp(-(V - 15)/13)}\right)
        m^3 n u \, (E_K - V)

    with the gate kinetics

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V + 30.3)/9.6)} \\
        \tau_m &= \frac{1}{3(\alpha_m + \beta_m)} T_{\text{act}},
        \quad
        \alpha_m = \frac{0.65 \, q_{10}}
                        {\exp(-(V + 10)/8.5) + \exp(-(V - 30)/59)},
        \quad
        \beta_m = \frac{0.65 \, q_{10}}{2.5 + \exp((V + 82)/17)} \\
        n_\infty &= 0.25 + \frac{1}{1.35 + \exp((V + 7)/14)} \\
        \tau_n &= \frac{1}{3(\alpha_n + \beta_n)} T_{\text{inactf}},
        \quad
        \alpha_n = \frac{0.001 \, q_{10}}
                        {2.4 + 10.9 \exp(-(V + 90)/78)},
        \quad
        \beta_n = 0.001 \, q_{10} \exp((V - 168)/16) \\
        u_\infty &= 0.1 + \frac{1}{1.1 + \exp((V + 7)/14)} \\
        \tau_u &= 6800 \, T_{\text{inacts}}
        \end{aligned}

    where :math:`V` is in millivolts, :math:`\tau` in milliseconds and
    :math:`q_{10} = 2.2^{(T - 37)/10}` with :math:`T` in degrees
    Celsius. The reversal potential comes from the potassium ion
    object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default
        ``0.13195e-3 siemens/cm2``. This is the BrainCell name for the
        NEURON ``gKur`` parameter and is exactly its ``.mod`` value.
    temp : array-like, optional
        Absolute temperature entering :math:`q_{10}`, default
        37 degrees Celsius. This equals the :math:`q_{10}` reference
        temperature, so the default factor is exactly 1.
    Tauact : array-like or callable, optional
        Activation time-scale multiplier for :math:`\tau_m`, default
        ``1.0`` (dimensionless).
    Tauinactf : array-like or callable, optional
        Fast-inactivation time-scale multiplier for :math:`\tau_n`,
        default ``1.0`` (dimensionless).
    Tauinacts : array-like or callable, optional
        Slow-inactivation time-scale multiplier for :math:`\tau_u`,
        default ``1.0`` (dimensionless).
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.channel.Kv1p5_MA2020_GrC : Granule-cell subclass that
        inherits these gate kinetics and adds the nonspecific cation
        current component.
    Kv1p1_MA2024_PC : Low-threshold Kv1 current of the same Purkinje
        cell model.

    Notes
    -----
    Ported from ``PC/channel/Kv1p5_MA24_PC.mod``.

    **This is a cardiac mechanism, not a cerebellar one.** The
    ``.mod`` ``TITLE`` reads "Cardiac IKur current & nonspec cation
    current with identical kinetics", and its kinetics were fitted to
    human atrial myocyte recordings [1]_, not to any cerebellar
    recording. The Purkinje-cell citation [2]_ names the model
    BrainCell imported this parameterisation from, not the origin of
    the kinetics.

    **Only the potassium path is converted.** The Purkinje ``.mod``
    file computes a nonspecific cation current ``ino`` with kinetics
    identical to ``ik``, but its ``USEION no WRITE ino`` line is
    commented out, so ``ino`` survives only as a ``RANGE`` variable
    with no current owner. BrainCell converts the default ``ik`` path
    alone; this class has no ``gnonspec`` parameter and emits no
    nonspecific current. The granule-cell sibling
    :class:`braincell.channel.Kv1p5_MA2020_GrC`, whose ``.mod`` file
    leaves that line enabled, subclasses this one and adds the second
    component.

    **q10 asymmetry in** :meth:`f_u_tau`. Temperature scaling is not
    attached through the gate objects: none of the three gates sets
    ``phi`` or ``q10``, so :meth:`HH.gate_phi` resolves to ``1.0`` for
    ``m``, ``n`` and ``u`` alike. Instead the private ``_q10`` method
    computes :math:`2.2^{(T - 37)/10}` and multiplies it into the
    ``alpha``/``beta`` rates used by :meth:`f_m_tau` and
    :meth:`f_n_tau` only. :meth:`f_u_tau` returns the constant
    ``6800 * Tauinacts`` milliseconds: it is voltage-independent, it
    receives no :math:`q_{10}` scaling, and unlike its two siblings it
    also carries no factor of :math:`1/3`. This reproduces the
    ``.mod`` file's ``utau = 6800*Tauinacts`` exactly and is the
    mechanism's own code path, not a BrainCell convention and not a
    closed-form temperature dependence printed in either cited paper.

    **Conductance default.** ``0.13195e-3 siemens/cm2`` is the
    deposit's tuned value, carried across from the ``.mod`` file. It
    is not a value printed in either cited paper.

    **Import deviations.** The integration method was changed from
    ``derivimplicit`` to ``cnexp``; the three gate ODEs are
    independent, so the substitution is exact. This mechanism carries
    no NMODL ``TABLE``, so no table-removal deviation applies.

    References
    ----------
    .. [1] Feng, J., Xu, D., Wang, Z., & Nattel, S. (1998). Ultrarapid
           delayed rectifier current inactivation in human atrial
           myocytes: properties and consequences. American Journal of
           Physiology-Heart and Circulatory Physiology, 275(5),
           H1717-H1725.
           doi:10.1152/ajpheart.1998.275.5.H1717
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

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
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.13195e-3 * (u.siemens / u.cm**2),
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
        alpha = self._q10() * 0.65 / (u.math.exp(-(V + 10.0) / 8.5) + u.math.exp(-(V - 30.0) / 59.0))
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
    r"""Kv3.3 high-threshold potassium current of the Purkinje model.

    Fast-activating, non-inactivating high-threshold potassium
    current imported from the human Purkinje cell model of Masoli
    et al. (2024) [3]_. Gating is a single ``n`` gate of power 4 in
    alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.22 \, \exp((V + 16) / 26.5) \\
        \beta_n &= 0.22 \, \exp(-(V + 16) / 26.5)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -16 mV, and the :math:`n^4` conductance reaches half its
    maximum near +6.1 mV -- the high activation threshold the source
    mechanism's ``TITLE`` line describes.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.005 S/cm2``,
        equivalently ``5.0 mS/cm2``, which is exactly the source
        mechanism's ``gbar = 0.005 S/cm2``. Most channels in this
        module spell their default in ``mS/cm2``; this one keeps the
        ``.mod`` file's own ``S/cm2``, so read the number with care.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2024_PC : Kv1.1 current of the same model, sharing this
        class's gating-current construction.
    Kv3p4_MA2024_PC : The other Kv3-family current of the same model.

    Notes
    -----
    Ported from ``PC/channel/Kv3p3_MA24_PC.mod``. No other cell type
    in this repository imports this mechanism, so unlike most
    channels in this module it has no sibling ports.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 1.9196` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.
    The construction is the one :class:`Kv1p1_MA2024_PC` uses; only
    :math:`z_n` and the rate constants differ.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    multiplies ``qt`` into ``alpha`` and ``beta``, which divides its
    ``taun`` by the same factor. The two forms are algebraically
    identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau_n(V)` data recorded from
    rat cerebellar Purkinje neurons by Martina et al. (2007) [1]_,
    and names the RIKEN implementation published by Akemann et al.
    (2009) [2]_ as its model reference. The Purkinje-cell paper [3]_
    names the model this parameterisation was imported from, not the
    origin of the equations.

    Two header transcription errors are worth knowing about, and
    neither is reproduced above. The header prints the Martina page
    range as "563-671"; the published range is 563-571, with a stray
    unbalanced parenthesis before it. And the mechanism itself is
    named for the Kv3 family as a whole -- its ``TITLE`` reads
    "Voltage-gated potassium channel from Kv3 subunits" -- with the
    specific ".3" arriving only through the ported file's own
    ``: Suffix from Kv3 to Kv3_3`` rename line, not from either
    origin paper.

    **Conductance default.** ``0.005 S/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Martina, M., Metz, A. E., & Bean, B. P. (2007).
           Voltage-dependent potassium currents during fast spikes of
           rat cerebellar Purkinje neurons: inhibition by BDS-I
           toxin. Journal of Neurophysiology, 97(1), 563-571.
           doi:10.1152/jn.00269.2006
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """


    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.005 * (u.siemens / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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


@register_channel("Kv3p4_MA2025_BC")
class Kv3p4_MA2025_BC(OhmicHH):
    r"""Fast TEA-sensitive potassium current of the basket cell model.

    High-threshold, fast-activating and only partially inactivating
    potassium current, imported from the
    cerebellar basket cell model of Masoli et al. (2025) [2]_.

    Gating is an ``m`` gate of power 3 and an ``h`` gate of power 1,
    both written in steady-state/time-constant form and both
    evaluated at a junction-potential-corrected voltage
    :math:`V' = V + 11`:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V' + 24) / 15.4)} \\
        h_\infty &= 0.31
          + \frac{0.69}{1 + \exp((V' + 5.802) / 11.2)}
        \end{aligned}

    where :math:`V'` is in millivolts. Inactivation is only partial:
    :math:`h_\infty` decays to a floor of 0.31 rather than to zero,
    so about 31 per cent of the conductance survives a maintained
    depolarisation. Both time constants are piecewise, in
    milliseconds:

    .. math::

        \tau_m = 10^3 \times \begin{cases}
        3 \, (3.4225 \times 10^{-5}
          + 0.00498 \, e^{V' / 28.29}), & V' < -35 \\
        1.2851 \times 10^{-4} + \dfrac{1}{e^{(V' + 100.7)/12.9}
          + e^{(V' - 56)/(-23.1)}}, & V' \ge -35
        \end{cases}

    .. math::

        \tau_h = 10^3 \times \begin{cases}
        0.0012 + 0.0023 \, e^{-0.141 \, V'}, & V' > 0 \\
        1.2202 \times 10^{-5}
          + 0.012 \, e^{-((V' + 56.3)/49.6)^2}, & V' \le 0
        \end{cases}

    Each ladder is selected with ``u.math.where`` on the predicates
    ``V' < -35`` and ``V' > 0``, so the boundary value itself falls
    to the second line in both.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.004 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. See the temperature note below: this
        default is BrainCell's own, not a value the source mechanism
        states.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv3p4_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv3p4_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv3p4_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv3p4_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``BC/channel/Kv3p4_MA25_BC.mod``. All five cell-type
    ports of this mechanism carry the same equations and the same
    constants, and so do the five BrainCell classes: everything above
    is shared verbatim with :class:`Kv3p4_MA2020_GoC`,
    :class:`Kv3p4_MA2020_GrC`, :class:`Kv3p4_MA2024_PC` and
    :class:`Kv3p4_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    The ``BC`` file additionally declares a ``g_equiv`` RANGE variable
    and assigns ``g_equiv = gkbar * m^3 * h`` in its ``BREAKPOINT``.
    That is a diagnostic output for the NEURON comparison harness
    rather than part of the dynamics, and BrainCell does not port it.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 37 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 37)/10}`. The ``.mod`` file instead
    divides its ``mtau`` and ``htau`` by the same factor. The two
    forms are algebraically identical, but it means :meth:`f_m_tau`
    and :meth:`f_h_tau` return q10-free time constants rather than
    the mechanism's ``mtau`` and ``htau``.

    **The default temperature is BrainCell's, not the mechanism's.**
    Unlike most channels in this module, this ``.mod`` file never
    declares a ``celsius`` value; it reads NEURON's global instead.
    The ``temp`` default of 22 degrees Celsius is therefore a
    BrainCell choice, and because it sits 15 degrees below the
    gates' reference it makes the default temperature factor
    :math:`\phi \approx 0.192` -- gating roughly 5.2 times slower
    than at 37 degrees Celsius. Callers reproducing a published
    simulation should set ``temp`` explicitly.

    **Provenance, and a name the sources do not support.** These files
    are ModelDB accession 48332's ``kpkj.mod`` renamed. The header
    lines ": HH TEA-sensitive Purkinje potassium current" and
    ": Created 8/5/02 - nwg" are the deposit's own, "nwg" being
    Nathan W. Gouwens, second author of Khaliq et al. (2003) [1]_, and
    the parameters ``mivh = -24 mV``, ``mik = 15.4`` and
    ``hiy0 = 0.31`` reproduce that paper's Table 1 row for the current
    it calls **K fast**. The trailing ``: Suffix from kpkj to Kv3_4``
    line is a BrainCell-local addition and is not in the deposit. The
    model paper [2]_ names the deposit this parameterisation was
    imported from, not the origin of the equations.

    That rename line is the whole basis of the ``Kv3p4`` name, and
    it does not survive contact with the source. Khaliq et al. call
    this current K fast throughout and never name a Kv subunit; the
    strings "Kv3.4", "Kv3.3" and "Kv3.1" appear nowhere in the
    paper, and its only mention of Kv3 at all is one Discussion
    sentence observing that the positive activation range is typical
    of the Kv3 family. **The class name therefore asserts a subunit
    identity the cited literature does not establish.** Under this
    project's mismatch policy the name is documented rather than
    changed: read ``Kv3p4`` as a label for the TEA-sensitive fast K
    current of Khaliq et al. (2003), which those authors associate
    with the Kv3 family, and not as a claim about Kv3.4.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here. Do not
    assume it shares the deviations of its ``Kv4p3`` neighbour,
    which has three.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [2] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
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
            self.mty0 + 1.0 / (u.math.exp((V + self.mtvh1) / self.mtk1) + u.math.exp((V + self.mtvh2) / self.mtk2)),
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
            1.2202e-05 + 0.012 * u.math.exp(-(((V + 56.3) / 49.6) ** 2)),
        )
        return 1000.0 * htau_func


@register_channel("Kv3p4_MA2024_PC")
class Kv3p4_MA2024_PC(OhmicHH):
    r"""Fast TEA-sensitive potassium current of the Purkinje cell model.

    High-threshold, fast-activating and only partially inactivating
    potassium current, imported from the
    human Purkinje cell model of Masoli et al. (2024) [2]_.

    Gating is an ``m`` gate of power 3 and an ``h`` gate of power 1,
    both written in steady-state/time-constant form and both
    evaluated at a junction-potential-corrected voltage
    :math:`V' = V + 11`:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V' + 24) / 15.4)} \\
        h_\infty &= 0.31
          + \frac{0.69}{1 + \exp((V' + 5.802) / 11.2)}
        \end{aligned}

    where :math:`V'` is in millivolts. Inactivation is only partial:
    :math:`h_\infty` decays to a floor of 0.31 rather than to zero,
    so about 31 per cent of the conductance survives a maintained
    depolarisation. Both time constants are piecewise, in
    milliseconds:

    .. math::

        \tau_m = 10^3 \times \begin{cases}
        3 \, (3.4225 \times 10^{-5}
          + 0.00498 \, e^{V' / 28.29}), & V' < -35 \\
        1.2851 \times 10^{-4} + \dfrac{1}{e^{(V' + 100.7)/12.9}
          + e^{(V' - 56)/(-23.1)}}, & V' \ge -35
        \end{cases}

    .. math::

        \tau_h = 10^3 \times \begin{cases}
        0.0012 + 0.0023 \, e^{-0.141 \, V'}, & V' > 0 \\
        1.2202 \times 10^{-5}
          + 0.012 \, e^{-((V' + 56.3)/49.6)^2}, & V' \le 0
        \end{cases}

    Each ladder is selected with ``u.math.where`` on the predicates
    ``V' < -35`` and ``V' > 0``, so the boundary value itself falls
    to the second line in both.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.004 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. See the temperature note below: this
        default is BrainCell's own, not a value the source mechanism
        states.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv3p4_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv3p4_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv3p4_MA2025_BC : Basket-cell port of the same mechanism.
    Kv3p4_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``PC/channel/Kv3p4_MA24_PC.mod``. All five cell-type
    ports of this mechanism carry the same equations and the same
    constants, and so do the five BrainCell classes: everything above
    is shared verbatim with :class:`Kv3p4_MA2020_GoC`,
    :class:`Kv3p4_MA2020_GrC`, :class:`Kv3p4_MA2025_BC` and
    :class:`Kv3p4_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 37 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 37)/10}`. The ``.mod`` file instead
    divides its ``mtau`` and ``htau`` by the same factor. The two
    forms are algebraically identical, but it means :meth:`f_m_tau`
    and :meth:`f_h_tau` return q10-free time constants rather than
    the mechanism's ``mtau`` and ``htau``.

    **The default temperature is BrainCell's, not the mechanism's.**
    Unlike most channels in this module, this ``.mod`` file never
    declares a ``celsius`` value; it reads NEURON's global instead.
    The ``temp`` default of 22 degrees Celsius is therefore a
    BrainCell choice, and because it sits 15 degrees below the
    gates' reference it makes the default temperature factor
    :math:`\phi \approx 0.192` -- gating roughly 5.2 times slower
    than at 37 degrees Celsius. Callers reproducing a published
    simulation should set ``temp`` explicitly.

    **Provenance, and a name the sources do not support.** These files
    are ModelDB accession 48332's ``kpkj.mod`` renamed. The header
    lines ": HH TEA-sensitive Purkinje potassium current" and
    ": Created 8/5/02 - nwg" are the deposit's own, "nwg" being
    Nathan W. Gouwens, second author of Khaliq et al. (2003) [1]_, and
    the parameters ``mivh = -24 mV``, ``mik = 15.4`` and
    ``hiy0 = 0.31`` reproduce that paper's Table 1 row for the current
    it calls **K fast**. The trailing ``: Suffix from kpkj to Kv3_4``
    line is a BrainCell-local addition and is not in the deposit. The
    model paper [2]_ names the deposit this parameterisation was
    imported from, not the origin of the equations.

    That rename line is the whole basis of the ``Kv3p4`` name, and
    it does not survive contact with the source. Khaliq et al. call
    this current K fast throughout and never name a Kv subunit; the
    strings "Kv3.4", "Kv3.3" and "Kv3.1" appear nowhere in the
    paper, and its only mention of Kv3 at all is one Discussion
    sentence observing that the positive activation range is typical
    of the Kv3 family. **The class name therefore asserts a subunit
    identity the cited literature does not establish.** Under this
    project's mismatch policy the name is documented rather than
    changed: read ``Kv3p4`` as a label for the TEA-sensitive fast K
    current of Khaliq et al. (2003), which those authors associate
    with the Kv3 family, and not as a claim about Kv3.4.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here. Do not
    assume it shares the deviations of its ``Kv4p3`` neighbour,
    which has three.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
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
            self.mty0 + 1.0 / (u.math.exp((V + self.mtvh1) / self.mtk1) + u.math.exp((V + self.mtvh2) / self.mtk2)),
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
            1.2202e-05 + 0.012 * u.math.exp(-(((V + 56.3) / 49.6) ** 2)),
        )
        return 1000.0 * htau_func


@register_channel("Kv3p4_RI2021_SC")
class Kv3p4_RI2021_SC(OhmicHH):
    r"""Fast TEA-sensitive potassium current of the stellate cell model.

    High-threshold, fast-activating and only partially inactivating
    potassium current, imported from the
    cerebellar stellate cell model of Rizza et al. (2021) [2]_.

    Gating is an ``m`` gate of power 3 and an ``h`` gate of power 1,
    both written in steady-state/time-constant form and both
    evaluated at a junction-potential-corrected voltage
    :math:`V' = V + 11`:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V' + 24) / 15.4)} \\
        h_\infty &= 0.31
          + \frac{0.69}{1 + \exp((V' + 5.802) / 11.2)}
        \end{aligned}

    where :math:`V'` is in millivolts. Inactivation is only partial:
    :math:`h_\infty` decays to a floor of 0.31 rather than to zero,
    so about 31 per cent of the conductance survives a maintained
    depolarisation. Both time constants are piecewise, in
    milliseconds:

    .. math::

        \tau_m = 10^3 \times \begin{cases}
        3 \, (3.4225 \times 10^{-5}
          + 0.00498 \, e^{V' / 28.29}), & V' < -35 \\
        1.2851 \times 10^{-4} + \dfrac{1}{e^{(V' + 100.7)/12.9}
          + e^{(V' - 56)/(-23.1)}}, & V' \ge -35
        \end{cases}

    .. math::

        \tau_h = 10^3 \times \begin{cases}
        0.0012 + 0.0023 \, e^{-0.141 \, V'}, & V' > 0 \\
        1.2202 \times 10^{-5}
          + 0.012 \, e^{-((V' + 56.3)/49.6)^2}, & V' \le 0
        \end{cases}

    Each ladder is selected with ``u.math.where`` on the predicates
    ``V' < -35`` and ``V' > 0``, so the boundary value itself falls
    to the second line in both.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.004 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. See the temperature note below: this
        default is BrainCell's own, not a value the source mechanism
        states.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv3p4_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv3p4_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv3p4_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv3p4_MA2025_BC : Basket-cell port of the same mechanism.

    Notes
    -----
    Ported from ``SC/channel/Kv3p4_RI21_SC.mod``. All five cell-type
    ports of this mechanism carry the same equations and the same
    constants, and so do the five BrainCell classes: everything above
    is shared verbatim with :class:`Kv3p4_MA2020_GoC`,
    :class:`Kv3p4_MA2020_GrC`, :class:`Kv3p4_MA2024_PC` and
    :class:`Kv3p4_MA2025_BC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    The ``SC`` file additionally declares a ``g_equiv`` RANGE variable
    and assigns ``g_equiv = gkbar * m^3 * h`` in its ``BREAKPOINT``.
    That is a diagnostic output for the NEURON comparison harness
    rather than part of the dynamics, and BrainCell does not port it.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 37 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 37)/10}`. The ``.mod`` file instead
    divides its ``mtau`` and ``htau`` by the same factor. The two
    forms are algebraically identical, but it means :meth:`f_m_tau`
    and :meth:`f_h_tau` return q10-free time constants rather than
    the mechanism's ``mtau`` and ``htau``.

    **The default temperature is BrainCell's, not the mechanism's.**
    Unlike most channels in this module, this ``.mod`` file never
    declares a ``celsius`` value; it reads NEURON's global instead.
    The ``temp`` default of 22 degrees Celsius is therefore a
    BrainCell choice, and because it sits 15 degrees below the
    gates' reference it makes the default temperature factor
    :math:`\phi \approx 0.192` -- gating roughly 5.2 times slower
    than at 37 degrees Celsius. Callers reproducing a published
    simulation should set ``temp`` explicitly.

    **Provenance, and a name the sources do not support.** These files
    are ModelDB accession 48332's ``kpkj.mod`` renamed. The header
    lines ": HH TEA-sensitive Purkinje potassium current" and
    ": Created 8/5/02 - nwg" are the deposit's own, "nwg" being
    Nathan W. Gouwens, second author of Khaliq et al. (2003) [1]_, and
    the parameters ``mivh = -24 mV``, ``mik = 15.4`` and
    ``hiy0 = 0.31`` reproduce that paper's Table 1 row for the current
    it calls **K fast**. The trailing ``: Suffix from kpkj to Kv3_4``
    line is a BrainCell-local addition and is not in the deposit. The
    model paper [2]_ names the deposit this parameterisation was
    imported from, not the origin of the equations.

    That rename line is the whole basis of the ``Kv3p4`` name, and
    it does not survive contact with the source. Khaliq et al. call
    this current K fast throughout and never name a Kv subunit; the
    strings "Kv3.4", "Kv3.3" and "Kv3.1" appear nowhere in the
    paper, and its only mention of Kv3 at all is one Discussion
    sentence observing that the positive activation range is typical
    of the Kv3 family. **The class name therefore asserts a subunit
    identity the cited literature does not establish.** Under this
    project's mismatch policy the name is documented rather than
    changed: read ``Kv3p4`` as a label for the TEA-sensitive fast K
    current of Khaliq et al. (2003), which those authors associate
    with the Kv3 family, and not as a claim about Kv3.4.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here. Do not
    assume it shares the deviations of its ``Kv4p3`` neighbour,
    which has three.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [2] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
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
            self.mty0 + 1.0 / (u.math.exp((V + self.mtvh1) / self.mtk1) + u.math.exp((V + self.mtvh2) / self.mtk2)),
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
            1.2202e-05 + 0.012 * u.math.exp(-(((V + 56.3) / 49.6) ** 2)),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA2025_BC")
class Kv4p3_MA2025_BC(OhmicHH):
    r"""A-type transient potassium current of the basket cell model.

    Fast-inactivating A-type potassium current, imported from the
    cerebellar basket cell model of Masoli et al. (2025) [2]_.

    Gating is an ``a`` gate of power 3 and a ``b`` gate of power 1,
    each with an explicit Boltzmann steady state and a time constant
    built from an alpha/beta pair:

    .. math::

        \begin{aligned}
        a_\infty &= \frac{1}{1 + \exp((V + 38) / (-17))} \\
        b_\infty &= \frac{1}{1 + \exp((V + 78.8) / 8.4)} \\
        \tau_a &= \frac{1}{\alpha_a + \beta_a}, \qquad
        \tau_b = \frac{1}{\alpha_b + \beta_b}
        \end{aligned}

    with, writing :math:`\sigma(x, y) = 1 / (e^{x/y} + 1)` for the
    module-level ``_sigm`` helper,

    .. math::

        \begin{aligned}
        \alpha_a &= 0.8147 \, \sigma(V + 9.17203,\ -23.3271) \\
        \beta_a &= 0.1655 \big/ e^{(V + 18.2791) / 19.4718} \\
        \alpha_b &= 0.0368 \, \sigma(V + 111.332,\ 12.8433) \\
        \beta_b &= 0.0345 \, \sigma(V + 49.9537,\ -8.90123)
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and the time constants are in milliseconds. This class applies
    no voltage shift, and the reversal potential comes from the
    potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``3.2 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.0032 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv4p3_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv4p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv4p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv4p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``BC/channel/Kv4p3_MA25_BC.mod``. The five cell-type
    ports of this mechanism are byte-identical apart from their
    ``SUFFIX`` line and their ``celsius`` default, and the five
    BrainCell classes carry the same rate constants: every equation
    above is shared verbatim with :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2020_GrC`, :class:`Kv4p3_MA2024_PC` and
    :class:`Kv4p3_RI2021_SC`. What differs is the default ``temp``,
    the deposit each was imported from, and therefore the model paper
    cited below.

    ``Kalpha_a`` is stored here as a bare float and passed straight to
    ``_sigm``, whereas :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2020_GrC` and :class:`Kv4p3_RI2021_SC` attach
    ``u.mV`` to it and convert with ``.to_decimal(u.mV)`` at use. The
    arithmetic is identical -- both forms divide a millivolt
    difference by -23.3271 -- but the inconsistency is real, and on
    this documentation-only branch it is recorded rather than fixed.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 25.5 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 25.5)/10}` (about 1.64 at the default
    30 degrees Celsius). The ``.mod`` file instead multiplies
    the same ``Q10`` into ``alp_a``, ``bet_a``, ``alp_b`` and
    ``bet_b``, which divides its ``tau_a`` and ``tau_b`` by that
    factor. The two forms are algebraically identical, but it means
    :meth:`f_a_tau` and :meth:`f_b_tau` return q10-free time
    constants rather than the mechanism's ``tau_a`` and ``tau_b``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries
    ``Author: E.D'Angelo, T.Nieus, A. Fontana``. That credit line is
    copy-pasted verbatim across every cell-type port of this
    mechanism, and it names authors 1, 2 and 7 of an eight-author
    paper, so it is not treated as a citation here. The kinetics
    originate in the cerebellar granule cell model of D'Angelo et al.
    (2001) [1]_; the model paper [2]_ names the deposit this
    parameterisation was imported from, not the origin of the
    equations. The commented-out alternatives that remain in the
    ``.mod`` file -- ``linoid`` forms of each rate, and a
    "Bardoni Belluzzi" steady-state block -- are never evaluated by
    the mechanism and are not reproduced here.

    **Conductance default.** ``3.2 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** Three, all recorded against this symbol.
    First, the original mechanism's NMODL ``TABLE`` over
    ``[-100, 30] mV``, covering ``a_inf``, ``tau_a``, ``b_inf`` and
    ``tau_b``, is not reproduced: all four expressions are evaluated
    per call. NEURON clamped tabulated values to the boundary
    outside that window, so any BrainCell-versus-NEURON divergence
    below -100 mV or above 30 mV is expected rather than a port
    error. Second, the integration method was changed from
    ``derivimplicit`` to ``cnexp``; the two gate ODEs are
    independent, so the substitution is exact. Third, four
    parameters are carried at NEURON's compiled six-significant-
    figure precision rather than at the ``.mod`` source text's:
    ``Kalpha_a`` ``-23.32708`` becomes ``-23.3271``, ``Kbeta_a``
    ``19.47175`` becomes ``19.4718``, ``V0beta_a`` ``-18.27914``
    becomes ``-18.2791`` and ``V0alpha_b`` ``-111.33209`` becomes
    ``-111.332``. That rewrite reaches those four parameters only;
    ordinary in-formula literals keep their source values.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm**2),
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

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp((V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV))

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
class Kv4p3_MA2024_PC(OhmicHH):
    r"""A-type transient potassium current of the Purkinje cell model.

    Fast-inactivating A-type potassium current, imported from the
    human Purkinje cell model of Masoli et al. (2024) [2]_.

    Gating is an ``a`` gate of power 3 and a ``b`` gate of power 1,
    each with an explicit Boltzmann steady state and a time constant
    built from an alpha/beta pair:

    .. math::

        \begin{aligned}
        a_\infty &= \frac{1}{1 + \exp((V + 38) / (-17))} \\
        b_\infty &= \frac{1}{1 + \exp((V + 78.8) / 8.4)} \\
        \tau_a &= \frac{1}{\alpha_a + \beta_a}, \qquad
        \tau_b = \frac{1}{\alpha_b + \beta_b}
        \end{aligned}

    with, writing :math:`\sigma(x, y) = 1 / (e^{x/y} + 1)` for the
    module-level ``_sigm`` helper,

    .. math::

        \begin{aligned}
        \alpha_a &= 0.8147 \, \sigma(V + 9.17203,\ -23.3271) \\
        \beta_a &= 0.1655 \big/ e^{(V + 18.2791) / 19.4718} \\
        \alpha_b &= 0.0368 \, \sigma(V + 111.332,\ 12.8433) \\
        \beta_b &= 0.0345 \, \sigma(V + 49.9537,\ -8.90123)
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and the time constants are in milliseconds. This class applies
    no voltage shift, and the reversal potential comes from the
    potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``3.2 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.0032 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv4p3_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv4p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv4p3_MA2025_BC : Basket-cell port of the same mechanism.
    Kv4p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``PC/channel/Kv4p3_MA24_PC.mod``. The five cell-type
    ports of this mechanism are byte-identical apart from their
    ``SUFFIX`` line and their ``celsius`` default, and the five
    BrainCell classes carry the same rate constants: every equation
    above is shared verbatim with :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2020_GrC`, :class:`Kv4p3_MA2025_BC` and
    :class:`Kv4p3_RI2021_SC`. What differs is the default ``temp``,
    the deposit each was imported from, and therefore the model paper
    cited below.

    ``Kalpha_a`` is stored here as a bare float and passed straight to
    ``_sigm``, whereas :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2020_GrC` and :class:`Kv4p3_RI2021_SC` attach
    ``u.mV`` to it and convert with ``.to_decimal(u.mV)`` at use. The
    arithmetic is identical -- both forms divide a millivolt
    difference by -23.3271 -- but the inconsistency is real, and on
    this documentation-only branch it is recorded rather than fixed.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 25.5 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 25.5)/10}` (about 1.64 at the default
    30 degrees Celsius). The ``.mod`` file instead multiplies
    the same ``Q10`` into ``alp_a``, ``bet_a``, ``alp_b`` and
    ``bet_b``, which divides its ``tau_a`` and ``tau_b`` by that
    factor. The two forms are algebraically identical, but it means
    :meth:`f_a_tau` and :meth:`f_b_tau` return q10-free time
    constants rather than the mechanism's ``tau_a`` and ``tau_b``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries
    ``Author: E.D'Angelo, T.Nieus, A. Fontana``. That credit line is
    copy-pasted verbatim across every cell-type port of this
    mechanism, and it names authors 1, 2 and 7 of an eight-author
    paper, so it is not treated as a citation here. The kinetics
    originate in the cerebellar granule cell model of D'Angelo et al.
    (2001) [1]_; the model paper [2]_ names the deposit this
    parameterisation was imported from, not the origin of the
    equations. The commented-out alternatives that remain in the
    ``.mod`` file -- ``linoid`` forms of each rate, and a
    "Bardoni Belluzzi" steady-state block -- are never evaluated by
    the mechanism and are not reproduced here.

    **Conductance default.** ``3.2 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** Three, all recorded against this symbol.
    First, the original mechanism's NMODL ``TABLE`` over
    ``[-100, 30] mV``, covering ``a_inf``, ``tau_a``, ``b_inf`` and
    ``tau_b``, is not reproduced: all four expressions are evaluated
    per call. NEURON clamped tabulated values to the boundary
    outside that window, so any BrainCell-versus-NEURON divergence
    below -100 mV or above 30 mV is expected rather than a port
    error. Second, the integration method was changed from
    ``derivimplicit`` to ``cnexp``; the two gate ODEs are
    independent, so the substitution is exact. Third, four
    parameters are carried at NEURON's compiled six-significant-
    figure precision rather than at the ``.mod`` source text's:
    ``Kalpha_a`` ``-23.32708`` becomes ``-23.3271``, ``Kbeta_a``
    ``19.47175`` becomes ``19.4718``, ``V0beta_a`` ``-18.27914``
    becomes ``-18.2791`` and ``V0alpha_b`` ``-111.33209`` becomes
    ``-111.332``. That rewrite reaches those four parameters only;
    ordinary in-formula literals keep their source values.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm**2),
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

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a)

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp((V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV))

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
class Kv4p3_RI2021_SC(OhmicHH):
    r"""A-type transient potassium current of the stellate cell model.

    Fast-inactivating A-type potassium current, imported from the
    cerebellar stellate cell model of Rizza et al. (2021) [2]_.

    Gating is an ``a`` gate of power 3 and a ``b`` gate of power 1,
    each with an explicit Boltzmann steady state and a time constant
    built from an alpha/beta pair:

    .. math::

        \begin{aligned}
        a_\infty &= \frac{1}{1 + \exp((V + 38) / (-17))} \\
        b_\infty &= \frac{1}{1 + \exp((V + 78.8) / 8.4)} \\
        \tau_a &= \frac{1}{\alpha_a + \beta_a}, \qquad
        \tau_b = \frac{1}{\alpha_b + \beta_b}
        \end{aligned}

    with, writing :math:`\sigma(x, y) = 1 / (e^{x/y} + 1)` for the
    module-level ``_sigm`` helper,

    .. math::

        \begin{aligned}
        \alpha_a &= 0.8147 \, \sigma(V + 9.17203,\ -23.3271) \\
        \beta_a &= 0.1655 \big/ e^{(V + 18.2791) / 19.4718} \\
        \alpha_b &= 0.0368 \, \sigma(V + 111.332,\ 12.8433) \\
        \beta_b &= 0.0345 \, \sigma(V + 49.9537,\ -8.90123)
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and the time constants are in milliseconds. This class applies
    no voltage shift, and the reversal potential comes from the
    potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``3.2 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.0032 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv4p3_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv4p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv4p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv4p3_MA2025_BC : Basket-cell port of the same mechanism.

    Notes
    -----
    Ported from ``SC/channel/Kv4p3_RI21_SC.mod``. The five cell-type
    ports of this mechanism are byte-identical apart from their
    ``SUFFIX`` line and their ``celsius`` default, and the five
    BrainCell classes carry the same rate constants: every equation
    above is shared verbatim with :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2020_GrC`, :class:`Kv4p3_MA2024_PC` and
    :class:`Kv4p3_MA2025_BC`. What differs is the default ``temp``,
    the deposit each was imported from, and therefore the model paper
    cited below.

    ``Kalpha_a`` is stored here as ``-23.3271 * u.mV`` and converted
    with ``.to_decimal(u.mV)`` at use, whereas
    :class:`Kv4p3_MA2024_PC` and :class:`Kv4p3_MA2025_BC` store the
    same number as a bare float and pass it straight to ``_sigm``. The
    arithmetic is identical -- both forms divide a millivolt
    difference by -23.3271 -- but the inconsistency is real, and on
    this documentation-only branch it is recorded rather than fixed.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 25.5 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 25.5)/10}` (about 1.64 at the default
    30 degrees Celsius). The ``.mod`` file instead multiplies
    the same ``Q10`` into ``alp_a``, ``bet_a``, ``alp_b`` and
    ``bet_b``, which divides its ``tau_a`` and ``tau_b`` by that
    factor. The two forms are algebraically identical, but it means
    :meth:`f_a_tau` and :meth:`f_b_tau` return q10-free time
    constants rather than the mechanism's ``tau_a`` and ``tau_b``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries
    ``Author: E.D'Angelo, T.Nieus, A. Fontana``. That credit line is
    copy-pasted verbatim across every cell-type port of this
    mechanism, and it names authors 1, 2 and 7 of an eight-author
    paper, so it is not treated as a citation here. The kinetics
    originate in the cerebellar granule cell model of D'Angelo et al.
    (2001) [1]_; the model paper [2]_ names the deposit this
    parameterisation was imported from, not the origin of the
    equations. The commented-out alternatives that remain in the
    ``.mod`` file -- ``linoid`` forms of each rate, and a
    "Bardoni Belluzzi" steady-state block -- are never evaluated by
    the mechanism and are not reproduced here.

    **Conductance default.** ``3.2 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** Three, all recorded against this symbol.
    First, the original mechanism's NMODL ``TABLE`` over
    ``[-100, 30] mV``, covering ``a_inf``, ``tau_a``, ``b_inf`` and
    ``tau_b``, is not reproduced: all four expressions are evaluated
    per call. NEURON clamped tabulated values to the boundary
    outside that window, so any BrainCell-versus-NEURON divergence
    below -100 mV or above 30 mV is expected rather than a port
    error. Second, the integration method was changed from
    ``derivimplicit`` to ``cnexp``; the two gate ODEs are
    independent, so the substitution is exact. Third, four
    parameters are carried at NEURON's compiled six-significant-
    figure precision rather than at the ``.mod`` source text's:
    ``Kalpha_a`` ``-23.32708`` becomes ``-23.3271``, ``Kbeta_a``
    ``19.47175`` becomes ``19.4718``, ``V0beta_a`` ``-18.27914``
    becomes ``-18.2791`` and ``V0alpha_b`` ``-111.33209`` becomes
    ``-111.332``. That rewrite reaches those four parameters only;
    ordinary in-formula literals keep their source values.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm**2),
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

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV))

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp((V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV))

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


@register_channel("KM_MA2020_GoC")
class KM_MA2020_GoC(OhmicHH):
    r"""M-type potassium current of the Golgi cell model.

    Slow, non-inactivating M-type potassium current imported from the
    cerebellar Golgi cell model of Masoli et al. (2020) [2]_. A single
    first-order ``n`` gate of power 1 drives an ohmic current:

    .. math::

        \begin{aligned}
        n_\infty &= \frac{1}{1 + \exp(-(V + 35) / 6)} \\
        \alpha_n &= 0.0033 \, \exp((V + 30) / 40) \\
        \beta_n &= 0.0033 \, \exp(-(V + 30) / 20) \\
        \tau_n &= \frac{1}{\alpha_n + \beta_n}
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and :math:`\tau_n` is in milliseconds. This class applies no
    voltage shift, and the reversal potential comes from the potassium
    ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.25 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.00025 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KM_MA2020_GrC : Granule-cell port of the same mechanism.
    KM_RI2021_SC : Stellate-cell port of the same mechanism.
    Kv1p1_MA2020_GoC : Low-threshold Kv1.1 current of the same Golgi
        cell model.

    Notes
    -----
    Ported from ``GoC/channel/KM_MA20_GoC.mod``. That file, the
    granule port ``GrC/channel/KM_MA20_GrC.mod`` and the stellate port
    ``SC/channel/KM_RI21_SC.mod`` are byte-identical apart from their
    ``SUFFIX`` line, and so are the three BrainCell classes: the rate
    constants above are shared verbatim with :class:`KM_MA2020_GrC`
    and :class:`KM_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    The mechanism does not use the steady state implied by its own
    rates. Its ``n_inf = a_n/(a_n + b_n)`` line is commented out in
    the ``.mod`` source and replaced by the explicit Boltzmann shown
    above, so :math:`n_\infty` and :math:`\tau_n` are independent
    expressions here.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`(n_\infty - n)/\tau_n` term by
    :math:`\phi = 3^{(T - 22)/10}` (about 2.41 at the default 30
    degrees Celsius). The ``.mod`` file instead multiplies ``Q10``
    into ``alp_n`` and ``bet_n``, which divides its ``tau_n`` by the
    same factor. The two forms are algebraically identical, but it
    means :meth:`f_n_tau` returns the q10-free time constant rather
    than the mechanism's ``tau_n``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries ``Author: A. Fontana`` and
    ``CoAuthor: T.Nieus``. That credit line is copy-pasted verbatim
    across every cell-type port of this mechanism and names people
    unrelated to the Golgi-cell key, so it is not treated as a
    citation here. The kinetics originate in the cerebellar granule
    cell model of D'Angelo et al. (2001) [1]_; the Golgi-cell paper
    [2]_ names the model this parameterisation was imported from, not
    the origin of the equations. Reference [2]_ is the Golgi-cell
    paper specifically -- the companion granule-cell paper of the same
    year belongs to :class:`KM_MA2020_GrC`, not to this class.

    **Conductance default.** ``0.25 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 30] mV``, covering ``n_inf`` and ``tau_n``, is not
    reproduced: both expressions are evaluated per call. NEURON
    clamped tabulated values to the boundary outside that window, so
    any BrainCell-versus-NEURON divergence below -100 mV or above
    30 mV is expected rather than a port error. The integration
    method was also changed from ``derivimplicit`` to ``cnexp``;
    with one independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm**2),
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

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp((V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV))

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp((V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV))

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV)))

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kv1p1_MA2020_GoC")
class Kv1p1_MA2020_GoC(HH):
    r"""Kv1.1 low-threshold potassium current of the Golgi cell model.

    Non-inactivating, low-threshold potassium current carried by Kv1.1
    subunits, imported from the cerebellar Golgi cell model of Masoli
    et al. (2020) [3]_. Gating is a single ``n`` gate of power 4 in
    alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.12889 \, \exp((V + 45) / 33.90877) \\
        \beta_n &= 0.12889 \, \exp(-(V + 45) / 12.42101)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -45 mV, and the :math:`n^4` conductance reaches half its
    maximum near -29.9 mV -- consistent with the
    ``Vhalf = -28.8 +- 2.3 mV`` that the ``.mod`` header quotes from
    the Zerr et al. (1998) recordings [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``, which
        is exactly the source mechanism's ``gbar = 0.004 S/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv1p1_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv1p1_MA2025_BC : Basket-cell port of the same mechanism.
    Kv1p1_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GoC/channel/Kv1p1_MA20_GoC.mod``. That file and the
    granule, Purkinje, basket and stellate ports are identical apart
    from their ``SUFFIX`` line and one line of indentation, and the
    five BrainCell classes are identical too: the rate constants and
    the gating-current constants above are shared verbatim with
    :class:`Kv1p1_MA2020_GrC`, :class:`Kv1p1_MA2024_PC`,
    :class:`Kv1p1_MA2025_BC` and :class:`Kv1p1_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 2.7978` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    divides its ``taun`` by the same factor. The two forms are
    algebraically identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau(V)` data from human Kv1.1
    expressed in Xenopus oocytes by Zerr et al. (1998) [1]_, and
    names the RIKEN implementation published by Akemann et al. (2009)
    [2]_ as its model reference. The Golgi-cell paper [3]_ names the
    model this parameterisation was imported from, not the origin of
    the equations; the companion granule-cell paper of the same year
    belongs to :class:`Kv1p1_MA2020_GrC`, not to this class.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic
           ataxia mutations in Kv1.1 alter potassium channel function
           by dominant negative effects or haploinsufficiency. The
           Journal of Neuroscience, 18(8), 2842-2848.
           doi:10.1523/JNEUROSCI.18-08-02842.1998
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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


@register_channel("Kv3p4_MA2020_GoC")
class Kv3p4_MA2020_GoC(OhmicHH):
    r"""Fast TEA-sensitive potassium current of the Golgi cell model.

    High-threshold, fast-activating and only partially inactivating
    potassium current, imported from the
    cerebellar Golgi cell model of Masoli et al. (2020) [2]_.

    Gating is an ``m`` gate of power 3 and an ``h`` gate of power 1,
    both written in steady-state/time-constant form and both
    evaluated at a junction-potential-corrected voltage
    :math:`V' = V + 11`:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V' + 24) / 15.4)} \\
        h_\infty &= 0.31
          + \frac{0.69}{1 + \exp((V' + 5.802) / 11.2)}
        \end{aligned}

    where :math:`V'` is in millivolts. Inactivation is only partial:
    :math:`h_\infty` decays to a floor of 0.31 rather than to zero,
    so about 31 per cent of the conductance survives a maintained
    depolarisation. Both time constants are piecewise, in
    milliseconds:

    .. math::

        \tau_m = 10^3 \times \begin{cases}
        3 \, (3.4225 \times 10^{-5}
          + 0.00498 \, e^{V' / 28.29}), & V' < -35 \\
        1.2851 \times 10^{-4} + \dfrac{1}{e^{(V' + 100.7)/12.9}
          + e^{(V' - 56)/(-23.1)}}, & V' \ge -35
        \end{cases}

    .. math::

        \tau_h = 10^3 \times \begin{cases}
        0.0012 + 0.0023 \, e^{-0.141 \, V'}, & V' > 0 \\
        1.2202 \times 10^{-5}
          + 0.012 \, e^{-((V' + 56.3)/49.6)^2}, & V' \le 0
        \end{cases}

    Each ladder is selected with ``u.math.where`` on the predicates
    ``V' < -35`` and ``V' > 0``, so the boundary value itself falls
    to the second line in both.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.004 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. See the temperature note below: this
        default is BrainCell's own, not a value the source mechanism
        states.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv3p4_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv3p4_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv3p4_MA2025_BC : Basket-cell port of the same mechanism.
    Kv3p4_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GoC/channel/Kv3p4_MA20_GoC.mod``. All five cell-type
    ports of this mechanism carry the same equations and the same
    constants, and so do the five BrainCell classes: everything above
    is shared verbatim with :class:`Kv3p4_MA2020_GrC`,
    :class:`Kv3p4_MA2024_PC`, :class:`Kv3p4_MA2025_BC` and
    :class:`Kv3p4_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.
    Reference [2]_ is the Golgi-cell paper specifically -- the
    companion granule-cell paper of the same year belongs to
    :class:`Kv3p4_MA2020_GrC`, not to this class.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 37 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 37)/10}`. The ``.mod`` file instead
    divides its ``mtau`` and ``htau`` by the same factor. The two
    forms are algebraically identical, but it means :meth:`f_m_tau`
    and :meth:`f_h_tau` return q10-free time constants rather than
    the mechanism's ``mtau`` and ``htau``.

    **The default temperature is BrainCell's, not the mechanism's.**
    Unlike most channels in this module, this ``.mod`` file never
    declares a ``celsius`` value; it reads NEURON's global instead.
    The ``temp`` default of 22 degrees Celsius is therefore a
    BrainCell choice, and because it sits 15 degrees below the
    gates' reference it makes the default temperature factor
    :math:`\phi \approx 0.192` -- gating roughly 5.2 times slower
    than at 37 degrees Celsius. Callers reproducing a published
    simulation should set ``temp`` explicitly.

    **Provenance, and a name the sources do not support.** These files
    are ModelDB accession 48332's ``kpkj.mod`` renamed. The header
    lines ": HH TEA-sensitive Purkinje potassium current" and
    ": Created 8/5/02 - nwg" are the deposit's own, "nwg" being
    Nathan W. Gouwens, second author of Khaliq et al. (2003) [1]_, and
    the parameters ``mivh = -24 mV``, ``mik = 15.4`` and
    ``hiy0 = 0.31`` reproduce that paper's Table 1 row for the current
    it calls **K fast**. The trailing ``: Suffix from kpkj to Kv3_4``
    line is a BrainCell-local addition and is not in the deposit. The
    model paper [2]_ names the deposit this parameterisation was
    imported from, not the origin of the equations.

    That rename line is the whole basis of the ``Kv3p4`` name, and
    it does not survive contact with the source. Khaliq et al. call
    this current K fast throughout and never name a Kv subunit; the
    strings "Kv3.4", "Kv3.3" and "Kv3.1" appear nowhere in the
    paper, and its only mention of Kv3 at all is one Discussion
    sentence observing that the positive activation range is typical
    of the Kv3 family. **The class name therefore asserts a subunit
    identity the cited literature does not establish.** Under this
    project's mismatch policy the name is documented rather than
    changed: read ``Kv3p4`` as a label for the TEA-sensitive fast K
    current of Khaliq et al. (2003), which those authors associate
    with the Kv3 family, and not as a claim about Kv3.4.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here. Do not
    assume it shares the deviations of its ``Kv4p3`` neighbour,
    which has three.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [2] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
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
            self.mty0 + 1.0 / (u.math.exp((V + self.mtvh1) / self.mtk1) + u.math.exp((V + self.mtvh2) / self.mtk2)),
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
            1.2202e-05 + 0.012 * u.math.exp(-(((V + 56.3) / 49.6) ** 2)),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA2020_GoC")
class Kv4p3_MA2020_GoC(OhmicHH):
    r"""A-type transient potassium current of the Golgi cell model.

    Fast-inactivating A-type potassium current, imported from the
    cerebellar Golgi cell model of Masoli et al. (2020) [2]_.

    Gating is an ``a`` gate of power 3 and a ``b`` gate of power 1,
    each with an explicit Boltzmann steady state and a time constant
    built from an alpha/beta pair:

    .. math::

        \begin{aligned}
        a_\infty &= \frac{1}{1 + \exp((V + 38) / (-17))} \\
        b_\infty &= \frac{1}{1 + \exp((V + 78.8) / 8.4)} \\
        \tau_a &= \frac{1}{\alpha_a + \beta_a}, \qquad
        \tau_b = \frac{1}{\alpha_b + \beta_b}
        \end{aligned}

    with, writing :math:`\sigma(x, y) = 1 / (e^{x/y} + 1)` for the
    module-level ``_sigm`` helper,

    .. math::

        \begin{aligned}
        \alpha_a &= 0.8147 \, \sigma(V + 9.17203,\ -23.3271) \\
        \beta_a &= 0.1655 \big/ e^{(V + 18.2791) / 19.4718} \\
        \alpha_b &= 0.0368 \, \sigma(V + 111.332,\ 12.8433) \\
        \beta_b &= 0.0345 \, \sigma(V + 49.9537,\ -8.90123)
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and the time constants are in milliseconds. This class applies
    no voltage shift, and the reversal potential comes from the
    potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``3.2 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.0032 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. This matches the ``celsius = 22 (degC)``
        written in the source mechanism's ``PARAMETER`` block -- the
        one line that distinguishes it from the other four ports,
        which all write 30.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv4p3_MA2020_GrC : Granule-cell port of the same mechanism.
    Kv4p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv4p3_MA2025_BC : Basket-cell port of the same mechanism.
    Kv4p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GoC/channel/Kv4p3_MA20_GoC.mod``. The five cell-type
    ports of this mechanism are byte-identical apart from their
    ``SUFFIX`` line and their ``celsius`` default, and the five
    BrainCell classes carry the same rate constants: every equation
    above is shared verbatim with :class:`Kv4p3_MA2020_GrC`,
    :class:`Kv4p3_MA2024_PC`, :class:`Kv4p3_MA2025_BC` and
    :class:`Kv4p3_RI2021_SC`. What differs is the default ``temp``,
    the deposit each was imported from, and therefore the model paper
    cited below. Reference [2]_ is the Golgi-cell paper specifically
    -- the companion granule-cell paper of the same year belongs to
    :class:`Kv4p3_MA2020_GrC`, not to this class.

    ``Kalpha_a`` is stored here as ``-23.3271 * u.mV`` and converted
    with ``.to_decimal(u.mV)`` at use, whereas
    :class:`Kv4p3_MA2024_PC` and :class:`Kv4p3_MA2025_BC` store the
    same number as a bare float and pass it straight to ``_sigm``. The
    arithmetic is identical -- both forms divide a millivolt
    difference by -23.3271 -- but the inconsistency is real, and on
    this documentation-only branch it is recorded rather than fixed.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 25.5 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 25.5)/10}` (about 0.681 at the default
    22 degrees Celsius). The ``.mod`` file instead multiplies
    the same ``Q10`` into ``alp_a``, ``bet_a``, ``alp_b`` and
    ``bet_b``, which divides its ``tau_a`` and ``tau_b`` by that
    factor. The two forms are algebraically identical, but it means
    :meth:`f_a_tau` and :meth:`f_b_tau` return q10-free time
    constants rather than the mechanism's ``tau_a`` and ``tau_b``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries
    ``Author: E.D'Angelo, T.Nieus, A. Fontana``. That credit line is
    copy-pasted verbatim across every cell-type port of this
    mechanism, and it names authors 1, 2 and 7 of an eight-author
    paper, so it is not treated as a citation here. The kinetics
    originate in the cerebellar granule cell model of D'Angelo et al.
    (2001) [1]_; the model paper [2]_ names the deposit this
    parameterisation was imported from, not the origin of the
    equations. The commented-out alternatives that remain in the
    ``.mod`` file -- ``linoid`` forms of each rate, and a
    "Bardoni Belluzzi" steady-state block -- are never evaluated by
    the mechanism and are not reproduced here.

    **Conductance default.** ``3.2 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** Three, all recorded against this symbol.
    First, the original mechanism's NMODL ``TABLE`` over
    ``[-100, 30] mV``, covering ``a_inf``, ``tau_a``, ``b_inf`` and
    ``tau_b``, is not reproduced: all four expressions are evaluated
    per call. NEURON clamped tabulated values to the boundary
    outside that window, so any BrainCell-versus-NEURON divergence
    below -100 mV or above 30 mV is expected rather than a port
    error. Second, the integration method was changed from
    ``derivimplicit`` to ``cnexp``; the two gate ODEs are
    independent, so the substitution is exact. Third, four
    parameters are carried at NEURON's compiled six-significant-
    figure precision rather than at the ``.mod`` source text's:
    ``Kalpha_a`` ``-23.32708`` becomes ``-23.3271``, ``Kbeta_a``
    ``19.47175`` becomes ``19.4718``, ``V0beta_a`` ``-18.27914``
    becomes ``-18.2791`` and ``V0alpha_b`` ``-111.33209`` becomes
    ``-111.332``. That rewrite reaches those four parameters only;
    ordinary in-formula literals keep their source values.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm**2),
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

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV))

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp((V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV))

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


@register_channel("KM_MA2020_GrC")
class KM_MA2020_GrC(OhmicHH):
    r"""M-type potassium current of the granule cell model.

    Slow, non-inactivating M-type potassium current imported from the
    cerebellar granule cell model of Masoli et al. (2020) [2]_. A
    single first-order ``n`` gate of power 1 drives an ohmic current:

    .. math::

        \begin{aligned}
        n_\infty &= \frac{1}{1 + \exp(-(V + 35) / 6)} \\
        \alpha_n &= 0.0033 \, \exp((V + 30) / 40) \\
        \beta_n &= 0.0033 \, \exp(-(V + 30) / 20) \\
        \tau_n &= \frac{1}{\alpha_n + \beta_n}
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and :math:`\tau_n` is in milliseconds. This class applies no
    voltage shift, and the reversal potential comes from the potassium
    ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.25 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.00025 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    KM_MA2020_GoC : Golgi-cell port of the same mechanism.
    KM_RI2021_SC : Stellate-cell port of the same mechanism.
    Kir2p3_MA2020_GrC : Inward rectifier of the same granule cell
        model, sharing the same origin paper.

    Notes
    -----
    Ported from ``GrC/channel/KM_MA20_GrC.mod``. That file, the Golgi
    port ``GoC/channel/KM_MA20_GoC.mod`` and the stellate port
    ``SC/channel/KM_RI21_SC.mod`` are byte-identical apart from their
    ``SUFFIX`` line, and so are the three BrainCell classes: the rate
    constants above are shared verbatim with :class:`KM_MA2020_GoC`
    and :class:`KM_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.

    The mechanism does not use the steady state implied by its own
    rates. Its ``n_inf = a_n/(a_n + b_n)`` line is commented out in
    the ``.mod`` source and replaced by the explicit Boltzmann shown
    above, so :math:`n_\infty` and :math:`\tau_n` are independent
    expressions here.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`(n_\infty - n)/\tau_n` term by
    :math:`\phi = 3^{(T - 22)/10}` (about 2.41 at the default 30
    degrees Celsius). The ``.mod`` file instead multiplies ``Q10``
    into ``alp_n`` and ``bet_n``, which divides its ``tau_n`` by the
    same factor. The two forms are algebraically identical, but it
    means :meth:`f_n_tau` returns the q10-free time constant rather
    than the mechanism's ``tau_n``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries ``Author: A. Fontana`` and
    ``CoAuthor: T.Nieus``. That credit line is copy-pasted verbatim
    across every cell-type port of this mechanism and names people
    unrelated to the granule-cell key, so it is not treated as a
    citation here. The kinetics originate in the cerebellar granule
    cell model of D'Angelo et al. (2001) [1]_; the granule-cell paper
    [2]_ names the model this parameterisation was imported from, not
    the origin of the equations. Reference [2]_ is the granule-cell
    paper specifically -- the companion Golgi-cell paper of the same
    year belongs to :class:`KM_MA2020_GoC`, not to this class.

    **Conductance default.** ``0.25 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 30] mV``, covering ``n_inf`` and ``tau_n``, is not
    reproduced: both expressions are evaluated per call. NEURON
    clamped tabulated values to the boundary outside that window, so
    any BrainCell-versus-NEURON divergence below -100 mV or above
    30 mV is expected rather than a port error. The integration
    method was also changed from ``derivimplicit`` to ``cnexp``;
    with one independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", q10=3.0, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.25 * (u.mS / u.cm**2),
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

    def _n_alpha(self, V):
        V = V.to_decimal(u.mV)
        return self.Aalpha_n * u.math.exp((V - self.V0alpha_n.to_decimal(u.mV)) / self.Kalpha_n.to_decimal(u.mV))

    def _n_beta(self, V):
        V = V.to_decimal(u.mV)
        return self.Abeta_n * u.math.exp((V - self.V0beta_n.to_decimal(u.mV)) / self.Kbeta_n.to_decimal(u.mV))

    def f_n_inf(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V - self.V0_ninf.to_decimal(u.mV)) / self.B_ninf.to_decimal(u.mV)))

    def f_n_tau(self, V, K: IonInfo):
        return 1.0 / (self._n_alpha(V) + self._n_beta(V))


@register_channel("Kir2p3_MA2020_GrC")
class Kir2p3_MA2020_GrC(OhmicHH):
    r"""Kir2.3 inward-rectifier current of the granule cell model.

    Hyperpolarization-activated inward-rectifier potassium current
    imported from the cerebellar granule cell model of Masoli et al.
    (2020) [2]_. A single first-order ``d`` gate of power 1 drives an
    ohmic current, with the gate written in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_d &= 0.13289 \, \exp(-(V + 83.94) / 24.3902) \\
        \beta_d &= 0.16994 \, \exp((V + 83.94) / 35.714)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. The template forms
    :math:`d_\infty = \alpha_d / (\alpha_d + \beta_d)` and
    :math:`\tau_d = 1 / (\alpha_d + \beta_d)` from these; half
    activation falls near -87.5 mV, and :math:`d_\infty` rises towards
    1 as the membrane hyperpolarizes. This class applies no voltage
    shift, and the reversal potential comes from the potassium ion
    object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.9 mS/cm2``, which
        is exactly the source mechanism's ``gkbar = 0.0009 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kir2p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kir2p3_MA2025_BC : Basket-cell port of the same mechanism.
    Kir2p3_RI2021_SC : Stellate-cell port of the same mechanism.
    KM_MA2020_GrC : M-type current of the same granule cell model,
        sharing the same origin paper.

    Notes
    -----
    Ported from ``GrC/channel/Kir2p3_MA20_GrC.mod``. That file and the
    Purkinje, basket and stellate ports are byte-identical apart from
    their ``SUFFIX`` line and, in the Purkinje port only, the
    mechanism-local ``celsius`` default. The four BrainCell classes
    are likewise identical, so the rate constants above are shared
    verbatim with :class:`Kir2p3_MA2024_PC`,
    :class:`Kir2p3_MA2025_BC` and :class:`Kir2p3_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **The rectification lives in the gate, not in the current.** The
    current expression is the plain ohmic
    ``g_max * d * (E_K - V)`` supplied by :class:`OhmicHH`; there is
    no Mg2+ or polyamine block term anywhere in the mechanism. The
    inward-rectifier behaviour comes entirely from :math:`d_\infty`
    increasing as the membrane hyperpolarizes.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 3.0`` at a reference of 20 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the whole
    :math:`\alpha_d (1 - d) - \beta_d d` term by
    :math:`\phi = 3^{(T - 20)/10}`, which is exactly 3 at the default
    30 degrees Celsius. The ``.mod`` file instead multiplies ``Q10``
    into ``alp_d`` and ``bet_d``. The two forms are algebraically
    identical, but it means :meth:`f_d_alpha` and :meth:`f_d_beta`
    return the q10-free rates rather than the mechanism's
    ``alpha_d``/``beta_d``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` ``COMMENT`` carries a reference string that is the
    published title of D'Angelo et al. (2001) truncated mid-subtitle,
    plus the porting note "Suffix from Ubc_Kir to Kir2_3". Neither is
    treated as a citation here. The kinetics originate in the
    cerebellar granule cell model of D'Angelo et al. (2001) [1]_; the
    granule-cell paper [2]_ names the model this parameterisation was
    imported from, not the origin of the equations. Reference [2]_ is
    the granule-cell paper specifically -- the companion Golgi-cell
    paper of the same year covers a different deposit and is not
    cited here.

    **Conductance default.** ``0.9 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** The original mechanism's NMODL ``TABLE``
    over ``[-100, 100] mV``, covering ``d_inf`` and ``tau_d``, is not
    reproduced: both expressions are evaluated per call. NEURON used
    the boundary value outside that window, so any
    BrainCell-versus-NEURON divergence below -100 mV or above 100 mV
    is expected rather than a port error. The integration method was
    also changed from ``derivimplicit`` to ``cnexp``; with one
    independent gate ODE that substitution is exact.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("d", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.9 * (u.mS / u.cm**2),
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

    def f_d_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_d * u.math.exp((V - self.V0alpha_d.to_decimal(u.mV)) / self.Kalpha_d.to_decimal(u.mV))

    def f_d_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_d * u.math.exp((V - self.V0beta_d.to_decimal(u.mV)) / self.Kbeta_d.to_decimal(u.mV))


@register_channel("Kv1p1_MA2020_GrC")
class Kv1p1_MA2020_GrC(HH):
    r"""Kv1.1 low-threshold potassium current of the granule model.

    Non-inactivating, low-threshold potassium current carried by Kv1.1
    subunits, imported from the cerebellar granule cell model of
    Masoli et al. (2020) [3]_. Gating is a single ``n`` gate of power
    4 in alpha/beta form:

    .. math::

        \begin{aligned}
        \alpha_n &= 0.12889 \, \exp((V + 45) / 33.90877) \\
        \beta_n &= 0.12889 \, \exp(-(V + 45) / 12.42101)
        \end{aligned}

    where :math:`V` is in millivolts and the rates are per
    millisecond. Because the two prefactors are equal,
    :math:`n_\infty = \alpha_n / (\alpha_n + \beta_n)` is exactly 0.5
    at -45 mV, and the :math:`n^4` conductance reaches half its
    maximum near -29.9 mV -- consistent with the
    ``Vhalf = -28.8 +- 2.3 mV`` that the ``.mod`` header quotes from
    the Zerr et al. (1998) recordings [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``, which
        is exactly the source mechanism's ``gbar = 0.004 S/cm2``.
    temp : array-like, optional
        Absolute temperature driving the gate's q10 factor, default
        22 degrees Celsius. This equals the gate's reference
        temperature, so the default temperature factor is exactly 1.
    gateCurrent : array-like or callable, optional
        Gating-current switch, default ``0.0`` (dimensionless, off).
        Any non-zero value enables the gating-current term described
        below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv1p1_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv1p1_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv1p1_MA2025_BC : Basket-cell port of the same mechanism.
    Kv1p1_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GrC/channel/Kv1p1_MA20_GrC.mod``. That file and the
    Golgi, Purkinje, basket and stellate ports are identical apart
    from their ``SUFFIX`` line and one line of indentation, and the
    five BrainCell classes are identical too: the rate constants and
    the gating-current constants above are shared verbatim with
    :class:`Kv1p1_MA2020_GoC`, :class:`Kv1p1_MA2024_PC`,
    :class:`Kv1p1_MA2025_BC` and :class:`Kv1p1_RI2021_SC`. What
    differs is only the deposit each was imported from, and therefore
    the model paper cited below.

    **Why this class overrides** ``current``. The mechanism emits an
    optional Kv gating current alongside the ionic current, so
    :class:`OhmicHH` is not enough. :meth:`current` returns

    .. math::

        g_{\max} \, n^4 \, (E_K - V) - I_{\text{gate}}

    with

    .. math::

        I_{\text{gate}} = n_c \cdot 10^{6} \cdot e_0 \cdot 4 \, z_n
        \cdot \frac{\mathrm{d}n}{\mathrm{d}t}, \qquad
        n_c = 10^{12} \, \frac{g_{\max}}{g_{\text{unit}}}

    where :math:`g_{\text{unit}} = 16` pS is the unitary channel
    conductance, :math:`z_n = 2.7978` the n-gate valence,
    :math:`e_0 = 1.60217646 \times 10^{-19}` C the elementary charge
    and :math:`\mathrm{d}n/\mathrm{d}t` the same
    :math:`\phi(\alpha_n (1 - n) - \beta_n n)` the gate integrates.
    NEURON emits this term as a separate ``NONSPECIFIC_CURRENT``;
    BrainCell folds it into the single ``current()`` return, and the
    subtraction reflects the package's inward-positive sign
    convention against NMODL's outward-positive one. The term is
    selected with ``u.math.where`` rather than a Python branch, so
    ``gateCurrent`` may be an array and the choice stays traceable.

    **Where the q10 factor is applied.** The gate declares
    ``q10 = 2.7`` at a reference of 22 degrees Celsius, so
    :meth:`HH.compute_derivative` scales the rate balance by
    :math:`\phi = 2.7^{(T - 22)/10}`. The ``.mod`` file instead
    divides its ``taun`` by the same factor. The two forms are
    algebraically identical.

    **Provenance.** The ``.mod`` header states that the six rate
    parameters were obtained by least-squares fits to
    :math:`G/G_{\max}(V)` and :math:`\tau(V)` data from human Kv1.1
    expressed in Xenopus oocytes by Zerr et al. (1998) [1]_, and
    names the RIKEN implementation published by Akemann et al. (2009)
    [2]_ as its model reference. The granule-cell paper [3]_ names
    the model this parameterisation was imported from, not the origin
    of the equations; the companion Golgi-cell paper of the same year
    belongs to :class:`Kv1p1_MA2020_GoC`, not to this class.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here.

    References
    ----------
    .. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic
           ataxia mutations in Kv1.1 alter potassium channel function
           by dominant negative effects or haploinsufficiency. The
           Journal of Neuroscience, 18(8), 2842-2848.
           doi:10.1523/JNEUROSCI.18-08-02842.1998
    .. [2] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
           Effect of voltage sensitive fluorescent proteins on
           neuronal excitability. Biophysical Journal, 96(10),
           3959-3976.
           doi:10.1016/j.bpj.2009.02.046
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.7, temp_ref=u.celsius2kelvin(22.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
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


@register_channel("Kv2p2_0010_MA2020_GrC")
class Kv2p2_0010_MA2020_GrC(OhmicHH):
    r"""Kv2.2 delayed-rectifier current of the granule cell model.

    Slowly inactivating delayed-rectifier potassium current attributed
    to Kv2.2, imported from the cerebellar granule cell model of
    Masoli et al. (2020) [3]_. Two first-order gates of power 1, an
    activation ``m`` and an inactivation ``h``, drive an ohmic
    current:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V - 5) / 12)} \\
        \tau_m &= \frac{130}{1 + \exp(-(V + 46.56) / 44.14)} \\
        h_\infty &= \frac{1}{1 + \exp((V + 16.3) / 4.8)} \\
        \tau_h &= \frac{10000}{1 + \exp(-(V + 46.56) / 44.14)}
        \end{aligned}

    where :math:`V` is in millivolts and both time constants are in
    milliseconds. The two time constants share a denominator and
    differ only by a factor of about 77, so inactivation is very slow
    -- :math:`\tau_h` approaches 10 seconds at depolarized potentials.
    This class applies no voltage shift, and the reversal potential
    comes from the potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.01 mS/cm2``,
        which is exactly the source mechanism's
        ``gKv2_2bar = 0.00001 S/cm2``.
    BBiD : array-like or callable, optional
        Channelpedia ion-channel identifier, default ``10.0``
        (dimensionless). This is inert metadata, not a kinetic
        parameter; see the note below.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kir2p3_MA2020_GrC : Inward rectifier of the same granule cell
        model.
    KM_MA2020_GrC : M-type current of the same granule cell model.

    Notes
    -----
    Ported from ``GrC/channel/Kv2p2_0010_MA20_GrC.mod``.

    **No temperature dependence.** Neither gate declares ``phi`` or
    ``q10``, so :meth:`HH.gate_phi` resolves to ``1.0``, and the class
    takes no ``temp`` parameter at all. This is faithful: the ``.mod``
    file contains no ``celsius`` reference and no Q10 term.

    **``BBiD`` is metadata, not a parameter of the model.** In the
    ``.mod`` file ``BBiD = 10`` is declared ``RANGE`` but never
    appears in any equation, and BrainCell likewise stores it and
    never reads it. It is the Channelpedia identifier for Kv2.2 (gene
    *KCNB2*), and it is also the ``0010`` embedded in the class name.

    **How this mechanism was produced.** The ``.mod`` file is machine
    generated, not hand written: its version-control keywords name the
    EPFL Blue Brain ``xmlTomod/CreateMOD.c`` generator behind the
    Channelpedia database of Ranjan et al. (2011) [2]_. That record is
    cited here for the toolchain and the channel identity only -- it
    is not the source of the rate constants above. Those come from the
    delayed-rectifier component identified in gastrointestinal smooth
    muscle by Schmalz et al. (1998) [1]_, which the ``.mod`` header's
    own ``:Reference :`` line names. The granule-cell paper [3]_ names
    the model this parameterisation was imported from; the companion
    Golgi-cell paper of the same year covers a different deposit and
    is not cited here.

    **Conductance default.** ``0.01 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in any of the cited papers.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream. Its
    rate block is wrapped in ``UNITSOFF``/``UNITSON``, which BrainCell
    reproduces by evaluating the expressions on the dimensionless
    millivolt value of ``V``.

    References
    ----------
    .. [1] Schmalz, F., Kinsella, J., Koh, S. D., Vogalis, F.,
           Schneider, A., Flynn, E. R. M., Kenyon, J. L., & Horowitz,
           B. (1998). Molecular identification of a component of
           delayed rectifier current in gastrointestinal smooth
           muscles. American Journal of Physiology-Gastrointestinal
           and Liver Physiology, 274(5), G901-G911.
           doi:10.1152/ajpgi.1998.274.5.G901
    .. [2] Ranjan, R., Khazen, G., Gambazzi, L., Ramaswamy, S., Hill,
           S. L., Schurmann, F., & Markram, H. (2011). Channelpedia:
           an integrative and interactive database for ion channels.
           Frontiers in Neuroinformatics, 5, 36.
           doi:10.3389/fninf.2011.00036
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m"),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        BBiD: Union[brainstate.typing.ArrayLike, Callable] = 10.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.BBiD = braintools.init.param(BBiD, self.varshape, allow_none=False)

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
class Kv3p4_MA2020_GrC(OhmicHH):
    r"""Fast TEA-sensitive potassium current of the granule cell model.

    High-threshold, fast-activating and only partially inactivating
    potassium current, imported from the
    cerebellar granule cell model of Masoli et al. (2020) [2]_.

    Gating is an ``m`` gate of power 3 and an ``h`` gate of power 1,
    both written in steady-state/time-constant form and both
    evaluated at a junction-potential-corrected voltage
    :math:`V' = V + 11`:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp(-(V' + 24) / 15.4)} \\
        h_\infty &= 0.31
          + \frac{0.69}{1 + \exp((V' + 5.802) / 11.2)}
        \end{aligned}

    where :math:`V'` is in millivolts. Inactivation is only partial:
    :math:`h_\infty` decays to a floor of 0.31 rather than to zero,
    so about 31 per cent of the conductance survives a maintained
    depolarisation. Both time constants are piecewise, in
    milliseconds:

    .. math::

        \tau_m = 10^3 \times \begin{cases}
        3 \, (3.4225 \times 10^{-5}
          + 0.00498 \, e^{V' / 28.29}), & V' < -35 \\
        1.2851 \times 10^{-4} + \dfrac{1}{e^{(V' + 100.7)/12.9}
          + e^{(V' - 56)/(-23.1)}}, & V' \ge -35
        \end{cases}

    .. math::

        \tau_h = 10^3 \times \begin{cases}
        0.0012 + 0.0023 \, e^{-0.141 \, V'}, & V' > 0 \\
        1.2202 \times 10^{-5}
          + 0.012 \, e^{-((V' + 56.3)/49.6)^2}, & V' \le 0
        \end{cases}

    Each ladder is selected with ``u.math.where`` on the predicates
    ``V' < -35`` and ``V' > 0``, so the boundary value itself falls
    to the second line in both.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``4.0 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.004 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        22 degrees Celsius. See the temperature note below: this
        default is BrainCell's own, not a value the source mechanism
        states.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv3p4_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv3p4_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv3p4_MA2025_BC : Basket-cell port of the same mechanism.
    Kv3p4_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GrC/channel/Kv3p4_MA20_GrC.mod``. All five cell-type
    ports of this mechanism carry the same equations and the same
    constants, and so do the five BrainCell classes: everything above
    is shared verbatim with :class:`Kv3p4_MA2020_GoC`,
    :class:`Kv3p4_MA2024_PC`, :class:`Kv3p4_MA2025_BC` and
    :class:`Kv3p4_RI2021_SC`. What differs is only the deposit each
    was imported from, and therefore the model paper cited below.
    Reference [2]_ is the granule-cell paper specifically -- the
    companion Golgi-cell paper of the same year belongs to
    :class:`Kv3p4_MA2020_GoC`, not to this class.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 37 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 37)/10}`. The ``.mod`` file instead
    divides its ``mtau`` and ``htau`` by the same factor. The two
    forms are algebraically identical, but it means :meth:`f_m_tau`
    and :meth:`f_h_tau` return q10-free time constants rather than
    the mechanism's ``mtau`` and ``htau``.

    **The default temperature is BrainCell's, not the mechanism's.**
    Unlike most channels in this module, this ``.mod`` file never
    declares a ``celsius`` value; it reads NEURON's global instead.
    The ``temp`` default of 22 degrees Celsius is therefore a
    BrainCell choice, and because it sits 15 degrees below the
    gates' reference it makes the default temperature factor
    :math:`\phi \approx 0.192` -- gating roughly 5.2 times slower
    than at 37 degrees Celsius. Callers reproducing a published
    simulation should set ``temp`` explicitly.

    **Provenance, and a name the sources do not support.** These files
    are ModelDB accession 48332's ``kpkj.mod`` renamed. The header
    lines ": HH TEA-sensitive Purkinje potassium current" and
    ": Created 8/5/02 - nwg" are the deposit's own, "nwg" being
    Nathan W. Gouwens, second author of Khaliq et al. (2003) [1]_, and
    the parameters ``mivh = -24 mV``, ``mik = 15.4`` and
    ``hiy0 = 0.31`` reproduce that paper's Table 1 row for the current
    it calls **K fast**. The trailing ``: Suffix from kpkj to Kv3_4``
    line is a BrainCell-local addition and is not in the deposit. The
    model paper [2]_ names the deposit this parameterisation was
    imported from, not the origin of the equations.

    That rename line is the whole basis of the ``Kv3p4`` name, and
    it does not survive contact with the source. Khaliq et al. call
    this current K fast throughout and never name a Kv subunit; the
    strings "Kv3.4", "Kv3.3" and "Kv3.1" appear nowhere in the
    paper, and its only mention of Kv3 at all is one Discussion
    sentence observing that the positive activation range is typical
    of the Kv3 family. **The class name therefore asserts a subunit
    identity the cited literature does not establish.** Under this
    project's mismatch policy the name is documented rather than
    changed: read ``Kv3p4`` as a label for the TEA-sensitive fast K
    current of Khaliq et al. (2003), which those authors associate
    with the Kv3 family, and not as a claim about Kv3.4.

    **Conductance default.** ``4.0 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations: none.** This mechanism carries no NMODL
    ``TABLE`` and was already integrated with ``cnexp`` upstream, so
    neither the table-removal nor the ``derivimplicit`` substitution
    recorded for other channels in this model applies here. Do not
    assume it shares the deviations of its ``Kv4p3`` neighbour,
    which has three.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
        Gate("h", q10=3.0, temp_ref=u.celsius2kelvin(37.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 4.0 * (u.mS / u.cm**2),
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
            self.mty0 + 1.0 / (u.math.exp((V + self.mtvh1) / self.mtk1) + u.math.exp((V + self.mtvh2) / self.mtk2)),
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
            1.2202e-05 + 0.012 * u.math.exp(-(((V + 56.3) / 49.6) ** 2)),
        )
        return 1000.0 * htau_func


@register_channel("Kv4p3_MA2020_GrC")
class Kv4p3_MA2020_GrC(OhmicHH):
    r"""A-type transient potassium current of the granule cell model.

    Fast-inactivating A-type potassium current, imported from the
    cerebellar granule cell model of Masoli et al. (2020) [2]_.

    Gating is an ``a`` gate of power 3 and a ``b`` gate of power 1,
    each with an explicit Boltzmann steady state and a time constant
    built from an alpha/beta pair:

    .. math::

        \begin{aligned}
        a_\infty &= \frac{1}{1 + \exp((V + 38) / (-17))} \\
        b_\infty &= \frac{1}{1 + \exp((V + 78.8) / 8.4)} \\
        \tau_a &= \frac{1}{\alpha_a + \beta_a}, \qquad
        \tau_b = \frac{1}{\alpha_b + \beta_b}
        \end{aligned}

    with, writing :math:`\sigma(x, y) = 1 / (e^{x/y} + 1)` for the
    module-level ``_sigm`` helper,

    .. math::

        \begin{aligned}
        \alpha_a &= 0.8147 \, \sigma(V + 9.17203,\ -23.3271) \\
        \beta_a &= 0.1655 \big/ e^{(V + 18.2791) / 19.4718} \\
        \alpha_b &= 0.0368 \, \sigma(V + 111.332,\ 12.8433) \\
        \beta_b &= 0.0345 \, \sigma(V + 49.9537,\ -8.90123)
        \end{aligned}

    where :math:`V` is in millivolts, the rates are per millisecond
    and the time constants are in milliseconds. This class applies
    no voltage shift, and the reversal potential comes from the
    potassium ion object rather than from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``3.2 mS/cm2``,
        which is exactly the source mechanism's
        ``gkbar = 0.0032 mho/cm2``.
    temp : array-like, optional
        Absolute temperature driving both gates' q10 factor, default
        30 degrees Celsius. This matches the ``celsius = 30 (degC)``
        written in the source mechanism's ``PARAMETER`` block.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kv4p3_MA2020_GoC : Golgi-cell port of the same mechanism.
    Kv4p3_MA2024_PC : Purkinje-cell port of the same mechanism.
    Kv4p3_MA2025_BC : Basket-cell port of the same mechanism.
    Kv4p3_RI2021_SC : Stellate-cell port of the same mechanism.

    Notes
    -----
    Ported from ``GrC/channel/Kv4p3_MA20_GrC.mod``. The five cell-type
    ports of this mechanism are byte-identical apart from their
    ``SUFFIX`` line and their ``celsius`` default, and the five
    BrainCell classes carry the same rate constants: every equation
    above is shared verbatim with :class:`Kv4p3_MA2020_GoC`,
    :class:`Kv4p3_MA2024_PC`, :class:`Kv4p3_MA2025_BC` and
    :class:`Kv4p3_RI2021_SC`. What differs is the default ``temp``,
    the deposit each was imported from, and therefore the model paper
    cited below. Reference [2]_ is the granule-cell paper specifically
    -- the companion Golgi-cell paper of the same year belongs to
    :class:`Kv4p3_MA2020_GoC`, not to this class.

    ``Kalpha_a`` is stored here as ``-23.3271 * u.mV`` and converted
    with ``.to_decimal(u.mV)`` at use, whereas
    :class:`Kv4p3_MA2024_PC` and :class:`Kv4p3_MA2025_BC` store the
    same number as a bare float and pass it straight to ``_sigm``. The
    arithmetic is identical -- both forms divide a millivolt
    difference by -23.3271 -- but the inconsistency is real, and on
    this documentation-only branch it is recorded rather than fixed.

    **Where the q10 factor is applied.** Both gates declare
    ``q10 = 3.0`` at a reference of 25.5 degrees Celsius, so
    :meth:`HH.compute_derivative` scales each
    :math:`(x_\infty - x)/\tau_x` term by
    :math:`\phi = 3^{(T - 25.5)/10}` (about 1.64 at the default
    30 degrees Celsius). The ``.mod`` file instead multiplies
    the same ``Q10`` into ``alp_a``, ``bet_a``, ``alp_b`` and
    ``bet_b``, which divides its ``tau_a`` and ``tau_b`` by that
    factor. The two forms are algebraically identical, but it means
    :meth:`f_a_tau` and :meth:`f_b_tau` return q10-free time
    constants rather than the mechanism's ``tau_a`` and ``tau_b``.

    **Provenance, and what the header does not establish.** The
    ``.mod`` header carries
    ``Author: E.D'Angelo, T.Nieus, A. Fontana``. That credit line is
    copy-pasted verbatim across every cell-type port of this
    mechanism, and it names authors 1, 2 and 7 of an eight-author
    paper, so it is not treated as a citation here. The kinetics
    originate in the cerebellar granule cell model of D'Angelo et al.
    (2001) [1]_; the model paper [2]_ names the deposit this
    parameterisation was imported from, not the origin of the
    equations. The commented-out alternatives that remain in the
    ``.mod`` file -- ``linoid`` forms of each rate, and a
    "Bardoni Belluzzi" steady-state block -- are never evaluated by
    the mechanism and are not reproduced here.

    **Conductance default.** ``3.2 mS/cm2`` is the deposit's tuned
    value, carried across from the ``.mod`` file. It is not a value
    printed in either cited paper.

    **Import deviations.** Three, all recorded against this symbol.
    First, the original mechanism's NMODL ``TABLE`` over
    ``[-100, 30] mV``, covering ``a_inf``, ``tau_a``, ``b_inf`` and
    ``tau_b``, is not reproduced: all four expressions are evaluated
    per call. NEURON clamped tabulated values to the boundary
    outside that window, so any BrainCell-versus-NEURON divergence
    below -100 mV or above 30 mV is expected rather than a port
    error. Second, the integration method was changed from
    ``derivimplicit`` to ``cnexp``; the two gate ODEs are
    independent, so the substitution is exact. Third, four
    parameters are carried at NEURON's compiled six-significant-
    figure precision rather than at the ``.mod`` source text's:
    ``Kalpha_a`` ``-23.32708`` becomes ``-23.3271``, ``Kbeta_a``
    ``19.47175`` becomes ``19.4718``, ``V0beta_a`` ``-18.27914``
    becomes ``-18.2791`` and ``V0alpha_b`` ``-111.33209`` becomes
    ``-111.332``. That rewrite reaches those four parameters only;
    ordinary in-formula literals keep their source values.

    References
    ----------
    .. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
           Taglietti, V., Fontana, A., & Naldi, G. (2001).
           Theta-frequency bursting and resonance in cerebellar
           granule cells: experimental evidence and modeling of a
           slow K+-dependent mechanism. The Journal of Neuroscience,
           21(3), 759-770.
           doi:10.1523/JNEUROSCI.21-03-00759.2001
    .. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Potassium
    gates = (
        Gate("a", power=3, q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
        Gate("b", q10=3.0, temp_ref=u.celsius2kelvin(25.5)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 3.2 * (u.mS / u.cm**2),
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

    def _a_alpha(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_a * _sigm(V - self.V0alpha_a.to_decimal(u.mV), self.Kalpha_a.to_decimal(u.mV))

    def _a_beta(self, V, K: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_a / u.math.exp((V - self.V0beta_a.to_decimal(u.mV)) / self.Kbeta_a.to_decimal(u.mV))

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


@register_channel("Kdr_ZH2019_IO")
class Kdr_ZH2019_IO(OhmicHH):
    r"""Delayed rectifier of the inferior-olive model (Zhang 2019).

    The somatic delayed-rectifier potassium current of the
    single-compartment inferior olive (IO) neurons in the
    essential-tremor cortico-cerebello-thalamo-cortical loop model of
    (Zhang & Santaniello, 2019) [2]_, with :math:`n^4` HH gating and
    an ohmic driving force:

    .. math::

        \begin{aligned}
        \alpha_n &= 10 \, S\!\left(\frac{V + 41}{10}\right),
                    \quad S(x) = \frac{x}{1 - \exp(-x)} \\
        \beta_n &= 12.5 \exp(-(V + 51) / 80) \\
        n_\infty &= \frac{\alpha_n}{\alpha_n + \beta_n}, \qquad
        \tau_n = \frac{10}{\alpha_n + \beta_n}
        \end{aligned}

    where :math:`V` is in millivolts -- this class applies no voltage
    shift -- the rates are in :math:`\mathrm{ms}^{-1}` and
    :math:`\tau_n` is in milliseconds. Expanding :math:`S` gives the
    familiar linoid
    :math:`\alpha_n = (V + 41)/(1 - \exp(-(V + 41)/10))` everywhere
    except at its singular point (see Notes). The reversal potential
    comes from the potassium ion object, not from the class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``18.0 mS/cm2``,
        exactly the source mechanism's ``gbar``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.channel.Na_ZH2019_IO : Sodium current of the same
        inferior-olive import family and the same bibliographic
        origin.
    braincell.channel.hyperpolarization_activated.HCN_ZH2019_IO :
        H current of the same family.

    Notes
    -----
    Ported from ``IO/channel/Kdr_ZH19_IO.mod``, whose header reads
    "K_dr channel from Schweighofer et al 1999. The referred model is
    an inferior olive neuron" with porter credit "B. Torben-Nielsen @
    HUJI, 21-10-2010". The kinetics therefore originate with
    Schweighofer, Doya, & Kawato (1999) [1]_ and reached this class
    through the NEURON port of Torben-Nielsen, Segev, & Yarom (2012),
    which Zhang & Santaniello (2019) [2]_ reused without further
    modification credit. The 2012 port paper is named in this prose
    only, per house style, and not as a numbered reference.

    The inferior olive neurons in both of those models are
    **single-compartment** (``nseg = 1``); the multi-compartment part
    of that lineage is a separate Purkinje-cell population. This class
    must not be described as part of a multi-compartment inferior
    olive mechanism.

    **Singularity guard: the mod file's branch is not reproduced.**
    ``Kdr_ZH19_IO.mod`` guards the removable singularity of
    :math:`\alpha_n` with ``if (fabs(v + 41.0) < 1e-6)`` and, inside
    that branch, substitutes the perturbed literal ``41.00001``.
    BrainCell instead routes the expression through the stable helper
    :math:`S(x) = x/(1 - \exp(-x))`, which returns :math:`1 + x/2`
    when ``abs(x) < 1e-6``. Because the guard now applies to the
    scaled argument :math:`x = (V + 41)/10`, it triggers for
    ``abs(V + 41) < 1e-5 mV`` and yields
    :math:`\alpha_n \approx 10\ \mathrm{ms}^{-1}` there. The
    substitution is exact away from the singular point and
    better-behaved at it, but it is a BrainCell choice: the import
    README explicitly excludes those in-formula literals from its
    NMODL default-precision rewrites.

    **Import deviation.** As for every mechanism of this ``ZH19``/IO
    family, the ``rates(v)`` call moved from the NMODL ``BREAKPOINT``
    into ``DERIVATIVE states``, so ``ninf``/``taun`` are refreshed
    before the ``cnexp`` state update rather than after it. That is a
    semantic change, not a cosmetic one.

    Two further details of the source, recorded so they are not
    mistaken for port errors: the mod file's ``ek = -75 mV``
    ``PARAMETER`` is absent here because the reversal potential is
    supplied by the ion object, and its ``taun`` numerator carries the
    upstream comment ": was 5", i.e. the factor 10 above replaced an
    earlier 5 before this port.

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
    root_type = Potassium
    gates = (Gate("n", power=4),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 18.0 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

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
