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


"""Voltage-dependent sodium channels built directly on HH templates."""

from typing import Optional

import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell._typing import ArrayLike, Initializer, Size
from braincell.channel._base import Gate, HH, Markov, OhmicHH
from braincell.ion import Sodium
from braincell.mech import register_channel
from braincell.quad.protocol import IndependentIntegration

__all__ = [
    "Na_Ba2002",
    "Na_TM1991",
    "Na_HH1952",
    "NaF_SU2015_DCN",
    "NaP_SU2015_DCN",
    "Na_ZH2019_IO",
    "Nav1p6_MA2020_GoC",
    "Nav1p6_MA2024_PC",
    "Nav1p6_MA2025_BC",
    "Nav1p6_RI2021_SC",
    "Nav1p1_MA2025_BC",
    "Nav1p1_RI2021_SC",
    "Nav_MA2020_GrC",
    "NaFHF_MA2020_GrC",
]


def _x_over_one_minus_exp_neg_stable(x):
    return u.math.where(
        u.math.abs(x) < 1e-6,
        1.0 + x / 2.0,
        x / (1.0 - u.math.exp(-x)),
    )


@register_channel("Na_Ba2002")
class Na_Ba2002(OhmicHH):
    r"""Bazhenov 2002 sodium current with :math:`p^3 q` HH gating.

    The thalamocortical-cell fast sodium current used in the slow-wave
    sleep oscillation model of (Bazhenov et al., 2002) [1]_, with the
    mirrored-sign Traub-Miles rate forms (see Notes):

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.32 \times 4}
                    {\operatorname{exprel}(-(V' - 13) / 4)} \\
        \beta_p &= \frac{0.28 \times 5}
                   {\operatorname{exprel}((V' - 40) / 5)} \\
        \alpha_q &= 0.128 \exp(-(V' - 17) / 18) \\
        \beta_q &= \frac{4}{1 + \exp(-(V' - 40) / 5)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\operatorname{exprel}(x) = (\exp(x) - 1) / x`, used only to
    remove the removable singularity at each Boltzmann midpoint; it
    does not change the function value. Gating is :math:`p^3 q`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``90.0 mS/cm2`` --
        matches the TC-cell value reported in (Bazhenov et al., 2002)
        [1]_.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for both gates, default ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for the Q10 formula, default 36
        degrees Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``-50.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Na_TM1991 : Same Traub-Miles rate functions with ``V_sh = -63 mV``
        and ``q10 = 1.0`` instead.

    Notes
    -----
    Algebraically these are the same Traub & Miles (1991) rate
    functions used by :class:`Na_TM1991`, written with the sign of
    ``V - V_sh`` mirrored; the only numeric differences are this
    class's ``V_sh = -50.0 mV`` in place of ``Na_TM1991``'s
    ``-63.0 mV``, and ``q10 = 3.0`` in place of ``1.0``. (Bazhenov et
    al., 2002) [1]_ attribute the current's kinetics to Traub & Miles
    (1991) but do **not** print the alpha/beta expressions themselves
    -- the paper's Methods defer them to Bazhenov et al. (1998). This
    docstring's equations are transcribed from this class's own rate
    methods, confirmed algebraically equivalent to the Traub-Miles
    forms, not copied from the 2002 paper's text.

    References
    ----------
    .. [1] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
           (2002). Model of thalamocortical slow-wave sleep
           oscillations and transitions to activated states. The
           Journal of Neuroscience, 22(19), 8691-8704.
           doi:10.1523/JNEUROSCI.22-19-08691.2002
    """

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("p", power=3, q10="q10", temp_ref="temp_ref"),
        Gate("q", q10="q10", temp_ref="temp_ref"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 90.0 * (u.mS / u.cm**2),
        temp: ArrayLike = u.celsius2kelvin(36.0),
        q10: Initializer = 3.0,
        temp_ref: ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Initializer = -50.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 13.0
        return 0.32 * 4.0 / u.math.exprel(-temp / 4.0)

    def f_p_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 40.0
        return 0.28 * 5.0 / u.math.exprel(temp / 5.0)

    def f_q_alpha(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.128 * u.math.exp(-(temp - 17.0) / 18.0)

    def f_q_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 4.0 / (1.0 + u.math.exp(-(temp - 40.0) / 5.0))


@register_channel("Na_TM1991")
class Na_TM1991(OhmicHH):
    r"""Traub and Miles 1991 sodium current with :math:`p^3 q` HH gating.

    The hippocampal pyramidal-cell fast sodium current from the
    Traub & Miles (1991) [1]_ reduced model, as packaged in the
    NEURON ``HH2.mod`` mechanism:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.32 \times 4}
                    {\operatorname{exprel}(-(V' - 13) / 4)} \\
        \beta_p &= \frac{0.28 \times 5}
                   {\operatorname{exprel}((V' - 40) / 5)} \\
        \alpha_q &= 0.128 \exp(-(V' - 17) / 18) \\
        \beta_q &= \frac{4}{1 + \exp(-(V' - 40) / 5)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\operatorname{exprel}(x) = (\exp(x) - 1) / x` removes the
    removable singularity at each Boltzmann midpoint without changing
    the function value. Gating is :math:`p^3 q`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``120.0 mS/cm2``. This
        value is a BrainCell default, not a Traub & Miles (1991)
        [1]_ figure -- ``HH2.mod`` itself ships ``gnabar = 0.1
        mho/cm2`` (100 mS/cm2).
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for both gates, default ``1.0`` (no
        temperature scaling at the reference temperature).
    temp_ref : array-like, optional
        Reference temperature for the Q10 formula, default 36
        degrees Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``-63.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Na_Ba2002 : Same Traub-Miles rate functions with ``V_sh = -50 mV``
        and ``q10 = 3.0`` instead.

    Notes
    -----
    Ported from ``HH2.mod``. Algebraically these are the same rate
    functions used by :class:`Na_Ba2002`; the only numeric
    differences are this class's ``V_sh = -63.0 mV`` in place of
    ``Na_Ba2002``'s ``-50.0 mV``, and ``q10 = 1.0`` in place of
    ``3.0``.

    The ``-63.0 mV`` default is the value Destexhe's network ``.hoc``
    files assign to ``vtraub`` for this current, and it is the offset
    associated with the Traub-Miles hippocampal pyramidal cell that
    this class is named for. It is **not** ``HH2.mod``'s own default,
    which is ``-55 mV``. This class's potassium sibling,
    :class:`K_TM1991`, ships a different default shift
    (``-60.0 mV``) from the same ``.mod`` file; the two classes do
    **not** share a shift, and neither should be read as implying a
    single canonical Traub-Miles offset.

    References
    ----------
    .. [1] Traub, R. D., & Miles, R. (1991). Neuronal networks of the
           hippocampus. Cambridge University Press.
           doi:10.1017/CBO9780511895401
    """

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("p", power=3, q10="q10", temp_ref="temp_ref"),
        Gate("q", q10="q10", temp_ref="temp_ref"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 120.0 * (u.mS / u.cm**2),
        temp: ArrayLike = u.celsius2kelvin(36.0),
        q10: Initializer = 1.0,
        temp_ref: ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Initializer = -63.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V, Na: IonInfo):
        temp = (self.V_sh - V).to_decimal(u.mV)
        return 0.32 * 4.0 / u.math.exprel((13.0 + temp) / 4.0)

    def f_p_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 40.0
        return 0.28 * 5.0 / u.math.exprel(temp / 5.0)

    def f_q_alpha(self, V, Na: IonInfo):
        temp = (self.V_sh - V).to_decimal(u.mV)
        return 0.128 * u.math.exp((17.0 + temp) / 18.0)

    def f_q_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 4.0 / (1.0 + u.math.exp(-(temp - 40.0) / 5.0))


@register_channel("Na_HH1952")
class Na_HH1952(OhmicHH):
    r"""Hodgkin-Huxley 1952 sodium current with :math:`p^3 q` HH gating.

    The original squid giant axon fast sodium current of
    (Hodgkin & Huxley, 1952) [1]_:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{1}{\operatorname{exprel}(-(V' - 5) / 10)} \\
        \beta_p &= 4 \exp(-(V' + 20) / 18) \\
        \alpha_q &= 0.07 \exp(-(V' + 20) / 20) \\
        \beta_q &= \frac{1}{1 + \exp(-(V' - 10) / 10)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\operatorname{exprel}(x) = (\exp(x) - 1) / x` removes the
    removable singularity at the Boltzmann midpoint without changing
    the function value. With the default ``V_sh = -45 mV`` (so
    ``V' = (V/mV) + 45``, placing rest at -65 mV in the modern
    absolute-potential convention), these expand to exactly the
    published forms
    :math:`\alpha_m = 0.1 (V+40) / (1 - \exp(-(V+40)/10))`,
    :math:`\beta_m = 4 \exp(-(V+65)/18)`,
    :math:`\alpha_h = 0.07 \exp(-(V+65)/20)`,
    :math:`\beta_h = 1 / (1 + \exp(-(V+35)/10))`. Gating is
    :math:`p^3 q`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``120.0 mS/cm2`` --
        matches (Hodgkin & Huxley, 1952) [1]_'s :math:`g_{Na}`.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for both gates, default ``3.0`` -- the
        paper's own factor-of-3-per-10-degree-Celsius correction.
    temp_ref : array-like, optional
        Reference temperature for the Q10 formula, default 36
        degrees Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``-45.0 mV``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    (Hodgkin & Huxley, 1952) [1]_ measured these rates at 6.3 degrees
    Celsius, not this class's ``temp_ref = 36`` degrees Celsius
    default. Because ``temp`` and ``temp_ref`` are equal by default,
    the Q10 correction is a no-op at construction time; the defaults
    do **not** reproduce the original 6.3-degree-Celsius kinetics.

    References
    ----------
    .. [1] Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative
           description of membrane current and its application to
           conduction and excitation in nerve. The Journal of
           Physiology, 117(4), 500-544.
           doi:10.1113/jphysiol.1952.sp004764
    """

    __module__ = "braincell.channel"
    root_type = Sodium
    gates = (
        Gate("p", power=3, q10="q10", temp_ref="temp_ref"),
        Gate("q", q10="q10", temp_ref="temp_ref"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 120.0 * (u.mS / u.cm**2),
        temp: ArrayLike = u.celsius2kelvin(36.0),
        q10: Initializer = 3.0,
        temp_ref: ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Initializer = -45.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10 = braintools.init.param(q10, self.varshape, allow_none=False)
        self.temp_ref = braintools.init.param(temp_ref, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)

    def f_p_alpha(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV) - 5.0
        return 1.0 / u.math.exprel(-temp / 10.0)

    def f_p_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 4.0 * u.math.exp(-(temp + 20.0) / 18.0)

    def f_q_alpha(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 0.07 * u.math.exp(-(temp + 20.0) / 20.0)

    def f_q_beta(self, V, Na: IonInfo):
        temp = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(temp - 10.0) / 10.0))


@register_channel("NaF_SU2015_DCN")
class NaF_SU2015_DCN(OhmicHH):
    r"""Fast sodium current of the deep cerebellar nuclei model.

    Deep cerebellar nucleus (DCN) fast sodium current, part of the
    model published as (Sudhakar et al., 2015) [2]_:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp((V + 45) / -7.3)} \\
        \tau_m &= \left[\frac{5.83}{\exp((V - 6.4)/-9)
                   + \exp((V + 97)/17)} + 0.025\right] / q \\
        h_\infty &= \frac{1}{1 + \exp((V + 42) / 5.9)} \\
        \tau_h &= \left[\frac{16.67}{\exp((V - 8.3)/-29)
                   + \exp((V + 66)/9)} + 0.2\right] / q
        \end{aligned}

    with :math:`V` in mV and :math:`q` = ``qdeltat``. Gating is
    :math:`m^3 h`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.01 mS/cm2``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    Ported from ``NaF_SU15_DCN.mod``. The mechanism name ``NaF`` does
    not appear anywhere in the text of (Sudhakar et al., 2015) [2]_;
    this docstring says only that the mechanism is part of the
    published model, not that the paper names or describes it.

    Kinetics originate in the GENESIS deep cerebellar nucleus model of
    Steuber et al. (2011) [1]_, translated to NEURON by Luthman et al.
    (2011), and reused by (Sudhakar et al., 2015) [2]_, which cites
    only the GENESIS model and not the NEURON translation.

    The former NEURON ``TABLE`` tabulated ``minf``, ``taum``, ``hinf``
    and ``tauh`` over ``[-150, 100] mV`` and clamped outside that
    range; BrainCell evaluates the continuous formulas above at every
    call instead, so values outside ``[-150, 100] mV`` are expected to
    diverge from the original NEURON boundary-clamped output.

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
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.01 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def f_m_inf(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 45.0) / -7.3))

    def f_m_tau(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return (5.83 / (u.math.exp((V - 6.4) / -9.0) + u.math.exp((V + 97.0) / 17.0)) + 0.025) / self.qdeltat

    def f_h_inf(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 42.0) / 5.9))

    def f_h_tau(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return (16.67 / (u.math.exp((V - 8.3) / -29.0) + u.math.exp((V + 66.0) / 9.0)) + 0.2) / self.qdeltat


@register_channel("NaP_SU2015_DCN")
class NaP_SU2015_DCN(OhmicHH):
    r"""Persistent sodium current of the deep cerebellar nuclei model.

    Deep cerebellar nucleus (DCN) persistent sodium current, part of
    the model published as (Sudhakar et al., 2015) [2]_:

    .. math::

        \begin{aligned}
        m_\infty &= \frac{1}{1 + \exp((V + 70) / -4.1)} \\
        \tau_m &= 50 / q \\
        h_\infty &= \frac{1}{1 + \exp((V + 80) / 4)} \\
        \tau_h &= \left[\frac{1750}{1 + \exp((V + 65)/-8)}
                   + 250\right] / q
        \end{aligned}

    with :math:`V` in mV and :math:`q` = ``qdeltat``; :math:`\tau_m`
    is a bare constant with no voltage dependence. Gating is
    :math:`m^3 h`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.01 mS/cm2``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    Ported from ``NaP_SU15_DCN.mod``. ``NaP`` is one of only four
    mechanism names (with ``CaLVA``, ``HCN`` and ``SK``) that actually
    occur in the text of (Sudhakar et al., 2015) [2]_; even so, the
    paper does not print these Boltzmann constants.

    Kinetics originate in the GENESIS deep cerebellar nucleus model of
    Steuber et al. (2011) [1]_, translated to NEURON by Luthman et al.
    (2011), and reused by (Sudhakar et al., 2015) [2]_, which cites
    only the GENESIS model and not the NEURON translation.

    The former NEURON ``TABLE`` tabulated ``minf``, ``hinf`` and
    ``tauh`` (but not ``taum``, which does not depend on voltage) over
    ``[-150, 100] mV`` and clamped outside that range; BrainCell
    evaluates the continuous formulas above at every call instead, so
    values outside ``[-150, 100] mV`` are expected to diverge from the
    original NEURON boundary-clamped output.

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
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.01 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = 1.0

    def f_m_inf(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 70.0) / -4.1))

    def f_m_tau(self, V, Na: IonInfo):
        return 50.0 / self.qdeltat

    def f_h_inf(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 4.0))

    def f_h_tau(self, V, Na: IonInfo):
        V = V.to_decimal(u.mV)
        return (1750.0 / (1.0 + u.math.exp((V + 65.0) / -8.0)) + 250.0) / self.qdeltat


@register_channel("Na_ZH2019_IO")
class Na_ZH2019_IO(OhmicHH):
    r"""Inferior olive somatic sodium current, Schweighofer kinetics.

    Inferior olive (IO) somatic fast sodium current from the
    essential-tremor cortico-cerebello-thalamo-cortical loop model of
    (Zhang & Santaniello, 2019) [2]_:

    .. math::

        \begin{aligned}
        \alpha_m &= x/(1 - \exp(-x)),\ x = (V + 41) / 10 \\
        \beta_m &= 9 \exp(-(V + 66) / 20) \\
        \alpha_h &= 5 \exp(-(V + 60) / 15) \\
        \beta_h &= 10 y/(1 - \exp(-y)),\ y = (V + 50) / 10
        \end{aligned}

    with :math:`V` in mV,
    :math:`m_\infty = \alpha_m / (\alpha_m + \beta_m)`,
    :math:`\tau_m = 0.001` ms (effectively instantaneous),
    :math:`h_\infty = \alpha_h / (\alpha_h + \beta_h)`,
    :math:`\tau_h = 250 / (\alpha_h + \beta_h)`. Gating is
    :math:`m^3 h`. Both :math:`x/(1-\exp(-x))` terms are evaluated by
    the module-level helper ``_x_over_one_minus_exp_neg_stable``,
    which substitutes the Taylor form ``1 + x/2`` for
    ``|x| < 1e-6`` and the closed form otherwise -- this is the
    numerically stable replacement for the removable singularity at
    :math:`x = 0`, not the naive textbook fraction.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``70.0 mS/cm2``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    Ported from ``Na_ZH19_IO.mod``. Kinetics originate in the
    single-compartment inferior olive model of Schweighofer, Doya &
    Kawato (1999) [1]_, reaching this class through the NEURON port of
    Torben-Nielsen, Segev & Yarom (2012), and reused by (Zhang &
    Santaniello, 2019) [2]_.

    ``Na_ZH19_IO.mod`` guards the removable singularities in
    ``alpha_m`` and ``beta_h`` with an explicit
    ``if (fabs(v + 41.0) < 1e-6)`` / ``if (fabs(v + 50.0) < 1e-6)``
    branch that substitutes the perturbed literal ``41.000001`` /
    ``50.000001``. BrainCell replaces both branches with
    ``_x_over_one_minus_exp_neg_stable`` instead of reproducing the
    perturbed-literal branch; this is exact away from the singularity
    and better-behaved at it, but it is a BrainCell substitution, not
    a reproduction of the ``.mod`` file's own guard.

    The upstream ``rates(v)`` call was relocated from ``BREAKPOINT``
    into ``DERIVATIVE states``, so ``m``/``h`` steady states and time
    constants are refreshed before the ``cnexp`` state update rather
    than after it -- a semantic change carried over from the source
    ``.mod`` file, not introduced by this port.

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
    root_type = Sodium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 70.0 * (u.mS / u.cm**2),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

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

    def f_m_inf(self, V, Na: IonInfo):
        alpha = self._m_alpha(V)
        beta = self._m_beta(V)
        return alpha / (alpha + beta)

    def f_m_tau(self, V, Na: IonInfo):
        return 0.001

    def f_h_inf(self, V, Na: IonInfo):
        alpha = self._h_alpha(V)
        beta = self._h_beta(V)
        return alpha / (alpha + beta)

    def f_h_tau(self, V, Na: IonInfo):
        alpha = self._h_alpha(V)
        beta = self._h_beta(V)
        return 250.0 / (alpha + beta)


@register_channel("Nav1p6_MA2020_GoC")
class Nav1p6_MA2020_GoC(Markov):
    r"""Resurgent Nav1.6 sodium current, Golgi-cell parameterisation.

    The 13-state Raman & Bean [1]_ / Khaliq et al. [2]_ resurgent
    sodium Markov scheme, as implemented by Akemann & Knopfel (2006)
    [3]_ and parameterised for the Golgi cell model of
    (Masoli et al., 2020) [4]_. Five closed states ``C1``-``C5`` form
    an activation ladder that opens into ``O``; ``O`` can additionally
    transition into a blocked state ``B`` (the resurgent-current
    mechanism) or into a deep-inactivated state ``I6``; each closed
    state ``Cn`` has a matching inactivated state ``In``
    (``n = 1..5``), themselves connected in the same ladder topology
    as the closed states and converging on ``I6`` from ``I5``.
    ``I6`` is eliminated algebraically (``dependent_state``).

    .. math::

        \begin{aligned}
        f_{0,n} &= (5-n)\,\alpha \exp(V'/x_1)\,\phi,
                   &\quad b_{0,n} &= n\,\beta
                   \exp((V' + v_a)/(x_2 + v_k))\,\phi \\
        f_{0O} &= \gamma \exp(V'/x_3)\,\phi,
                  &\quad b_{0O} &= \delta \exp(V'/x_4)\,\phi \\
        f_{ip} &= \epsilon \exp(V'/x_5)\,\phi,
                  &\quad b_{ip} &= \zeta \exp(V'/x_6)\,\phi \\
        f_{1,n} &= (5-n)\,\alpha\,a \exp((V' + v_i)/x_1)\,\phi,
                   &\quad b_{1,n} &= n\,\beta\,b
                   \exp((V' + v_i)/x_2)\,\phi \\
        f_{1n} &= \gamma \exp(V'/x_3)\,\phi,
                  &\quad b_{1n} &= \delta \exp(V'/x_4)\,\phi \\
        f_{i,n} &= C_{on}\,a^{\,n-1}\,\phi,
                   &\quad b_{i,n} &= C_{off}\,b^{\,n-1}\,\phi \\
        f_{in} &= O_{on}\,\phi, &\quad b_{in} &= O_{off}\,\phi
        \end{aligned}

    for :math:`n = 1..4` (:math:`f_{0,n}`/:math:`b_{0,n}`,
    :math:`f_{1,n}`/:math:`b_{1,n}`) and :math:`n = 1..5`
    (:math:`f_{i,n}`/:math:`b_{i,n}`), where :math:`V' = V/\mathrm{mV}`,
    :math:`a` = ``alfac`` :math:`= (O_{on}/C_{on})^{1/4}`, :math:`b` =
    ``btfac`` :math:`= (O_{off}/C_{off})^{1/4}`, and
    :math:`\phi = 3^{(T - 22)/10}` with :math:`T` in degrees Celsius.
    The current is :math:`g_{max} \cdot O \cdot (E_{Na} - V)`, taken
    directly from the open-state occupancy rather than a
    ``power``-weighted gate product.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius (the reference temperature itself, so
        :math:`\phi = 1`).
    g_max : array-like or callable, optional
        Maximal conductance density, default ``16.0 mS/cm2``.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p6_MA2024_PC : Same scheme and constants, Purkinje-cell model
        citation.
    Nav1p6_MA2025_BC : Same scheme and constants, basket-cell model
        citation.
    Nav1p6_RI2021_SC : Same scheme and constants, stellate-cell model
        citation.
    Nav1p1_MA2025_BC : Same 13-state topology with an independently
        overridden constant set for the non-resurgent Nav1.1 variant.

    Notes
    -----
    Ported from ``Nav1p6_MA20_GoC.mod``. All of ``Con``, ``Coff``,
    ``Oon``, ``Ooff``, ``alpha``, ``beta``, ``gamma``, ``delta``,
    ``epsilon``, ``zeta``, ``x1``-``x6``, ``vshifta``, ``vshifti``,
    ``vshiftk``, ``alfac`` and ``btfac`` are fixed internal constants
    set in ``__init__`` and are not exposed as ``__init__`` parameters.

    No import deviation (``TABLE`` removal, ``derivimplicit`` ->
    ``cnexp`` substitution, rate-refresh relocation, or NMODL
    default-precision rewrite) is recorded for this mechanism in the
    bibliography's ``MA2020`` import-deviations tables.

    References
    ----------
    .. [1] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [2] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [3] Akemann, W., & Knopfel, T. (2006). Interaction of Kv3
           potassium channels and resurgent sodium current influences
           the rate of spontaneous firing of Purkinje neurons. The
           Journal of Neuroscience, 26(17), 4602-4612.
           doi:10.1523/JNEUROSCI.5204-05.2006
    .. [4] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C1", "I1", "fi1", "bi1"),
        ("I1", "I2", "f11", "b11"),
        ("C2", "I2", "fi2", "bi2"),
        ("I2", "I3", "f12", "b12"),
        ("C3", "I3", "fi3", "bi3"),
        ("I3", "I4", "f13", "b13"),
        ("C4", "I4", "fi4", "bi4"),
        ("I4", "I5", "f14", "b14"),
        ("C5", "I5", "fi5", "bi5"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "B", "fip", "bip"),
        ("I5", "I6", "f1n", "b1n"),
        ("O", "I6", "fin", "bin"),
    )
    dependent_state = "I6"

    def __init__(
        self,
        size: Size,
        temp: ArrayLike = u.celsius2kelvin(22.0),
        g_max: Initializer = 16.0 * (u.mS / u.cm**2),
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

        self.Con = 0.005
        self.Coff = 0.5
        self.Oon = 0.75
        self.Ooff = 0.005
        self.alpha = 150.0
        self.beta = 3.0
        self.gamma = 150.0
        self.delta = 40.0
        self.epsilon = 1.75
        self.zeta = 0.03

        self.x1 = 20.0
        self.x2 = -20.0
        self.x3 = 1e12
        self.x4 = -1e12
        self.x5 = 1e12
        self.x6 = -25.0
        self.vshifta = 0.0
        self.vshifti = 0.0
        self.vshiftk = 0.0

        self.alfac = (self.Oon / self.Con) ** (1 / 4)
        self.btfac = (self.Ooff / self.Coff) ** (1 / 4)

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    f01 = lambda self, V: 4 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f02 = lambda self, V: 3 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f03 = lambda self, V: 2 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f04 = lambda self, V: 1 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f0O = lambda self, V: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fip = lambda self, V: self.epsilon * u.math.exp((V / u.mV) / self.x5) * self.phi
    f11 = lambda self, V: 4 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f12 = lambda self, V: 3 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f13 = lambda self, V: 2 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f14 = lambda self, V: 1 * self.alpha * self.alfac * u.math.exp((V / u.mV + self.vshifti) / self.x1) * self.phi
    f1n = lambda self, V: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fi1 = lambda self, V: self.Con * self.phi
    fi2 = lambda self, V: self.Con * self.alfac * self.phi
    fi3 = lambda self, V: self.Con * self.alfac**2 * self.phi
    fi4 = lambda self, V: self.Con * self.alfac**3 * self.phi
    fi5 = lambda self, V: self.Con * self.alfac**4 * self.phi
    fin = lambda self, V: self.Oon * self.phi

    b01 = lambda self, V: 1 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b02 = lambda self, V: 2 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b03 = lambda self, V: 3 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b04 = lambda self, V: 4 * self.beta * u.math.exp((V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b0O = lambda self, V: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bip = lambda self, V: self.zeta * u.math.exp(V / u.mV / self.x6) * self.phi
    b11 = lambda self, V: 1 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b12 = lambda self, V: 2 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b13 = lambda self, V: 3 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b14 = lambda self, V: 4 * self.beta * self.btfac * u.math.exp((V / u.mV + self.vshifti) / self.x2) * self.phi
    b1n = lambda self, V: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bi1 = lambda self, V: self.Coff * self.phi
    bi2 = lambda self, V: self.Coff * self.btfac * self.phi
    bi3 = lambda self, V: self.Coff * self.btfac**2 * self.phi
    bi4 = lambda self, V: self.Coff * self.btfac**3 * self.phi
    bi5 = lambda self, V: self.Coff * self.btfac**4 * self.phi
    bin = lambda self, V: self.Ooff * self.phi

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p6_MA2024_PC")
class Nav1p6_MA2024_PC(Nav1p6_MA2020_GoC):
    r"""Resurgent Nav1.6 sodium current, Purkinje-cell parameterisation.

    The same 13-state Raman & Bean [1]_ / Khaliq et al. [2]_ resurgent
    sodium Markov scheme documented in :class:`Nav1p6_MA2020_GoC`, as
    implemented by Akemann & Knopfel (2006) [3]_ and reused unchanged
    for the human Purkinje-cell model of
    (Masoli et al., 2024) [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius. Inherited from :class:`Nav1p6_MA2020_GoC`.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``16.0 mS/cm2``.
        Inherited from :class:`Nav1p6_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p6_MA2020_GoC : The base class; full equations, state
        topology and constant values are documented there.
    Nav1p6_MA2025_BC : Same scheme and constants, basket-cell model
        citation.
    Nav1p6_RI2021_SC : Same scheme and constants, stellate-cell model
        citation.

    Notes
    -----
    Ported from ``Nav1p6_MA24_PC.mod``. This class does not override
    ``__init__``: the constructor, the transition-rate lambdas, the
    fixed kinetic constants and :meth:`current` are all inherited
    unchanged from :class:`Nav1p6_MA2020_GoC`. Only the
    ``register_channel`` key and this docstring's model citation
    differ -- the ``.mod`` file this class ports from parameterises
    the identical mechanism for a different cell type, not a
    different kinetic scheme.

    No import deviation is recorded for this mechanism in the
    bibliography's ``MA2024`` import-deviations tables.

    References
    ----------
    .. [1] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [2] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [3] Akemann, W., & Knopfel, T. (2006). Interaction of Kv3
           potassium channels and resurgent sodium current influences
           the rate of spontaneous firing of Purkinje neurons. The
           Journal of Neuroscience, 26(17), 4602-4612.
           doi:10.1523/JNEUROSCI.5204-05.2006
    .. [4] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p6_MA2025_BC")
class Nav1p6_MA2025_BC(Nav1p6_MA2020_GoC):
    r"""Resurgent Nav1.6 sodium current, basket-cell parameterisation.

    The same 13-state Raman & Bean [1]_ / Khaliq et al. [2]_ resurgent
    sodium Markov scheme documented in :class:`Nav1p6_MA2020_GoC`, as
    implemented by Akemann & Knopfel (2006) [3]_ and reused unchanged
    for the basket-cell model of
    (Masoli et al., 2025) [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius. Inherited from :class:`Nav1p6_MA2020_GoC`.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``16.0 mS/cm2``.
        Inherited from :class:`Nav1p6_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p6_MA2020_GoC : The base class; full equations, state
        topology and constant values are documented there.
    Nav1p6_MA2024_PC : Same scheme and constants, Purkinje-cell model
        citation.
    Nav1p6_RI2021_SC : Same scheme and constants, stellate-cell model
        citation.
    Nav1p1_MA2025_BC : Non-resurgent Nav1.1 sibling for the same
        basket-cell model, with its own overridden constants.

    Notes
    -----
    Ported from ``Nav1p6_MA25_BC.mod``. This class does not override
    ``__init__``: the constructor, the transition-rate lambdas, the
    fixed kinetic constants and :meth:`current` are all inherited
    unchanged from :class:`Nav1p6_MA2020_GoC`. Only the
    ``register_channel`` key and this docstring's model citation
    differ.

    No import deviation is recorded for this mechanism in the
    bibliography's ``MA2025`` import-deviations tables.

    References
    ----------
    .. [1] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [2] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [3] Akemann, W., & Knopfel, T. (2006). Interaction of Kv3
           potassium channels and resurgent sodium current influences
           the rate of spontaneous firing of Purkinje neurons. The
           Journal of Neuroscience, 26(17), 4602-4612.
           doi:10.1523/JNEUROSCI.5204-05.2006
    .. [4] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p6_RI2021_SC")
class Nav1p6_RI2021_SC(Nav1p6_MA2020_GoC):
    r"""Resurgent Nav1.6 sodium current, stellate-cell parameterisation.

    The same 13-state Raman & Bean [1]_ / Khaliq et al. [2]_ resurgent
    sodium Markov scheme documented in :class:`Nav1p6_MA2020_GoC`, as
    implemented by Akemann & Knopfel (2006) [3]_ and reused unchanged
    for the stellate-cell model of
    (Rizza et al., 2021) [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius. Inherited from :class:`Nav1p6_MA2020_GoC`.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``16.0 mS/cm2``.
        Inherited from :class:`Nav1p6_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p6_MA2020_GoC : The base class; full equations, state
        topology and constant values are documented there.
    Nav1p6_MA2024_PC : Same scheme and constants, Purkinje-cell model
        citation.
    Nav1p6_MA2025_BC : Same scheme and constants, basket-cell model
        citation.
    Nav1p1_RI2021_SC : Non-resurgent Nav1.1 sibling for the same
        stellate-cell model.

    Notes
    -----
    Ported from ``Nav1p6_RI21_SC.mod``. This class does not override
    ``__init__``: the constructor, the transition-rate lambdas, the
    fixed kinetic constants and :meth:`current` are all inherited
    unchanged from :class:`Nav1p6_MA2020_GoC`. Only the
    ``register_channel`` key and this docstring's model citation
    differ.

    No import deviation is recorded for this mechanism in the
    bibliography's ``RI2021`` import-deviations tables.

    References
    ----------
    .. [1] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [2] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
    .. [3] Akemann, W., & Knopfel, T. (2006). Interaction of Kv3
           potassium channels and resurgent sodium current influences
           the rate of spontaneous firing of Purkinje neurons. The
           Journal of Neuroscience, 26(17), 4602-4612.
           doi:10.1523/JNEUROSCI.5204-05.2006
    .. [4] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p1_MA2025_BC")
class Nav1p1_MA2025_BC(Nav1p6_MA2020_GoC):
    r"""Non-resurgent Nav1.1 sodium current, basket-cell parameterisation.

    The non-resurgent ``Narsg`` sodium current derived from the
    Khaliq et al. (2003) [1]_ resurgent-current model and used for
    Kv3/Nav1.1 excitability studies by Akemann, Lundby, Mutoh &
    Knopfel (2009) [2]_, reparameterised for the basket-cell model of
    (Masoli et al., 2025) [3]_.

    This class inherits the 13-state Markov ``pairs`` topology, the
    transition-rate lambdas and the :math:`\phi` temperature-scaling
    formula from :class:`Nav1p6_MA2020_GoC` (see that class for the
    full state diagram and rate equations), but overrides three of
    the fixed kinetic constants and recomputes :math:`\phi` with a
    different base:

    .. math::

        \phi = 2.7^{(T - 22)/10}, \quad
        O_{on} = 2.3\ \mathrm{ms^{-1}}, \quad
        \epsilon = 10^{-12}\ \mathrm{ms^{-1}}, \quad
        a = \left(\frac{O_{on}}{C_{on}}\right)^{1/4}

    with :math:`C_{on}` unchanged from :class:`Nav1p6_MA2020_GoC`'s
    ``0.005``, so ``alfac`` is recomputed to a different value than
    the base class's. :meth:`current` adds an optional gating-current
    correction on top of the inherited ionic current:

    .. math::

        \begin{aligned}
        I &= g_{max}\, O\, (E_{Na} - V) - \mathbb{1}[\text{gateCurrent}
             \neq 0]\, I_{gate} \\
        I_{gate} &= n_c \times 10^6\, e_0\, z_{gate}\,
                     \dot{q} \\
        \dot{q} &= f_{01} C_1 + (f_{02}{-}b_{01}) C_2
                    + (f_{03}{-}b_{02}) C_3
                    + (f_{04}{-}b_{03}) C_4 - b_{04} C_5 \\
                 &\ + f_{11} I_1 + (f_{12}{-}b_{11}) I_2
                    + (f_{13}{-}b_{12}) I_3
                    + (f_{14}{-}b_{13}) I_4 - b_{14} I_5 \\
        n_c &= 10^{12}\, g_{max} / g_{unit}
        \end{aligned}

    where :math:`\dot{q}` is the net probability flux through the
    activation/inactivation ladders (excluding ``O``, ``B`` and
    ``I6``), :math:`n_c` estimates the channel count from ``g_max``
    and a fixed unitary conductance ``gunit``, and :math:`e_0` is the
    elementary charge.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``8.0 mS/cm2`` (half
        :class:`Nav1p6_MA2020_GoC`'s default).
    gateCurrent : array-like or callable, optional
        Enables the gating-current correction in :meth:`current` when
        nonzero, default ``0.0`` (disabled).
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p1_RI2021_SC : Same scheme and constants, stellate-cell model
        citation.
    Nav1p6_MA2025_BC : Resurgent Nav1.6 sibling for the same
        basket-cell model, with its own (unoverridden) constants.

    Notes
    -----
    Ported from ``Nav1p1_MA25_BC.mod``, whose own header describes
    the mechanism as "derived from the Narsg channel of Khaliq et al.,
    J. Neurosci. 23(2003)4899" -- i.e. via :class:`Nav1p6_MA2020_GoC`'s
    origin, not an independently republished scheme.

    ``zgate = 2.5435``, ``gunit = 15e-9 mS`` and
    ``e0 = 1.60217646e-19 C`` are fixed internal constants set in
    ``__init__`` and are not exposed as ``__init__`` parameters.

    No import deviation is recorded for this mechanism in the
    bibliography's ``MA2025`` import-deviations tables.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
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

    def __init__(
        self,
        size: Size,
        temp: ArrayLike = u.celsius2kelvin(22.0),
        g_max: Initializer = 8.0 * (u.mS / u.cm**2),
        gateCurrent: Initializer = 0.0,
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(
            size=size,
            temp=temp,
            g_max=g_max,
            name=name,
            solver=solver,
            substeps=substeps,
        )
        self.phi = 2.7 ** (((self.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.gateCurrent = braintools.init.param(gateCurrent, self.varshape, allow_none=False)
        self.Oon = 2.3
        self.epsilon = 1e-12
        self.zgate = 2.5435
        self.gunit = 15.0e-9 * u.mS
        self.e0 = 1.60217646e-19 * u.coulomb
        self.alfac = (self.Oon / self.Con) ** (1 / 4)

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)

    def current(self, V, Na: IonInfo):
        conductive = self.g_max * self.O.value * (Na.E - V)
        gate_flip = (
            self.f01(V) * self.C1.value
            + (self.f02(V) - self.b01(V)) * self.C2.value
            + (self.f03(V) - self.b02(V)) * self.C3.value
            + (self.f04(V) - self.b03(V)) * self.C4.value
            - self.b04(V) * self.C5.value
            + self.f11(V) * self.I1.value
            + (self.f12(V) - self.b11(V)) * self.I2.value
            + (self.f13(V) - self.b12(V)) * self.I3.value
            + (self.f14(V) - self.b13(V)) * self.I4.value
            - self.b14(V) * self.I5.value
        ) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * self.zgate * gate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Nav1p1_RI2021_SC")
class Nav1p1_RI2021_SC(Nav1p1_MA2025_BC):
    r"""Non-resurgent Nav1.1 sodium current, stellate-cell parameterisation.

    The same non-resurgent ``Narsg`` sodium current documented on
    :class:`Nav1p1_MA2025_BC` -- derived from the Khaliq et al. (2003)
    [1]_ resurgent-current model and used for Kv3/Nav1.1 excitability
    studies by Akemann, Lundby, Mutoh & Knopfel (2009) [2]_ -- imported
    here for the stellate-cell model of Rizza et al. (2021) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving the :math:`\phi` factor, default
        22 degrees Celsius.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``8.0 mS/cm2``.
    gateCurrent : array-like or callable, optional
        Enables the gating-current correction in :meth:`current` when
        nonzero, default ``0.0`` (disabled).
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav1p1_MA2025_BC : Same 13-state scheme and overridden constants,
        basket-cell model citation; the full kinetics, :math:`\phi`
        formula and gating-current formula are documented there.
    Nav1p6_RI2021_SC : Resurgent Nav1.6 sibling for the same
        stellate-cell model.

    Notes
    -----
    Ported from ``Nav1p1_RI21_SC.mod``. This class does not override
    ``__init__``: the constructor signature, the 13-state kinetics,
    the :math:`\phi` temperature scaling, the ``Oon``/``epsilon``/
    ``zgate``/``gunit``/``e0`` constants and the gating-current term
    in :meth:`current` are all inherited unchanged from
    :class:`Nav1p1_MA2025_BC`; only the citation and registration key
    differ.

    No import deviation is recorded for this mechanism in the
    bibliography's ``RI2021`` import-deviations tables.

    References
    ----------
    .. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
           contribution of resurgent sodium current to high-frequency
           firing in Purkinje neurons: an experimental and modeling
           study. The Journal of Neuroscience, 23(12), 4899-4912.
           doi:10.1523/JNEUROSCI.23-12-04899.2003
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

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav_MA2020_GrC")
class Nav_MA2020_GrC(Markov, IndependentIntegration):
    r"""Resurgent Nav sodium current, granule-cell parameterisation.

    A 13-state Raman & Bean (2001) [2]_-style resurgent sodium Markov
    scheme, refitted to the transient/persistent/resurgent granule-cell
    recordings and kinetic scheme of Magistretti et al. (2006) [1]_ and
    imported here for the granule-cell model of (Masoli et al., 2020)
    [3]_. Five closed states ``C1``-``C5`` form an activation ladder
    mirrored by five inactivated states ``I1``-``I5``; ``C5`` opens
    into ``O``, which can transition into a blocked state ``OB`` or
    into the shared deep-inactivated state ``I6`` (also reachable from
    ``I5``); ``I6`` is algebraically eliminated as ``dependent_state``.

    All transition rates share one temperature factor and a small set
    of voltage-dependent and constant primitives:

    .. math::

        \begin{aligned}
        \phi &= 3^{(T - 20)/10} \\
        \alpha(V) &= \phi\, A_\alpha\, e^{V/V_\alpha}, \quad
        \beta(V) = \phi\, A_\beta\, e^{-V/V_\beta}, \quad
        \theta(V) = \phi\, A_\theta\, e^{-V/V_\theta} \\
        \gamma &= \phi A_\gamma, \quad \delta = \phi A_\delta, \quad
        \varepsilon = \phi A_\varepsilon \\
        C_{on} &= \phi A_{Con}, \quad C_{off} = \phi A_{Coff}, \quad
        O_{on} = \phi A_{Oon}, \quad O_{off} = \phi A_{Ooff} \\
        a &= (O_{on}/C_{on})^{1/4}, \quad b = (O_{off}/C_{off})^{1/4}
        \end{aligned}

    with :math:`V` in mV (unitless argument to the exponentials) and
    :math:`T` the temperature in degrees Celsius. The closed and
    inactivated ladders share the same scaling weights
    :math:`n_1{=}5.422,\ n_2{=}3.279,\ n_3{=}1.83,\ n_4{=}0.738`:

    .. math::

        \begin{aligned}
        f_{0k} &= n_k\, \alpha(V), &
        b_{0k} &= n_{5-k}\, \beta(V), & k&=1,\dots,4 \\
        f_{1k} &= n_k\, \alpha(V)\, a, &
        b_{1k} &= n_{5-k}\, \beta(V)\, b, & k&=1,\dots,4 \\
        f_{ik} &= C_{on}\, a^{\,k-1}, &
        b_{ik} &= C_{off}\, b^{\,k-1}, & k&=1,\dots,5 \\
        f_{0O} &= \gamma, & b_{0O} &= \delta \\
        f_{ip} &= \varepsilon, & b_{ip} &= \theta \\
        f_{1n} &= \gamma, & b_{1n} &= \delta \\
        f_{in} &= O_{on}, & b_{in} &= O_{off}
        \end{aligned}

    where subscript ``0k``/``1k`` index the ``Ck<->Ck+1``/
    ``Ik<->Ik+1`` ladder steps, ``ik`` the ``Ck<->Ik`` cross-links,
    ``0O`` the ``C5<->O`` opening step, ``ip`` the ``O<->OB`` blocking
    step, and ``1n``/``in`` the ``I5<->I6``/``O<->I6`` deep-inactivation
    links.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving :math:`\phi`, default 32 degrees
        Celsius.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``13.0 mS/cm2``.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    NaFHF_MA2020_GrC : Same 13-state scheme with an additional
        slow-blocked branch (``L3``-``L6``) enabled.
    Nav1p6_MA2020_GoC : Same general resurgent-Markov family, fitted
        instead to Purkinje-cell kinetics with its own constants.

    Notes
    -----
    Ported from ``Nav_MA20_GrC.mod``, whose header attributes the
    scheme to "Raman 13 state model. Adapted from Magistretti et al,
    2006." This class does not subclass :class:`Nav1p6_MA2020_GoC`;
    it is an independent implementation with its own ``__init__`` and
    rate constants. :math:`\phi` is referenced to 20 degrees Celsius
    here, unlike the Nav1.6/Nav1.1 Golgi/Purkinje/basket/stellate
    family, which references 22 degrees Celsius.

    **Discrepancy between code and bibliography record.** This class
    ships ``ACon = 0.005`` and ``AOoff = 0.005``. The bibliography's
    cross-checked fingerprint for the ``MA2020`` granule sodium pair
    states that ``Nav_MA2020_GrC`` and ``NaFHF_MA2020_GrC`` share
    ``ACon = 0.025`` and ``AOoff = 0.002`` -- those values are correct
    for :class:`NaFHF_MA2020_GrC` (confirmed against its own code) but
    not for this class. The values documented above are read directly
    from this class's ``__init__`` and are the ones in effect at
    runtime.

    No import deviation is recorded for this mechanism in the
    bibliography's ``MA2020`` import-deviations tables.

    References
    ----------
    .. [1] Magistretti, J., Castelli, L., Forti, L., & D'Angelo, E.
           (2006). Kinetic and functional analysis of transient,
           persistent and resurgent sodium currents in rat cerebellar
           granule cells in situ: an electrophysiological and
           modelling study. The Journal of Physiology, 573(1), 83-106.
           doi:10.1113/jphysiol.2006.106682
    .. [2] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "OB", "fip", "bip"),
        ("I1", "I2", "f11", "b11"),
        ("I2", "I3", "f12", "b12"),
        ("I3", "I4", "f13", "b13"),
        ("I4", "I5", "f14", "b14"),
        ("C1", "I1", "fi1", "bi1"),
        ("C2", "I2", "fi2", "bi2"),
        ("C3", "I3", "fi3", "bi3"),
        ("C4", "I4", "fi4", "bi4"),
        ("C5", "I5", "fi5", "bi5"),
        ("O", "I6", "fin", "bin"),
        ("I5", "I6", "f1n", "b1n"),
    )
    dependent_state = "I6"

    def __init__(
        self,
        size: Size,
        temp: ArrayLike = u.celsius2kelvin(32.0),
        g_max: Initializer = 13.0 * (u.mS / u.cm**2),
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)

        self.Aalfa = 353.91
        self.Valfa = 13.99
        self.Abeta = 1.272
        self.Vbeta = 13.99
        self.Agamma = 150.0
        self.Adelta = 40.0
        self.Aepsilon = 1.75
        self.Ateta = 0.0201
        self.Vteta = 25.0
        self.ACon = 0.005
        self.ACoff = 0.5
        self.AOon = 0.75
        self.AOoff = 0.005
        self.n1 = 5.422
        self.n2 = 3.279
        self.n3 = 1.83
        self.n4 = 0.738

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    def init_state(self, V, Na: IonInfo, batch_size: int = None):
        super().init_state(V, Na, batch_size=batch_size)
        self.reset_steady_state(V, Na, batch_size=batch_size)

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)

    alfa = lambda self, V: self.phi * self.Aalfa * u.math.exp((V / u.mV) / self.Valfa)
    beta = lambda self, V: self.phi * self.Abeta * u.math.exp(-(V / u.mV) / self.Vbeta)
    teta = lambda self, V: self.phi * self.Ateta * u.math.exp(-(V / u.mV) / self.Vteta)
    gamma = lambda self, V: self.phi * self.Agamma
    delta = lambda self, V: self.phi * self.Adelta
    epsilon = lambda self, V: self.phi * self.Aepsilon
    Con = lambda self, V: self.phi * self.ACon
    Coff = lambda self, V: self.phi * self.ACoff
    Oon = lambda self, V: self.phi * self.AOon
    Ooff = lambda self, V: self.phi * self.AOoff
    a_factor = lambda self, V: (self.Oon(V) / self.Con(V)) ** 0.25
    b_factor = lambda self, V: (self.Ooff(V) / self.Coff(V)) ** 0.25

    f01 = lambda self, V: self.n1 * self.alfa(V)
    f02 = lambda self, V: self.n2 * self.alfa(V)
    f03 = lambda self, V: self.n3 * self.alfa(V)
    f04 = lambda self, V: self.n4 * self.alfa(V)
    f0O = lambda self, V: self.gamma(V)
    fip = lambda self, V: self.epsilon(V)
    f11 = lambda self, V: self.n1 * self.alfa(V) * self.a_factor(V)
    f12 = lambda self, V: self.n2 * self.alfa(V) * self.a_factor(V)
    f13 = lambda self, V: self.n3 * self.alfa(V) * self.a_factor(V)
    f14 = lambda self, V: self.n4 * self.alfa(V) * self.a_factor(V)
    f1n = lambda self, V: self.gamma(V)
    fi1 = lambda self, V: self.Con(V)
    fi2 = lambda self, V: self.Con(V) * self.a_factor(V)
    fi3 = lambda self, V: self.Con(V) * self.a_factor(V) ** 2
    fi4 = lambda self, V: self.Con(V) * self.a_factor(V) ** 3
    fi5 = lambda self, V: self.Con(V) * self.a_factor(V) ** 4
    fin = lambda self, V: self.Oon(V)

    b01 = lambda self, V: self.n4 * self.beta(V)
    b02 = lambda self, V: self.n3 * self.beta(V)
    b03 = lambda self, V: self.n2 * self.beta(V)
    b04 = lambda self, V: self.n1 * self.beta(V)
    b0O = lambda self, V: self.delta(V)
    bip = lambda self, V: self.teta(V)
    b11 = lambda self, V: self.n4 * self.beta(V) * self.b_factor(V)
    b12 = lambda self, V: self.n3 * self.beta(V) * self.b_factor(V)
    b13 = lambda self, V: self.n2 * self.beta(V) * self.b_factor(V)
    b14 = lambda self, V: self.n1 * self.beta(V) * self.b_factor(V)
    b1n = lambda self, V: self.delta(V)
    bi1 = lambda self, V: self.Coff(V)
    bi2 = lambda self, V: self.Coff(V) * self.b_factor(V)
    bi3 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 2
    bi4 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 3
    bi5 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 4
    bin = lambda self, V: self.Ooff(V)


@register_channel("NaFHF_MA2020_GrC")
class NaFHF_MA2020_GrC(Markov, IndependentIntegration):
    r"""Resurgent Nav sodium current with slow block, granule-cell model.

    The same 13-state Raman & Bean (2001) [2]_-style resurgent sodium
    Markov scheme documented on :class:`Nav_MA2020_GrC` -- fitted to
    the granule-cell kinetics of Magistretti et al. (2006) [1]_ and
    imported for the granule-cell model of (Masoli et al., 2020) [3]_
    -- extended with a second, slower blocked-state ladder
    ``L3``-``L6`` branching off ``C3``, ``C4``, ``C5`` and ``O``. It
    is the same mechanism as :class:`Nav_MA2020_GrC` with this
    slow-blocked branch enabled, not an independently derived model.

    The shared primitives (:math:`\phi`, :math:`\alpha`, :math:`\beta`,
    :math:`\theta`, :math:`\gamma`, :math:`\delta`, :math:`\varepsilon`,
    :math:`C_{on}`, :math:`C_{off}`, :math:`O_{on}`, :math:`O_{off}`,
    :math:`a`, :math:`b`) and the ``C1``-``C5``/``I1``-``I5``/``O``/
    ``OB``/``I6`` transition-rate formulas are exactly as given on
    :class:`Nav_MA2020_GrC`. The added ``L3``-``L6`` branch uses two
    further constant rate factors :math:`L_{on} = \phi A_{Lon}` and
    :math:`L_{off} = \phi A_{Loff}` and fixed scale factors
    :math:`c = 20.0`, :math:`d = 0.075`:

    .. math::

        \begin{aligned}
        f_{33} &= n_3\, \alpha(V)\, c, &
        b_{33} &= n_2\, \alpha(V)\, d \\
        f_{34} &= n_4\, \alpha(V)\, c, &
        b_{34} &= n_1\, \alpha(V)\, d \\
        f_{3n} &= \gamma, & b_{3n} &= \delta \\
        f_{l3} &= L_{on}, & b_{l3} &= L_{off} \\
        f_{l4} &= L_{on}\, c, & b_{l4} &= L_{off}\, d \\
        f_{l5} &= L_{on}\, c^2, & b_{l5} &= L_{off}\, d^2 \\
        f_{l6} &= L_{on}\, c^2, & b_{l6} &= L_{off}\, d^2
        \end{aligned}

    where ``f33``/``b33`` and ``f34``/``b34`` are the ``C3<->C4``/
    ``C4<->C5`` steps of the ``L``-branch's own closed-ladder mirror
    (``L3``/``L4``/``L5`` gate through the same alfa/beta primitives
    as ``C3``-``C5`` but scaled by ``c``/``d``), ``f3n``/``b3n`` is the
    ``L5<->L6`` step, and ``fl3``-``fl6``/``bl3``-``bl6`` are the
    ``C3<->L3``, ``C4<->L4``, ``C5<->L5`` and ``O<->L6`` cross-links.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    temp : array-like, optional
        Absolute temperature driving :math:`\phi`, default 32 degrees
        Celsius.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``13.0 mS/cm2``.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Override for :class:`Markov`'s default ODE solver.
    substeps : int, optional
        Override for :class:`Markov`'s default substep count.

    See Also
    --------
    Nav_MA2020_GrC : Same scheme without the slow-blocked branch; the
        shared primitives and ladder formulas are documented there.

    Notes
    -----
    Ported from ``NaFHF_MA20_GrC.mod``. Its own ``COMMENT`` block is
    empty, which is not a provenance gap: it inherits
    ``Nav_MA20_GrC.mod``'s "Based on Raman 13 state model. Adapted
    from Magistretti et al, 2006." attribution, per the bibliography.

    This class ships ``ACon = 0.025`` and ``AOoff = 0.002`` -- unlike
    :class:`Nav_MA2020_GrC`, these values match the bibliography's
    cross-checked fingerprint for the ``MA2020`` granule sodium pair
    exactly. See :class:`Nav_MA2020_GrC`'s Notes for the discrepancy
    recorded against its own ``ACon``/``AOoff`` values.

    No import deviation is recorded for this mechanism in the
    bibliography's ``MA2020`` import-deviations tables.

    References
    ----------
    .. [1] Magistretti, J., Castelli, L., Forti, L., & D'Angelo, E.
           (2006). Kinetic and functional analysis of transient,
           persistent and resurgent sodium currents in rat cerebellar
           granule cells in situ: an electrophysiological and
           modelling study. The Journal of Physiology, 573(1), 83-106.
           doi:10.1113/jphysiol.2006.106682
    .. [2] Raman, I. M., & Bean, B. P. (2001). Inactivation and
           recovery of sodium currents in cerebellar Purkinje neurons:
           evidence for two mechanisms. Biophysical Journal, 80(2),
           729-737.
           doi:10.1016/S0006-3495(01)76052-3
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "OB", "fip", "bip"),
        ("I1", "I2", "f11", "b11"),
        ("I2", "I3", "f12", "b12"),
        ("I3", "I4", "f13", "b13"),
        ("I4", "I5", "f14", "b14"),
        ("L3", "L4", "f33", "b33"),
        ("L4", "L5", "f34", "b34"),
        ("L5", "L6", "f3n", "b3n"),
        ("C1", "I1", "fi1", "bi1"),
        ("C2", "I2", "fi2", "bi2"),
        ("C3", "I3", "fi3", "bi3"),
        ("C4", "I4", "fi4", "bi4"),
        ("C5", "I5", "fi5", "bi5"),
        ("C3", "L3", "fl3", "bl3"),
        ("C4", "L4", "fl4", "bl4"),
        ("C5", "L5", "fl5", "bl5"),
        ("O", "L6", "fl6", "bl6"),
        ("O", "I6", "fin", "bin"),
        ("I5", "I6", "f1n", "b1n"),
    )
    dependent_state = "I6"

    def __init__(
        self,
        size: Size,
        temp: ArrayLike = u.celsius2kelvin(32.0),
        g_max: Initializer = 13.0 * (u.mS / u.cm**2),
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)

        self.Aalfa = 353.91
        self.Valfa = 13.99
        self.Abeta = 1.272
        self.Vbeta = 13.99
        self.Agamma = 150.0
        self.Adelta = 40.0
        self.Aepsilon = 1.75
        self.Ateta = 0.0201
        self.Vteta = 25.0
        self.ACon = 0.025
        self.ACoff = 0.5
        self.AOon = 0.75
        self.AOoff = 0.002
        self.n1 = 5.422
        self.n2 = 3.279
        self.n3 = 1.83
        self.n4 = 0.738
        self.ALon = 0.001
        self.ALoff = 0.5
        self.c = 20.0
        self.d = 0.075

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    def init_state(self, V, Na: IonInfo, batch_size: int = None):
        super().init_state(V, Na, batch_size=batch_size)
        self.reset_steady_state(V, Na, batch_size=batch_size)

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)

    alfa = lambda self, V: self.phi * self.Aalfa * u.math.exp((V / u.mV) / self.Valfa)
    beta = lambda self, V: self.phi * self.Abeta * u.math.exp(-(V / u.mV) / self.Vbeta)
    teta = lambda self, V: self.phi * self.Ateta * u.math.exp(-(V / u.mV) / self.Vteta)
    gamma = lambda self, V: self.phi * self.Agamma
    delta = lambda self, V: self.phi * self.Adelta
    epsilon = lambda self, V: self.phi * self.Aepsilon
    Con = lambda self, V: self.phi * self.ACon
    Coff = lambda self, V: self.phi * self.ACoff
    Oon = lambda self, V: self.phi * self.AOon
    Ooff = lambda self, V: self.phi * self.AOoff
    a_factor = lambda self, V: (self.Oon(V) / self.Con(V)) ** 0.25
    b_factor = lambda self, V: (self.Ooff(V) / self.Coff(V)) ** 0.25
    Lon = lambda self, V: self.phi * self.ALon
    Loff = lambda self, V: self.phi * self.ALoff

    f01 = lambda self, V: self.n1 * self.alfa(V)
    f02 = lambda self, V: self.n2 * self.alfa(V)
    f03 = lambda self, V: self.n3 * self.alfa(V)
    f04 = lambda self, V: self.n4 * self.alfa(V)
    f0O = lambda self, V: self.gamma(V)
    fip = lambda self, V: self.epsilon(V)
    f11 = lambda self, V: self.n1 * self.alfa(V) * self.a_factor(V)
    f12 = lambda self, V: self.n2 * self.alfa(V) * self.a_factor(V)
    f13 = lambda self, V: self.n3 * self.alfa(V) * self.a_factor(V)
    f14 = lambda self, V: self.n4 * self.alfa(V) * self.a_factor(V)
    f1n = lambda self, V: self.gamma(V)
    f33 = lambda self, V: self.n3 * self.alfa(V) * self.c
    f34 = lambda self, V: self.n4 * self.alfa(V) * self.c
    f3n = lambda self, V: self.gamma(V)
    fi1 = lambda self, V: self.Con(V)
    fi2 = lambda self, V: self.Con(V) * self.a_factor(V)
    fi3 = lambda self, V: self.Con(V) * self.a_factor(V) ** 2
    fi4 = lambda self, V: self.Con(V) * self.a_factor(V) ** 3
    fi5 = lambda self, V: self.Con(V) * self.a_factor(V) ** 4
    fin = lambda self, V: self.Oon(V)
    fl3 = lambda self, V: self.Lon(V)
    fl4 = lambda self, V: self.Lon(V) * self.c
    fl5 = lambda self, V: self.Lon(V) * self.c**2
    fl6 = lambda self, V: self.Lon(V) * self.c**2

    b01 = lambda self, V: self.n4 * self.beta(V)
    b02 = lambda self, V: self.n3 * self.beta(V)
    b03 = lambda self, V: self.n2 * self.beta(V)
    b04 = lambda self, V: self.n1 * self.beta(V)
    b0O = lambda self, V: self.delta(V)
    bip = lambda self, V: self.teta(V)
    b11 = lambda self, V: self.n4 * self.beta(V) * self.b_factor(V)
    b12 = lambda self, V: self.n3 * self.beta(V) * self.b_factor(V)
    b13 = lambda self, V: self.n2 * self.beta(V) * self.b_factor(V)
    b14 = lambda self, V: self.n1 * self.beta(V) * self.b_factor(V)
    b1n = lambda self, V: self.delta(V)
    b33 = lambda self, V: self.n2 * self.alfa(V) * self.d
    b34 = lambda self, V: self.n1 * self.alfa(V) * self.d
    b3n = lambda self, V: self.delta(V)
    bi1 = lambda self, V: self.Coff(V)
    bi2 = lambda self, V: self.Coff(V) * self.b_factor(V)
    bi3 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 2
    bi4 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 3
    bi5 = lambda self, V: self.Coff(V) * self.b_factor(V) ** 4
    bin = lambda self, V: self.Ooff(V)
    bl3 = lambda self, V: self.Loff(V)
    bl4 = lambda self, V: self.Loff(V) * self.d
    bl5 = lambda self, V: self.Loff(V) * self.d**2
    bl6 = lambda self, V: self.Loff(V) * self.d**2
