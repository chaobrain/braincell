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
import jax

from braincell._base import HHTypedNeuron, IonInfo
from braincell.channel._base import Gate, HH, OhmicHH, ghk_flux
from braincell.ion import Calcium
from braincell.mech import register_channel

_CAV3P1_NMODL_FARADAY = 9.6485e4 * (u.coulomb / u.mol)
_CAV3P1_NMODL_GAS_CONSTANT = 8.3145 * (u.joule / (u.kelvin * u.mol))
_CAV3P1_NMODL_TEMP_OFFSET = 0.04 * u.kelvin
_CAV3P3_NMODL_FARADAY = 96520.0 * (u.coulomb / u.mol)
_CAV3P3_NMODL_GAS_CONSTANT = 8.3134 * (u.joule / (u.kelvin * u.mol))
_CAV3P3_NMODL_TEMP_OFFSET = -0.01 * u.kelvin

__all__ = [
    "CaN_IS2008",
    "CaT_HM1992",
    "CaT_HP1992",
    "CaHT_HM1992",
    "CaHT_Re1993",
    "CaL_IS2008",
    "CaHVA_SU2015_DCN",
    "CaL_SU2015_DCN",
    "CaLVA_SU2015_DCN",
    "Cav1p2_MA2020_GoC",
    "Cav1p2_MA2025_BC",
    "Cav1p3_MA2020_GoC",
    "Cav1p3_MA2025_BC",
    "Cav3p1_MA2020_GoC",
    "Cav3p1_MA2020_GoC_Frozen",
    "Cav3p1_MA2024_PC",
    "Cav3p1_MA2024_PC_Frozen",
    "Cav3p1Test_PC24",
    "Cav2p1_MA2025_BC",
    "Cav2p1_MA2025_BC_Frozen",
    "Cav2p1_MA2024_PC",
    "Cav2p1_MA2024_PC_Frozen",
    "Cav2p1_RI2021_SC",
    "Cav2p1_RI2021_SC_Frozen",
    "Cav3p2_MA2025_BC",
    "Cav3p2_MA2024_PC",
    "Cav3p2_RI2021_SC",
    "Cav3p3_MA2024_PC",
    "Cav3p3_MA2024_PC_Frozen",
    "Cav3p3_RI2021_SC",
    "CaHVA_MA2020_GoC",
    "CaHVA_MA2020_GrC",
    "Cav2p3_MA2020_GoC",
    "Ca_ZH2019_IO",
    "Ca_ZH2019_IO_Frozen",
]


def _cav3p1_nmodl_ghk_flux(V, ci, co, z, temp):
    """GHK helper matching the constants and Kelvin conversion in Cav3p1 NMODL."""
    ghk_temp = temp + _CAV3P1_NMODL_TEMP_OFFSET
    zeta = (z * _CAV3P1_NMODL_FARADAY * V) / (_CAV3P1_NMODL_GAS_CONSTANT * ghk_temp)
    exp_term = u.math.exp(-zeta)
    numerator = ci - co * exp_term
    small_branch = (z * _CAV3P1_NMODL_FARADAY) * numerator * (1 + zeta / 2)
    regular_branch = (z * zeta * _CAV3P1_NMODL_FARADAY) * numerator / (1 - exp_term)
    return u.math.where(u.math.abs(1 - exp_term) < 1e-6, small_branch, regular_branch)


def _cav3p3_nmodl_ghk_flux(V, ci, co, z, temp):
    """GHK helper matching the constants and Kelvin conversion in Cav3p3 NMODL."""
    ghk_temp = temp + _CAV3P3_NMODL_TEMP_OFFSET
    w = (z * _CAV3P3_NMODL_FARADAY * V) / (_CAV3P3_NMODL_GAS_CONSTANT * ghk_temp)
    exp_term = u.math.exp(w)
    numerator = co - ci * exp_term
    small_branch = -z * _CAV3P3_NMODL_FARADAY * numerator * (1 - w / 2)
    regular_branch = -z * _CAV3P3_NMODL_FARADAY * numerator * w / (exp_term - 1)
    return u.math.where(u.math.abs(exp_term - 1) < 1e-6, small_branch, regular_branch)


def _freeze_quantity_gradient(value):
    return u.Quantity(
        jax.lax.stop_gradient(u.get_mantissa(value)),
        u.get_unit(value),
    )


@register_channel("CaN_IS2008")
class CaN_IS2008(HH):
    r"""Inoue & Strowbridge 2008 Ca-activated nonselective cation current.

    A calcium- and voltage-dependent non-selective cation current
    (:math:`I_{CAN}`) gated by a single activation variable and scaled
    by a saturating calcium-modulation factor:

    .. math::

        \begin{aligned}
        I_{CAN} &= \bar g \cdot M([Ca]_i) \cdot p \cdot (E - V) \\
        M([Ca]_i) &= \frac{[Ca]_i}{[Ca]_i + 0.2\ \mathrm{mM}} \\
        p_\infty &= \frac{1}{1 + \exp(-(V + 43) / 5.2)} \\
        \tau_p &= \frac{2.7}{\exp(-(V + 55) / 15) + \exp((V + 55) / 15)}
                  + 1.6
        \end{aligned}

    where :math:`V` is read in millivolts, :math:`[Ca]_i` is the
    intracellular calcium concentration, and :math:`p` relaxes toward
    :math:`p_\infty` with time constant :math:`\tau_p` (in
    milliseconds) scaled by
    :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    E : array-like or callable, optional
        Reversal potential. Defaults to ``10.0 mV``.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``1.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factor, default 36
        degrees Celsius.
    q10 : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``1.0``
        (no temperature scaling at the reference temperature).
    temp_ref : array-like, optional
        Reference temperature for the Q10 formula, default 36
        degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaL_IS2008 : Sibling channel from the same source paper, whose
        L-type attribution is contradicted by the paper's own text
        (see its Notes).

    Notes
    -----
    Ported verbatim from BrainPy's ``ICaN_IS2008``
    (``brainpy/dyn/channels/calcium.py``): every rate constant and
    the ``M([Ca]_i)`` modulation term match exactly. BrainPy in turn
    attributes the current to two sources, only one of which is
    Inoue & Strowbridge (2008): the
    ``M([Ca]_i) = [Ca]_i / ([Ca]_i + 0.2 mM)`` modulation is the
    calcium-activated non-selective cation form used by Destexhe,
    Contreras, Steriade, Sejnowski & Huguenard (1994), while the
    voltage dependence of ``p`` is attributed to Inoue & Strowbridge
    (2008).

    Inoue, T., & Strowbridge, B. W. (2008) models a calcium- and
    voltage-dependent nonselective cation current
    (:math:`I_{CAN}`) in an olfactory bulb granule-cell model, where
    it generates the calcium-dependent afterdepolarization central to
    the paper's thesis about persistent activity -- this substantially
    supports the attribution of this class to that paper. However,
    the paper's Methods defer the gating constants to supplementary
    materials that are not included with the PMC-deposited author
    manuscript and were not otherwise obtainable, and no ModelDB
    deposit exists for this paper. **No published source has been
    read that prints the** ``p_inf``/``tau_p`` **constants above**, so
    this docstring ships no ``References`` section: the current is
    documented as a ported implementation of substantially supported
    but not fully confirmed provenance, not as a transcription of a
    specific paper's printed equations.
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (Gate("p", q10="q10", temp_ref="temp_ref"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * u.mV,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 43.0) / 5.2))

    def f_p_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 2.7 / (u.math.exp(-(V + 55.0) / 15.0) + u.math.exp((V + 55.0) / 15.0)) + 1.6


@register_channel("CaT_HM1992")
class CaT_HM1992(OhmicHH):
    r"""Huguenard & McCormick 1992 low-threshold T-type calcium current.

    The thalamic relay-neuron low-threshold (T-type) calcium current
    :math:`I_T` of (Huguenard & McCormick, 1992) [1]_, with
    :math:`p^2 q` HH gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 59) / 6.2)} \\
        \tau_p &= \frac{1}
                  {\exp(-(V' + 132) / 16.7) + \exp((V' + 16.8) / 18.2)}
                  + 0.612 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 83) / 4)} \\
        \tau_q &= \begin{cases}
                  \exp((V' + 467) / 66.6) & V' < -80 \\
                  \exp(-(V' + 22) / 10.5) + 28 & V' \geq -80
                  \end{cases}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\tau_p`/:math:`\tau_q` (in milliseconds) are further scaled
    by :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``2.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default
        ``3.55``. This value could not be traced to the paper or to
        any reference NEURON implementation; treat it as a
        BrainCell/BrainPy default (see Notes).
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 24 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default
        ``3.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 24 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``-3.0 mV`` (see Notes).
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaHT_HM1992 : Same gating functions with ``V_sh`` moved to
        ``+25.0 mV``, relabelled as a high-threshold current; that
        relabelling has no source in this paper (see its Notes).
    CaT_HP1992 : Independently sourced T-type current for reticular
        nucleus neurons, with the same ``p^2 q`` gating shape but a
        different Boltzmann parameterisation.

    Notes
    -----
    Compared against ``ITGHK.mod`` (ModelDB accession 279),
    Destexhe's NEURON implementation of this paper, headed "Model of
    Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992". The
    mod file's own ``shift = 2 mV`` (screening charge at 2 mM
    external calcium) is folded into this class's Boltzmann
    midpoints only -- 57 to 59 in ``p_inf``, 81 to 83 in ``q_inf`` --
    **not** into the time-constant expressions: ``tau_p`` carries the
    mod file's bare 132/16.8 and ``tau_q`` its bare 467/22 with the
    branch at :math:`V' = -80`, i.e. exactly ``ITGHK.mod``'s numbers
    read at ``shift = 0``, not at its own shipped ``shift = 2 mV``.
    So the equations above match ``ITGHK.mod``'s ``tau_p``, piecewise
    ``tau_q`` and ``p^2 q`` gating exactly against a ``shift = 0``
    reading, while the steady-state midpoints match the mod file
    *with* the 2 mV shift folded in. Do not describe this class as
    reproducing the mod file "with the 2 mV screening-charge shift
    folded in" everywhere -- that is true of the midpoints and false
    of the time constants.

    A second, unrelated mod file, ``IT.mod`` (ModelDB accession
    3817), is *not* a source for any constant here despite modelling
    the same paper's data: its header reads "Model **based on the
    data of** Huguenard & McCormick... **and** Huguenard & Prince...",
    it shares only ``m_inf``/``h_inf`` with ``ITGHK.mod``, has no
    real ``tau_m`` (activation is taken at steady state), and its
    piecewise ``tau_h`` is commented out in favour of a different
    bi-exponential fit.

    This class applies a further ``V_sh = -3.0 mV`` on top of the
    already-folded 2 mV shift, so the shipped defaults sit 3 mV from
    ``ITGHK.mod``'s own defaults. This is a documented free
    parameter, not a citation error.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 59.0) / 6.2))

    def f_p_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp(-(V + 132.0) / 16.7) + u.math.exp((V + 16.8) / 18.2)) + 0.612

    def f_q_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 83.0) / 4.0))

    def f_q_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V >= -80.0,
            u.math.exp(-(V + 22.0) / 10.5) + 28.0,
            u.math.exp((V + 467.0) / 66.6),
        )


@register_channel("CaT_HP1992")
class CaT_HP1992(OhmicHH):
    r"""Huguenard & Prince 1992 T-type calcium current for reticular nucleus.

    The slowly inactivating transient (T-type) calcium current
    :math:`I_{Ts}` recorded in rat thalamic reticular nucleus (nRt)
    neurons by (Huguenard & Prince, 1992) [1]_, with :math:`p^2 q` HH
    gating and an ohmic driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 52) / 7.4)} \\
        \tau_p &= 3 + \frac{1}
                  {\exp((V' + 27) / 10) + \exp(-(V' + 102) / 15)} \\
        q_\infty &= \frac{1}{1 + \exp((V' + 80) / 5)} \\
        \tau_q &= 85 + \frac{1}
                  {\exp((V' + 48) / 4) + \exp(-(V' + 407) / 50)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\tau_p`/:math:`\tau_q` (in milliseconds) are further scaled
    by :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``1.75 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``5.0``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 24 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default
        ``3.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 24 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``-3.0 mV`` (see Notes).
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaT_HM1992 : Independently sourced low-threshold T-type current
        for thalamic relay neurons, with the same ``p^2 q`` gating
        shape but a different Boltzmann parameterisation.

    Notes
    -----
    Compared against ``IT2.mod`` (ModelDB accession 3670), whose
    header reads: "The kinetics is described by standard equations
    (NOT GHK) using a m2h format, according to the voltage-clamp data
    (whole cell patch clamp) of Huguenard & Prince, J Neurosci. 12:
    3804-3817, 1992", with "Q10 changed to 5 and 3". Unlike
    :class:`CaT_HM1992`, the mod file's ``shift = 2 mV`` (screening
    charge for external calcium = 2 mM) is folded into **every**
    constant here -- both the Boltzmann midpoints and the
    time-constant offsets -- so all six mod-file numbers (50, 78, 25,
    100, 46, 405) shift uniformly to the 52, 80, 27, 102, 48, 407
    used above. ``q10_p = 5.0`` and ``q10_q = 3.0`` at
    ``temp_ref = 24 degC`` match the mod file's
    ``phi_m = 5^((celsius-24)/10)`` and ``phi_h = 3^((celsius-24)/10)``
    exactly.

    This class applies a further ``V_sh = -3.0 mV`` on top of the
    already-folded 2 mV shift, so the shipped defaults sit 3 mV
    depolarized relative to ``IT2.mod``'s own defaults. This is a
    documented free parameter, not a citation error.
    ``g_max = 1.75 mS/cm2`` matches ``IT2.mod``'s
    ``gcabar = 0.00175 mho/cm2``.

    References
    ----------
    .. [1] Huguenard, J. R., & Prince, D. A. (1992). A novel T-type
           current underlies prolonged Ca2+-dependent burst firing in
           GABAergic neurons of rat thalamic reticular nucleus. The
           Journal of Neuroscience, 12(10), 3804-3817.
           doi:10.1523/JNEUROSCI.12-10-03804.1992
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.75 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 52.0) / 7.4))

    def f_p_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 3.0 + 1.0 / (u.math.exp((V + 27.0) / 10.0) + u.math.exp(-(V + 102.0) / 15.0))

    def f_q_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 5.0))

    def f_q_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 85.0 + 1.0 / (u.math.exp((V + 48.0) / 4.0) + u.math.exp(-(V + 407.0) / 50.0))


@register_channel("CaHT_HM1992")
class CaHT_HM1992(OhmicHH):
    r"""Depolarized-shift variant of the Huguenard & McCormick 1992 T current.

    :math:`p^2 q` HH gating with an ohmic driving force, using the
    same rate functions as :class:`CaT_HM1992` but with the threshold
    shift moved from ``-3.0 mV`` to ``+25.0 mV`` (see Notes for why
    this is not a citation for a genuine high-threshold current):

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 59) / 6.2)} \\
        \tau_p &= \frac{1}
                  {\exp(-(V' + 132) / 16.7) + \exp((V' + 16.8) / 18.2)}
                  + 0.612 \\
        q_\infty &= \frac{1}{1 + \exp((V' + 83) / 4)} \\
        \tau_q &= \begin{cases}
                  \exp((V' + 467) / 66.6) & V' < -80 \\
                  \exp(-(V' + 22) / 10.5) + 28 & V' \geq -80
                  \end{cases}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}` and
    :math:`\tau_p`/:math:`\tau_q` (in milliseconds) are further scaled
    by :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``2.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default
        ``3.55``, inherited unchanged from :class:`CaT_HM1992` (see
        that class's Notes on this value's provenance).
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 24 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default
        ``3.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 24 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``+25.0 mV`` -- see Notes; this is the only numeric
        difference from :class:`CaT_HM1992`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaT_HM1992 : The unshifted low-threshold T current this class is
        character-for-character derived from (``V_sh = -3.0 mV``);
        see its Notes for how the source paper's 2 mV screening
        shift is folded into these same rate functions.
    CaHT_Re1993 : Independently sourced high-threshold calcium
        current with a genuine high-voltage-activated attribution.

    Notes
    -----
    **This class does not implement a current described in
    (Huguenard & McCormick, 1992).** That paper [1]_ models exactly
    four currents -- a *low*-threshold T-type calcium current, two
    potassium currents and a hyperpolarization-activated current --
    and contains no high-threshold or high-voltage-activated calcium
    current at all. Reading the code confirms this class's rate
    functions are character-for-character identical to
    :class:`CaT_HM1992`'s (same 59, 6.2, 0.612, 132, 16.7, 16.8,
    18.2, 83, 4.0, 467, 66.6, 22, 10.5, 28 constants, the same
    :math:`V' = -80` branch point, the same ``p^2 q`` gating and the
    same ``q10_p = 3.55`` / ``q10_q = 3.0`` at 24 degC defaults). The
    only difference is ``V_sh``: ``+25.0 mV`` here versus ``-3.0 mV``
    in :class:`CaT_HM1992`. This class is the paper's low-threshold T
    current translated 28 mV depolarized and relabelled
    "high-threshold" -- a BrainCell/BrainPy-derived variant, not a
    current the cited paper reports. The ``+25 mV`` shift itself has
    no traceable source in the paper.

    The citation below is included because it correctly identifies
    the origin of the *gating kinetics* (they are exactly Huguenard &
    McCormick's T-current rate functions -- see :class:`CaT_HM1992`
    for the full derivation and the 2 mV screening-shift caveat that
    also applies here), not because the paper describes a
    high-threshold current under this name. A genuine high-threshold
    thalamic calcium current, :math:`I_L`, is described in the
    companion paper McCormick, D. A., & Huguenard, J. R. (1992), "A
    model of the electrophysiological properties of thalamocortical
    relay neurons", Journal of Neurophysiology, 68(4), 1384-1400,
    doi:10.1152/jn.1992.68.4.1384 -- but this class's gating functions
    do not match that current's kinetics either, so that paper is
    named here only as a lead for future re-derivation, not cited.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 59.0) / 6.2))

    def f_p_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (u.math.exp(-(V + 132.0) / 16.7) + u.math.exp((V + 16.8) / 18.2)) + 0.612

    def f_q_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 83.0) / 4.0))

    def f_q_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return u.math.where(
            V >= -80.0,
            u.math.exp(-(V + 22.0) / 10.5) + 28.0,
            u.math.exp((V + 467.0) / 66.6),
        )


@register_channel("CaHT_Re1993")
class CaHT_Re1993(OhmicHH):
    r"""Reuveni 1993 high-threshold calcium current.

    The high-voltage-activated (HVA) calcium current underlying the
    calcium plateau spike in neocortical pyramidal cells, from
    (Reuveni et al., 1993) [1]_, with :math:`p^2 q` HH gating (an
    :math:`m^2 h` scheme in the source mod file's naming) and an
    ohmic driving force:

    .. math::

        \begin{aligned}
        \alpha_p &= \frac{0.055 (V'' - 27)}{\exp((V'' - 27) / 3.8) - 1} \\
        \beta_p &= 0.94 \exp((V'' - 75) / 17) \\
        \alpha_q &= 0.000457 \exp((V'' - 13) / 50) \\
        \beta_q &= \frac{0.0065}{\exp((V'' - 15) / 28) + 1}
        \end{aligned}

    where :math:`V'' = (V_{sh} - V) / \mathrm{mV}` is the negated-
    voltage form the source mod file uses inline, so with the default
    ``V_sh = 0.0 mV`` this reduces to :math:`V'' = -V / \mathrm{mV}`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``1.0 mS/cm2``.
    temp : array-like, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``2.3``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 23 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default
        ``2.3``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 23 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift entering the negated-voltage form above,
        default ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaHT_HM1992 : Nominally also a high-threshold current, but
        actually a relabelled low-threshold T current with no
        genuine high-threshold attribution (see its Notes).

    Notes
    -----
    Compared against ``ca.mod`` (ModelDB accession 2488), headed "HVA
    Ca current / Based on Reuveni, Friedman, Amitai and Gutnick
    (1993) / J. Neurosci. 13:4609-4621" (implementation by Zach
    Mainen, Salk Institute, 1994). Every rate constant above matches
    the mod file's ``alpha_m``/``beta_m``/``alpha_h``/``beta_h``
    term for term, and the temperature parameters match exactly:
    ``ca.mod`` uses ``temp = 23 degC, q10 = 2.3``; this class
    defaults to ``q10_p = q10_q = 2.3`` at ``temp_ref = 23 degC``.
    Gating is :math:`m^2 h` in both.

    References
    ----------
    .. [1] Reuveni, I., Friedman, A., Amitai, Y., & Gutnick, M. J.
           (1993). Stepwise repolarization from Ca2+ plateaus in
           neocortical pyramidal cells: evidence for nonhomogeneous
           distribution of HVA Ca2+ channels in dendrites. The
           Journal of Neuroscience, 13(11), 4609-4621.
           doi:10.1523/JNEUROSCI.13-11-04609.1993
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm**2),
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

    def f_p_alpha(self, V, Ca: IonInfo):
        temp = (-V + self.V_sh).to_decimal(u.mV)
        delta = -27.0 + temp
        return 0.055 * delta / (u.math.exp(delta / 3.8) - 1.0)

    def f_p_beta(self, V, Ca: IonInfo):
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.94 * u.math.exp((-75.0 + temp) / 17.0)

    def f_q_alpha(self, V, Ca: IonInfo):
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.000457 * u.math.exp((-13.0 + temp) / 50.0)

    def f_q_beta(self, V, Ca: IonInfo):
        temp = (-V + self.V_sh).to_decimal(u.mV)
        return 0.0065 / (u.math.exp((-15.0 + temp) / 28.0) + 1.0)


@register_channel("CaL_IS2008")
class CaL_IS2008(OhmicHH):
    r"""Inoue & Strowbridge 2008 L-type calcium current.

    A voltage-gated calcium current with :math:`p^2 q` HH gating and
    an ohmic driving force:

    .. math::

        \begin{aligned}
        p_\infty &= \frac{1}{1 + \exp(-(V' + 10) / 4)} \\
        \tau_p &= 0.4 + \frac{0.7}
                  {\exp(-(V' + 5) / 15) + \exp((V' + 5) / 15)} \\
        q_\infty &= \frac{1}{1 + \exp((V' + 25) / 2)} \\
        \tau_q &= 300 + \frac{100}
                  {\exp((V' + 40) / 9.5) + \exp(-(V' + 40) / 9.5)}
        \end{aligned}

    where :math:`V' = (V - V_{sh}) / \mathrm{mV}`, gating is
    :math:`p^2 q`, and :math:`\tau_p`/:math:`\tau_q` (in milliseconds)
    are further scaled by
    :meth:`~braincell.channel._base.HH.gate_phi`.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``1.0 mS/cm2``.
    temp : array-like or callable, optional
        Absolute temperature driving the Q10 factors, default 36
        degrees Celsius.
    q10_p : array-like or callable, optional
        Q10 scaling factor for the activation gate, default ``3.55``.
    temp_ref_p : array-like, optional
        Reference temperature for ``q10_p``, default 24 degrees
        Celsius.
    q10_q : array-like or callable, optional
        Q10 scaling factor for the inactivation gate, default
        ``3.0``.
    temp_ref_q : array-like, optional
        Reference temperature for ``q10_q``, default 24 degrees
        Celsius.
    V_sh : array-like or callable, optional
        Threshold shift applied to both gates' rates, default
        ``0.0 mV``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaN_IS2008 : Sibling channel from the same source paper, whose
        calcium-activated non-selective cation attribution is
        substantially, though not fully, supported (see its Notes).

    Notes
    -----
    Ported verbatim from BrainPy's ``ICaL_IS2008``
    (``brainpy/dyn/channels/calcium.py``): every rate constant and
    the ``p^2 q`` gating scheme match exactly, and BrainPy attributes
    this current to Inoue & Strowbridge (2008) alone.

    **This attribution is contradicted by the cited paper's own
    text.** Inoue, T., & Strowbridge, B. W. (2008) models exactly two
    voltage-gated calcium currents in its olfactory bulb granule-cell
    simulations: "a low-threshold (T-type) Ca current, and a
    high-threshold (P/N-type) Ca current." The strings "L-type",
    "L type" and "ICaL" do not occur anywhere in the article text --
    the paper has **no L-type calcium current** for this class to be
    a port of. Two further, independent signs point the same way:
    this class's ``q10_p = 3.55`` / ``q10_q = 3.0`` at 24 degrees
    Celsius defaults are byte-identical to :class:`CaT_HM1992`'s
    temperature defaults (24 degC is the Huguenard & McCormick
    reference temperature, not an olfactory-bulb one), and no ModelDB
    deposit exists for this paper to compare constants against. The
    paper's Methods also defer whatever gating constants it does
    report to supplementary materials not included with the
    PMC-deposited author manuscript, so even the P/N-type or T-type
    currents it does contain cannot be checked against this class
    from the text alone.

    Because of this, either this class implements the paper's
    high-threshold P/N-type current under the wrong name, or its true
    source is a different paper entirely; both possibilities remain
    open. This docstring ships no ``References`` section: printing a
    confident citation to Inoue & Strowbridge (2008) for an *L-type*
    current would misattribute a current that paper does not contain.
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2, q10="q10_p", temp_ref="temp_ref_p"),
        Gate("q", q10="q10_q", temp_ref="temp_ref_q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm**2),
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

    def f_p_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 10.0) / 4.0))

    def f_p_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.4 + 0.7 / (u.math.exp(-(V + 5.0) / 15.0) + u.math.exp((V + 5.0) / 15.0))

    def f_q_inf(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 25.0) / 2.0))

    def f_q_tau(self, V, Ca: IonInfo):
        V = (V - self.V_sh).to_decimal(u.mV)
        return 300.0 + 100.0 / (u.math.exp((V + 40.0) / 9.5) + u.math.exp(-(V + 40.0) / 9.5))


@register_channel("CaHVA_SU2015_DCN")
class CaHVA_SU2015_DCN(HH):
    r"""HVA calcium current of the DCN model (Sudhakar 2015).

    A GHK-driven, high-voltage-activated (HVA) calcium current with a
    single :math:`m^3` activation gate, used for the deep cerebellar
    nucleus (DCN) neuron model of (Sudhakar et al., 2015) [2]_:

    .. math::

        \begin{aligned}
        I_{CaHVA} &= -P \cdot m^3 \cdot \Phi(V, [Ca]_i, [Ca]_o, T) \\
        m_\infty &= \frac{1}{1 + \exp(-(V + 34.5) / -9.0)} \\
        \tau_m &= \frac{1}{\dfrac{31.746}{\exp((V - 5) / -13.89) + 1}
                  + \dfrac{3.97 \times 10^{-4} (V + 8.9)}
                  {\exp((V + 8.9) / 5) - 1}} \Big/ q_{\Delta t}
        \end{aligned}

    where :math:`\Phi` is the constant-field GHK flux (see
    :func:`~braincell.channel._base.ghk_flux`) evaluated with this
    class's own inline Faraday/gas-constant literals rather than the
    shared helper (see Notes), and :math:`P` is the permeability
    parameter.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    perm : array-like or callable, optional
        Permeability entering the GHK flux term. Defaults to
        ``7.5e-6 cm/s``.
    temp : array-like, optional
        Absolute temperature entering the GHK flux term and the
        activation time constant. Defaults to 36 degrees Celsius.
    qdeltat : array-like or callable, optional
        Divisor applied to :math:`\tau_m`; a NEURON-style ``Q10``-free
        rate scale, not a :class:`~braincell.channel._base.Gate`
        ``phi`` factor. Defaults to ``1.0``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaLVA_SU2015_DCN : Sibling DCN calcium current using the same
        inline GHK constants and permeability parameterisation, with
        low-voltage-activated (:math:`m^2 h`) gating instead.
    CaL_SU2015_DCN : Sibling DCN calcium current with the same
        low-voltage-activated gating shape as
        :class:`CaLVA_SU2015_DCN`, but driven ohmically against a
        fixed reversal potential rather than through GHK.
    braincell.channel._base.ghk_flux : Shared GHK flux helper; this
        class does not call it directly (see Notes).

    Notes
    -----
    Ported from ``DCN/channel/CaHVA_SU15_DCN.mod``, whose ``TITLE``
    reads "High voltage activated calcium current (CaHVA) of deep
    cerebellar nucleus (DCN) neuron". The origin of the DCN kinetics
    is the GENESIS model of Steuber, Schultheiss, Silver, De Schutter
    & Jaeger (2011) [1]_, translated from GENESIS to NEURON by
    Luthman, Hoebeek, Maex, Davey, Adams, De Zeeuw & Steuber (2011)
    and reused, without modification credit, in Sudhakar et al.
    (2015) [2]_. The string "CaHVA" does not occur in the Sudhakar et
    al. (2015) article text; this docstring records only that the
    mechanism is part of the model published as [2]_, not that the
    paper names or describes it, and it does not claim that either
    paper prints the ``m_inf``/``tau_m`` constants above.

    ``current()`` evaluates the GHK constant-field equation inline
    with the mod file's own hard-coded literals
    (``4.47814e6``, ``-23.20764929``) rather than calling
    :func:`~braincell.channel._base.ghk_flux`, matching the pattern
    used by the module-level ``_cav3p1_nmodl_ghk_flux`` /
    ``_cav3p3_nmodl_ghk_flux`` helpers elsewhere in this file.
    NEURON's raw ``ica`` for this mechanism is outward-positive; the
    sign is flipped in ``current()`` to match BrainCell's repo-wide
    inward-positive convention.

    The original mechanism's ``TABLE`` directive tabulated ``minf``
    and ``taum`` over ``[-150, 100] mV`` (plus a ``DEPEND T`` table);
    BrainCell removes the table and evaluates both expressions
    per-call instead.

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
    root_type = Calcium
    gates = (Gate("m", power=3),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        perm: Union[brainstate.typing.ArrayLike, Callable] = 7.5e-6 * (u.cm / u.second),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        qdeltat: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.perm = braintools.init.param(perm, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.qdeltat = braintools.init.param(qdeltat, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        v_mV = V.to_decimal(u.mV)
        temp = self.temp.to_decimal(u.kelvin)
        ci = Ca.Ci.to_decimal(u.mM)
        co = Ca.Co.to_decimal(u.mM)
        perm = self.perm.to_decimal(u.cm / u.second)
        A = u.math.exp(-23.20764929 * v_mV / temp)
        drive = (4.47814e6 * v_mV / temp) * ((ci / 1000.0) - (co / 1000.0) * A) / (1.0 - A)
        current_value = perm * self.m.value**3 * drive
        # NEURON's raw ``ica`` is outward-positive, so inward calcium entry
        # appears as a negative current. BrainCell channel currents use the
        # repo-wide inward-positive convention, so imported mechanisms flip
        # the sign here and comparisons should use ``-neuron_ica``.
        return -current_value * (u.mA / (u.cm**2))

    def f_m_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 34.5) / -9.0))

    def f_m_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        activation = 31.746 / (u.math.exp((V - 5.0) / -13.89) + 1.0)
        correction = 3.97e-4 * (V + 8.9) / (u.math.exp((V + 8.9) / 5.0) - 1.0)
        return 1.0 / (activation + correction) / self.qdeltat


@register_channel("CaL_SU2015_DCN")
class CaL_SU2015_DCN(OhmicHH):
    r"""Ohmic-drive LVA calcium current of the DCN model (Sudhakar 2015).

    An ohmically-driven calcium current with :math:`m^2 h` HH gating
    against a fixed reversal potential, used for the deep cerebellar
    nucleus (DCN) neuron model of (Sudhakar et al., 2015) [2]_:

    .. math::

        \begin{aligned}
        I_{CaL} &= g_{\mathrm{max}} \, m^2 h \, (E - V) \\
        m_\infty &= \frac{1}{1 + \exp(-(V + 56) / -6.2)} \\
        \tau_m &= \left(\frac{0.333}
                  {\exp((V + 131) / -16.7) + \exp((V + 15.8) / 18.2)}
                  + 0.204\right) \Big/ q_{\Delta t} \\
        h_\infty &= \frac{1}{1 + \exp((V + 80) / 4)} \\
        \tau_h &= \frac{1}{q_{\Delta t}} \times \begin{cases}
                  0.333 \exp((V + 466) / 66) & V < -81 \\
                  0.333 \exp((V + 21) / -10.5) + 9.32 & V \geq -81
                  \end{cases}
        \end{aligned}

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.01 mS/cm2``.
    E : array-like or callable, optional
        Fixed reversal potential used in place of an ion-derived
        value (see Notes). Defaults to ``139.0 mV``.
    qdeltat : array-like or callable, optional
        Divisor applied to :math:`\tau_m` and :math:`\tau_h`; a
        NEURON-style ``Q10``-free rate scale, not a
        :class:`~braincell.channel._base.Gate` ``phi`` factor.
        Defaults to ``1.0``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaLVA_SU2015_DCN : Sibling DCN calcium current with the same
        :math:`m^2 h` gating shape and identical rate constants (see
        Notes), driven through GHK against ion concentrations rather
        than ohmically against a fixed ``E``.
    CaHVA_SU2015_DCN : Sibling DCN calcium current with
        high-voltage-activated (:math:`m^3`) gating instead.

    Notes
    -----
    Ported from ``DCN/channel/CaL_SU15_DCN.mod``. **Its own ``TITLE``
    reads "LVA calcium current (CaLVA) of deep cerebellar nucleus
    (DCN) neuron"** -- i.e. the mod file named ``CaL_SU15_DCN.mod``
    describes itself as an LVA / "CaLVA" current, not an L-type
    current, despite the ``CaL`` symbol name. This class's rate
    functions are in fact algebraically identical to
    :class:`CaLVA_SU2015_DCN`'s: both use
    ``m_inf = 1/(1 + exp((V+56)/-6.2))`` and the same
    :math:`\tau_m`/:math:`h_\infty`/:math:`\tau_h` expressions above.
    The only structural difference between the two classes is the
    current law -- this class overrides
    :meth:`~braincell.channel._base.OhmicHH.reversal_potential` to
    return the fixed ``E`` parameter and derives its driving force
    ohmically, while :class:`CaLVA_SU2015_DCN` derives its driving
    force from a GHK flux against explicit ion concentrations. The
    default ``E = 139.0 mV`` was not independently verified against
    the mod file's own ``PARAMETER`` block for this task; it is
    documented here as a shipped BrainCell default, not attributed to
    a specific value printed in Sudhakar et al. (2015) [2]_.

    The origin of the DCN kinetics is the GENESIS model of Steuber,
    Schultheiss, Silver, De Schutter & Jaeger (2011) [1]_, translated
    from GENESIS to NEURON by Luthman, Hoebeek, Maex, Davey, Adams,
    De Zeeuw & Steuber (2011) and reused, without modification
    credit, in Sudhakar et al. (2015) [2]_. The string "CaL" does not
    occur in the Sudhakar et al. (2015) article text; this docstring
    records only that the mechanism is part of the model published as
    [2]_, not that the paper names or describes it, and it does not
    claim that either paper prints the constants above.

    The original mechanism's ``TABLE`` directive tabulated ``minf``,
    ``taum``, ``hinf`` and ``tauh`` over ``[-150, 100] mV`` (with no
    ``DEPEND T`` table, unlike :class:`CaHVA_SU2015_DCN` and
    :class:`CaLVA_SU2015_DCN`); BrainCell removes the table and
    evaluates all four expressions per-call instead.

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
    gates = (
        Gate("m", power=2),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        E: Union[brainstate.typing.ArrayLike, Callable] = 139.0 * u.mV,
        qdeltat: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.qdeltat = braintools.init.param(qdeltat, self.varshape, allow_none=False)

    def reversal_potential(self, V, *ions):
        return self.E

    def f_m_inf(self, V):
        return self._m_inf_formula(V.to_decimal(u.mV))

    def f_m_tau(self, V):
        return self._m_tau_formula(V.to_decimal(u.mV)) / self.qdeltat

    def f_h_inf(self, V):
        return self._h_inf_formula(V.to_decimal(u.mV))

    def f_h_tau(self, V):
        return self._h_tau_formula(V.to_decimal(u.mV)) / self.qdeltat

    def _m_inf_formula(self, V):
        return 1.0 / (1.0 + u.math.exp((V + 56.0) / -6.2))

    def _m_tau_formula(self, V):
        return 0.333 / (u.math.exp((V + 131.0) / -16.7) + u.math.exp((V + 15.8) / 18.2)) + 0.204

    def _h_inf_formula(self, V):
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 4.0))

    def _h_tau_formula(self, V):
        return u.math.where(
            V < -81.0,
            0.333 * u.math.exp((V + 466.0) / 66.0),
            0.333 * u.math.exp((V + 21.0) / -10.5) + 9.32,
        )


@register_channel("CaLVA_SU2015_DCN")
class CaLVA_SU2015_DCN(HH):
    r"""GHK-drive LVA calcium current of the DCN model (Sudhakar 2015).

    A GHK-driven, low-voltage-activated (LVA) calcium current with
    :math:`m^2 h` gating, used for the deep cerebellar nucleus (DCN)
    neuron model of (Sudhakar et al., 2015) [2]_:

    .. math::

        \begin{aligned}
        I_{CaLVA} &= -P \cdot m^2 h \cdot \Phi(V, [Ca]_i, [Ca]_o, T) \\
        m_\infty &= \frac{1}{1 + \exp(-(V + 56) / -6.2)} \\
        \tau_m &= \left(\frac{0.333}
                  {\exp((V + 131) / -16.7) + \exp((V + 15.8) / 18.2)}
                  + 0.204\right) \Big/ q_{\Delta t} \\
        h_\infty &= \frac{1}{1 + \exp((V + 80) / 4)} \\
        \tau_h &= \frac{1}{q_{\Delta t}} \times \begin{cases}
                  0.333 \exp((V + 466) / 66) & V < -81 \\
                  0.333 \exp((V + 21) / -10.5) + 9.32 & V \geq -81
                  \end{cases}
        \end{aligned}

    where :math:`\Phi` is the constant-field GHK flux (see
    :func:`~braincell.channel._base.ghk_flux`) evaluated with this
    class's own inline Faraday/gas-constant literals rather than the
    shared helper (see Notes), and :math:`P` is the permeability
    parameter.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    perm : array-like or callable, optional
        Permeability entering the GHK flux term. Defaults to
        ``1.0 cm/s``.
    temp : array-like, optional
        Absolute temperature entering the GHK flux term and the
        activation/inactivation time constants. Defaults to 36
        degrees Celsius.
    qdeltat : array-like or callable, optional
        Divisor applied to :math:`\tau_m` and :math:`\tau_h`; a
        NEURON-style ``Q10``-free rate scale, not a
        :class:`~braincell.channel._base.Gate` ``phi`` factor.
        Defaults to ``1.0``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaL_SU2015_DCN : Sibling DCN calcium current with the same
        :math:`m^2 h` gating shape and identical rate constants (see
        Notes), driven ohmically against a fixed reversal potential
        rather than through GHK.
    CaHVA_SU2015_DCN : Sibling DCN calcium current with
        high-voltage-activated (:math:`m^3`) gating instead.
    braincell.ion.CdpLVA_SU2015_DCN : Dedicated LVA-current-driven
        calcium pool BrainCell ships for this current; see Notes for
        how it relates to :attr:`root_type`.
    braincell.channel._base.ghk_flux : Shared GHK flux helper; this
        class does not call it directly (see Notes).

    Notes
    -----
    Ported from ``DCN/channel/CaLVA_SU15_DCN.mod``, whose ``TITLE``
    reads "Low voltage activated calcium current (CaLVA) of deep
    cerebellar nucleus (DCN) neuron". This class's rate functions are
    algebraically identical to :class:`CaL_SU2015_DCN`'s, whose own
    mod file (``CaL_SU15_DCN.mod``) is *also* titled "LVA calcium
    current (CaLVA) of deep cerebellar nucleus (DCN) neuron" despite
    its different ``CaL`` symbol name -- the two BrainCell classes
    are the same LVA current family exposed through two different
    current laws (see :class:`CaL_SU2015_DCN`'s Notes).

    ``root_type`` here is the generic :class:`~braincell.ion.Calcium`
    base, identical to :class:`CaHVA_SU2015_DCN`'s -- nothing in this
    channel class itself restricts it to a particular ion pool.
    BrainCell separately ships
    :class:`~braincell.ion.CdpLVA_SU2015_DCN`, a dedicated
    LVA-current-driven calcium pool mirroring the imported NMODL's
    separate ``cali``/``cal`` pool (as opposed to
    ``CdpHVA_SU2015_DCN``'s ``cai``, which the original NEURON
    mechanism used to let the LVA and HVA currents drive
    independently trackable calcium pools). Preserving that
    separation in a BrainCell model is a compositional choice made
    when attaching this channel to a specific ion instance, not a
    guarantee enforced by this class's ``root_type``.

    ``current()`` evaluates the GHK constant-field equation inline
    with the mod file's own hard-coded literals
    (``4.47814e6``, ``-23.20764929``) rather than calling
    :func:`~braincell.channel._base.ghk_flux`, matching the pattern
    used by the module-level ``_cav3p1_nmodl_ghk_flux`` /
    ``_cav3p3_nmodl_ghk_flux`` helpers elsewhere in this file.
    NEURON's raw ``ical`` for this mechanism is outward-positive; the
    sign is flipped in ``current()`` to match BrainCell's repo-wide
    inward-positive convention.

    The origin of the DCN kinetics is the GENESIS model of Steuber,
    Schultheiss, Silver, De Schutter & Jaeger (2011) [1]_, translated
    from GENESIS to NEURON by Luthman, Hoebeek, Maex, Davey, Adams,
    De Zeeuw & Steuber (2011) and reused, without modification
    credit, in Sudhakar et al. (2015) [2]_. The string "CaLVA" does
    occur in the Sudhakar et al. (2015) article text, but this
    docstring does not claim that either paper prints the
    ``m_inf``/``tau_m``/``h_inf``/``tau_h`` constants above.

    The original mechanism's ``TABLE`` directive tabulated ``minf``,
    ``taum``, ``hinf`` and ``tauh`` over ``[-150, 100] mV`` (plus a
    ``DEPEND T`` table); BrainCell removes the table and evaluates
    all four expressions per-call instead.

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
    root_type = Calcium
    gates = (
        Gate("m", power=2),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        perm: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.cm / u.second),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        qdeltat: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.perm = braintools.init.param(perm, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.qdeltat = braintools.init.param(qdeltat, self.varshape, allow_none=False)

    def current(self, V, Ca: IonInfo):
        v_mV = V.to_decimal(u.mV)
        temp = self.temp.to_decimal(u.kelvin)
        ci = Ca.Ci.to_decimal(u.mM)
        co = Ca.Co.to_decimal(u.mM)
        perm = self.perm.to_decimal(u.cm / u.second)
        A = u.math.exp(-23.20764929 * v_mV / temp)
        drive = (4.47814e6 * v_mV / temp) * ((ci / 1000.0) - (co / 1000.0) * A) / (1.0 - A)
        current_value = perm * self.m.value**2 * self.h.value * drive
        # NEURON's raw ``ical`` is outward-positive, so inward calcium entry
        # appears as a negative current. BrainCell channel currents use the
        # repo-wide inward-positive convention, so imported mechanisms flip
        # the sign here and comparisons should use ``-neuron_ical``.
        return -current_value * (u.mA / (u.cm**2))

    def f_m_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 56.0) / -6.2))

    def f_m_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return (0.333 / (u.math.exp((V + 131.0) / -16.7) + u.math.exp((V + 15.8) / 18.2)) + 0.204) / self.qdeltat

    def f_h_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 80.0) / 4.0))

    def f_h_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return (
            u.math.where(
                V < -81.0,
                0.333 * u.math.exp((V + 466.0) / 66.0),
                0.333 * u.math.exp((V + 21.0) / -10.5) + 9.32,
            )
            / self.qdeltat
        )


@register_channel("Cav1p2_MA2020_GoC")
class Cav1p2_MA2020_GoC(OhmicHH):
    r"""Golgi cell Cav1.2 L-type calcium current with Ca inactivation.

    The Cav1.2 (L-type) calcium current of the cerebellar Golgi cell
    model of (Masoli et al., 2020) [3]_. Its kinetics are the GENESIS
    Cav1.2 model of (Evans, Maniar & Blackwell, 2013) [1]_, transferred
    from GENESIS to NEURON by (Beining et al., 2017) [2]_. Gating is
    :math:`m\,h\,n` with an ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, m \, h \, n \, (E_{Ca} - V) \\
        m_\infty &= \frac{1}{1 + \exp(-(V' + 8.9) / 6.7)} \\
        \tau_m &= \frac{1}{\alpha_m + \beta_m} \\
        \alpha_m &= \frac{39800 \, (V' + 8.124)}
                    {\exp((V' + 8.124) / 9.005) - 1} \\
        \beta_m &= 990 \exp(V' / 31.4) \\
        h_\infty &= \frac{\mathrm{VDI}}{1 + \exp((V' + 55) / 8)}
                    + (1 - \mathrm{VDI}) \\
        \tau_h &= 44.3 \\
        n_\infty &= \frac{k_f}{k_f + [Ca]_i / \mathrm{mM}} \\
        \tau_n &= 0.5
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}`, the time constants are in
    milliseconds, :math:`\mathrm{VDI} = 0.17` and
    :math:`k_f = 0.0005`. The :math:`n` gate is the imported
    mechanism's ``h2`` state: a calcium-dependent inactivation whose
    steady state depends on the internal calcium concentration alone
    and which relaxes with a fixed 0.5 ms time constant. Because
    :math:`\mathrm{VDI} = 0.17`, the voltage-dependent inactivation
    :math:`h_\infty` never falls below 0.83.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.0002 S/cm2``,
        i.e. ``0.2 mS/cm2`` (see Notes).
    V_sh : array-like or callable, optional
        Threshold shift. Accepted and stored, but read by no rate
        method of this class (see Notes). Defaults to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature driving the gate Q10 factors. Defaults
        to 22 degrees Celsius.
    q10 : array-like or callable, optional
        Q10 factor shared by all three gates. Defaults to ``1.0``,
        which makes the temperature scaling a no-op (see Notes).
    temp_ref : array-like, optional
        Reference temperature for ``q10``. Defaults to 22 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav1p3_MA2020_GoC : Sibling Cav1.3 current from the same import
        family and the same two origin papers, with its own
        Boltzmann parameters and an ``exprel``-guarded ``tau_m``.
    Cav1p2_MA2025_BC : The same mechanism re-imported for the basket
        cell model; identical kinetics, different model citation.
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``GoC/channel/Cav1p2_MA20_GoC.mod``. That file has no
    ``TITLE``; its whole header is the comment "model from Evans et al
    2013, transferred from GENESIS to NEURON by Beining et al (2016),
    'A novel comprehensive and consistent electrophysiologcal model of
    dentate granule cells'" (typo in the original). Both fields that
    header gives are wrong: the transfer paper appeared in 2017, and
    the quoted title corresponds to no published paper or preprint --
    it reads as a pre-publication working title. The citable record is
    the 2017 eLife paper [2]_. Note also that Evans et al. (2013) is a
    striatal medium spiny neuron paper, not a dentate granule cell
    paper, despite what the header's phrasing invites.

    The mod file's second header line, "also added Calcium dependent
    inactivation", and an inline comment crediting the ``h2`` state to
    "santhakumar 05" are recorded here as prose only. The verified
    bibliography resolves this mechanism to the two origin papers
    cited above and does not resolve that third credit, so no
    reference entry is written for it.

    ``V_sh`` is accepted, stored and never read: no rate method of
    this class uses it. That reproduces the mod file, whose
    ``PARAMETER`` block likewise declares ``vshift = 0 (mV)`` and
    never uses it. In the same spirit, the three gates are wired for
    Q10 scaling through ``Gate(q10="q10", temp_ref="temp_ref")``, but
    the shipped defaults (``q10 = 1.0`` and
    ``temp = temp_ref = 22 degC``) make ``phi`` exactly 1, matching
    the mod file, which applies no temperature scaling at all.

    **Import deviation -- rate-refresh relocation.** The ``rates()``
    call moved from ``BREAKPOINT`` into ``DERIVATIVE state``, so
    ``inf``/``tau`` are refreshed before the ``cnexp`` state update
    rather than after it.

    **Open question -- a possible unit-scale defect inherited from
    upstream.** The GENESIS originals (``CaL12CDI.g``,
    ``CaL13CDI.g``) evaluate the ``mTau`` linoid in volts and return
    seconds, whereas the NEURON ports apply the same numeric
    coefficients in mV and declare ``mTau (ms)``. The worked
    comparison is given in :class:`Cav1p3_MA2020_GoC`'s Notes, and
    Cav1.2 shows the same pattern. That comparison is derived
    arithmetic, not a fetched claim, and it was not confirmed against
    a NEURON run, so it is recorded as an open question rather than
    as a defect: BrainCell reproduces the mod file faithfully under
    either reading, and this docstring asserts neither.

    NEURON's raw ``ica`` here is ``g * (v - eca)``, i.e.
    outward-positive; :class:`~braincell.channel._base.OhmicHH`
    computes ``g_max * m h n * (E - V)``, the same current under
    BrainCell's repo-wide inward-positive convention.

    ``g_max``'s default is the ``gbar`` of the cell-model deposit this
    mechanism was imported from -- a value tuned for that model, not a
    conductance reported by either origin paper.

    References
    ----------
    .. [1] Evans, R. C., Maniar, Y. M., & Blackwell, K. T. (2013).
           Dynamic modulation of spike timing-dependent calcium
           influx during corticostriatal upstates. Journal of
           Neurophysiology, 110(7), 1631-1645.
           doi:10.1152/jn.00232.2013
    .. [2] Beining, M., Mongiat, L. A., Schwarzacher, S. W., Cuntz,
           H., & Jedlicka, P. (2017). T2N as a new tool for robust
           electrophysiological modeling demonstrated for mature and
           adult-born dentate granule cells. eLife, 6, e26517.
           doi:10.7554/eLife.26517
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
        Gate("n", q10="q10", temp_ref="temp_ref"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0002 * (u.siemens / u.cm**2),
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

    def f_m_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 8.9) / -6.7))

    def f_h_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.VDI / (1.0 + u.math.exp((V + 55.0) / 8.0)) + (1.0 - self.VDI)

    def f_n_inf(self, V, Ca: IonInfo):
        return self.kf / (self.kf + Ca.Ci / u.mM)

    def f_m_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        m_alpha = 39800.0 * (V + 8.124) / (u.math.exp((V + 8.124) / 9.005) - 1.0)
        m_beta = 990.0 * u.math.exp(V / 31.4)
        return 1.0 / (m_alpha + m_beta)

    def f_h_tau(self, V, Ca: IonInfo):
        return 44.3

    def f_n_tau(self, V, Ca: IonInfo):
        return 0.5


@register_channel("Cav1p2_MA2025_BC")
class Cav1p2_MA2025_BC(Cav1p2_MA2020_GoC):
    r"""Cav1.2 L-type calcium current, basket-cell parameterisation.

    The same :math:`m\,h\,n` Cav1.2 kinetics documented in
    :class:`Cav1p2_MA2020_GoC`, reused unchanged for the cerebellar
    basket cell model of (Masoli et al., 2025) [3]_. The kinetics
    remain those of (Evans, Maniar & Blackwell, 2013) [1]_ as
    transferred from GENESIS to NEURON by (Beining et al., 2017) [2]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.0002 S/cm2``
        (``0.2 mS/cm2``). Inherited from :class:`Cav1p2_MA2020_GoC`.
    V_sh : array-like or callable, optional
        Accepted but not read by any rate method (see
        :class:`Cav1p2_MA2020_GoC` Notes). Default ``0.0 mV``.
        Inherited from :class:`Cav1p2_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature driving the gate Q10 factors, default 22
        degrees Celsius. Inherited from :class:`Cav1p2_MA2020_GoC`.
    q10 : array-like or callable, optional
        Q10 factor shared by all three gates, default ``1.0``.
        Inherited from :class:`Cav1p2_MA2020_GoC`.
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 22 degrees
        Celsius. Inherited from :class:`Cav1p2_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav1p2_MA2020_GoC : The base class; the full equation set, the
        header corrections, the inert ``V_sh`` and the ``mTau``
        unit-scale open question are all documented there.
    Cav1p3_MA2025_BC : Sibling Cav1.3 current of the same basket cell
        model, from the same two origin papers.

    Notes
    -----
    Ported from ``BC/channel/Cav1p2_MA25_BC.mod``, which is identical
    to ``GoC/channel/Cav1p2_MA20_GoC.mod`` except for its ``SUFFIX``
    line. Accordingly this class overrides nothing: the constructor,
    the gate declarations, the rate methods and the ``kf``/``VDI``
    constants are all inherited from :class:`Cav1p2_MA2020_GoC`. Only
    the ``register_channel`` key and this docstring's model citation
    differ.

    The ``MA2025`` import-deviations table records the same
    rate-refresh relocation already documented on
    :class:`Cav1p2_MA2020_GoC`: ``rates()`` moved from ``BREAKPOINT``
    into ``DERIVATIVE state``, so ``inf``/``tau`` are refreshed before
    the ``cnexp`` state update. The ``mTau`` unit-scale open question
    recorded for the ``MA2020`` Cav1.2/Cav1.3 pair applies identically
    here.

    References
    ----------
    .. [1] Evans, R. C., Maniar, Y. M., & Blackwell, K. T. (2013).
           Dynamic modulation of spike timing-dependent calcium
           influx during corticostriatal upstates. Journal of
           Neurophysiology, 110(7), 1631-1645.
           doi:10.1152/jn.00232.2013
    .. [2] Beining, M., Mongiat, L. A., Schwarzacher, S. W., Cuntz,
           H., & Jedlicka, P. (2017). T2N as a new tool for robust
           electrophysiological modeling demonstrated for mature and
           adult-born dentate granule cells. eLife, 6, e26517.
           doi:10.7554/eLife.26517
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Cav1p3_MA2020_GoC")
class Cav1p3_MA2020_GoC(OhmicHH):
    r"""Golgi cell Cav1.3 L-type calcium current with Ca inactivation.

    The Cav1.3 (L-type) calcium current of the cerebellar Golgi cell
    model of (Masoli et al., 2020) [3]_. Its kinetics are the GENESIS
    Cav1.3 model of (Evans, Maniar & Blackwell, 2013) [1]_, transferred
    from GENESIS to NEURON by (Beining et al., 2017) [2]_. Gating is
    :math:`m\,h\,n` with an ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, m \, h \, n \, (E_{Ca} - V) \\
        m_\infty &= \frac{1}{\exp(-(V' + 40) / 5) + 1} \\
        \tau_m &= \frac{1}{\alpha_m + \beta_m} \\
        \alpha_m &= \frac{39800 \times 15.005}
                    {\mathrm{exprel}((V' + 67.24) / 15.005)} \\
        \beta_m &= 3500 \exp(V' / 31.4) \\
        h_\infty &= \frac{\mathrm{VDI}}{\exp((V' + 37) / 5) + 1}
                    + (1 - \mathrm{VDI}) \\
        \tau_h &= 44.3 \\
        n_\infty &= \frac{k_f}{k_f + [Ca]_i / \mathrm{mM}} \\
        \tau_n &= 0.5
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}`, the time constants are in
    milliseconds, :math:`\mathrm{VDI} = 1.0` and
    :math:`k_f = 0.0005`. Unlike :class:`Cav1p2_MA2020_GoC`
    (:math:`\mathrm{VDI} = 0.17`), :math:`\mathrm{VDI} = 1.0` here
    leaves no non-inactivating floor, so :math:`h_\infty` reduces to
    the plain Boltzmann :math:`1 / (1 + \exp((V' + 37) / 5))`. The
    :math:`n` gate is the imported mechanism's ``h2`` state, a
    calcium-dependent inactivation with a fixed 0.5 ms time constant.

    :math:`\mathrm{exprel}(x) = (\exp(x) - 1) / x`, so
    :math:`\alpha_m` equals the mod file's literal
    :math:`39800 (V' + 67.24) / (\exp((V' + 67.24) / 15.005) - 1)`
    while remaining finite at :math:`V' = -67.24` mV, where the
    literal form is an indeterminate :math:`0/0`. At that voltage
    :math:`\mathrm{exprel}(0) = 1` and
    :math:`\alpha_m = 39800 \times 15.005`, the limit of the mod
    file's expression.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.000005 S/cm2``,
        i.e. ``0.005 mS/cm2`` (see Notes).
    V_sh : array-like or callable, optional
        Threshold shift. Accepted and stored, but read by no rate
        method of this class (see Notes). Defaults to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature driving the gate Q10 factors. Defaults
        to 22 degrees Celsius.
    q10 : array-like or callable, optional
        Q10 factor shared by all three gates. Defaults to ``1.0``,
        which makes the temperature scaling a no-op (see Notes).
    temp_ref : array-like, optional
        Reference temperature for ``q10``. Defaults to 22 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav1p2_MA2020_GoC : Sibling Cav1.2 current from the same import
        family and the same two origin papers; its ``alpha_m`` keeps
        the mod file's literal, unguarded quotient.
    Cav1p3_MA2025_BC : The same mechanism re-imported for the basket
        cell model; identical kinetics, different model citation.
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``GoC/channel/Cav1p3_MA20_GoC.mod``. That file has no
    ``TITLE``; its whole header is the comment "model from Evans et al
    2013, transferred from GENESIS to NEURON by Beining et al (2016),
    'A novel comprehensive and consistent electrophysiologcal model of
    dentate granule cells'" (typo in the original). Both fields that
    header gives are wrong: the transfer paper appeared in 2017, and
    the quoted title corresponds to no published paper or preprint --
    it reads as a pre-publication working title. The citable record is
    the 2017 eLife paper [2]_. Evans et al. (2013) is a striatal
    medium spiny neuron paper, not a dentate granule cell paper,
    despite what the header's phrasing invites.

    The mod file's second header line, "also added Calcium dependent
    inactivation", and an inline comment crediting the ``h2`` state to
    "santhakumar 05" are recorded here as prose only. The verified
    bibliography resolves this mechanism to the two origin papers
    cited above and does not resolve that third credit, so no
    reference entry is written for it.

    ``V_sh`` is accepted, stored and never read: no rate method of
    this class uses it. That reproduces the mod file, whose
    ``PARAMETER`` block likewise declares ``vshift = 0 (mV)`` and
    never uses it. In the same spirit, the three gates are wired for
    Q10 scaling through ``Gate(q10="q10", temp_ref="temp_ref")``, but
    the shipped defaults (``q10 = 1.0`` and
    ``temp = temp_ref = 22 degC``) make ``phi`` exactly 1, matching
    the mod file, which applies no temperature scaling at all.

    **Import deviation -- rate-refresh relocation.** The ``rates()``
    call moved from ``BREAKPOINT`` into ``DERIVATIVE state``, so
    ``inf``/``tau`` are refreshed before the ``cnexp`` state update
    rather than after it.

    **Open question -- a possible unit-scale defect inherited from
    upstream.** The GENESIS original ``CaL13CDI.g`` evaluates the
    ``mTau`` linoid in volts and returns seconds, whereas the NEURON
    port applies the same numeric coefficients in mV and declares
    ``mTau (ms)``. Worked at :math:`V = 0` mV, the GENESIS reading
    gives ``tau_m`` about 0.283 ms while the mod file -- and hence
    this class -- gives about 2.9e-5 ms, roughly 1e4 times smaller,
    i.e. effectively instantaneous activation. Cav1.2 shows the same
    pattern, and the transfer paper's Methods mention no intentional
    rescaling. This is derived arithmetic, not a fetched claim, and
    it was not confirmed against a NEURON run, so it is recorded as
    an open question rather than as a defect: BrainCell reproduces
    the mod file faithfully under either reading, and this docstring
    asserts neither.

    NEURON's raw ``ica`` here is ``g * (v - eca)``, i.e.
    outward-positive; :class:`~braincell.channel._base.OhmicHH`
    computes ``g_max * m h n * (E - V)``, the same current under
    BrainCell's repo-wide inward-positive convention.

    ``g_max``'s default is the ``gbar`` of the cell-model deposit this
    mechanism was imported from -- a value tuned for that model, not a
    conductance reported by either origin paper.

    References
    ----------
    .. [1] Evans, R. C., Maniar, Y. M., & Blackwell, K. T. (2013).
           Dynamic modulation of spike timing-dependent calcium
           influx during corticostriatal upstates. Journal of
           Neurophysiology, 110(7), 1631-1645.
           doi:10.1152/jn.00232.2013
    .. [2] Beining, M., Mongiat, L. A., Schwarzacher, S. W., Cuntz,
           H., & Jedlicka, P. (2017). T2N as a new tool for robust
           electrophysiological modeling demonstrated for mature and
           adult-born dentate granule cells. eLife, 6, e26517.
           doi:10.7554/eLife.26517
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", q10="q10", temp_ref="temp_ref"),
        Gate("h", q10="q10", temp_ref="temp_ref"),
        Gate("n", q10="q10", temp_ref="temp_ref"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.000005 * (u.siemens / u.cm**2),
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

    def f_m_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (u.math.exp((V - (-40.0)) / -5.0) + 1.0)

    def f_h_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.VDI / (u.math.exp((V - (-37.0)) / 5.0) + 1.0) + (1.0 - self.VDI)

    def f_n_inf(self, V, Ca: IonInfo):
        return self.kf / (self.kf + Ca.Ci / u.mM)

    def f_m_tau(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        m_alpha = 39800.0 * 15.005 / u.math.exprel((V + 67.24) / 15.005)
        m_beta = 3500.0 * u.math.exp(V / 31.4)
        return 1.0 / (m_alpha + m_beta)

    def f_h_tau(self, V, Ca: IonInfo):
        return 44.3

    def f_n_tau(self, V, Ca: IonInfo):
        return 0.5


@register_channel("Cav1p3_MA2025_BC")
class Cav1p3_MA2025_BC(Cav1p3_MA2020_GoC):
    r"""Cav1.3 L-type calcium current, basket-cell parameterisation.

    The same :math:`m\,h\,n` Cav1.3 kinetics documented in
    :class:`Cav1p3_MA2020_GoC`, reused unchanged for the cerebellar
    basket cell model of (Masoli et al., 2025) [3]_. The kinetics
    remain those of (Evans, Maniar & Blackwell, 2013) [1]_ as
    transferred from GENESIS to NEURON by (Beining et al., 2017) [2]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.000005 S/cm2``
        (``0.005 mS/cm2``). Inherited from
        :class:`Cav1p3_MA2020_GoC`.
    V_sh : array-like or callable, optional
        Accepted but not read by any rate method (see
        :class:`Cav1p3_MA2020_GoC` Notes). Default ``0.0 mV``.
        Inherited from :class:`Cav1p3_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature driving the gate Q10 factors, default 22
        degrees Celsius. Inherited from :class:`Cav1p3_MA2020_GoC`.
    q10 : array-like or callable, optional
        Q10 factor shared by all three gates, default ``1.0``.
        Inherited from :class:`Cav1p3_MA2020_GoC`.
    temp_ref : array-like, optional
        Reference temperature for ``q10``, default 22 degrees
        Celsius. Inherited from :class:`Cav1p3_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav1p3_MA2020_GoC : The base class; the full equation set, the
        ``exprel`` guard, the header corrections, the inert ``V_sh``
        and the ``mTau`` unit-scale open question are documented
        there.
    Cav1p2_MA2025_BC : Sibling Cav1.2 current of the same basket cell
        model, from the same two origin papers.

    Notes
    -----
    Ported from ``BC/channel/Cav1p3_MA25_BC.mod``, which is identical
    to ``GoC/channel/Cav1p3_MA20_GoC.mod`` except for its ``SUFFIX``
    line. Accordingly this class overrides nothing: the constructor,
    the gate declarations, the rate methods and the ``kf``/``VDI``
    constants are all inherited from :class:`Cav1p3_MA2020_GoC`. Only
    the ``register_channel`` key and this docstring's model citation
    differ.

    The ``MA2025`` import-deviations table records the same
    rate-refresh relocation already documented on
    :class:`Cav1p3_MA2020_GoC`: ``rates()`` moved from ``BREAKPOINT``
    into ``DERIVATIVE state``, so ``inf``/``tau`` are refreshed before
    the ``cnexp`` state update. The ``mTau`` unit-scale open question
    recorded for the ``MA2020`` Cav1.2/Cav1.3 pair applies identically
    here.

    References
    ----------
    .. [1] Evans, R. C., Maniar, Y. M., & Blackwell, K. T. (2013).
           Dynamic modulation of spike timing-dependent calcium
           influx during corticostriatal upstates. Journal of
           Neurophysiology, 110(7), 1631-1645.
           doi:10.1152/jn.00232.2013
    .. [2] Beining, M., Mongiat, L. A., Schwarzacher, S. W., Cuntz,
           H., & Jedlicka, P. (2017). T2N as a new tool for robust
           electrophysiological modeling demonstrated for mature and
           adult-born dentate granule cells. eLife, 6, e26517.
           doi:10.7554/eLife.26517
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Cav3p1_MA2020_GoC")
class Cav3p1_MA2020_GoC(HH):
    r"""Golgi cell Cav3.1 low-threshold calcium current with GHK drive.

    The Cav3.1 (T-type) low-threshold calcium current of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [3]_. Its
    kinetics were fitted to the Cav3.1 temperature-dependence
    recordings of (Iftinca et al., 2006) [1]_ and published in the
    Purkinje-cell calcium-buffering model of (Anwar, Hong & De
    Schutter, 2012) [2]_. Gating is :math:`p^2 q`, driven by a
    constant-field (GHK) calcium flux rather than an ohmic term:

    .. math::

        \begin{aligned}
        I_{Ca} &= -P \, p^2 q \, \Phi(V, [Ca]_i, [Ca]_o, z{=}2, T) \\
        p_\infty &= \frac{1}{1 + \exp((V - v_{0,m}) / k_m)} \\
        q_\infty &= \frac{1}{1 + \exp((V - v_{0,h}) / k_h)} \\
        \tau_p &= \begin{cases}
                  1 & V \leq -90\ \mathrm{mV} \\
                  \dfrac{1}{q_t}\left(C_{\tau m} + \dfrac{A_{\tau m}}
                  {e^{(V - v_{\tau m1})/k_{\tau m1}}
                   + e^{(V - v_{\tau m2})/k_{\tau m2}}}\right)
                  & V > -90\ \mathrm{mV}
                  \end{cases} \\
        \tau_q &= \frac{1}{q_t}\left(C_{\tau h}
                  + \frac{A_{\tau h}}
                  {e^{(V - v_{\tau h1})/k_{\tau h1}}}\right) \\
        q_t &= Q_{10}^{(T - T_{ref}) / 10}
        \end{aligned}

    with :math:`v_{0,m} = -52` mV, :math:`k_m = -5` mV,
    :math:`v_{0,h} = -72` mV, :math:`k_h = 7` mV,
    :math:`C_{\tau m} = A_{\tau m} = A_{\tau h} = 1`,
    :math:`C_{\tau h} = 15`, :math:`v_{\tau m1} = -40` mV,
    :math:`v_{\tau m2} = -102` mV, :math:`k_{\tau m1} = 9` mV,
    :math:`k_{\tau m2} = -18` mV, :math:`v_{\tau h1} = -32` mV and
    :math:`k_{\tau h1} = 7` mV; the time constants are in
    milliseconds and :math:`P` is the calcium permeability. The
    :math:`\tau_p` branch point is inclusive at exactly
    :math:`-90` mV, and its constant branch is **not** divided by
    :math:`q_t` (see Notes).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability :math:`P` entering the GHK flux -- the
        mod file's ``pcabar`` -- despite the ``g_max`` name and
        conductance-like spelling. Defaults to ``2.5e-4 cm/s`` (see
        Notes).
    V_sh : array-like or callable, optional
        Threshold shift. Accepted and stored, but read by no method
        of this class (see Notes). Defaults to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature. Enters both :math:`q_t` and the GHK
        flux. Defaults to 22 degrees Celsius.
    q10 : array-like or callable, optional
        Q10 factor for :math:`q_t`. Defaults to ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for :math:`q_t`. Defaults to 37 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p1_MA2024_PC : The same mechanism re-imported for the human
        Purkinje cell model; identical kinetics, different model
        citation.
    Cav3p1_MA2020_GoC_Frozen : This class with the GHK term's
        voltage dependence removed from the autodiff graph.
    Cav3p1Test_PC24 : Anonymous test variant that replaces the GHK
        drive with a direct conductance-density current law.
    braincell.channel._base.ghk_flux : Shared GHK flux helper; this
        class does not call it directly (see Notes).

    Notes
    -----
    Ported from ``GoC/channel/Cav3p1_MA20_GoC.mod``. That file's
    ``TITLE`` reads "Low threshold calcium current Cerebellum Purkinje
    Cell Model" and its ``COMMENT`` records the rename "Suffix from
    CaT3_1 to CaV3_1" -- both inherited verbatim from Anwar's original
    Purkinje-cell mechanism (ModelDB 138382). This class is
    nonetheless the **Golgi cell** port (``SUFFIX Cav3p1_MA20_GoC``),
    used in the Golgi cell model cited as [3]_; the ``TITLE``'s
    "Purkinje" is upstream provenance, not a description of this
    class. An earlier revision of this docstring's summary line
    repeated it as though it were, and that is corrected here.

    Two corrections to the mod header's own reference line, "Anwar H,
    Hong S, De Schutter E (2010) ... in Purkinje cell": the citable
    record is 2012 -- 2010 is the online-first date, which is also why
    the DOI carries ``-010-`` -- and the published title ends
    "Purkinje **cells**", plural. Entry [2]_ below is the corrected
    form. The header's "Written by Haroon Anwar" line names the
    mechanism's author and is deliberately not turned into a citation.

    **Temperature handling is encoded in the tau expressions, not in
    the gate declaration.** The source NMODL applies :math:`q_t`
    directly inside the tau formulas instead of through a uniform
    gate-level ``phi``:

    - for ``p``/``m`` the ``v <= -90`` branch is hard-coded to
      ``1 ms`` and is **not** divided by ``qt``;
    - in the other branch the full ``C_tau_m + A_tau_m / (...)``
      expression is divided by ``qt``;
    - for ``q``/``h`` the full ``C_tau_h + A_tau_h / exp(...)``
      expression is also divided by ``qt``.

    That does not match the generic ``HH`` gate temperature path,
    where ``Gate(q10=..., temp_ref=...)`` multiplies the whole
    derivative by ``phi`` and therefore divides the whole tau by
    ``qt``. Gate ``phi`` is intentionally left at 1 here and the
    source-mod temperature handling is encoded directly in
    :meth:`f_p_tau` and :meth:`f_q_tau`.

    ``current()`` evaluates the GHK constant-field equation through
    the module-level ``_cav3p1_nmodl_ghk_flux`` helper rather than
    :func:`~braincell.channel._base.ghk_flux`, so that it reproduces
    the mod file's own constants exactly: ``F = 9.6485e4 C/mol``,
    ``R = 8.3145 J/(K mol)`` and the mod file's ``kelvinfkt``
    conversion ``273.19 + celsius``, whose 0.04 K offset from
    :func:`brainunit.celsius2kelvin` is carried as
    ``_CAV3P1_NMODL_TEMP_OFFSET``. The helper also keeps the mod
    file's small-``zeta`` series branch, taken when
    ``|1 - exp(-zeta)| < 1e-6``. NEURON's raw ``ica`` for this
    mechanism is outward-positive; ``current()`` negates it to match
    BrainCell's repo-wide inward-positive convention.

    ``g_max`` is a permeability in ``cm/s``, not a conductance
    density: the name is BrainCell's uniform parameter name for the
    scale factor in front of the gating product, and this mechanism's
    current law is a permeability-scaled GHK flux. ``V_sh`` is
    accepted, stored and never read; the mod file declares no
    corresponding parameter at all.

    ``g_max``'s default is the ``pcabar`` of the cell-model deposit
    this mechanism was imported from -- a value tuned for that model,
    not a permeability reported by either origin paper.

    References
    ----------
    .. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
           Turner, R. W., & Zamponi, G. W. (2006). Temperature
           dependence of T-type calcium channel gating.
           Neuroscience, 142(4), 1031-1042.
           doi:10.1016/j.neuroscience.2006.07.010
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2),
        Gate("q"),
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
        drive = _cav3p1_nmodl_ghk_flux(V=V, ci=Ca.Ci, co=Ca.Co, z=self.z, temp=self.temp)
        return -self.g_max * self.conductance_factor(V, Ca) * drive

    def f_p_inf(self, V, Ca: IonInfo):
        return 1.0 / (1.0 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V, Ca: IonInfo):
        return 1.0 / (1.0 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V, Ca: IonInfo):
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return u.math.where(
            V <= -90.0 * u.mV,
            1.0,
            (
                self.C_tau_m
                + (
                    self.A_tau_m
                    / (
                        u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1)
                        + u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
                    )
                )
            )
            / qt,
        )

    def f_q_tau(self, V, Ca: IonInfo):
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return (self.C_tau_h + self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1)) / qt


@register_channel("Cav3p1_MA2024_PC")
class Cav3p1_MA2024_PC(Cav3p1_MA2020_GoC):
    r"""Purkinje cell Cav3.1 low-threshold calcium current, GHK drive.

    The same :math:`p^2 q` Cav3.1 kinetics and GHK current law
    documented in :class:`Cav3p1_MA2020_GoC`, reused unchanged for the
    human Purkinje cell model of (Masoli et al., 2024) [3]_. The
    kinetics remain the fit to (Iftinca et al., 2006) [1]_ published
    in (Anwar, Hong & De Schutter, 2012) [2]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.5e-4 cm/s``. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    V_sh : array-like or callable, optional
        Accepted but read by no method of this class (see
        :class:`Cav3p1_MA2020_GoC` Notes). Default ``0.0 mV``.
        Inherited from :class:`Cav3p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature entering both the ``qt`` factor and the
        GHK flux, default 22 degrees Celsius. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    q10 : array-like or callable, optional
        Q10 factor for ``qt``, default ``3.0``. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    temp_ref : array-like, optional
        Reference temperature for ``qt``, default 37 degrees Celsius.
        Inherited from :class:`Cav3p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p1_MA2020_GoC : The base class; the full equation set, the
        GHK helper's constants, the tau-embedded temperature handling
        and the header corrections are documented there.
    Cav3p1_MA2024_PC_Frozen : Purkinje-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class (see its Notes).
    Cav3p1Test_PC24 : Anonymous test variant of the same mechanism
        with the GHK drive replaced by a direct conductance-density
        current law.

    Notes
    -----
    Ported from ``PC/channel/Cav3p1_MA24_PC.mod``. An earlier revision
    of this docstring named the source file ``Cav3p1_MA2024_PC.mod``,
    which does not exist; the shipped file uses the two-digit year
    code, and that is corrected here. That file is identical to
    ``GoC/channel/Cav3p1_MA20_GoC.mod`` except for its ``SUFFIX`` line
    and a dropped ``INDEPENDENT`` statement, neither of which affects
    the kinetics. Accordingly this class overrides nothing: the
    constructor, the gate declarations, the named Boltzmann/tau
    parameter block and the ``current`` method are all inherited from
    :class:`Cav3p1_MA2020_GoC`. Only the ``register_channel`` key and
    this docstring's model citation differ.

    The ``MA2024`` import-deviations tables list no ``TABLE`` removal,
    no integration-method substitution and no rate-refresh relocation
    for ``Cav3p1``.

    References
    ----------
    .. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
           Turner, R. W., & Zamponi, G. W. (2006). Temperature
           dependence of T-type calcium channel gating.
           Neuroscience, 142(4), 1031-1042.
           doi:10.1016/j.neuroscience.2006.07.010
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("Cav3p1_MA2020_GoC_Frozen")
class Cav3p1_MA2020_GoC_Frozen(Cav3p1_MA2020_GoC):
    r"""Golgi cell Cav3.1 with the GHK drive frozen for autodiff.

    A subclass of :class:`Cav3p1_MA2020_GoC` that overrides
    :meth:`current` alone, stopping the gradient through the membrane
    potential where it enters the constant-field (GHK) flux term. The
    forward current is bit-for-bit that of the base class; only
    reverse-mode derivatives differ. Every kinetic equation, the whole
    parameter block and the Golgi cell model attribution of (Masoli et
    al., 2020) [3]_ -- with kinetics from (Iftinca et al., 2006) [1]_
    as published in (Anwar, Hong & De Schutter, 2012) [2]_ -- are
    inherited unchanged.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.5e-4 cm/s``. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    V_sh : array-like or callable, optional
        Accepted but read by no method of this class (see
        :class:`Cav3p1_MA2020_GoC` Notes). Default ``0.0 mV``.
        Inherited from :class:`Cav3p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature entering both the ``qt`` factor and the
        GHK flux, default 22 degrees Celsius. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    q10 : array-like or callable, optional
        Q10 factor for ``qt``, default ``3.0``. Inherited from
        :class:`Cav3p1_MA2020_GoC`.
    temp_ref : array-like, optional
        Reference temperature for ``qt``, default 37 degrees Celsius.
        Inherited from :class:`Cav3p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p1_MA2020_GoC : The base class; every equation, the GHK
        helper's constants and the header corrections are documented
        there.
    Cav3p1_MA2024_PC_Frozen : Purkinje-cell counterpart with
        numerically identical behaviour but a different class graph:
        it subclasses :class:`~braincell.channel._base.HH` directly
        and re-declares everything rather than inheriting.

    Notes
    -----
    "Frozen" describes the gradient path, not the forward numerics and
    not the channel states. Both gates keep integrating exactly as in
    the base class -- nothing stops evolving. What
    :meth:`current` changes is that the membrane potential handed to
    the GHK helper is first passed through the module-level
    ``_freeze_quantity_gradient``, which rebuilds the quantity from
    :func:`jax.lax.stop_gradient` applied to its mantissa. The
    explicit voltage dependence of the GHK driving force therefore
    contributes nothing to reverse-mode gradients, while the value it
    computes is unchanged.

    The unfrozen ``V`` is still passed to
    :meth:`~braincell.channel._base.HH.conductance_factor`, but that
    method ignores its voltage argument entirely and reads only the
    gate states, so the distinction has no effect on either the value
    or the gradient. The gate states themselves are not frozen.

    Provenance, the tau-embedded temperature handling, the GHK
    helper's constants and the ``g_max``-is-a-permeability discrepancy
    are identical to :class:`Cav3p1_MA2020_GoC`'s and are not repeated
    here. Per the bibliography's attribution scan, this subclass
    contributes no rate-function code of its own.

    References
    ----------
    .. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
           Turner, R. W., & Zamponi, G. W. (2006). Temperature
           dependence of T-type calcium channel gating.
           Neuroscience, 142(4), 1031-1042.
           doi:10.1016/j.neuroscience.2006.07.010
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"

    def current(self, V, Ca: IonInfo):
        frozen_V = _freeze_quantity_gradient(V)
        drive = _cav3p1_nmodl_ghk_flux(V=frozen_V, ci=Ca.Ci, co=Ca.Co, z=self.z, temp=self.temp)
        return -self.g_max * self.conductance_factor(V, Ca) * drive


@register_channel("Cav3p1_MA2024_PC_Frozen")
class Cav3p1_MA2024_PC_Frozen(HH):
    r"""Purkinje cell Cav3.1 with the GHK drive frozen for autodiff.

    A standalone Purkinje-cell Cav3.1 mechanism that stops the
    gradient through the membrane potential where it enters the
    constant-field (GHK) flux term. Its kinetics, its named
    Boltzmann/tau parameter block and its forward current are
    numerically identical to :class:`Cav3p1_MA2020_GoC`'s, and its
    attribution is that of :class:`Cav3p1_MA2024_PC`: the Cav3.1 fit
    to (Iftinca et al., 2006) [1]_ published in (Anwar, Hong & De
    Schutter, 2012) [2]_, imported for the human Purkinje cell model
    of (Masoli et al., 2024) [3]_. Unlike its Golgi-cell counterpart
    it inherits none of that -- see Notes.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), despite the ``g_max`` name. Defaults to
        ``2.5e-4 cm/s``.
    V_sh : array-like or callable, optional
        Threshold shift. Accepted and stored, but read by no method
        of this class (see Notes). Defaults to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature. Enters both the ``qt`` factor and the
        GHK flux. Defaults to 22 degrees Celsius.
    q10 : array-like or callable, optional
        Q10 factor for ``qt``. Defaults to ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for ``qt``. Defaults to 37 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p1_MA2024_PC : The unfrozen Purkinje-cell import of the same
        mechanism. This class is **not** derived from it.
    Cav3p1_MA2020_GoC : Where the equation set, the GHK helper's
        constants, the tau-embedded temperature handling and the mod
        header corrections are documented in full.
    Cav3p1_MA2020_GoC_Frozen : Golgi-cell counterpart with the same
        freezing behaviour, reached by inheritance instead.

    Notes
    -----
    **This class is not a subclass of the unfrozen Purkinje-cell
    import, and the two frozen Cav3.1 classes in this module are not
    built the same way.** :class:`Cav3p1_MA2020_GoC_Frozen` derives
    from :class:`Cav3p1_MA2020_GoC` and overrides :meth:`current`
    alone.
    This class derives from :class:`~braincell.channel._base.HH`
    directly and re-declares everything: ``root_type``, both gates,
    the whole constructor and parameter block, ``f_p_inf``,
    ``f_q_inf``, ``f_p_tau``, ``f_q_tau`` and ``current``. The two
    frozen variants therefore compute the same numbers while sharing
    no code, and a change to :class:`Cav3p1_MA2020_GoC` propagates to
    one of them and not to the other.

    "Frozen" describes the gradient path, not the forward numerics and
    not the channel states. Both gates keep integrating normally --
    nothing stops evolving. :meth:`current` passes the membrane
    potential through the module-level ``_freeze_quantity_gradient``,
    which rebuilds the quantity from :func:`jax.lax.stop_gradient`
    applied to its mantissa, before handing it to the GHK helper. The
    unfrozen ``V`` still reaches
    :meth:`~braincell.channel._base.HH.conductance_factor`, but that
    method ignores its voltage argument and reads only the gate
    states, so the distinction affects neither value nor gradient.

    The kinetics are those of ``PC/channel/Cav3p1_MA24_PC.mod``, which
    is identical to ``GoC/channel/Cav3p1_MA20_GoC.mod`` except for its
    ``SUFFIX`` line and a dropped ``INDEPENDENT`` statement. No mod
    file ships a frozen-gradient variant: freezing is a BrainCell
    autodiff facility with no counterpart in NMODL, so it is not an
    import deviation and changes nothing a NEURON comparison would
    observe.

    As in :class:`Cav3p1_MA2020_GoC`, ``g_max`` is a permeability in
    ``cm/s`` rather than a conductance density, ``V_sh`` is accepted
    and never read with no corresponding parameter in the mod file,
    the temperature factor ``qt`` is embedded in the tau expressions
    instead of in ``Gate``, the ``tau_p`` constant branch at
    ``V <= -90 mV`` is not divided by ``qt``, and ``current()`` uses
    the module-level ``_cav3p1_nmodl_ghk_flux`` helper with the mod
    file's own ``F``/``R`` constants and its ``273.19 + celsius``
    Kelvin conversion. The permeability default is the deposit's tuned
    ``pcabar``, not a value reported by either origin paper.

    References
    ----------
    .. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
           Turner, R. W., & Zamponi, G. W. (2006). Temperature
           dependence of T-type calcium channel gating.
           Neuroscience, 142(4), 1031-1042.
           doi:10.1016/j.neuroscience.2006.07.010
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2),
        Gate("q"),
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
        frozen_V = _freeze_quantity_gradient(V)
        drive = _cav3p1_nmodl_ghk_flux(V=frozen_V, ci=Ca.Ci, co=Ca.Co, z=self.z, temp=self.temp)
        return -self.g_max * self.conductance_factor(V, Ca) * drive

    def f_p_inf(self, V, Ca: IonInfo):
        return 1.0 / (1.0 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V, Ca: IonInfo):
        return 1.0 / (1.0 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V, Ca: IonInfo):
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return u.math.where(
            V <= -90.0 * u.mV,
            1.0,
            (
                self.C_tau_m
                + (
                    self.A_tau_m
                    / (
                        u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1)
                        + u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
                    )
                )
            )
            / qt,
        )

    def f_q_tau(self, V, Ca: IonInfo):
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return (self.C_tau_h + self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1)) / qt


@register_channel("Cav3p1Test_PC24")
class Cav3p1Test_PC24(HH):
    r"""Cav3.1 test variant with the GHK drive replaced by a constant.

    A test variant of the Cav3.1 mechanism whose gating kinetics are
    exactly those of :class:`Cav3p1_MA2024_PC` -- the Cav3.1 fit to
    (Iftinca et al., 2006) [1]_ published in (Anwar, Hong & De
    Schutter, 2012) [2]_ and imported for the human Purkinje cell
    model of (Masoli et al., 2024) [3]_ -- but with the constant-field
    (GHK) drive replaced by a direct conductance-density current law:

    .. math::

       I_{Ca} = g_{max} \, p^2 q

    **This ohmic form is not the published one.** The published
    mechanism is a GHK-driven permeability; this variant is not, and
    its ``g_max = 2.5e-4`` carries ``S/cm2`` where the published
    mechanism's ``pcabar = 2.5e-4`` carries ``cm/s`` -- the same
    number in a different dimension. The class name's "Test" is
    accurate: nothing in the source file or the deposit indicates
    that this variant produced any published result. See Notes for
    what the current law does and does not depend on.

    The steady-state and tau formulas, including the temperature
    factor embedded in the tau expressions and the ``V <= -90 mV``
    constant branch of :math:`\tau_p`, are identical to
    :class:`Cav3p1_MA2020_GoC`'s and are written out there.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Conductance density multiplying the gating product. Defaults
        to ``2.5e-4 S/cm2`` (see above and Notes). This class takes
        no ``V_sh``, unlike its Cav3.1 siblings.
    temp : array-like, optional
        Absolute temperature driving the ``qt`` factor. Defaults to
        22 degrees Celsius. It does **not** enter a GHK term here,
        because there is none.
    q10 : array-like or callable, optional
        Q10 factor for ``qt``. Defaults to ``3.0``.
    temp_ref : array-like, optional
        Reference temperature for ``qt``. Defaults to 37 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p1_MA2024_PC : The published, GHK-driven Purkinje-cell
        mechanism this variant's kinetics are taken from.
    Cav3p1_MA2020_GoC : Where the full equation set and the
        tau-embedded temperature handling are documented.

    Notes
    -----
    Ported from ``PC/channel/Cav3_1_test.mod``, which carries no
    ``TITLE``, no ``COMMENT``, no author line and no reference of any
    kind. Its provenance was established by comparison instead: it
    declares the same named parameter block and the same four rate
    expressions as ``PC/channel/Cav3p1_MA24_PC.mod``, character for
    character, so it inherits that mechanism's attribution chain. The
    sibling file ``Cav3_1_test2.mod`` is claimed by no BrainCell
    symbol.

    **The single difference between the two files is the current
    law.** ``Cav3p1_MA24_PC.mod`` computes
    ``ica = (1e3) * pcabar * m*m*h * g`` with ``g`` from GHK and
    ``pcabar`` in ``cm/s``; ``Cav3_1_test.mod`` drops the GHK drive
    entirely, redeclares ``pcabar = 2.5e-4 (S/cm2)`` as a conductance
    density and computes ``ica = pcabar * m*m*h``. That right-hand
    side is dimensionally inconsistent with the ``(mA/cm2)`` the same
    file declares for ``ica``, so BrainCell's :meth:`current`
    multiplies by ``1 mV`` purely to lift the value to a
    current-density unit that can flow through the standard
    compare/runtime paths. The multiplier is a fixed constant, not a
    driving force: the resulting current carries no ``(v - eca)``
    factor and no GHK factor, and therefore depends on voltage only
    through the gates.

    ``current()`` also negates the result, as every imported
    mechanism in this module does, to match BrainCell's repo-wide
    inward-positive convention against NEURON's outward-positive
    ``ica``.

    No import deviations are recorded for this mechanism: the file
    carries no ``TABLE``, no ``derivimplicit`` and no rate-refresh
    relocation, and it appears in none of the comparison suite's
    deviation tables.

    References
    ----------
    .. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
           Turner, R. W., & Zamponi, G. W. (2006). Temperature
           dependence of T-type calcium channel gating.
           Neuroscience, 142(4), 1031-1042.
           doi:10.1016/j.neuroscience.2006.07.010
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("p", power=2),
        Gate("q"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.5e-4 * (u.siemens / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        q10: Union[brainstate.typing.ArrayLike, Callable] = 3.0,
        temp_ref: brainstate.typing.ArrayLike = u.celsius2kelvin(37.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
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

    def current(self, V, Ca: IonInfo):
        _ = (V, Ca)
        # ``Cav3_1_test.mod`` drops both ``(v-eca)`` and GHK drive, so the raw
        # NMODL right-hand side is numerically ``pcabar * p^2 * q`` despite the
        # declared current unit. We multiply by ``1 mV`` here solely to lift the
        # conductance-density-like quantity to a current-density unit that can
        # flow through the standard BrainCell compare/runtime paths.
        return -self.g_max * self.conductance_factor(V, Ca) * (1.0 * u.mV)

    def f_p_inf(self, V, Ca: IonInfo):
        _ = Ca
        return 1.0 / (1.0 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V, Ca: IonInfo):
        _ = Ca
        return 1.0 / (1.0 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V, Ca: IonInfo):
        _ = Ca
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return u.math.where(
            V <= -90.0 * u.mV,
            1.0,
            (
                self.C_tau_m
                + (
                    self.A_tau_m
                    / (
                        u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1)
                        + u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
                    )
                )
            )
            / qt,
        )

    def f_q_tau(self, V, Ca: IonInfo):
        _ = Ca
        qt = self.q10 ** (((self.temp - self.temp_ref) / u.kelvin) / 10.0)
        return (self.C_tau_h + self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1)) / qt


@register_channel("Cav2p1_RI2021_SC")
class Cav2p1_RI2021_SC(HH):
    r"""Stellate cell Cav2.1 P-type calcium current with GHK drive.

    The Cav2.1 (P-type) calcium current of the cerebellar stellate
    cell model of (Rizza et al., 2021) [3]_. Its kinetics were built
    from dissociated Purkinje neuron recordings reported by (Swensen
    & Bean, 2005) [1]_ and published as part of the Purkinje-cell
    calcium-buffering model of (Anwar, Hong & De Schutter, 2012)
    [2]_. Gating is :math:`m^3` with a single activation state,
    driven by a constant-field (GHK) calcium flux rather than an
    ohmic term:

    .. math::

        \begin{aligned}
        I_{Ca} &= -P \, m^3 \,
                  \Phi(V', [Ca]_i, [Ca]_o, z{=}2, T) \\
        m_\infty &= \frac{1}{1 + \exp(-(V' - v_{1/2}) / k)} \\
        \tau_m &= \frac{1}{\phi_m} \begin{cases}
                  0.2702 + 1.1622\, e^{-(V' + 26.798)^2 / 164.19}
                  & V' \geq -40 \\
                  0.6923\, e^{V' / 1089.372} & V' < -40
                  \end{cases} \\
        \phi_m &= 3^{(T - 23\,^\circ\mathrm{C}) / 10}
        \end{aligned}

    where :math:`V' = V - V_{sh}`, :math:`v_{1/2} = -29.458` mV,
    :math:`k = 8.429` mV, :math:`P` is the calcium permeability and
    :math:`\Phi` is :func:`~braincell.channel._base.ghk_flux`. The
    :math:`\tau_m` branch reads :math:`V'` in millivolts and returns
    milliseconds; its branch point is inclusive at exactly
    :math:`-40` mV. :math:`\phi_m` is supplied by the ``Gate``
    declaration, which divides :math:`\tau_m` by it.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability :math:`P` entering the GHK flux -- the
        mod file's ``pcabar`` -- despite the ``g_max`` name and
        conductance-like spelling. Defaults to ``2.2e-4 cm/s`` (see
        Notes).
    V_sh : array-like or callable, optional
        Voltage shift subtracted from :math:`V` before every rate
        and before the GHK term; the mod file's ``vshift``. Defaults
        to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature. Enters both :math:`\phi_m` and the GHK
        flux. Defaults to 23 degrees Celsius, at which
        :math:`\phi_m = 1`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_MA2024_PC : The same mechanism re-imported for the human
        Purkinje cell model; identical kinetics, different model
        citation.
    Cav2p1_MA2025_BC : The same mechanism re-imported for the basket
        cell model; identical kinetics, different model citation.
    Cav2p1_RI2021_SC_Frozen : Stellate-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class, and it is not
        numerically identical to it (see its Notes).
    braincell.channel._base.ghk_flux : Shared GHK flux helper called
        by :meth:`current`; the constant-field derivation lives
        there and is not restated here.

    Notes
    -----
    Ported from ``SC/channel/Cav2p1_RI21_SC.mod``, whose ``COMMENT``
    records that the mechanism was "Constructed from the recording
    data provided by Bruce Bean" and cites (Swensen & Bean, 2005)
    for those recordings. The same file also records the rename
    "Suffix from newCaP to Cav2_1".

    Two corrections to the mod header's own model-reference line,
    "Anwar H, Hong S, De Schutter E (2010) ... in Purkinje cell":
    the citable record is 2012 -- 2010 is the online-first date,
    which is also why the DOI carries ``-010-`` -- and the published
    title ends "Purkinje **cells**", plural. Entry [2]_ below is the
    corrected form. The header's "Written by Sungho Hong" line names
    the mechanism's author and is deliberately not turned into a
    citation.

    ``current()`` evaluates ``-g_max * m^3 * drive`` with ``drive``
    from :func:`~braincell.channel._base.ghk_flux`, so ``g_max``
    occupies that function's permeability slot :math:`P_s` and is a
    permeability in ``cm/s``, not a conductance density. The
    negation matches BrainCell's repo-wide inward-positive
    convention against NEURON's outward-positive ``ica``.

    **The shared helper's physical constants are not the mod
    file's.** :func:`~braincell.channel._base.ghk_flux` uses the
    CODATA Faraday and gas constants and the temperature exactly as
    passed, whereas this mod file declares ``F = 9.6485e4``,
    ``R = 8.3145`` and a ``kelvinfkt`` conversion of
    ``273.19 + celsius``. The resulting flux differs by of order
    :math:`10^{-4}` in relative terms. The frozen variants of this
    mechanism route through a helper that does carry the mod file's
    constants; see :class:`Cav2p1_MA2024_PC_Frozen`.

    The mod file's ``vhalfh = -11.039 (mV)`` and
    ``cvh = 16.098 (mV)`` are declared but never used -- the
    mechanism has no inactivation state -- and BrainCell drops them.

    The mod file computes ``taum = taumfkt(v - vshift) / qt`` with
    ``qt = q10^((celsius - 23)/10)`` and ``q10 = 3``. That is
    exactly the generic gate temperature path, in which
    ``Gate(q10=..., temp_ref=...)`` divides the whole tau by the
    factor, so the Q10 is declared on the gate here rather than
    written into :meth:`f_m_tau`.

    The ``RI2021`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for this mechanism.

    ``g_max``'s default is the ``pcabar`` of the cell-model deposit
    this mechanism was imported from -- a value tuned for that
    model, not a permeability reported by either origin paper.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(23.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.2e-4 * (u.cm / u.second),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.vhalfm = -29.458 * u.mV
        self.cvm = 8.429 * u.mV
        self.z = 2

    def _shifted_voltage(self, V):
        return V - self.V_sh

    def current(self, V, Ca: IonInfo):
        drive = ghk_flux(
            V=self._shifted_voltage(V),
            ci=Ca.Ci,
            co=Ca.Co,
            z=self.z,
            temp=self.temp,
        )
        return -self.g_max * self.conductance_factor(V, Ca) * drive

    def f_m_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfm) / self.cvm))

    def f_m_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V >= -40.0,
            0.2702 + 1.1622 * u.math.exp(-((V + 26.798) ** 2) / 164.19),
            0.6923 * u.math.exp(V / 1089.372),
        )


@register_channel("Cav2p1_MA2025_BC")
class Cav2p1_MA2025_BC(Cav2p1_RI2021_SC):
    r"""Basket cell Cav2.1 P-type calcium current with GHK drive.

    The same :math:`m^3` Cav2.1 kinetics and GHK current law
    documented in :class:`Cav2p1_RI2021_SC`, reused unchanged for
    the cerebellar basket cell model of (Masoli et al., 2025) [3]_.
    The kinetics remain those built from the dissociated Purkinje
    neuron recordings of (Swensen & Bean, 2005) [1]_ and published
    in (Anwar, Hong & De Schutter, 2012) [2]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.2e-4 cm/s``. Inherited from
        :class:`Cav2p1_RI2021_SC`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``vshift``, default
        ``0.0 mV``. Inherited from :class:`Cav2p1_RI2021_SC`.
    temp : array-like, optional
        Absolute temperature entering both the gate's Q10 factor and
        the GHK flux, default 23 degrees Celsius. Inherited from
        :class:`Cav2p1_RI2021_SC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_RI2021_SC : The base class; the full equation set, the
        permeability discrepancy and the mod header corrections are
        documented there.
    Cav2p1_MA2024_PC : The same mechanism imported for the human
        Purkinje cell model.
    Cav2p1_MA2025_BC_Frozen : Basket-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class (see its Notes).

    Notes
    -----
    Ported from ``BC/channel/Cav2p1_MA25_BC.mod``, which is
    byte-identical to ``SC/channel/Cav2p1_RI21_SC.mod`` except for
    its ``SUFFIX`` line. Accordingly this class overrides nothing:
    the constructor, the gate declaration, the named parameter block
    and ``current`` are all inherited from
    :class:`Cav2p1_RI2021_SC`. Only the ``register_channel`` key and
    this docstring's model citation differ. Per the bibliography's
    attribution scan, this subclass contributes no rate-function
    code of its own; its zero literal overlap in that scan is an
    artefact of the constants living in the base class.

    The ``MA2025`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav2p1``.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2024_PC")
class Cav2p1_MA2024_PC(Cav2p1_RI2021_SC):
    r"""Purkinje cell Cav2.1 P-type calcium current, GHK drive.

    The same :math:`m^3` Cav2.1 kinetics and GHK current law
    documented in :class:`Cav2p1_RI2021_SC`, reused unchanged for
    the human Purkinje cell model of (Masoli et al., 2024) [3]_. The
    kinetics remain those built from the dissociated Purkinje neuron
    recordings of (Swensen & Bean, 2005) [1]_ and published in
    (Anwar, Hong & De Schutter, 2012) [2]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.2e-4 cm/s``. Inherited from
        :class:`Cav2p1_RI2021_SC`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``vshift``, default
        ``0.0 mV``. Inherited from :class:`Cav2p1_RI2021_SC`.
    temp : array-like, optional
        Absolute temperature entering both the gate's Q10 factor and
        the GHK flux, default 23 degrees Celsius. Inherited from
        :class:`Cav2p1_RI2021_SC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_RI2021_SC : The base class; the full equation set, the
        permeability discrepancy and the mod header corrections are
        documented there.
    Cav2p1_MA2025_BC : The same mechanism imported for the basket
        cell model.
    Cav2p1_MA2024_PC_Frozen : Purkinje-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class, and it is not
        numerically identical to it (see its Notes).

    Notes
    -----
    Ported from ``PC/channel/Cav2p1_MA24_PC.mod``, which is
    identical to ``SC/channel/Cav2p1_RI21_SC.mod`` apart from its
    ``SUFFIX`` line and the BrainCell-local ``g_equiv`` diagnostic
    the stellate copy carries. Accordingly this class overrides
    nothing: the constructor, the gate declaration, the named
    parameter block and ``current`` are all inherited from
    :class:`Cav2p1_RI2021_SC`. Only the ``register_channel`` key and
    this docstring's model citation differ. Per the bibliography's
    attribution scan, this subclass contributes no rate-function
    code of its own; its zero literal overlap in that scan is an
    artefact of the constants living in the base class.

    The ``MA2024`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav2p1``.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2024_PC_Frozen")
class Cav2p1_MA2024_PC_Frozen(HH):
    r"""Purkinje cell Cav2.1 with the GHK drive frozen for autodiff.

    A standalone Purkinje-cell Cav2.1 mechanism that stops the
    gradient through the membrane potential where it enters the
    constant-field (GHK) flux term. Its kinetics and its named
    parameter block are those of :class:`Cav2p1_RI2021_SC`, and its
    attribution is that of :class:`Cav2p1_MA2024_PC`: the Cav2.1
    kinetics built from the dissociated Purkinje neuron recordings
    of (Swensen & Bean, 2005) [1]_ and published in (Anwar, Hong &
    De Schutter, 2012) [2]_, imported for the human Purkinje cell
    model of (Masoli et al., 2024) [3]_. It inherits none of that
    code, and its forward current is **not** identical to the
    unfrozen class's -- see Notes.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), despite the ``g_max`` name. Defaults to
        ``2.2e-4 cm/s``.
    V_sh : array-like or callable, optional
        Voltage shift subtracted from :math:`V` before every rate
        and before the GHK term; the mod file's ``vshift``. Defaults
        to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature. Enters both the gate's Q10 factor and
        the GHK flux. Defaults to 23 degrees Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_MA2024_PC : The unfrozen Purkinje-cell import of the same
        mechanism. This class is **not** derived from it.
    Cav2p1_RI2021_SC : Where the equation set, the permeability
        discrepancy and the mod header corrections are documented in
        full.
    Cav2p1_RI2021_SC_Frozen : Stellate-cell frozen variant, which
        *is* a subclass of this class and overrides nothing.
    Cav2p1_MA2025_BC_Frozen : Basket-cell frozen variant, likewise a
        subclass of this class overriding nothing.

    Notes
    -----
    **This class is not a subclass of the unfrozen Purkinje-cell
    import.** It derives from
    :class:`~braincell.channel._base.HH` directly and re-declares
    everything: ``root_type``, the gate, the whole constructor and
    parameter block, ``f_m_inf``, ``f_m_tau`` and ``current``. A
    change to :class:`Cav2p1_RI2021_SC` therefore does not propagate
    here. The two Cav2.1 frozen siblings,
    :class:`Cav2p1_RI2021_SC_Frozen` and
    :class:`Cav2p1_MA2025_BC_Frozen`, are subclasses of *this* class
    and add nothing but a registry key.

    "Frozen" describes the gradient path, not the channel states.
    The ``m`` gate keeps integrating exactly as in the unfrozen
    class -- nothing stops evolving. :meth:`current` passes the
    membrane potential through the module-level
    ``_freeze_quantity_gradient``, which rebuilds the quantity from
    :func:`jax.lax.stop_gradient` applied to its mantissa, before
    handing it to the GHK helper. The unfrozen ``V`` still reaches
    :meth:`~braincell.channel._base.HH.conductance_factor`, but that
    method ignores its voltage argument and reads only the gate
    states, so the distinction affects neither value nor gradient
    there.

    **The freezing is not the only difference from the unfrozen
    class.** :meth:`current` here calls the module-level
    ``_cav3p1_nmodl_ghk_flux`` helper, whereas
    :class:`Cav2p1_RI2021_SC` calls the shared
    :func:`~braincell.channel._base.ghk_flux`. The two helpers use
    different physical constants: the shared one uses CODATA values
    and the temperature as passed, while the NMODL helper uses
    ``F = 9.6485e4 C/mol``, ``R = 8.3145 J/(K mol)`` and a
    ``273.19 + celsius`` Kelvin conversion carried as
    ``_CAV3P1_NMODL_TEMP_OFFSET``. Those are exactly the constants
    ``Cav2p1_MA24_PC.mod`` itself declares, so the helper's
    ``_cav3p1_`` name -- inherited from the Cav3.1 import, whose mod
    file declares the same values -- understates its applicability
    here. The consequence is that this class's forward current
    differs from :class:`Cav2p1_MA2024_PC`'s by of order
    :math:`10^{-4}` in relative terms, and that this class is the
    closer of the two to the mod file. Nothing in the mod file
    corresponds to the freezing itself: it is a BrainCell autodiff
    facility with no NMODL counterpart, so it is not an import
    deviation and changes nothing a NEURON comparison would observe.

    As in :class:`Cav2p1_RI2021_SC`, ``g_max`` is a permeability in
    ``cm/s`` rather than a conductance density, the mod file's
    unused ``vhalfh``/``cvh`` are dropped, the Q10 is declared on
    the ``Gate`` because the mod file divides the whole tau by it,
    and ``current()`` negates NEURON's outward-positive ``ica`` to
    match BrainCell's inward-positive convention. The permeability
    default is the deposit's tuned ``pcabar``, not a value reported
    by either origin paper.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(23.0)),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 2.2e-4 * (u.cm / u.second),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(23.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.vhalfm = -29.458 * u.mV
        self.cvm = 8.429 * u.mV
        self.z = 2

    def _shifted_voltage(self, V):
        return V - self.V_sh

    def current(self, V, Ca: IonInfo):
        frozen_V = _freeze_quantity_gradient(V)
        drive = _cav3p1_nmodl_ghk_flux(
            V=self._shifted_voltage(frozen_V),
            ci=Ca.Ci,
            co=Ca.Co,
            z=self.z,
            temp=self.temp,
        )
        return -self.g_max * self.conductance_factor(V, Ca) * drive

    def f_m_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfm) / self.cvm))

    def f_m_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V >= -40.0,
            0.2702 + 1.1622 * u.math.exp(-((V + 26.798) ** 2) / 164.19),
            0.6923 * u.math.exp(V / 1089.372),
        )


@register_channel("Cav2p1_RI2021_SC_Frozen")
class Cav2p1_RI2021_SC_Frozen(Cav2p1_MA2024_PC_Frozen):
    r"""Stellate cell Cav2.1 with the GHK drive frozen for autodiff.

    The stellate-cell registration of the frozen-GHK Cav2.1
    mechanism implemented by :class:`Cav2p1_MA2024_PC_Frozen`. The
    kinetics are those built from the dissociated Purkinje neuron
    recordings of (Swensen & Bean, 2005) [1]_ and published in
    (Anwar, Hong & De Schutter, 2012) [2]_, imported here for the
    cerebellar stellate cell model of (Rizza et al., 2021) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.2e-4 cm/s``. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``vshift``, default
        ``0.0 mV``. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    temp : array-like, optional
        Absolute temperature entering both the gate's Q10 factor and
        the GHK flux, default 23 degrees Celsius. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_MA2024_PC_Frozen : The base class; the freezing
        mechanism, the GHK helper's constants and the difference
        from the unfrozen classes are documented there.
    Cav2p1_RI2021_SC : The unfrozen stellate-cell import of the same
        mechanism. This class is **not** derived from it, and is not
        numerically identical to it.

    Notes
    -----
    **The cell-type suffix in this class's name describes its model
    citation, not its implementation.** The class body is empty
    apart from ``__module__``: every gate, constant, rate function
    and the ``current`` method come from
    :class:`Cav2p1_MA2024_PC_Frozen`, the Purkinje-cell frozen
    variant. That is sound because ``SC/channel/Cav2p1_RI21_SC.mod``
    and ``PC/channel/Cav2p1_MA24_PC.mod`` are identical apart from
    their ``SUFFIX`` lines and a ``g_equiv`` diagnostic, but it does
    mean the stellate-cell class inherits from a Purkinje-cell one
    rather than from :class:`Cav2p1_RI2021_SC`. Per the
    bibliography's attribution scan this subclass contributes no
    rate-function code of its own; its zero literal overlap in that
    scan is an artefact of the constants living in the base class.

    Everything in :class:`Cav2p1_MA2024_PC_Frozen`'s Notes applies
    unchanged and is not repeated here: no channel state stops
    evolving, the freezing affects reverse-mode gradients only, and
    the frozen path uses the NMODL-constant GHK helper rather than
    the shared :func:`~braincell.channel._base.ghk_flux`, so the
    forward current differs slightly from
    :class:`Cav2p1_RI2021_SC`'s.

    The ``RI2021`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav2p1``.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2025_BC_Frozen")
class Cav2p1_MA2025_BC_Frozen(Cav2p1_MA2024_PC_Frozen):
    r"""Basket cell Cav2.1 with the GHK drive frozen for autodiff.

    The basket-cell registration of the frozen-GHK Cav2.1 mechanism
    implemented by :class:`Cav2p1_MA2024_PC_Frozen`. The kinetics
    are those built from the dissociated Purkinje neuron recordings
    of (Swensen & Bean, 2005) [1]_ and published in (Anwar, Hong &
    De Schutter, 2012) [2]_, imported here for the cerebellar basket
    cell model of (Masoli et al., 2025) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``2.2e-4 cm/s``. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``vshift``, default
        ``0.0 mV``. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    temp : array-like, optional
        Absolute temperature entering both the gate's Q10 factor and
        the GHK flux, default 23 degrees Celsius. Inherited from
        :class:`Cav2p1_MA2024_PC_Frozen`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav2p1_MA2024_PC_Frozen : The base class; the freezing
        mechanism, the GHK helper's constants and the difference
        from the unfrozen classes are documented there.
    Cav2p1_MA2025_BC : The unfrozen basket-cell import of the same
        mechanism. This class is **not** derived from it, and is not
        numerically identical to it.

    Notes
    -----
    **The cell-type suffix in this class's name describes its model
    citation, not its implementation.** The class body is empty
    apart from ``__module__``: every gate, constant, rate function
    and the ``current`` method come from
    :class:`Cav2p1_MA2024_PC_Frozen`, the Purkinje-cell frozen
    variant. That is sound because ``BC/channel/Cav2p1_MA25_BC.mod``
    and ``PC/channel/Cav2p1_MA24_PC.mod`` are identical apart from
    their ``SUFFIX`` lines, but it does mean the basket-cell class
    inherits from a Purkinje-cell one rather than from
    :class:`Cav2p1_MA2025_BC`. Per the bibliography's attribution
    scan this subclass contributes no rate-function code of its own;
    its zero literal overlap in that scan is an artefact of the
    constants living in the base class.

    Everything in :class:`Cav2p1_MA2024_PC_Frozen`'s Notes applies
    unchanged and is not repeated here: no channel state stops
    evolving, the freezing affects reverse-mode gradients only, and
    the frozen path uses the NMODL-constant GHK helper rather than
    the shared :func:`~braincell.channel._base.ghk_flux`, so the
    forward current differs slightly from
    :class:`Cav2p1_MA2025_BC`'s.

    The ``MA2025`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav2p1``.

    References
    ----------
    .. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
           firing in dissociated Purkinje neurons with acute or
           long-term reductions in sodium conductance. The Journal
           of Neuroscience, 25(14), 3509-3520.
           doi:10.1523/JNEUROSCI.3929-04.2005
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Cav3p3_MA2024_PC_Frozen")
class Cav3p3_MA2024_PC_Frozen(HH):
    r"""Purkinje cell Cav3.3 with the GHK drive frozen for autodiff.

    A standalone Purkinje-cell Cav3.3 mechanism that stops the
    gradient through the membrane potential where it enters the
    constant-field (GHK) flux term. Its kinetics, its named
    parameter block and its forward current are numerically
    identical to :class:`Cav3p3_MA2024_PC`'s, and its attribution is
    the same: the Cav3.3 kinetics of the CA3 hippocampal pyramidal
    neuron model of (Xu & Clancy, 2008) [1]_, imported for the human
    Purkinje cell model of (Masoli et al., 2024) [2]_. It inherits
    none of that code -- see Notes.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    perm : array-like or callable, optional
        Calcium permeability entering the GHK flux, the mod file's
        ``pcabar``. Defaults to ``1.0e-4 cm/s``.
    g_scale : array-like or callable, optional
        Dimensionless empirical scale factor multiplying the flux,
        carrying the numeric value of the mod file's
        ``gCav3_3bar``. Defaults to ``1.0e-5``.
    temp : array-like, optional
        Absolute temperature. Enters both the gates' Q10 factor and
        the GHK flux. Defaults to 36 degrees Celsius.
    V_sh : array-like or callable, optional
        Voltage shift subtracted from :math:`V` before every rate
        and before the GHK term. Defaults to ``0.0 mV``. The mod
        file declares no corresponding parameter.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p3_MA2024_PC : The unfrozen Purkinje-cell import of the same
        mechanism. This class is **not** derived from it.
    Cav3p3_RI2021_SC : Where the equation set, the GHK helper's
        constants and the current-law scaling caveat are documented
        in full.
    Cav2p1_MA2024_PC_Frozen : The module's other standalone frozen
        variant, which additionally swaps GHK helpers and so is
        *not* numerically identical to its unfrozen counterpart.

    Notes
    -----
    **This class is not a subclass of the unfrozen Purkinje-cell
    import, and the module's frozen variants are not all built the
    same way.** :class:`Cav3p3_MA2024_PC` derives from
    :class:`Cav3p3_RI2021_SC`; this class derives from
    :class:`~braincell.channel._base.HH` directly and re-declares
    everything: ``root_type``, both gates, the whole constructor and
    parameter block, ``f_n_inf``, ``f_l_inf``, ``f_n_tau``,
    ``f_l_tau`` and ``current``. The two therefore compute the same
    forward numbers while sharing no code, and a change to
    :class:`Cav3p3_RI2021_SC` propagates to one and not the other.

    "Frozen" describes the gradient path, not the forward numerics
    and not the channel states. Both gates keep integrating normally
    -- nothing stops evolving. :meth:`current` passes the membrane
    potential through the module-level
    ``_freeze_quantity_gradient``, which rebuilds the quantity from
    :func:`jax.lax.stop_gradient` applied to its mantissa, before
    handing it to the GHK helper. The unfrozen ``V`` still reaches
    :meth:`~braincell.channel._base.HH.conductance_factor`, but that
    method ignores its voltage argument and reads only the gate
    states, so the distinction affects neither value nor gradient
    there. **Unlike** :class:`Cav2p1_MA2024_PC_Frozen`, this class
    calls the same ``_cav3p3_nmodl_ghk_flux`` helper its unfrozen
    counterpart calls, so freezing is the only difference between
    the two and the forward current is unchanged.

    The kinetics are those of ``PC/channel/Cav3p3_MA24_PC.mod``,
    which is identical to ``SC/channel/Cav3p3_RI21_SC.mod`` apart
    from its ``SUFFIX`` line and a ``g_equiv`` diagnostic. No mod
    file ships a frozen-gradient variant: freezing is a BrainCell
    autodiff facility with no counterpart in NMODL, so it is not an
    import deviation and changes nothing a NEURON comparison would
    observe.

    As in :class:`Cav3p3_RI2021_SC`, the mod file's current-law
    scaling is not dimensionally self-consistent and ``g_scale`` is
    therefore dimensionless, the GHK helper carries the mod file's
    own ``F``/``R`` constants and its ``celsius + 273.14`` Kelvin
    conversion plus a small-argument series branch the mod file
    lacks, ``V_sh`` has no counterpart in the mod file, the Q10 is
    declared on the ``Gate`` objects, and ``current()`` negates
    NEURON's outward-positive ``ica``. The ``perm`` and ``g_scale``
    defaults are the deposit's tuned values, not values reported by
    the origin paper.

    References
    ----------
    .. [1] Xu, J., & Clancy, C. E. (2008). Ionic mechanisms of
           endogenous bursting in CA3 hippocampal pyramidal neurons:
           A model study. PLoS ONE, 3(4), e2056.
           doi:10.1371/journal.pone.0002056
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("n", power=2, q10=2.3, temp_ref=u.celsius2kelvin(28.0)),
        Gate("l", q10=2.3, temp_ref=u.celsius2kelvin(28.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        perm: Union[brainstate.typing.ArrayLike, Callable] = 1.0e-4 * (u.cm / u.second),
        g_scale: Union[brainstate.typing.ArrayLike, Callable] = 1.0e-5,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.perm = braintools.init.param(perm, self.varshape, allow_none=False)
        self.g_scale = braintools.init.param(g_scale, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.vhalfn = -41.5 * u.mV
        self.vhalfl = -69.8 * u.mV
        self.kn = 6.2 * u.mV
        self.kl = -6.1 * u.mV
        self.z = 2

    def _shifted_voltage(self, V):
        return V - self.V_sh

    def current(self, V, Ca: IonInfo):
        frozen_V = _freeze_quantity_gradient(V)
        drive = _cav3p3_nmodl_ghk_flux(
            V=self._shifted_voltage(frozen_V),
            ci=Ca.Ci,
            co=Ca.Co,
            z=self.z,
            temp=self.temp,
        )
        return -self.g_scale * self.perm * self.conductance_factor(V, Ca) * drive

    def f_n_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfn) / self.kn))

    def f_l_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfl) / self.kl))

    def f_n_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V > -60.0,
            7.2 + 0.02 * u.math.exp(-V / 14.7),
            0.875 * u.math.exp((V + 120.0) / 41.0),
        )

    def f_l_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V > -60.0,
            79.5 + 2.0 * u.math.exp(-V / 9.3),
            260.0,
        )


@register_channel("Cav3p2_RI2021_SC")
class Cav3p2_RI2021_SC(OhmicHH):
    r"""Stellate cell Cav3.2 low-threshold T-type calcium current.

    The Cav3.2 (T-type, alpha1H) low-threshold calcium current of
    the cerebellar stellate cell model of (Rizza et al., 2021) [4]_.
    It is Destexhe's 1992 NEURON implementation of the
    low-threshold calcium current of (Huguenard & McCormick, 1992)
    [1]_, with the biophysical properties refitted to recordings of
    human recombinant Cav3.2 channels in HEK-293 cells by (Vitko et
    al., 2005) [2]_ and transformed from those 23-25 degrees Celsius
    data to 36 degrees Celsius using Q10 factors credited to
    (Coulter, Huguenard & Prince, 1989) [3]_ (see Notes). Gating is
    :math:`m^2 h` with an ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, m^2 h \, (E_{Ca} - V) \\
        m_\infty &= \frac{1}{1 + \exp(-(V' + 54.8) / 7.4)} \\
        h_\infty &= \frac{1}{1 + \exp((V' + 85.5) / 7.18)} \\
        \tau_m &= \frac{1}{\phi_m}\left(1.9 +
                  \frac{1}{e^{(V' + 37)/11.9}
                  + e^{-(V' + 131.6)/21}}\right) \\
        \tau_h &= 13.7 + \frac{1}{\phi_h}
                  \cdot \frac{1942 + e^{(V' + 164)/9.2}}
                  {1 + e^{(V' + 89.3)/3.7}} \\
        \phi_m &= 5^{(36 - 24)/10}, \quad
        \phi_h = 3^{(36 - 24)/10}
        \end{aligned}

    where :math:`V' = V + V_{sh}` read in millivolts and the time
    constants are in milliseconds. :math:`\phi_m` is supplied by the
    ``m`` gate's fixed ``phi``, while :math:`\phi_h` is written
    directly into :meth:`f_h_tau` and the ``h`` gate's ``phi`` is
    left at 1 -- because the additive ``13.7`` sits outside the
    division and so does not fit the template's uniform
    ``tau / phi`` shape. Both factors are constants, not functions
    of ``temp``; see Notes.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to
        ``8.0e-4 mS/cm2``, which is **not** the mod file's
        ``gcabar`` converted (see Notes).
    V_sh : array-like or callable, optional
        Voltage shift added to :math:`V` before every rate, the mod
        file's ``shift``. Defaults to ``0.0 mV``.
    temp : array-like, optional
        Absolute temperature. Accepted and stored, but read by no
        method of this class (see Notes). Defaults to 36 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p2_MA2024_PC : The same mechanism re-imported for the human
        Purkinje cell model; identical kinetics, different model
        citation.
    Cav3p2_MA2025_BC : The same mechanism re-imported for the basket
        cell model; identical kinetics, different model citation.
    CaT_HM1992 : The other import in this module tracing to
        (Huguenard & McCormick, 1992), by way of a different
        Destexhe implementation and with different constants.
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``SC/channel/Cav3p2_RI21_SC.mod``, whose header
    reads "Model of Huguenard & McCormick, J Neurophysiol 68:
    1373-1383, 1992", "Written by Alain Destexhe, Salk Institute,
    Sept 18, 1992" and "Biophysical properties of the T current were
    from recordings of human recombinant Cav3.2 T-channel in HEK-293
    cells -- see Vitko et al." It also records the rename "Suffix
    from CaT3_2 to Cav3_2". The Destexhe authorship line names the
    mechanism's implementer and is deliberately not turned into a
    citation; entry [1]_ is the paper his implementation models.
    Note that the Boltzmann and tau constants above are the Vitko
    refit, not the numbers of the original 1992 parameterisation.

    **What the Q10 citation does and does not support.** The mod
    file's ``INITIAL`` block comments the 24-to-36 degrees Celsius
    transformation as "assuming Q10 of 5 and 3 for m and h (as in
    Coulter et al., J Physiol 414: 587, 1989)". Entry [3]_ is a
    Q10 source only: it reports that the low-threshold current's
    kinetic properties were temperature sensitive with Q10 values
    greater than 2.5, and does **not** print the specific 5 and 3
    used here. That split is Destexhe's parameterisation derived
    from those data, and this docstring does not present [3]_ as a
    source of kinetics.

    **The temperature conversion is baked in, and ``temp`` is
    dead.** The mod file computes ``phi_m = 5^(12/10)`` and
    ``phi_h = 3^(12/10)`` once, from the fixed literals 36 and 24
    rather than from NEURON's ``celsius``. BrainCell reproduces that
    exactly: the ``m`` gate carries ``phi=5.0 ** ((36.0 - 24.0) /
    10.0)`` and :meth:`f_h_tau` recomputes the matching ``phi_h``
    inline. Consequently the ``temp`` constructor parameter is
    stored on the instance and never read -- neither gate declares a
    ``q10``, and this mechanism has no GHK term for ``temp`` to
    enter. Changing ``temp`` changes nothing.

    **``g_max``'s default does not match the mod file.** The mod
    file declares ``gcabar = .0008 (mho/cm2)``, i.e. 0.8 mS/cm2;
    this class defaults to ``8.0e-4 mS/cm2``, the same numeric
    literal carrying the millisiemens unit, which is a thousand
    times smaller. Sibling imports in this module resolve the same
    ``mho/cm2`` declaration the other way -- ``CaHVA_MA2020_GoC``
    turns ``0.00046 mho/cm2`` into ``0.46 mS/cm2``, and
    ``Cav1p2_MA2020_GoC`` keeps ``0.0002 S/cm2`` outright. The
    divergence is recorded here rather than corrected: this is a
    documentation-only description of the shipped default. It is
    invisible to the NEURON comparison suite, which always passes
    ``g_max`` explicitly in ``S/cm2``. Note also that even a
    correctly converted default would be the cell-model deposit's
    tuned ``gcabar``, not a conductance reported by any of the
    origin papers.

    **The mod file's fixed calcium concentrations are not read
    here.** It declares ``cai = 2.4e-4 (mM)`` and ``cao = 2 (mM)``,
    computes its own reversal potential from them by the Nernst
    equation, and notes that ``cai`` was "adjusted for eca=120 mV".
    :class:`~braincell.channel._base.OhmicHH` instead takes
    :math:`E_{Ca}` from the attached
    :class:`~braincell.ion.Calcium` ion object, so the comparison
    path has to pin those concentrations externally to reproduce the
    mod file's driving force.

    Taken together -- the hard-coded 36 degrees Celsius conversion,
    the externally pinned concentrations and the irregular
    ``tau_h`` shape -- this mod file is not a clean reusable
    temperature- and concentration-general mechanism. The
    implementation here preserves its quirks deliberately, so that
    BrainCell matches NEURON one for one; a more general rewrite
    would have to break that correspondence.

    NEURON's raw ``ica`` here is ``gcabar * m*m*h * (v - carev)``,
    i.e. outward-positive;
    :class:`~braincell.channel._base.OhmicHH` computes
    ``g_max * m^2 h * (E - V)``, the same current under BrainCell's
    repo-wide inward-positive convention.

    The ``RI2021`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for this mechanism.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    .. [2] Vitko, I., Chen, Y., Arias, J. M., Shen, Y., Wu, X.-R., &
           Perez-Reyes, E. (2005). Functional characterization and
           neuronal modeling of the effects of childhood absence
           epilepsy variants of CACNA1H, a T-type calcium channel.
           The Journal of Neuroscience, 25(19), 4844-4855.
           doi:10.1523/JNEUROSCI.0847-05.2005
    .. [3] Coulter, D. A., Huguenard, J. R., & Prince, D. A. (1989).
           Calcium currents in rat thalamocortical relay neurones:
           kinetic properties of the transient, low-threshold
           current. The Journal of Physiology, 414(1), 587-604.
           doi:10.1113/jphysiol.1989.sp017705
    .. [4] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", power=2, phi=5.0 ** ((36.0 - 24.0) / 10.0)),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 8.0e-4 * (u.mS / u.cm**2),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)

    def _shifted_voltage(self, V):
        return V + self.V_sh

    def f_m_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 54.8) / 7.4))

    def f_h_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 85.5) / 7.18))

    def f_m_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return 1.9 + 1.0 / (u.math.exp((V + 37.0) / 11.9) + u.math.exp(-(V + 131.6) / 21.0))

    def f_h_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        # Source mod writes:
        #   tau_h = 13.7 + (1942 + exp(...)) / (1 + exp(...)) / phi_h
        # which is not the usual "tau / phi" shape handled by HH.gate_phi().
        # We therefore keep h-gate phi at 1 and encode the fixed 36C
        # conversion directly in tau_h here.
        phi_h = 3.0 ** ((36.0 - 24.0) / 10.0)
        term = (1942.0 + u.math.exp((V + 164.0) / 9.2)) / (1.0 + u.math.exp((V + 89.3) / 3.7))
        return 13.7 + term / phi_h


@register_channel("Cav3p2_MA2025_BC")
class Cav3p2_MA2025_BC(Cav3p2_RI2021_SC):
    r"""Basket cell Cav3.2 low-threshold T-type calcium current.

    The same :math:`m^2 h` Cav3.2 kinetics and ohmic current law
    documented in :class:`Cav3p2_RI2021_SC`, reused unchanged for
    the cerebellar basket cell model of (Masoli et al., 2025) [4]_.
    The kinetics remain Destexhe's 1992 implementation of the
    low-threshold current of (Huguenard & McCormick, 1992) [1]_,
    refitted to the human recombinant Cav3.2 recordings of (Vitko et
    al., 2005) [2]_ and transformed to 36 degrees Celsius with Q10
    factors credited to (Coulter, Huguenard & Prince, 1989) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``8.0e-4 mS/cm2``,
        which is not the mod file's ``gcabar`` converted (see
        :class:`Cav3p2_RI2021_SC` Notes). Inherited from
        :class:`Cav3p2_RI2021_SC`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``shift``, default
        ``0.0 mV``. Inherited from :class:`Cav3p2_RI2021_SC`.
    temp : array-like, optional
        Accepted but read by no method of this class (see
        :class:`Cav3p2_RI2021_SC` Notes). Default 36 degrees
        Celsius. Inherited from :class:`Cav3p2_RI2021_SC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p2_RI2021_SC : The base class; the full equation set, the
        baked-in temperature conversion, the dead ``temp``
        parameter and the ``g_max`` default divergence are
        documented there.
    Cav3p2_MA2024_PC : The same mechanism imported for the human
        Purkinje cell model.

    Notes
    -----
    Ported from ``BC/channel/Cav3p2_MA25_BC.mod``, which is
    byte-identical to ``SC/channel/Cav3p2_RI21_SC.mod`` except for
    its ``SUFFIX`` line. Accordingly this class overrides nothing:
    the constructor, both gate declarations and the four rate
    methods are all inherited from :class:`Cav3p2_RI2021_SC`. Only
    the ``register_channel`` key and this docstring's model citation
    differ. Per the bibliography's attribution scan, this subclass
    contributes no rate-function code of its own; its zero literal
    overlap in that scan is an artefact of the constants living in
    the base class.

    The ``MA2025`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav3p2``.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    .. [2] Vitko, I., Chen, Y., Arias, J. M., Shen, Y., Wu, X.-R., &
           Perez-Reyes, E. (2005). Functional characterization and
           neuronal modeling of the effects of childhood absence
           epilepsy variants of CACNA1H, a T-type calcium channel.
           The Journal of Neuroscience, 25(19), 4844-4855.
           doi:10.1523/JNEUROSCI.0847-05.2005
    .. [3] Coulter, D. A., Huguenard, J. R., & Prince, D. A. (1989).
           Calcium currents in rat thalamocortical relay neurones:
           kinetic properties of the transient, low-threshold
           current. The Journal of Physiology, 414(1), 587-604.
           doi:10.1113/jphysiol.1989.sp017705
    .. [4] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Cav3p2_MA2024_PC")
class Cav3p2_MA2024_PC(Cav3p2_RI2021_SC):
    r"""Purkinje cell Cav3.2 low-threshold T-type calcium current.

    The same :math:`m^2 h` Cav3.2 kinetics and ohmic current law
    documented in :class:`Cav3p2_RI2021_SC`, reused unchanged for
    the human Purkinje cell model of (Masoli et al., 2024) [4]_. The
    kinetics remain Destexhe's 1992 implementation of the
    low-threshold current of (Huguenard & McCormick, 1992) [1]_,
    refitted to the human recombinant Cav3.2 recordings of (Vitko et
    al., 2005) [2]_ and transformed to 36 degrees Celsius with Q10
    factors credited to (Coulter, Huguenard & Prince, 1989) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``8.0e-4 mS/cm2``,
        which is not the mod file's ``gcabar`` converted (see
        :class:`Cav3p2_RI2021_SC` Notes). Inherited from
        :class:`Cav3p2_RI2021_SC`.
    V_sh : array-like or callable, optional
        Voltage shift, the mod file's ``shift``, default
        ``0.0 mV``. Inherited from :class:`Cav3p2_RI2021_SC`.
    temp : array-like, optional
        Accepted but read by no method of this class (see
        :class:`Cav3p2_RI2021_SC` Notes). Default 36 degrees
        Celsius. Inherited from :class:`Cav3p2_RI2021_SC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p2_RI2021_SC : The base class; the full equation set, the
        baked-in temperature conversion, the dead ``temp``
        parameter and the ``g_max`` default divergence are
        documented there.
    Cav3p2_MA2025_BC : The same mechanism imported for the basket
        cell model.

    Notes
    -----
    Ported from ``PC/channel/Cav3p2_MA24_PC.mod``, which is
    identical to ``SC/channel/Cav3p2_RI21_SC.mod`` apart from its
    ``SUFFIX`` line and the BrainCell-local ``g_equiv`` diagnostic
    the stellate copy carries. Accordingly this class overrides
    nothing: the constructor, both gate declarations and the four
    rate methods are all inherited from
    :class:`Cav3p2_RI2021_SC`. Only the ``register_channel`` key and
    this docstring's model citation differ. Per the bibliography's
    attribution scan, this subclass contributes no rate-function
    code of its own; its zero literal overlap in that scan is an
    artefact of the constants living in the base class.

    The ``MA2024`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav3p2``.

    References
    ----------
    .. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of
           the currents involved in rhythmic oscillations in thalamic
           relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
           doi:10.1152/jn.1992.68.4.1373
    .. [2] Vitko, I., Chen, Y., Arias, J. M., Shen, Y., Wu, X.-R., &
           Perez-Reyes, E. (2005). Functional characterization and
           neuronal modeling of the effects of childhood absence
           epilepsy variants of CACNA1H, a T-type calcium channel.
           The Journal of Neuroscience, 25(19), 4844-4855.
           doi:10.1523/JNEUROSCI.0847-05.2005
    .. [3] Coulter, D. A., Huguenard, J. R., & Prince, D. A. (1989).
           Calcium currents in rat thalamocortical relay neurones:
           kinetic properties of the transient, low-threshold
           current. The Journal of Physiology, 414(1), 587-604.
           doi:10.1113/jphysiol.1989.sp017705
    .. [4] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("Cav3p3_RI2021_SC")
class Cav3p3_RI2021_SC(HH):
    r"""Stellate cell Cav3.3 low-threshold calcium current, GHK drive.

    The Cav3.3 (T-type, alpha1I) low-threshold calcium current of
    the cerebellar stellate cell model of (Rizza et al., 2021) [2]_.
    Its kinetics are those of the CA3 hippocampal pyramidal neuron
    model of (Xu & Clancy, 2008) [1]_, reused unchanged for the
    stellate cell. Gating is :math:`n^2 l`, driven by a
    hand-written constant-field (GHK) calcium flux rather than an
    ohmic term:

    .. math::

        \begin{aligned}
        I_{Ca} &= -s \, P \, n^2 l \,
                  \Phi(V', [Ca]_i, [Ca]_o, z{=}2, T) \\
        n_\infty &= \frac{1}{1 + \exp(-(V' - v_{1/2,n}) / k_n)} \\
        l_\infty &= \frac{1}{1 + \exp(-(V' - v_{1/2,l}) / k_l)} \\
        \tau_n &= \frac{1}{\phi} \begin{cases}
                  7.2 + 0.02\, e^{-V' / 14.7} & V' > -60 \\
                  0.875\, e^{(V' + 120) / 41} & V' \leq -60
                  \end{cases} \\
        \tau_l &= \frac{1}{\phi} \begin{cases}
                  79.5 + 2\, e^{-V' / 9.3} & V' > -60 \\
                  260 & V' \leq -60
                  \end{cases} \\
        \phi &= 2.3^{(T - 28\,^\circ\mathrm{C}) / 10}
        \end{aligned}

    where :math:`V' = V - V_{sh}`, :math:`v_{1/2,n} = -41.5` mV,
    :math:`k_n = 6.2` mV, :math:`v_{1/2,l} = -69.8` mV,
    :math:`k_l = -6.1` mV, :math:`s` is ``g_scale``, :math:`P` is
    ``perm`` and :math:`\Phi` is the module-level
    ``_cav3p3_nmodl_ghk_flux`` helper (see Notes). The tau branches
    read :math:`V'` in millivolts and return milliseconds; both are
    exclusive at exactly :math:`-60` mV, and neither is continuous
    across that point -- at :math:`V' = -60` mV the upper branches
    give about 8.4 ms and 1347 ms against the lower branches' 3.8 ms
    and 260 ms. That discontinuity is the mod file's, reproduced
    rather than smoothed. :math:`\phi` is supplied by the ``Gate``
    declarations, which divide each tau by it.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    perm : array-like or callable, optional
        Calcium permeability :math:`P` entering the GHK flux, the
        mod file's ``pcabar``. Defaults to ``1.0e-4 cm/s`` (see
        Notes).
    g_scale : array-like or callable, optional
        Dimensionless empirical scale factor multiplying the flux,
        carrying the numeric value of the mod file's
        ``gCav3_3bar``. Defaults to ``1.0e-5`` (see Notes).
    temp : array-like, optional
        Absolute temperature. Enters both :math:`\phi` and the GHK
        flux. Defaults to 36 degrees Celsius.
    V_sh : array-like or callable, optional
        Voltage shift subtracted from :math:`V` before every rate
        and before the GHK term. Defaults to ``0.0 mV``. The mod
        file declares no corresponding parameter (see Notes).
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p3_MA2024_PC : The same mechanism re-imported for the human
        Purkinje cell model; identical kinetics, different model
        citation.
    Cav3p3_MA2024_PC_Frozen : Purkinje-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class (see its Notes).
    braincell.channel._base.ghk_flux : The shared GHK helper, which
        this class does **not** call (see Notes).

    Notes
    -----
    Ported from ``SC/channel/Cav3p3_RI21_SC.mod``, whose ``TITLE``
    reads "CaV 3.3 CA3 hippocampal neuron" and whose ``COMMENT``
    records "Created by jun xu @ Clancy Lab of Cornell University
    Medical College" and cites (Xu & Clancy, 2008). The "CA3
    hippocampal neuron" of the title is upstream provenance: this
    class is the stellate cell port, used in the model cited as
    [2]_.

    **The mod file's current-law scaling is not dimensionally
    self-consistent, and BrainCell's parameter units follow from
    that.** The file mixes ``pcabar`` (declared like a permeability,
    ``cm/s``), ``gCav3_3bar`` (declared like a conductance density,
    ``S/cm2``) and a hand-written GHK expression carrying its own
    built-in unit conversion constants, then multiplies all three
    together. To keep the BrainCell current law dimensionally
    consistent while still matching NEURON numerically, ``g_scale``
    is treated here as a dimensionless empirical scale factor rather
    than as a physical conductance density; ``perm`` remains the
    permeability-like term.

    ``current()`` evaluates the constant-field equation through the
    module-level ``_cav3p3_nmodl_ghk_flux`` helper rather than
    :func:`~braincell.channel._base.ghk_flux`, so that it reproduces
    this mod file's own constants exactly: ``F = 96520 C/mol``,
    ``R = 8.3134 J/(K mol)`` and the mod file's
    ``T = celsius + 273.14`` conversion, whose 0.01 K offset from
    :func:`brainunit.celsius2kelvin` is carried as
    ``_CAV3P3_NMODL_TEMP_OFFSET``. The helper writes the flux as
    :math:`-zF(c_o - c_i e^{w}) w / (e^{w} - 1)` with
    :math:`w = zFV'/(RT)`, matching the mod file line for line, and
    adds one thing the mod file has no counterpart for: a
    small-:math:`w` series branch
    :math:`-zF(c_o - c_i e^{w})(1 - w/2)`, taken when
    :math:`|e^{w} - 1| < 10^{-6}`, which avoids the division by
    zero the mod file's expression has at :math:`V' = 0`.
    ``current()`` negates the result to match BrainCell's repo-wide
    inward-positive convention against NEURON's outward-positive
    ``ica``.

    ``V_sh`` has no counterpart in the mod file at all; it is a
    BrainCell extension. Unlike the ``V_sh`` of the Cav3.1 classes
    in this module it *is* read -- by all four rate methods and by
    :meth:`current` -- but its ``0.0 mV`` default leaves the
    mechanism at the mod file's behaviour.

    The mod file applies its temperature factor as
    ``tau = tau / qt`` with ``qt = q10^((celsius - 28)/10)`` and
    ``q10 = 2.3``. That is exactly the generic gate temperature
    path, so the Q10 is declared on the ``Gate`` objects here rather
    than written into :meth:`f_n_tau` and :meth:`f_l_tau`. The mod
    file reads NEURON's global ``celsius`` where this class reads
    its own ``temp`` parameter, which defaults to 36 degrees
    Celsius.

    The mod file writes ``vhalfn``, ``vhalfl``, ``kn`` and ``kl`` as
    bare numbers with a ``:mv`` comment rather than as united
    quantities; BrainCell attaches ``u.mV`` to all four.

    The ``RI2021`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for this mechanism.

    ``perm`` and ``g_scale`` default to the ``pcabar`` and
    ``gCav3_3bar`` of the cell-model deposit this mechanism was
    imported from -- values tuned for that model, not values
    reported by the origin paper.

    References
    ----------
    .. [1] Xu, J., & Clancy, C. E. (2008). Ionic mechanisms of
           endogenous bursting in CA3 hippocampal pyramidal neurons:
           A model study. PLoS ONE, 3(4), e2056.
           doi:10.1371/journal.pone.0002056
    .. [2] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("n", power=2, q10=2.3, temp_ref=u.celsius2kelvin(28.0)),
        Gate("l", q10=2.3, temp_ref=u.celsius2kelvin(28.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        perm: Union[brainstate.typing.ArrayLike, Callable] = 1.0e-4 * (u.cm / u.second),
        g_scale: Union[brainstate.typing.ArrayLike, Callable] = 1.0e-5,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(36.0),
        V_sh: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.perm = braintools.init.param(perm, self.varshape, allow_none=False)
        self.g_scale = braintools.init.param(g_scale, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(V_sh, self.varshape, allow_none=False)
        self.vhalfn = -41.5 * u.mV
        self.vhalfl = -69.8 * u.mV
        self.kn = 6.2 * u.mV
        self.kl = -6.1 * u.mV
        self.z = 2

    def _shifted_voltage(self, V):
        return V - self.V_sh

    def current(self, V, Ca: IonInfo):
        drive = _cav3p3_nmodl_ghk_flux(
            V=self._shifted_voltage(V),
            ci=Ca.Ci,
            co=Ca.Co,
            z=self.z,
            temp=self.temp,
        )
        return -self.g_scale * self.perm * self.conductance_factor(V, Ca) * drive

    def f_n_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfn) / self.kn))

    def f_l_inf(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V)
        return 1.0 / (1.0 + u.math.exp(-(V - self.vhalfl) / self.kl))

    def f_n_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V > -60.0,
            7.2 + 0.02 * u.math.exp(-V / 14.7),
            0.875 * u.math.exp((V + 120.0) / 41.0),
        )

    def f_l_tau(self, V, Ca: IonInfo):
        V = self._shifted_voltage(V).to_decimal(u.mV)
        return u.math.where(
            V > -60.0,
            79.5 + 2.0 * u.math.exp(-V / 9.3),
            260.0,
        )


@register_channel("Cav3p3_MA2024_PC")
class Cav3p3_MA2024_PC(Cav3p3_RI2021_SC):
    r"""Purkinje cell Cav3.3 low-threshold calcium current, GHK drive.

    The same :math:`n^2 l` Cav3.3 kinetics and hand-written GHK
    current law documented in :class:`Cav3p3_RI2021_SC`, reused
    unchanged for the human Purkinje cell model of (Masoli et al.,
    2024) [2]_. The kinetics remain those of the CA3 hippocampal
    pyramidal neuron model of (Xu & Clancy, 2008) [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    perm : array-like or callable, optional
        Calcium permeability entering the GHK flux (the mod file's
        ``pcabar``), default ``1.0e-4 cm/s``. Inherited from
        :class:`Cav3p3_RI2021_SC`.
    g_scale : array-like or callable, optional
        Dimensionless scale factor carrying the numeric value of the
        mod file's ``gCav3_3bar``, default ``1.0e-5``. Inherited
        from :class:`Cav3p3_RI2021_SC`.
    temp : array-like, optional
        Absolute temperature entering both the gates' Q10 factor and
        the GHK flux, default 36 degrees Celsius. Inherited from
        :class:`Cav3p3_RI2021_SC`.
    V_sh : array-like or callable, optional
        Voltage shift with no counterpart in the mod file, default
        ``0.0 mV``. Inherited from :class:`Cav3p3_RI2021_SC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Cav3p3_RI2021_SC : The base class; the full equation set, the
        GHK helper's constants and the current-law scaling caveat
        are documented there.
    Cav3p3_MA2024_PC_Frozen : Purkinje-cell variant with the GHK
        term's voltage dependence removed from the autodiff graph.
        It is **not** a subclass of this class (see its Notes).

    Notes
    -----
    Ported from ``PC/channel/Cav3p3_MA24_PC.mod``, which is
    identical to ``SC/channel/Cav3p3_RI21_SC.mod`` apart from its
    ``SUFFIX`` line and the BrainCell-local ``g_equiv`` diagnostic
    the stellate copy carries. Accordingly this class overrides
    nothing: the constructor, both gate declarations, the named
    parameter block, the four rate methods and ``current`` are all
    inherited from :class:`Cav3p3_RI2021_SC`. Only the
    ``register_channel`` key and this docstring's model citation
    differ. Per the bibliography's attribution scan, this subclass
    contributes no rate-function code of its own.

    The ``MA2024`` import-deviations tables list no ``TABLE``
    removal, no ``derivimplicit`` -> ``cnexp`` substitution and no
    rate-refresh relocation for ``Cav3p3``.

    References
    ----------
    .. [1] Xu, J., & Clancy, C. E. (2008). Ionic mechanisms of
           endogenous bursting in CA3 hippocampal pyramidal neurons:
           A model study. PLoS ONE, 3(4), e2056.
           doi:10.1371/journal.pone.0002056
    .. [2] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("CaHVA_MA2020_GoC")
class CaHVA_MA2020_GoC(OhmicHH):
    r"""Golgi cell high-voltage-activated calcium current.

    The high-voltage-activated (HVA) calcium current of the cerebellar
    Golgi cell model of (Masoli et al., 2020) [2]_. Its kinetics are
    those of the cerebellar granule cell model of (D'Angelo et al.,
    2001) [1]_, reused unchanged for the Golgi cell. Gating is
    :math:`s^2 u` in alpha/beta form with an ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, s^2 u \, (E_{Ca} - V) \\
        \alpha_s &= 0.04944 \exp((V' + 29.06) / 15.873) \\
        \beta_s &= 0.08298 \exp((V' + 18.66) / -25.641) \\
        \alpha_u &= 0.0013 \exp((V' + 48) / -18.183) \\
        \beta_u &= 0.0013 \exp((V' + 48) / 83.33)
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}` and the rates are per
    millisecond. Both gates are further scaled by
    :meth:`~braincell.channel._base.HH.gate_phi` with
    :math:`Q_{10} = 3` referred to 20 degrees Celsius (see Notes).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.46 mS/cm2``,
        the mod file's ``gcabar = 0.00046 mho/cm2`` (see Notes).
    temp : array-like, optional
        Absolute temperature driving the gates' Q10 factor. Defaults
        to 30 degrees Celsius, matching the mod file's
        ``celsius = 30``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaHVA_MA2020_GrC : The same mechanism imported from the granule
        cell deposit; identical constants, different model citation,
        and one different import deviation (see its Notes).
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``GoC/channel/CaHVA_MA20_GoC.mod``. An earlier
    revision of this docstring named the source file
    ``CaHVA_MA2020_GoC.mod``, which does not exist; the shipped file
    uses the two-digit year code, and that is corrected here. The
    file's ``TITLE`` reads "Cerebellum Granule Cell Model" and its
    ``COMMENT`` credits "E.D'Angelo, T.Nieus, A. Fontana" -- both
    inherited from the granule cell original, which is where these
    kinetics come from; the file is that mechanism re-deposited for
    the Golgi cell model cited as [2]_. That credit names authors 1,
    2 and 7 of the eight-author origin paper, so it is not turned
    into a citation: entry [1]_ below lists all eight.

    The mod file applies its Q10 factor inside each of the four rate
    functions, as ``Q10 = 3^((celsius - 20)/10)`` multiplying
    ``alp_s``, ``bet_s``, ``alp_u`` and ``bet_u``. BrainCell hoists it
    to the gate level instead, as
    ``Gate(q10=3.0, temp_ref=20 degC)``. For the alpha/beta form the
    two are algebraically identical: with
    :math:`\alpha = Q_{10} a` and :math:`\beta = Q_{10} b`,
    :math:`\alpha (1 - x) - \beta x = Q_{10} (a (1 - x) - b x)`, which
    is exactly what ``phi`` multiplies. The rate methods here
    therefore return the unscaled ``a``/``b``.

    The mod file's ``eca = 129.33 (mV)`` is not read by this class:
    the reversal potential is supplied by the attached
    :class:`~braincell.ion.Calcium` ion object.

    **Import deviation -- interpolation table removed.** The original
    ``TABLE`` directive tabulated ``s_inf``, ``tau_s``, ``u_inf`` and
    ``tau_u`` over ``[-100, 30]`` mV, clamping to the boundary value
    outside that range; BrainCell evaluates the continuous formulas
    per call, so any BrainCell/NEURON divergence outside that window
    is expected.

    **Not an integration-method substitution.** Unlike its granule
    cell twin, this mechanism was already ``cnexp`` upstream, so no
    ``derivimplicit`` -> ``cnexp`` change was made for it.

    **Import deviation -- NMODL default-precision rewrite.**
    ``Kalpha_s`` is written ``15.87301587302`` in the mod source and
    ``15.873`` here, because BrainCell aligns with the roughly
    six-significant-figure defaults NEURON's generated C emits rather
    than with the source text. Ordinary in-formula literals are not
    subject to this rewrite and keep their source values.

    NEURON's raw ``ica`` here is ``g * (v - eca)``, i.e.
    outward-positive; :class:`~braincell.channel._base.OhmicHH`
    computes ``g_max * s^2 u * (E - V)``, the same current under
    BrainCell's repo-wide inward-positive convention.

    ``g_max``'s default is the ``gcabar`` of the cell-model deposit
    this mechanism was imported from -- a value tuned for that model,
    not a conductance reported by the origin paper.

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
    root_type = Calcium
    gates = (
        Gate("s", power=2, q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
        Gate("u", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.46 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_s = 0.04944
        self.Kalpha_s = 15.873  # 01587302
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

    def f_s_alpha(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_s * u.math.exp((V - self.V0alpha_s) / self.Kalpha_s)

    def f_s_beta(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_s * u.math.exp((V - self.V0beta_s) / self.Kbeta_s)

    def f_u_alpha(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_u * u.math.exp((V - self.V0alpha_u) / self.Kalpha_u)

    def f_u_beta(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_u * u.math.exp((V - self.V0beta_u) / self.Kbeta_u)


@register_channel("CaHVA_MA2020_GrC")
class CaHVA_MA2020_GrC(OhmicHH):
    r"""Granule cell high-voltage-activated calcium current.

    The high-voltage-activated (HVA) calcium current of the cerebellar
    granule cell model of (Masoli et al., 2020) [2]_, whose kinetics
    are those of the earlier granule cell model of (D'Angelo et al.,
    2001) [1]_. Gating is :math:`s^2 u` in alpha/beta form with an
    ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, s^2 u \, (E_{Ca} - V) \\
        \alpha_s &= 0.04944 \exp((V' + 29.06) / 15.873) \\
        \beta_s &= 0.08298 \exp((V' + 18.66) / -25.641) \\
        \alpha_u &= 0.0013 \exp((V' + 48) / -18.183) \\
        \beta_u &= 0.0013 \exp((V' + 48) / 83.33)
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}` and the rates are per
    millisecond. Both gates are further scaled by
    :meth:`~braincell.channel._base.HH.gate_phi` with
    :math:`Q_{10} = 3` referred to 20 degrees Celsius (see Notes).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.46 mS/cm2``,
        the mod file's ``gcabar = 0.00046 mho/cm2`` (see Notes).
    temp : array-like, optional
        Absolute temperature driving the gates' Q10 factor. Defaults
        to 30 degrees Celsius, matching the mod file's
        ``celsius = 30``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaHVA_MA2020_GoC : The same mechanism imported from the Golgi
        cell deposit; identical constants, different model citation,
        and one fewer import deviation (see its Notes).
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``GrC/channel/CaHVA_MA20_GrC.mod``. An earlier
    revision of this docstring named the source file
    ``CaHVA_MA2020_GrC.mod``, which does not exist; the shipped file
    uses the two-digit year code, and that is corrected here. That
    file and ``GoC/channel/CaHVA_MA20_GoC.mod`` are identical apart
    from the ``SUFFIX`` line, which is why the two BrainCell classes
    carry the same constants; they are separate classes, not a
    subclass pair, because they were imported from two different
    deposits and therefore take two different model citations.

    The file's ``COMMENT`` credits "E.D'Angelo, T.Nieus, A. Fontana",
    naming authors 1, 2 and 7 of the eight-author origin paper. That
    credit is not turned into a citation: entry [1]_ below lists all
    eight.

    The mod file applies its Q10 factor inside each of the four rate
    functions, as ``Q10 = 3^((celsius - 20)/10)`` multiplying
    ``alp_s``, ``bet_s``, ``alp_u`` and ``bet_u``. BrainCell hoists it
    to the gate level instead, as
    ``Gate(q10=3.0, temp_ref=20 degC)``. For the alpha/beta form the
    two are algebraically identical: with
    :math:`\alpha = Q_{10} a` and :math:`\beta = Q_{10} b`,
    :math:`\alpha (1 - x) - \beta x = Q_{10} (a (1 - x) - b x)`, which
    is exactly what ``phi`` multiplies. The rate methods here
    therefore return the unscaled ``a``/``b``.

    The mod file's ``eca = 129.33 (mV)`` is not read by this class:
    the reversal potential is supplied by the attached
    :class:`~braincell.ion.Calcium` ion object.

    **Import deviation -- interpolation table removed.** The original
    ``TABLE`` directive tabulated ``s_inf``, ``tau_s``, ``u_inf`` and
    ``tau_u`` over ``[-100, 30]`` mV, clamping to the boundary value
    outside that range; BrainCell evaluates the continuous formulas
    per call, so any BrainCell/NEURON divergence outside that window
    is expected.

    **Import deviation -- integration method substituted.** The
    upstream ``derivimplicit`` is replaced by ``cnexp``. This is the
    granule cell mechanism's one difference in status from its Golgi
    cell twin, which was already ``cnexp`` upstream. The substitution
    is exact here, because the ``s`` and ``u`` gate ODEs are
    independent of one another.

    **Import deviation -- NMODL default-precision rewrite.**
    ``Kalpha_s`` is written ``15.87301587302`` in the mod source and
    ``15.873`` here, because BrainCell aligns with the roughly
    six-significant-figure defaults NEURON's generated C emits rather
    than with the source text. Ordinary in-formula literals are not
    subject to this rewrite and keep their source values.

    NEURON's raw ``ica`` here is ``g * (v - eca)``, i.e.
    outward-positive; :class:`~braincell.channel._base.OhmicHH`
    computes ``g_max * s^2 u * (E - V)``, the same current under
    BrainCell's repo-wide inward-positive convention.

    ``g_max``'s default is the ``gcabar`` of the cell-model deposit
    this mechanism was imported from -- a value tuned for that model,
    not a conductance reported by the origin paper.

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
    root_type = Calcium
    gates = (
        Gate("s", power=2, q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
        Gate("u", q10=3.0, temp_ref=u.celsius2kelvin(20.0)),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.46 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(30.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.Aalpha_s = 0.04944
        self.Kalpha_s = 15.873  # 01587302
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

    def f_s_alpha(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_s * u.math.exp((V - self.V0alpha_s) / self.Kalpha_s)

    def f_s_beta(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_s * u.math.exp((V - self.V0beta_s) / self.Kbeta_s)

    def f_u_alpha(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Aalpha_u * u.math.exp((V - self.V0alpha_u) / self.Kalpha_u)

    def f_u_beta(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return self.Abeta_u * u.math.exp((V - self.V0beta_u) / self.Kbeta_u)


@register_channel("Cav2p3_MA2020_GoC")
class Cav2p3_MA2020_GoC(OhmicHH):
    r"""Golgi cell Cav2.3 R-type, medium-threshold calcium current.

    The Cav2.3 (R-type) medium-threshold calcium current of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [3]_. Its
    kinetics are those of the ``car`` mechanism of the CA1 pyramidal
    cell model of (Poirazi, Brannon & Mel, 2003) [1]_ [2]_, reused
    unchanged for the Golgi cell. Gating is :math:`m^3 h` with
    voltage-independent time constants and an ohmic driving force:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \, m^3 h \, (E_{Ca} - V) \\
        m_\infty &= \frac{1}{1 + \exp((V' + 48.5) / -3)} \\
        h_\infty &= \frac{1}{1 + \exp((V' + 53) / 1)} \\
        \tau_m &= 50 \ \mathrm{ms}, \quad \tau_h = 5 \ \mathrm{ms}
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}`. Neither gate declares a Q10
    or a ``phi``, so no temperature factor is applied to either time
    constant.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density. Defaults to ``0.0 mS/cm2``,
        matching the mod file's ``gcabar = 0 (mho/cm2)``, so the
        mechanism contributes no current until a cell model sets it
        (see Notes).
    temp : array-like, optional
        Absolute temperature. Accepted and stored, but read by no
        method of this class (see Notes). Defaults to 34 degrees
        Celsius, matching the mod file's ``celsius = 34``.
    name : str, optional
        Optional channel name.

    See Also
    --------
    CaHVA_MA2020_GoC : The Golgi cell model's other
        high-voltage-activated calcium current, from a different
        origin.
    braincell.channel._base.OhmicHH : Template supplying the ohmic
        driving force used above.

    Notes
    -----
    Ported from ``GoC/channel/Cav2p3_MA20_GoC.mod``, whose ``TITLE``
    reads "Ca R-type channel with medium threshold for activation"
    and whose header records that the mechanism is "used in distal
    dendritic regions, together with calH.mod", that it "uses
    channel conductance (not permeability)", that it was "written by
    Yiota Poirazi on 11/13/00" and that BrainCell renamed it "From
    car to Cav2_3". The Poirazi credit is unusual among these
    cerebellar imports in naming an author of the origin papers
    rather than an unrelated porter, but it is still not turned into
    a citation on its own: entries [1]_ and [2]_ below are the
    published records. The upstream ``car.mod`` and its sibling
    ``calH.mod`` are both files of ModelDB accession 20212, the
    shared CA1 pyramidal model deposit serving the two companion
    papers; because the mechanism belongs to that shared biophysics
    and cannot be assigned to one paper, both are cited.

    **Import deviation -- interpolation table removed.** The
    original ``TABLE`` directive tabulated the indexed ``inf`` and
    ``tau`` arrays over ``[-100, 100]`` mV; NEURON clamped to the
    boundary value outside that range, so any BrainCell/NEURON
    divergence outside that window is expected. BrainCell evaluates
    the formulas per call instead.

    **The indexed rate structure was split into named methods.** The
    mod file stores its steady states and time constants in
    ``inf[2]`` and ``tau[2]``, filled by a ``FROM i=0 TO 1`` loop
    over the helpers ``varss(v, i)`` and ``vartau(v, i)``. Index 0
    is activation and index 1 is inactivation, per the mod file's
    own trailing comments and its ``INITIAL`` block, which assigns
    ``m = inf[0]`` and ``h = inf[1]``. BrainCell therefore maps
    index 0 onto :meth:`f_m_inf` / :meth:`f_m_tau` -- the
    ``(v + 48.5) / -3`` curve with ``tau = 50`` -- and index 1 onto
    :meth:`f_h_inf` / :meth:`f_h_tau` -- the ``(v + 53) / 1`` curve
    with ``tau = 5``. Both branch values were read back against the
    mod file's ``if (i==0)`` / ``else if (i==1)`` arms.

    **The default conductance is genuinely zero.** The mod file's
    ``gcabar = 0 (mho/cm2)`` is commented "initialized conductance":
    the deposit expects the cell-setup code to assign a density per
    region, and BrainCell carries the same zero. This is not a
    missing value or a failed unit conversion.

    ``temp`` is accepted, stored on the instance and never read.
    Neither ``Gate`` carries a ``q10`` or a ``phi``, and this
    mechanism has no GHK term, so nothing in the class consumes a
    temperature. The mod file likewise declares ``celsius = 34``
    without using it in any rate expression.

    The mod file's ``eca = 140 (mV)`` is not read by this class: the
    reversal potential is supplied by the attached
    :class:`~braincell.ion.Calcium` ion object. Its ``gmax``
    running-maximum diagnostic, updated in ``BREAKPOINT`` by
    ``if (g > gmax) { gmax = g }``, is a monitoring variable that
    never feeds the current, and BrainCell drops it.

    NEURON's raw ``ica`` here is ``g * (v - eca)``, i.e.
    outward-positive; :class:`~braincell.channel._base.OhmicHH`
    computes ``g_max * m^3 h * (E - V)``, the same current under
    BrainCell's repo-wide inward-positive convention.

    The mod file is already ``cnexp`` upstream, so no
    ``derivimplicit`` -> ``cnexp`` substitution was made for it, and
    the ``MA2020`` tables record no rate-refresh relocation and no
    NMODL default-precision rewrite for this mechanism.

    References
    ----------
    .. [1] Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic
           of subthreshold synaptic summation in a model CA1
           pyramidal cell. Neuron, 37(6), 977-987.
           doi:10.1016/S0896-6273(03)00148-X
    .. [2] Poirazi, P., Brannon, T., & Mel, B. W. (2003). Pyramidal
           neuron as two-layer neural network. Neuron, 37(6),
           989-999.
           doi:10.1016/S0896-6273(03)00149-1
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = Calcium
    gates = (
        Gate("m", power=3),
        Gate("h"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.mS / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(34.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)

    def f_m_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 48.5) / -3.0))

    def f_h_inf(self, V, Ca: IonInfo):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 53.0) / 1.0))

    def f_m_tau(self, V, Ca: IonInfo):
        return 50.0

    def f_h_tau(self, V, Ca: IonInfo):
        return 5.0


@register_channel("Ca_ZH2019_IO")
class Ca_ZH2019_IO(HH):
    r"""Somatic calcium current of the inferior-olive model (Zhang 2019).

    A fixed-reversal-potential calcium current with an instantaneous,
    non-stateful cubic activation and one dynamic inactivation gate,
    used for the single-compartment inferior-olive neuron model of
    (Zhang & Santaniello, 2019) [2]_:

    .. math::

        \begin{aligned}
        I_{Ca} &= g_{max} \cdot m_\infty \cdot h \cdot (E - V) \\
        m_\infty &= \left(\frac{1}{1 + \exp((V_{mid} - V) / 4.2)}\right)^3 \\
        h_\infty &= \frac{1}{1 + \exp((V + 85.5) / 8.6)} \\
        \tau_h &= 40 + 30 \cdot \frac{1}{1 + \exp((V + 84) / 7.3)}
                  \cdot \exp\!\left(\frac{V + 160}{30}\right)
        \end{aligned}

    Only :math:`h` is an integrated :class:`~braincell.channel._base.Gate`
    state (``gates = (Gate("h"),)``); :math:`m_\infty` is recomputed
    from :math:`V` on every call to :meth:`current` rather than tracked
    as a channel state (see Notes).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximum conductance density. Defaults to ``0.4 mS/cm**2``.
    E : array-like or callable, optional
        Fixed calcium current reversal potential (this class does not
        derive :math:`E` from an ion concentration). Defaults to
        ``120.0 mV``.
    mMidV : array-like or callable, optional
        Midpoint voltage of the instantaneous activation curve.
        Defaults to ``-61.0 mV``.
    freeze_m_inf : bool, optional
        If ``True`` (the default), block autodiff through the
        instantaneous activation factor with
        :func:`jax.lax.stop_gradient` while leaving the forward
        current value unchanged; see Notes.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Ca_ZH2019_IO_Frozen : Registry alias that forces
        ``freeze_m_inf=True`` unconditionally.
    braincell.channel.hyperpolarization_activated.HCN_ZH2019_IO :
        Sibling inferior-olive current from the same import family
        (different bibliography origin; see Notes).

    Notes
    -----
    Ported from ``IO/channel/Ca_ZH19_IO.mod``, whose header credits
    "Ca channel from Manor (Rinzel, Segev, Yarom) 1997" and porter
    "B. Torben-Nielsen @ HUJI, 7-10-2010" -- i.e. the kinetics
    originate with Manor, Rinzel, Segev & Yarom (1997) [1]_ and were
    ported to NEURON by Torben-Nielsen, Segev & Yarom (2012), whose
    inferior-olive model in turn was reused, without further
    modification credit, by Zhang & Santaniello (2019) [2]_. The 2012
    port paper is cited here only in this prose, per house style, not
    as a numbered reference.

    The inferior-olive neurons in both the Torben-Nielsen et al.
    (2012) and Zhang & Santaniello (2019) models are
    single-compartment (``nseg = 1``); the multi-compartment part of
    that lineage is a separate Purkinje-cell population, not the
    inferior olive. This docstring does not describe this mechanism
    as multi-compartment.

    In the imported ``.mod`` file, the shared ``rates(v)`` helper that
    recomputes ``minf``/``hinf``/``htau`` was called from
    ``BREAKPOINT`` in the original NEURON mechanism; BrainCell's port
    (like the rest of this ``ZH19``/``IO`` import family) evaluates
    the equivalent expressions from ``DERIVATIVE``-time state updates
    instead, so ``m_inf``/``h_inf``/``tau_h`` are refreshed before,
    rather than after, the state integration step within a given call.

    ``m`` carries no persistent state and is not part of ``gates``:
    :meth:`current` calls :meth:`f_m_inf` fresh on every evaluation,
    so nothing "evolves" over time for the activation term -- it is a
    purely algebraic function of the instantaneous voltage. Setting
    ``freeze_m_inf=True`` (the default) wraps that same forward value
    in :func:`jax.lax.stop_gradient`; it changes only which terms
    appear in gradients taken through this channel, never the forward
    current or the value of ``m_inf`` itself.

    References
    ----------
    .. [1] Manor, Y., Rinzel, J., Segev, I., & Yarom, Y. (1997).
           Low-amplitude oscillations in the inferior olive: A model
           based on electrical coupling of neurons with heterogeneous
           channel densities. Journal of Neurophysiology, 77(5),
           2736-2752.
           doi:10.1152/jn.1997.77.5.2736
    .. [2] Zhang, X., & Santaniello, S. (2019). Role of cerebellar
           GABAergic dysfunctions in the origins of essential tremor.
           Proceedings of the National Academy of Sciences of the
           United States of America, 116(27), 13592-13601.
           doi:10.1073/pnas.1817689116
    """

    __module__ = "braincell.channel"
    root_type = HHTypedNeuron
    gates = (Gate("h"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.4 * (u.mS / u.cm**2),
        E: Union[brainstate.typing.ArrayLike, Callable] = 120.0 * u.mV,
        mMidV: Union[brainstate.typing.ArrayLike, Callable] = -61.0 * u.mV,
        freeze_m_inf: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.mMidV = braintools.init.param(mMidV, self.varshape, allow_none=False)
        self.freeze_m_inf = bool(freeze_m_inf)

    def current(self, V):
        m_inf = self.f_m_inf(V)
        if self.freeze_m_inf:
            m_inf = jax.lax.stop_gradient(m_inf)
        return self.g_max * m_inf * self.h.value * (self.E - V)

    def f_m_inf(self, V):
        V = V.to_decimal(u.mV)
        m_mid = self.mMidV.to_decimal(u.mV)
        term = 1.0 + u.math.exp((m_mid - V) / 4.2)
        return 1.0 / (term * term * term)

    def f_h_inf(self, V):
        V = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 85.5) / 8.6))

    def f_h_tau(self, V):
        V = V.to_decimal(u.mV)
        return 40.0 + 30.0 * (1.0 / (1.0 + u.math.exp((V + 84.0) / 7.3))) * u.math.exp((V + 160.0) / 30.0)


@register_channel("Ca_ZH2019_IO_Frozen")
class Ca_ZH2019_IO_Frozen(Ca_ZH2019_IO):
    r""":class:`Ca_ZH2019_IO` with ``freeze_m_inf`` pinned to ``True``.

    A registry-only subclass of :class:`Ca_ZH2019_IO` that always
    stops autodiff through the instantaneous activation factor,
    regardless of what a caller passes for ``freeze_m_inf``.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`Ca_ZH2019_IO`.
    **kwargs
        Keyword arguments forwarded to :class:`Ca_ZH2019_IO`; any
        ``freeze_m_inf`` entry is overwritten with ``True`` before the
        parent constructor runs (see Notes).

    See Also
    --------
    Ca_ZH2019_IO : Base class; already defaults to
        ``freeze_m_inf=True`` (see Notes for what this subclass adds
        on top of that default).

    Notes
    -----
    "Frozen" describes the gradient path through the instantaneous
    activation term, not the forward numerics: :class:`Ca_ZH2019_IO`
    already defaults to ``freeze_m_inf=True``, so this subclass does
    not change the default forward current or default gradient
    behaviour by itself. What it changes is *configurability* --
    ``__init__`` here unconditionally sets
    ``kwargs["freeze_m_inf"] = True`` before delegating to
    :class:`Ca_ZH2019_IO`, so a caller cannot recover the unfrozen
    (full-gradient) behaviour by passing ``freeze_m_inf=False`` to
    this class, whereas it can to the base class. No channel state
    stops evolving: ``m`` was never a stateful
    :class:`~braincell.channel._base.Gate` in the base class either
    (only ``h`` is), and :func:`jax.lax.stop_gradient` affects only
    backward-mode differentiation, never the forward value computed
    by :meth:`~Ca_ZH2019_IO.f_m_inf`.

    Provenance, the single-compartment caveat, and the
    ``rates``-relocation import deviation are identical to
    :class:`Ca_ZH2019_IO`'s and are not repeated here; see that
    class's Notes. Per the bibliography's attribution scan, this
    subclass contributes no rate-function code of its own -- it
    inherits every kinetic equation unchanged from
    :class:`Ca_ZH2019_IO`, that is from Manor, Rinzel, Segev &
    Yarom (1997) [1]_ as imported by Zhang & Santaniello
    (2019) [2]_.

    References
    ----------
    .. [1] Manor, Y., Rinzel, J., Segev, I., & Yarom, Y. (1997).
           Low-amplitude oscillations in the inferior olive: A model
           based on electrical coupling of neurons with heterogeneous
           channel densities. Journal of Neurophysiology, 77(5),
           2736-2752.
           doi:10.1152/jn.1997.77.5.2736
    .. [2] Zhang, X., & Santaniello, S. (2019). Role of cerebellar
           GABAergic dysfunctions in the origins of essential tremor.
           Proceedings of the National Academy of Sciences of the
           United States of America, 116(27), 13592-13601.
           doi:10.1073/pnas.1817689116
    """

    __module__ = "braincell.channel"

    def __init__(self, *args, **kwargs):
        kwargs["freeze_m_inf"] = True
        super().__init__(*args, **kwargs)
