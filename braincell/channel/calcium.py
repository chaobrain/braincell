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
    r"""Inoue & Strowbridge 2008 calcium-activated nonselective cation current.

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
    (Huguenard & McCormick, 1992) [1]_.** That paper models exactly
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
    r"""Evans/Beining Cav1.2 calcium current with calcium-dependent inactivation."""

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
    r"""Template-based import of ``Cav1p2_MA25_BC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav1p3_MA2020_GoC")
class Cav1p3_MA2020_GoC(OhmicHH):
    r"""Evans/Beining Cav1.3 calcium current with calcium-dependent inactivation."""

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
    r"""Template-based import of ``Cav1p3_MA25_BC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav3p1_MA2020_GoC")
class Cav3p1_MA2020_GoC(HH):
    r"""Purkinje cell Cav3.1 low-threshold calcium current with GHK drive.

    Notes
    -----
    The source NMODL applies temperature scaling directly inside the tau
    formulas instead of through a uniform gate-level ``phi`` factor:

    - for ``p``/``m`` the ``v <= -90`` branch is hard-coded to ``1 ms`` and is
      **not** divided by ``qt``;
    - in the other branch the full ``C_tau_m + A_tau_m / (...)`` expression is
      divided by ``qt``;
    - for ``q``/``h`` the full ``C_tau_h + A_tau_h / exp(...)`` expression is
      also divided by ``qt``.

    That behavior does not match the generic ``HH`` gate temperature path,
    where ``Gate(q10=..., temp_ref=...)`` would multiply the full derivative by
    ``phi`` and therefore divide the whole tau by ``qt``. We intentionally keep
    gate ``phi=1`` here and encode the source-mod temperature handling directly
    in :meth:`f_p_tau` and :meth:`f_q_tau`.
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
    """Template-based import of ``Cav3p1_MA2024_PC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav3p1_MA2020_GoC_Frozen")
class Cav3p1_MA2020_GoC_Frozen(Cav3p1_MA2020_GoC):
    """GoC Cav3.1 variant that freezes GHK voltage dependence in autodiff."""

    __module__ = "braincell.channel"

    def current(self, V, Ca: IonInfo):
        frozen_V = _freeze_quantity_gradient(V)
        drive = _cav3p1_nmodl_ghk_flux(V=frozen_V, ci=Ca.Ci, co=Ca.Co, z=self.z, temp=self.temp)
        return -self.g_max * self.conductance_factor(V, Ca) * drive


@register_channel("Cav3p1_MA2024_PC_Frozen")
class Cav3p1_MA2024_PC_Frozen(HH):
    """Experimental Cav3.1 variant that freezes GHK voltage dependence in autodiff."""

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
    r"""Template-based import of ``Cav3_1_test.mod``.

    This PC24 test variant removes the GHK drive entirely and uses a direct
    conductance-density style current law:

    .. math::

       I_{Ca} = g_{max} \, p^2 q

    The source NMODL keeps the same steady-state and tau formulas as the
    Cav3.1 template, including the gate-temperature handling encoded directly
    in the tau expressions.
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
    """Template-based import of ``Cav2p1_RI21_SC.mod``."""

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
    """Template-based import of ``Cav2p1_MA2025_BC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2024_PC")
class Cav2p1_MA2024_PC(Cav2p1_RI2021_SC):
    """Template-based import of ``Cav2p1_MA2024_PC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2024_PC_Frozen")
class Cav2p1_MA2024_PC_Frozen(HH):
    """Experimental Cav2.1 variant that freezes GHK voltage dependence in autodiff."""

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
    """SC Cav2.1 variant reusing the frozen-GHK PC24 implementation."""

    __module__ = "braincell.channel"


@register_channel("Cav2p1_MA2025_BC_Frozen")
class Cav2p1_MA2025_BC_Frozen(Cav2p1_MA2024_PC_Frozen):
    """BC Cav2.1 variant reusing the frozen-GHK PC24 implementation."""

    __module__ = "braincell.channel"


@register_channel("Cav3p3_MA2024_PC_Frozen")
class Cav3p3_MA2024_PC_Frozen(HH):
    """Experimental Cav3.3 variant that freezes GHK voltage dependence in autodiff."""

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
    """Template-based import of ``Cav3p2_RI21_SC.mod``.

    Notes
    -----
    This source mod is not especially clean as a reusable temperature- and
    concentration-general mechanism:

    - the original mod effectively bakes the gate-temperature conversion to
      36 C into fixed phi factors derived from 24 C data;
    - the compare path here uses fixed Ca concentrations to match the original
      mod assumptions;
    - ``tau_h`` is written in a special ``13.7 + term / phi_h`` form rather
      than the usual ``tau / phi`` pattern used by most HH-style templates.

    The implementation below intentionally preserves those quirks so the
    BrainCell behavior matches NEURON for one-to-one comparison. Longer term
    this channel should probably be rewritten into a more general form instead
    of carrying over the source mod's baked-in assumptions.
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
    """Template-based import of ``Cav3p2_MA2025_BC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav3p2_MA2024_PC")
class Cav3p2_MA2024_PC(Cav3p2_RI2021_SC):
    """Template-based import of ``Cav3p2_MA2024_PC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Cav3p3_RI2021_SC")
class Cav3p3_RI2021_SC(HH):
    """Template-based import of ``Cav3p3_RI21_SC.mod``.

    Notes
    -----
    The source mod uses a somewhat inconsistent current-law scaling: it mixes
    ``pcabar`` (documented like a permeability, ``cm/s``), ``gCav3_3bar``
    (documented like ``S/cm^2``), and a hand-written GHK expression with its
    own built-in unit conversion constants. To keep the BrainCell current law
    dimensionally consistent while still matching NEURON numerically,
    ``g_scale`` is treated here as a dimensionless empirical scale factor
    rather than as a physical conductance density. ``perm`` remains the
    permeability-like term.
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
    """Template-based import of ``Cav3p3_MA2024_PC.mod``."""

    __module__ = "braincell.channel"


@register_channel("CaHVA_MA2020_GoC")
class CaHVA_MA2020_GoC(OhmicHH):
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
    :class:`Ca_ZH2019_IO`.

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
