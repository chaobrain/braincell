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

"""Calcium-dependent potassium channels built directly on templates.

Two mechanism families live in this module. The first is gated purely
by intracellular calcium: :class:`AHP_De1994`, :class:`SK_SU2015_DCN`,
the ``Kca3p1_*`` group and the ``Kca2p2_*`` group all read only
``Ca.Ci`` in their transition/gate rates, with no voltage dependence.
The second is the BK/mslo family, the ``Kca1p1_*`` group: a
large-conductance channel gated jointly by voltage and [Ca]_i,
following the Horrigan-Aldrich allosteric scheme, in which calcium
binding and the voltage-dependent conformational change act on
independent legs of the same state graph.

Every public channel class here sets
``root_type = _KCA_ROOT_TYPE = brainstate.mixin.JointTypes[Potassium,
Calcium]``: the host cell must provide both a
:class:`~braincell.ion.Potassium` and a :class:`~braincell.ion.Calcium`
ion, because every rate method takes both ``K`` and ``Ca`` as
:class:`~braincell._base.IonInfo` arguments, even where only ``Ca.Ci``
actually drives the kinetics.
"""

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u
import jax

from braincell._base import IonInfo
from braincell.channel._base import Gate, Markov, OhmicHH, Transition
from braincell.ion import Calcium, Potassium
from braincell.mech import register_channel

__all__ = [
    "AHP_De1994",
    "SK_SU2015_DCN",
    "Kca3p1_MA2020_GoC",
    "Kca3p1_MA2025_BC",
    "Kca3p1_MA2024_PC",
    "Kca2p2_MA2020_GoC",
    "Kca2p2_MA2025_BC",
    "Kca2p2_MA2020_GrC",
    "Kca2p2_MA2024_PC",
    "Kca2p2_RI2021_SC",
    "Kca1p1_MA2020_GoC",
    "Kca1p1_MA2025_BC",
    "Kca1p1_MA2020_GrC",
    "Kca1p1_MA2024_PC",
    "Kca1p1_RI2021_SC",
]


_KCA_ROOT_TYPE = brainstate.mixin.JointTypes[Potassium, Calcium]


def _q10_factor(temp, q10, *, ref_celsius: float):
    return q10 ** (((temp - u.celsius2kelvin(ref_celsius)) / u.kelvin) / 10.0)


@register_channel("AHP_De1994")
class AHP_De1994(OhmicHH):
    r"""Calcium-activated after-hyperpolarization current.

    Reproduces the slow Ca2+-activated K+ (AHP) current of the
    thalamic reticular nucleus spindle-rhythmicity model of
    (Destexhe et al., 1994) [1]_: a two-site calcium-binding gate
    with no voltage dependence, of the closed form
    ``<closed> + n Ca_i <-> <open> (alpha, beta)``.

    .. math::

        \begin{aligned}
        I_{AHP} &= g_{\mathrm{max}} \, p^2 \, (E_K - V) \\
        \frac{dp}{dt} &= \phi \left(\alpha_p \, (1 - p) -
                          \beta_p \, p\right) \\
        \alpha_p &= \alpha \, \left([\mathrm{Ca}]_i /
                     \mathrm{mM}\right)^n \\
        \beta_p &= \beta
        \end{aligned}

    i.e. a Hodgkin-Huxley-style gate whose forward rate is a power
    law in intracellular calcium and whose backward rate is a fixed
    constant.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    n : array-like or callable, optional
        Calcium-binding exponent used inside :math:`\alpha_p`,
        default ``2``. Independent of the gate's conductance exponent
        (see Notes): changing ``n`` alters only the calcium-binding
        rate law, not the ``p ** 2`` factor in the current.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
    alpha : array-like or callable, optional
        Forward (calcium-binding) rate coefficient, default ``48.0``
        (units ``mM^-n ms^-1``).
    beta : array-like or callable, optional
        Backward (unbinding) rate, default ``0.09 ms^-1``. See Notes:
        this default is three times the reference value.
    phi : array-like or callable, optional
        Rate-scaling factor multiplying both :math:`\alpha_p` and
        :math:`\beta_p`, default ``1.0``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    No ``.mod`` file under this repository carries a
    ``De1994``/``De19`` filename fragment; this class has no in-repo
    NMODL counterpart to name here.

    The gate's conductance exponent is fixed at ``power=2`` by this
    class's ``gates`` declaration and is independent of the ``n``
    constructor parameter, which only exponentiates
    ``[Ca]_i`` inside the forward rate. Both default to ``2``, which
    is why the two are easy to conflate; they do not have to agree.

    **BrainCell's ``beta`` default is three times the reference
    value.** BrainCell defaults to ``beta = 0.09 ms^-1``, but the
    reference value is ``beta = 0.03 ms^-1`` -- the value used by the
    authors' reference implementation, and quoted from the paper by
    BrainPy. (The 1994 paper itself is paywalled; only its PubMed
    abstract was read for this documentation pass, so the 0.03 figure
    could not be checked directly against the published text.) The
    ``alpha = 48.0`` default is unaffected by this discrepancy: it is
    exactly ``beta / cac ** n = 0.03 / 0.025 ** 2 = 48 mM^-2 ms^-1``
    from the reference implementation's own ``beta`` and ``cac``, and
    BrainCell reproduces it exactly -- evidence that the rest of the
    parameterisation is faithful even though the ``beta`` default
    drifted.

    References
    ----------
    .. [1] Destexhe, A., Contreras, D., Sejnowski, T. J., & Steriade,
           M. (1994). A model of spindle rhythmicity in the isolated
           thalamic reticular nucleus. Journal of Neurophysiology,
           72(2), 803-818.
           doi:10.1152/jn.1994.72.2.803
    """

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    gates = (Gate("p", power=2, phi="phi"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        n: Union[brainstate.typing.ArrayLike, Callable] = 2,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 48.0,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.09,
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.n = braintools.init.param(n, self.varshape, allow_none=False)
        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.phi = braintools.init.param(phi, self.varshape, allow_none=False)

    def f_p_alpha(self, V, K: IonInfo, Ca: IonInfo):
        return self.alpha * u.math.power(Ca.Ci / u.mM, self.n)

    def f_p_beta(self, V, K: IonInfo, Ca: IonInfo):
        return self.beta


@register_channel("SK_SU2015_DCN")
class SK_SU2015_DCN(OhmicHH):
    r"""Small-conductance calcium-activated potassium current, DCN.

    Template-based import of ``SK_SU15_DCN.mod``, part of the deep
    cerebellar nuclei (DCN) neuron model of (Sudhakar et al., 2015)
    [2]_.

    .. math::

        \begin{aligned}
        I_{SK} &= g_{\mathrm{max}} \, z \, (E_K - V) \\
        \frac{dz}{dt} &= \frac{z_{\infty} - z}{\tau_z} \\
        z_{\infty} &= \frac{[\mathrm{Ca}]_i^4}
                           {[\mathrm{Ca}]_i^4 + 8.1 \times 10^{-15}} \\
        \tau_z &= \begin{cases}
            1 - 186.67 \, [\mathrm{Ca}]_i &
                [\mathrm{Ca}]_i < 0.005~\mathrm{mM} \\
            0.0667 & [\mathrm{Ca}]_i \geq 0.005~\mathrm{mM}
        \end{cases}
        \end{aligned}

    where :math:`[\mathrm{Ca}]_i` is expressed in mM and
    :math:`\tau_z` in ms before being divided by ``qdeltat``.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``0.01 mS/cm2``.
    qdeltat : array-like or callable, optional
        Divisor applied to :math:`\tau_z` after evaluation, default
        ``1.0``.
    name : str, optional
        Optional channel name.

    Notes
    -----
    Ported from ``SK_SU15_DCN.mod``. The mechanism name ``SK`` and
    the rate constants above appear only in the deposited NMODL code,
    not in the body of the Sudhakar et al. (2015) article, which
    documents the DCN model as reused from a previously published
    model rather than reprinting per-mechanism constants. The origin
    chain, reached only through the ``.mod`` file's own credit line
    (not through either paper's text): kinetics from the deep
    cerebellar nucleus model of Steuber et al. (2011) [1]_,
    translated from GENESIS to NEURON by Luthman et al. (2011), and
    used unchanged in Sudhakar et al. (2015) [2]_. Luthman et al.
    (2011) does not appear anywhere in the Sudhakar et al. (2015)
    text and is recorded here only as the translation step, not as a
    numbered reference.

    **Import deviation, distinct from the code's own branch.** The
    upstream ``.mod`` file tabulated ``zinf`` and ``tauz`` over a
    clamped ``cai`` range of ``[0, 0.01]`` mM via a NEURON ``TABLE``
    statement; BrainCell evaluates the closed-form expressions above
    on every call instead. This former-``TABLE`` clamp is a separate
    fact from the ``[Ca]_i < 0.005`` mM branch inside :math:`\tau_z`
    itself, which is the model's own piecewise definition and is not
    an artefact of removing the table.

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
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    gates = (Gate("z"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.01 * (u.mS / u.cm**2),
        qdeltat: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.qdeltat = braintools.init.param(qdeltat, self.varshape, allow_none=False)

    def f_z_inf(self, V, K: IonInfo, Ca: IonInfo):
        _ = (V, K)
        return self._z_inf_formula(Ca.Ci.to_decimal(u.mM))

    def f_z_tau(self, V, K: IonInfo, Ca: IonInfo):
        _ = (V, K)
        return self._z_tau_formula(Ca.Ci.to_decimal(u.mM)) / self.qdeltat

    def _z_inf_formula(self, cai):
        cai4 = cai**4
        return cai4 / (cai4 + 8.1e-15)

    def _z_tau_formula(self, cai):
        return u.math.where(cai < 0.005, 1.0 - 186.67 * cai, 0.0667)


@register_channel("Kca3p1_MA2020_GoC")
class Kca3p1_MA2020_GoC(OhmicHH):
    r"""Kca3.1 (IK) calcium-activated K current, Golgi cell.

    Template-based import of ``Kca3p1_MA20_GoC.mod``, part of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [4]_. A
    single gate whose forward rate factors into an independent
    voltage-dependent term and a piecewise calcium-dependent term.

    .. math::

        \begin{aligned}
        I &= g_{\mathrm{max}} \, p \, (E_K - V) \\
        \frac{dp}{dt} &= \frac{p_{\infty} - p}{\tau_p} \\
        p_{\infty} &= \frac{p_{\alpha}}{p_{\alpha} + p_{\beta}}, \quad
        \tau_p = \frac{1}{p_{\alpha} + p_{\beta}} \\
        p_{\alpha} &= \exp\!\left(\frac{V' + 70}{27}\right)
                      \cdot Y_{\mathrm{concdep}} \\
        Y_{\mathrm{concdep}} &= \begin{cases}
            \dfrac{500 \times 0.0013}
                  {\operatorname{exprel}\!\left(\dfrac{0.015 -
                   [\mathrm{Ca}]_i}{0.0013}\right)} &
                [\mathrm{Ca}]_i < 0.01~\mathrm{mM} \\[6pt]
            \dfrac{500 \times 0.005}{\exp(0.005 / 0.0013) - 1} &
                [\mathrm{Ca}]_i \geq 0.01~\mathrm{mM}
        \end{cases} \\
        p_{\beta} &= 0.05
        \end{aligned}

    where :math:`V' = V / \mathrm{mV}`, :math:`[\mathrm{Ca}]_i` is
    expressed in mM, and :math:`\operatorname{exprel}(y) =
    (e^y - 1) / y` (evaluated without the removable singularity at
    :math:`y = 0`).

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``120.0 mS/cm2``.
    q10_base : array-like, optional
        Accepted but not used; see Notes. Default ``3.0``.
    temp : array-like, optional
        Accepted but not used; see Notes. Default 22 degrees
        Celsius.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kca3p1_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca3p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.

    Notes
    -----
    Ported from ``Kca3p1_MA20_GoC.mod``.

    **``q10_base`` and ``temp`` are accepted but never read.** Both
    are stored on ``self`` in ``__init__`` but no method in this
    class -- there is no ``_phi()`` here, unlike the ``Kca2p2_*`` and
    ``Kca1p1_*`` classes below -- references either attribute. This
    is a discrepancy between the constructor signature and the
    implemented kinetics; it is documented rather than fixed, and
    the signature is left unchanged. The same holds for
    :class:`Kca3p1_MA2025_BC` and :class:`Kca3p1_MA2024_PC`, whose
    ``__init__`` is inherited unchanged from this class.

    ``p_beta = 0.05`` is a fixed internal constant assigned in
    ``__init__``, not a constructor parameter.

    **Import deviation, distinct from the code's own branch.** The
    upstream ``.mod`` file tabulated ``Yvdep`` and ``Yconcdep`` via a
    NEURON ``TABLE`` statement over ``V`` in ``[-100, 100]`` mV and
    ``cai`` clamped to ``[0, 0.01]`` mM; BrainCell evaluates the
    closed-form expressions above on every call instead. This former
    table clamp happens to share its upper concentration bound
    (0.01 mM) with the model's own ``[Ca]_i < 0.01`` mM branch inside
    :math:`Y_{\mathrm{concdep}}`, but the two are independent facts:
    the branch is the model's own definition, unaffected by table
    removal; the clamp is a NEURON interpolation-range artefact that
    no longer applies.

    References
    ----------
    .. [1] Rubin, D. B., & Cleland, T. A. (2006). Dynamical
           mechanisms of odor processing in olfactory bulb mitral
           cells. Journal of Neurophysiology, 96(2), 555-568.
           doi:10.1152/jn.00264.2006
    .. [2] Bhalla, U. S., & Bower, J. M. (1993). Exploring parameter
           space in detailed single neuron models: simulations of
           the mitral and granule cells of the olfactory bulb.
           Journal of Neurophysiology, 69(6), 1948-1965.
           doi:10.1152/jn.1993.69.6.1948
    .. [3] Davison, A. P., Feng, J., & Brown, D. (2000). A reduced
           compartmental model of the mitral cell for use in network
           models of the olfactory bulb. Brain Research Bulletin,
           51(5), 393-399.
           doi:10.1016/S0361-9230(99)00256-7
    .. [4] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    gates = (Gate("p"),)

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120.0 * (u.mS / u.cm**2),
        q10_base: brainstate.typing.ArrayLike = 3.0,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_base = braintools.init.param(q10_base, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.p_beta = 0.05

    def p_tau(self, V, Ca):
        return 1 / (self.p_alpha(V, Ca) + self.p_beta)

    def p_inf(self, V, Ca):
        return self.p_alpha(V, Ca) / (self.p_alpha(V, Ca) + self.p_beta)

    def p_alpha(self, V, Ca):
        V = V / u.mV
        return self.p_vdep(V) * self.p_concdep(Ca)

    def p_vdep(self, V):
        return u.math.exp((V + 70.0) / 27.0)

    def p_concdep(self, Ca):
        concdep_1 = 500 * 0.0013 / u.math.exprel((0.015 - Ca.Ci / u.mM) / 0.0013)
        with jax.ensure_compile_time_eval():
            concdep_2 = 500 * 0.005 / (u.math.exp(0.005 / 0.0013) - 1)
        return u.math.where(Ca.Ci / u.mM < 0.01, concdep_1, concdep_2)

    def f_p_alpha(self, V, K: IonInfo, Ca: IonInfo):
        return self.p_alpha(V, Ca)

    def f_p_beta(self, V, K: IonInfo, Ca: IonInfo):
        return self.p_beta


@register_channel("Kca3p1_MA2025_BC")
class Kca3p1_MA2025_BC(Kca3p1_MA2020_GoC):
    r"""Kca3.1 calcium-activated K current, basket-cell parameterisation.

    The same single-gate Kca3.1 kinetics documented in
    :class:`Kca3p1_MA2020_GoC`, reused unchanged for the cerebellar
    basket cell model of (Masoli et al., 2025) [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``120.0 mS/cm2``.
        Inherited from :class:`Kca3p1_MA2020_GoC`.
    q10_base : array-like, optional
        Accepted but not used (see :class:`Kca3p1_MA2020_GoC`
        Notes). Default ``3.0``. Inherited from
        :class:`Kca3p1_MA2020_GoC`.
    temp : array-like, optional
        Accepted but not used (see :class:`Kca3p1_MA2020_GoC`
        Notes). Default 22 degrees Celsius. Inherited from
        :class:`Kca3p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kca3p1_MA2020_GoC : The base class; full equations and the
        ``q10_base``/``temp`` unused-parameter discrepancy are
        documented there.
    Kca3p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.

    Notes
    -----
    Ported from ``Kca3p1_MA25_BC.mod``. This class does not override
    ``__init__``: the constructor, the rate methods and the fixed
    ``p_beta = 0.05`` constant are all inherited unchanged from
    :class:`Kca3p1_MA2020_GoC`. Only the ``register_channel`` key and
    this docstring's model citation differ -- the ``.mod`` file this
    class ports from parameterises the identical mechanism for a
    different cell type, not a different kinetic scheme.

    No import deviation is recorded for this mechanism beyond the
    ``TABLE`` removal already documented on
    :class:`Kca3p1_MA2020_GoC`, which applies identically here
    (``examples/neuron_compare/Cerebellum_mod/README.md``'s
    ``MA2025`` import-deviations table lists the same ``V``/``cai``
    range and tabulated quantities).

    References
    ----------
    .. [1] Rubin, D. B., & Cleland, T. A. (2006). Dynamical
           mechanisms of odor processing in olfactory bulb mitral
           cells. Journal of Neurophysiology, 96(2), 555-568.
           doi:10.1152/jn.00264.2006
    .. [2] Bhalla, U. S., & Bower, J. M. (1993). Exploring parameter
           space in detailed single neuron models: simulations of
           the mitral and granule cells of the olfactory bulb.
           Journal of Neurophysiology, 69(6), 1948-1965.
           doi:10.1152/jn.1993.69.6.1948
    .. [3] Davison, A. P., Feng, J., & Brown, D. (2000). A reduced
           compartmental model of the mitral cell for use in network
           models of the olfactory bulb. Brain Research Bulletin,
           51(5), 393-399.
           doi:10.1016/S0361-9230(99)00256-7
    .. [4] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Kca3p1_MA2024_PC")
class Kca3p1_MA2024_PC(Kca3p1_MA2020_GoC):
    r"""Kca3.1 calcium-activated K current, Purkinje-cell parameterisation.

    The same single-gate Kca3.1 kinetics documented in
    :class:`Kca3p1_MA2020_GoC`, reused unchanged for the human
    Purkinje cell model of (Masoli et al., 2024) [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``120.0 mS/cm2``.
        Inherited from :class:`Kca3p1_MA2020_GoC`.
    q10_base : array-like, optional
        Accepted but not used (see :class:`Kca3p1_MA2020_GoC`
        Notes). Default ``3.0``. Inherited from
        :class:`Kca3p1_MA2020_GoC`.
    temp : array-like, optional
        Accepted but not used (see :class:`Kca3p1_MA2020_GoC`
        Notes). Default 22 degrees Celsius. Inherited from
        :class:`Kca3p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.

    See Also
    --------
    Kca3p1_MA2020_GoC : The base class; full equations and the
        ``q10_base``/``temp`` unused-parameter discrepancy are
        documented there.
    Kca3p1_MA2025_BC : Same kinetics, basket-cell model citation.

    Notes
    -----
    Ported from ``Kca3p1_MA24_PC.mod``. This class does not override
    ``__init__``: the constructor, the rate methods and the fixed
    ``p_beta = 0.05`` constant are all inherited unchanged from
    :class:`Kca3p1_MA2020_GoC`. Only the ``register_channel`` key and
    this docstring's model citation differ -- the ``.mod`` file this
    class ports from parameterises the identical mechanism for a
    different cell type, not a different kinetic scheme.

    The ``MA2024`` import-deviations table records the same ``TABLE``
    removal already documented on :class:`Kca3p1_MA2020_GoC`
    (``V`` in ``[-100, 100] mV``, ``cai`` clamped to
    ``[0, 0.01] mM``, tabulating ``Yvdep``/``Yconcdep``).

    References
    ----------
    .. [1] Rubin, D. B., & Cleland, T. A. (2006). Dynamical
           mechanisms of odor processing in olfactory bulb mitral
           cells. Journal of Neurophysiology, 96(2), 555-568.
           doi:10.1152/jn.00264.2006
    .. [2] Bhalla, U. S., & Bower, J. M. (1993). Exploring parameter
           space in detailed single neuron models: simulations of
           the mitral and granule cells of the olfactory bulb.
           Journal of Neurophysiology, 69(6), 1948-1965.
           doi:10.1152/jn.1993.69.6.1948
    .. [3] Davison, A. P., Feng, J., & Brown, D. (2000). A reduced
           compartmental model of the mitral cell for use in network
           models of the olfactory bulb. Brain Research Bulletin,
           51(5), 393-399.
           doi:10.1016/S0361-9230(99)00256-7
    .. [4] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("Kca2p2_MA2020_GoC")
class Kca2p2_MA2020_GoC(Markov):
    r"""Kca2.2 (SK2) calcium-activated K current, Golgi cell.

    Template-based import of ``Kca2p2_MA20_GoC.mod``, part of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [3]_, with
    kinetics from the recombinant SK-channel gating scheme of
    (Hirschberg et al., 1998) [1]_ as implemented for the Golgi cell
    by (Solinas et al., 2007) [2]_. A seven-state, purely
    calcium-dependent Markov scheme: four closed states in series
    (``C1``-``C4``) with two open states (``O1``, ``O2``) branching
    off ``C3`` and ``C4``. ``C1`` is the algebraically eliminated
    state.

    .. math::

        \begin{aligned}
        I &= g_{\mathrm{max}} \, (O_1 + O_2) \, (E_K - V) \\
        C_1 \underset{k_{c1}}{\overset{k_{c2}[\mathrm{Ca}]_i}{
            \rightleftharpoons}} C_2
            \underset{k_{c2}}{\overset{k_{c3}[\mathrm{Ca}]_i}{
            \rightleftharpoons}} C_3
            \underset{k_{c3}}{\overset{k_{c4}[\mathrm{Ca}]_i}{
            \rightleftharpoons}} C_4 \\
        C_3 \underset{k_{o1}^{-}}{\overset{k_{o1}^{+}}{
            \rightleftharpoons}} O_1, \qquad
        C_4 \underset{k_{o2}^{-}}{\overset{k_{o2}^{+}}{
            \rightleftharpoons}} O_2
        \end{aligned}

    where every forward (calcium-binding) rate is
    :math:`\phi \cdot k / d` with :math:`d` the ``diff`` diffusion
    factor and :math:`[\mathrm{Ca}]_i` in mM, every backward
    (unbinding) rate is :math:`\phi \cdot k`, and
    :math:`\phi = q_{10}^{(T - 296.15\,\mathrm{K}) / 10}` uses
    :math:`23\,^{\circ}\mathrm{C}` (296.15 K) as the reference
    temperature regardless of the ``temp`` default below.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``38.0 mS/cm2``.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``.
    diff : array-like, optional
        Calcium diffusion-shell divisor applied to every forward
        (calcium-binding) rate, default ``3.0``.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. See Notes
        for the 22/23 degree distinction.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca2p2_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca2p2_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca2p2_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca2p2_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca2p2_MA20_GoC.mod``. ``current`` is overridden to
    sum only the two open states, ``O1`` and ``O2``; ``C1``-``C4`` do
    not conduct.

    ``_phi`` evaluates the Q10 scaling relative to a fixed
    :math:`23\,^{\circ}\mathrm{C}` reference (``ref_celsius=23.0``),
    while the ``temp`` constructor default is
    :math:`22\,^{\circ}\mathrm{C}`. The two temperatures are
    independent: the reference is the condition the rate constants
    below were tuned at, and the default is simply this class's
    default operating temperature, one degree below that reference.
    This is unlike :class:`Kca3p1_MA2020_GoC`, whose analogous
    ``q10_base``/``temp`` parameters are accepted but never read.

    References
    ----------
    .. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion,
           N. V. (1998). Gating of recombinant small-conductance
           Ca-activated K+ channels by calcium. The Journal of
           General Physiology, 111(4), 565-581.
           doi:10.1085/jgp.111.4.565
    .. [2] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [3] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    pairs = (
        Transition("C1", "C2", "dirc2_t_ca", "invc1_t"),
        Transition("C2", "C3", "dirc3_t_ca", "invc2_t"),
        Transition("C3", "C4", "dirc4_t_ca", "invc3_t"),
        Transition("C3", "O1", "diro1_t", "invo1_t"),
        Transition("C4", "O2", "diro2_t", "invo2_t"),
    )
    dependent_state = "C1"

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 38.0 * (u.mS / u.cm**2),
        q10_base: brainstate.typing.ArrayLike = 3.0,
        diff: brainstate.typing.ArrayLike = 3.0,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_base = braintools.init.param(q10_base, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.diff = braintools.init.param(diff, self.varshape, allow_none=False)

        self.invc1 = 80e-3
        self.invc2 = 80e-3
        self.invc3 = 200e-3

        self.invo1 = 1.0
        self.invo2 = 100e-3
        self.diro1 = 160e-3
        self.diro2 = 1.2

        self.dirc2 = 200.0
        self.dirc3 = 160.0
        self.dirc4 = 80.0

    def _phi(self):
        return _q10_factor(self.temp, self.q10_base, ref_celsius=23.0)

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, K, Ca, batch_size=batch_size)

    def current(self, V, K: IonInfo, Ca: IonInfo):
        states = self.state_values()
        return self.g_max * (states["O1"] + states["O2"]) * (K.E - V)

    def dirc2_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc2 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def dirc3_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc3 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def dirc4_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc4 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def invc1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc1 * self._phi()

    def invc2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc2 * self._phi()

    def invc3_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc3 * self._phi()

    def invo1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invo1 * self._phi()

    def invo2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invo2 * self._phi()

    def diro1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.diro1 * self._phi()

    def diro2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.diro2 * self._phi()


@register_channel("Kca2p2_MA2025_BC")
class Kca2p2_MA2025_BC(Kca2p2_MA2020_GoC):
    r"""Kca2.2 calcium-activated K current, basket-cell parameterisation.

    The same seven-state Kca2.2 Markov scheme documented in
    :class:`Kca2p2_MA2020_GoC`, reused unchanged for the cerebellar
    basket cell model of (Masoli et al., 2025) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``38.0 mS/cm2``.
        Inherited from :class:`Kca2p2_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    diff : array-like, optional
        Calcium diffusion-shell divisor, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca2p2_MA2020_GoC : The base class; full state topology,
        equations, and the 22/23 degree Celsius temperature note
        are documented there.
    Kca2p2_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca2p2_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca2p2_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca2p2_MA25_BC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca2p2_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion,
           N. V. (1998). Gating of recombinant small-conductance
           Ca-activated K+ channels by calcium. The Journal of
           General Physiology, 111(4), 565-581.
           doi:10.1085/jgp.111.4.565
    .. [2] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [3] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.channel"


@register_channel("Kca2p2_MA2020_GrC")
class Kca2p2_MA2020_GrC(Kca2p2_MA2020_GoC):
    r"""Kca2.2 calcium-activated K current, granule-cell parameterisation.

    The same seven-state Kca2.2 Markov scheme documented in
    :class:`Kca2p2_MA2020_GoC`, reused unchanged for the cerebellar
    granule cell subtype model of (Masoli et al., 2020) [3]_. This
    class is a Python subclass of :class:`Kca2p2_MA2020_GoC`, but
    that inheritance relationship is purely a code-reuse device: the
    kinetics and model citation below belong to the granule cell
    paper, not to the Golgi cell paper cited on the base class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``38.0 mS/cm2``.
        Inherited from :class:`Kca2p2_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    diff : array-like, optional
        Calcium diffusion-shell divisor, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca2p2_MA2020_GoC : The Python base class supplying the shared
        state topology and rate equations; cites the Golgi cell
        paper, not the granule cell paper cited here.
    Kca2p2_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca2p2_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca2p2_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca2p2_MA20_GrC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca2p2_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ. Despite the Python inheritance from the *Golgi*
    cell class, the correct model citation is the *granule* cell
    paper below -- the two cerebellar cell-type papers were published
    together and this citation is not interchangeable with the one
    on :class:`Kca2p2_MA2020_GoC`.

    References
    ----------
    .. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion,
           N. V. (1998). Gating of recombinant small-conductance
           Ca-activated K+ channels by calcium. The Journal of
           General Physiology, 111(4), 565-581.
           doi:10.1085/jgp.111.4.565
    .. [2] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"


@register_channel("Kca2p2_MA2024_PC")
class Kca2p2_MA2024_PC(Kca2p2_MA2020_GoC):
    r"""Kca2.2 calcium-activated K current, Purkinje-cell parameterisation.

    The same seven-state Kca2.2 Markov scheme documented in
    :class:`Kca2p2_MA2020_GoC`, reused unchanged for the human
    Purkinje cell model of (Masoli et al., 2024) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``38.0 mS/cm2``.
        Inherited from :class:`Kca2p2_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    diff : array-like, optional
        Calcium diffusion-shell divisor, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca2p2_MA2020_GoC : The base class; full state topology,
        equations, and the 22/23 degree Celsius temperature note
        are documented there.
    Kca2p2_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca2p2_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca2p2_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca2p2_MA24_PC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca2p2_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion,
           N. V. (1998). Gating of recombinant small-conductance
           Ca-activated K+ channels by calcium. The Journal of
           General Physiology, 111(4), 565-581.
           doi:10.1085/jgp.111.4.565
    .. [2] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [3] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.channel"


@register_channel("Kca2p2_RI2021_SC")
class Kca2p2_RI2021_SC(Kca2p2_MA2020_GoC):
    r"""Kca2.2 calcium-activated K current, stellate-cell parameterisation.

    The same seven-state Kca2.2 Markov scheme documented in
    :class:`Kca2p2_MA2020_GoC`, reused unchanged for the cerebellar
    stellate cell model of (Rizza et al., 2021) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``38.0 mS/cm2``.
        Inherited from :class:`Kca2p2_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    diff : array-like, optional
        Calcium diffusion-shell divisor, default ``3.0``. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca2p2_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca2p2_MA2020_GoC : The base class; full state topology,
        equations, and the 22/23 degree Celsius temperature note
        are documented there.
    Kca2p2_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca2p2_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca2p2_MA2024_PC : Same kinetics, Purkinje-cell model citation.

    Notes
    -----
    Ported from ``Kca2p2_RI21_SC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca2p2_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion,
           N. V. (1998). Gating of recombinant small-conductance
           Ca-activated K+ channels by calcium. The Journal of
           General Physiology, 111(4), 565-581.
           doi:10.1085/jgp.111.4.565
    .. [2] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De
           Schutter, E., & D'Angelo, E. (2007). Computational
           reconstruction of pacemaking and intrinsic
           electroresponsiveness in cerebellar Golgi cells. Frontiers
           in Cellular Neuroscience, 1, 2.
           doi:10.3389/neuro.03.002.2007
    .. [3] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.channel"


@register_channel("Kca1p1_MA2020_GoC")
class Kca1p1_MA2020_GoC(Markov):
    r"""Kca1.1 (BK/mslo) Ca- and voltage-activated K current, Golgi cell.

    Template-based import of ``Kca1p1_MA20_GoC.mod``, part of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [3]_, with
    parameters from the allosteric BK-channel gating scheme of
    (Cox, Cui, & Aldrich, 1997) [1]_ as adapted for calcium buffering
    in Purkinje cells by (Anwar, Hong, & De Schutter, 2012) [2]_. Ten
    states arranged as a 5x2 grid: a closed ladder ``C0``-``C4`` and
    an open ladder ``O0``-``O4``, each ladder stepping through 0-4
    bound calcium ions, with five vertical closed-to-open
    transitions, one per occupancy level. ``C0`` is the algebraically
    eliminated state.

    .. math::

        \begin{aligned}
        I &= g_{\mathrm{max}} \, (O_0 + O_1 + O_2 + O_3 + O_4) \,
             (E_K - V) \\
        C_i \underset{(i+1) K_c k_1}{\overset{(4-i)[\mathrm{Ca}]_i k_1}{
            \rightleftharpoons}} C_{i+1}, &\qquad
        O_i \underset{(i+1) K_o k_1}{\overset{(4-i)[\mathrm{Ca}]_i k_1}{
            \rightleftharpoons}} O_{i+1}, \qquad i = 0,\dots,3 \\
        C_i \underset{\mathrm{pb}_i \, \beta(V)}{\overset{
            \mathrm{pf}_i \, \alpha(V)}{\rightleftharpoons}} O_i,
            &\qquad i = 0,\dots,4 \\
        \alpha(V) &= \exp\!\left(\frac{Q_o F V}{R T}\right), \qquad
        \beta(V) = \exp\!\left(\frac{Q_c F V}{R T}\right)
        \end{aligned}

    where :math:`[\mathrm{Ca}]_i` is expressed in mM, :math:`k_1`,
    :math:`K_c`, :math:`K_o` are fixed rate/dissociation constants;
    each unbinding step is standard mass-action, so its rate scales
    with the number of calcium ions bound in the state being left,
    i.e. :math:`(i+1)` for the departure from :math:`C_{i+1}` or
    :math:`O_{i+1}`. Every rate additionally carries the multiplicative factor
    :math:`\phi = q_{10}^{(T - 296.15\,\mathrm{K}) / 10}`
    (:math:`23\,^{\circ}\mathrm{C}` reference), :math:`F` is the
    Faraday constant, :math:`R` the gas constant, and :math:`Q_o`,
    :math:`Q_c` are the Horrigan-Aldrich gating charges for the
    opening and closing conformational change.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. See Notes
        for the 22/23 degree distinction.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca1p1_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca1p1_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca1p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca1p1_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca1p1_MA20_GoC.mod``. ``current`` is overridden to
    sum all five open states, ``O0`` through ``O4``; ``C0``-``C4`` do
    not conduct.

    ``_phi`` evaluates the Q10 scaling relative to a fixed
    :math:`23\,^{\circ}\mathrm{C}` reference (``ref_celsius=23.0``),
    while the ``temp`` constructor default is
    :math:`22\,^{\circ}\mathrm{C}`, one degree below that reference.
    This is unlike :class:`Kca3p1_MA2020_GoC`, whose analogous
    ``q10_base``/``temp`` parameters are accepted but never read.

    ``self.L0 = 1806`` is assigned in ``__init__`` alongside the
    other Horrigan-Aldrich constants but is never read by any rate
    method or by ``current``; it is documented here rather than
    removed, since removing an assigned-but-unused attribute is a
    behavior change out of scope for a documentation pass.

    References
    ----------
    .. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric
           gating of a large conductance Ca-activated K+ channel.
           The Journal of General Physiology, 110(3), 257-281.
           doi:10.1085/jgp.110.3.257
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
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    pairs = (
        Transition("C0", "C1", "c01", "c10"),
        Transition("C1", "C2", "c12", "c21"),
        Transition("C2", "C3", "c23", "c32"),
        Transition("C3", "C4", "c34", "c43"),
        Transition("O0", "O1", "o01", "o10"),
        Transition("O1", "O2", "o12", "o21"),
        Transition("O2", "O3", "o23", "o32"),
        Transition("O3", "O4", "o34", "o43"),
        Transition("C0", "O0", "f0", "b0"),
        Transition("C1", "O1", "f1", "b1"),
        Transition("C2", "O2", "f2", "b2"),
        Transition("C3", "O3", "f3", "b3"),
        Transition("C4", "O4", "f4", "b4"),
    )
    dependent_state = "C0"

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm**2),
        q10_base: brainstate.typing.ArrayLike = 3.0,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.q10_base = braintools.init.param(q10_base, self.varshape, allow_none=False)

        self.Qo = 0.73
        self.Qc = -0.67
        self.k1 = 1.0e3
        self.onoffrate = 1.0
        self.L0 = 1806
        self.Kc = 11.0e-3
        self.Ko = 1.1e-3

        self.pf0 = 2.39e-3
        self.pf1 = 7.0e-3
        self.pf2 = 40e-3
        self.pf3 = 295e-3
        self.pf4 = 557e-3

        self.pb0 = 3936e-3
        self.pb1 = 1152e-3
        self.pb2 = 659e-3
        self.pb3 = 486e-3
        self.pb4 = 92e-3

    def _phi(self):
        return _q10_factor(self.temp, self.q10_base, ref_celsius=23.0)

    def _alpha_factor(self, V):
        return u.math.exp((self.Qo * u.faraday_constant * V) / (u.gas_constant * self.temp))

    def _beta_factor(self, V):
        return u.math.exp((self.Qc * u.faraday_constant * V) / (u.gas_constant * self.temp))

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, K, Ca, batch_size=batch_size)

    def current(self, V, K: IonInfo, Ca: IonInfo):
        states = self.state_values()
        return self.g_max * (states["O0"] + states["O1"] + states["O2"] + states["O3"] + states["O4"]) * (K.E - V)

    def c01(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c12(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c23(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c34(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o01(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o12(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o23(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o34(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c10(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c21(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c32(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c43(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def o10(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o21(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o32(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o43(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def f0(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf0 * self._alpha_factor(V) * self._phi()

    def f1(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf1 * self._alpha_factor(V) * self._phi()

    def f2(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf2 * self._alpha_factor(V) * self._phi()

    def f3(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf3 * self._alpha_factor(V) * self._phi()

    def f4(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf4 * self._alpha_factor(V) * self._phi()

    def b0(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb0 * self._beta_factor(V) * self._phi()

    def b1(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb1 * self._beta_factor(V) * self._phi()

    def b2(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb2 * self._beta_factor(V) * self._phi()

    def b3(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb3 * self._beta_factor(V) * self._phi()

    def b4(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb4 * self._beta_factor(V) * self._phi()


@register_channel("Kca1p1_MA2025_BC")
class Kca1p1_MA2025_BC(Kca1p1_MA2020_GoC):
    r"""Kca1.1 Ca- and voltage-activated K current, basket-cell variant.

    The same ten-state Kca1.1 Markov scheme documented in
    :class:`Kca1p1_MA2020_GoC`, reused unchanged for the cerebellar
    basket cell model of (Masoli et al., 2025) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
        Inherited from :class:`Kca1p1_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca1p1_MA2020_GoC : The base class; full state topology,
        equations, the 22/23 degree Celsius temperature note, and
        the unused ``L0`` attribute are documented there.
    Kca1p1_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca1p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca1p1_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca1p1_MA25_BC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca1p1_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric
           gating of a large conductance Ca-activated K+ channel.
           The Journal of General Physiology, 110(3), 257-281.
           doi:10.1085/jgp.110.3.257
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


@register_channel("Kca1p1_MA2020_GrC")
class Kca1p1_MA2020_GrC(Kca1p1_MA2020_GoC):
    r"""Kca1.1 Ca- and voltage-activated K current, granule-cell variant.

    The same ten-state Kca1.1 Markov scheme documented in
    :class:`Kca1p1_MA2020_GoC`, reused unchanged for the cerebellar
    granule cell subtype model of (Masoli et al., 2020) [3]_. This
    class is a Python subclass of :class:`Kca1p1_MA2020_GoC`, but
    that inheritance relationship is purely a code-reuse device: the
    kinetics and model citation below belong to the granule cell
    paper, not to the Golgi cell paper cited on the base class.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
        Inherited from :class:`Kca1p1_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca1p1_MA2020_GoC : The Python base class supplying the shared
        state topology and rate equations; cites the Golgi cell
        paper, not the granule cell paper cited here.
    Kca1p1_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca1p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.
    Kca1p1_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca1p1_MA20_GrC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca1p1_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ. Despite the Python inheritance from the *Golgi*
    cell class, the correct model citation is the *granule* cell
    paper below -- the two cerebellar cell-type papers were published
    together and this citation is not interchangeable with the one
    on :class:`Kca1p1_MA2020_GoC`.

    References
    ----------
    .. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric
           gating of a large conductance Ca-activated K+ channel.
           The Journal of General Physiology, 110(3), 257-281.
           doi:10.1085/jgp.110.3.257
    .. [2] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [3] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"


@register_channel("Kca1p1_MA2024_PC")
class Kca1p1_MA2024_PC(Kca1p1_MA2020_GoC):
    r"""Kca1.1 Ca- and voltage-activated K current, Purkinje-cell variant.

    The same ten-state Kca1.1 Markov scheme documented in
    :class:`Kca1p1_MA2020_GoC`, reused unchanged for the human
    Purkinje cell model of (Masoli et al., 2024) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
        Inherited from :class:`Kca1p1_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca1p1_MA2020_GoC : The base class; full state topology,
        equations, the 22/23 degree Celsius temperature note, and
        the unused ``L0`` attribute are documented there.
    Kca1p1_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca1p1_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca1p1_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``Kca1p1_MA24_PC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca1p1_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric
           gating of a large conductance Ca-activated K+ channel.
           The Journal of General Physiology, 110(3), 257-281.
           doi:10.1085/jgp.110.3.257
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


@register_channel("Kca1p1_RI2021_SC")
class Kca1p1_RI2021_SC(Kca1p1_MA2020_GoC):
    r"""Kca1.1 Ca- and voltage-activated K current, stellate-cell variant.

    The same ten-state Kca1.1 Markov scheme documented in
    :class:`Kca1p1_MA2020_GoC`, reused unchanged for the cerebellar
    stellate cell model of (Rizza et al., 2021) [3]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximal conductance density, default ``10.0 mS/cm2``.
        Inherited from :class:`Kca1p1_MA2020_GoC`.
    q10_base : array-like, optional
        Q10 temperature coefficient, default ``3.0``. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    temp : array-like, optional
        Absolute temperature, default 22 degrees Celsius. Inherited
        from :class:`Kca1p1_MA2020_GoC`.
    name : str, optional
        Optional channel name.
    solver : str, optional
        Markov-chain integration scheme.
    substeps : int, optional
        Number of solver substeps per integration step.

    See Also
    --------
    Kca1p1_MA2020_GoC : The base class; full state topology,
        equations, the 22/23 degree Celsius temperature note, and
        the unused ``L0`` attribute are documented there.
    Kca1p1_MA2025_BC : Same kinetics, basket-cell model citation.
    Kca1p1_MA2020_GrC : Same kinetics, granule-cell model citation.
    Kca1p1_MA2024_PC : Same kinetics, Purkinje-cell model citation.

    Notes
    -----
    Ported from ``Kca1p1_RI21_SC.mod``. This class does not override
    ``__init__``: the constructor, state topology, and rate methods
    are all inherited unchanged from :class:`Kca1p1_MA2020_GoC`.
    Only the ``register_channel`` key and this docstring's model
    citation differ.

    References
    ----------
    .. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric
           gating of a large conductance Ca-activated K+ channel.
           The Journal of General Physiology, 110(3), 257-281.
           doi:10.1085/jgp.110.3.257
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
