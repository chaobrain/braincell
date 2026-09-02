# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Optional

import brainstate
import braintools
import brainunit as u

from braincell._base_ion import Ion
from braincell._typing import Initializer, Size
from braincell.mech import register_ion
from braincell.ion._base import (
    Conserve,
    DynamicNernstIon,
    Factor,
    FixedIon,
    InitNernstIon,
    KineticIon,
    Reaction,
    Source,
    Species,
    _RadialShellGeometry,
    species_initializer_view,
)

__all__ = [
    'Calcium',
    'CalciumFixed',
    'CalciumInitNernst',
    'CalciumDetailed',
    'CalciumFirstOrder',
    'ToyCaBindingKinetic_SU2015_DCN',
    'ToyCaBindingSourceKinetic_SU2015_DCN',
    'ToyCaBindingIcaSourceKinetic_SU2015_DCN',
    'ToyDiamFactorKinetic_SU2015_DCN',
    'ToyCaPumpFactorKinetic_SU2015_DCN',
    'CdpStC_CAMOnly_MA2020_GoC',
    'CdpStC_NoCAM_MA2020_GoC',
    'CdpStC_MA2025_BC',
    'CdpStC_MA2020_GoC',
    'CdpCAM_MA2024_PC',
    'CdpCR_MA2020_GrC',
    'CdpStC_RI2021_SC',
    'CdpHVA_SU2015_DCN',
    'CdpLVA_SU2015_DCN',
]


class _ParvalbuminEquilibrium(brainstate.mixin.Mixin):
    r"""Resting occupancy of a parvalbumin pool competing for Ca and Mg.

    Parvalbumin binds calcium and magnesium at the same site, so its
    three states are fixed by two dissociation ratios rather than one:
    :math:`k_{dc} = c_{null} m_1 / m_2` for calcium and
    :math:`k_{dm} = mg_{null} p_1 / p_2` for magnesium. The free,
    calcium-bound, and magnesium-bound fractions then partition
    ``PVnull`` as :math:`1 : k_{dc} : k_{dm}`.

    The chemistry is identical wherever the pool appears, so the three
    ``cdp`` mechanisms that carry parvalbumin mix this in rather than
    restating it. It is calcium chemistry, not shell geometry, which is
    why it lives here and not on
    :class:`~braincell.ion._base._RadialShellGeometry`.

    Requires ``cainull``, ``mginull``, ``PVnull``, and the four rate
    parameters ``m1``, ``m2``, ``p1``, ``p2`` on the concrete ion.

    See Also
    --------
    CdpStC_NoCAM_MA2020_GoC : Golgi-cell pool carrying parvalbumin.
    CdpStC_MA2020_GoC : The same pool with calmodulin added.
    CdpCAM_MA2024_PC : Purkinje-cell pool with parvalbumin and calbindin.
    """

    def _pv_dissociation_ratios(self):
        """Return the ``(calcium, magnesium)`` dissociation ratios."""
        return (self.cainull * self.m1) / self.m2, (self.mginull * self.p1) / self.p2

    def _ss_pv_free(self):
        """Return the metal-free parvalbumin fraction at equilibrium."""
        kdc, kdm = self._pv_dissociation_ratios()
        return self.PVnull / (1.0 + kdc + kdm)

    def _ss_pv_ca(self):
        """Return the calcium-bound parvalbumin fraction at equilibrium."""
        kdc, kdm = self._pv_dissociation_ratios()
        return (self.PVnull * kdc) / (1.0 + kdc + kdm)

    def _ss_pv_mg(self):
        """Return the magnesium-bound parvalbumin fraction at equilibrium."""
        kdc, kdm = self._pv_dissociation_ratios()
        return (self.PVnull * kdm) / (1.0 + kdc + kdm)


class Calcium(Ion):
    """Base class for modeling the calcium ion species.

    ``Calcium`` collects the physiological defaults shared by every
    concrete calcium ion model in BrainCell and provides the
    ``Ion``/``IonChannel`` container interface calcium channels
    attach to. It carries no dynamics of its own.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Ion`.
    name : str or None, optional
        Runtime instance name. Defaults to ``None``, in which case the
        instance is unnamed. Forwarded unchanged to :class:`Ion`.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Ion`.

    See Also
    --------
    CalciumFixed : Fixed-parameter calcium ion built on this base.
    CalciumInitNernst : Calcium ion with ``E`` initialized from the
        Nernst equation.
    CalciumDetailed : Dynamic calcium ion with a first-order
        concentration model and Nernst-computed ``E``.
    CalciumFirstOrder : Sibling dynamic calcium ion with a different
        first-order form.

    Notes
    -----
    This is an abstract base class and must be subclassed (for
    example by :class:`CalciumFixed` or :class:`CalciumDetailed`) to
    obtain a concrete calcium ion model with defined reversal-
    potential dynamics. ``default_Ci``, ``default_Co``, and
    ``default_valence`` below are values conventional in the
    calcium-modeling literature, not measurements reported by a
    single paper; no citation is asserted for them.

    Attributes
    ----------
    ion_symbol : str
        Symbol used for runtime family lookup. Set to ``'Ca'``.
    default_Ci : brainunit.Quantity
        Default intracellular calcium concentration, ``5e-5 mM``.
    default_Co : brainunit.Quantity
        Default extracellular calcium concentration, ``2.0 mM``.
    default_valence : int
        Default ionic valence, ``2``.
    """

    __module__ = 'braincell.ion'

    ion_symbol = 'Ca'
    default_Ci = 5e-05 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2


@register_ion("CalciumFixed")
class CalciumFixed(Calcium, FixedIon):
    r"""Fixed calcium dynamics.

    This calcium model has no dynamics. It holds a fixed reversal
    potential :math:`E` and fixed concentrations :math:`C_i`/:math:`C_o`.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    E : array-like or callable or None, optional
        Fixed reversal potential. Defaults to ``+120 mV``. Passing
        ``None`` explicitly raises :class:`ValueError`; there is no
        class-default fallback for this argument.
    Ci : array-like or callable or None, optional
        Intracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Ci` inside
        :meth:`FixedIon._init_fixed_ion`.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`FixedIon._init_fixed_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Calcium.default_valence` inside
        :meth:`FixedIon._init_fixed_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``E`` is explicitly passed as ``None``.
        :meth:`FixedIon._init_fixed_ion` requires an explicit fixed
        reversal potential and does not fall back to a class default
        for ``E``.

    See Also
    --------
    Calcium : Base calcium ion family this class fixes.
    CalciumInitNernst : Sibling calcium model whose ``E`` is computed
        from the Nernst equation instead of fixed.

    Notes
    -----
    With the shipped class defaults (``Co = 2.0 mM``, ``Ci = 5e-5
    mM``, ``valence = 2``) at 36 degrees Celsius, the Nernst equation
    gives ``E = +141.15 mV``. ``CalciumFixed`` instead defaults ``E``
    to ``+120 mV``, so the two sibling classes disagree by about
    21 mV when both are constructed with no arguments.
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        E: Optional[Initializer] = 120.0 * u.mV,
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("CalciumInitNernst")
class CalciumInitNernst(Calcium, InitNernstIon):
    r"""Fixed ``Ci``/``Co`` calcium model with ``E`` initialized from Nernst.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    Ci : array-like or callable or None, optional
        Intracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Ci` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Calcium.default_valence` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`InitNernstIon._init_nernst_ion` requires an explicit
        temperature and does not fall back to a class default.

    See Also
    --------
    Calcium : Base calcium ion family this class computes a reversal
        potential for.
    CalciumFixed : Sibling calcium model with a fixed reversal
        potential instead of a Nernst-computed one.

    Notes
    -----
    ``E`` is not computed at construction time. ``_init_nernst_ion``
    sets ``self.E = None`` immediately, and the actual reversal
    potential is filled in only later by
    ``InitNernstIon._update_reversal``, which fires from the
    ``_ion_init_state_hook`` and ``_ion_reset_state_hook`` lifecycle
    hooks. Between construction and the first state initialization or
    reset, ``E`` is ``None``.

    The stored formula, transcribed as ``_update_reversal`` writes it,
    is the Nernst equation

    .. math::

        E = \frac{R \cdot \mathrm{temp}}{\mathrm{valence} \cdot F}
            \log\!\left(\frac{C_o}{C_i}\right)

    where :math:`R` is the gas constant and :math:`F` is the Faraday
    constant. The argument to the logarithm is :math:`C_o / C_i`
    (extracellular over intracellular), and ``valence`` divides inside
    the prefactor rather than appearing as a separate multiplicative
    term. House policy treats the Nernst equation as a textbook
    result, so it is named here without a citation.
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_nernst_ion(Ci=Ci, Co=Co, temp=temp, valence=valence)


@register_ion("CalciumDetailed")
class CalciumDetailed(Calcium, DynamicNernstIon):
    r"""Dynamic calcium concentration with a Nernst-computed reversal.

    :meth:`derivative` implements only the first-order relaxation
    model of section 2 below. Section 1 reproduces, for background
    only, the fuller ATP-pump model this family of models is drawn
    from: **the pump kinetics in section 1 are NOT implemented by
    this class.** ``derivative`` is three lines long and contains no
    Michaelis-Menten term and no pump parameter.

    **1. Background: the dynamics of intracellular** :math:`Ca^{2+}`
    **(not implemented)**

    The dynamics of intracellular :math:`Ca^{2+}` were determined by two contributions [3]_ :

    *(i) Influx of* :math:`Ca^{2+}` *due to Calcium currents*

    :math:`Ca^{2+}` ion enter through :math:`Ca^{2+}` channel and diffuse into the
    interior of the cell. Only the :math:`Ca^{2+}` concentration in a thin shell beneath
    the membrane was modeled. The influx of :math:`Ca^{2+}` into such a thin shell followed:

    .. math::

        [Ca]_{i}=-\frac{I_{Ca}}{2 F d}

    where :math:`F=96489\, \mathrm{C\, mol^{-1}}` is the Faraday constant,
    :math:`d=1\, \mathrm{\mu m}` is the depth of the shell beneath the membrane,
    :math:`I_T` in :math:`\mathrm{\mu A/cm^{2}}` and :math:`[Ca]_{i}` in millimolar,
    and :math:`I_{Ca}` is the summation of all :math:`Ca^{2+}` currents.

    *(ii) Efflux of* :math:`Ca^{2+}` *due to an active pump*

    In a thin shell beneath the membrane, :math:`Ca^{2+}` retrieval usually consists of a
    combination of several processes, such as binding to :math:`Ca^{2+}` buffers, calcium
    efflux due to :math:`Ca^{2+}` ATPase pump activity and diffusion to neighboring shells.
    Only the :math:`Ca^{2+}` pump was modeled here. We adopted the following kinetic scheme:

    .. math::

        Ca _{i}^{2+}+ P \overset{c_1}{\underset{c_2}{\rightleftharpoons}} CaP \xrightarrow{c_3} P+ Ca _{0}^{2+}

    where P represents the :math:`Ca^{2+}` pump, CaP is an intermediate state,
    :math:`Ca _{ o }^{2+}` is the extracellular :math:`Ca^{2+}` concentration,
    and :math:`c_{1}, c_{2}` and :math:`c_{3}` are rate constants. :math:`Ca^{2+}`
    ion have a high affinity for the pump :math:`P`, whereas extrusion of
    :math:`Ca^{2+}` follows a slower process (Blaustein, 1988 ). Therefore,
    :math:`c_{3}` is low compared to :math:`c_{1}` and :math:`c_{2}` and the
    Michaelis-Menten approximation can be used for describing the kinetics of the pump.
    According to such a scheme, the kinetic equation for the :math:`Ca^{2+}` pump is:

    .. math::

        \frac{[Ca^{2+}]_{i}}{dt}=-\frac{K_{T}[Ca]_{i}}{[Ca]_{i}+K_{d}}

    where :math:`K_{T}=10^{-4}\, \mathrm{mM\, ms^{-1}}` is the product of :math:`c_{3}`
    with the total concentration of :math:`P` and :math:`K_{d}=c_{2} / c_{1}=10^{-4}\, \mathrm{mM}`
    is the dissociation constant, which can be interpreted here as the value of
    :math:`[Ca]_{i}` at which the pump is half activated (if :math:`[Ca]_{i} \ll K_{d}`
    then the efflux is negligible). None of this pump exists in
    :meth:`derivative` below -- no saturating term and no pump parameter
    appears anywhere in this class.

    **2. The implemented model: a simple first-order relaxation**

    What :meth:`derivative` actually computes is the first-order model of
    (Bazhenov, et al., 1998) [2]_, which that paper's own Appendix credits
    to (Destexhe et al., 1994) [1]_:

    .. math::

        \frac{d\left[Ca^{2+}\right]_{i}}{d t}=-\frac{I_{Ca}}{z F d}+\frac{\left[Ca^{2+}\right]_{rest}-\left[C a^{2+}\right]_{i}}{\tau_{Ca}}

    where :math:`I_{Ca}` is the summation of all :math:`Ca ^{2+}` currents, :math:`d`
    is the thickness of the perimembrane "shell" in which calcium is able to affect
    membrane properties :math:`(1.\, \mathrm{\mu m})`, :math:`z=2` is the valence of the
    :math:`Ca ^{2+}` ion, :math:`F` is the Faraday constant, and :math:`\tau_{C a}` is
    the :math:`Ca ^{2+}` removal rate. The resting :math:`Ca ^{2+}` concentration was
    set to be :math:`\left[ Ca ^{2+}\right]_{\text {rest}}=2.4\times 10^{-4}\, \mathrm{mM}`
    (:math:`0.24\, \mathrm{\mu M}`), matching the default ``C_rest`` below exactly.
    BrainCell additionally clamps the influx term at zero before adding it (see
    Notes), which neither paper does.

    **3. The reversal potential**

    The reversal potential of calcium :math:`Ca ^{2+}` is calculated according to the
    Nernst equation:

    .. math::

        E = k'{RT \over 2F} log{[Ca^{2+}]_0 \over [Ca^{2+}]_i}

    where :math:`R=8.31441 \, \mathrm{J} /(\mathrm{mol}^{\circ} \mathrm{K})`,
    :math:`T=309.15^{\circ} \mathrm{K}`,
    :math:`F=96,489 \mathrm{C} / \mathrm{mol}`,
    and :math:`\left[\mathrm{Ca}^{2+}\right]_{0}=2 \mathrm{mM}`.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.DynamicNernstIon.E`. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    d : array-like or callable, optional
        Thickness :math:`d` of the peri-membrane calcium shell.
        Defaults to ``1.0 um``.
    tau : array-like or callable, optional
        Time constant :math:`\tau_{Ca}` of the calcium removal rate.
        Defaults to ``5.0 ms``.
    C_rest : array-like or callable, optional
        Resting intracellular calcium concentration
        :math:`[Ca^{2+}]_{rest}` that ``Ci`` relaxes toward. Defaults
        to ``2.4e-4 mM`` (``0.24 uM``).
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the dynamic ``Ci`` state. Defaults to a
        constant ``2.4e-4 mM`` initializer, matching ``C_rest``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion` requires an
        explicit temperature and does not fall back to a class
        default.

    See Also
    --------
    Calcium : Base calcium ion family this class computes a reversal
        potential for.
    CalciumFirstOrder : Sibling dynamic calcium ion with a different
        first-order form.

    Notes
    -----
    ``derivative`` rectifies the influx term with
    ``u.math.maximum(drive, 0)`` so that only inward calcium current
    raises ``Ci``; no such clamp appears in either Destexhe et al.
    (1994) [1]_ or Bazhenov et al. (1998) [2]_, and it is a BrainCell
    addition.

    Bazhenov's Appendix writes the influx term with one lumped
    constant, :math:`A = 5.18 \times 10^{-5}\, \mathrm{mM\, cm^2 /
    (ms\, \mu A)}`. BrainCell instead writes it out as
    :math:`1/(zFd)`, with :math:`z=2` hard-coded and ``d`` exposed as
    a constructor parameter. The two are the same term parameterized
    differently, not a divergence -- it is why ``d`` is a BrainCell
    parameter and not one of Bazhenov's.

    Section 1's Faraday and gas constants, :math:`F = 96489\,
    \mathrm{C/mol}` and :math:`R = 8.31441\, \mathrm{J/(mol\, K)}`,
    are the paper's own literal values, quoted above for reference
    only. The code instead uses the CODATA constants
    ``u.faraday_constant`` and ``u.gas_constant`` (via
    :attr:`~braincell.ion._base.DynamicNernstIon.E`), neither of
    which is a constructor parameter.

    References
    ----------
    .. [1] Destexhe, A., Contreras, D., Sejnowski, T. J., & Steriade, M.
           (1994). A model of spindle rhythmicity in the isolated thalamic
           reticular nucleus. Journal of Neurophysiology, 72(2), 803-818.
           doi:10.1152/jn.1994.72.2.803
    .. [2] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
           (1998). Cellular and network models for intrathalamic
           augmenting responses during 10-Hz stimulation. Journal of
           Neurophysiology, 79(5), 2730-2748.
           doi:10.1152/jn.1998.79.5.2730
    .. [3] Destexhe, A., Babloyantz, A., & Sejnowski, T. J. (1993). Ionic
           mechanisms for intrinsic slow oscillations in thalamic relay
           neurons. Biophysical Journal, 65(4), 1538-1552.
           doi:10.1016/S0006-3495(93)81190-1
    """

    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        d: Initializer = 1.0 * u.um,
        tau: Initializer = 5.0 * u.ms,
        C_rest: Initializer = 2.4e-4 * u.mM,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = braintools.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        # parameters
        self.d = braintools.init.param(d, self.varshape, allow_none=False)
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)
        self.C_rest = braintools.init.param(C_rest, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        drive = total_current / (2 * u.faraday_constant * self.d)
        drive = u.math.maximum(drive, u.math.zeros_like(drive))
        return drive + (self.C_rest - Ci) / self.tau


@register_ion("CalciumFirstOrder")
class CalciumFirstOrder(Calcium, DynamicNernstIon):
    r"""First-order calcium concentration model with rectified current drive.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.DynamicNernstIon.E`. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    alpha : array-like or callable, optional
        Scale factor applied to the rectified current-drive term in
        :meth:`derivative`. Defaults to ``0.13`` (a bare, unitless
        number).
    beta : array-like or callable, optional
        First-order decay rate applied to ``Ci`` in :meth:`derivative`.
        Defaults to ``0.075`` (a bare, unitless number).
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the dynamic ``Ci`` state. Defaults to a
        constant ``2.4e-4 mM`` initializer.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion` requires an
        explicit temperature and does not fall back to a class
        default.

    See Also
    --------
    Calcium : Base calcium ion family this class computes a reversal
        potential for.
    CalciumDetailed : Sibling dynamic calcium ion with a first-order
        relaxation model driven by a unit-converted current term.

    Notes
    -----
    :meth:`derivative` computes

    .. math::

        \frac{dCi}{dt} = \max(\alpha \cdot I_{Ca},\ 0) - \beta \cdot Ci

    i.e. a *rectified*, positively-scaled current drive minus a
    first-order decay -- not a symmetric ``-alpha*I_Ca - beta*Ca``
    form. ``alpha`` and ``beta`` are generic first-order-model
    coefficients with no identified literature source; they are not
    traceable to a specific paper from the code alone, so none is
    cited here.

    ``alpha`` and ``beta`` are stored as bare, unitless numbers, while
    ``total_current`` (the calcium current summed over attached
    channels) carries current-density units such as
    :math:`\mathrm{\mu A/cm^2}`. Because :meth:`derivative` compares
    ``self.alpha * total_current`` directly against ``0.0 * u.mM`` in
    ``u.math.maximum``, calling ``derivative`` with any real,
    unit-typed ``total_current`` raises
    ``brainunit.UnitMismatchError``; calling it with no channels
    attached instead raises ``TypeError``, because ``current()``
    returns ``None`` rather than a zero quantity when there are no
    channels to sum. Unlike :class:`CalciumDetailed`, which divides
    its current term by :math:`2Fd` before clamping (so the clamped
    quantity and the zero it is compared against share the same
    derived unit), this class performs no such conversion, so
    ``derivative`` is not currently callable with a real
    ``total_current`` in either configuration. This is an existing
    implementation defect, not a documentation issue; it is recorded
    here rather than fixed, since this change is documentation-only.
    """

    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        alpha: Initializer = 0.13,
        beta: Initializer = 0.075,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = braintools.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        # parameters
        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        drive = u.math.maximum(self.alpha * total_current, 0.0 * u.mM)
        return drive - self.beta * Ci


@register_ion("ToyCaBindingKinetic_SU2015_DCN")
class ToyCaBindingKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy for ``KineticIon`` validation.

    This is a BrainCell import-path validation fixture, not a model of
    a published mechanism -- the ``SU2015_DCN`` suffix follows
    BrainCell's naming convention only. The mechanism models one
    reversible buffering step:

    .. math::

       Ca_i + B \rightleftharpoons BC

    with the conserved pool:

    .. math::

       B + BC = B_{tot}

    ``B`` is solved algebraically from the conservation rule while ``Ci`` and
    ``BC`` are integrated as differential species.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 36
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    kf : array-like or callable, optional
        Forward rate constant of the ``Ci + B -> BC`` reaction.
        Defaults to ``2.0 / (mM * ms)``.
    kb : array-like or callable, optional
        Backward rate constant of the ``BC -> Ci + B`` reaction.
        Defaults to ``0.5 / ms``.
    Btot : array-like or callable, optional
        Total conserved concentration of ``B + BC``. Defaults to
        ``1.0 mM``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the ``Ci`` species. Defaults to ``0.10 mM``.
    BC_initializer : array-like or callable, optional
        Initializer for the ``BC`` species. Defaults to ``0.00 mM``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"rk4"``.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``5``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``, or if
        ``substeps`` is less than ``1``. Both come from
        :meth:`KineticIon._init_kinetic_ion`.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    ToyCaBindingSourceKinetic_SU2015_DCN : Same reaction network plus
        a constant source on ``Ci``.
    ToyCaBindingIcaSourceKinetic_SU2015_DCN : Same reaction network
        plus a current-driven source on ``Ci``.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all five toy
        fixtures.

    Notes
    -----
    ``B`` is not a differential species: it is declared in
    :attr:`species` for readability, but :attr:`conserves` marks it as
    the ``algebraic`` member of the ``B + BC = Btot`` conservation
    relation, so its value is recovered from ``BC`` rather than
    integrated.
    """

    __module__ = "braincell.ion"

    #: View onto ``species_initializers["BC"]``, kept readable
    #: because ``_compute.ions`` reflects over ``__init__``.
    BC_initializer = species_initializer_view("BC")

    species = (
        Species("Ci", init=0.10 * u.mM),
        Species("B", init=1.00 * u.mM),
        Species("BC", init=0.00 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("B", "BC"),
            algebraic="B",
            total=lambda self, V, x: self.Btot,
        ),
    )

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kf: Initializer = 2.0 / (u.mM * u.ms),
        kb: Initializer = 0.5 / u.ms,
        Btot: Initializer = 1.0 * u.mM,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = 0.10 * u.mM,
        BC_initializer: Initializer = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)


@register_ion("ToyCaBindingSourceKinetic_SU2015_DCN")
class ToyCaBindingSourceKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy with a constant ``Ci`` source.

    This is a BrainCell import-path validation fixture, not a model of
    a published mechanism -- the ``SU2015_DCN`` suffix follows
    BrainCell's naming convention only. The mechanism keeps the same
    reversible binding network as :class:`ToyCaBindingKinetic_SU2015_DCN`:

    .. math::

       Ca_i + B \rightleftharpoons BC

    and adds one constant source term on ``Ci``:

    .. math::

       \frac{d Ca_i}{dt}\Big|_{\text{source}} = s_{Ca}

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 36
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    kf : array-like or callable, optional
        Forward rate constant of the ``Ci + B -> BC`` reaction.
        Defaults to ``2.0 / (mM * ms)``.
    kb : array-like or callable, optional
        Backward rate constant of the ``BC -> Ci + B`` reaction.
        Defaults to ``0.5 / ms``.
    Btot : array-like or callable, optional
        Total conserved concentration of ``B + BC``. Defaults to
        ``1.0 mM``.
    ci_source : array-like or callable, optional
        Constant source rate :math:`s_{Ca}` added to ``dCi/dt``.
        Defaults to ``0.002 mM / ms``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the ``Ci`` species. Defaults to ``0.10 mM``.
    BC_initializer : array-like or callable, optional
        Initializer for the ``BC`` species. Defaults to ``0.00 mM``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"rk4"``.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``5``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``, or if
        ``substeps`` is less than ``1``. Both come from
        :meth:`KineticIon._init_kinetic_ion`.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    ToyCaBindingKinetic_SU2015_DCN : Same reaction network without the
        constant source.
    ToyCaBindingIcaSourceKinetic_SU2015_DCN : Sibling fixture whose
        source is driven by current instead of a constant rate.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all five toy
        fixtures.

    Notes
    -----
    ``B`` is not a differential species: :attr:`conserves` marks it as
    the ``algebraic`` member of the ``B + BC = Btot`` conservation
    relation, so its value is recovered from ``BC`` rather than
    integrated, exactly as in :class:`ToyCaBindingKinetic_SU2015_DCN`.
    """

    __module__ = "braincell.ion"

    #: View onto ``species_initializers["BC"]``, kept readable
    #: because ``_compute.ions`` reflects over ``__init__``.
    BC_initializer = species_initializer_view("BC")

    species = ToyCaBindingKinetic_SU2015_DCN.species
    reactions = ToyCaBindingKinetic_SU2015_DCN.reactions
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: self.ci_source,
        ),
    )
    conserves = ToyCaBindingKinetic_SU2015_DCN.conserves

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kf: Initializer = 2.0 / (u.mM * u.ms),
        kb: Initializer = 0.5 / u.ms,
        Btot: Initializer = 1.0 * u.mM,
        ci_source: Initializer = 0.002 * u.mM / u.ms,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = 0.10 * u.mM,
        BC_initializer: Initializer = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)
        self.ci_source = braintools.init.param(ci_source, self.varshape, allow_none=False)


@register_ion("ToyCaBindingIcaSourceKinetic_SU2015_DCN")
class ToyCaBindingIcaSourceKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy with current-driven ``Ci`` source.

    This is a BrainCell import-path validation fixture, not a model of
    a published mechanism -- the ``SU2015_DCN`` suffix follows
    BrainCell's naming convention only. The mechanism keeps the same
    reversible binding network as the earlier toy kinetic ions and
    drives ``Ci`` with inward-positive calcium current.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 36
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    kf : array-like or callable, optional
        Forward rate constant of the ``Ci + B -> BC`` reaction.
        Defaults to ``2.0 / (mM * ms)``.
    kb : array-like or callable, optional
        Backward rate constant of the ``BC -> Ci + B`` reaction.
        Defaults to ``0.5 / ms``.
    Btot : array-like or callable, optional
        Total conserved concentration of ``B + BC``. Defaults to
        ``1.0 mM``.
    kCa : array-like or callable, optional
        Current-to-flux scale factor used by the ``Ci`` source.
        Defaults to ``3.45e-7 / coulomb``.
    depth : array-like or callable, optional
        Shell depth used by the ``Ci`` source's unit conversion.
        Defaults to ``0.2 um``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the ``Ci`` species. Defaults to ``0.10 mM``.
    BC_initializer : array-like or callable, optional
        Initializer for the ``BC`` species. Defaults to ``0.00 mM``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"rk4"``.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``5``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``, or if
        ``substeps`` is less than ``1``. Both come from
        :meth:`KineticIon._init_kinetic_ion`.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    ToyCaBindingKinetic_SU2015_DCN : Same reaction network without any
        source term.
    ToyCaBindingSourceKinetic_SU2015_DCN : Sibling fixture with a
        constant, rather than current-driven, source.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all five toy
        fixtures.

    Notes
    -----
    The ``Ci`` source evaluates to zero when ``total_current`` is
    ``None`` (no channels attached), and otherwise to
    ``(kCa / depth) * total_current * 1e4``, with ``total_current``
    read in ``mA/cm**2`` and the ``1e4`` factor converting that
    current density into the ``mM/ms`` units the source needs -- the
    standard cross-sectional-area conversion used across BrainCell's
    current-driven calcium sources. This class sets
    ``uses_total_current = True``, so :meth:`KineticIon`'s derivative
    hook always supplies a real ``total_current`` once at least one
    channel is attached, unlike :class:`CalciumFirstOrder`.
    """

    __module__ = "braincell.ion"

    #: View onto ``species_initializers["BC"]``, kept readable
    #: because ``_compute.ions`` reflects over ``__init__``.
    BC_initializer = species_initializer_view("BC")
    uses_total_current = True

    species = ToyCaBindingKinetic_SU2015_DCN.species
    reactions = ToyCaBindingKinetic_SU2015_DCN.reactions
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: (
                braintools.init.param(0.0 * (u.mM / u.ms), self.varshape)
                if total_current is None
                else (
                    u.get_mantissa(self.kCa)
                    / self.depth.to_decimal(u.um)
                    * total_current.to_decimal(u.mA / u.cm**2)
                    * 1e4
                )
                * (u.mM / u.ms)
            ),
        ),
    )
    conserves = ToyCaBindingKinetic_SU2015_DCN.conserves

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kf: Initializer = 2.0 / (u.mM * u.ms),
        kb: Initializer = 0.5 / u.ms,
        Btot: Initializer = 1.0 * u.mM,
        kCa: Initializer = 3.45e-7 / u.coulomb,
        depth: Initializer = 0.2 * u.um,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = 0.10 * u.mM,
        BC_initializer: Initializer = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)
        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)


@register_ion("ToyCaPumpFactorKinetic_SU2015_DCN")
class ToyCaPumpFactorKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal factor-crossing toy with cytosolic and pump-area compartments.

    This is a BrainCell import-path validation fixture, not a model of
    a published mechanism -- the ``SU2015_DCN`` suffix follows
    BrainCell's naming convention only. ``Ci`` lives in a cytosolic
    volume factor while pump states live in an area-like factor. The
    toy keeps the state count minimal while exercising mixed-factor
    reaction, conservation, and current-driven source paths, plus an
    irreversible (one-way) release reaction.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 36
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    kf : array-like or callable, optional
        Forward rate constant of the ``Ci + PumpFree -> PumpBound``
        binding reaction. Defaults to ``2.0 / (mM * ms)``.
    kb : array-like or callable, optional
        Backward rate constant of the ``PumpBound -> Ci + PumpFree``
        unbinding reaction. Defaults to ``0.5 / ms``.
    k_rel : array-like or callable, optional
        Rate constant of the irreversible ``PumpBound -> PumpFree``
        release reaction (no back-reaction; ``backward=None``).
        Defaults to ``0.05 / ms``.
    PumpTot : array-like or callable, optional
        Per-area total pump concentration used by the conserved pool
        ``PumpFree + PumpBound = PumpTot * pump_area``. Defaults to
        ``1.0 mM * um``.
    kCa : array-like or callable, optional
        Current-to-flux scale factor used by the ``Ci`` source.
        Defaults to ``3.45e-7 / coulomb``.
    depth : array-like or callable, optional
        Shell depth used by the ``Ci`` source's unit conversion.
        Defaults to ``0.2 um``.
    cyt_volume : array-like or callable, optional
        Constant cytosolic volume backing the ``cyto`` factor that
        ``Ci`` lives in. Defaults to ``3.0 um**3``. Unlike
        :class:`ToyDiamFactorKinetic_SU2015_DCN`, this value is a
        plain constructor constant, not derived from compartment
        geometry.
    pump_area : array-like or callable, optional
        Constant membrane area backing the ``pump_area`` factor that
        ``PumpFree``/``PumpBound`` live in. Defaults to ``3.0 um**2``.
        Also a plain constructor constant, not geometry-derived.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the ``Ci`` species. Defaults to ``0.10 mM``.
    PumpBound_initializer : array-like or callable, optional
        Initializer for the ``PumpBound`` species. Defaults to
        ``0.00 mM * um``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"rk4"``.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``5``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``, or if
        ``substeps`` is less than ``1``. Both come from
        :meth:`KineticIon._init_kinetic_ion`.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    ToyCaBindingIcaSourceKinetic_SU2015_DCN : Sibling fixture with the
        same current-driven ``Ci`` source but a single, non-factor
        binding partner.
    ToyDiamFactorKinetic_SU2015_DCN : Sibling fixture whose factors
        are derived from runtime compartment geometry instead of
        fixed constructor constants.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all five toy
        fixtures, including the ``Factor`` mechanism.

    Notes
    -----
    ``PumpFree`` is the algebraic member of the conserved pool
    ``PumpFree + PumpBound = PumpTot * pump_area``; it is recovered
    from ``PumpBound`` by subtraction rather than integrated directly.
    The second reaction, ``PumpBound -> PumpFree``, has
    ``backward=None`` and is therefore irreversible: it drains
    ``PumpBound`` at rate ``k_rel * pump_area`` with no corresponding
    forward-binding contribution of its own.

    The ``Ci`` source evaluates to zero when ``total_current`` is
    ``None`` (no channels attached), and otherwise to
    ``cyt_volume * (kCa / depth) * total_current * 1e4``, with
    ``total_current`` read in ``mA/cm**2``. This mirrors
    :class:`ToyCaBindingIcaSourceKinetic_SU2015_DCN`'s source formula,
    additionally scaled by ``cyt_volume`` because ``Ci`` here lives in
    a volume factor rather than being unfactored. This class sets
    ``uses_total_current = True``.
    """

    __module__ = "braincell.ion"

    #: View onto ``species_initializers["PumpBound"]``, kept readable
    #: because ``_compute.ions`` reflects over ``__init__``.
    PumpBound_initializer = species_initializer_view("PumpBound")
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.cyt_volume),
        Factor("pump_area", lambda self: self.pump_area),
    )
    species = (
        Species("Ci", init=0.10 * u.mM, factor="cyto"),
        Species("PumpFree", init=1.00 * u.mM * u.um, factor="pump_area"),
        Species("PumpBound", init=0.00 * u.mM * u.um, factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "PumpFree": 1},
            rhs={"PumpBound": 1},
            forward=lambda self, V, x: self.kf * self.pump_area,
            backward=lambda self, V, x: self.kb * self.pump_area,
        ),
        Reaction(
            lhs={"PumpBound": 1},
            rhs={"PumpFree": 1},
            forward=lambda self, V, x: self.k_rel * self.pump_area,
            backward=None,
        ),
    )
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: (
                braintools.init.param(0.0 * (u.mM * u.um**3 / u.ms), self.varshape)
                if total_current is None
                else self.cyt_volume
                * (
                    (
                        u.get_mantissa(self.kCa)
                        / self.depth.to_decimal(u.um)
                        * total_current.to_decimal(u.mA / u.cm**2)
                        * 1e4
                    )
                    * (u.mM / u.ms)
                )
            ),
        ),
    )
    conserves = (
        Conserve(
            species=("PumpFree", "PumpBound"),
            algebraic="PumpFree",
            total=lambda self, V, x: self.PumpTot * self.pump_area,
        ),
    )

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kf: Initializer = 2.0 / (u.mM * u.ms),
        kb: Initializer = 0.5 / u.ms,
        k_rel: Initializer = 0.05 / u.ms,
        PumpTot: Initializer = 1.0 * u.mM * u.um,
        kCa: Initializer = 3.45e-7 / u.coulomb,
        depth: Initializer = 0.2 * u.um,
        cyt_volume: Initializer = 3.0 * u.um**3,
        pump_area: Initializer = 3.0 * u.um**2,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = 0.10 * u.mM,
        PumpBound_initializer: Initializer = 0.00 * u.mM * u.um,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "PumpBound": PumpBound_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.k_rel = braintools.init.param(k_rel, self.varshape, allow_none=False)
        self.PumpTot = braintools.init.param(PumpTot, self.varshape, allow_none=False)
        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)
        self.cyt_volume = braintools.init.param(cyt_volume, self.varshape, allow_none=False)
        self.pump_area = braintools.init.param(pump_area, self.varshape, allow_none=False)


@register_ion("ToyDiamFactorKinetic_SU2015_DCN")
class ToyDiamFactorKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal geometry-factor toy with runtime-derived cytosolic strip factors.

    This is a BrainCell import-path validation fixture, not a model of
    a published mechanism -- the ``SU2015_DCN`` suffix follows
    BrainCell's naming convention only. ``Ci`` lives in a thin strip
    volume derived from the runtime midpoint diameter, while
    ``PumpFree`` and ``PumpBound`` live on a line-like membrane factor:

    .. math::

       cyto = \pi \cdot diam_{mid} \cdot depth

       pump\_area = \pi \cdot diam_{mid}

    The mechanism then exercises a reversible reaction

    .. math::

       Ca_i + PumpFree \rightleftharpoons PumpBound

    together with the conserved pool

    .. math::

       PumpFree + PumpBound = PumpTot \cdot pump\_area

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 36
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    kf : array-like or callable, optional
        Forward rate constant of the ``Ci + PumpFree -> PumpBound``
        reaction, applied per unit ``pi * diam_mid``. Defaults to
        ``2.0 / (mM * ms)``.
    kb : array-like or callable, optional
        Backward rate constant of the ``PumpBound -> Ci + PumpFree``
        reaction, applied per unit ``pi * diam_mid``. Defaults to
        ``0.5 / ms``.
    PumpTot : array-like or callable, optional
        Per-area total pump concentration used by the conserved pool
        ``PumpFree + PumpBound = PumpTot * pi * diam_mid``. Defaults
        to ``1.0 mM * um``.
    depth : array-like or callable, optional
        Strip depth multiplying ``pi * diam_mid`` to form the
        ``cyto`` factor that ``Ci`` lives in. Defaults to ``1.0 um``,
        which differs from every other toy fixture in this module
        (they default ``depth`` to ``0.2 um``).
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable, optional
        Initializer for the ``Ci`` species. Defaults to ``0.10 mM``.
    PumpBound_initializer : array-like or callable, optional
        Initializer for the ``PumpBound`` species. Defaults to
        ``0.00 mM * um``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"backward_euler"``, unlike every other toy fixture in this
        module (they default to ``"rk4"``).
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``1``, unlike every other toy fixture in this
        module (they default to ``5``).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``, or if
        ``substeps`` is less than ``1``. Both come from
        :meth:`KineticIon._init_kinetic_ion`.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    ToyCaPumpFactorKinetic_SU2015_DCN : Sibling fixture with the same
        binding/pump topology, but whose ``cyto``/``pump_area``
        factors are fixed constructor constants instead of
        geometry-derived quantities.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all five toy
        fixtures, including the ``Factor`` mechanism.

    Notes
    -----
    ``diam_mid`` is **not** a constructor parameter of this class. It
    is a compartment-geometry attribute injected at runtime onto every
    ion instance from the enclosing compartment's context (see
    ``braincell.mech._context``, populated from
    ``braincell._discretization``), documented there as the diameter
    at the control-volume midpoint. The ``cyto`` and ``pump_area``
    factors, and the reaction/conservation coefficients above, read
    ``self.diam_mid`` directly and so are only well-defined once this
    ion is attached to a compartment that supplies that context.

    This class has no ``sources``: ``Ci`` is driven only by the
    binding reaction, with no current-driven or constant influx term.

    ``PumpFree`` is the algebraic member of the conserved pool; it is
    recovered from ``PumpBound`` by subtraction rather than integrated
    directly.

    The corresponding ``.mod`` fixture's ``pump_area``/``cyto``
    constants are compiled by NEURON to the fixed decimal
    ``62.8319``, a rounding of the exact value
    ``pi * diam_mid * depth`` (``62.83185307179586`` for the
    fixture's reference geometry) would produce. BrainCell does not
    substitute that compiled decimal: both factors are recomputed at
    runtime from ``self.diam_mid``, so results track the live geometry
    exactly rather than reproducing NEURON's rounded constant. No
    other toy or ``Cdp*`` mechanism in this module carries this
    exception.
    """

    __module__ = "braincell.ion"

    #: View onto ``species_initializers["PumpBound"]``, kept readable
    #: because ``_compute.ions`` reflects over ``__init__``.
    PumpBound_initializer = species_initializer_view("PumpBound")

    factors = (
        Factor("cyto", lambda self: u.math.pi * self.diam_mid * self.depth),
        Factor("pump_area", lambda self: u.math.pi * self.diam_mid),
    )
    species = (
        Species("Ci", init=0.10 * u.mM, factor="cyto"),
        Species("PumpFree", init=1.00 * u.mM * u.um, factor="pump_area"),
        Species("PumpBound", init=0.00 * u.mM * u.um, factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "PumpFree": 1},
            rhs={"PumpBound": 1},
            forward=lambda self, V, x: self.kf * u.math.pi * self.diam_mid,
            backward=lambda self, V, x: self.kb * u.math.pi * self.diam_mid,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("PumpFree", "PumpBound"),
            algebraic="PumpFree",
            total=lambda self, V, x: self.PumpTot * u.math.pi * self.diam_mid,
        ),
    )

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kf: Initializer = 2.0 / (u.mM * u.ms),
        kb: Initializer = 0.5 / u.ms,
        PumpTot: Initializer = 1.0 * u.mM * u.um,
        depth: Initializer = 1.0 * u.um,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = 0.10 * u.mM,
        PumpBound_initializer: Initializer = 0.00 * u.mM * u.um,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "PumpBound": PumpBound_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.PumpTot = braintools.init.param(PumpTot, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)


# ---------------------------------------------------------------------------
# Declaration blocks shared by the ported ``cdp`` calcium pools.
#
# The five ``Cdp*`` mechanisms below are different reaction networks over a
# common substrate: the same cytosolic buffer set, the same parvalbumin
# triple, the same surface pump pair, and -- where calmodulin is modelled at
# all -- the same nine CaM states. Each block is declared once here and
# composed into the class tables, so a correction to the shared chemistry
# reaches every model that uses it instead of three or four hand-kept copies.
# ---------------------------------------------------------------------------

_CI_SPECIES = (Species("Ci", init=0.0 * u.mM, factor="cyto"),)

_BUFFER_SPECIES = (
    Species("mg", init=0.0 * u.mM, factor="cyto"),
    Species("Buff1", init=0.0 * u.mM, factor="cyto"),
    Species("Buff1_ca", init=0.0 * u.mM, factor="cyto"),
    Species("Buff2", init=0.0 * u.mM, factor="cyto"),
    Species("Buff2_ca", init=0.0 * u.mM, factor="cyto"),
    Species("BTC", init=0.0 * u.mM, factor="cyto"),
    Species("BTC_ca", init=0.0 * u.mM, factor="cyto"),
    Species("DMNPE", init=0.0 * u.mM, factor="cyto"),
    Species("DMNPE_ca", init=0.0 * u.mM, factor="cyto"),
)

_PV_SPECIES = (
    Species("PV", init=0.0 * u.mM, factor="cyto"),
    Species("PV_ca", init=0.0 * u.mM, factor="cyto"),
    Species("PV_mg", init=0.0 * u.mM, factor="cyto"),
)

_PUMP_SPECIES = (
    Species("pump", init=0.0 * (u.mol / u.cm**2), factor="pump_area"),
    Species("pumpca", init=0.0 * (u.mol / u.cm**2), factor="pump_area"),
)

#: Calmodulin states, in the binding order the ``cdp`` mod files declare them.
_CAM_STATES = (
    "CAM0",
    "CAM1C",
    "CAM2C",
    "CAM1N2C",
    "CAM1N",
    "CAM2N",
    "CAM2N1C",
    "CAM1C1N",
    "CAM4",
)


_BUFFER_REACTIONS = (
    Reaction(
        lhs={"pump": 1, "Ci": 1},
        rhs={"pumpca": 1},
        forward=lambda self, V, x: self.kpmp1 * self.parea,
        backward=lambda self, V, x: self.kpmp2 * self.parea,
    ),
    Reaction(
        lhs={"pumpca": 1},
        rhs={"pump": 1},
        forward=lambda self, V, x: self.kpmp3 * self.parea,
        backward=None,
    ),
    Reaction(
        lhs={"Ci": 1, "Buff1": 1},
        rhs={"Buff1_ca": 1},
        forward=lambda self, V, x: self.rf1 * self.dsqvol,
        backward=lambda self, V, x: self.rf2 * self.dsqvol,
    ),
    Reaction(
        lhs={"Ci": 1, "Buff2": 1},
        rhs={"Buff2_ca": 1},
        forward=lambda self, V, x: self.rf3 * self.dsqvol,
        backward=lambda self, V, x: self.rf4 * self.dsqvol,
    ),
    Reaction(
        lhs={"Ci": 1, "BTC": 1},
        rhs={"BTC_ca": 1},
        forward=lambda self, V, x: self.b1 * self.dsqvol,
        backward=lambda self, V, x: self.b2 * self.dsqvol,
    ),
    Reaction(
        lhs={"Ci": 1, "DMNPE": 1},
        rhs={"DMNPE_ca": 1},
        forward=lambda self, V, x: self.c1 * self.dsqvol,
        backward=lambda self, V, x: self.c2 * self.dsqvol,
    ),
)

_PV_REACTIONS = (
    Reaction(
        lhs={"Ci": 1, "PV": 1},
        rhs={"PV_ca": 1},
        forward=lambda self, V, x: self.m1 * self.dsqvol,
        backward=lambda self, V, x: self.m2 * self.dsqvol,
    ),
    Reaction(
        lhs={"mg": 1, "PV": 1},
        rhs={"PV_mg": 1},
        forward=lambda self, V, x: self.p1 * self.dsqvol,
        backward=lambda self, V, x: self.p2 * self.dsqvol,
    ),
)


def _cam_species(factor: str) -> tuple:
    """Return the nine calmodulin species, scaled by ``factor``.

    The Golgi-cell pools give calmodulin its own unit-carrying
    ``"cam_unit"`` factor, so its states are not scaled by the shell
    volume a second time; the Purkinje-cell pool scales them by
    ``"cyto"`` like every other cytosolic species. The states themselves
    are the same nine either way, which is why the factor is the only
    parameter.

    Parameters
    ----------
    factor : str
        Name of the :class:`~braincell.ion._base.Factor` these states
        are converted through.

    Returns
    -------
    tuple of braincell.ion._base.Species
        The nine calmodulin states, in ``_CAM_STATES`` order.
    """
    return tuple(Species(name, init=0.0 * u.mM, factor=factor) for name in _CAM_STATES)


@register_ion("CdpStC_CAMOnly_MA2020_GoC")
class CdpStC_CAMOnly_MA2020_GoC(Calcium, _RadialShellGeometry, KineticIon):
    r"""Import of the calmodulin-only ``CdpStC_CAMOnly_MA20_GoC.mod``.

    Isolates the calmodulin (CaM) subnetwork of the imported Golgi-cell
    calcium pool so its binding kinetics can be validated independently
    of the pump and non-CaM buffers that :class:`CdpStC_MA2020_GoC` also
    tracks. The scheme is a two-lobe CaM binding model: an independent
    C-lobe and N-lobe, each with two sequential, reversible
    calcium-binding steps, reaching the fully-loaded ``CAM4`` state
    through four distinct binding orders.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 25
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    Nannuli : array-like or callable, optional
        Radial-shell count inherited from the NEURON multi-shell
        diffusion template. BrainCell tracks a single well-mixed
        ``Ci`` pool, so ``Nannuli`` only shapes the effective volume
        fraction returned by :attr:`vrat` (``dr2 = 0.25 /
        (Nannuli - 1)``); no shell diffusion is performed. Defaults
        to ``10.9495``.
    cainull : array-like or callable, optional
        Baseline/initial free calcium concentration ``Ci``. Defaults
        to ``45e-6 mM``.
    CAM_start : array-like or callable, optional
        Initial concentration of apo-calmodulin, ``CAM0``. Defaults
        to ``0.03 mM``.
    K1Coff, K1Con : array-like or callable, optional
        Backward and forward rate constants of the first C-lobe
        binding step. Default ``0.04 /ms`` and ``5.4 /(mM*ms)``.
    K2Coff, K2Con : array-like or callable, optional
        Backward and forward rate constants of the second C-lobe
        binding step. Default ``0.00925 /ms`` and ``15.0 /(mM*ms)``.
    K1Noff, K1Non : array-like or callable, optional
        Backward and forward rate constants of the first N-lobe
        binding step. Default ``2.5 /ms`` and ``142.5 /(mM*ms)``.
    K2Noff, K2Non : array-like or callable, optional
        Backward and forward rate constants of the second N-lobe
        binding step. Default ``0.75 /ms`` and ``175.0 /(mM*ms)``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the ``Ci`` species. Defaults to ``None``,
        which falls back to ``cainull``.
    species_initializers : dict or None, optional
        Per-species initializer overrides, keyed by one of this
        class's ten differential species (``Ci``, ``CAM0``,
        ``CAM1C``, ``CAM2C``, ``CAM1N2C``, ``CAM1N``, ``CAM2N``,
        ``CAM2N1C``, ``CAM1C1N``, ``CAM4``). Defaults to ``None``
        (no overrides).
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"backward_euler"``, matching
        :attr:`~braincell.ion._base.KineticIon.default_solver`.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``1``, matching
        :attr:`~braincell.ion._base.KineticIon.default_substeps`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``species_initializers`` names a species outside the ten
        listed above, or if ``temp`` is explicitly passed as
        ``None``, or ``substeps`` is less than ``1`` (the latter two
        raised by :meth:`KineticIon._init_kinetic_ion`).
    AttributeError
        Raised during state initialization or reset if this ion's
        compartment geometry (``diam_arc_mean``) has not been
        attached yet.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    CdpStC_NoCAM_MA2020_GoC : Sibling decomposition keeping the pump
        and non-CaM buffers while dropping this class's CaM network.
    CdpStC_MA2020_GoC : The undivided mechanism, combining this CaM
        network with the pump and non-CaM buffers.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all ``Cdp*``
        mechanisms.

    Notes
    -----
    Ported from ``GoC/ion/CdpStC_CAMOnly_MA20_GoC.mod``, part of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [4]_. That
    file carries a title only, no credit block; the CaM subnetwork it
    isolates belongs to the ``CdpStC`` mechanism of Anwar, Hong & De
    Schutter [1]_, whose extended buffer parameters come from Schmidt
    et al. (2003) [2]_ and whose pump rate was tuned to data from
    Maeda et al. (1999) [3]_.
    ``uses_total_current = False`` and ``sources = ()``: this pool has
    no calcium influx or pump of its own, so ``Ci`` is consumed by the
    twelve CaM reactions above and never resupplied. It is meant to be
    exercised in isolation, matching the ``.mod`` file's role of
    validating the CaM subnetwork rather than serving as a standalone
    physiological pool.

    The ``cam_unit`` factor scales the nine CaM-state species by a
    unit-magnitude, ``um**2``-dimensioned array rather than by
    :attr:`dsqvol` again: the imported NMODL ``COMPARTMENT`` scaling
    applies once, to the shared cytosolic volume that ``Ci`` occupies,
    and must not be applied a second time to each CaM row.

    ``CdpStC_NoCAM_MA2020_GoC`` hardcodes ``solver="backward_euler"``
    and ``substeps=1`` as literal defaults, as this class does,
    whereas :class:`CdpStC_MA2020_GoC` instead defaults both to
    ``None`` and lets :meth:`KineticIon._init_kinetic_ion` fall back
    to the same class-level
    :attr:`~braincell.ion._base.KineticIon.default_solver` and
    :attr:`~braincell.ion._base.KineticIon.default_substeps`; the
    observable defaults are identical either way.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.ion"
    uses_total_current = False

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        # NEURON sparse treats the CAM rows differently from ``ca``: their
        # reaction rates carry the same ``dsqvol`` unit bridge, but the CAM
        # state rows are not multiplied by the cytosolic compartment factor
        # again. We match that by giving CAM states a unit-compatible factor
        # whose magnitude is 1 instead of ``dsqvol``.
        Factor(
            "cam_unit",
            lambda self: 1.0 * u.um**2,
        ),
    )
    species = _CI_SPECIES + _cam_species("cam_unit")
    reactions = (
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2C": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N2C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM2N": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N1C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
    )
    sources = ()
    conserves = ()

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(25.0),
        Nannuli: Initializer = 10.9495,
        cainull: Initializer = 45e-6 * u.mM,
        CAM_start: Initializer = 0.03 * u.mM,
        K1Coff: Initializer = 0.04 / u.ms,
        K1Con: Initializer = 5.4 / (u.mM * u.ms),
        K2Coff: Initializer = 0.00925 / u.ms,
        K2Con: Initializer = 15.0 / (u.mM * u.ms),
        K1Noff: Initializer = 2.5 / u.ms,
        K1Non: Initializer = 142.5 / (u.mM * u.ms),
        K2Noff: Initializer = 0.75 / u.ms,
        K2Non: Initializer = 175.0 / (u.mM * u.ms),
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.CAM_start = braintools.init.param(CAM_start, self.varshape, allow_none=False)
        self.K1Coff = braintools.init.param(K1Coff, self.varshape, allow_none=False)
        self.K1Con = braintools.init.param(K1Con, self.varshape, allow_none=False)
        self.K2Coff = braintools.init.param(K2Coff, self.varshape, allow_none=False)
        self.K2Con = braintools.init.param(K2Con, self.varshape, allow_none=False)
        self.K1Noff = braintools.init.param(K1Noff, self.varshape, allow_none=False)
        self.K1Non = braintools.init.param(K1Non, self.varshape, allow_none=False)
        self.K2Noff = braintools.init.param(K2Noff, self.varshape, allow_none=False)
        self.K2Non = braintools.init.param(K2Non, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "CAM0": self.CAM_start,
            "CAM1C": 0.0 * u.mM,
            "CAM2C": 0.0 * u.mM,
            "CAM1N2C": 0.0 * u.mM,
            "CAM1N": 0.0 * u.mM,
            "CAM2N": 0.0 * u.mM,
            "CAM2N1C": 0.0 * u.mM,
            "CAM1C1N": 0.0 * u.mM,
            "CAM4": 0.0 * u.mM,
        }


@register_ion("CdpStC_NoCAM_MA2020_GoC")
class CdpStC_NoCAM_MA2020_GoC(Calcium, _ParvalbuminEquilibrium, _RadialShellGeometry, KineticIon):
    r"""BrainCell-factored calcium pool: pump, non-CaM buffers, no CaM.

    Keeps the pump and non-calmodulin buffer subnetworks of the
    imported Golgi-cell ``CdpStC`` calcium pool while dropping the
    calmodulin (CaM) reactions of :class:`CdpStC_CAMOnly_MA2020_GoC`
    entirely. Unlike its sibling classes, this one is not itself a
    direct port of a ``.mod`` file: no
    ``CdpStC_NoCAM_MA20_GoC.mod`` exists in the imported source tree.
    It is a BrainCell-factored base whose literal parameter set
    matches ``BC/ion/CdpStC_MA25_BC.mod`` and
    ``SC/ion/CdpStC_RI21_SC.mod`` -- both of which are exactly this
    GoC ``CdpStC`` mechanism with the CaM subnetwork commented out --
    so it exists to be shared by :class:`CdpStC_MA2025_BC` and
    :class:`CdpStC_RI2021_SC` rather than to stand for a distinct
    published mechanism. The CAM reactions were removed, not
    replaced by different kinetics.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 25
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    Nannuli : array-like or callable, optional
        Radial-shell count inherited from the NEURON multi-shell
        diffusion template; only shapes the single effective volume
        fraction :attr:`vrat`. Defaults to ``10.9495``.
    cainull : array-like or callable, optional
        Baseline/initial free calcium concentration ``Ci``. Defaults
        to ``45e-6 mM``.
    mginull : array-like or callable, optional
        Baseline/initial magnesium concentration ``mg``. Defaults to
        ``0.59 mM``.
    Buffnull1 : array-like or callable, optional
        Total concentration of the first generic buffer, ``Buff1 +
        Buff1_ca``. Defaults to ``0.0 mM``.
    rf1, rf2 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff1``
        binding step. Default ``0.0134329 /(mM*ms)`` and
        ``0.0397469 /ms``.
    Buffnull2 : array-like or callable, optional
        Total concentration of the second generic buffer, ``Buff2 +
        Buff2_ca``. Defaults to ``60.9091 mM``.
    rf3, rf4 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff2``
        binding step. Default ``0.1435 /(mM*ms)`` and ``0.0014 /ms``.
    BTCnull : array-like or callable, optional
        Total concentration of the BTC indicator dye buffer, ``BTC +
        BTC_ca``. Defaults to ``0.0 mM``.
    b1, b2 : array-like or callable, optional
        Forward and backward rate constants of the ``BTC`` binding
        step. Default ``5.33 /(mM*ms)`` and ``0.08 /ms``.
    DMNPEnull : array-like or callable, optional
        Total concentration of the caged-calcium buffer DMNPE,
        ``DMNPE + DMNPE_ca``. Defaults to ``0.0 mM``.
    c1, c2 : array-like or callable, optional
        Forward and backward rate constants of the ``DMNPE`` binding
        step. Default ``5.63 /(mM*ms)`` and ``0.107e-3 /ms``.
    PVnull : array-like or callable, optional
        Total concentration of parvalbumin, ``PV + PV_ca + PV_mg``.
        Defaults to ``0.08 mM``.
    m1, m2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV`` calcium
        binding step. Default ``1.07e2 /(mM*ms)`` and
        ``9.5e-4 /ms``.
    p1, p2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV``
        magnesium binding step. Default ``0.8 /(mM*ms)`` and
        ``2.5e-2 /ms``.
    kpmp1, kpmp2 : array-like or callable, optional
        Forward and backward rate constants of the ``pump + Ci ->
        pumpca`` binding step. Default ``3e-3 /(mM*ms)`` and
        ``1.75e-5 /ms``.
    kpmp3 : array-like or callable, optional
        Rate constant of the irreversible extrusion step,
        ``pumpca -> pump``. Defaults to ``7.255e-5 /ms``.
    TotalPump : array-like or callable, optional
        Areal pump-site density; the conserved sum of ``pump +
        pumpca`` per unit membrane area. Defaults to
        ``1e-9 mol/cm2``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the ``Ci`` species. Defaults to ``None``,
        which falls back to ``cainull``.
    species_initializers : dict or None, optional
        Per-species initializer overrides, keyed by one of this
        class's fourteen differential species (``Ci``, ``mg``,
        ``Buff1``, ``Buff1_ca``, ``Buff2``, ``Buff2_ca``, ``BTC``,
        ``BTC_ca``, ``DMNPE``, ``DMNPE_ca``, ``PV``, ``PV_ca``,
        ``PV_mg``, ``pump``). Defaults to ``None`` (no overrides);
        unset buffer/PV species default to their steady-state
        occupancy at ``cainull``/``mginull``, and ``pump`` defaults
        to ``TotalPump``.
    solver : str, optional
        Integrator name used for the reaction network. Defaults to
        ``"backward_euler"``, matching
        :attr:`~braincell.ion._base.KineticIon.default_solver`.
    substeps : int, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``1``, matching
        :attr:`~braincell.ion._base.KineticIon.default_substeps`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``species_initializers`` names a species outside the
        fourteen listed above, or if ``temp`` is explicitly passed
        as ``None``, or ``substeps`` is less than ``1`` (the latter
        two raised by :meth:`KineticIon._init_kinetic_ion`).
    AttributeError
        Raised during state initialization or reset, or from
        :attr:`parea`/:attr:`dsq`, if this ion's compartment geometry
        (``diam_arc_mean``) has not been attached yet.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    CdpStC_CAMOnly_MA2020_GoC : Sibling decomposition keeping only
        the CaM network this class omits.
    CdpStC_MA2020_GoC : The undivided mechanism, combining this
        pump/buffer network with the CaM network.
    CdpStC_MA2025_BC : Thin basket-cell subclass reusing this class's
        network unchanged.
    CdpStC_RI2021_SC : Thin stellate-cell subclass reusing this
        class's network unchanged.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all ``Cdp*``
        mechanisms.

    Notes
    -----
    This class has no source ``.mod`` file of its own; see the
    extended summary. Its credits are those of the GoC ``CdpStC``
    mechanism it factors, part of the cerebellar Golgi cell model of
    (Masoli et al., 2020) [4]_: Anwar, Hong & De Schutter [1]_ as the
    reference for the mechanism, the extended buffer parameters from
    Schmidt et al. (2003) [2]_, and the pump rate tuned to data from
    Maeda et al. (1999) [3]_. ``uses_total_current = True`` and one
    :class:`~braincell.ion._base.Source` drives ``Ci`` from the
    channel current supplied at each step:
    ``_ci_source_flux`` returns zero when no current is supplied, and
    otherwise ``total_current * pi * diam_arc_mean / (2 *
    faraday_constant)``. NEURON's raw GoC ``ica`` is efflux-positive,
    but BrainCell channel currents follow the repo-wide inward-positive
    convention, so a positive ``total_current`` here increases ``Ci``.

    One :class:`~braincell.ion._base.Conserve` constrains ``pump +
    pumpca = TotalPump * parea``, with ``pumpca`` recovered
    algebraically rather than integrated; ``pump`` is the only pump
    state among the fourteen differential species.

    Buffer- and PV-bound species initialize at the equilibrium
    occupancy implied by their dissociation constants and
    ``cainull``/``mginull`` (see ``_ss_buffer_free``,
    ``_ss_buffer_bound``, ``_ss_pv_free``, ``_ss_pv_ca``,
    ``_ss_pv_mg``), not at zero, so steady state is reached without a
    long settling transient.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
    )
    species = _CI_SPECIES + _BUFFER_SPECIES + _PV_SPECIES + _PUMP_SPECIES
    reactions = _BUFFER_REACTIONS + _PV_REACTIONS
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: self._ci_source_flux(total_current),
        ),
    )
    conserves = (
        Conserve(
            species=("pump", "pumpca"),
            algebraic="pumpca",
            total=lambda self, V, x: self.TotalPump * self.parea,
        ),
    )

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(25.0),
        Nannuli: Initializer = 10.9495,
        cainull: Initializer = 45e-6 * u.mM,
        mginull: Initializer = 0.59 * u.mM,
        Buffnull1: Initializer = 0.0 * u.mM,
        rf1: Initializer = 0.0134329 / (u.mM * u.ms),
        rf2: Initializer = 0.0397469 / u.ms,
        Buffnull2: Initializer = 60.9091 * u.mM,
        rf3: Initializer = 0.1435 / (u.mM * u.ms),
        rf4: Initializer = 0.0014 / u.ms,
        BTCnull: Initializer = 0.0 * u.mM,
        b1: Initializer = 5.33 / (u.mM * u.ms),
        b2: Initializer = 0.08 / u.ms,
        DMNPEnull: Initializer = 0.0 * u.mM,
        c1: Initializer = 5.63 / (u.mM * u.ms),
        c2: Initializer = 0.107e-3 / u.ms,
        PVnull: Initializer = 0.08 * u.mM,
        m1: Initializer = 1.07e2 / (u.mM * u.ms),
        m2: Initializer = 9.5e-4 / u.ms,
        p1: Initializer = 0.8 / (u.mM * u.ms),
        p2: Initializer = 2.5e-2 / u.ms,
        kpmp1: Initializer = 3e-3 / (u.mM * u.ms),
        kpmp2: Initializer = 1.75e-5 / u.ms,
        kpmp3: Initializer = 7.255e-5 / u.ms,
        TotalPump: Initializer = 1e-9 * (u.mol / u.cm**2),
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.PVnull = braintools.init.param(PVnull, self.varshape, allow_none=False)
        self.m1 = braintools.init.param(m1, self.varshape, allow_none=False)
        self.m2 = braintools.init.param(m2, self.varshape, allow_none=False)
        self.p1 = braintools.init.param(p1, self.varshape, allow_none=False)
        self.p2 = braintools.init.param(p2, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "PV": self._ss_pv_free(),
            "PV_ca": self._ss_pv_ca(),
            "PV_mg": self._ss_pv_mg(),
            "pump": self.TotalPump,
        }


@register_ion("CdpStC_MA2025_BC")
class CdpStC_MA2025_BC(CdpStC_NoCAM_MA2020_GoC):
    r"""Calcium pool for the basket cell, no calmodulin buffering.

    The same pump/non-CaM-buffer/parvalbumin kinetic network
    documented in :class:`CdpStC_NoCAM_MA2020_GoC`, reused unchanged
    for the cerebellar basket cell model of Masoli et al. (2025)
    [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Ion state shape. Inherited from :class:`CdpStC_NoCAM_MA2020_GoC`.
    temp, Nannuli, cainull, mginull, Buffnull1, rf1, rf2, Buffnull2
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    rf3, rf4, BTCnull, b1, b2, DMNPEnull, c1, c2, PVnull, m1, m2
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    p1, p2, kpmp1, kpmp2, kpmp3, TotalPump, Co, Ci_initializer
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    species_initializers, solver, substeps, name, **channels
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.

    See Also
    --------
    CdpStC_NoCAM_MA2020_GoC : The base class; full network,
        parameters and equilibrium initializers are documented
        there.
    CdpStC_RI2021_SC : Same kinetics, stellate-cell model citation.

    Notes
    -----
    Ported from ``BC/ion/CdpStC_MA25_BC.mod``, whose header names
    Anwar, Hong & De Schutter [1]_ as the reference for the
    mechanism, credits the extended buffer parameters to Schmidt
    et al. (2003) [2]_, and records the pump rate as tuned to data
    from Maeda et al. (1999) [3]_. This class does not
    override ``__init__``: the reaction network, the source and the
    conservation relation are all inherited unchanged from
    :class:`CdpStC_NoCAM_MA2020_GoC`. Only the ``register_ion`` key
    and this docstring's model citation differ -- the ``.mod`` file
    this class ports from is the same GoC ``CdpStC`` mechanism with
    its calmodulin block commented out, per the imported README's
    "Ion_dyn inherited variants" table, not a distinct kinetic
    scheme.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2025).
           Cerebellar basket cell filtering of Purkinje cell
           responses elicited by low frequency parallel fibre
           transmission. Scientific Reports, 15(1), 25192.
           doi:10.1038/s41598-025-09964-2
    """

    __module__ = "braincell.ion"


@register_ion("CdpStC_RI2021_SC")
class CdpStC_RI2021_SC(CdpStC_NoCAM_MA2020_GoC):
    r"""Calcium pool for the stellate cell, no calmodulin buffering.

    The same pump/non-CaM-buffer/parvalbumin kinetic network
    documented in :class:`CdpStC_NoCAM_MA2020_GoC`, reused unchanged
    for the cerebellar stellate cell model of Rizza et al. (2021)
    [4]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        Ion state shape. Inherited from :class:`CdpStC_NoCAM_MA2020_GoC`.
    temp, Nannuli, cainull, mginull, Buffnull1, rf1, rf2, Buffnull2
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    rf3, rf4, BTCnull, b1, b2, DMNPEnull, c1, c2, PVnull, m1, m2
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    p1, p2, kpmp1, kpmp2, kpmp3, TotalPump, Co, Ci_initializer
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.
    species_initializers, solver, substeps, name, **channels
        Identical in meaning and default to
        :class:`CdpStC_NoCAM_MA2020_GoC`; not restated here.

    See Also
    --------
    CdpStC_NoCAM_MA2020_GoC : The base class; full network,
        parameters and equilibrium initializers are documented
        there.
    CdpStC_MA2025_BC : Same kinetics, basket-cell model citation.

    Notes
    -----
    Ported from ``SC/ion/CdpStC_RI21_SC.mod``, whose header names
    Anwar, Hong & De Schutter [1]_ as the reference for the
    mechanism, credits the extended buffer parameters to Schmidt
    et al. (2003) [2]_, and records the pump rate as tuned to data
    from Maeda et al. (1999) [3]_. This class does not
    override ``__init__``: the reaction network, the source and the
    conservation relation are all inherited unchanged from
    :class:`CdpStC_NoCAM_MA2020_GoC`. Only the ``register_ion`` key
    and this docstring's model citation differ. The source ``.mod``
    file also reads an extracellular calcium variable ``cao`` that
    its kinetic equations never use; BrainCell drops that unused
    read rather than reproducing it.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
           Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate
           cell computational modeling predicts signal filtering in
           the molecular layer circuit of cerebellum. Scientific
           Reports, 11(1), 3873.
           doi:10.1038/s41598-021-83209-w
    """

    __module__ = "braincell.ion"


@register_ion("CdpStC_MA2020_GoC")
class CdpStC_MA2020_GoC(Calcium, _ParvalbuminEquilibrium, _RadialShellGeometry, KineticIon):
    r"""Golgi-cell calcium pool: pump, generic buffers, PV, and CaM.

    Undivided import of the Golgi-cell ``CdpStC`` calcium pool:
    ``Ci`` (the NMODL ``ca``/``cai`` pool) is buffered by two generic
    first-order buffers, the indicator dyes BTC and DMNPE, and
    parvalbumin (PV), extruded by a membrane pump, and additionally
    binds calmodulin (CaM) through the same four-site cooperative
    scheme documented in :class:`CdpStC_CAMOnly_MA2020_GoC`. This is
    the combination that :class:`CdpStC_CAMOnly_MA2020_GoC` (CaM
    branch only) and :class:`CdpStC_NoCAM_MA2020_GoC` (pump/buffer/PV
    branch only) each keep half of.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 25
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    Nannuli : array-like or callable, optional
        Radial-shell count inherited from the NEURON multi-shell
        diffusion template; only shapes the single effective volume
        fraction :attr:`vrat`. Defaults to ``10.9495``.
    cainull : array-like or callable, optional
        Baseline/initial free calcium concentration ``Ci``. Defaults
        to ``45e-6 mM``.
    mginull : array-like or callable, optional
        Baseline/initial magnesium concentration ``mg``. Defaults to
        ``0.59 mM``.
    Buffnull1 : array-like or callable, optional
        Total concentration of the first generic buffer, ``Buff1 +
        Buff1_ca``. Defaults to ``0.0 mM``.
    rf1, rf2 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff1``
        binding step. Default ``0.0134329 /(mM*ms)`` and
        ``0.0397469 /ms``.
    Buffnull2 : array-like or callable, optional
        Total concentration of the second generic buffer, ``Buff2 +
        Buff2_ca``. Defaults to ``60.9091 mM``.
    rf3, rf4 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff2``
        binding step. Default ``0.1435 /(mM*ms)`` and ``0.0014 /ms``.
    BTCnull : array-like or callable, optional
        Total concentration of the BTC indicator dye buffer, ``BTC +
        BTC_ca``. Defaults to ``0.0 mM``.
    b1, b2 : array-like or callable, optional
        Forward and backward rate constants of the ``BTC`` binding
        step. Default ``5.33 /(mM*ms)`` and ``0.08 /ms``.
    DMNPEnull : array-like or callable, optional
        Total concentration of the caged-calcium buffer DMNPE,
        ``DMNPE + DMNPE_ca``. Defaults to ``0.0 mM``.
    c1, c2 : array-like or callable, optional
        Forward and backward rate constants of the ``DMNPE`` binding
        step. Default ``5.63 /(mM*ms)`` and ``0.107e-3 /ms``.
    PVnull : array-like or callable, optional
        Total concentration of parvalbumin, ``PV + PV_ca + PV_mg``.
        Defaults to ``0.08 mM``.
    m1, m2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV`` calcium
        binding step. Default ``1.07e2 /(mM*ms)`` and
        ``9.5e-4 /ms``.
    p1, p2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV``
        magnesium binding step. Default ``0.8 /(mM*ms)`` and
        ``2.5e-2 /ms``.
    CAM_start : array-like or callable, optional
        Baseline/initial concentration of unbound calmodulin,
        ``CAM0``. Defaults to ``0.03 mM``.
    K1Coff, K1Con : array-like or callable, optional
        Backward and forward rate constants of the first C-lobe
        calcium-binding step. Default ``0.04 /ms`` and
        ``5.4 /(mM*ms)``.
    K2Coff, K2Con : array-like or callable, optional
        Backward and forward rate constants of the second C-lobe
        calcium-binding step. Default ``0.00925 /ms`` and
        ``15.0 /(mM*ms)``.
    K1Noff, K1Non : array-like or callable, optional
        Backward and forward rate constants of the first N-lobe
        calcium-binding step. Default ``2.5 /ms`` and
        ``142.5 /(mM*ms)``.
    K2Noff, K2Non : array-like or callable, optional
        Backward and forward rate constants of the second N-lobe
        calcium-binding step. Default ``0.75 /ms`` and
        ``175.0 /(mM*ms)``.
    kpmp1, kpmp2 : array-like or callable, optional
        Forward and backward rate constants of the ``pump + Ci ->
        pumpca`` binding step. Default ``3e-3 /(mM*ms)`` and
        ``1.75e-5 /ms``.
    kpmp3 : array-like or callable, optional
        Rate constant of the irreversible extrusion step,
        ``pumpca -> pump``. Defaults to ``7.255e-5 /ms``.
    TotalPump : array-like or callable, optional
        Areal pump-site density; the conserved sum of ``pump +
        pumpca`` per unit membrane area. Defaults to
        ``1e-9 mol/cm2``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the ``Ci`` species. Defaults to ``None``,
        which falls back to ``cainull``.
    species_initializers : dict or None, optional
        Per-species initializer overrides, keyed by one of this
        class's twenty-three differential species (the fourteen of
        :class:`CdpStC_NoCAM_MA2020_GoC` plus the nine CaM species
        ``CAM0``, ``CAM1C``, ``CAM2C``, ``CAM1N2C``, ``CAM1N``,
        ``CAM2N``, ``CAM2N1C``, ``CAM1C1N``, ``CAM4``). Defaults to
        ``None`` (no overrides); unset buffer/PV species default to
        their steady-state occupancy at ``cainull``/``mginull``,
        ``pump`` defaults to ``TotalPump``, ``CAM0`` defaults to
        ``CAM_start``, and every other CaM species defaults to
        ``0.0 mM``.
    solver : str or None, optional
        Integrator name used for the reaction network. Defaults to
        ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_solver`
        (``"backward_euler"``).
    substeps : int or None, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_substeps`
        (``1``).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``species_initializers`` names a species outside the
        twenty-three listed above, or if ``temp`` is explicitly
        passed as ``None``, or ``substeps`` is less than ``1`` (the
        latter two raised by :meth:`KineticIon._init_kinetic_ion`).
    AttributeError
        Raised during state initialization or reset, or from
        :attr:`parea`/:attr:`dsq`, if this ion's compartment geometry
        (``diam_arc_mean``) has not been attached yet.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    CdpStC_CAMOnly_MA2020_GoC : Sibling decomposition keeping only
        the CaM network this class also includes.
    CdpStC_NoCAM_MA2020_GoC : Sibling decomposition keeping only the
        pump/buffer/PV network this class also includes.
    CdpCAM_MA2024_PC : Purkinje-cell mechanism reusing this class's
        pump/buffer network unchanged and adding Calbindin.
    CdpCR_MA2020_GrC : Granule-cell mechanism reusing this class's
        pump/buffer network unchanged and substituting Calretinin
        for parvalbumin.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all ``Cdp*``
        mechanisms.

    Notes
    -----
    Ported from ``GoC/ion/CdpStC_MA20_GoC.mod``, part of the
    cerebellar Golgi cell model of (Masoli et al., 2020) [4]_; its
    header names Anwar, Hong & De Schutter [1]_ as the reference for
    the mechanism, credits the extended buffer parameters to Schmidt
    et al. (2003) [2]_, and records the pump rate as tuned to data
    from Maeda et al. (1999) [3]_. ``uses_total_current
    = True`` and one :class:`~braincell.ion._base.Source` drives
    ``Ci`` from the channel current supplied at each step:
    ``_ci_source_flux`` returns zero when no current is supplied, and
    otherwise ``total_current * pi * diam_arc_mean / (2 *
    faraday_constant)``. NEURON's raw GoC ``ica`` is efflux-positive,
    but BrainCell channel currents follow the repo-wide inward-positive
    convention, so a positive ``total_current`` here increases ``Ci``.

    One :class:`~braincell.ion._base.Conserve` constrains ``pump +
    pumpca = TotalPump * parea``, with ``pumpca`` recovered
    algebraically rather than integrated; ``pump`` is the only pump
    state among the twenty-three differential species.

    Twenty reactions couple the twenty-four species: the two pump
    steps and the six buffer/PV steps described in
    :class:`CdpStC_NoCAM_MA2020_GoC`, plus twelve reactions forming
    the same two-lobe cooperative calmodulin scheme described in
    :class:`CdpStC_CAMOnly_MA2020_GoC` (``CAM0`` through ``CAM4``,
    reached via either the C-lobe-first or N-lobe-first binding
    order). Buffer- and PV-bound species initialize at the
    equilibrium occupancy implied by their dissociation constants and
    ``cainull``/``mginull``, not at zero.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E.
           (2020). Cerebellar Golgi cell models predict dendritic
           processing and mechanisms of synaptic plasticity. PLOS
           Computational Biology, 16(12), e1007937.
           doi:10.1371/journal.pcbi.1007937
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    # The imported NMODL uses ``COMPARTMENT (1e10)*parea`` because ``pump``
    # and ``pumpca`` are stored visibly in ``mol/cm2`` and NEURON needs an
    # explicit area conversion to reach amount space. BrainCell factors already
    # provide that visible-to-amount mapping, so keeping the extra ``1e10``
    # here would double-apply the pump compartment scaling.
    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
        Factor(
            "cam_unit",
            lambda self: 1.0 * u.um**2,
        ),
    )
    species = _CI_SPECIES + _BUFFER_SPECIES + _PV_SPECIES + _cam_species("cam_unit") + _PUMP_SPECIES
    reactions = CdpStC_NoCAM_MA2020_GoC.reactions + CdpStC_CAMOnly_MA2020_GoC.reactions
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: self._ci_source_flux(total_current),
        ),
    )
    conserves = (
        Conserve(
            species=("pump", "pumpca"),
            algebraic="pumpca",
            total=lambda self, V, x: self.TotalPump * self.parea,
        ),
    )

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(25.0),
        Nannuli: Initializer = 10.9495,
        cainull: Initializer = 45e-6 * u.mM,
        mginull: Initializer = 0.59 * u.mM,
        Buffnull1: Initializer = 0.0 * u.mM,
        rf1: Initializer = 0.0134329 / (u.mM * u.ms),
        rf2: Initializer = 0.0397469 / u.ms,
        Buffnull2: Initializer = 60.9091 * u.mM,
        rf3: Initializer = 0.1435 / (u.mM * u.ms),
        rf4: Initializer = 0.0014 / u.ms,
        BTCnull: Initializer = 0.0 * u.mM,
        b1: Initializer = 5.33 / (u.mM * u.ms),
        b2: Initializer = 0.08 / u.ms,
        DMNPEnull: Initializer = 0.0 * u.mM,
        c1: Initializer = 5.63 / (u.mM * u.ms),
        c2: Initializer = 0.107e-3 / u.ms,
        PVnull: Initializer = 0.08 * u.mM,
        m1: Initializer = 1.07e2 / (u.mM * u.ms),
        m2: Initializer = 9.5e-4 / u.ms,
        p1: Initializer = 0.8 / (u.mM * u.ms),
        p2: Initializer = 2.5e-2 / u.ms,
        CAM_start: Initializer = 0.03 * u.mM,
        K1Coff: Initializer = 0.04 / u.ms,
        K1Con: Initializer = 5.4 / (u.mM * u.ms),
        K2Coff: Initializer = 0.00925 / u.ms,
        K2Con: Initializer = 15.0 / (u.mM * u.ms),
        K1Noff: Initializer = 2.5 / u.ms,
        K1Non: Initializer = 142.5 / (u.mM * u.ms),
        K2Noff: Initializer = 0.75 / u.ms,
        K2Non: Initializer = 175.0 / (u.mM * u.ms),
        kpmp1: Initializer = 3e-3 / (u.mM * u.ms),
        kpmp2: Initializer = 1.75e-5 / u.ms,
        kpmp3: Initializer = 7.255e-5 / u.ms,
        TotalPump: Initializer = 1e-9 * (u.mol / u.cm**2),
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str | None = None,
        substeps: int | None = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.PVnull = braintools.init.param(PVnull, self.varshape, allow_none=False)
        self.m1 = braintools.init.param(m1, self.varshape, allow_none=False)
        self.m2 = braintools.init.param(m2, self.varshape, allow_none=False)
        self.p1 = braintools.init.param(p1, self.varshape, allow_none=False)
        self.p2 = braintools.init.param(p2, self.varshape, allow_none=False)
        self.CAM_start = braintools.init.param(CAM_start, self.varshape, allow_none=False)
        self.K1Coff = braintools.init.param(K1Coff, self.varshape, allow_none=False)
        self.K1Con = braintools.init.param(K1Con, self.varshape, allow_none=False)
        self.K2Coff = braintools.init.param(K2Coff, self.varshape, allow_none=False)
        self.K2Con = braintools.init.param(K2Con, self.varshape, allow_none=False)
        self.K1Noff = braintools.init.param(K1Noff, self.varshape, allow_none=False)
        self.K1Non = braintools.init.param(K1Non, self.varshape, allow_none=False)
        self.K2Noff = braintools.init.param(K2Noff, self.varshape, allow_none=False)
        self.K2Non = braintools.init.param(K2Non, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "PV": self._ss_pv_free(),
            "PV_ca": self._ss_pv_ca(),
            "PV_mg": self._ss_pv_mg(),
            "CAM0": self.CAM_start,
            "CAM1C": 0.0 * u.mM,
            "CAM2C": 0.0 * u.mM,
            "CAM1N2C": 0.0 * u.mM,
            "CAM1N": 0.0 * u.mM,
            "CAM2N": 0.0 * u.mM,
            "CAM2N1C": 0.0 * u.mM,
            "CAM1C1N": 0.0 * u.mM,
            "CAM4": 0.0 * u.mM,
            "pump": self.TotalPump,
        }


@register_ion("CdpCAM_MA2024_PC")
class CdpCAM_MA2024_PC(Calcium, _ParvalbuminEquilibrium, _RadialShellGeometry, KineticIon):
    r"""Purkinje-cell calcium pool: pump, buffers, Calbindin, PV, CaM.

    Extends the Golgi-cell pump/generic-buffer/parvalbumin/calmodulin
    network of :class:`CdpStC_MA2020_GoC` with a four-state Calbindin
    D-28k (CB) cooperative binding scheme. This is the same
    ``CdpStC`` scaffold with the CB subnetwork enabled and both CB
    and CaM species placed in the cytosolic compartment, per the
    imported source tree's "Ion_dyn implementation notes"; the pump,
    generic-buffer, PV and CaM reactions are otherwise unchanged from
    :class:`CdpStC_MA2020_GoC`.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 25
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    Nannuli : array-like or callable, optional
        Radial-shell count inherited from the NEURON multi-shell
        diffusion template; only shapes the single effective volume
        fraction :attr:`vrat`. Defaults to ``10.9495``.
    cainull : array-like or callable, optional
        Baseline/initial free calcium concentration ``Ci``. Defaults
        to ``45e-6 mM``.
    mginull : array-like or callable, optional
        Baseline/initial magnesium concentration ``mg``. Defaults to
        ``0.59 mM``.
    Buffnull1 : array-like or callable, optional
        Total concentration of the first generic buffer, ``Buff1 +
        Buff1_ca``. Defaults to ``0.0 mM``.
    rf1, rf2 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff1``
        binding step. Default ``0.0134329 /(mM*ms)`` and
        ``0.0397469 /ms``.
    Buffnull2 : array-like or callable, optional
        Total concentration of the second generic buffer, ``Buff2 +
        Buff2_ca``. Defaults to ``60.9091 mM``.
    rf3, rf4 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff2``
        binding step. Default ``0.1435 /(mM*ms)`` and ``0.0014 /ms``.
    BTCnull : array-like or callable, optional
        Total concentration of the BTC indicator dye buffer, ``BTC +
        BTC_ca``. Defaults to ``0.0 mM``.
    b1, b2 : array-like or callable, optional
        Forward and backward rate constants of the ``BTC`` binding
        step. Default ``5.33 /(mM*ms)`` and ``0.08 /ms``.
    DMNPEnull : array-like or callable, optional
        Total concentration of the caged-calcium buffer DMNPE,
        ``DMNPE + DMNPE_ca``. Defaults to ``0.0 mM``.
    c1, c2 : array-like or callable, optional
        Forward and backward rate constants of the ``DMNPE`` binding
        step. Default ``5.63 /(mM*ms)`` and ``0.107e-3 /ms``.
    CBnull : array-like or callable, optional
        Total concentration of Calbindin D-28k, ``CB + CB_f_ca +
        CB_ca_s + CB_ca_ca``. Defaults to ``0.16 mM``.
    nf1, nf2 : array-like or callable, optional
        Forward and backward rate constants of Calbindin's fast
        binding sites (used for both the ``CB -> CB_ca_s`` and
        ``CB_f_ca -> CB_ca_ca`` steps). Default ``43.5 /(mM*ms)``
        and ``3.58e-2 /ms``.
    ns1, ns2 : array-like or callable, optional
        Forward and backward rate constants of Calbindin's slow
        binding sites (used for both the ``CB -> CB_f_ca`` and
        ``CB_ca_s -> CB_ca_ca`` steps). Default ``5.5 /(mM*ms)``
        and ``0.26e-2 /ms``.
    PVnull : array-like or callable, optional
        Total concentration of parvalbumin, ``PV + PV_ca + PV_mg``.
        Defaults to ``0.08 mM``.
    m1, m2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV`` calcium
        binding step. Default ``1.07e2 /(mM*ms)`` and
        ``9.5e-4 /ms``.
    p1, p2 : array-like or callable, optional
        Forward and backward rate constants of the ``PV``
        magnesium binding step. Default ``0.8 /(mM*ms)`` and
        ``2.5e-2 /ms``.
    CAM_start : array-like or callable, optional
        Baseline/initial concentration of unbound calmodulin,
        ``CAM0``. Defaults to ``0.03 mM``.
    K1Coff, K1Con : array-like or callable, optional
        Backward and forward rate constants of the first C-lobe
        calcium-binding step. Default ``0.04 /ms`` and
        ``5.4 /(mM*ms)``.
    K2Coff, K2Con : array-like or callable, optional
        Backward and forward rate constants of the second C-lobe
        calcium-binding step. Default ``0.00925 /ms`` and
        ``15.0 /(mM*ms)``.
    K1Noff, K1Non : array-like or callable, optional
        Backward and forward rate constants of the first N-lobe
        calcium-binding step. Default ``2.5 /ms`` and
        ``142.5 /(mM*ms)``.
    K2Noff, K2Non : array-like or callable, optional
        Backward and forward rate constants of the second N-lobe
        calcium-binding step. Default ``0.75 /ms`` and
        ``175.0 /(mM*ms)``.
    kpmp1, kpmp2 : array-like or callable, optional
        Forward and backward rate constants of the ``pump + Ci ->
        pumpca`` binding step. Default ``3e-3 /(mM*ms)`` and
        ``1.75e-5 /ms``.
    kpmp3 : array-like or callable, optional
        Rate constant of the irreversible extrusion step,
        ``pumpca -> pump``. Defaults to ``7.255e-5 /ms``.
    TotalPump : array-like or callable, optional
        Areal pump-site density; the conserved sum of ``pump +
        pumpca`` per unit membrane area. Defaults to
        ``1e-9 mol/cm2``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the ``Ci`` species. Defaults to ``None``,
        which falls back to ``cainull``.
    species_initializers : dict or None, optional
        Per-species initializer overrides, keyed by one of this
        class's twenty-seven differential species: the fourteen of
        :class:`CdpStC_NoCAM_MA2020_GoC`, the four Calbindin species
        (``CB``, ``CB_f_ca``, ``CB_ca_s``, ``CB_ca_ca``), and the
        nine CaM species (``CAM0``, ``CAM1C``, ``CAM2C``,
        ``CAM1N2C``, ``CAM1N``, ``CAM2N``, ``CAM2N1C``, ``CAM1C1N``,
        ``CAM4``). Defaults to ``None`` (no overrides); unset
        buffer/Calbindin/PV species default to their steady-state
        occupancy at ``cainull``/``mginull``, ``pump`` defaults to
        ``TotalPump``, ``CAM0`` defaults to ``CAM_start``, and every
        other CaM species defaults to ``0.0 mM``.
    solver : str or None, optional
        Integrator name used for the reaction network. Defaults to
        ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_solver`
        (``"backward_euler"``).
    substeps : int or None, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_substeps`
        (``1``).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``species_initializers`` names a species outside the
        twenty-seven listed above, or if ``temp`` is explicitly
        passed as ``None``, or ``substeps`` is less than ``1`` (the
        latter two raised by :meth:`KineticIon._init_kinetic_ion`).
    AttributeError
        Raised during state initialization or reset, or from
        :attr:`parea`/:attr:`dsq`, if this ion's compartment geometry
        (``diam_arc_mean``) has not been attached yet.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    CdpStC_MA2020_GoC : Golgi-cell mechanism supplying the pump,
        generic-buffer, PV and CaM reactions this class reuses
        (``sources`` and ``conserves`` are the same tuple objects,
        and several helper methods delegate to it directly).
    CdpCR_MA2020_GrC : Sibling Purkinje-network variant that
        substitutes Calretinin for Calbindin.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all ``Cdp*``
        mechanisms.

    Notes
    -----
    Ported from ``PC/ion/CdpCAM_MA24_PC.mod``, part of the human
    Purkinje cell model of (Masoli et al., 2024) [4]_; its header
    names Anwar, Hong & De Schutter [1]_ as the reference for the
    mechanism, credits the extended buffer parameters to Schmidt
    et al. (2003) [2]_, and records the pump rate as tuned to data
    from Maeda et al. (1999) [3]_. ``uses_total_current =
    True``; ``sources`` and ``conserves`` are the exact tuple objects
    defined on :class:`CdpStC_MA2020_GoC` (same ``Ci``-driving
    :class:`~braincell.ion._base.Source` and same ``pump + pumpca =
    TotalPump * parea`` :class:`~braincell.ion._base.Conserve`). The
    shell geometry and the current-driven ``Ci`` source
    (:attr:`vrat`, :attr:`parea`, :attr:`dsq`, :attr:`dsqvol`,
    ``_require_diam_arc_mean``, ``_ci_source_flux``) are inherited from
    :class:`~braincell.ion._base._RadialShellGeometry`, and the
    parvalbumin equilibrium from :class:`_ParvalbuminEquilibrium`.

    Twenty-four reactions couple the twenty-eight species: the two
    pump steps and four of the six buffer/PV steps from
    :class:`CdpStC_NoCAM_MA2020_GoC` (``Buff1``, ``Buff2``, ``BTC``,
    ``DMNPE``; PV keeps only its two binding steps, unaffected by
    Calbindin), four Calbindin reactions forming its two-site
    fast/slow cooperative scheme (``CB -> CB_ca_s`` via the fast
    rates ``nf1``/``nf2``, ``CB -> CB_f_ca`` via the slow rates
    ``ns1``/``ns2``, then both intermediates converging on
    ``CB_ca_ca`` via the complementary rate pair), and the same
    twelve calmodulin reactions documented in
    :class:`CdpStC_CAMOnly_MA2020_GoC`.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
           Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
           Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz,
           A., & D'Angelo, E. (2024). Human Purkinje cells outperform
           mouse Purkinje cells in dendritic complexity and
           computational capacity. Communications Biology, 7(1), 5.
           doi:10.1038/s42003-023-05689-y
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
    )
    species = (
        _CI_SPECIES
        + _BUFFER_SPECIES
        + (
            Species("CB", init=0.0 * u.mM, factor="cyto"),
            Species("CB_f_ca", init=0.0 * u.mM, factor="cyto"),
            Species("CB_ca_s", init=0.0 * u.mM, factor="cyto"),
            Species("CB_ca_ca", init=0.0 * u.mM, factor="cyto"),
        )
        + _PV_SPECIES
        + _cam_species("cyto")
        + _PUMP_SPECIES
    )
    reactions = (
        _BUFFER_REACTIONS
        + (
            Reaction(
                lhs={"Ci": 1, "CB": 1},
                rhs={"CB_ca_s": 1},
                forward=lambda self, V, x: self.nf1 * self.dsqvol,
                backward=lambda self, V, x: self.nf2 * self.dsqvol,
            ),
            Reaction(
                lhs={"Ci": 1, "CB": 1},
                rhs={"CB_f_ca": 1},
                forward=lambda self, V, x: self.ns1 * self.dsqvol,
                backward=lambda self, V, x: self.ns2 * self.dsqvol,
            ),
            Reaction(
                lhs={"Ci": 1, "CB_f_ca": 1},
                rhs={"CB_ca_ca": 1},
                forward=lambda self, V, x: self.nf1 * self.dsqvol,
                backward=lambda self, V, x: self.nf2 * self.dsqvol,
            ),
            Reaction(
                lhs={"Ci": 1, "CB_ca_s": 1},
                rhs={"CB_ca_ca": 1},
                forward=lambda self, V, x: self.ns1 * self.dsqvol,
                backward=lambda self, V, x: self.ns2 * self.dsqvol,
            ),
        )
        + _PV_REACTIONS
        + CdpStC_CAMOnly_MA2020_GoC.reactions
    )
    sources = CdpStC_MA2020_GoC.sources
    conserves = CdpStC_MA2020_GoC.conserves

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(25.0),
        Nannuli: Initializer = 10.9495,
        cainull: Initializer = 45e-6 * u.mM,
        mginull: Initializer = 0.59 * u.mM,
        Buffnull1: Initializer = 0.0 * u.mM,
        rf1: Initializer = 0.0134329 / (u.mM * u.ms),
        rf2: Initializer = 0.0397469 / u.ms,
        Buffnull2: Initializer = 60.9091 * u.mM,
        rf3: Initializer = 0.1435 / (u.mM * u.ms),
        rf4: Initializer = 0.0014 / u.ms,
        BTCnull: Initializer = 0.0 * u.mM,
        b1: Initializer = 5.33 / (u.mM * u.ms),
        b2: Initializer = 0.08 / u.ms,
        DMNPEnull: Initializer = 0.0 * u.mM,
        c1: Initializer = 5.63 / (u.mM * u.ms),
        c2: Initializer = 0.107e-3 / u.ms,
        CBnull: Initializer = 0.16 * u.mM,
        nf1: Initializer = 43.5 / (u.mM * u.ms),
        nf2: Initializer = 3.58e-2 / u.ms,
        ns1: Initializer = 5.5 / (u.mM * u.ms),
        ns2: Initializer = 0.26e-2 / u.ms,
        PVnull: Initializer = 0.08 * u.mM,
        m1: Initializer = 1.07e2 / (u.mM * u.ms),
        m2: Initializer = 9.5e-4 / u.ms,
        p1: Initializer = 0.8 / (u.mM * u.ms),
        p2: Initializer = 2.5e-2 / u.ms,
        CAM_start: Initializer = 0.03 * u.mM,
        K1Coff: Initializer = 0.04 / u.ms,
        K1Con: Initializer = 5.4 / (u.mM * u.ms),
        K2Coff: Initializer = 0.00925 / u.ms,
        K2Con: Initializer = 15.0 / (u.mM * u.ms),
        K1Noff: Initializer = 2.5 / u.ms,
        K1Non: Initializer = 142.5 / (u.mM * u.ms),
        K2Noff: Initializer = 0.75 / u.ms,
        K2Non: Initializer = 175.0 / (u.mM * u.ms),
        kpmp1: Initializer = 3e-3 / (u.mM * u.ms),
        kpmp2: Initializer = 1.75e-5 / u.ms,
        kpmp3: Initializer = 7.255e-5 / u.ms,
        TotalPump: Initializer = 1e-9 * (u.mol / u.cm**2),
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str | None = None,
        substeps: int | None = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.CBnull = braintools.init.param(CBnull, self.varshape, allow_none=False)
        self.nf1 = braintools.init.param(nf1, self.varshape, allow_none=False)
        self.nf2 = braintools.init.param(nf2, self.varshape, allow_none=False)
        self.ns1 = braintools.init.param(ns1, self.varshape, allow_none=False)
        self.ns2 = braintools.init.param(ns2, self.varshape, allow_none=False)
        self.PVnull = braintools.init.param(PVnull, self.varshape, allow_none=False)
        self.m1 = braintools.init.param(m1, self.varshape, allow_none=False)
        self.m2 = braintools.init.param(m2, self.varshape, allow_none=False)
        self.p1 = braintools.init.param(p1, self.varshape, allow_none=False)
        self.p2 = braintools.init.param(p2, self.varshape, allow_none=False)
        self.CAM_start = braintools.init.param(CAM_start, self.varshape, allow_none=False)
        self.K1Coff = braintools.init.param(K1Coff, self.varshape, allow_none=False)
        self.K1Con = braintools.init.param(K1Con, self.varshape, allow_none=False)
        self.K2Coff = braintools.init.param(K2Coff, self.varshape, allow_none=False)
        self.K2Con = braintools.init.param(K2Con, self.varshape, allow_none=False)
        self.K1Noff = braintools.init.param(K1Noff, self.varshape, allow_none=False)
        self.K1Non = braintools.init.param(K1Non, self.varshape, allow_none=False)
        self.K2Noff = braintools.init.param(K2Noff, self.varshape, allow_none=False)
        self.K2Non = braintools.init.param(K2Non, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "CB": self._ss_cb_free(),
            "CB_f_ca": self._ss_cb_fast(),
            "CB_ca_s": self._ss_cb_slow(),
            "CB_ca_ca": self._ss_cb_ca(),
            "PV": self._ss_pv_free(),
            "PV_ca": self._ss_pv_ca(),
            "PV_mg": self._ss_pv_mg(),
            "CAM0": self.CAM_start,
            "CAM1C": 0.0 * u.mM,
            "CAM2C": 0.0 * u.mM,
            "CAM1N2C": 0.0 * u.mM,
            "CAM1N": 0.0 * u.mM,
            "CAM2N": 0.0 * u.mM,
            "CAM2N1C": 0.0 * u.mM,
            "CAM1C1N": 0.0 * u.mM,
            "CAM4": 0.0 * u.mM,
            "pump": self.TotalPump,
        }

    def _kdf(self):
        return (self.cainull * self.nf1) / self.nf2

    def _kds(self):
        return (self.cainull * self.ns1) / self.ns2

    def _ss_cb_free(self):
        kdf = self._kdf()
        kds = self._kds()
        return self.CBnull / (1.0 + kdf + kds + kdf * kds)

    def _ss_cb_fast(self):
        kdf = self._kdf()
        kds = self._kds()
        return (self.CBnull * kds) / (1.0 + kdf + kds + kdf * kds)

    def _ss_cb_slow(self):
        kdf = self._kdf()
        kds = self._kds()
        return (self.CBnull * kdf) / (1.0 + kdf + kds + kdf * kds)

    def _ss_cb_ca(self):
        kdf = self._kdf()
        kds = self._kds()
        return (self.CBnull * kdf * kds) / (1.0 + kdf + kds + kdf * kds)


@register_ion("CdpCR_MA2020_GrC")
class CdpCR_MA2020_GrC(Calcium, _RadialShellGeometry, KineticIon):
    r"""Granule-cell calcium pool: pump, generic buffers, Calretinin.

    Reuses the pump and generic-buffer (``Buff1``, ``Buff2``, BTC,
    DMNPE) network of :class:`CdpStC_MA2020_GoC`, but replaces its
    parvalbumin and calmodulin branches with a two-site-per-lobe
    cooperative Calretinin (CR) binding scheme plus one separate,
    uncoupled "vestigial" CR site. Parvalbumin is absent from this
    mechanism; Calretinin is the endogenous calcium buffer.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.KineticIon.E`. Defaults to 25
        degrees Celsius, converted to kelvin via ``u.celsius2kelvin``
        before being stored.
    Nannuli : array-like or callable, optional
        Radial-shell count inherited from the NEURON multi-shell
        diffusion template; only shapes the single effective volume
        fraction :attr:`vrat`. Defaults to ``10.9495``.
    cainull : array-like or callable, optional
        Baseline/initial free calcium concentration ``Ci``. Defaults
        to ``45e-6 mM``.
    mginull : array-like or callable, optional
        Baseline/initial magnesium concentration ``mg``. Defaults to
        ``0.59 mM``.
    Buffnull1 : array-like or callable, optional
        Total concentration of the first generic buffer, ``Buff1 +
        Buff1_ca``. Defaults to ``0.0 mM``.
    rf1, rf2 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff1``
        binding step. Default ``0.0134329 /(mM*ms)`` and
        ``0.0397469 /ms``.
    Buffnull2 : array-like or callable, optional
        Total concentration of the second generic buffer, ``Buff2 +
        Buff2_ca``. Defaults to ``60.9091 mM``.
    rf3, rf4 : array-like or callable, optional
        Forward and backward rate constants of the ``Buff2``
        binding step. Default ``0.1435 /(mM*ms)`` and ``0.0014 /ms``.
    BTCnull : array-like or callable, optional
        Total concentration of the BTC indicator dye buffer, ``BTC +
        BTC_ca``. Defaults to ``0.0 mM``.
    b1, b2 : array-like or callable, optional
        Forward and backward rate constants of the ``BTC`` binding
        step. Default ``5.33 /(mM*ms)`` and ``0.08 /ms``.
    DMNPEnull : array-like or callable, optional
        Total concentration of the caged-calcium buffer DMNPE,
        ``DMNPE + DMNPE_ca``. Defaults to ``0.0 mM``.
    c1, c2 : array-like or callable, optional
        Forward and backward rate constants of the ``DMNPE`` binding
        step. Default ``5.63 /(mM*ms)`` and ``0.107e-3 /ms``.
    CRnull : array-like or callable, optional
        Total concentration of unbound Calretinin, the initializer
        for ``CR``. Defaults to ``0.9 mM``.
    nT1, nT2 : array-like or callable, optional
        Forward and backward rate constants of a Calretinin site's
        *first* calcium-binding step, on either lobe. Default
        ``1.8 /(mM*ms)`` and ``0.053 /ms``.
    nR1, nR2 : array-like or callable, optional
        Forward and backward rate constants of a Calretinin site's
        *second*, cooperative calcium-binding step, on either lobe.
        Default ``310.0 /(mM*ms)`` and ``0.02 /ms``.
    nV1, nV2 : array-like or callable, optional
        Forward and backward rate constants of the separate
        vestigial Calretinin site, ``CR -> CR_1V``. Default
        ``7.3 /(mM*ms)`` and ``0.24 /ms``.
    kpmp1, kpmp2 : array-like or callable, optional
        Forward and backward rate constants of the ``pump + Ci ->
        pumpca`` binding step. Default ``3e-3 /(mM*ms)`` and
        ``1.75e-5 /ms``.
    kpmp3 : array-like or callable, optional
        Rate constant of the irreversible extrusion step,
        ``pumpca -> pump``. Defaults to ``7.255e-5 /ms``.
    TotalPump : array-like or callable, optional
        Areal pump-site density; the conserved sum of ``pump +
        pumpca`` per unit membrane area. Defaults to
        ``1e-9 mol/cm2``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`KineticIon._init_kinetic_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the ``Ci`` species. Defaults to ``None``,
        which falls back to ``cainull``.
    species_initializers : dict or None, optional
        Per-species initializer overrides, keyed by one of this
        class's twenty-one differential species: ``Ci``, ``mg``,
        the four generic-buffer species (``Buff1``, ``Buff1_ca``,
        ``Buff2``, ``Buff2_ca``), the BTC and DMNPE pairs, the ten
        Calretinin species (``CR``, ``CR_1C_0N``, ``CR_2C_0N``,
        ``CR_2C_1N``, ``CR_1C_1N``, ``CR_0C_1N``, ``CR_0C_2N``,
        ``CR_1C_2N``, ``CR_2C_2N``, ``CR_1V``), and ``pump``.
        Defaults to ``None`` (no overrides); unset generic-buffer
        species default to their steady-state occupancy at
        ``cainull``, ``CR`` defaults to ``CRnull``, every other
        Calretinin species defaults to ``0.0 mM``, and ``pump``
        defaults to ``TotalPump``.
    solver : str or None, optional
        Integrator name used for the reaction network. Defaults to
        ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_solver`
        (``"backward_euler"``).
    substeps : int or None, optional
        Number of solver substeps run inside one parent update.
        Defaults to ``None``, which falls back to
        :attr:`~braincell.ion._base.KineticIon.default_substeps`
        (``1``).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``species_initializers`` names a species outside the
        twenty-one listed above, or if ``temp`` is explicitly passed
        as ``None``, or ``substeps`` is less than ``1`` (the latter
        two raised by :meth:`KineticIon._init_kinetic_ion`).
    AttributeError
        Raised during state initialization or reset, or from
        :attr:`parea`/:attr:`dsq`, if this ion's compartment geometry
        (``diam_arc_mean``) has not been attached yet.

    See Also
    --------
    Calcium : Base calcium ion family this class attaches the
        reaction network to.
    CdpStC_MA2020_GoC : Golgi-cell mechanism supplying the pump and
        generic-buffer reactions this class reuses (``sources`` and
        ``conserves`` are the same tuple objects, and several helper
        methods delegate to it directly).
    CdpCAM_MA2024_PC : Sibling Purkinje-network variant built on the
        same base but adding Calbindin instead of Calretinin.
    braincell.ion._base.KineticIon : Template this class instantiates;
        documents the NMODL-style semantics shared by all ``Cdp*``
        mechanisms.

    Notes
    -----
    Ported from ``GrC/ion/CdpCR_MA20_GrC.mod``, part of the granule
    cell subtype model of (Masoli et al., 2020) [4]_; its header
    names Anwar, Hong & De Schutter [1]_ as the reference for the
    mechanism, credits the extended buffer parameters to Schmidt
    et al. (2003) [2]_, and records the pump rate as tuned to data
    from Maeda et al. (1999) [3]_. ``uses_total_current
    = True``; ``sources`` and ``conserves`` are the exact tuple
    objects defined on :class:`CdpStC_MA2020_GoC` (same
    ``Ci``-driving :class:`~braincell.ion._base.Source` and same
    ``pump + pumpca = TotalPump * parea``
    :class:`~braincell.ion._base.Conserve`). The shell geometry and
    the current-driven ``Ci`` source (:attr:`vrat`, :attr:`parea`,
    :attr:`dsq`, :attr:`dsqvol`, ``_require_diam_arc_mean``,
    ``_ci_source_flux``) are inherited from
    :class:`~braincell.ion._base._RadialShellGeometry`.

    Nineteen reactions couple the twenty-two species: the two pump
    steps, four generic-buffer steps (``Buff1``, ``Buff2``, ``BTC``,
    ``DMNPE``; unlike :class:`CdpStC_MA2020_GoC` there is no PV
    branch here), thirteen Calretinin reactions, and no calmodulin
    branch at all. The thirteen Calretinin reactions form a
    two-site-per-lobe cooperative lattice: nine states index how many
    of Calretinin's two "C-lobe" sites (0, 1 or 2) and two "N-lobe"
    sites (0, 1 or 2) are calcium-bound (``CR`` is the (0, 0) state,
    ``CR_2C_2N`` the fully bound state), every *first*-site binding
    step on either lobe uses the ``nT1``/``nT2`` rate pair, and every
    *second*, cooperative-site binding step uses the faster
    ``nR1``/``nR2`` pair. A tenth, separate Calretinin species,
    ``CR_1V``, binds calcium directly from ``CR`` via its own
    ``nV1``/``nV2`` rate pair and does not couple further into the
    nine-state lattice.

    References
    ----------
    .. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
           Ca2+-activated K+ channels with models of Ca2+ buffering
           in Purkinje cells. The Cerebellum, 11(3), 681-693.
           doi:10.1007/s12311-010-0224-3
    .. [2] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
           Eilers, J. (2003). Mutational analysis of dendritic Ca2+
           kinetics in rodent Purkinje cells: role of parvalbumin and
           calbindin D28k. The Journal of Physiology, 551(1), 13-32.
           doi:10.1113/jphysiol.2002.035824
    .. [3] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y.,
           & Kasai, H. (1999). Supralinear Ca2+ signaling by
           cooperative and mobile Ca2+ buffering in Purkinje neurons.
           Neuron, 24(4), 989-1002.
           doi:10.1016/S0896-6273(00)81045-4
    .. [4] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
    )
    species = (
        _CI_SPECIES
        + _BUFFER_SPECIES
        + (
            Species("CR", init=0.0 * u.mM, factor="cyto"),
            Species("CR_1C_0N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_2C_0N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_2C_1N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_1C_1N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_0C_1N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_0C_2N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_1C_2N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_2C_2N", init=0.0 * u.mM, factor="cyto"),
            Species("CR_1V", init=0.0 * u.mM, factor="cyto"),
        )
        + _PUMP_SPECIES
    )
    reactions = _BUFFER_REACTIONS + (
        Reaction(
            lhs={"Ci": 1, "CR": 1},
            rhs={"CR_1C_0N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_1C_0N": 1},
            rhs={"CR_2C_0N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_2C_0N": 1},
            rhs={"CR_2C_1N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR": 1},
            rhs={"CR_0C_1N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_0C_1N": 1},
            rhs={"CR_0C_2N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_0C_2N": 1},
            rhs={"CR_1C_2N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_2C_1N": 1},
            rhs={"CR_2C_2N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_1C_2N": 1},
            rhs={"CR_2C_2N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_1C_0N": 1},
            rhs={"CR_1C_1N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_0C_1N": 1},
            rhs={"CR_1C_1N": 1},
            forward=lambda self, V, x: self.nT1 * self.dsqvol,
            backward=lambda self, V, x: self.nT2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_1C_1N": 1},
            rhs={"CR_2C_1N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR_1C_1N": 1},
            rhs={"CR_1C_2N": 1},
            forward=lambda self, V, x: self.nR1 * self.dsqvol,
            backward=lambda self, V, x: self.nR2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CR": 1},
            rhs={"CR_1V": 1},
            forward=lambda self, V, x: self.nV1 * self.dsqvol,
            backward=lambda self, V, x: self.nV2 * self.dsqvol,
        ),
    )
    sources = CdpStC_MA2020_GoC.sources
    conserves = CdpStC_MA2020_GoC.conserves

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(25.0),
        Nannuli: Initializer = 10.9495,
        cainull: Initializer = 45e-6 * u.mM,
        mginull: Initializer = 0.59 * u.mM,
        Buffnull1: Initializer = 0.0 * u.mM,
        rf1: Initializer = 0.0134329 / (u.mM * u.ms),
        rf2: Initializer = 0.0397469 / u.ms,
        Buffnull2: Initializer = 60.9091 * u.mM,
        rf3: Initializer = 0.1435 / (u.mM * u.ms),
        rf4: Initializer = 0.0014 / u.ms,
        BTCnull: Initializer = 0.0 * u.mM,
        b1: Initializer = 5.33 / (u.mM * u.ms),
        b2: Initializer = 0.08 / u.ms,
        DMNPEnull: Initializer = 0.0 * u.mM,
        c1: Initializer = 5.63 / (u.mM * u.ms),
        c2: Initializer = 0.107e-3 / u.ms,
        CRnull: Initializer = 0.9 * u.mM,
        nT1: Initializer = 1.8 / (u.mM * u.ms),
        nT2: Initializer = 0.053 / u.ms,
        nR1: Initializer = 310.0 / (u.mM * u.ms),
        nR2: Initializer = 0.02 / u.ms,
        nV1: Initializer = 7.3 / (u.mM * u.ms),
        nV2: Initializer = 0.24 / u.ms,
        kpmp1: Initializer = 3e-3 / (u.mM * u.ms),
        kpmp2: Initializer = 1.75e-5 / u.ms,
        kpmp3: Initializer = 7.255e-5 / u.ms,
        TotalPump: Initializer = 1e-9 * (u.mol / u.cm**2),
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str | None = None,
        substeps: int | None = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.CRnull = braintools.init.param(CRnull, self.varshape, allow_none=False)
        self.nT1 = braintools.init.param(nT1, self.varshape, allow_none=False)
        self.nT2 = braintools.init.param(nT2, self.varshape, allow_none=False)
        self.nR1 = braintools.init.param(nR1, self.varshape, allow_none=False)
        self.nR2 = braintools.init.param(nR2, self.varshape, allow_none=False)
        self.nV1 = braintools.init.param(nV1, self.varshape, allow_none=False)
        self.nV2 = braintools.init.param(nV2, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "CR": self.CRnull,
            "CR_1C_0N": 0.0 * u.mM,
            "CR_2C_0N": 0.0 * u.mM,
            "CR_2C_1N": 0.0 * u.mM,
            "CR_1C_1N": 0.0 * u.mM,
            "CR_0C_1N": 0.0 * u.mM,
            "CR_0C_2N": 0.0 * u.mM,
            "CR_1C_2N": 0.0 * u.mM,
            "CR_2C_2N": 0.0 * u.mM,
            "CR_1V": 0.0 * u.mM,
            "pump": self.TotalPump,
        }


@register_ion("CdpHVA_SU2015_DCN")
class CdpHVA_SU2015_DCN(Calcium, DynamicNernstIon):
    r"""HVA-current-driven calcium pool for the deep cerebellar nuclei.

    First-order relaxation model for the calcium pool associated
    with high-voltage-activated (HVA) calcium channels in deep
    cerebellar nucleus (DCN) neurons: the current drive from attached
    HVA channels pushes ``Ci`` away from a fixed baseline, and a
    single time constant relaxes it back.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.DynamicNernstIon.E`. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    kCa : array-like or callable, optional
        Current-to-concentration scale factor in :meth:`derivative`.
        Defaults to ``3.45e-7 /coulomb``.
    tauCa : array-like or callable, optional
        Relaxation time constant toward ``caiBase`` in
        :meth:`derivative`. Defaults to ``70.0 ms``.
    caiBase : array-like or callable, optional
        Baseline calcium concentration that ``Ci`` relaxes toward,
        and the default value of ``Ci_initializer`` when none is
        given. Defaults to ``50e-6 mM``.
    depth : array-like or callable, optional
        Shell depth dividing the current-drive term in
        :meth:`derivative`. Defaults to ``0.2 um``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the dynamic ``Ci`` state. Defaults to
        ``None``, which falls back to a constant initializer at
        ``caiBase``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion` requires an
        explicit temperature and does not fall back to a class
        default.

    See Also
    --------
    Calcium : Base calcium ion family this class computes a reversal
        potential for.
    CdpLVA_SU2015_DCN : Sibling DCN calcium pool with an identical
        relaxation model driven by low-voltage-activated current.
    braincell.ion._base.DynamicNernstIon : Mixin this class builds
        on; documents the ``Ci``/``Co``/``E`` interface shared by
        all dynamic calcium ions.

    Notes
    -----
    This class subclasses :class:`~braincell.ion._base.DynamicNernstIon`,
    not :class:`~braincell.ion._base.KineticIon`: it defines a single
    :meth:`derivative` method rather than declarative
    ``Factor``/``Species``/``Reaction`` tuples, unlike the other
    ``Cdp*`` mechanisms in this module.

    Ported from ``DCN/ion/CdpHVA_SU15_DCN.mod``. :meth:`derivative`
    implements

    .. math::

        \frac{dCi}{dt} = -\frac{kCa}{depth} \cdot I_{total} \cdot 10^4
        - \frac{Ci - caiBase}{tauCa}

    NEURON's raw ``ica`` is efflux-positive; BrainCell channel
    currents follow the repo-wide inward-positive convention, so the
    sign of the current-drive term is flipped relative to the
    original NMODL expression to keep a positive ``total_current``
    increasing ``Ci``. When no channels are attached,
    ``total_current`` defaults to zero and :meth:`derivative` reduces
    to pure relaxation toward ``caiBase``. The current-drive term is
    computed from unitless decimals (via ``to_decimal``) rather than
    through ``brainunit`` dimensional arithmetic, matching the
    imported NMODL's ``* 1e4`` unit-conversion literal rather than
    deriving it dimensionally.

    The kinetic constants (``kCa``, ``tauCa``, ``caiBase``, ``depth``)
    are DCN-model parameters; no paper's text can be cited as
    printing this specific relaxation-model form. The kinetics trace
    to the deep cerebellar nucleus model of Steuber et al. (2011)
    [1]_, translated from GENESIS to NEURON by Luthman et al. (2011)
    and used in the NEURON DCN model of Sudhakar et al. (2015) [2]_,
    from which this mechanism is imported.

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

    __module__ = "braincell.ion"
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kCa: Initializer = 3.45e-7 / u.coulomb,
        tauCa: Initializer = 70.0 * u.ms,
        caiBase: Initializer = 50e-6 * u.mM,
        depth: Initializer = 0.2 * u.um,
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        if Ci_initializer is None:
            Ci_initializer = braintools.init.Constant(caiBase)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.tauCa = braintools.init.param(tauCa, self.varshape, allow_none=False)
        self.caiBase = braintools.init.param(caiBase, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        if total_current is None:
            total_current = braintools.init.param(0.0 * (u.mA / u.cm**2), self.varshape)
        # The imported NMODL uses:
        #   cai' = -(kCa / depth) * ica * 1e4 - (cai - caiBase) / tauCa
        # where NEURON raw ``ica`` is negative for inward current. BrainCell
        # channels follow the repo-wide inward-positive current convention, so
        # the equivalent imported-ion drive here is positive in ``total_current``.
        drive_value = (
            u.get_mantissa(self.kCa) / self.depth.to_decimal(u.um) * total_current.to_decimal(u.mA / u.cm**2) * 1e4
        )
        drive = drive_value * (u.mM / u.ms)
        return drive - (Ci - self.caiBase) / self.tauCa


@register_ion("CdpLVA_SU2015_DCN")
class CdpLVA_SU2015_DCN(Calcium, DynamicNernstIon):
    r"""LVA-current-driven calcium pool for the deep cerebellar nuclei.

    First-order relaxation model for the calcium pool associated
    with low-voltage-activated (LVA) calcium channels in deep
    cerebellar nucleus (DCN) neurons: the current drive from attached
    LVA channels pushes ``Ci`` away from a fixed baseline, and a
    single time constant relaxes it back. Structurally identical to
    :class:`CdpHVA_SU2015_DCN`; only the parameter names differ,
    matching the imported NMODL's separate ``cali``/``cal`` pool.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Calcium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation in
        :attr:`~braincell.ion._base.DynamicNernstIon.E`. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    kCal : array-like or callable, optional
        Current-to-concentration scale factor in :meth:`derivative`.
        Defaults to ``3.45e-7 /coulomb``.
    tauCal : array-like or callable, optional
        Relaxation time constant toward ``caliBase`` in
        :meth:`derivative`. Defaults to ``70.0 ms``.
    caliBase : array-like or callable, optional
        Baseline calcium concentration that ``Ci`` relaxes toward,
        and the default value of ``Ci_initializer`` when none is
        given. Defaults to ``50e-6 mM``.
    depth : array-like or callable, optional
        Shell depth dividing the current-drive term in
        :meth:`derivative`. Defaults to ``0.2 um``.
    Co : array-like or callable or None, optional
        Extracellular calcium concentration. Defaults to ``None``,
        which falls back to :attr:`Calcium.default_Co` inside
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion`.
    Ci_initializer : array-like or callable or None, optional
        Initializer for the dynamic ``Ci`` state. Defaults to
        ``None``, which falls back to a constant initializer at
        ``caliBase``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged
        to :class:`Calcium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`DynamicNernstIon._init_dynamic_nernst_ion` requires an
        explicit temperature and does not fall back to a class
        default.

    See Also
    --------
    Calcium : Base calcium ion family this class computes a reversal
        potential for.
    CdpHVA_SU2015_DCN : Sibling DCN calcium pool with an identical
        relaxation model driven by high-voltage-activated current.
    braincell.ion._base.DynamicNernstIon : Mixin this class builds
        on; documents the ``Ci``/``Co``/``E`` interface shared by
        all dynamic calcium ions.

    Notes
    -----
    This class subclasses :class:`~braincell.ion._base.DynamicNernstIon`,
    not :class:`~braincell.ion._base.KineticIon`: it defines a single
    :meth:`derivative` method rather than declarative
    ``Factor``/``Species``/``Reaction`` tuples, unlike the other
    ``Cdp*`` mechanisms in this module. ``Ci`` here corresponds to
    the NMODL ``cali`` pool (as opposed to ``CdpHVA_SU2015_DCN``'s
    ``cai``), exposed through the same standard ``Ci``/``Co``/``E``
    interface.

    Ported from ``DCN/ion/CdpLVA_SU15_DCN.mod``. :meth:`derivative`
    implements

    .. math::

        \frac{dCi}{dt} = -\frac{kCal}{depth} \cdot I_{total} \cdot 10^4
        - \frac{Ci - caliBase}{tauCal}

    NEURON's raw ``ical`` is efflux-positive; BrainCell channel
    currents follow the repo-wide inward-positive convention, so the
    sign of the current-drive term is flipped relative to the
    original NMODL expression to keep a positive ``total_current``
    increasing ``Ci``. When no channels are attached,
    ``total_current`` defaults to zero and :meth:`derivative` reduces
    to pure relaxation toward ``caliBase``. The current-drive term is
    computed from unitless decimals (via ``to_decimal``) rather than
    through ``brainunit`` dimensional arithmetic, matching the
    imported NMODL's ``* 1e4`` unit-conversion literal rather than
    deriving it dimensionally.

    The kinetic constants (``kCal``, ``tauCal``, ``caliBase``,
    ``depth``) are DCN-model parameters; no paper's text can be cited
    as printing this specific relaxation-model form. The kinetics
    trace to the deep cerebellar nucleus model of Steuber et al.
    (2011) [1]_, translated from GENESIS to NEURON by Luthman et al.
    (2011) and used in the NEURON DCN model of Sudhakar et al. (2015)
    [2]_, from which this mechanism is imported.

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

    __module__ = "braincell.ion"
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        kCal: Initializer = 3.45e-7 / u.coulomb,
        tauCal: Initializer = 70.0 * u.ms,
        caliBase: Initializer = 50e-6 * u.mM,
        depth: Initializer = 0.2 * u.um,
        Co: Optional[Initializer] = None,
        Ci_initializer: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        if Ci_initializer is None:
            Ci_initializer = braintools.init.Constant(caliBase)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        self.kCal = braintools.init.param(kCal, self.varshape, allow_none=False)
        self.tauCal = braintools.init.param(tauCal, self.varshape, allow_none=False)
        self.caliBase = braintools.init.param(caliBase, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        if total_current is None:
            total_current = braintools.init.param(0.0 * (u.mA / u.cm**2), self.varshape)
        # The imported NMODL uses:
        #   cali' = -(kCal / depth) * ical * 1e4 - (cali - caliBase) / tauCal
        # where NEURON raw ``ical`` is negative for inward current. BrainCell
        # channel currents use the repo-wide inward-positive convention, so
        # the equivalent imported-ion drive here is positive in ``total_current``.
        drive_value = (
            u.get_mantissa(self.kCal) / self.depth.to_decimal(u.um) * total_current.to_decimal(u.mA / u.cm**2) * 1e4
        )
        drive = drive_value * (u.mM / u.ms)
        return drive - (Ci - self.caliBase) / self.tauCal
