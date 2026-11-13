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


"""Potassium ion species, plus fixed and Nernst-derived reversal models."""

from typing import Optional

import braintools
import brainunit as u

from braincell._base_ion import Ion
from braincell._constants import FARADAY_CONSTANT
from braincell._typing import Initializer, Size
from braincell.mech import register_ion
from braincell.ion._base import (
    DynamicNernstIon,
    FixedIon,
    InitNernstIon,
    _initial_state_value,
    _shell_influx,
    _total_current_input,
)
from braincell.quad.protocol import state

__all__ = [
    'Potassium',
    'PotassiumDetailed',
    'PotassiumDetailedFH',
    'PotassiumFixed',
    'PotassiumInitNernst',
]


class Potassium(Ion):
    """Base class for modeling the potassium ion species.

    ``Potassium`` collects the physiological defaults shared by every
    concrete potassium ion model in BrainCell and provides the
    ``Ion``/``IonChannel`` container interface potassium channels attach
    to. It carries no dynamics of its own.

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
    PotassiumFixed : Fixed-parameter potassium ion built on this base.
    PotassiumInitNernst : Potassium ion with ``E`` initialized from the
        Nernst equation.

    Notes
    -----
    This is an abstract base class and must be subclassed (for example
    by :class:`PotassiumFixed` or :class:`PotassiumInitNernst`) to
    obtain a concrete potassium ion model with defined reversal-
    potential dynamics.

    Attributes
    ----------
    ion_symbol : str
        Symbol used for runtime family lookup. Set to ``'K'``.
    default_Ci : brainunit.Quantity
        Default intracellular potassium concentration, ``54.4 mM``.
    default_Co : brainunit.Quantity
        Default extracellular potassium concentration, ``2.5 mM``.
    default_valence : int
        Default ionic valence, ``1``.
    """

    __module__ = 'braincell.ion'
    ion_symbol = 'K'
    default_Ci = 54.4 * u.mM
    default_Co = 2.5 * u.mM
    default_valence = 1


@register_ion("PotassiumFixed")
class PotassiumFixed(Potassium, FixedIon):
    r"""Fixed potassium dynamics.

    This potassium model has no dynamics. It holds a fixed reversal
    potential :math:`E` and fixed concentrations :math:`C_i`/:math:`C_o`.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Potassium`.
    E : array-like or callable or None, optional
        Fixed reversal potential. Defaults to ``-95 mV``. Passing
        ``None`` explicitly raises :class:`ValueError`; there is no
        class-default fallback for this argument.
    Ci : array-like or callable or None, optional
        Intracellular potassium concentration. Defaults to ``None``,
        which falls back to :attr:`Potassium.default_Ci` inside
        :meth:`FixedIon._init_fixed_ion`.
    Co : array-like or callable or None, optional
        Extracellular potassium concentration. Defaults to ``None``,
        which falls back to :attr:`Potassium.default_Co` inside
        :meth:`FixedIon._init_fixed_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Potassium.default_valence` inside
        :meth:`FixedIon._init_fixed_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Potassium`.

    Raises
    ------
    ValueError
        If ``E`` is explicitly passed as ``None``.
        :meth:`FixedIon._init_fixed_ion` requires an explicit fixed
        reversal potential and does not fall back to a class default
        for ``E``.

    See Also
    --------
    Potassium : Base potassium ion family this class fixes.
    PotassiumInitNernst : Sibling potassium model whose ``E`` is
        computed from the Nernst equation instead of fixed.

    Notes
    -----
    With the shipped class defaults (``Co = 2.5 mM``, ``Ci = 54.4 mM``,
    ``valence = 1``) at 36 degrees Celsius, the Nernst equation gives
    ``E = -82.05 mV``. ``PotassiumFixed`` instead defaults ``E`` to
    ``-95 mV``, so the two sibling classes disagree by about 13 mV when
    both are constructed with no arguments.
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        E: Optional[Initializer] = -95.0 * u.mV,
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("PotassiumInitNernst")
class PotassiumInitNernst(Potassium, InitNernstIon):
    r"""Fixed ``Ci``/``Co`` potassium model with Nernst-initialized ``E``.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Potassium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    Ci : array-like or callable or None, optional
        Intracellular potassium concentration. Defaults to ``None``,
        which falls back to :attr:`Potassium.default_Ci` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    Co : array-like or callable or None, optional
        Extracellular potassium concentration. Defaults to ``None``,
        which falls back to :attr:`Potassium.default_Co` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Potassium.default_valence` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Potassium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`InitNernstIon._init_nernst_ion` requires an explicit
        temperature and does not fall back to a class default.

    See Also
    --------
    Potassium : Base potassium ion family this class computes a
        reversal potential for.
    PotassiumFixed : Sibling potassium model with a fixed reversal
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
    is

    .. math::

        E = \frac{R \cdot \mathrm{temp}}{\mathrm{valence} \cdot F}
            \log\!\left(\frac{C_o}{C_i}\right)

    where :math:`R` is the gas constant and :math:`F` is the Faraday
    constant. The argument to the logarithm is :math:`C_o / C_i`
    (extracellular over intracellular), and ``valence`` divides inside
    the prefactor rather than appearing as a separate multiplicative
    term.
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


@register_ion("PotassiumDetailed")
class PotassiumDetailed(Potassium, DynamicNernstIon):
    r"""Dynamic intracellular potassium driven by the potassium current, with Nernst ``E``.

    The model of :class:`~braincell.ion.SodiumDetailed` for potassium:
    ``Ci`` is a state, ``Co`` a fixed parameter, and

    .. math::

        \frac{dC_i}{dt} = \frac{I}{z F d} + \beta\,(C_{rest} - C_i),
        \qquad
        E = \frac{RT}{zF}\,\ln\frac{C_o}{C_i},

    with :math:`I` the potassium current density (positive inward, own
    channels plus registered external currents, or the Cell's
    step-start snapshot), :math:`z` the ``valence`` (+1) and :math:`d`
    the volume per unit membrane area. The equation is that of the
    sodium pool of Canavier (1999) [1]_, whose potassium concentrations
    are fixed.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Potassium`.
    temp : array-like or callable, optional
        Absolute temperature of the Nernst equation. Defaults to 36
        degrees Celsius (``309.15 K``).
    d : array-like or callable, optional
        Volume per unit membrane area that the current fills. Defaults
        to ``1.0 um``.
    beta : array-like or callable, optional
        Relaxation rate of ``Ci`` toward ``C_rest``. Defaults to
        ``0.0 /ms`` (no relaxation).
    C_rest : array-like or callable, optional
        Concentration ``Ci`` relaxes toward. Defaults to ``54.4 mM``.
    Co : array-like or callable or None, optional
        Extracellular potassium concentration. Defaults to ``None``,
        which falls back to :attr:`Potassium.default_Co` (``2.5 mM``).
    Ci_initializer : array-like or callable, optional
        Initial ``Ci``. Defaults to a constant ``54.4 mM``
        (:attr:`Potassium.default_Ci`).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Potassium`.

    See Also
    --------
    SodiumDetailed : The same model for sodium.
    braincell.channel.K_Leak : A potassium leak that reads this ion's
        ``E``.

    Notes
    -----
    ``d`` is a shell depth, or the whole compartment's volume / lateral
    area, ``d=lambda ctx: ctx.diam_mid / 4`` for a cylinder. The influx
    is not rectified, so an outward potassium current lowers ``Ci``.
    No literature source fixes the defaults: ``d = 1 um`` is
    :class:`~braincell.ion.CalciumDetailed`'s, and ``54.4 / 2.5 mM`` are the potassium
    family's conventional concentrations. Every value carries its unit;
    a bare number or a wrong unit raises at ``init_state`` in a Cell.

    References
    ----------
    .. [1] Canavier, C. C. (1999). Sodium dynamics underlying burst firing
           and putative mechanisms for the regulation of the firing pattern
           in midbrain dopamine neurons: a computational approach. Journal
           of Computational Neuroscience, 6(1), 49-69.
           doi:10.1023/A:1008809000182
    """

    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        d: Initializer = 1.0 * u.um,
        beta: Initializer = 0.0 / u.ms,
        C_rest: Initializer = 54.4 * u.mM,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = braintools.init.Constant(54.4 * u.mM),
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(Co=Co, temp=temp, Ci_initializer=Ci_initializer)
        self.d = braintools.init.param(d, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.C_rest = braintools.init.param(C_rest, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        return _shell_influx(self, total_current) + self.beta * (self.C_rest - Ci)


@register_ion("PotassiumDetailedFH")
class PotassiumDetailedFH(Potassium, DynamicNernstIon):
    r"""Dynamic intra- and extracellular potassium in a Frankenhaeuser--Hodgkin space.

    :class:`PotassiumDetailed` with the extracellular concentration a
    state too: a thin layer of finite volume outside the membrane, the
    Frankenhaeuser--Hodgkin space, receives the potassium the membrane releases. Both concentrations
    relax to their own resting values,

    .. math::

        \frac{dC_i}{dt} = \frac{I}{z F d} + \beta\,(C_{rest} - C_i),
        \qquad
        \frac{dC_o}{dt} = -\frac{I}{z F d_o} + \beta_o\,(C_{o,rest} - C_o),

    and :math:`E = \frac{RT}{zF}\ln(C_o / C_i)` follows both. :math:`I` is
    the potassium current density (positive inward, own channels plus
    registered external currents, or the Cell's step-start snapshot),
    read once per derivative evaluation for both equations; :math:`z` is
    the ``valence`` (+1), :math:`d` the intracellular and :math:`d_o` the
    extracellular volume per unit membrane area. The space is local to
    each compartment: neighbouring compartments and other cells do not
    share it, and nothing diffuses between them. The equations are those
    of the potassium pool of Botta et al. (2010) [1]_.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to
        :class:`Potassium`.
    temp : array-like or callable, optional
        Absolute temperature of the Nernst equation. Defaults to 36
        degrees Celsius (``309.15 K``).
    d : array-like or callable, optional
        Intracellular volume per unit membrane area. Defaults to
        ``1.0 um``.
    d_o : array-like or callable, optional
        Extracellular (space) volume per unit membrane area. Defaults to
        ``1.0 um``.
    beta : array-like or callable, optional
        Relaxation rate of ``Ci`` toward ``C_rest``. Defaults to
        ``0.0 /ms`` (no relaxation).
    beta_o : array-like or callable, optional
        Relaxation rate of ``Co`` toward ``Co_rest``. Defaults to
        ``0.0 /ms``.
    C_rest : array-like or callable, optional
        Concentration ``Ci`` relaxes toward. Defaults to ``54.4 mM``.
    Co_rest : array-like or callable, optional
        Concentration ``Co`` relaxes toward. Defaults to ``2.5 mM``.
    Ci_initializer : array-like or callable, optional
        Initial ``Ci``. Defaults to a constant ``54.4 mM``
        (:attr:`Potassium.default_Ci`).
    Co_initializer : array-like or callable, optional
        Initial ``Co``. Defaults to a constant ``2.5 mM``
        (:attr:`Potassium.default_Co`).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Potassium`.

    See Also
    --------
    PotassiumDetailed : The same intracellular pool with a fixed ``Co``.
    braincell.channel.K_Leak : A potassium leak that reads this ion's
        ``E``.

    Notes
    -----
    ``Ci`` and ``Co`` are both created by ``init_state`` and restored to
    their initializers by ``reset_state``; record ``Co`` in a Cell with
    ``observe.ion(name).state("Co")``. Before ``init_state``, ``Co``
    holds the family default ``2.5 mM``.

    No constructor parameter is called ``Co``: a :class:`~braincell.Cell`
    replaces every declared constructor parameter with a runtime
    parameter of the same name, which would overwrite the ``Co`` state.
    The extracellular values are therefore ``Co_rest`` and
    ``Co_initializer``.

    Botta et al. write the influx of both potassium concentrations as
    ``ik / (2 FARADAY d)``, the divalent form of a calcium pool, in
    ``K_conc.mod`` (ModelDB 127021). For a monovalent ion that is the
    :math:`z = 1` equation above with twice the depth: the ``.mod`` file's
    ``d = 0.2 um`` is ``d = d_o = 0.4 um`` here. Its ``beta`` (1.3 /ms in
    the ``.mod`` file, 0.075 /ms in the published simulation) is both
    ``beta`` and ``beta_o``, and ``ki0`` / ``ko0`` (140 / 5 mM) are
    ``C_rest`` / ``Co_rest`` and the initial values.

    The influx is not rectified. No literature source fixes the defaults:
    ``d = d_o = 1 um`` is :class:`~braincell.ion.CalciumDetailed`'s,
    and ``54.4 / 2.5 mM`` are the potassium family's conventional
    concentrations. Every value carries its unit.

    References
    ----------
    .. [1] Botta, P., de Souza, F. M. S., Sangrey, T., De Schutter, E., &
           Valenzuela, C. F. (2010). Alcohol excites cerebellar Golgi cells
           by inhibiting the Na+/K+ ATPase. Neuropsychopharmacology, 35(9),
           1984-1996.
           doi:10.1038/npp.2010.76

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.ion import PotassiumDetailedFH
        >>> k = PotassiumDetailedFH(1, d=0.4 * u.um, d_o=0.4 * u.um)
        >>> k.init_state(u.math.full(1, -65.0) * u.mV)
        >>> k.Co.value
        Quantity([2.5], "mM")
    """

    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        d: Initializer = 1.0 * u.um,
        d_o: Initializer = 1.0 * u.um,
        beta: Initializer = 0.0 / u.ms,
        beta_o: Initializer = 0.0 / u.ms,
        C_rest: Initializer = 54.4 * u.mM,
        Co_rest: Initializer = 2.5 * u.mM,
        Ci_initializer: Initializer = braintools.init.Constant(54.4 * u.mM),
        Co_initializer: Initializer = braintools.init.Constant(2.5 * u.mM),
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(temp=temp, Ci_initializer=Ci_initializer)
        self.d = braintools.init.param(d, self.varshape, allow_none=False)
        self.d_o = braintools.init.param(d_o, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.beta_o = braintools.init.param(beta_o, self.varshape, allow_none=False)
        self.C_rest = braintools.init.param(C_rest, self.varshape, allow_none=False)
        self.Co_rest = braintools.init.param(Co_rest, self.varshape, allow_none=False)
        # Public, unlike ``_Ci_initializer``: a Cell replaces the attribute
        # named after each declared parameter, and only ``Ci_initializer``
        # is mapped to a private name.
        self.Co_initializer = Co_initializer

    def _ion_init_state_hook(self, V, batch_size: int = None):
        """Create the ``Ci`` and ``Co`` states from their initializers."""
        super()._ion_init_state_hook(V, batch_size)
        self.Co = state(_initial_state_value(self.Co_initializer, self.varshape, batch_size))

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        """Reset ``Ci`` and ``Co`` to their initializers."""
        super()._ion_reset_state_hook(V, batch_size)
        self.Co.value = _initial_state_value(self.Co_initializer, self.varshape, batch_size)

    def _ion_compute_derivative_hook(self, V):
        """Write ``Ci.derivative`` and ``Co.derivative`` from one current reading."""
        total_current = _total_current_input(self, V)
        self.Ci.derivative = self.derivative(self.Ci.value, V, total_current=total_current)
        self.Co.derivative = self.derivative_Co(self.Co.value, V, total_current=total_current)

    def derivative(self, Ci, V, total_current=None):
        """Return ``dCi/dt = I / (z F d) + beta (C_rest - Ci)``."""
        _ = V
        return _shell_influx(self, total_current) + self.beta * (self.C_rest - Ci)

    def derivative_Co(self, Co, V, total_current=None):
        """Return ``dCo/dt = -I / (z F d_o) + beta_o (Co_rest - Co)``.

        The current that raises ``Ci`` leaves the space outside the
        membrane, hence the minus sign.
        """
        _ = V
        efflux = self._drive_current(total_current) / (self.valence * FARADAY_CONSTANT * self.d_o)
        return -efflux + self.beta_o * (self.Co_rest - Co)
