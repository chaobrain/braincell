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


"""Sodium ion species, plus fixed and Nernst-derived reversal models."""

from typing import Optional

import braintools
import brainunit as u

from braincell._base_ion import Ion
from braincell._typing import Initializer, Size
from braincell.mech import register_ion
from braincell.ion._base import DynamicNernstIon, FixedIon, InitNernstIon, _shell_influx

__all__ = [
    'Sodium',
    'SodiumDetailed',
    'SodiumFixed',
    'SodiumInitNernst',
]


class Sodium(Ion):
    """Base class for modeling the sodium ion species.

    ``Sodium`` collects the physiological defaults shared by every
    concrete sodium ion model in BrainCell and provides the
    ``Ion``/``IonChannel`` container interface sodium channels attach
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
    SodiumFixed : Fixed-parameter sodium ion built on this base.
    SodiumInitNernst : Sodium ion with ``E`` initialized from the
        Nernst equation.

    Notes
    -----
    This is an abstract base class and must be subclassed (for example
    by :class:`SodiumFixed` or :class:`SodiumInitNernst`) to obtain a
    concrete sodium ion model with defined reversal-potential dynamics.

    Attributes
    ----------
    ion_symbol : str
        Symbol used for runtime family lookup. Set to ``'Na'``.
    default_Ci : brainunit.Quantity
        Default intracellular sodium concentration, ``10.0 mM``.
    default_Co : brainunit.Quantity
        Default extracellular sodium concentration, ``140.0 mM``.
    default_valence : int
        Default ionic valence, ``1``.
    """

    __module__ = 'braincell.ion'
    ion_symbol = 'Na'
    default_Ci = 10.0 * u.mM
    default_Co = 140.0 * u.mM
    default_valence = 1


@register_ion("SodiumFixed")
class SodiumFixed(Sodium, FixedIon):
    r"""Fixed Sodium dynamics.

    This sodium model has no dynamics. It holds a fixed reversal
    potential :math:`E` and fixed concentrations :math:`C_i`/:math:`C_o`.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Sodium`.
    E : array-like or callable or None, optional
        Fixed reversal potential. Defaults to ``+50 mV``. Passing
        ``None`` explicitly raises :class:`ValueError`; there is no
        class-default fallback for this argument.
    Ci : array-like or callable or None, optional
        Intracellular sodium concentration. Defaults to ``None``,
        which falls back to :attr:`Sodium.default_Ci` inside
        :meth:`FixedIon._init_fixed_ion`.
    Co : array-like or callable or None, optional
        Extracellular sodium concentration. Defaults to ``None``,
        which falls back to :attr:`Sodium.default_Co` inside
        :meth:`FixedIon._init_fixed_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Sodium.default_valence` inside
        :meth:`FixedIon._init_fixed_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Sodium`.

    Raises
    ------
    ValueError
        If ``E`` is explicitly passed as ``None``.
        :meth:`FixedIon._init_fixed_ion` requires an explicit fixed
        reversal potential and does not fall back to a class default
        for ``E``.

    See Also
    --------
    Sodium : Base sodium ion family this class fixes.
    SodiumInitNernst : Sibling sodium model whose ``E`` is computed
        from the Nernst equation instead of fixed.

    Notes
    -----
    With the shipped class defaults (``Co = 140.0 mM``, ``Ci = 10.0
    mM``, ``valence = 1``) at 36 degrees Celsius, the Nernst equation
    gives ``E = +70.31 mV``. ``SodiumFixed`` instead defaults ``E`` to
    ``+50 mV``, so the two sibling classes disagree by about 20 mV
    when both are constructed with no arguments.
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        E: Optional[Initializer] = 50.0 * u.mV,
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("SodiumInitNernst")
class SodiumInitNernst(Sodium, InitNernstIon):
    r"""Fixed ``Ci``/``Co`` sodium model with ``E`` initialized from Nernst.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Sodium`.
    temp : array-like or callable, optional
        Absolute temperature used by the Nernst equation. Defaults to
        36 degrees Celsius, converted to kelvin via
        ``u.celsius2kelvin`` before being stored.
    Ci : array-like or callable or None, optional
        Intracellular sodium concentration. Defaults to ``None``,
        which falls back to :attr:`Sodium.default_Ci` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    Co : array-like or callable or None, optional
        Extracellular sodium concentration. Defaults to ``None``,
        which falls back to :attr:`Sodium.default_Co` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    valence : array-like or callable or None, optional
        Ionic valence. Defaults to ``None``, which falls back to
        :attr:`Sodium.default_valence` inside
        :meth:`InitNernstIon._init_nernst_ion`.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Sodium`.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.
        :meth:`InitNernstIon._init_nernst_ion` requires an explicit
        temperature and does not fall back to a class default.

    See Also
    --------
    Sodium : Base sodium ion family this class computes a reversal
        potential for.
    SodiumFixed : Sibling sodium model with a fixed reversal
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


@register_ion("SodiumDetailed")
class SodiumDetailed(Sodium, DynamicNernstIon):
    r"""Dynamic intracellular sodium driven by the sodium current, with Nernst ``E``.

    ``Ci`` is a state; ``Co`` is a fixed parameter. The sodium current
    fills a volume ``d`` per unit membrane area, and ``Ci`` relaxes to
    ``C_rest`` at the rate :math:`\beta`:

    .. math::

        \frac{dC_i}{dt} = \frac{I}{z F d} + \beta\,(C_{rest} - C_i),
        \qquad
        E = \frac{RT}{zF}\,\ln\frac{C_o}{C_i}.

    :math:`I` is the sodium current density, positive inward as for
    every braincell channel: the sum of this ion's channels and of the
    currents other channels register for it (``current(V,
    include_external=True)``), or the step-start snapshot a Cell sets.
    :math:`z` is the ion's ``valence`` (+1) and :math:`F` the Faraday
    constant. With the default :math:`\beta = 0` this is pure
    accumulation, the sodium pool of Canavier (1999) [1]_ (ModelDB 6763,
    ``nadifl.mod``) without its longitudinal diffusion.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically the number of
        neurons or compartments. Forwarded unchanged to :class:`Sodium`.
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
        Concentration ``Ci`` relaxes toward. Defaults to ``10.0 mM``.
    Co : array-like or callable or None, optional
        Extracellular sodium concentration. Defaults to ``None``, which
        falls back to :attr:`Sodium.default_Co` (``140.0 mM``).
    Ci_initializer : array-like or callable, optional
        Initial ``Ci``. Defaults to a constant ``10.0 mM``
        (:attr:`Sodium.default_Ci`).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion, forwarded unchanged to
        :class:`Sodium`.

    See Also
    --------
    PotassiumDetailed : The same model for potassium.
    CalciumDetailed : The calcium shell, which rectifies its influx.
    braincell.channel.Na_Leak : A sodium leak that reads this ion's
        ``E``.

    Notes
    -----
    ``d`` has two readings. For a thin shell under the membrane it is
    the shell depth. For the whole volume of a cylindrical
    compartment it is volume / lateral area = diam / 4; in a Cell, pass
    it as a spatial callable, ``d=lambda ctx: ctx.diam_mid / 4``, which
    is the volume factor of ``nadifl.mod`` (``COMPARTMENT
    PI*diam*diam/4`` against the source ``PI*diam``).

    The influx is not rectified, unlike :class:`~braincell.ion.CalciumDetailed`'s: an
    outward sodium current (a Na+/K+ pump's) must be able to lower
    ``Ci``. The valence is the ion's, not a literal, so the same term
    has the right sign for an anion.

    No literature source fixes the defaults: ``d = 1 um`` is
    :class:`~braincell.ion.CalciumDetailed`'s, and ``C_rest``, ``Ci_initializer`` and
    ``Co`` are the sodium family's conventional concentrations. Canavier
    (1999) [1]_ uses the whole-compartment volume, ``nai`` = 4 mM and
    ``nao`` = 145 mM at 35 degC. Every value carries its unit; a bare
    number or a wrong unit raises at ``init_state`` in a Cell.

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
        C_rest: Initializer = 10.0 * u.mM,
        Co: Optional[Initializer] = None,
        Ci_initializer: Initializer = braintools.init.Constant(10.0 * u.mM),
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
