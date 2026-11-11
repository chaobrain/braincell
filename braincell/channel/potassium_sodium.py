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


"""Channels with coupled potassium, sodium, and nonspecific current paths."""

from typing import Optional

import brainstate
import braintools
import brainunit as u

from braincell._base_channel import Channel, IonInfo
from braincell._typing import ArrayLike, Initializer, Size
from braincell.channel.potassium import Kv1p5_MA2024_PC
from braincell.ion import NonSpecific, Potassium, Sodium
from braincell.mech import register_channel

__all__ = [
    "Kv1p5_MA2020_GrC",
    "NaKPump_Ca1999",
]


@register_channel("Kv1p5_MA2020_GrC")
class Kv1p5_MA2020_GrC(Kv1p5_MA2024_PC):
    r"""Granule-cell Kv1.5 channel with two current owners.

    This channel imports the two-current form of the NEURON mechanism
    ``Kv1p5_MA20_GrC.mod``. The relevant source lines are quoted here
    because the mechanism depends on potassium and sodium concentrations
    while writing two separate current variables::

        "USEION k READ ek,ki,ko WRITE ik"
        "USEION na READ nai,nao"
        "USEION no WRITE ino VALENCE 1: nonspecific cation current"
        "ik = gKur*(0.1 + 1/(1 + exp(-(v - 15)/13)))*m*m*m*n*u*(v - ek)"
        "ino=gnonspec*(0.1 + 1/(1 + exp(-(v - 15)/13)))*m*m*m*n*u*(v - z*log((nao+ko)/(nai+ki)))"

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    g_max : array-like or callable, optional
        Maximum potassium conductance, default ``0.13195e-3 S/cm2``.
        This is the BrainCell name for the NEURON ``gKur`` parameter.
    gnonspec : array-like or callable, optional
        Maximum nonspecific cation conductance for the ``ino``
        component, default ``0.0 S/cm2``.
    temp : array-like, optional
        Absolute temperature used by the q10 and nonspecific reversal
        expressions, default 37 degrees Celsius.
    Tauact : array-like or callable, optional
        Activation time-scale multiplier, default ``1.0``
        (dimensionless).
    Tauinactf : array-like or callable, optional
        Fast inactivation time-scale multiplier, default ``1.0``
        (dimensionless).
    Tauinacts : array-like or callable, optional
        Slow inactivation time-scale multiplier, default ``1.0``
        (dimensionless).
    name : str, optional
        Optional module name.

    See Also
    --------
    Kv1p5_MA2024_PC : Purkinje-cell parent class that supplies the
        inherited gate kinetics.

    Notes
    -----
    BrainCell keeps ``current(...)`` as the total membrane-current API.
    The special multi-owner case is exposed through
    :meth:`current_components`, which returns ``{"k": ik, "no": ino}``.
    Sodium is a read-only concentration dependency for the nonspecific
    reversal expression and is not a current owner. ``No`` is a
    placeholder ion that receives the nonspecific current contribution.

    The inherited Kv1.5 gate kinetics are potassium kinetics and are
    reused verbatim from :class:`Kv1p5_MA2024_PC`. Gate methods declare
    only the ion arguments they read, so the inherited
    ``f_m_inf(self, V, K)`` and friends bind to potassium alone even
    though this channel is rooted on three ions. The parent declares
    three gates -- ``m`` (power 3), ``n`` (power 1) and ``u`` (power
    1) -- so the shared gate product used by both current components
    is

    .. math::

        m^3 \, n \, u

    and both components are further scaled by the shared voltage
    factor computed in ``_voltage_factor``:

    .. math::

        0.1 + \frac{1}{1 + \exp\left(-\dfrac{V - 15\ \text{mV}}
        {13\ \text{mV}}\right)}

    Temperature scaling is *not* attached through the gate objects:
    the parent's ``gates`` tuple sets neither ``phi`` nor ``q10``, so
    :meth:`HH.gate_phi` resolves to the default ``1.0`` for ``m``,
    ``n`` and ``u`` alike. Instead the parent's own ``_q10`` method
    computes ``2.2 ** ((temp_K - 310.15) / 10)`` (``310.15 K`` is 37
    degrees Celsius) and multiplies it into the ``alpha``/``beta``
    rates used inside ``f_m_tau`` and ``f_n_tau`` only; ``f_u_tau``
    returns the constant ``6800 * Tauinacts`` and receives no q10
    scaling at all. This is the mechanism's own code path, not a
    general BrainCell convention, and it is reproduced here rather
    than any closed-form temperature dependence printed in the cited
    papers.

    BrainCell channel currents use the package convention
    ``conductance * (E - V)``. The quoted NEURON source assigns
    ``(v - E)`` currents, so this implementation uses the sign
    convention already used by the surrounding BrainCell channel
    catalogue.

    This mechanism is a **cardiac** IKur channel, not a cerebellar
    one: the ``.mod`` file ``Kv1p5_MA20_GrC.mod`` carries the ``TITLE``
    "Cardiac IKur current & nonspec cation current with identical
    kinetics", and its kinetics were fitted to human atrial myocyte
    recordings by Feng et al. (1998) [1]_, not to any cerebellar
    recording. The granule-cell citation [2]_ names the model
    BrainCell imported this parameterisation from -- the ``MA2020``
    granule-cell deposit -- not the origin of the kinetics.

    References
    ----------
    .. [1] Feng, J., Xu, D., Wang, Z., & Nattel, S. (1998). Ultrarapid
           delayed rectifier current inactivation in human atrial
           myocytes: properties and consequences. American Journal of
           Physiology-Heart and Circulatory Physiology, 275(5),
           H1717-H1725.
           doi:10.1152/ajpheart.1998.275.5.H1717
    .. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
           D'Angelo, E. (2020). Parameter tuning differentiates
           granule cell subtypes enriching transmission properties
           at the cerebellum input stage. Communications Biology,
           3(1), 222.
           doi:10.1038/s42003-020-0953-x
    """

    __module__ = "braincell.channel"
    root_type = brainstate.mixin.JointTypes[Potassium, Sodium, NonSpecific]
    current_owner_types = {
        "k": Potassium,
        "no": NonSpecific,
    }

    def __init__(
        self,
        size: Size,
        g_max: Initializer = 0.13195e-3 * (u.siemens / u.cm**2),
        gnonspec: Initializer = 0.0 * (u.siemens / u.cm**2),
        temp: ArrayLike = u.celsius2kelvin(37.0),
        Tauact: Initializer = 1.0,
        Tauinactf: Initializer = 1.0,
        Tauinacts: Initializer = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            size=size,
            g_max=g_max,
            temp=temp,
            Tauact=Tauact,
            Tauinactf=Tauinactf,
            Tauinacts=Tauinacts,
            name=name,
        )
        self.gnonspec = braintools.init.param(gnonspec, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return total Kv1.5 membrane current.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information used by the ``ik`` component.
        Na : IonInfo
            Sodium ion information read by the nonspecific reversal
            expression.
        No : IonInfo
            Nonspecific current-owner placeholder information.

        Returns
        -------
        array-like
            Sum of the potassium ``ik`` and nonspecific ``ino``
            components.

        Notes
        -----
        This method remains the value consumed by the membrane voltage
        solver. Owner-specific ion totals use
        :meth:`current_components` instead.
        """
        components = self.current_components(V, K, Na, No)
        return components["k"] + components["no"]

    def current_components(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return owner-specific Kv1.5 current components.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information supplying ``E``, ``Ci``, and
            ``Co``.
        Na : IonInfo
            Sodium ion information supplying ``Ci`` and ``Co`` for the
            nonspecific reversal expression.
        No : IonInfo
            Nonspecific current-owner placeholder. It is accepted for
            root-type compatibility and ownership but does not enter the
            NEURON formula directly.

        Returns
        -------
        dict
            Mapping ``"k"`` to the potassium component and ``"no"`` to
            the nonspecific cation component.

        Notes
        -----
        The NEURON source computes::

            "ik = gKur*(0.1 + 1/(1 + exp(-(v - 15)/13)))*m*m*m*n*u*(v - ek)"
            "ino=gnonspec*(0.1 + 1/(1 + exp(-(v - 15)/13)))*m*m*m*n*u*(v - z*log((nao+ko)/(nai+ki)))"

        BrainCell uses ``(E - V)`` current signs. The nonspecific
        reversal expression is therefore expanded directly in this
        method as ``z * log((nao + ko) / (nai + ki))`` and used in
        ``gnonspec * gates * (E_no - V)``.

        The source computes ``z = (R*(celsius + 273.15))/F`` with ``R``
        in joule/degC and ``F`` in coulombs, which is a number in volts
        (about 0.027 at 37 degrees Celsius); ``v - z*log(...)`` then
        uses it as millivolts. ``E_no`` therefore sits 1000 times closer
        to 0 mV than the Nernst-like value ``R*T/F*log(...)``.
        This method reproduces the source, so a nonzero ``gnonspec``
        matches NEURON.
        """
        conductance = self._voltage_factor(V) * self.conductance_factor(V, K, Na, No)
        ik = self.g_max * conductance * (K.E - V)
        # ``z = (R*(celsius+273.15))/F`` with ``R`` in joule/degC and ``F`` in
        # coulombs is a number in volts; ``v - z*log(...)`` then reads it as
        # millivolts. Reproduce that, not the physical Nernst value.
        z = (u.gas_constant * self.temp / u.faraday_constant).to_decimal(u.volt)
        e_no = z * u.math.log((Na.Co + K.Co) / (Na.Ci + K.Ci)) * u.mV
        ino = self.gnonspec * conductance * (e_no - V)
        return {"k": ik, "no": ino}


@register_channel("NaKPump_Ca1999")
class NaKPump_Ca1999(Channel):
    r"""Na+/K+ pump of Canavier (1999): three sodium out, two potassium in.

    The pump current depends on the intracellular sodium concentration
    only:

    .. math::

        I_p = \frac{i_{max}}{1 + \left(k_m / [\mathrm{Na}^+]_i\right)^n}

    ``I_p`` is the net outward current density. Each cycle moves three
    sodium ions out and two potassium ions in, so the sodium ion receives
    ``-3 I_p`` and the potassium ion ``+2 I_p`` (BrainCell's sign
    convention, positive inward). The membrane receives their sum,
    ``-I_p``: the pump hyperpolarizes. With a dynamic sodium ion such as
    :class:`~braincell.ion.SodiumDetailed` the pump lowers ``[Na+]i``;
    with :class:`~braincell.ion.PotassiumDetailed` it raises ``[K+]i``.
    With fixed ions (``SodiumFixed``, ``PotassiumFixed`` or the
    ``*InitNernst`` classes) it contributes the membrane current only.

    Parameters
    ----------
    size : brainstate.typing.Size
        Channel state shape.
    imax : array-like or callable, optional
        Maximum pump current density, the ``ipumpmax`` of ``pump.mod``,
        default ``0.04 mA/cm2``.
    km : array-like or callable, optional
        Intracellular sodium concentration of half-maximal pumping,
        default ``10 mM``.
    n : array-like or callable, optional
        Hill exponent, default ``1.5`` (dimensionless).
    name : str, optional
        Optional channel name.

    See Also
    --------
    braincell.ion.SodiumDetailed : Dynamic sodium ion the pump depletes.
    braincell.ion.PotassiumDetailed : Dynamic potassium ion the pump fills.
    braincell.channel.Na_Leak : Sodium leak that reads the dynamic
        ``E_Na``.

    Notes
    -----
    The class reproduces ``pump.mod`` of Canavier's midbrain dopamine
    neuron model (ModelDB 6763) [1]_, whose ``BREAKPOINT`` reads::

        "inapump = ipumpmax*(1/(1 + pow(km/nai,n)))"
        "ina = 3.0*inapump"
        "ik = -2.0*inapump"

    NEURON's currents are positive outward, so ``ina = 3 I_p`` is
    BrainCell's ``-3 I_p``. The mechanism has no gating state and does
    not depend on the membrane potential, the temperature, ``[Na+]o`` or
    any potassium concentration.

    The defaults are the ``PARAMETER`` values of ``pump.mod``. The
    published simulation (``seg1.nrn``) sets ``ipumpmax = 0.012 mA/cm2``
    in the soma and in every dendrite; pass ``imax=0.012 * u.mA /
    u.cm**2`` to reproduce it.

    ``root_type`` joins :class:`~braincell.ion.Sodium` and
    :class:`~braincell.ion.Potassium`, so a Cell needs both ions where the
    pump is painted. ``current_owner_types`` names both as current owners:
    :meth:`current_components` returns ``{"na": -3 I_p, "k": 2 I_p}`` and
    each ion receives its own component, while :meth:`current` returns
    their sum for the membrane.

    References
    ----------
    .. [1] Canavier, C. C. (1999). Sodium dynamics underlying burst firing
           and putative mechanisms for the regulation of the firing pattern
           in midbrain dopamine neurons: a computational approach. Journal
           of Computational Neuroscience, 6(1), 49-69.
           doi:10.1023/A:1008809000182

    Examples
    --------
    >>> import brainunit as u
    >>> import jax.numpy as jnp
    >>> from braincell import IonInfo
    >>> from braincell.channel import NaKPump_Ca1999
    >>> pump = NaKPump_Ca1999(1, imax=0.012 * u.mA / u.cm**2)
    >>> na = IonInfo(Ci=jnp.full(1, 10.0) * u.mM, Co=jnp.full(1, 145.0) * u.mM,
    ...              E=jnp.full(1, 70.0) * u.mV, valence=1)
    >>> k = IonInfo(Ci=jnp.full(1, 140.0) * u.mM, Co=jnp.full(1, 5.0) * u.mM,
    ...             E=jnp.full(1, -90.0) * u.mV, valence=1)
    >>> V = jnp.full(1, -60.0) * u.mV
    >>> parts = pump.current_components(V, na, k)  # Ci = km: I_p = imax / 2
    >>> print(parts["na"].to_decimal(u.mA / u.cm**2), parts["k"].to_decimal(u.mA / u.cm**2))
    [-0.018] [0.012]
    """

    __module__ = "braincell.channel"
    root_type = brainstate.mixin.JointTypes[Sodium, Potassium]
    current_owner_types = {
        "na": Sodium,
        "k": Potassium,
    }

    def __init__(
        self,
        size: Size,
        imax: Initializer = 0.04 * (u.mA / u.cm**2),
        km: Initializer = 10.0 * u.mM,
        n: Initializer = 1.5,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.imax = braintools.init.param(imax, self.varshape, allow_none=False)
        self.km = braintools.init.param(km, self.varshape, allow_none=False)
        self.n = braintools.init.param(n, self.varshape, allow_none=False)

    def compute_derivative(self, V, Na: IonInfo, K: IonInfo):
        """Do nothing: the pump has no state to integrate.

        Parameters
        ----------
        V : array-like
            Membrane potential (unused).
        Na : IonInfo
            Sodium ion information (unused).
        K : IonInfo
            Potassium ion information (unused).
        """

    def pump_current(self, Na: IonInfo):
        """Return the net outward pump current density ``I_p``.

        Parameters
        ----------
        Na : IonInfo
            Sodium ion information supplying ``Ci``.

        Returns
        -------
        array-like
            ``imax / (1 + (km / Ci) ** n)``, a current density, positive
            outward.
        """
        ratio = u.math.power(self.km / Na.Ci, self.n)
        return self.imax / (1.0 + ratio)

    def current_components(self, V, Na: IonInfo, K: IonInfo):
        """Return the sodium and potassium components of the pump current.

        Parameters
        ----------
        V : array-like
            Membrane potential (unused: the pump is voltage independent).
        Na : IonInfo
            Sodium ion information supplying ``Ci``.
        K : IonInfo
            Potassium ion information (unused, required by ``root_type``).

        Returns
        -------
        dict
            ``{"na": -3 * I_p, "k": 2 * I_p}``, positive inward.
        """
        i_p = self.pump_current(Na)
        return {"na": -3.0 * i_p, "k": 2.0 * i_p}

    def current(self, V, Na: IonInfo, K: IonInfo):
        """Return the net membrane current of the pump, ``-I_p``.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        Na : IonInfo
            Sodium ion information supplying ``Ci``.
        K : IonInfo
            Potassium ion information.

        Returns
        -------
        array-like
            Sum of the two :meth:`current_components`, positive inward.
        """
        components = self.current_components(V, Na, K)
        return components["na"] + components["k"]
