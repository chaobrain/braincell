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

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel.potassium import Kv1p5_MA2024_PC
from braincell.ion import NonSpecific, Potassium, Sodium
from braincell.mech import register_channel

__all__ = [
    "Kv1p5_MA2020_GrC",
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
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.13195e-3 * (u.siemens / u.cm**2),
        gnonspec: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.siemens / u.cm**2),
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(37.0),
        Tauact: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        Tauinactf: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        Tauinacts: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
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
        """
        conductance = self._voltage_factor(V) * self.conductance_factor(V, K, Na, No)
        ik = self.g_max * conductance * (K.E - V)
        ino = (
            self.gnonspec
            * conductance
            * ((u.gas_constant * self.temp / u.faraday_constant * u.math.log((Na.Co + K.Co) / (Na.Ci + K.Ci))) - V)
        )
        return {"k": ik, "no": ino}
