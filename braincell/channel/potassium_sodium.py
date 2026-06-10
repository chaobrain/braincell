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
    """Granule-cell Kv1.5 channel with two current owners.

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
        Maximum potassium conductance. This is the BrainCell name for
        the NEURON ``gKur`` parameter.
    gnonspec : array-like or callable, optional
        Maximum nonspecific cation conductance for the ``ino``
        component.
    temp : array-like, optional
        Absolute temperature used by the q10 and nonspecific reversal
        expressions.
    Tauact : array-like or callable, optional
        Activation time-scale multiplier.
    Tauinactf : array-like or callable, optional
        Fast inactivation time-scale multiplier.
    Tauinacts : array-like or callable, optional
        Slow inactivation time-scale multiplier.
    name : str, optional
        Optional module name.

    Notes
    -----
    BrainCell keeps ``current(...)`` as the total membrane-current API.
    The special multi-owner case is exposed through
    :meth:`current_components`, which returns ``{"k": ik, "no": ino}``.
    Sodium is a read-only concentration dependency for the nonspecific
    reversal expression and is not a current owner. ``No`` is a
    placeholder ion that receives the nonspecific current contribution.

    BrainCell channel currents use the package convention
    ``conductance * (E - V)``. The quoted NEURON source assigns
    ``(v - E)`` currents, so this implementation uses the sign
    convention already used by the surrounding BrainCell channel
    catalogue.
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
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.13195e-3 * (u.siemens / u.cm ** 2),
        gnonspec: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * (u.siemens / u.cm ** 2),
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
            * (
                (
                    u.gas_constant
                    * self.temp
                    / u.faraday_constant
                    * u.math.log((Na.Co + K.Co) / (Na.Ci + K.Ci))
                )
                - V
            )
        )
        return {"k": ik, "no": ino}

    def f_m_inf(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the steady-state activation gate value.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information. The inherited gate expression
            only depends on voltage.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Steady-state value for the ``m`` activation gate.

        Notes
        -----
        The sodium and nonspecific arguments are intentionally unused:
        this special channel reads them for current routing and
        nonspecific reversal, while the inherited Kv1.5 gate kinetics
        remain potassium-channel kinetics.
        """
        return super().f_m_inf(V, K)

    def f_m_tau(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the activation gate time constant.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information. The inherited gate expression
            only depends on voltage.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Time constant for the ``m`` activation gate.

        Notes
        -----
        The extra ion arguments keep the method compatible with the
        ``JointTypes[Potassium, Sodium, NonSpecific]`` channel binding.
        """
        return super().f_m_tau(V, K)

    def f_n_inf(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the steady-state fast-inactivation gate value.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information. The inherited gate expression
            only depends on voltage.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Steady-state value for the ``n`` inactivation gate.

        Notes
        -----
        The sodium and nonspecific arguments are intentionally unused by
        the gate equation.
        """
        return super().f_n_inf(V, K)

    def f_n_tau(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the fast-inactivation gate time constant.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information. The inherited gate expression
            only depends on voltage.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Time constant for the ``n`` inactivation gate.

        Notes
        -----
        The extra ion arguments keep the method compatible with the
        multi-ion channel root type.
        """
        return super().f_n_tau(V, K)

    def f_u_inf(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the steady-state slow-inactivation gate value.

        Parameters
        ----------
        V : array-like
            Membrane potential.
        K : IonInfo
            Potassium ion information. The inherited gate expression
            only depends on voltage.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Steady-state value for the ``u`` inactivation gate.

        Notes
        -----
        The sodium and nonspecific arguments are intentionally unused by
        the gate equation.
        """
        return super().f_u_inf(V, K)

    def f_u_tau(self, V, K: IonInfo, Na: IonInfo, No: IonInfo):
        """Return the slow-inactivation gate time constant.

        Parameters
        ----------
        V : array-like
            Membrane potential. Accepted for the standard gate
            signature.
        K : IonInfo
            Potassium ion information. Accepted for the standard gate
            signature.
        Na : IonInfo
            Sodium ion information. Accepted for joint-ion signature
            compatibility.
        No : IonInfo
            Nonspecific ion placeholder. Accepted for joint-ion
            signature compatibility.

        Returns
        -------
        array-like
            Time constant for the ``u`` inactivation gate.

        Notes
        -----
        The inherited slow gate has a voltage-independent time constant;
        all ion arguments are accepted only to match the multi-ion
        channel binding signature.
        """
        return super().f_u_tau(V, K)
