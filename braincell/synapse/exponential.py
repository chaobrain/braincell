# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Exponential-decay conductance synapses.

Both models here are event-driven conductance jumps that relax
exponentially: the NEURON ``expsyn.mod`` single exponential and the
``exp2syn.mod`` difference of two exponentials.
"""

import brainunit as u
import numpy as np

from braincell._base_channel import Synapse
from braincell._base_neuron import HHTypedNeuron
from braincell.mech import (
    ParameterSpec,
    ScalarEventInput,
    StateSpec,
    positive,
    register_synapse,
)

__all__ = [
    'ExpSyn',
    'Exp2Syn',
]


@register_synapse("ExpSyn")
class ExpSyn(Synapse):
    """NEURON-compatible `ExpSyn` template.

    This class follows the dynamics in NEURON's ``expsyn.mod``:

    - state: ``g`` in ``uS``
    - decay: ``g' = -g / tau``
    - step-boundary event: ``g <- g + weighted_pre_drive``
    - inward-positive point current: ``I = g * (e - V_post)``
    """

    root_type = HHTypedNeuron
    parameters = {
        "tau": ParameterSpec(0.1 * u.ms, validator=positive),
        "e": ParameterSpec(0.0 * u.mV),
    }
    states = {"g": StateSpec(0.0 * u.uS)}
    event_input = ScalarEventInput(u.uS, aggregation="sum")

    def apply_events(self, payload, V_post=None):
        _ = V_post
        self.event_input.validate_payload(payload)
        self.g.value = self.g.value + payload

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.g.derivative = -self.g.value / self.tau

    def current(self, V_post):
        return self.g.value * (self.e - V_post)


@register_synapse("Exp2Syn")
class Exp2Syn(Synapse):
    """NEURON-compatible `Exp2Syn` template.

    This class follows the dynamics in NEURON's ``exp2syn.mod``:

    - states: ``A`` and ``B`` in ``uS``
    - decay: ``A' = -A / tau1``, ``B' = -B / tau2``
    - conductance: ``g = B - A``
    - inward-positive point current: ``I = g * (e - V_post)``
    - step-boundary event: ``A <- A + weighted_pre_drive * factor`` and same for ``B``
    """

    root_type = HHTypedNeuron
    parameters = {
        "tau1": ParameterSpec(0.1 * u.ms, validator=positive),
        "tau2": ParameterSpec(10.0 * u.ms, validator=positive),
        "e": ParameterSpec(0.0 * u.mV),
    }
    states = {
        "A": StateSpec(0.0 * u.uS),
        "B": StateSpec(0.0 * u.uS),
    }
    event_input = ScalarEventInput(u.uS, aggregation="sum")

    @classmethod
    def validate_parameter_values(cls, parameters) -> None:
        """Require ``tau1 < tau2`` elementwise.

        Notes
        -----
        The comparison runs in **numpy**, not ``u.math``/``jnp``. Under an
        open JAX trace every jax operation returns a tracer even when all its
        inputs are concrete, so ``bool()`` on a ``u.math.any`` reduction would
        raise ``TracerBoolConversionError`` and make this class impossible to
        construct inside a jitted setup function. numpy stages nothing, and
        it is what the sibling :func:`~braincell.mech.positive` validator
        already uses.

        Both mantissas are taken in ``ms`` rather than compared as
        quantities, so that a caller mixing ``us`` and ``ms`` is ordered on
        physical duration rather than on raw magnitude.
        """
        tau1 = np.asarray(u.Quantity(parameters["tau1"]).to_decimal(u.ms))
        tau2 = np.asarray(u.Quantity(parameters["tau2"]).to_decimal(u.ms))
        if np.any(tau1 >= tau2):
            raise ValueError("Exp2Syn requires tau1 < tau2.")

    def _compute_factor(self):
        tp = (self.tau1 * self.tau2) / (self.tau2 - self.tau1) * u.math.log(self.tau2 / self.tau1)
        factor = -u.math.exp(-(tp / self.tau1)) + u.math.exp(-(tp / self.tau2))
        return 1.0 / factor

    @property
    def g(self):
        return self.B.value - self.A.value

    def apply_events(self, payload, V_post=None):
        _ = V_post
        self.event_input.validate_payload(payload)
        delta = payload * self._compute_factor()
        self.A.value = self.A.value + delta
        self.B.value = self.B.value + delta

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.A.derivative = -self.A.value / self.tau1
        self.B.derivative = -self.B.value / self.tau2

    def current(self, V_post):
        return self.g * (self.e - V_post)
