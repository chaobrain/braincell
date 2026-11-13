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
from braincell._typing import Initializer, Size
from braincell.mech import (
    ScalarEventInput,
    StateSpec,
    positive,
    register_synapse,
)

__all__ = [
    'ExpSyn',
    'Exp2Syn',
]

#: Above this ``tau1 / tau2``, tau1 runs as this times tau2, as in the INITIAL
#: block of NEURON's ``exp2syn.mod`` (and ``gabaA_Cl.mod``).
_TAU_RATIO_LIMIT = 0.9999


def _effective_tau1(tau1, tau2):
    """Return the rise time constant the dynamics use: ``tau1``, or ``0.9999 * tau2`` where ``tau1 / tau2 > 0.9999``.

    NEURON's ``exp2syn.mod`` sets ``tau1 = 0.9999*tau2`` in INITIAL when
    ``tau1/tau2 > 0.9999``. Python and numpy scalars keep their own
    arithmetic; arrays and tracers go through ``where``, which passes an
    unclamped ``tau1`` through bit for bit.
    """
    clamped = tau1 / tau2 > _TAU_RATIO_LIMIT
    if isinstance(clamped, (bool, np.bool_)):
        return _TAU_RATIO_LIMIT * tau2 if clamped else tau1
    return u.math.where(clamped, _TAU_RATIO_LIMIT * tau2, tau1)


def _peak_factor(tau1, tau2):
    """Return ``1 / max_t (exp(-t / tau2) - exp(-t / tau1))``, the expression of ``exp2syn.mod``."""
    tp = (tau1 * tau2) / (tau2 - tau1) * u.math.log(tau2 / tau1)
    factor = -u.math.exp(-(tp / tau1)) + u.math.exp(-(tp / tau2))
    return 1.0 / factor


@register_synapse("ExpSyn")
class ExpSyn(Synapse):
    """NEURON-compatible `ExpSyn` template.

    This class follows the dynamics in NEURON's ``expsyn.mod``:

    - state: ``g`` in ``uS``
    - decay: ``g' = -g / tau``
    - step-boundary event: ``g <- g + weighted_pre_drive``
    - inward-positive point current: ``I = g * (e - V_post)``

    Parameters
    ----------
    size : brainstate.typing.Size
        Packed synapse shape.
    name : str or None, optional
        Runtime node name.
    tau : brainstate.typing.ArrayLike or callable, optional
        Positive decay time constant, default 0.1 ms.
    e : brainstate.typing.ArrayLike or callable, optional
        Reversal voltage, default 0 mV.
    """

    root_type = HHTypedNeuron
    states = {"g": StateSpec(0.0 * u.uS)}
    event_input = ScalarEventInput(u.uS, aggregation="sum")

    def __init__(self, size: Size, name: str | None = None, tau: Initializer = 0.1 * u.ms, e: Initializer = 0.0 * u.mV):
        super().__init__(size=size, name=name)
        self._init_parameters(tau=tau, e=e)

    @classmethod
    def validate_parameter_values(cls, parameters):
        """Require a positive decay time constant."""
        positive(parameters["tau"], "tau")

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

    Parameters
    ----------
    size : brainstate.typing.Size
        Packed synapse shape.
    name : str or None, optional
        Runtime node name.
    tau1 : brainstate.typing.ArrayLike or callable, optional
        Positive rise time constant, default 0.1 ms. Where ``tau1 / tau2 >
        0.9999`` the dynamics use ``0.9999 * tau2`` (see Notes); the declared
        value is kept.
    tau2 : brainstate.typing.ArrayLike or callable, optional
        Decay time constant greater than tau1, default 10 ms.
    e : brainstate.typing.ArrayLike or callable, optional
        Reversal voltage, default 0 mV.

    Notes
    -----
    ``factor = 1 / max_t (exp(-t / tau2) - exp(-t / tau1))`` makes an event of
    weight ``w`` peak at ``g = w``. As ``tau1`` approaches ``tau2`` the
    expression loses about ``eps / delta`` of its precision (``delta = 1 -
    tau1 / tau2``), and ``g = B - A`` subtracts two states of about ``e w /
    delta``. Like the INITIAL block of ``exp2syn.mod``, ``tau1 / tau2 >
    0.9999`` therefore runs as ``tau1 = 0.9999 * tau2``, in the factor and in
    ``A' = -A / tau1``: one event's conductance is then the ``delta = 1e-4``
    kernel, within ``4.2e-5 w`` of the requested one, and the gradient with
    respect to ``tau1`` is zero there. For ``delta >= 1e-4`` the float64 factor is within
    ``2.4e-12`` of its closed form, the same expression and error as NEURON.
    ``validation/neuron/synapse/reference_test.py`` compares both regimes with
    NEURON's ``Exp2Syn``: within ``1e-12 uS`` at ``delta = 1e-2`` and ``1e-5``.
    """

    root_type = HHTypedNeuron
    states = {
        "A": StateSpec(0.0 * u.uS),
        "B": StateSpec(0.0 * u.uS),
    }
    event_input = ScalarEventInput(u.uS, aggregation="sum")

    def __init__(
        self,
        size: Size,
        name: str | None = None,
        tau1: Initializer = 0.1 * u.ms,
        tau2: Initializer = 10.0 * u.ms,
        e: Initializer = 0.0 * u.mV,
    ):
        super().__init__(size=size, name=name)
        self._init_parameters(tau1=tau1, tau2=tau2, e=e)

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
        positive(parameters["tau1"], "tau1")
        positive(parameters["tau2"], "tau2")
        tau1 = np.asarray(u.Quantity(parameters["tau1"]).to_decimal(u.ms))
        tau2 = np.asarray(u.Quantity(parameters["tau2"]).to_decimal(u.ms))
        if np.any(tau1 >= tau2):
            raise ValueError("Exp2Syn requires tau1 < tau2.")

    def _compute_factor(self):
        return _peak_factor(_effective_tau1(self.tau1, self.tau2), self.tau2)

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
        self.A.derivative = -self.A.value / _effective_tau1(self.tau1, self.tau2)
        self.B.derivative = -self.B.value / self.tau2

    def current(self, V_post):
        return self.g * (self.e - V_post)
