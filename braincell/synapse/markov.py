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

# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainstate
import braintools
import brainunit as u

from braincell._base import HHTypedNeuron, Synapse
from braincell.mech import register_synapse
from braincell.quad.protocol import DiffEqState

__all__ = [
    'ExpSyn',
    'Exp2Syn',
    'AMPA',
    'GABAa',
    'NMDA',
]


def _decay_factor(dt, tau):
    """Return the exact exponential decay factor over one timestep."""
    return u.math.exp(-(dt / tau))


def _syn_uS_state(shape, batch_size=None):
    """Allocate one synapse state with conductance unit ``uS``."""
    return DiffEqState(
        u.Quantity(braintools.init.param(u.math.zeros, shape, batch_size), u.uS)
    )


def _syn_state(shape, batch_size=None):
    """Allocate one dimensionless synapse ODE state."""
    return DiffEqState(braintools.init.param(u.math.zeros, shape, batch_size))


def _event_payload(pre_drive, default_weight):
    """Return weighted synaptic event payload.

    Network projections write already-weighted quantities into ``pre_drive``.
    Legacy local drives such as ``NetStim(weight=1.0)`` remain dimensionless and
    therefore use the placed synapse's default ``weight``.
    """
    if isinstance(pre_drive, u.Quantity):
        return pre_drive
    return pre_drive * default_weight


@register_synapse("ExpSyn")
class ExpSyn(Synapse):
    """NEURON-compatible `ExpSyn` template.

    This class follows the dynamics in NEURON's ``expsyn.mod``:

    - state: ``g`` in ``uS``
    - decay: ``g' = -g / tau``
    - step-boundary event: ``g <- g + weighted_pre_drive``
    - current: ``i = g * (V_post - e)``
    """

    root_type = HHTypedNeuron
    current_sign = "neuron"
    current_units = "total"

    def __init__(
        self,
        size: brainstate.typing.Size,
        tau: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * u.ms,
        e: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        weight: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.uS,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)
        self.e = braintools.init.param(e, self.varshape, allow_none=False)
        self.weight = braintools.init.param(weight, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        self.g = _syn_uS_state(self.varshape, batch_size=batch_size)

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        self.g.value = u.Quantity(
            braintools.init.param(u.math.zeros, self.varshape, batch_size), u.uS
        )

    def pre_integral(self, V_post=None):
        _ = V_post

    def apply_discrete_events(self, V_post=None):
        _ = V_post
        self.g.value = self.g.value + _event_payload(self.pre_drive(), self.weight)

    def post_integral(self, V_post=None):
        _ = V_post

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.g.derivative = -self.g.value / self.tau

    def current(self, V_post):
        return self.g.value * (V_post - self.e)


@register_synapse("Exp2Syn")
class Exp2Syn(Synapse):
    """NEURON-compatible `Exp2Syn` template.

    This class follows the dynamics in NEURON's ``exp2syn.mod``:

    - states: ``A`` and ``B`` in ``uS``
    - decay: ``A' = -A / tau1``, ``B' = -B / tau2``
    - conductance: ``g = B - A``
    - current: ``i = g * (V_post - e)``
    - step-boundary event: ``A <- A + weighted_pre_drive * factor`` and same for ``B``
    """

    root_type = HHTypedNeuron
    current_sign = "neuron"
    current_units = "total"

    def __init__(
        self,
        size: brainstate.typing.Size,
        tau1: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * u.ms,
        tau2: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * u.ms,
        e: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        weight: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.uS,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.tau1 = braintools.init.param(tau1, self.varshape, allow_none=False)
        self.tau2 = braintools.init.param(tau2, self.varshape, allow_none=False)
        self.e = braintools.init.param(e, self.varshape, allow_none=False)
        self.weight = braintools.init.param(weight, self.varshape, allow_none=False)
        self.factor = self._compute_factor()

    def _effective_tau1(self):
        ratio = u.math.asarray(self.tau1 / self.tau2)
        tau1_eff = u.math.where(ratio > 0.9999, self.tau2 * 0.9999, self.tau1)
        tau1_eff = u.math.where(ratio < 1e-9, self.tau2 * 1e-9, tau1_eff)
        return tau1_eff

    def _compute_factor(self):
        tau1_eff = self._effective_tau1()
        tp = (tau1_eff * self.tau2) / (self.tau2 - tau1_eff) * u.math.log(
            u.math.asarray(self.tau2 / tau1_eff)
        )
        factor = -u.math.exp(-(tp / tau1_eff)) + u.math.exp(
            -(tp / self.tau2)
        )
        return 1.0 / factor

    def _on_param_updated(self, var_name: str, new_value) -> None:
        _ = (var_name, new_value)
        self.factor = self._compute_factor()

    @property
    def g(self):
        return self.B.value - self.A.value

    def init_state(self, V_post=None, batch_size=None):
        super().init_state(V_post=V_post, batch_size=batch_size)
        self.A = _syn_uS_state(self.varshape, batch_size=batch_size)
        self.B = _syn_uS_state(self.varshape, batch_size=batch_size)

    def reset_state(self, V_post=None, batch_size=None):
        super().reset_state(V_post=V_post, batch_size=batch_size)
        self.A.value = u.Quantity(
            braintools.init.param(u.math.zeros, self.varshape, batch_size), u.uS
        )
        self.B.value = u.Quantity(
            braintools.init.param(u.math.zeros, self.varshape, batch_size), u.uS
        )

    def pre_integral(self, V_post=None):
        _ = V_post

    def apply_discrete_events(self, V_post=None):
        _ = V_post
        delta = _event_payload(self.pre_drive(), self.weight) * self.factor
        self.A.value = self.A.value + delta
        self.B.value = self.B.value + delta

    def post_integral(self, V_post=None):
        _ = V_post

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.A.derivative = -self.A.value / self._effective_tau1()
        self.B.derivative = -self.B.value / self.tau2

    def current(self, V_post):
        return self.g * (V_post - self.e)


@register_synapse("AMPA")
class AMPA(Synapse):
    """Single-exponential AMPA synapse.

    Parameters
    ----------
    size : brainstate.typing.Size
        Point-space target shape.
    alpha : array-like or callable, optional
        Rise rate in ``ms^-1``.
    beta : array-like or callable, optional
        Decay rate in ``ms^-1``.
    T : array-like or callable, optional
        Presynaptic scaling factor.
    g_max : array-like or callable, optional
        Maximum synaptic conductance.
    E_rev : array-like or callable, optional
        Reversal potential of the synapse.
    name : str, optional
        Runtime node name.
    """

    root_type = HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.98 / u.ms,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.18 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 0.5,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        E_rev: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)

        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        """Initialize synaptic state."""
        super().init_state(V_post=V_post, batch_size=batch_size)
        self.g = _syn_state(self.varshape, batch_size=batch_size)

    def reset_state(self, V_post=None, batch_size=None):
        """Reset synaptic state."""
        super().reset_state(V_post=V_post, batch_size=batch_size)
        self.g.value = braintools.init.param(u.math.zeros, self.varshape, batch_size)

    def compute_derivative(self, V_post=None):
        """Advance one timestep of synaptic conductance dynamics."""
        _ = V_post
        self.g.derivative = self.alpha * self.pre_drive() * self.T * (1 - self.g.value) - self.beta * self.g.value

    def current(self, V_post):
        """Return the postsynaptic point current."""
        return self.g_max * self.g.value * (self.E_rev - V_post)


@register_synapse("GABAa")
class GABAa(Synapse):
    """Single-exponential GABAa synapse."""

    root_type = HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.53 / u.ms,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.18 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        E_rev: Union[brainstate.typing.ArrayLike, Callable] = -70.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)

        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        """Initialize synaptic state."""
        super().init_state(V_post=V_post, batch_size=batch_size)
        self.g = _syn_state(self.varshape, batch_size=batch_size)

    def reset_state(self, V_post=None, batch_size=None):
        """Reset synaptic state."""
        super().reset_state(V_post=V_post, batch_size=batch_size)
        self.g.value = braintools.init.param(u.math.zeros, self.varshape, batch_size)

    def compute_derivative(self, V_post=None):
        """Advance one timestep of synaptic conductance dynamics."""
        _ = V_post
        self.g.derivative = self.alpha * self.pre_drive() * self.T * (1 - self.g.value) - self.beta * self.g.value

    def current(self, V_post):
        """Return the postsynaptic point current."""
        return self.g_max * self.g.value * (self.E_rev - V_post)


@register_synapse("NMDA")
class NMDA(Synapse):
    """Double-exponential NMDA synapse."""

    root_type = HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        alpha1: Union[brainstate.typing.ArrayLike, Callable] = 2. / u.ms,
        beta1: Union[brainstate.typing.ArrayLike, Callable] = 0.01 / u.ms,
        alpha2: Union[brainstate.typing.ArrayLike, Callable] = 1. / u.ms,
        beta2: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * (u.mS / u.cm ** 2),
        E_rev: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)

        self.alpha1 = braintools.init.param(alpha1, self.varshape, allow_none=False)
        self.beta1 = braintools.init.param(beta1, self.varshape, allow_none=False)
        self.alpha2 = braintools.init.param(alpha2, self.varshape, allow_none=False)
        self.beta2 = braintools.init.param(beta2, self.varshape, allow_none=False)
        self.T = braintools.init.param(T, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.E_rev = braintools.init.param(E_rev, self.varshape, allow_none=False)

    def init_state(self, V_post=None, batch_size=None):
        """Initialize synaptic state."""
        super().init_state(V_post=V_post, batch_size=batch_size)
        self.g = _syn_state(self.varshape, batch_size=batch_size)
        self.x = _syn_state(self.varshape, batch_size=batch_size)

    def reset_state(self, V_post=None, batch_size=None):
        """Reset synaptic state."""
        super().reset_state(V_post=V_post, batch_size=batch_size)
        self.g.value = braintools.init.param(u.math.zeros, self.varshape, batch_size)
        self.x.value = braintools.init.param(u.math.zeros, self.varshape, batch_size)

    def compute_derivative(self, V_post=None):
        """Advance one timestep of synaptic conductance dynamics."""
        _ = V_post
        self.g.derivative = self.alpha1 * self.x.value * (1 - self.g.value) - self.beta1 * self.g.value
        self.x.derivative = self.alpha2 * self.pre_drive() * self.T * (1 - self.x.value) - self.beta2 * self.x.value

    def current(self, V_post):
        """Return the postsynaptic point current."""
        return self.g_max * self.g.value * (self.E_rev - V_post)
