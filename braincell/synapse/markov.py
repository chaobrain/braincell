# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Callable

import brainunit as u

import brainstate

__all__ = [
    'AMPA',
    'GABAa',
    'NMDA',
]


class AMPA(brainstate.nn.Synapse):
    """
    AMPA synapse model class.

    This class implements the dynamics of an AMPA-type synapse using an exponential Euler integration scheme.
    The synaptic conductance is updated based on presynaptic spike input and time constants.

    Args:
        in_size (brainstate.typing.Size): The input size or shape of the synapse.
        alpha (Union[brainstate.typing.ArrayLike, Callable], optional):
            The rise rate constant (default: 0.98 / u.ms).
        beta (Union[brainstate.typing.ArrayLike, Callable], optional):
            The decay rate constant (default: 0.18 / u.ms).
        T (Union[brainstate.typing.ArrayLike, Callable], optional):
            The synaptic efficacy or scaling factor (default: 0.5).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.98 / u.ms,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.18 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 0.5,
    ):
        super().__init__(in_size=in_size)

        # parameters
        self.alpha = brainstate.init.param(alpha)
        self.beta = brainstate.init.param(beta)
        self.T = brainstate.init.param(T)

    def init_state(self, **kwargs):
        """
        Initialize the hidden state `g` (synaptic conductance) to zeros.
        """
        self.g = brainstate.HiddenState(brainstate.init.param(u.math.zeros, self.varshape))

    def reset_state(self, **kwargs):
        """
        Reset the hidden state `g` (synaptic conductance) to zeros.
        """
        self.g.value = brainstate.init.param(u.math.zeros, self.varshape)

    def update(self, pre_spike):
        """
        Update the synaptic conductance `g` based on the presynaptic spike input.

        The update follows the equation:
            dg/dt = alpha * pre_spike * T * (1 - g) - beta * g

        Args:
            pre_spike: Presynaptic spike input.

        Returns:
            Updated synaptic conductance value.
        """
        dg = lambda g: self.alpha * pre_spike * self.T * (1 - g) - self.beta * g
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        return self.g.value


class GABAa(brainstate.nn.Synapse):
    """
    GABAa synapse model class.

    This class implements the dynamics of a GABAa-type synapse using an exponential Euler integration scheme.
    The synaptic conductance is updated based on presynaptic spike input and time constants.

    Args:
        in_size (brainstate.typing.Size): The input size or shape of the synapse.
        alpha (Union[brainstate.typing.ArrayLike, Callable], optional):
            The rise rate constant of the synaptic conductance. Defaults to 0.53 / u.ms.
        beta (Union[brainstate.typing.ArrayLike, Callable], optional):
            The decay rate constant of the synaptic conductance. Defaults to 0.18 / u.ms.
        T (Union[brainstate.typing.ArrayLike, Callable], optional):
            The synaptic efficacy or scaling factor. Defaults to 1.0.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.53 / u.ms,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.18 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
    ):
        super().__init__(in_size=in_size)

        # parameters
        self.alpha = brainstate.init.param(alpha)
        self.beta = brainstate.init.param(beta)
        self.T = brainstate.init.param(T)

    def init_state(self, **kwargs):
        """
        Initialize the hidden state `g` (synaptic conductance) to zeros.
        """
        self.g = brainstate.HiddenState(brainstate.init.param(u.math.zeros, self.varshape))

    def reset_state(self, **kwargs):
        """
        Reset the hidden state `g` (synaptic conductance) to zeros.
        """
        self.g.value = brainstate.init.param(u.math.zeros, self.varshape)

    def update(self, pre_spike):
        """
        Update the synaptic conductance `g` based on the presynaptic spike input.

        The update follows the equation:
            dg/dt = alpha * pre_spike * T * (1 - g) - beta * g

        Args:
            pre_spike: Presynaptic spike input.

        Returns:
            Updated synaptic conductance value.
        """
        dg = lambda g: self.alpha * pre_spike * self.T * (1 - g) - self.beta * g
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        return self.g.value


class NMDA(brainstate.nn.Synapse):
    """
    NMDA synapse model class.

    This class implements the dynamics of an NMDA-type synapse using an exponential Euler integration scheme.
    The synaptic conductance is updated based on presynaptic spike input and multiple time constants.

    Args:
        in_size (brainstate.typing.Size): The input size or shape of the synapse.
        alpha1 (Union[brainstate.typing.ArrayLike, Callable], optional):
            The rise rate constant for the synaptic conductance `g`. Defaults to 2 / ms.
        beta1 (Union[brainstate.typing.ArrayLike, Callable], optional):
            The decay rate constant for the synaptic conductance `g`. Defaults to 0.01 / ms.
        alpha2 (Union[brainstate.typing.ArrayLike, Callable], optional):
            The rise rate constant for the auxiliary variable `x`. Defaults to 1 / ms.
        beta2 (Union[brainstate.typing.ArrayLike, Callable], optional):
            The decay rate constant for the auxiliary variable `x`. Defaults to 0.5 / ms.
        T (Union[brainstate.typing.ArrayLike, Callable], optional):
            The synaptic efficacy or scaling factor. Defaults to 1.0.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        alpha1: Union[brainstate.typing.ArrayLike, Callable] = 2. / u.ms,
        beta1: Union[brainstate.typing.ArrayLike, Callable] = 0.01 / u.ms,
        alpha2: Union[brainstate.typing.ArrayLike, Callable] = 1. / u.ms,
        beta2: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        T: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
    ):
        super().__init__(in_size=in_size)

        # Initialize model parameters
        self.alpha1 = brainstate.init.param(alpha1)
        self.beta1 = brainstate.init.param(beta1)
        self.alpha2 = brainstate.init.param(alpha2)
        self.beta2 = brainstate.init.param(beta2)
        self.T = brainstate.init.param(T)

    def init_state(self, **kwargs):
        """
        Initialize the hidden states `g` (synaptic conductance) and `x` (auxiliary variable) to zeros.
        """
        self.g = brainstate.HiddenState(brainstate.init.param(u.math.zeros, self.varshape))
        self.x = brainstate.HiddenState(brainstate.init.param(u.math.zeros, self.varshape))

    def reset_state(self, **kwargs):
        """
        Reset the hidden states `g` (synaptic conductance) and `x` (auxiliary variable) to zeros.
        """
        self.g.value = brainstate.init.param(u.math.zeros, self.varshape)
        self.x.value = brainstate.init.param(u.math.zeros, self.varshape)

    def update(self, pre_spike):
        """
        Update the synaptic conductance `g` and auxiliary variable `x` based on the presynaptic spike input.

        The updates follow the differential equations:
            dg/dt = alpha1 * x * (1 - g) - beta1 * g
            braincell/dt = alpha2 * pre_spike * T * (1 - x) - beta2 * x

        Args:
            pre_spike: Presynaptic spike input.

        Returns:
            Updated synaptic conductance value `g`.
        """
        # Define the differential equations for g and x
        dg = lambda g, x: self.alpha1 * x * (1 - g) - self.beta1 * g
        dx = lambda x: self.alpha2 * pre_spike * self.T * (1 - x) - self.beta2 * x
        # Update the hidden states using exponential Euler integration
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value, self.x.value)
        self.x.value = brainstate.nn.exp_euler_step(dx, self.x.value)
        return self.g.value
