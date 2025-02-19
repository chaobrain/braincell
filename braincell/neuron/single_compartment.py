# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from typing import Union, Optional, Callable, Tuple

import brainstate as bst
import brainunit as u

from braincell._base import HHTypedNeuron, IonChannel
from braincell._integrators import DiffEqState
from braincell._integrators import get_integrator

__all__ = [
    'SingleCompartment',
]


class SingleCompartment(HHTypedNeuron):
    r"""
    Base class to model conductance-based neuron group.

    The standard formulation for a conductance-based model is given as

    .. math::

        C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

    where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
    reversal potential, :math:`M` is the activation variable, and :math:`N` is the
    inactivation variable.

    :math:`M` and :math:`N` have the dynamics of

    .. math::

        {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

    where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
    :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
    Equivalently, the above equation can be written as:

    .. math::

        \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

    where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.


    Parameters
    ----------
    size : int, sequence of int
      The network size of this neuron group.
    name : optional, str
      The neuron group name.
    """
    __module__ = 'braincell.neuron'

    def __init__(
        self,
        size: bst.typing.Size,
        C: Union[bst.typing.ArrayLike, Callable] = 1. * u.uF / u.cm ** 2,
        V_th: Union[bst.typing.ArrayLike, Callable] = 0. * u.mV,
        V_initializer: Union[bst.typing.ArrayLike, Callable] = bst.init.Uniform(-70 * u.mV, -60. * u.mV),
        spk_fun: Callable = bst.surrogate.ReluGrad(),
        solver: str = 'rk2',
        name: Optional[str] = None,
        **ion_channels
    ):
        super().__init__(size, name=name, **ion_channels)
        assert self.n_compartment == 1, "SingleCompartment neuron should have only one compartment."
        self.C = bst.init.param(C, self.varshape)
        self.V_th = bst.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer
        self.spk_fun = spk_fun
        self.solver = get_integrator(solver)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        return self.varshape

    @property
    def n_compartment(self) -> int:
        return 1

    def init_state(self, batch_size=None):
        self.V = DiffEqState(bst.init.param(self.V_initializer, self.varshape, batch_size))
        super().init_state(batch_size)

    def reset_state(self, batch_size=None):
        self.V.value = bst.init.param(self.V_initializer, self.varshape, batch_size)
        super().init_state(batch_size)

    def pre_integral(self, I_ext=0. * u.nA / u.cm ** 2):
        # pre integral
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.pre_integral(self.V.value)

    def compute_derivative(self, I_ext=0. * u.nA / u.cm ** 2):
        # [ Compute the derivative of membrane potential ]
        # 1. inputs + 2. synapses
        I_ext = self.sum_current_inputs(I_ext, self.V.value)

        # 3. channel
        for key, ch in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            I_ext = I_ext + ch.current(self.V.value)

        # 4. derivatives
        self.V.derivative = I_ext / self.C

        # [ integrate dynamics of ion and ion channel ]
        # check whether the children channel have the correct parents.
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.compute_derivative(self.V.value)

    def post_integral(self, I_ext=0. * u.nA / u.cm ** 2):
        # post integral
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.post_integral(self.V.value)

    def update(self, I_ext=0. * u.nA / u.cm ** 2):
        last_V = self.V.value
        # integration
        t = bst.environ.get('t')
        self.solver(self, t, I_ext)
        # post integral
        self.post_integral(I_ext)
        # return spike
        return self.get_spike(last_V, self.V.value)

    def get_spike(self, last_V, next_V):
        denom = 20.0 * u.mV
        return (
            self.spk_fun((next_V - self.V_th) / denom) *
            self.spk_fun((self.V_th - last_V) / denom)
        )
