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

from typing import Optional, Callable, Tuple, Union

import brainstate
import brainunit as u

from ._base import HHTypedNeuron, IonChannel
from ._integrator import get_integrator
from ._protocol import DiffEqState
from ._typing import Initializer

__all__ = [
    'SingleCompartment',
]


class SingleCompartment(HHTypedNeuron):
    r"""Base class to model conductance-based neurons with single compartment.

    The standard formulation for a conductance-based point neuron model is given as

    .. math::

        C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

    where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
    reversal potential, :math:`M` is the activation variable, :math:`N` is the
    inactivation variable, $\bar{g}_{j}$ is the maximum conductance.

    :math:`M` and :math:`N` have the dynamics of

    .. math::

        {braincell \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

    where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
    :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
    Equivalently, the above equation can be written as:

    .. math::

        \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

    where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.

    The implementations of $x$ please see ``braincell.ion``, and ``braincell.channel`` modules.

    Parameters
    ----------
    size : int, sequence of int
        The network size of this neuron group.
    C : Initializer, optional
        Membrane capacitance. Default is 1.0 uF/cm²
    V_th : Initializer, optional
        Threshold voltage for spike detection. Default is 0.0 mV
    V_initializer : Initializer, optional
        Initial membrane potential distribution. Default is uniform between -70 mV and -60 mV
    spk_fun : Callable, optional
        Spike function for threshold crossing detection. Default is ReLU gradient
    solver : str or Callable, optional
        Numerical integration method. Default is 'rk2' (second-order Runge-Kutta)
    name : str, optional
        The neuron group name.
    **ion_channels : dict
        Additional ion channels to include in the neuron model

    Notes
    -----
    This class is subclassed from :class:`braincell.HHTypedNeuron`.
    """
    __module__ = 'braincell'

    def __init__(
        self,
        size: brainstate.typing.Size,
        C: Initializer = 1. * u.uF / u.cm ** 2,
        V_th: Initializer = 0. * u.mV,
        V_initializer: Initializer = brainstate.init.Uniform(-70 * u.mV, -60. * u.mV),
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),
        solver: Union[str, Callable] = 'rk2',
        name: Optional[str] = None,
        **ion_channels
    ):
        """
        Initialize a :class:`SingleCompartment` neuron.

        Parameters
        ----------
        size : brainstate.typing.Size
            The size of the neuron group.
        C : Union[brainstate.typing.ArrayLike, Callable], optional
            Membrane capacitance. Default is 1. * u.uF / u.cm ** 2.
        V_th : Union[brainstate.typing.ArrayLike, Callable], optional
            Threshold voltage. Default is 0. * u.mV.
        V_initializer : Union[brainstate.typing.ArrayLike, Callable], optional
            Initial membrane potential. Default is uniform distribution between -70 mV and -60 mV.
        spk_fun : Callable, optional
            Spike function. Default is brainstate.surrogate.ReluGrad().
        solver : str | Callable, optional
            Numerical solver for integration. Default is 'rk2'.
        name : Optional[str], optional
            Name of the neuron group. Default is None.
        **ion_channels : dict
            Additional ion channels to be added to the neuron.
        """
        super().__init__(size, name=name, **ion_channels)
        assert self.n_compartment == 1, "SingleCompartment neuron should have only one compartment."
        self.C = brainstate.init.param(C, self.varshape)
        self.V_th = brainstate.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer
        self.spk_fun = spk_fun
        self.solver = get_integrator(solver)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        """
        Get the population size of the neuron group.

        This property returns the shape of the neuron population, which determines
        how many individual neurons are in this group. For example, a shape of (10,)
        means 10 neurons, while (2, 5) would represent a 2D grid of neurons.

        Returns
        -------
        Tuple[int, ...]
            The shape of the neuron population as a tuple of integers.
        """
        return self.varshape

    @property
    def n_compartment(self) -> int:
        """
        Get the number of compartments in this neuron model.

        For the :class:`SingleCompartment` model, this always returns 1 since it's a point
        neuron model with only one compartment. Multi-compartment models would
        override this property to return their respective number of compartments.

        Returns
        -------
        int
            The number of compartments, which is 1 for :class:`SingleCompartment` neurons.
        """
        return 1

    def init_state(self, batch_size=None):
        """
        Initialize the state of the neuron.

        This method sets up the initial membrane potential (V) of the neuron using the
        V_initializer and initializes other state variables through the parent class.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for initialization. If None, no batch dimension is added.

        Returns
        -------
        None
        """
        self.V = DiffEqState(brainstate.init.param(self.V_initializer, self.varshape, batch_size))
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        super().init_state(batch_size)

    def reset_state(self, batch_size=None):
        """
        Reset the state of the neuron.

        This method resets the membrane potential (V) to its initial value and
        reinitializes other state variables through the parent class.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for resetting. If None, no batch dimension is added.

        Returns
        -------
        None
        """
        self.V.value = brainstate.init.param(self.V_initializer, self.varshape, batch_size)
        self.spike = self.get_spike(self.V.value, self.V.value)
        super().init_state(batch_size)

    def pre_integral(self, I_ext=0. * u.nA / u.cm ** 2):
        """
        Perform pre-integration operations.

        This method calls the pre_integral method of all ion channels associated
        with this neuron before the main integration step.

        Parameters
        ----------
        I_ext : float, optional
            External current input. Default is 0 nA/cm^2.
        """
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.pre_integral(self.V.value)

    def compute_derivative(self, I_ext=0. * u.nA / u.cm ** 2):
        """
        Compute the derivative of the membrane potential.

        This method calculates the derivative of the membrane potential considering
        external inputs, synaptic currents, and ion channel currents.

        Parameters
        ----------
        I_ext : float, optional
            External current input. Default is 0 nA/cm^2.
        """
        I_ext = self.sum_current_inputs(I_ext, self.V.value)
        for key, ch in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            try:
                I_ext = I_ext + ch.current(self.V.value)
            except Exception as e:
                raise ValueError(
                    f"Error in computing current for ion channel '{key}': \n"
                    f"{ch}\n"
                    f"Error: {e}"
                )
        self.V.derivative = I_ext / self.C
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.compute_derivative(self.V.value)

    def post_integral(self, I_ext=0. * u.nA / u.cm ** 2):
        """
        Perform post-integration operations.

        This method updates the membrane potential with delta inputs and calls
        the post_integral method of all associated ion channels.

        Parameters
        ----------
        I_ext : float, optional
            External current input. Default is 0 nA/cm^2.
        """
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.post_integral(self.V.value)

    def update(self, I_ext=0. * u.nA / u.cm ** 2):
        """
        Update the neuron state and check for spikes.

        This method performs the integration step to update the neuron's state
        and checks if a spike has occurred.

        Parameters
        ----------
        I_ext : float, optional
            External current input. Default is 0 nA/cm^2.

        Returns
        -------
        spike : array-like
            An array indicating whether a spike occurred (1) or not (0) for each neuron.
        """
        last_V = self.V.value
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get('dt')
        self.solver(self, t, dt, I_ext)
        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    def get_spike(self, last_V, next_V):
        """
        Determine if a spike has occurred.

        This method checks if a spike has occurred by comparing the previous and
        current membrane potentials against the threshold.

        Parameters
        ----------
        last_V : array-like
            The membrane potential at the previous time step.
        next_V : array-like
            The membrane potential at the current time step.

        Returns
        -------
        spike : array-like
            An array indicating whether a spike occurred (1) or not (0) for each neuron.
        """
        denom = 20.0 * u.mV
        return (
            self.spk_fun((next_V - self.V_th) / denom) *
            self.spk_fun((self.V_th - last_V) / denom)
        )

    def soma_spike(self):
        denom = 20.0 * u.mV
        return self.spk_fun((self.V.value - self.V_th) / denom)
