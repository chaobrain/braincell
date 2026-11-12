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

"""Point-neuron math: the private base of the lumped Cell's engine.

:class:`PointNeuron` integrates ``C dV/dt = sum of channel currents + I_ext``
for a population with no spatial axis. The lumped Cell's engine
(``LumpedRuntime``, ``braincell/_multi_compartment/lumped.py``) subclasses it
and replaces the membrane area with that of the Cell's one CV.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import brainstate
import braintools
import brainunit as u

from braincell._base_channel import IonChannel
from braincell._base_ion import Ion
from braincell._base_neuron import HHTypedNeuron, _zero_spike_like
from braincell._misc import cast_like as _cast_like
from braincell._typing import Initializer, Size
from braincell.quad import get_integrator
from braincell.quad.protocol import IndependentIntegration, state, state_grouping

__all__ = ["PointNeuron"]


class PointNeuron(HHTypedNeuron):
    r"""Conductance-based neuron population with no spatial axis.

    The voltage of each member obeys

    .. math::

        C {dV \over dt} = \sum_j I_j(V) + I_{ext}

    where the :math:`I_j` are the currents of the ion channels and ions
    attached to the neuron, all current densities. The solver advances ``V``
    together with the states of the channels that do not integrate
    themselves (:class:`~braincell.IndependentIntegration`); :meth:`update`
    first calls every channel's ``ind_update``.

    This is the base of the lumped :class:`~braincell.Cell`'s engine
    (``LumpedRuntime``); it is not public. It holds the math of the point-neuron
    class that braincell 0.3 removed: a point neuron is now written as
    ``Cell.point(...)``.

    Parameters
    ----------
    size : int, sequence of int
        The population shape.
    C : Initializer, optional
        Specific membrane capacitance. Default is 1.0 uF/cm^2.
    V_th : Initializer, optional
        Spike threshold. Default is 0.0 mV.
    V_initializer : Initializer, optional
        Initial membrane potential. Default is uniform between -70 mV and
        -60 mV.
    spk_fun : Callable, optional
        Surrogate spike function. Default is ``braintools.surrogate.ReluGrad()``.
    solver : str or Callable, optional
        A name accepted by :func:`braincell.quad.get_integrator` or a step
        function. Default is ``'ralston2'``.
    name : str, optional
        The neuron group name.
    **ion_channels
        Ions and channels to attach, keyed by attribute name.
    """

    def __init__(
        self,
        size: Size,
        C: Initializer = 1.0 * u.uF / u.cm**2,
        V_th: Initializer = 0.0 * u.mV,
        V_initializer: Initializer = braintools.init.Uniform(-70 * u.mV, -60.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: Union[str, Callable] = 'ralston2',
        name: Optional[str] = None,
        **ion_channels,
    ):
        super().__init__(size, name=name, **ion_channels)
        self.C = braintools.init.param(C, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
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

        A point neuron has one compartment.

        Returns
        -------
        int
            The number of compartments, which is 1.
        """
        return 1

    def _ion_channels(self) -> Dict[str, IonChannel]:
        """Return the directly attached ion channels, keyed by attribute name.

        The one spelling of this graph walk. It is a walk rather than a
        stored list because ``HHTypedNeuron.__init__`` only records the
        channels passed to it as keyword arguments, while a channel may also
        be assigned as an attribute afterwards.

        Returns
        -------
        dict of str to IonChannel
            Channels one level below this neuron, in attribute order.
        """
        return self.nodes(IonChannel, allowed_hierarchy=(1, 1))

    def _run_driven_hook(self, hook: str, channels: Dict[str, IonChannel]) -> None:
        """Call one integration hook on the mechanisms this neuron's solver advances.

        A channel or ion carrying the :class:`~braincell.IndependentIntegration`
        marker advances its own state in ``ind_update``, so the neuron's
        ``pre_integral`` / ``compute_derivative`` / ``post_integral`` hooks
        skip it — stepping it there as well would integrate it twice per
        step. The dependent channels under such an ion (a kinetic ion's HH
        calcium channel) are still the solver's. :meth:`update` calls
        ``ind_update`` on every channel, including the marked ones.

        Parameters
        ----------
        hook : str
            ``"pre_integral"``, ``"compute_derivative"`` or ``"post_integral"``.
        channels : dict of str to IonChannel
            The result of :meth:`_ion_channels`.
        """
        for ch in channels.values():
            if not isinstance(ch, IndependentIntegration):
                getattr(ch, hook)(self.V.value)
            elif isinstance(ch, Ion):
                ch._run_on_dependent_children(ch._channels(), hook, self.V.value)

    def init_state(self, batch_size=None):
        """
        Initialize the state of the neuron.

        This method sets up the initial membrane potential (V) of the neuron using the
        V_initializer and initializes other state variables through the parent class.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for initialization. If None, no batch dimension is added.
        """
        # A point neuron has no spatial axis, so its hidden states are plain
        # (non-grouped) states. The scope is set explicitly rather than left
        # to the default so that a program building both a point neuron and
        # a cable Cell is correct regardless of construction order.
        with state_grouping(False):
            self.V = state(braintools.init.param(self.V_initializer, self.varshape, batch_size))
            self.spike = brainstate.ShortTermState(_zero_spike_like(self.V.value))
            super().init_state(batch_size)

    def reset_state(self, batch_size=None):
        """
        Reset the state of the neuron.

        This method resets the membrane potential (V) to its initial value and
        reinitialized other state variables through the parent class.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for resetting. If None, no batch dimension is added.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.spike.value = _zero_spike_like(self.V.value)
        with state_grouping(False):
            super().reset_state(batch_size)

    def pre_integral(self, I_ext=0.0 * u.nA / u.cm**2):
        """
        Perform pre-integration operations.

        This method calls the pre_integral method of all ion channels associated
        with this neuron before the main integration step.

        Parameters
        ----------
        I_ext : float, optional
            External current input. Default is 0 nA/cm^2.
        """
        self._run_driven_hook("pre_integral", self._ion_channels())

    def compute_derivative(self, I_ext=0.0 * u.nA / u.cm**2):
        """
        Compute the derivative of the membrane potential.

        This method calculates the derivative of the membrane potential considering
        external inputs, synaptic currents, and ion channel currents.

        Parameters
        ----------
        I_ext : float, optional
            External current input, in the unit system of ``C``: a current
            density with the default ``C`` in uF/cm^2.
        """
        channels = self._ion_channels()
        I_ext = self.sum_current_inputs(I_ext, self.V.value)
        for key, ch in channels.items():
            try:
                contribution = ch.current(self.V.value)
            except (TypeError, ValueError, RuntimeError, ArithmeticError) as e:
                raise ValueError(f"Error in computing current for ion channel '{key}': \n{ch}\nError: {e}") from e
            # An ion pool with no channels of its own returns ``None``: it has
            # nothing to add, not a zero in some unit it would have to guess.
            # The common KCa setup -- K and Ca carrying only a mixed channel --
            # hits this, and used to fail with a unit mismatch against ``None``.
            if contribution is not None:
                I_ext = I_ext + contribution
        self.V.derivative = I_ext / self.C
        self._run_driven_hook("compute_derivative", channels)

    def post_integral(self, I_ext=0.0 * u.nA / u.cm**2):
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
        self._run_driven_hook("post_integral", self._ion_channels())

    def update(self, I_ext=0.0 * u.nA / u.cm**2):
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
        # A value input (add_current_input, add_delta_input) acts on this
        # step only: every solver stage reads it, and the step removes it.
        with self._value_input_step():
            # Every channel, including the self-integrating ones that the
            # three hook methods above skip.
            for ch in self._ion_channels().values():
                ch.ind_update(self.V.value)

            # numerical integration
            last_V = self.V.value
            self.solver(self, I_ext)
            spk = self.get_spike(last_V, self.V.value)
            self.spike.value = spk
        return spk

    def soma_spike(self):
        denom = _cast_like(20.0 * u.mV, self.V.value)
        V_th = _cast_like(self.V_th, self.V.value)
        return self.spk_fun((self.V.value - V_th) / denom)
