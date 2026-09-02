# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""The Hodgkin-Huxley typed neuron base class.

``HHTypedNeuron`` lives in its own module so that
:mod:`braincell._base_ion` can name it as ``root_type`` with a plain
top-of-file import. The class and the ion classes once shared a module
and reached each other through bottom-of-file imports; that arrangement
worked only because of the exact order of statements across two files.
Keeping the class here makes the cycle structurally impossible rather
than merely avoided, which is what
``_base_neuron_test.BaseNeuronExportTest`` pins.
"""

from typing import Optional, Tuple

import brainpy
import brainunit as u

from braincell._typing import Size
from ._base_channel import IonChannel
from ._misc import Container, TreeNode, cast_like as _cast_like
from .quad.protocol import DiffEqModule

__all__ = ["HHTypedNeuron"]


def _zero_spike_like(V):
    """Return a zero spike buffer matching a membrane-voltage value.

    Parameters
    ----------
    V : array-like or Quantity
        Membrane voltage value whose shape should be mirrored.

    Returns
    -------
    array-like
        Unitless zero-valued spike buffer with the same shape as ``V``.

    Notes
    -----
    ``init_state`` and ``reset_state`` seed ``last_V`` and ``next_V`` with
    the same value, so no threshold crossing can occur.  Building the zero
    buffer directly avoids tracing the surrogate spike function during these
    lifecycle phases.
    """
    return u.math.zeros_like(u.get_magnitude(V))


class HHTypedNeuron(brainpy.state.Dynamics, Container, DiffEqModule):
    """Base class for Hodgkin-Huxley typed neuronal membrane dynamics.

    Subclasses own the membrane voltage ``V`` and the integration scheme;
    this class owns the ion-channel container and the lifecycle calls that
    fan out to it.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target: an int, or a tuple/list of ints
        read as ``(..., n_neuron, n_compartment)``.
    name : str, optional
        The name of the instance. A default is generated when omitted.
    **ion_channels
        Ion channel instances to add to the neuron, keyed by attribute name.

    Raises
    ------
    ValueError
        If ``size`` is neither an int nor a non-empty tuple/list of ints.
    """

    __module__ = 'braincell'
    _container_name = 'ion_channels'

    def __init__(self, size: Size, name: Optional[str] = None, **ion_channels):
        super().__init__(size, name=name)

        # attribute for ``Container``
        self.ion_channels = self._format_elements(IonChannel, **ion_channels)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        """Neuron-population shape, that is ``size`` without the compartment axis.

        Returns
        -------
        tuple of int
            The population size in each dimension: ``(100, 50)`` is a 2-D
            population of 100 by 50 neurons.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    @property
    def n_compartment(self) -> int:
        """Number of compartments per neuron.

        Returns
        -------
        int
            How many compartments -- soma, dendrite, axon sections -- each
            neuron in the group is divided into.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    def current(self, *args, **kwargs):
        """Sum the membrane current contributed by every ion channel.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Returns
        -------
        brainunit.Quantity
            Total membrane current, in the subclass's current units.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def pre_integral(self, *args, **kwargs):
        """Run whatever must happen before each integration step.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    def compute_derivative(self, *args, **kwargs):
        """Fill in the derivative of every state this neuron integrates.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def post_integral(self, *args, **kwargs):
        """Run whatever must happen after each integration step.

        Does nothing by default. Subclasses override it to publish spikes,
        clamp concentrations, or otherwise fix up state the solver wrote.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.
        """

    def init_state(self, batch_size=None):
        """Allocate the state of every direct ion-channel child.

        Parameters
        ----------
        batch_size : int, optional
            Leading batch dimension to allocate channel states with.

        Raises
        ------
        TypeError
            If a child channel's ``root_type`` does not accept this neuron.

        Notes
        -----
        Channels are seeded from the current membrane potential
        ``self.V.value``, so ``V`` must already be initialized.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        TreeNode.check_hierarchies(self.__class__, *nodes)
        for channel in nodes:
            channel.init_state(self.V.value, batch_size=batch_size)

    def reset_state(self, batch_size=None):
        """Return every direct ion-channel child to its initial state.

        Parameters
        ----------
        batch_size : int, optional
            Leading batch dimension to reset channel states with.

        Notes
        -----
        Unlike :meth:`init_state` this does not re-check hierarchies: the
        state already exists, so the check has already run.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        for channel in nodes:
            channel.reset_state(self.V.value, batch_size=batch_size)

    def add(self, **elements):
        """Add ion channels to the neuron after construction.

        Parameters
        ----------
        **elements
            Ion channel instances to add, keyed by attribute name.

        Raises
        ------
        TypeError
            If an element's ``root_type`` does not accept this neuron, or if
            it is not an :class:`~braincell.IonChannel`.
        """
        TreeNode.check_hierarchies(type(self), **elements)
        self.ion_channels.update(self._format_elements(IonChannel, **elements))

    def get_spike(self, last_V, next_V):
        """Surrogate-gradient spike indicator at the ``V_th`` crossing.

        Uses ``self.V_th`` (threshold voltage) and ``self.spk_fun``
        (surrogate-gradient callable) supplied by subclasses. The
        product of rising- and falling-crossing terms produces a
        non-zero value only when ``last_V < V_th <= next_V``.
        """
        denom = _cast_like(20.0 * u.mV, next_V)
        V_th = _cast_like(self.V_th, next_V)
        return self.spk_fun((next_V - V_th) / denom) * self.spk_fun((V_th - last_V) / denom)
