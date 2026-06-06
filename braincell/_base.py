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

# -*- coding: utf-8 -*-

"""
Base classes and core components for modeling neurons and ion channels in the BrainCell library.

This module defines the fundamental classes and structures for creating Hodgkin-Huxley type
neuron models and various ion channels. It provides a hierarchical structure that allows
for the creation of complex, biologically accurate neural simulations.

Class Hierarchy:
----------------

1. HHTypedNeuron
   Base class for Hodgkin-Huxley type neuron models.
   - SingleCompartment (not defined in this file)
     A subclass representing a single-compartment neuron model.

2. IonChannel
   Base class for all ion channel types.

   - Ion
     Base class for specific ion channels.
     - Calcium
       Represents calcium ion channels.
     - Potassium
       Represents potassium ion channels.
     - Sodium
       Represents sodium ion channels.

   - MixIons
     Represents channels that involve multiple ion types.

   - Channel
     A generic channel class, possibly for custom or complex channel types.

Key Components:
---------------
- HHTypedNeuron: The foundation for creating Hodgkin-Huxley type neuron models.
  It manages ion channels and provides methods for current calculation and state updates.

- IonChannel: The base class for all ion channel types, defining the interface
  for channel behavior including current calculation and state management.

- Ion: Specializes IonChannel for specific ion types (Calcium, Potassium, Sodium).
  Each subclass represents the dynamics of its respective ion.

- MixIons: Allows for the creation of channels that involve multiple ion types,
  useful for modeling more complex channel behaviors.

- Channel: A generic channel class that can be used for custom channel types or
  as a base for more specific channel implementations.

This structure allows for a flexible and extensible framework for modeling various
types of neurons and ion channels, from simple single-compartment models to more
complex multi-compartment or custom channel configurations.

Usage:
------
Users can subclass these base classes to create specific neuron models or ion channel
types, leveraging the provided structure and methods to implement detailed, biologically
accurate simulations.

Note:
-----
The actual implementations of some classes (e.g., SingleCompartment, specific Ion subclasses)
may be defined in other files within the BrainCell library.
"""

from typing import Optional, Dict, Sequence, Callable, NamedTuple, Tuple, Type, Hashable

import brainstate
import brainpy
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.mixin import _JointGenericAlias

from .quad.protocol import DiffEqModule, IndependentIntegration
from ._misc import cast_like as _cast_like, set_module_as, Container, TreeNode


__all__ = [
    'HHTypedNeuron',

    'IonChannel',

    'Ion',
    'MixIons',
    'Channel',
    'Synapse',

    'mix_ions',
    'IonInfo',
]


class HHTypedNeuron(brainpy.state.Dynamics, Container, DiffEqModule):
    """
    The base class for the Hodgkin-Huxley typed neuronal membrane dynamics.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target. Can be an integer or a tuple of integers.
        Must be at least 1-dimensional, representing (..., n_neuron, n_compartment).

    name : Optional[str]
        The name of the HHTypedNeuron instance. If not provided, a default name will be used.

    **ion_channels
        A dictionary of ion channel instances to be added to the neuron.
        Each key-value pair represents an ion channel name and its corresponding instance.

    Raises
    ------
    ValueError
        If the size parameter is not correctly formatted (must be int or tuple/list of int).

    AssertionError
        If the size is less than 1-dimensional.
    """
    __module__ = 'braincell'
    _container_name = 'ion_channels'

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        **ion_channels
    ):
        super().__init__(size, name=name)

        # attribute for ``Container``
        self.ion_channels = self._format_elements(IonChannel, **ion_channels)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        """
        Get the population size of the neuron group.

        This property returns the size of the neuron population, which represents
        the number of neurons in each dimension of the group.

        Returns
        -------
        Tuple[int, ...]
            A tuple of integers representing the population size in each dimension.
            For example, (100, 50) would represent a 2D population with 100 neurons
            in the first dimension and 50 in the second.

        Raises
        ------
        NotImplementedError
            This method is not implemented in the base class and must be
            implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def n_compartment(self) -> int:
        """
        Get the number of compartments in the neuron group.

        This property represents the number of distinct compartments within each neuron
        in the group. Compartments are typically used to model different sections of a neuron,
        such as the soma, dendrites, and axon.

        Returns
        -------
        int
            The number of compartments in each neuron of the group.

        Raises
        ------
        NotImplementedError
            This method is not implemented in the base class and must be
            implemented by subclasses.
        """
        raise NotImplementedError

    def current(self, *args, **kwargs):
        """
        Generate ion channel current.

        This method calculates and returns the current generated by the ion channel.
        It must be implemented by subclasses to provide specific behavior for each
        type of ion channel.

        Parameters
        ----------
        *args
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Returns
        -------
        float or ndarray
            The calculated ion channel current. The exact type and shape of the
            return value depend on the specific implementation in the subclass.

        Raises
        ------
        NotImplementedError
            If this method is not overridden in a subclass.

        Notes
        -----
        This is an abstract method that must be implemented by all subclasses.
        The implementation should provide the logic for calculating the ion
        channel current based on the channel's properties and current state.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def pre_integral(self, *args, **kwargs):
        """
        Perform any necessary operations before the integral step in the simulation.

        This method is called before the integration of the differential equations
        in each time step. It allows for any preprocessing or setup required before
        the actual integration occurs.

        Parameters
        ----------
        *args
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses to provide specific
            pre-integration behavior.

        Notes
        -----
        Subclasses should override this method to implement any necessary
        operations that need to be performed before the integration step.
        This could include updating certain variables, checking conditions,
        or preparing data for the integration process.
        """
        raise NotImplementedError

    def compute_derivative(self, *args, **kwargs):
        """
        Compute the derivative of the state variables for the ion channel.

        This method calculates the rate of change of the state variables
        associated with the ion channel. It is an abstract method that must
        be implemented by subclasses to provide specific behavior for each
        type of ion channel.

        Parameters
        ----------
        *args
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        Subclasses should override this method to implement the specific
        equations governing the dynamics of the ion channel. The implementation
        should calculate how the state variables change over time based on
        the current state and any input parameters.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def post_integral(self, *args, **kwargs):
        """
        Perform any necessary operations after the integral step in the simulation.

        This method is called after the integration of the differential equations
        in each time step. For the neuron model, this typically corresponds to
        the `update()` function, where state variables are updated based on the
        results of the integration.

        Parameters
        ----------
        *args
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Notes
        -----
        Subclasses should override this method to implement any necessary
        operations that need to be performed after the integration step,
        such as updating membrane potentials, ion concentrations, or other
        state variables of the neuron model.
        """
        pass

    def init_state(self, batch_size=None):
        """
        Initialize the state variables of the neuron group.

        This method initializes the state variables for all ion channels in the neuron group.
        It retrieves all IonChannel nodes, checks their hierarchies, and calls the init_state
        method for each channel.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for the simulation. If provided, it will be passed to each
            channel's init_state method to initialize states with the specified batch size.

        Notes
        -----
        - This method uses the current membrane potential (self.V.value) when initializing
          the state of each channel.
        - The hierarchy of each IonChannel node is checked to ensure proper structure.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        TreeNode.check_hierarchies(self.__class__, *nodes)
        for channel in nodes:
            channel.init_state(self.V.value, batch_size=batch_size)

    def reset_state(self, batch_size=None):
        """
        Reset the state variables of the neuron group to their initial values.

        This method iterates through all IonChannel nodes in the neuron group and calls
        their respective reset_state methods. It's typically used to reinitialize the
        neuron group's state, often before starting a new simulation or when transitioning
        between different phases of a simulation.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for the simulation. If provided, it will be passed to each
            channel's reset_state method to reset states with the specified batch size.
            This is useful for maintaining consistency in batched simulations.

        Notes
        -----
        - The method uses the current membrane potential (self.V.value) when resetting
          the state of each channel.
        - Only IonChannel nodes with an allowed hierarchy of (1, 1) are considered.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        for channel in nodes:
            channel.reset_state(self.V.value, batch_size=batch_size)

    def add(self, **elements):
        """
        Add new elements to the neuron group.

        This method adds new ion channel elements to the neuron group. It checks the hierarchies
        of the new elements and updates the ion_channels dictionary with the formatted elements.

        Parameters
        ----------
        **elements: Any
            A dictionary of new elements to add. Each key-value pair represents an ion channel
            name and its corresponding instance.

        Raises
        ------
        TypeError
            If the hierarchies of the new elements are incompatible with the current structure.

        Notes
        -----
        The method uses TreeNode.check_hierarchies to ensure the new elements are compatible
        with the existing structure. It then formats the elements as IonChannel instances
        before adding them to the ion_channels dictionary.
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
        return (
            self.spk_fun((next_V - V_th) / denom)
            * self.spk_fun((V_th - last_V) / denom)
        )

# ---------------------------------------------------------------------------
# Re-exports (ARCH-03). The ion-channel family lives in
# ``braincell._base_channel``; ion species and the ``mix_ions`` factory
# live in ``braincell._base_ion``. Every existing caller using
# ``from braincell._base import X`` continues to work through these
# re-exports.
from ._base_channel import (  # noqa: E402
    Channel,
    IonChannel,
    IonInfo,
    Synapse,
)
from ._base_ion import (  # noqa: E402
    Ion,
    MixIons,
    mix_ions,
)
