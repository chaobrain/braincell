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

import numpy as np

import brainstate
from brainstate.mixin import _JointGenericAlias
from ._misc import set_module_as, Container, TreeNode
from ._protocol import DiffEqModule

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


class HHTypedNeuron(brainstate.nn.Dynamics, Container, DiffEqModule):
    """
    The base class for the Hodgkin-Huxley typed neuronal membrane dynamics.
    """
    __module__ = 'braincell'
    _container_name = 'ion_channels'

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        **ion_channels
    ):
        """
        Initialize a new instance of HHTypedNeuron.

        This function sets up the basic structure of a Hodgkin-Huxley typed neuron,
        including its size, name, and ion channels.

        Parameters
        ----------
        size : brainstate.typing.Size
            The size of the simulation target. Can be an integer or a tuple of integers.
            Must be at least 1-dimensional, representing (..., n_neuron, n_compartment).

        name : Optional[str]
            The name of the HHTypedNeuron instance. If not provided, a default name will be used.

        **ion_channels : dict
            A dictionary of ion channel instances to be added to the neuron.
            Each key-value pair represents an ion channel name and its corresponding instance.

        Raises
        ------
        ValueError
            If the size parameter is not correctly formatted (must be int or tuple/list of int).

        AssertionError
            If the size is less than 1-dimensional.
        """
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
        *args : tuple
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs : dict
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
        *args : tuple
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs : dict
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
        *args : tuple
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs : dict
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Returns
        -------
        None
            This method does not return a value directly. Instead, it should
            update the internal state of the object to reflect the computed
            derivatives.

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
        *args : tuple
            Variable length argument list. Specific arguments should be defined
            in the subclass implementation.

        **kwargs : dict
            Arbitrary keyword arguments. Specific keyword arguments should be
            defined in the subclass implementation.

        Returns
        -------
        None
            This method does not return a value. It performs in-place updates
            on the neuron's state variables.

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

        Returns
        -------
        None
            This method doesn't return any value. It performs in-place initialization
            of the state variables for all ion channels.

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

        Returns
        -------
        None
            This method doesn't return any value. It performs in-place reset of
            the state variables for all ion channels in the neuron group.

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

        Returns
        -------
        None
            This method doesn't return a value. It updates the ion_channels attribute in-place.

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


class IonChannel(brainstate.graph.Node, TreeNode, DiffEqModule):
    """
    Base class for modeling ion channel dynamics in neuronal simulations.

    The IonChannel class serves as a foundation for implementing various types of ion channels,
    including those for specific ions (e.g., sodium, potassium) or mixtures of ions. It provides
    a structure for defining the behavior and properties of ion channels within a neuron model.

    This class is designed to be subclassed to create specific ion channel models. Subclasses
    should implement the core methods to define the channel's behavior, such as current calculation,
    state initialization, and derivative computation.

    Attributes
    ----------
    in_size : tuple
        The dimensions of the ion channel, representing its size (e.g., number of neurons,
        number of compartments).
    out_size : tuple
        Same as in_size, representing the output dimensions of the channel.
    name : str, optional
        A name identifier for the ion channel.

    Notes
    -----
    - Subclasses should override the abstract methods (current, compute_derivative, init_state,
      reset_state) to define the specific behavior of the ion channel.
    - The class integrates with the broader neuron modeling framework, allowing for complex
      simulations of neuronal dynamics.
    - It's designed to work within a hierarchical structure of neuronal components, as indicated
      by its inheritance from TreeNode.

    Example
    -------

    .. code-block:: python

        class SodiumChannel(IonChannel):
            def __init__(self, size, g_max):
                super().__init__(size)
                self.g_max = g_max

            def current(self, V, Na):
                # Implement sodium current calculation
                pass

            def compute_derivative(self, V, Na):
                # Implement derivative computation for channel states
                pass

            def init_state(self, V, Na, batch_size=None):
                # Initialize channel states
                pass

            def reset_state(self, V, Na, batch_size=None):
                # Reset channel states
                pass
    """

    __module__ = 'braincell'

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
    ):
        """
        Initialize an IonChannel instance.

        This method sets up the basic properties of an ion channel, including its size and name.
        It performs type checking and formatting on the size parameter to ensure it meets
        the required specifications.

        Parameters
        ----------
        size : brainstate.typing.Size
            The size of the ion channel. This can be an integer, or a tuple/list of integers.
            It should represent the dimensions of the channel, with at least one dimension
            (e.g., number of neurons, number of compartments).

        name : Optional[str], default=None
            The name of the ion channel. If not provided, the channel will be unnamed.

        Raises
        ------
        ValueError
            If the size is not an integer or a non-empty tuple/list of integers.
        AssertionError
            If the size has less than one dimension.

        Notes
        -----
        - The size is stored as a tuple, even if a single integer is provided.
        - The size must have at least one dimension to represent the number of neurons
          or compartments.
        """
        # size
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise ValueError(f'size must be int, or a tuple/list of int. '
                                 f'But we got {type(size)}')
            if not isinstance(size[0], (int, np.integer)):
                raise ValueError('size must be int, or a tuple/list of int.'
                                 f'But we got {type(size)}')
            size = tuple(size)
        elif isinstance(size, (int, np.integer)):
            size = (size,)
        else:
            raise ValueError('size must be int, or a tuple/list of int.'
                             f'But we got {type(size)}')
        self.size = size
        assert len(size) >= 1, ('The size of the dendritic dynamics should be at '
                                'least 1D: (..., n_neuron, n_compartment).')
        self.name = name

    @property
    def varshape(self):
        """
        Get the shape of variables in the neuron group.

        Returns
        -------
        tuple
            The shape of variables, typically representing the dimensions of the neuron group.
        """
        return self.size

    def current(self, *args, **kwargs):
        """
        Calculate the current for this ion channel.

        This method should be implemented by subclasses to compute the current
        based on the channel's specific properties and state.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def pre_integral(self, *args, **kwargs):
        """
        Perform pre-integration operations.

        This method is called before the integration step in simulations.
        It can be used to prepare the channel's state or perform any necessary
        calculations before integration.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass

    def update_state(self, *args, **kwargs):

        pass

    def compute_derivative(self, *args, **kwargs):
        """
        Compute the derivative of the channel's state variables.

        This method should be implemented by subclasses to calculate how the
        channel's state changes over time.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def post_integral(self, *args, **kwargs):
        """
        Perform post-integration operations.

        This method is called after the integration step in simulations.
        It should be used to update the channel's state based on the results of integration.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        pass

    def reset_state(self, *args, **kwargs):
        """
        Reset the state of the ion channel.

        This method should reset all state variables of the channel to their initial values.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass

    def init_state(self, *args, **kwargs):
        """
        Initialize the state of the ion channel.

        This method should set up the initial state of all variables for the channel.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass


class IonInfo(NamedTuple):
    """
    A named tuple representing the information of an ion in a neuron model.

    This class encapsulates two key properties of an ion: its concentration
    and its reversal potential. It is used to store and pass ion-related
    information in various neuronal simulation contexts.

    Attributes:
        C (brainstate.typing.ArrayLike): The ion concentration.
            This represents the concentration of the ion, typically
            in units of millimoles per liter (mM).

        E (brainstate.typing.ArrayLike): The reversal potential.
            This represents the electrical potential at which there is no net
            flow of the ion across the membrane, typically in millivolts (mV).

    Note:
        Both C and E are expected to be array-like objects, allowing for
        representation of these properties across multiple neurons or
        compartments simultaneously.
    """
    C: brainstate.typing.ArrayLike
    E: brainstate.typing.ArrayLike


class Ion(IonChannel, Container):
    """
    The base class for modeling ion dynamics in neuronal simulations.

    This class represents a specific type of ion (e.g., sodium, potassium) and manages
    the associated ion channels and their dynamics. It inherits from both IonChannel
    and Container, allowing it to handle ion-specific behaviors and contain multiple
    channel instances.

    Args:
        size (brainstate.typing.Size): The size of the simulation target, typically
            representing the number of neurons or compartments.
        name (Optional[str]): The name of the Ion instance. Defaults to None.
        **channels: Additional keyword arguments for specifying Channel instances
            to be included in this Ion object.

    Attributes:
        channels (Dict[str, Channel]): A dictionary of Channel instances associated
            with this ion.
        external_currents (Dict[str, Callable]): A dictionary of external current
            functions that can be applied to this ion.

    The Ion class serves as a crucial component in modeling the behavior of specific
    ion types within a neuron or neural network simulation. It manages the collective
    behavior of multiple ion channels of the same ion type and provides methods for
    initializing, updating, and querying the state of these channels throughout the
    simulation process.
    """

    __module__ = 'braincell'
    _container_name = 'channels'

    # The type of the master object.
    root_type = HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        **channels
    ) -> None:
        """
        Initialize an Ion instance.

        This constructor sets up an Ion object with specified size and name,
        and initializes its channels and external currents.

        Parameters:
        -----------
        size : brainstate.typing.Size
            The size of the ion channel, typically representing the number of
            neurons or compartments.
        name : Optional[str], default=None
            The name of the Ion instance. If not provided, the instance will be unnamed.
        **channels : dict
            Additional keyword arguments for specifying Channel instances to be
            included in this Ion object.
        """
        super().__init__(size, name=name)
        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

        self._external_currents: Dict[str, Callable] = dict()

    @property
    def external_currents(self) -> Dict[str, Callable]:
        """
        Get the dictionary of external currents.

        Returns:
            Dict[str, Callable]: A dictionary where keys are strings identifying the external currents,
                                 and values are callable functions representing those currents.
        """
        return self._external_currents

    def pre_integral(self, V):
        """
        Perform pre-integration operations for all channels.

        This method is called before the integration step in simulations. It iterates through
        all Channel nodes and calls their pre_integral methods.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.pre_integral(V, self.pack_info())
    
    def update_state(self, V):
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.update_state(V, self.pack_info())

    def compute_derivative(self, V):
        """
        Compute derivatives for all channels.

        This method calculates the derivatives of state variables for all Channel nodes.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.compute_derivative(V, self.pack_info())

    def post_integral(self, V):
        """
        Perform post-integration operations for all channels.

        This method is called after the integration step in simulations. It iterates through
        all Channel nodes and calls their post_integral methods.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.post_integral(V, self.pack_info())

    def current(self, V, include_external: bool = False):
        """
        Generate ion channel current.

        This method calculates the total current from all channels and optionally includes external currents.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            include_external (bool): If True, include external currents in the calculation. Default is False.

        Returns:
            array-like: The total current generated by all channels (and external currents if included).
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        ion_info = self.pack_info()
        current = None
        if len(nodes) > 0:
            for node in nodes:
                node: Channel
                new_current = node.current(V, ion_info)
                current = new_current if current is None else (current + new_current)
        if include_external:
            for key, node in self._external_currents.items():
                node: Callable
                current = current + node(V, ion_info)
        return current

    def init_state(self, V, batch_size: int = None):
        """
        Initialize the state of all channels.

        This method initializes the state variables for all Channel nodes.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            batch_size (int, optional): The batch size for initialization. Default is None.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(type(self), *tuple(nodes))
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.init_state(V, ion_info, batch_size)

    def reset_state(self, V, batch_size: int = None):
        """
        Reset the state of all channels.

        This method resets the state variables for all Channel nodes to their initial values.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            batch_size (int, optional): The batch size for resetting. Default is None.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.reset_state(V, ion_info, batch_size)

    def register_external_current(self, key: Hashable, fun: Callable):
        """
        Register an external current function.

        This method adds a new external current function to the ion channel.

        Parameters:
            key (Hashable): A unique identifier for the external current.
            fun (Callable): The function that computes the external current.

        Raises:
            ValueError: If the key already exists in the external currents' dictionary.
        """
        if key in self._external_currents:
            raise ValueError
        self._external_currents[key] = fun

    def pack_info(self) -> IonInfo:
        """
        Pack the ion information into an IonInfo object.

        This method collects the reversal potential (E) and concentration (C) of the ion
        and packages them into an IonInfo named tuple.

        Returns:
            IonInfo: A named tuple containing:
                - E (array-like): The reversal potential of the ion.
                - C (array-like): The concentration of the ion.

        Notes:
            If E or C are instances of brainstate.State, their 'value' attribute is used.
            Otherwise, the E and C values are used directly.
        """
        E = self.E
        E = E.value if isinstance(E, brainstate.State) else E
        C = self.C.value if isinstance(self.C, brainstate.State) else self.C
        return IonInfo(E=E, C=C)

    def add(self, **elements):
        """
        Add new channel elements to the Ion instance.

        This method adds new Channel instances to the Ion object. It checks the
        hierarchies of the new elements and updates the channels dictionary.

        Parameters
        ----------
        **elements : Any
            A dictionary of new elements to add. Each key-value pair represents a
            channel name and its corresponding Channel instance.

        Raises
        ------
        TypeError
            If the hierarchies of the new elements are incompatible with the current structure.

        Notes
        -----
        - The method checks hierarchies using the check_hierarchies method.
        - New elements are formatted and added to the channels dictionary.
        """
        self.check_hierarchies(type(self), **elements)
        self.channels.update(self._format_elements(object, **elements))


class MixIons(IonChannel, Container):
    """
    A class for mixing multiple ion channels in neuronal simulations.

    This class combines multiple Ion instances to create a composite ion channel
    that can handle the dynamics of multiple ion types simultaneously.

    Args:
        *ions: Variable number of Ion instances. These define the types of ions
               that will be mixed in this channel.
        name (Optional[str]): The name of the MixIons instance. Defaults to None.
        **channels: Additional keyword arguments for specifying Channel instances.

    Attributes:
        ions (Sequence['Ion']): A tuple of Ion instances that are part of this mixed channel.
        ion_types (tuple): A tuple of the types of the Ion instances.
        channels (Dict[str, Channel]): A dictionary of Channel instances associated with this mixed channel.

    Raises:
        AssertionError: If fewer than two ions are provided, if any provided ion is not an Ion instance,
                        or if the sizes of all provided ions are not identical.
    """
    __module__ = 'braincell'

    root_type = HHTypedNeuron
    _container_name = 'channels'

    def __init__(self, *ions, name: Optional[str] = None, **channels):
        """
        Initialize a MixIons instance for combining multiple ion channels.

        This constructor creates a MixIons object that combines multiple Ion instances
        to model the behavior of a composite ion channel. It ensures that at least two
        ions are provided and that all ions have consistent sizes.

        Parameters
        ----------
        *ions : Ion
            Variable number of Ion instances to be combined in this mixed channel.
            At least two ions must be provided.

        name : Optional[str], default=None
            The name of this MixIons instance. If not provided, the instance will be unnamed.

        **channels : dict
            Additional keyword arguments for specifying Channel instances to be included
            in this MixIons object.

        Raises
        ------
        AssertionError
            If fewer than two ions are provided, if any provided ion is not an Ion instance,
            or if the sizes of all provided ions are not identical.

        Notes
        -----
        - The size of the MixIons instance is set to the size of the first provided ion.
        - All provided ions must have the same size.
        - The method stores both the ion instances and their types.
        - Additional channels can be added via keyword arguments.
        """
        # TODO: check "ion" should be independent from each other
        assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ion. '
        assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
        size = ions[0].size
        for ion in ions:
            assert ion.size == size, f'The size of all ion should be the same. But we got {ions}.'
        super().__init__(size=size, name=name)

        # Store the ion instances
        self.ions: Sequence['Ion'] = tuple(ions)
        self._ion_types = tuple([type(ion) for ion in self.ions])

        # Store the ion channel channel
        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

    @property
    def ion_types(self) -> Tuple[Type[Ion], ...]:
        """
        Get the types of ions in this mixed channel.

        Returns:
            Tuple[Type[Ion], ...]: A tuple containing the types of all Ion instances in this mixed channel.
        """
        return self._ion_types

    def pre_integral(self, V):
        """
        Perform pre-integration operations for all channels in the mixed ion group.

        This method is called before the integration step in the simulation. It iterates
        through all Channel nodes in the MixIons instance and calls their pre_integral
        methods with the current membrane potential and relevant ion information.

        Parameters
        ----------
        V : array-like
            The current membrane potential for all neurons in the simulation.

        Returns
        -------
        None
            This method doesn't return a value. It updates the state of each channel
            in place as necessary before integration.

        Notes
        -----
        The method retrieves all Channel nodes, packs the necessary ion information
        for each channel, and then calls the pre_integral method of each channel
        with the current membrane potential and the packed ion information.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.pre_integral(V, *ion_infos)

    def compute_derivative(self, V):
        """
        Compute the derivative of state variables for all channels in the mixed ion group.

        This method iterates through all Channel nodes in the MixIons instance and calls
        their compute_derivative methods with the current membrane potential and relevant
        ion information. It's typically used in the process of numerically solving the
        differential equations that describe the ion channel dynamics.

        Parameters
        ----------
        V : array-like
            The current membrane potential for all neurons in the simulation.

        Returns
        -------
        None
            This method doesn't return a value. It updates the state derivatives of each
            channel in place.

        Notes:
        ------
        The method retrieves all Channel nodes, packs the necessary ion information
        for each channel, and then calls the compute_derivative method of each channel
        with the current membrane potential and the packed ion information.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.compute_derivative(V, *ion_infos)

    def post_integral(self, V):
        """
        Perform post-integration operations for all channels in the mixed ion group.

        This method is called after the integration step in the simulation. It iterates
        through all Channel nodes in the MixIons instance and calls their post_integral
        methods with the current membrane potential and relevant ion information.

        Parameters
        ----------
        V : array-like
            The current membrane potential for all neurons in the simulation.

        Returns
        -------
        None
            This method doesn't return a value. It updates the state of each channel
            in place as necessary after integration.

        Notes
        -----
        The method retrieves all Channel nodes, packs the necessary ion information
        for each channel, and then calls the post_integral method of each channel
        with the current membrane potential and the packed ion information.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.post_integral(V, *ion_infos)

    def current(self, V):
        """
        Generate the total ion channel current for all channels in the mixed ion group.

        This method calculates the sum of currents from all Channel nodes within the
        MixIons instance. It iterates through each channel, computes its individual
        current, and aggregates them.

        Parameters
        ----------
        V : array-like
            The membrane potential for all neurons in the simulation. This is typically
            a numpy array or similar structure containing voltage values.

        Returns
        -------
        array-like or float
            The total current generated by all channels. If there are no channels,
            it returns 0. Otherwise, it returns an array-like object of the same
            shape as V, representing the summed current at each point.

        Notes
        -----
        - The method retrieves all Channel nodes and their corresponding ion information.
        - For each channel, it calls the channel's current method with the membrane
          potential and relevant ion information.
        - The currents from all channels are summed to produce the total current.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        if len(nodes) == 0:
            return 0.
        else:
            current = None
            for node in nodes:
                infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
                current = node.current(V, *infos) if current is None else (current + node.current(V, *infos))
            return current

    def init_state(self, V, batch_size: int = None):
        """
        Initialize the state of all channels in the mixed ion group.

        This method iterates through all Channel nodes in the MixIons instance and initializes
        their states. It checks the hierarchies of the nodes and packs the necessary ion
        information for each channel before calling their init_state methods.

        Parameters
        ----------
        V : array-like
            The initial membrane potential for all neurons in the simulation.
        batch_size : int, optional
            The batch size for the simulation. If provided, it will be used to initialize
            the states with the specified batch size.

        Returns
        -------
        None
            This method doesn't return a value. It initializes the state of each channel in place.

        Notes
        -----
        The method retrieves all Channel nodes, checks their hierarchies, packs the necessary
        ion information for each channel, and then calls the init_state method of each channel
        with the initial membrane potential, ion information, and batch size.
        """
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(self.ion_types, *tuple(nodes), check_fun=self._check_hierarchy)
        for node in nodes:
            node: Channel
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.init_state(V, *infos, batch_size)

    def reset_state(self, V, batch_size=None):
        """
        Reset the state of all channels in the mixed ion group.

        This method iterates through all Channel nodes in the MixIons instance and resets
        their states to initial values. It's typically used when restarting a simulation
        or when transitioning between different phases of a simulation.

        Parameters
        ----------
        V : array-like
            The current membrane potential for all neurons in the simulation.
            This is used to reset the channel states based on the current voltage.

        batch_size : int, optional
            The batch size for the simulation. If provided, it will be used to reset
            the states with the specified batch size, ensuring consistency in batched simulations.

        Returns
        -------
        None
            This method doesn't return a value. It resets the state of each channel in place.

        Notes
        -----
        The method retrieves all Channel nodes, packs the necessary ion information
        for each channel, and then calls the reset_state method of each channel
        with the current membrane potential, ion information, and batch size.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.reset_state(V, *infos, batch_size)

    def _check_hierarchy(self, ions, leaf):
        # 'root_type' should be a brainpy.mixin.JointType
        self._check_root(leaf)
        for cls in leaf.root_type.__args__:
            if not any([issubclass(root, cls) for root in ions]):
                raise TypeError(
                    f'Type does not match. {leaf} requires a master with type '
                    f'of {leaf.root_type}, but the master type now is {ions}.'
                )

    def add(self, **elements):
        """
        Add new channel elements to the MixIons instance.

        This method adds new Channel instances to the MixIons object. It checks the
        hierarchies of the new elements, updates the channels dictionary, and registers
        external currents for each new channel with its corresponding ion.

        Parameters
        ----------
        **elements : Any
            A dictionary of new elements to add. Each key-value pair represents a
            channel name and its corresponding Channel instance.

        Returns
        -------
        None
            This method doesn't return a value. It updates the MixIons instance in-place.

        Raises
        ------
        TypeError
            If the hierarchies of the new elements are incompatible with the current structure.

        Notes
        -----
        - The method checks hierarchies using the _check_hierarchy method.
        - It updates the channels dictionary with formatted Channel elements.
        - For each new channel, it registers an external current with each associated ion.
        """
        self.check_hierarchies(self.ion_types, check_fun=self._check_hierarchy, **elements)
        self.channels.update(self._format_elements(Channel, **elements))
        for elem in tuple(elements.values()):
            elem: Channel
            for ion_root in elem.root_type.__args__:
                ion = self._get_ion(ion_root)
                ion.register_external_current(id(elem), self._get_ion_fun(ion, elem))

    def _get_ion_fun(self, ion: 'Ion', node: 'Channel'):
        def fun(V, ion_info):
            infos = tuple(
                [(ion_info if isinstance(ion, root) else self._get_ion(root).pack_info())
                 for root in node.root_type.__args__]
            )
            return node.current(V, *infos)

        return fun

    def _get_ion(self, cls):
        for ion in self.ions:
            if isinstance(ion, cls):
                return ion
        else:
            raise ValueError(f'No instance of {cls} is found.')

    def _check_root(self, leaf):
        if not isinstance(leaf.root_type, _JointGenericAlias):
            raise TypeError(
                f'{self.__class__.__name__} requires leaf nodes that have the root_type of '
                f'"brainpy.mixin.JointType". However, we got {leaf.root_type}'
            )


@set_module_as('braincell')
def mix_ions(*ions) -> MixIons:
    """
    Create a mixed ion channel by combining multiple ion instances.

    This function takes one or more Ion instances and creates a MixIons object,
    which represents a channel that can handle multiple types of ions simultaneously.

    Parameters
    ----------
    *ions : Ion
        One or more instances of the Ion class. Each instance represents a specific
        type of ion (e.g., sodium, potassium, calcium) that will be part of the
        mixed ion channel.

    Returns
    -------
    MixIons
        An instance of the MixIons class that combines all the provided ion instances
        into a single mixed ion channel.

    Raises
    ------
    AssertionError
        If no ions are provided or if any of the provided arguments is not an instance
        of the Ion class.

    Examples:
    ---------
    >>> import braincell    
    >>> sodium_ion = braincell.ion.SodiumFixed(...)
    >>> potassium_ion = braincell.ion.PotassiumFixed(...)
    >>> mixed_channel = mix_ions(sodium_ion, potassium_ion)
    """
    for ion in ions:
        assert isinstance(ion, Ion), f'Must be instance of {Ion.__name__}. But got {type(ion)}'
    assert len(ions) > 0, ''
    return MixIons(*ions)


class Channel(IonChannel):
    """
    The base class for modeling channel dynamics in neuronal simulations.

    This class extends the IonChannel class to provide a framework for implementing
    specific ion channel models. It serves as a foundation for creating various types
    of ion channels, such as voltage-gated or ligand-gated channels.

    Note:
        Subclasses of Channel should implement specific methods like `current`,
        `compute_derivative`, etc., to define the behavior of the particular channel type.

    Example::

    .. code-block:: python

        class SodiumChannel(Channel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Additional initialization for sodium channel

            def current(self, V, *args):
                # Implement sodium current calculation
                pass

            # Implement other required methods
    """
    __module__ = 'braincell'


class Synapse(IonChannel):
    __module__ = 'braincell'
