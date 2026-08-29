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

"""Ion-channel family base classes.

Extracted from :mod:`braincell._base` during the ARCH-03 split. Houses
the abstract :class:`IonChannel`, its direct subclasses :class:`Channel`
and :class:`Synapse`, and the :class:`IonInfo` named-tuple used by
:meth:`braincell._base_ion.Ion.pack_info`. The public import path
``from braincell._base import ...`` continues to resolve via
re-exports in :mod:`braincell._base`.
"""

from typing import Mapping, NamedTuple, Optional

import brainstate
import braintools
import numpy as np
import brainunit as u

from braincell._typing import ArrayLike, Size
from ._misc import TreeNode
from .event import NoEventInput
from .quad.protocol import DiffEqModule, IndependentIntegration
from ._parameter_schema import DerivedSpec, ParameterSpec, RuntimeParameterState, StateSpec

__all__ = ["IonChannel", "IonInfo", "Channel", "Synapse"]


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
        size: Size,
        name: Optional[str] = None,
    ):
        # size
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise ValueError(f'size must be int, or a tuple/list of int. But we got {type(size)}')
            if not isinstance(size[0], (int, np.integer)):
                raise ValueError(f'size must be int, or a tuple/list of int.But we got {type(size)}')
            size = tuple(size)
        elif isinstance(size, (int, np.integer)):
            size = (size,)
        else:
            raise ValueError(f'size must be int, or a tuple/list of int.But we got {type(size)}')
        self.size = size
        assert len(size) >= 1, (
            'The size of the dendritic dynamics should be at least 1D: (..., n_neuron, n_compartment).'
        )
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
        *args
            Variable length argument list.
        **kwargs
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
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

    def compute_derivative(self, *args, **kwargs):
        """
        Compute the derivative of the channel's state variables.

        This method should be implemented by subclasses to calculate how the
        channel's state changes over time.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
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
        *args
            Variable length argument list.
        **kwargs
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
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

    def init_state(self, *args, **kwargs):
        """
        Initialize the state of the ion channel.

        This method should set up the initial state of all variables for the channel.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

    def ind_update(self, *args, **kwargs):
        if isinstance(self, IndependentIntegration):
            self.make_integration(*args, **kwargs)

    def _on_param_updated(self, var_name: str, new_value) -> None:
        """Hook invoked after runtime state writes a parameter.

        Default: no-op. Subclasses override to recompute derived
        values when a specific parameter changes (for example, channels
        can recompute temperature-derived factors when ``temp`` changes).

        Parameters
        ----------
        var_name : str
            The parameter name that was just updated.
        new_value : object
            The new value written to the runtime node attribute.
            Typically a :class:`brainunit.Quantity` but may be a plain
            array.
        """
        return None


class IonInfo(NamedTuple):
    """
    A named tuple representing the information of an ion in a neuron model.

    This class encapsulates the intracellular/extracellular concentrations of
    an ion, its reversal potential, and its valence. It is used to store and
    pass ion-related information in various neuronal simulation contexts.

    Attributes:
        Ci (brainstate.typing.ArrayLike): The intracellular ion concentration.
            This represents the concentration of the ion inside the cell,
            typically in units of millimoles per liter (mM).

        Co (brainstate.typing.ArrayLike): The extracellular ion concentration.
            This represents the concentration of the ion outside the cell,
            typically in units of millimoles per liter (mM).

        E (brainstate.typing.ArrayLike): The reversal potential.
            This represents the electrical potential at which there is no net
            flow of the ion across the membrane, typically in millivolts (mV).

        valence (brainstate.typing.ArrayLike): The ionic valence.
            This represents the charge number used in Nernst/GHK relations.

    Note:
        ``Ci``, ``Co``, ``E``, and ``valence`` are expected to be array-like
        objects or scalars, allowing representation of these properties across
        multiple neurons or compartments simultaneously.
    """

    Ci: ArrayLike
    Co: ArrayLike
    E: ArrayLike
    valence: ArrayLike


class Channel(IonChannel):
    """
    The base class for modeling channel dynamics in neuronal simulations.

    This class extends the IonChannel class to provide a framework for implementing
    specific ion channel models. It serves as a foundation for creating various types
    of ion channels, such as voltage-gated or ligand-gated channels.

    Note:
        Subclasses of Channel should implement specific methods like `current`,
        `compute_derivative`, etc., to define the behavior of the particular channel type.

    Examples
    --------

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

    parameters: Mapping[str, ParameterSpec] = {}
    states: Mapping[str, StateSpec] = {}
    derived: Mapping[str, DerivedSpec] = {}

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        return value.dense_value(masked=True) if isinstance(value, RuntimeParameterState) else value

    def __setattr__(self, name: str, value) -> None:
        current = vars(self).get(name)
        if isinstance(current, RuntimeParameterState) and not isinstance(value, RuntimeParameterState):
            current.value = value
            return
        super().__setattr__(name, value)


class Synapse(IonChannel):
    """Base class for vectorized runtime point-synapse mechanisms.

    Subclasses declare physical parameters, differential states, and public
    derived values through explicit schemas. ``current()`` always returns an
    inward-positive total point current. Discrete input is passed directly to
    :meth:`apply_events`; event buffers are owned by runtime routing.
    """

    __module__ = 'braincell'

    parameters: Mapping[str, ParameterSpec] = {}
    states: Mapping[str, StateSpec] = {}
    derived: Mapping[str, DerivedSpec] = {}
    event_input = NoEventInput()

    def __init__(self, size: brainstate.typing.Size, name: Optional[str] = None, **parameters):
        super().__init__(size=size, name=name)
        unknown = tuple(sorted(set(parameters).difference(self.parameters)))
        if unknown:
            raise TypeError(f"Unknown {type(self).__name__} parameters: {unknown!r}.")
        for field, spec in self.parameters.items():
            value = parameters.get(field, spec.default)
            value = braintools.init.param(value, self.varshape, allow_none=False)
            spec.validate(value, field)
            setattr(self, field, value)
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate relations involving more than one parameter."""
        self.validate_parameter_values({field: getattr(self, field) for field in self.parameters})

    @classmethod
    def validate_parameter_values(cls, parameters: Mapping[str, object]) -> None:
        """Validate cross-field physical invariants for canonical columns."""
        _ = parameters

    def init_state(self, V_post=None, batch_size=None):
        _ = V_post
        for field, spec in self.states.items():
            spec.validate(spec.initial, field)
            value = _broadcast_synapse_initial(spec.initial, self.varshape, batch_size=batch_size)
            setattr(self, field, _make_synapse_state(value))

    def reset_state(self, V_post=None, batch_size=None):
        _ = V_post
        for field, spec in self.states.items():
            value = _broadcast_synapse_initial(spec.initial, self.varshape, batch_size=batch_size)
            getattr(self, field).value = value

    def apply_events(self, payload, V_post=None):
        """Apply one already-aggregated event payload vector."""
        _ = payload, V_post


def _make_synapse_state(value):
    # Logical synapses are a packed SoA, not a Cell population/spatial grid.
    from .quad.protocol import DiffEqSingleState

    return DiffEqSingleState(value)


def _broadcast_synapse_initial(initial, shape, *, batch_size=None):
    target = ((int(batch_size),) + tuple(shape)) if batch_size is not None else tuple(shape)
    return u.math.full(target, initial)
