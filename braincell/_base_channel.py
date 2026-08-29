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

Houses the abstract :class:`IonChannel`, its direct subclasses
:class:`Channel` and :class:`Synapse`, and the :class:`IonInfo`
named-tuple used by :meth:`braincell._base_ion.Ion.pack_info`.

This is the bottom of the base-class layer: it imports neither
:mod:`braincell._base_ion` nor :mod:`braincell._base_neuron`, so the two
modules above it can name these classes with plain top-of-file imports.
The public import path is ``braincell`` itself, which re-exports all four.
"""

from typing import Mapping, NamedTuple, Optional

import brainstate
import braintools
import numpy as np
import brainunit as u

from braincell._typing import ArrayLike, Size
from braincell._parameter_schema import RuntimeParameterState
from ._misc import TreeNode
from .mech import NoEventInput, ParameterSpec, StateSpec
from .quad.protocol import DiffEqModule, DiffEqSingleState, IndependentIntegration

__all__ = ["IonChannel", "IonInfo", "Channel", "Synapse"]


def _normalize_size(size) -> tuple:
    """Normalize a channel ``size`` argument to a non-empty tuple of ints.

    Parameters
    ----------
    size : int or sequence of int
        Channel shape, ``(..., n_neuron, n_compartment)``.

    Returns
    -------
    tuple of int
        ``size`` as a tuple, with a bare int widened to a 1-tuple.

    Raises
    ------
    ValueError
        If ``size`` is not an int or a non-empty sequence of ints.

    Notes
    -----
    Every element is type-checked. The three inlined copies this replaced
    checked only ``size[0]``, so ``(4, "x")`` was accepted and failed much
    later with an unrelated message.
    """
    if isinstance(size, (int, np.integer)):
        return (int(size),)
    if isinstance(size, (list, tuple)) and len(size) > 0:
        if all(isinstance(item, (int, np.integer)) for item in size):
            return tuple(int(item) for item in size)
    raise ValueError(f'size must be int, or a non-empty tuple/list of int. But we got {size!r}')


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
    size : tuple of int
        The dimensions of the ion channel, ``(..., n_neuron, n_compartment)``.
    varshape : tuple of int
        ``size`` without its leading batch dimension, the shape channel
        state variables are allocated with.
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
        self.size = _normalize_size(size)
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
        """

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
    """Everything a channel is told about the ion it depends on.

    A channel never receives the :class:`~braincell.Ion` itself, only this
    snapshot, so it cannot reach back into the pool and mutate it.
    :meth:`braincell.Ion.pack_info` builds one per lifecycle call.

    Attributes
    ----------
    Ci : brainstate.typing.ArrayLike
        Intracellular ion concentration, conventionally in ``u.mM``.
    Co : brainstate.typing.ArrayLike
        Extracellular ion concentration, conventionally in ``u.mM``.
    E : brainstate.typing.ArrayLike
        Reversal potential -- the voltage at which the ion has no net flow
        across the membrane -- conventionally in ``u.mV``.
    valence : brainstate.typing.ArrayLike
        Charge number used in the Nernst and GHK relations.

    Notes
    -----
    Every field may be a scalar or an array shaped like the population, so
    one snapshot covers all neurons and compartments at once.
    """

    Ci: ArrayLike
    Co: ArrayLike
    E: ArrayLike
    valence: ArrayLike


class Channel(IonChannel):
    """Base class for a channel that draws its current from one ion pool.

    Subclasses implement :meth:`current` and, when they carry gating state,
    :meth:`compute_derivative`; every lifecycle method receives an
    :class:`IonInfo` for each ion named in the subclass's ``root_type``.

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

    Subclasses declare physical parameters and differential states through
    explicit schemas; any further public value is a plain property, resolved
    by name at the call site. ``current()`` always returns an inward-positive
    total point current. Discrete input is passed directly to
    :meth:`apply_events`; event buffers are owned by runtime routing.
    """

    __module__ = 'braincell'

    parameters: Mapping[str, ParameterSpec] = {}
    states: Mapping[str, StateSpec] = {}
    event_input = NoEventInput()

    def __init__(self, size: Size, name: Optional[str] = None, **parameters):
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
            # Logical synapses are a packed SoA, not a Cell population/spatial grid.
            setattr(self, field, DiffEqSingleState(value))

    def reset_state(self, V_post=None, batch_size=None):
        _ = V_post
        for field, spec in self.states.items():
            value = _broadcast_synapse_initial(spec.initial, self.varshape, batch_size=batch_size)
            getattr(self, field).value = value

    def apply_events(self, payload, V_post=None):
        """Apply one already-aggregated event payload vector."""
        _ = payload, V_post


def _broadcast_synapse_initial(initial, shape, *, batch_size=None):
    target = ((int(batch_size),) + tuple(shape)) if batch_size is not None else tuple(shape)
    return u.math.full(target, initial)
