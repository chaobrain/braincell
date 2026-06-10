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

"""Ion-species base classes.

Extracted from :mod:`braincell._base` during the ARCH-03 split. Houses
:class:`Ion`, :class:`MixIons`, and the :func:`mix_ions` factory.

``Ion.root_type`` / ``MixIons.root_type`` point at
:class:`braincell._base.HHTypedNeuron`; that import happens at module
load time but succeeds because ``_base`` defines ``HHTypedNeuron``
before executing its own bottom-of-file ``from ._base_ion import ...``
re-export line.
"""

from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple, Type

import brainstate
import brainunit as u
from brainstate.mixin import _JointGenericAlias

from ._base_channel import Channel, IonChannel, IonInfo
from ._misc import Container, set_module_as
from .quad.protocol import IndependentIntegration

__all__ = ["Ion", "MixIons", "mix_ions"]


def _channel_current_owner_specs(node):
    """Return current-owner specs declared by a channel.

    Parameters
    ----------
    node : Channel
        Channel instance whose current-owner declaration should be
        interpreted.

    Returns
    -------
    tuple of tuple
        A tuple of ``(component_key, owner_type)`` pairs. A
        ``component_key`` of ``None`` denotes the legacy single-owner
        path where the owner current is the channel's ``current(...)``
        return value. Non-``None`` keys denote component names resolved
        through ``current_components(...)``.

    Notes
    -----
    Existing channels normally declare ``current_owner_type`` and are
    returned as a single legacy owner. Channels that write more than one
    ion current may declare ``current_owner_types`` as a mapping from
    component key to owner ion type. Those channels must also implement
    ``current_components(...)``.
    """
    owners = getattr(node, "current_owner_types", None)
    if owners is not None:
        return tuple((key, owner_type) for key, owner_type in owners.items())
    owner = getattr(node, "current_owner_type", None)
    if owner is not None:
        return ((None, owner),)
    return ()


def _channel_component_current(node, component_key, V, *infos):
    """Return total or component current for one channel.

    Parameters
    ----------
    node : Channel
        Channel instance being evaluated.
    component_key : str or None
        Component key to read from ``node.current_components(...)``.
        ``None`` selects the legacy total-current path and calls
        ``node.current(...)`` directly.
    V : array-like
        Membrane potential passed to the channel.
    *infos
        Ion information objects passed to the channel.

    Returns
    -------
    array-like
        Current density returned by the channel for the requested owner.

    Raises
    ------
    AttributeError
        If ``component_key`` is not ``None`` and the channel does not
        implement ``current_components(...)``.
    KeyError
        If the requested component key is absent from the returned
        component mapping.

    Notes
    -----
    ``current(...)`` remains the total membrane current API. Component
    lookup is used only when a channel explicitly declares multiple
    current owners through ``current_owner_types``.
    """
    if component_key is None:
        return node.current(V, *infos)
    components = node.current_components(V, *infos)
    return components[component_key]


def _mask_inactive_current(current, point_mask):
    """Zero inactive points in a channel current.

    Parameters
    ----------
    current : array-like
        Current density returned by a channel. May be a
        :class:`brainunit.Quantity`.
    point_mask : array-like of bool
        Boolean mask whose ``True`` entries mark active runtime points.

    Returns
    -------
    array-like
        Current with inactive points replaced by zero while preserving
        units when ``current`` is a :class:`brainunit.Quantity`.

    Notes
    -----
    Dense runtime layouts store full point-shaped state and use masks to
    disable points outside the painted region. This helper keeps inactive
    points from contributing to ion-current totals.
    """
    if isinstance(current, u.Quantity):
        unit = current.unit
        mantissa = u.math.asarray(current.to_decimal(unit))
        return u.Quantity(u.math.where(point_mask, mantissa, 0.0), unit)
    return u.math.where(point_mask, current, 0.0)


def _safe_inactive_voltage(V, point_mask):
    """Replace inactive-point voltages with a benign value.

    Parameters
    ----------
    V : array-like
        Membrane potential passed to a channel. May be a
        :class:`brainunit.Quantity`.
    point_mask : array-like of bool
        Boolean mask whose ``True`` entries mark active runtime points.

    Returns
    -------
    array-like
        Voltage with inactive entries replaced by ``-65 mV`` while
        preserving units when ``V`` is a :class:`brainunit.Quantity`.

    Notes
    -----
    The returned value is used only to evaluate channel formulas at
    inactive points before their current is masked to zero. It prevents
    inactive points with arbitrary voltages from producing numerical
    overflow in voltage-dependent rate expressions.
    """
    if isinstance(V, u.Quantity):
        unit = V.unit
        mantissa = u.math.asarray(V.to_decimal(unit))
        safe = (-65.0 * u.mV).to_decimal(unit)
        return u.Quantity(u.math.where(point_mask, mantissa, safe), unit)
    return u.math.where(point_mask, V, -65.0)


class Ion(IonChannel, Container):
    """
    The base class for modeling ion dynamics in neuronal simulations.

    This class represents a specific type of ion (e.g., sodium, potassium) and manages
    the associated ion channels and their dynamics. It inherits from both IonChannel
    and Container, allowing it to handle ion-specific behaviors and contain multiple
    channel instances.

    The Ion class serves as a crucial component in modeling the behavior of specific
    ion types within a neuron or neural network simulation. It manages the collective
    behavior of multiple ion channels of the same ion type and provides methods for
    initializing, updating, and querying the state of these channels throughout the
    simulation process.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target, typically representing the number of
        neurons or compartments.
    name : Optional[str], default=None
        The name of the Ion instance. If not provided, the instance will be unnamed.
    channels
        Additional keyword arguments (``**channels``) specifying Channel instances
        to be included in this Ion object.

    Attributes
    ----------
    channels : Dict[str, Channel]
        A dictionary of Channel instances associated with this ion.
    """

    __module__ = 'braincell'
    _container_name = 'channels'

    # root_type is assigned below after the import-cycle with HHTypedNeuron
    # is safe to resolve. See module docstring.

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        **channels
    ) -> None:
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
        self._run_ion_hook("_ion_pre_integral_hook", V)
        if getattr(self, "_ion_channel_update_phase", None) == "ion":
            return
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(V, self.pack_info())

    def compute_derivative(self, V):
        """
        Compute derivatives for all channels.

        This method calculates the derivatives of state variables for all Channel nodes.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        self.check_hierarchies(type(self), *nodes)
        ion_info = self.pack_info()
        if getattr(self, "_ion_channel_update_phase", None) != "ion":
            for node in nodes:
                if not isinstance(node, IndependentIntegration):
                    node.compute_derivative(V, ion_info)
        self._run_ion_hook("_ion_compute_derivative_hook", V)

    def post_integral(self, V):
        """
        Perform post-integration operations for all channels.

        This method is called after the integration step in simulations. It iterates through
        all Channel nodes and calls their post_integral methods.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
        """
        if getattr(self, "_ion_channel_update_phase", None) != "ion":
            nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
            for node in nodes.values():
                if not isinstance(node, IndependentIntegration):
                    node.post_integral(V, self.pack_info())
        self._run_ion_hook("_ion_post_integral_hook", V)

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
                point_mask = getattr(node, "_point_mask", None)
                node_V = _safe_inactive_voltage(V, point_mask) if point_mask is not None else V
                new_current = node.current(node_V, ion_info)
                if point_mask is not None:
                    new_current = _mask_inactive_current(new_current, point_mask)
                current = new_current if current is None else (current + new_current)
        if include_external and self._external_currents:
            for key, node in self._external_currents.items():
                node: Callable
                contrib = node(V, ion_info)
                current = contrib if current is None else (current + contrib)
        return current

    def init_state(self, V, batch_size: int = None):
        """
        Initialize the state of all channels.

        This method initializes the state variables for all Channel nodes.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            batch_size (int, optional): The batch size for initialization. Default is None.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        self.check_hierarchies(type(self), *nodes)
        self._run_ion_hook("_ion_init_state_hook", V, batch_size=batch_size)
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.init_state(V, ion_info, batch_size=batch_size)

    def reset_state(self, V, batch_size: int = None):
        """
        Reset the state of all channels.

        This method resets the state variables for all Channel nodes to their initial values.

        Parameters:
            V (array-like): The membrane potential for all neurons/compartments.
            batch_size (int, optional): The batch size for resetting. Default is None.
        """
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        self.check_hierarchies(type(self), *nodes)
        self._run_ion_hook("_ion_reset_state_hook", V, batch_size=batch_size)
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.reset_state(V, ion_info, batch_size=batch_size)

    def update(self, V, *args, **kwargs):
        if isinstance(self, IndependentIntegration):
            from braincell.quad import ind_exp_euler_step

            self.make_integration(V)
            ion_info = self.pack_info()
            for key, node in brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).items():
                if isinstance(node, IndependentIntegration):
                    node.update(V, ion_info)
                else:
                    ind_exp_euler_step(node, V, ion_info)
            return
        ion_info = self.pack_info()
        for key, node in brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).items():
            node.update(V, ion_info)

    def _update_ion_state_only(self, V):
        old_phase = getattr(self, "_ion_channel_update_phase", None)
        self._ion_channel_update_phase = "ion"
        try:
            if isinstance(self, IndependentIntegration):
                self.make_integration(V)
        finally:
            if old_phase is None:
                delattr(self, "_ion_channel_update_phase")
            else:
                self._ion_channel_update_phase = old_phase

    def _run_ion_hook(self, name: str, *args, **kwargs):
        hook = getattr(self, name, None)
        if hook is not None:
            hook(*args, **kwargs)

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

        This method collects the intracellular/extracellular concentrations
        (Ci/Co), reversal potential (E), and valence of the ion and packages
        them into an IonInfo named tuple.

        Returns:
            IonInfo: A named tuple containing:

                - Ci (array-like): The intracellular ion concentration.
                - Co (array-like): The extracellular ion concentration.
                - E (array-like): The reversal potential of the ion.
                - valence (array-like): The ionic valence.

        Notes:
            If an attribute is an instance of brainstate.State, its ``value``
            attribute is used. Otherwise, the raw attribute value is used.
        """
        Ci = self.Ci
        Ci = Ci.value if isinstance(Ci, brainstate.State) else Ci
        Co = self.Co
        Co = Co.value if isinstance(Co, brainstate.State) else Co
        E = self.E
        E = E.value if isinstance(E, brainstate.State) else E
        valence = self.valence
        valence = valence.value if isinstance(valence, brainstate.State) else valence
        return IonInfo(Ci=Ci, Co=Co, E=E, valence=valence)

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
        self.channels.update(self._format_elements(Channel, **elements))


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

    # root_type assigned below after import-cycle is safe to resolve.
    _container_name = 'channels'

    def __init__(self, *ions, name: Optional[str] = None, **channels):
        """See class docstring."""
        assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ion. '
        assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
        size = ions[0].size
        for ion in ions:
            assert ion.size == size, f'The size of all ion should be the same. But we got {ions}.'
        super().__init__(size=size, name=name)

        self.ions: Sequence['Ion'] = tuple(ions)
        self._ion_types = tuple([type(ion) for ion in self.ions])

        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

    @property
    def ion_types(self) -> Tuple[Type[Ion], ...]:
        """Types of ions in this mixed channel."""
        return self._ion_types

    def pre_integral(self, V):
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            if not isinstance(node, IndependentIntegration):
                ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
                node.pre_integral(V, *ion_infos)

    def compute_derivative(self, V):
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            if not isinstance(node, IndependentIntegration):
                ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
                node.compute_derivative(V, *ion_infos)

    def post_integral(self, V):
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            if not isinstance(node, IndependentIntegration):
                ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
                node.post_integral(V, *ion_infos)

    def current(self, V):
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        if len(nodes) == 0:
            return 0.
        else:
            current = None
            for node in nodes:
                infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
                point_mask = getattr(node, "_point_mask", None)
                node_V = _safe_inactive_voltage(V, point_mask) if point_mask is not None else V
                new_current = node.current(node_V, *infos)
                if point_mask is not None:
                    new_current = _mask_inactive_current(new_current, point_mask)
                current = new_current if current is None else (current + new_current)
            return current

    def init_state(self, V, batch_size: int = None):
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(self.ion_types, *tuple(nodes), check_fun=self._check_hierarchy)
        for node in nodes:
            node: Channel
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.init_state(V, *infos, batch_size=batch_size)

    def reset_state(self, V, batch_size=None):
        nodes = tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.reset_state(V, *infos, batch_size=batch_size)

    def update(self, V, *args, **kwargs):
        for key, node in brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).items():
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.update(V, *infos)

    def _check_hierarchy(self, ions, leaf):
        self._check_root(leaf)
        for cls in leaf.root_type.__args__:
            if not any([issubclass(root, cls) for root in ions]):
                raise TypeError(
                    f'Type does not match. {leaf} requires a master with type '
                    f'of {leaf.root_type}, but the master type now is {ions}.'
                )

    def add(self, **elements):
        self.check_hierarchies(self.ion_types, check_fun=self._check_hierarchy, **elements)
        self.channels.update(self._format_elements(Channel, **elements))
        for elem in tuple(elements.values()):
            elem: Channel
            owner_specs = _channel_current_owner_specs(elem)
            if not owner_specs:
                owner_specs = tuple((None, ion_root) for ion_root in elem.root_type.__args__)
            for component_key, ion_root in owner_specs:
                ion = self._get_ion(ion_root)
                key = id(elem) if component_key is None else (id(elem), component_key)
                ion.register_external_current(key, self._get_ion_fun(ion, elem, component_key=component_key))

    def _get_ion_fun(self, ion: 'Ion', node: 'Channel', *, component_key=None):
        """Build an ion-external-current callback for a mixed channel.

        Parameters
        ----------
        ion : Ion
            Ion instance that owns the callback.
        node : Channel
            Mixed-ion channel instance whose current is being exposed
            through the owner ion.
        component_key : str or None, optional
            Component key to retrieve from
            ``node.current_components(...)``. ``None`` keeps the
            legacy behavior and exposes ``node.current(...)``.

        Returns
        -------
        callable
            Function with signature ``fun(V, ion_info)`` suitable for
            :meth:`Ion.register_external_current`.

        Notes
        -----
        This wrapper is the compatibility boundary for multi-owner
        currents. The membrane solver still calls ``node.current(...)``
        once for total current, while owner ions may receive individual
        components when ``component_key`` is set.
        """
        def fun(V, ion_info):
            infos = tuple([
                (ion_info if isinstance(ion, root) else self._get_ion(root).pack_info())
                for root in node.root_type.__args__
            ])
            return _channel_component_current(node, component_key, V, *infos)

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
    *ions
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
    assert len(ions) >= 2, f'mix_ions requires at least two ions, got {len(ions)}.'
    return MixIons(*ions)


# Late-bound root_type assignment: HHTypedNeuron lives in braincell._base
# and we defer its import until both Ion and MixIons are defined. This
# ordering guarantees that when ``_base`` imports Ion/MixIons back via its
# bottom-of-file re-export, HHTypedNeuron is already in the _base namespace.
from ._base import HHTypedNeuron  # noqa: E402

Ion.root_type = HHTypedNeuron
MixIons.root_type = HHTypedNeuron
