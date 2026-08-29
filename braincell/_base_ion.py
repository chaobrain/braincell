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

Houses :class:`Ion`, :class:`MixIons`, and the :func:`mix_ions` factory.

``Ion.root_type`` / ``MixIons.root_type`` are plain class attributes
naming :class:`braincell._base_neuron.HHTypedNeuron`, which this module
imports at the top like any other dependency. That works because
:mod:`braincell._base_neuron` sits below this module and never imports
back into it. They used to be patched on after both class bodies, from
when the two modules were one and reached each other through
bottom-of-file imports.
"""

from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple, Type

import brainstate
import brainunit as u
import jax
from brainstate.mixin import _JointGenericAlias

from braincell._typing import Size
from ._base_channel import Channel, IonChannel, IonInfo
from ._base_neuron import HHTypedNeuron
from ._misc import (
    Container,
    profile_barrier_current as _profile_barrier_current,
    profiler_safe_name,
    set_module_as,
)
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
        return jax.named_call(
            node.current,
            name=_channel_call_name("braincell:ion_current:channel", node),
        )(V, *infos)
    components = jax.named_call(
        node.current_components,
        name=_channel_call_name("braincell:ion_current:components", node),
    )(V, *infos)
    return components[component_key]


def _channel_call_name(prefix: str, node) -> str:
    """Build a profiler-safe ``jax.named_call`` name for an ion child channel."""
    class_name = type(getattr(node, "_channel", node)).__name__
    return profiler_safe_name(f"{prefix}:{class_name}")


def _external_current_call_name(prefix: str, key) -> str:
    """Build a profiler-safe ``jax.named_call`` name for external ion current."""
    return profiler_safe_name(f"{prefix}:{key!s}")


#: Voltage substituted at inactive points before their current is masked to
#: zero. Any finite resting-like value works; it exists only to keep
#: voltage-dependent rate expressions from overflowing on unpainted points.
_INACTIVE_POINT_VOLTAGE = -65.0 * u.mV


def _where_active(value, point_mask, fill):
    """Replace inactive points in ``value`` with ``fill``, preserving units.

    Parameters
    ----------
    value : array-like
        Per-point value. May be a :class:`brainunit.Quantity`.
    point_mask : array-like of bool
        Boolean mask whose ``True`` entries mark active runtime points.
    fill : array-like
        Replacement for inactive points, in ``value``'s dimension.

    Returns
    -------
    array-like
        ``value`` with inactive entries replaced, carrying ``value``'s unit.
    """
    if isinstance(value, u.Quantity):
        unit = value.unit
        mantissa = u.math.asarray(value.to_decimal(unit))
        replacement = fill.to_decimal(unit) if isinstance(fill, u.Quantity) else fill
        return u.Quantity(u.math.where(point_mask, mantissa, replacement), unit)
    return u.math.where(point_mask, value, u.get_mantissa(fill))


def _mask_inactive_current(current, point_mask):
    """Zero inactive points in a channel current.

    Parameters
    ----------
    current : array-like
        Current density returned by a channel. May be a
        :class:`brainunit.Quantity`.
    point_mask : array-like of bool
        Legacy-named boolean mask whose ``True`` entries mark active density CV rows.

    Returns
    -------
    array-like
        Current with inactive points replaced by zero while preserving
        units when ``current`` is a :class:`brainunit.Quantity`.

    Notes
    -----
    Dense density layouts store full CV-shaped state and use masks to disable
    CVs outside the painted region. This helper keeps inactive rows from
    contributing to ion-current totals.
    """
    return _where_active(current, point_mask, 0.0)


def _safe_inactive_voltage(V, point_mask):
    """Replace inactive-point voltages with a benign value.

    Parameters
    ----------
    V : array-like
        Membrane potential passed to a channel. May be a
        :class:`brainunit.Quantity`.
    point_mask : array-like of bool
        Legacy-named boolean mask whose ``True`` entries mark active density CV rows.

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
    return _where_active(V, point_mask, _INACTIVE_POINT_VOLTAGE)


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
    root_type = HHTypedNeuron

    def __init__(self, size: Size, name: Optional[str] = None, **channels) -> None:
        super().__init__(size, name=name)
        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

        self._external_currents: Dict[str, Callable] = dict()

    @property
    def external_currents(self) -> Dict[str, Callable]:
        """Currents contributed by channels this pool does not own.

        Returns
        -------
        dict of str to Callable
            Callbacks registered by :meth:`register_external_current`,
            keyed by the identifier the registrant chose. A mixed-ion
            channel appears here on each of its owner ions.
        """
        return self._external_currents

    def _channels(self) -> Tuple[Channel, ...]:
        """Direct channel children of this ion pool, in graph order."""
        return tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

    def _run_on_dependent_children(self, nodes, method: str, V) -> None:
        """Forward one lifecycle call to every child that does not self-integrate.

        ``pre_integral``, ``compute_derivative``, and ``post_integral``
        differ only in the name they forward and in where the ion's own hook
        runs relative to the children, so the loop itself is shared. ``nodes``
        is passed in rather than recomputed because ``compute_derivative``
        also needs it for its hierarchy check, and ``brainstate.graph.nodes``
        is not free at trace time.
        """
        ion_info = self.pack_info()
        for node in nodes:
            if not isinstance(node, IndependentIntegration):
                getattr(node, method)(V, ion_info)

    def pre_integral(self, V, recursive_child: bool = True):
        """Run this ion's pre-integration hook, then its children's.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        recursive_child : bool, default True
            Whether to forward the call to child channels. The
            multi-compartment family phases pass ``False`` and drive the
            children themselves.
        """
        self._run_ion_hook("_ion_pre_integral_hook", V)
        if recursive_child:
            self._run_on_dependent_children(self._channels(), "pre_integral", V)

    def compute_derivative(self, V, recursive_child: bool = True):
        """Fill in the derivatives of this ion and its children.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        recursive_child : bool, default True
            Whether to forward the call to child channels.

        Raises
        ------
        TypeError
            If a child channel's ``root_type`` does not accept this ion.

        Notes
        -----
        Children run first and this ion's own hook second, so a hook that
        reads a child-written derivative sees the updated value.
        """
        nodes = self._channels()
        self.check_hierarchies(type(self), *nodes)
        if recursive_child:
            self._run_on_dependent_children(nodes, "compute_derivative", V)
        self._run_ion_hook("_ion_compute_derivative_hook", V)

    def post_integral(self, V, recursive_child: bool = True):
        """Run this ion's children's post-integration hooks, then its own.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        recursive_child : bool, default True
            Whether to forward the call to child channels.
        """
        if recursive_child:
            self._run_on_dependent_children(self._channels(), "post_integral", V)
        self._run_ion_hook("_ion_post_integral_hook", V)

    def current(self, V, include_external: bool = False):
        """Sum the current carried by this ion across all its channels.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        include_external : bool, default False
            Whether to add the callbacks in :attr:`external_currents`, which
            is how a mixed-ion channel reaches its owner ions.

        Returns
        -------
        brainunit.Quantity or None
            The summed current, or ``None`` when this pool has no channels
            and no external currents to add.

        Notes
        -----
        A channel painted onto only some points carries a ``_point_mask``.
        Its voltage is replaced at the unpainted points before evaluation --
        rate expressions overflow on whatever junk is there otherwise -- and
        its current is masked back to zero afterwards.
        """
        nodes = self._channels()
        ion_info = self.pack_info()
        current = None
        for node in nodes:
            point_mask = getattr(node, "_point_mask", None)
            node_V = _safe_inactive_voltage(V, point_mask) if point_mask is not None else V
            new_current = jax.named_call(
                node.current,
                name=_channel_call_name("braincell:ion_current:channel", node),
            )(node_V, ion_info)
            new_current = _profile_barrier_current(new_current)
            if point_mask is not None:
                new_current = _mask_inactive_current(new_current, point_mask)
            current = new_current if current is None else (current + new_current)
        if include_external:
            for key, fun in self._external_currents.items():
                contrib = jax.named_call(
                    fun,
                    name=_external_current_call_name("braincell:ion_current:external", key),
                )(V, ion_info)
                contrib = _profile_barrier_current(contrib)
                current = contrib if current is None else (current + contrib)
        return current

    def _run_state_lifecycle(self, method: str, V, batch_size) -> None:
        """Share the body of :meth:`init_state` and :meth:`reset_state`.

        The two were byte-identical apart from the forwarded method name and
        the matching ``_ion_<method>_hook``. Unlike the integration
        lifecycle this reaches *every* child, including the ones that
        integrate themselves: a channel still needs its state allocated
        whoever advances it.
        """
        nodes = self._channels()
        self.check_hierarchies(type(self), *nodes)
        self._run_ion_hook(f"_ion_{method}_hook", V, batch_size=batch_size)
        ion_info = self.pack_info()
        for node in nodes:
            getattr(node, method)(V, ion_info, batch_size=batch_size)

    def init_state(self, V, batch_size: int = None):
        """Allocate the state of this ion and every child channel.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        batch_size : int, optional
            Leading batch dimension to allocate states with.

        Raises
        ------
        TypeError
            If a child channel's ``root_type`` does not accept this ion.
        """
        self._run_state_lifecycle("init_state", V, batch_size)

    def reset_state(self, V, batch_size: int = None):
        """Return this ion and every child channel to its initial state.

        Parameters
        ----------
        V : ArrayLike or brainunit.Quantity
            Membrane potential for all neurons and compartments.
        batch_size : int, optional
            Leading batch dimension to reset states with.

        Raises
        ------
        TypeError
            If a child channel's ``root_type`` does not accept this ion.
        """
        self._run_state_lifecycle("reset_state", V, batch_size)

    def ind_update(self, V, *args, recursive_child: bool = True, **kwargs):
        if isinstance(self, IndependentIntegration):
            self.make_integration(V, recursive_child=recursive_child)

        if not recursive_child:
            return

        ion_info = self.pack_info()
        for node in self._channels():
            node.ind_update(V, ion_info)

    def _run_ion_hook(self, name: str, *args, **kwargs):
        hook = getattr(self, name, None)
        if hook is not None:
            hook(*args, **kwargs)

    def register_external_current(self, key: Hashable, fun: Callable):
        """Register a current carried by this ion but owned elsewhere.

        Parameters
        ----------
        key : Hashable
            Unique identifier for the current. :class:`MixIons` uses the
            channel's ``id``, optionally paired with a component key.
        fun : Callable
            Callback with signature ``fun(V, ion_info)`` returning a
            current.

        Raises
        ------
        ValueError
            If ``key`` is already registered.
        """
        if key in self._external_currents:
            raise ValueError
        self._external_currents[key] = fun

    def pack_info(self) -> IonInfo:
        """Snapshot this ion's concentrations, reversal potential, and valence.

        Returns
        -------
        IonInfo
            Named tuple of ``Ci``, ``Co``, ``E``, and ``valence``. This is
            what every child channel receives instead of the ion itself, so
            a channel cannot reach back into the pool.

        Notes
        -----
        Each field goes through :func:`brainstate.maybe_state`, so a field
        held as a :class:`brainstate.State` is unwrapped to its ``value``
        and a plain array is passed through.
        """
        return IonInfo(
            Ci=brainstate.maybe_state(self.Ci),
            Co=brainstate.maybe_state(self.Co),
            E=brainstate.maybe_state(self.E),
            valence=brainstate.maybe_state(self.valence),
        )

    def add(self, **elements):
        """Add channels to this ion pool after construction.

        Parameters
        ----------
        **elements
            Channel instances to add, keyed by attribute name.

        Raises
        ------
        TypeError
            If an element's ``root_type`` does not accept this ion, or if it
            is not a :class:`~braincell.Channel`.
        """
        self.check_hierarchies(type(self), **elements)
        self.channels.update(self._format_elements(Channel, **elements))


class MixIons(IonChannel, Container):
    """A pool for channels whose current depends on more than one ion.

    A calcium-dependent potassium channel needs both the potassium
    reversal potential and the calcium concentration, so it cannot hang off
    either pool alone. It declares a joint ``root_type`` and is registered
    here instead; each of its owner ions then sees its current through
    :meth:`Ion.register_external_current`.

    Parameters
    ----------
    *ions : Ion
        Two or more ion pools to mix. All must have the same ``size``.
    name : str, optional
        The name of the instance. A default is generated when omitted.
    **channels
        Channel instances to add, keyed by attribute name.

    Attributes
    ----------
    ions : tuple of Ion
        The mixed ion pools, in the order given.
    ion_types : tuple of type
        The type of each entry in ``ions``.
    channels : dict of str to Channel
        Channels registered with this pool.

    Raises
    ------
    AssertionError
        If fewer than two ions are given, if any is not an
        :class:`~braincell.Ion`, or if their sizes differ.
    """

    __module__ = 'braincell'
    _container_name = 'channels'
    root_type = HHTypedNeuron

    def __init__(self, *ions, name: Optional[str] = None, **channels):
        """See class docstring."""
        assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ion. '
        assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
        size = ions[0].size
        for ion in ions:
            assert ion.size == size, f'The size of all ion should be the same. But we got {ions}.'
        super().__init__(size=size, name=name)

        self.ions: Sequence['Ion'] = tuple(ions)

        # ``add`` is the only door into ``channels``: besides validating the
        # root-type hierarchy it registers each channel's current with the
        # ions that own it. Updating the dict directly here skipped both, so
        # a channel passed to the constructor contributed no current at all
        # -- ``ion.current(V, include_external=True)`` returned ``None``.
        self.channels: Dict[str, Channel] = dict()
        self.add(**channels)

    @property
    def ion_types(self) -> Tuple[Type[Ion], ...]:
        """Types of ions in this mixed channel."""
        return tuple(type(ion) for ion in self.ions)

    def _channels(self) -> Tuple[Channel, ...]:
        """Child channels of this mixed pool, in graph order."""
        return tuple(brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

    def _pack_ion_infos(self) -> Dict[int, IonInfo]:
        """Pack one :class:`IonInfo` per mixed ion, keyed by identity.

        ``pack_info`` is not free. For an ion whose ``E`` is a computed
        Nernst property, reading it emits the whole expression into the
        jaxpr, so packing once per ``(channel, root)`` pair re-emitted that
        expression for every channel. A 32-channel KCa pool traced to 4222
        equations that way and traces to 3075 this way, and ``make_jaxpr``
        went from 0.477 s to 0.293 s. XLA common-subexpression-eliminates
        the duplicates, so the compiled step is unchanged -- this buys
        trace and compile time, and a smaller HLO.
        """
        return {id(ion): ion.pack_info() for ion in self.ions}

    def _infos_for(self, node: Channel, infos: Dict[int, IonInfo]) -> Tuple[IonInfo, ...]:
        """Select ``node``'s infos from ``infos``, in its declared root order."""
        return tuple(infos[id(self._get_ion(root))] for root in node.root_type.__args__)

    def _run_on_dependent_children(self, method: str, V) -> None:
        """Forward one lifecycle call to every child that does not self-integrate.

        ``pre_integral``, ``compute_derivative``, and ``post_integral``
        differ only in the name they forward, so they share this body.
        """
        infos = self._pack_ion_infos()
        for node in self._channels():
            if not isinstance(node, IndependentIntegration):
                getattr(node, method)(V, *self._infos_for(node, infos))

    def pre_integral(self, V):
        """Forward pre-integration to every dependent child channel."""
        self._run_on_dependent_children("pre_integral", V)

    def compute_derivative(self, V):
        """Forward derivative computation to every dependent child channel."""
        self._run_on_dependent_children("compute_derivative", V)

    def post_integral(self, V):
        """Forward post-integration to every dependent child channel."""
        self._run_on_dependent_children("post_integral", V)

    def current(self, V):
        """Sum the current carried by every channel in this mixed pool.

        Returns
        -------
        brainunit.Quantity or None
            The summed current, or ``None`` when the pool holds no
            channels. ``None`` is the same "nothing to contribute"
            sentinel :meth:`Ion.current` returns; this used to be a bare,
            unitless ``0.0``, which made summing over a pool crash with a
            unit mismatch instead.
        """
        ion_infos = self._pack_ion_infos()
        current = None
        for node in self._channels():
            infos = self._infos_for(node, ion_infos)
            point_mask = getattr(node, "_point_mask", None)
            node_V = _safe_inactive_voltage(V, point_mask) if point_mask is not None else V
            new_current = jax.named_call(
                node.current,
                name=_channel_call_name("braincell:mix_ion_current:channel", node),
            )(node_V, *infos)
            new_current = _profile_barrier_current(new_current)
            if point_mask is not None:
                new_current = _mask_inactive_current(new_current, point_mask)
            current = new_current if current is None else (current + new_current)
        return current

    def init_state(self, V, batch_size: int = None):
        """Allocate the state of every channel in this mixed pool."""
        nodes = self._channels()
        self.check_hierarchies(self.ion_types, *nodes, check_fun=self._check_hierarchy)
        infos = self._pack_ion_infos()
        for node in nodes:
            node.init_state(V, *self._infos_for(node, infos), batch_size=batch_size)

    def reset_state(self, V, batch_size=None):
        """Return every channel in this mixed pool to its initial state."""
        infos = self._pack_ion_infos()
        for node in self._channels():
            node.reset_state(V, *self._infos_for(node, infos), batch_size=batch_size)

    def ind_update(self, V, *args, **kwargs):
        """Let every self-integrating child in this pool advance itself."""
        infos = self._pack_ion_infos()
        for node in self._channels():
            node.ind_update(V, *self._infos_for(node, infos))

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

        Which root each argument position comes from is resolved here, at
        registration, not on every call: ``self.ions`` is fixed at
        construction, so the answer cannot change. That also keeps ``self``
        out of the closure. The callback lives in
        ``Ion._external_currents`` for the life of the model, so capturing
        ``self`` pinned the whole ``MixIons`` -- and through it every child
        channel -- behind a reference cycle only the cyclic collector could
        break. Building and dropping 40 half-million-compartment models
        peaked at 291-317 MB of RSS growth with that capture and 155-201 MB
        without it.
        """
        # ``None`` means "the ion this callback was registered with", which
        # supplies its own info at call time.
        sources = tuple(None if isinstance(ion, root) else self._get_ion(root) for root in node.root_type.__args__)

        def fun(V, ion_info):
            infos = tuple(ion_info if source is None else source.pack_info() for source in sources)
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

    Examples
    --------
    .. code-block:: python

        >>> import braincell
        >>> sodium_ion = braincell.ion.SodiumFixed(...)
        >>> potassium_ion = braincell.ion.PotassiumFixed(...)
        >>> mixed_channel = braincell.mix_ions(sodium_ion, potassium_ion)
    """
    for ion in ions:
        assert isinstance(ion, Ion), f'Must be instance of {Ion.__name__}. But got {type(ion)}'
    assert len(ions) >= 2, f'mix_ions requires at least two ions, got {len(ions)}.'
    return MixIons(*ions)
