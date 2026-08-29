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

"""Channel-to-ion runtime binding resolution and runtime node instantiation.

This module owns the second half of mechanism lowering: once
:mod:`braincell._compute.ions` has built the runtime ion instances, everything
here turns the remaining mechanism layouts into live runtime nodes and keeps
those nodes in step with the state buffers.

- :func:`_build_runtime_nodes` — the entry point. Builds the runtime ions,
  installs merged channel nodes, then instantiates one runtime node per
  remaining layout, returning the ion maps together with the runtime nodes,
  their bound ion keys, their current-owner keys, and the merged-channel layout
  groups.
- :func:`_resolve_channel_runtime_bindings`, :func:`_resolve_ion_instance_key`,
  :func:`_channel_family_slots`, :func:`_root_type_to_family`,
  :func:`_channel_current_owner_specs`, :func:`_channel_current_owner_family` —
  the binding rules. They read a channel class's ``root_type`` and
  ``current_owner_type`` / ``current_owner_types`` declarations, match them
  against the declared ion selectors, and decide which ion instance owns which
  component current.
- :class:`_BoundIonChannelRuntime` and
  :class:`_BoundIonChannelCurrentComponentRuntime` — the wrappers installed on
  an owner ion. The first forwards every hook to a mixed-ion channel with its
  bound ions packed in; the second additionally narrows ``current(...)`` to one
  key of ``current_components(...)`` so a channel that writes several ion
  currents can be attached to several ions without updating shared gating state
  more than once.
- :func:`_instantiate_runtime_node`, :func:`_unique_ion_channel_key`,
  :func:`_owner_channel_executable` — instantiation of a single layout's node
  and its attachment to the owning ion container.
- :func:`_install_merged_channel_nodes`, :func:`_is_mergeable_channel_layout`,
  :func:`_partition_non_overlapping_channel_layouts`,
  :func:`_merged_channel_constructor_params`,
  :func:`_initial_merged_channel_param`,
  :func:`_scatter_active_channel_param` — the merge path. Dense channel layouts
  that share a class, instance name, param set, schedule, and ion bindings and
  whose CV masks do not overlap collapse into one full-width runtime node,
  so a channel painted onto many sections costs one node instead of many.
- :func:`_runtime_param_value`, :func:`_runtime_constructor_params`,
  :func:`_sync_runtime_node_param`, :func:`_merged_channel_param_value`, and
  :data:`_CONDUCTANCE_PARAM_NAMES` — per-node parameter reads and the
  post-compilation write-back that rebuilds a node's params when a state buffer
  changes.
- :func:`_configure_runtime_subsolvers` and :func:`_is_root_level_runtime_node`
  — the two classification helpers the ``Cell`` lowering path calls directly.

Binding happens after ions are constructed, so this module depends downward on
:mod:`braincell._compute.ions` and :mod:`braincell._compute.layouts`, and
outward on ``braincell.mech``, ``braincell.quad``, ``braincell.ion`` and
``braincell.channel``. It imports nothing at runtime from
:mod:`braincell._compute.state`, which sits above it in the layer stack;
:class:`CellRuntimeState` appears only in annotations, through a
``TYPE_CHECKING`` import of that module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell import ion as runtime_ion
from braincell._base import Channel
from braincell.channel._base import Markov
from braincell.ion._base import KineticIon
from braincell.mech import (
    Density,
    SynapseSpec as SynapsePlacement,
    get_registry,
)
from braincell._compute.parameters import (
    RuntimeParameterState,
    density_parameter_names,
    parameter_state_value,
)
from braincell.quad import get_integrator
from .ions import _build_runtime_ions, _sync_runtime_ion
from .layouts import MechanismLayout

if TYPE_CHECKING:
    from .state import CellRuntimeState


def _configure_runtime_subsolvers(
    runtime: CellRuntimeState,
    *,
    solver,
    substeps: int,
) -> None:
    """Apply declaration-local or Cell-wide independent schedules."""
    fallback_solver = get_integrator(solver)
    records: dict[int, dict[str, object]] = {}

    for layout in runtime.layouts:
        declaration = runtime.layout_mechanisms[layout.id]
        if not isinstance(declaration, Density):
            continue
        node = runtime.runtime_nodes.get(layout.id)
        supports_schedule = isinstance(node, (Markov, KineticIon))
        has_override = declaration.solver is not None
        if has_override and not supports_schedule:
            raise ValueError(
                f"{type(declaration).__name__} declaration "
                f"{declaration.instance_name!r} sets solver/substeps, but "
                f"runtime {type(node).__name__!r} is neither a Markov channel "
                "nor a KineticIon."
            )
        if not supports_schedule:
            continue

        record = records.setdefault(
            id(node),
            {"node": node, "explicit": [], "declarations": []},
        )
        record["declarations"].append(declaration)
        if has_override:
            record["explicit"].append(
                (
                    get_integrator(declaration.solver),
                    declaration.substeps,
                    declaration,
                )
            )

    for record in records.values():
        explicit = record["explicit"]
        if explicit:
            selected_solver, selected_substeps, selected_declaration = explicit[0]
            for candidate_solver, candidate_substeps, candidate_declaration in explicit[1:]:
                if candidate_solver is not selected_solver or candidate_substeps != selected_substeps:
                    node = record["node"]
                    raise ValueError(
                        f"Runtime {type(node).__name__!r} receives conflicting "
                        "solver/substeps overrides from declarations "
                        f"{selected_declaration.instance_name!r} and "
                        f"{candidate_declaration.instance_name!r}. Use distinct "
                        "Ion names when different KineticIon schedules are required."
                    )
        else:
            selected_solver = fallback_solver
            selected_substeps = substeps

        node = record["node"]
        node.solver = selected_solver
        node.substeps = selected_substeps


def _build_runtime_nodes(
    *,
    n_cv: int,
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
    state_buffers: dict[tuple[int, str], np.ndarray],
    pop_size: tuple[int, ...] = (),
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[int, object],
    dict[int, tuple[str, ...]],
    dict[int, str | None],
    dict[int, tuple[int, ...]],
]:
    (
        ions,
        ion_aliases,
        ion_family_candidates,
        ion_class_candidates,
        ion_runtime_nodes,
    ) = _build_runtime_ions(
        n_cv=n_cv,
        layouts=layouts,
        layout_mechanisms=layout_mechanisms,
        state_buffers=state_buffers,
        pop_size=pop_size,
    )
    runtime_nodes: dict[int, object] = dict(ion_runtime_nodes)
    bound_ion_keys: dict[int, tuple[str, ...]] = {}
    current_owner_keys: dict[int, str | tuple[str, ...] | None] = {}

    merged_channel_layout_groups = _install_merged_channel_nodes(
        n_cv=n_cv,
        pop_size=pop_size,
        layouts=layouts,
        layout_mechanisms=layout_mechanisms,
        state_buffers=state_buffers,
        ions=ions,
        ion_aliases=ion_aliases,
        ion_family_candidates=ion_family_candidates,
        runtime_nodes=runtime_nodes,
        bound_ion_keys=bound_ion_keys,
        current_owner_keys=current_owner_keys,
    )

    for layout in layouts:
        if layout.id in merged_channel_layout_groups:
            continue
        mechanism = layout_mechanisms[layout.id]
        node, layout_bound_ion_keys, current_owner_key = _instantiate_runtime_node(
            layout=layout,
            mechanism=mechanism,
            state_buffers=state_buffers,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
            pop_size=pop_size,
        )
        if node is not None:
            runtime_nodes[layout.id] = node
            bound_ion_keys[layout.id] = layout_bound_ion_keys
            current_owner_keys[layout.id] = current_owner_key
    return (
        ions,
        ion_aliases,
        ion_family_candidates,
        ion_class_candidates,
        runtime_nodes,
        bound_ion_keys,
        current_owner_keys,
        merged_channel_layout_groups,
    )


class _BoundIonChannelRuntime(Channel):
    __module__ = "braincell._compute"

    def __init__(self, channel: object, *, bound_ions: tuple[object, ...], owner_ion: object):
        super().__init__(size=channel.size, name=getattr(channel, "name", None))
        self._channel = channel
        self._bound_ions = tuple(bound_ions)
        self.root_type = type(owner_ion)

    def _infos(self):
        return tuple(ion.pack_info() for ion in self._bound_ions)

    def pre_integral(self, V, *unused):
        return self._channel.pre_integral(V, *self._infos())

    def compute_derivative(self, V, *unused):
        return self._channel.compute_derivative(V, *self._infos())

    def post_integral(self, V, *unused):
        return self._channel.post_integral(V, *self._infos())

    def init_state(self, V, *unused, batch_size=None):
        return self._channel.init_state(V, *self._infos(), batch_size=batch_size)

    def reset_state(self, V, *unused, batch_size=None):
        return self._channel.reset_state(V, *self._infos(), batch_size=batch_size)

    def ind_update(self, V, *unused):
        return self._channel.ind_update(V, *self._infos())

    def current(self, V, *unused):
        return self._channel.current(V, *self._infos())


class _BoundIonChannelCurrentComponentRuntime(Channel):
    """Expose one component current of a bound mixed-ion channel.

    Parameters
    ----------
    channel : object
        Runtime channel instance that owns the actual gating state and
        total membrane-current implementation.
    bound_ions : tuple of object
        Ion instances required by ``channel.root_type`` in call order.
    owner_ion : object
        Ion instance that owns this component-current wrapper.
    component_key : str
        Key read from ``channel.current_components(...)``.
    owns_state : bool, optional
        Whether this wrapper forwards lifecycle and integration hooks to
        the wrapped channel. Exactly one wrapper should own state for a
        multi-owner channel.

    Notes
    -----
    Only one component wrapper should own state. Other wrappers are
    current-only and let ``owner_ion.current(...)`` return the component
    written to that ion without double-updating the shared gating state.
    """

    __module__ = "braincell._compute"

    def __init__(
        self,
        channel: object,
        *,
        bound_ions: tuple[object, ...],
        owner_ion: object,
        component_key: str,
        owns_state: bool = False,
    ):
        super().__init__(size=channel.size, name=getattr(channel, "name", None))
        self._channel = channel
        self._bound_ions = tuple(bound_ions)
        self._component_key = component_key
        self._owns_state = bool(owns_state)
        self._skip_family_update = not self._owns_state
        self.root_type = type(owner_ion)

    def _infos(self):
        """Return packed ion information for the wrapped channel.

        Returns
        -------
        tuple
            Packed ion information objects in the order expected by the
            wrapped channel.
        """
        return tuple(ion.pack_info() for ion in self._bound_ions)

    def pre_integral(self, V, *unused):
        """Forward pre-integration from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.

        Returns
        -------
        object or None
            Return value of the wrapped channel's ``pre_integral`` when
            this wrapper owns state; otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.pre_integral(V, *self._infos())
        return None

    def compute_derivative(self, V, *unused):
        """Forward derivative computation from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.

        Returns
        -------
        object or None
            Return value of the wrapped channel's
            ``compute_derivative`` when this wrapper owns state;
            otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.compute_derivative(V, *self._infos())
        return None

    def post_integral(self, V, *unused):
        """Forward post-integration from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.

        Returns
        -------
        object or None
            Return value of the wrapped channel's ``post_integral`` when
            this wrapper owns state; otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.post_integral(V, *self._infos())
        return None

    def init_state(self, V, *unused, batch_size=None):
        """Initialize shared channel state from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.
        batch_size : int or None, optional
            Optional batch size forwarded to the wrapped channel.

        Returns
        -------
        object or None
            Return value of the wrapped channel's ``init_state`` when
            this wrapper owns state; otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.init_state(V, *self._infos(), batch_size=batch_size)
        return None

    def reset_state(self, V, *unused, batch_size=None):
        """Reset shared channel state from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.
        batch_size : int or None, optional
            Optional batch size forwarded to the wrapped channel.

        Returns
        -------
        object or None
            Return value of the wrapped channel's ``reset_state`` when
            this wrapper owns state; otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.reset_state(V, *self._infos(), batch_size=batch_size)
        return None

    def update(self, V, *unused):
        """Update shared channel state from the state-owning wrapper.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by ion containers.

        Returns
        -------
        object or None
            Return value of the wrapped channel's ``update`` when this
            wrapper owns state; otherwise ``None``.
        """
        if self._owns_state:
            return self._channel.update(V, *self._infos())
        return None

    def current(self, V, *unused):
        """Return this owner ion's component current.

        Parameters
        ----------
        V : array-like
            Membrane potential passed to the wrapped channel.
        *unused
            Ignored compatibility arguments supplied by
            :meth:`braincell.Ion.current`.

        Returns
        -------
        array-like
            Current density stored under ``component_key`` in
            ``channel.current_components(...)``.

        Raises
        ------
        AttributeError
            If the wrapped channel does not implement
            ``current_components(...)``.
        KeyError
            If ``component_key`` is not returned by the wrapped channel.
        """
        return self._channel.current_components(V, *self._infos())[self._component_key]


def _instantiate_runtime_node(
    *,
    layout: MechanismLayout,
    mechanism: object,
    state_buffers: dict[tuple[int, str], np.ndarray],
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
    pop_size: tuple[int, ...],
) -> tuple[object | None, tuple[str, ...], str | tuple[str, ...] | None]:
    if isinstance(mechanism, SynapsePlacement):
        runtime_cls = get_registry().get("synapse", mechanism.synapse_type)
        parameter_names = tuple(
            var_name
            for layout_id, var_name in state_buffers
            if int(layout_id) == int(layout.id) and not var_name.startswith("_")
        )
        params = {
            var_name: _runtime_param_value(
                layout=layout,
                var_name=var_name,
                state_buffers=state_buffers,
            )
            for var_name in parameter_names
        }
        size = (int(layout.n_active),)
        node = runtime_cls(size=size, name=mechanism.synapse_type, **params)
        return node, (), None

    if layout.target != "density" or layout.layout != "dense":
        return None, (), None
    if not isinstance(mechanism, Density):
        return None, (), None
    if mechanism.category != "channel":
        return None, (), None

    runtime_cls = get_registry().get("channel", mechanism.class_name)
    params = _runtime_constructor_params(layout=layout, mechanism=mechanism, state_buffers=state_buffers)
    if len(params) > 0 and hasattr(next(iter(params.values())), "shape"):
        # Parameter buffers already carry the population axis.
        size = next(iter(params.values())).shape
    else:
        size = pop_size + (layout.spatial_axis_len,)
    node = runtime_cls(size=size, **params)
    _attach_runtime_parameter_states(node, params)
    bound_ions, current_owner_specs = _resolve_channel_runtime_bindings(
        runtime_cls=runtime_cls,
        mechanism=mechanism,
        ions=ions,
        ion_aliases=ion_aliases,
        ion_family_candidates=ion_family_candidates,
    )
    current_owner_keys = tuple(ion_key for _, ion_key in current_owner_specs)
    if current_owner_specs:
        if layout.layout == "dense" and layout.cv_mask is not None and not np.all(layout.cv_mask):
            setattr(node, "_point_mask", layout.cv_mask)
    state_owner_assigned = False
    for component_key, current_owner_key in current_owner_specs:
        owner_ion = ions[current_owner_key]
        channel_key = _unique_ion_channel_key(owner_ion, mechanism.instance_name, layout_id=layout.id)
        if component_key is None and len(bound_ions) == 1 and bound_ions[0][0] == current_owner_key:
            owner_ion.add(**{channel_key: node})
        elif component_key is None:
            wrapper = _BoundIonChannelRuntime(
                node,
                bound_ions=tuple(ion for _, ion in bound_ions),
                owner_ion=owner_ion,
            )
            if layout.layout == "dense" and layout.cv_mask is not None and not np.all(layout.cv_mask):
                setattr(wrapper, "_point_mask", layout.cv_mask)
            owner_ion.add(**{channel_key: wrapper})
        else:
            owns_state = not state_owner_assigned
            wrapper = _BoundIonChannelCurrentComponentRuntime(
                node,
                bound_ions=tuple(ion for _, ion in bound_ions),
                owner_ion=owner_ion,
                component_key=component_key,
                owns_state=owns_state,
            )
            state_owner_assigned = state_owner_assigned or owns_state
            if layout.layout == "dense" and layout.cv_mask is not None and not np.all(layout.cv_mask):
                setattr(wrapper, "_point_mask", layout.cv_mask)
            owner_ion.add(**{channel_key: wrapper})
    current_owner_key = (
        None
        if len(current_owner_keys) == 0
        else current_owner_keys[0]
        if len(current_owner_keys) == 1
        else current_owner_keys
    )
    return node, tuple(ion_key for ion_key, _ in bound_ions), current_owner_key


def _install_merged_channel_nodes(
    *,
    n_cv: int,
    pop_size: tuple[int, ...],
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
    state_buffers: dict[tuple[int, str], np.ndarray],
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
    runtime_nodes: dict[int, object],
    bound_ion_keys: dict[int, tuple[str, ...]],
    current_owner_keys: dict[int, str | tuple[str, ...] | None],
) -> dict[int, tuple[int, ...]]:
    groups: dict[
        tuple[object, ...],
        list[tuple[MechanismLayout, Density, type, tuple[tuple[str, object], ...], tuple[tuple[str | None, str], ...]]],
    ] = {}
    for layout in layouts:
        mechanism = layout_mechanisms[layout.id]
        if not _is_mergeable_channel_layout(layout, mechanism):
            continue
        runtime_cls = get_registry().get("channel", mechanism.class_name)
        bound_ions, owner_specs = _resolve_channel_runtime_bindings(
            runtime_cls=runtime_cls,
            mechanism=mechanism,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
        )
        if len(owner_specs) != 1:
            continue
        key = (
            runtime_cls,
            mechanism.instance_name,
            tuple(sorted(mechanism.params.keys())),
            mechanism.solver,
            mechanism.substeps,
            tuple(ion_key for ion_key, _ in bound_ions),
            owner_specs,
        )
        groups.setdefault(key, []).append((layout, mechanism, runtime_cls, bound_ions, owner_specs))

    installed: dict[int, tuple[int, ...]] = {}
    for items in groups.values():
        merge_sets = _partition_non_overlapping_channel_layouts(items)
        for merge_items in merge_sets:
            if len(merge_items) < 2:
                continue
            layout_ids = tuple(layout.id for layout, *_ in merge_items)
            layout0, mechanism0, runtime_cls, bound_ions, owner_specs = merge_items[0]
            params = _merged_channel_constructor_params(
                n_cv=n_cv,
                pop_size=pop_size,
                items=merge_items,
                state_buffers=state_buffers,
            )
            size = pop_size + (n_cv,)
            node = runtime_cls(size=size, **params)
            _attach_runtime_parameter_states(node, params)
            cv_mask = np.zeros((n_cv,), dtype=bool)
            for layout, *_ in merge_items:
                cv_mask |= np.asarray(layout.cv_mask, dtype=bool)
            if not np.all(cv_mask):
                setattr(node, "_point_mask", cv_mask)
            executable = _owner_channel_executable(
                node,
                bound_ions=bound_ions,
                owner_specs=owner_specs,
                owner_ion=ions[owner_specs[0][1]],
            )
            if executable is not node and not np.all(cv_mask):
                setattr(executable, "_point_mask", cv_mask)

            owner_ion = ions[owner_specs[0][1]]
            channel_key = _unique_ion_channel_key(owner_ion, mechanism0.instance_name, layout_id=layout0.id)
            owner_ion.add(**{channel_key: executable})
            current_owner_key = owner_specs[0][1]
            layout_bound_ion_keys = tuple(ion_key for ion_key, _ in bound_ions)
            for layout_id in layout_ids:
                runtime_nodes[layout_id] = node
                bound_ion_keys[layout_id] = layout_bound_ion_keys
                current_owner_keys[layout_id] = current_owner_key
                installed[layout_id] = layout_ids
    return installed


def _is_mergeable_channel_layout(layout: MechanismLayout, mechanism: object) -> bool:
    return (
        layout.target == "density"
        and layout.layout == "dense"
        and layout.cv_mask is not None
        and isinstance(mechanism, Density)
        and mechanism.category == "channel"
    )


def _partition_non_overlapping_channel_layouts(items):
    partitions = []
    for item in items:
        layout = item[0]
        mask = np.asarray(layout.cv_mask, dtype=bool)
        placed = False
        for partition in partitions:
            used = partition["mask"]
            if not np.any(used & mask):
                partition["items"].append(item)
                partition["mask"] = used | mask
                placed = True
                break
        if not placed:
            partitions.append({"items": [item], "mask": mask.copy()})
    return [partition["items"] for partition in partitions]


def _merged_channel_constructor_params(
    *,
    n_cv: int,
    pop_size: tuple[int, ...],
    items,
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> dict[str, object]:
    all_param_names = []
    for _layout, mechanism, *_ in items:
        for name in density_parameter_names(mechanism):
            if name not in all_param_names:
                all_param_names.append(name)

    params = {}
    full_shape = pop_size + (n_cv,)
    for var_name in all_param_names:
        value_items = [
            (layout, mechanism) for layout, mechanism, *_ in items if var_name in density_parameter_names(mechanism)
        ]
        if not value_items:
            continue
        first_layout, _first_mechanism = value_items[0]
        first_value = parameter_state_value(
            _runtime_param_value(
                layout=first_layout,
                var_name=var_name,
                state_buffers=state_buffers,
            )
        )
        merged = _initial_merged_channel_param(
            var_name=var_name,
            value=first_value,
            full_shape=full_shape,
        )
        for layout, _mechanism in value_items:
            value = parameter_state_value(
                _runtime_param_value(
                    layout=layout,
                    var_name=var_name,
                    state_buffers=state_buffers,
                )
            )
            merged = _scatter_active_channel_param(
                merged,
                value,
                cv_mask=np.asarray(layout.cv_mask, dtype=bool),
                full_shape=full_shape,
            )
        cv_mask = np.zeros((n_cv,), dtype=bool)
        for layout, _mechanism in value_items:
            cv_mask |= np.asarray(layout.cv_mask, dtype=bool)
        params[var_name] = RuntimeParameterState(
            merged,
            axis="row",
            full_shape=full_shape,
            point_mask=cv_mask,
            zero_inactive=var_name in _CONDUCTANCE_PARAM_NAMES,
        )
    return params


def _initial_merged_channel_param(*, var_name: str, value: object, full_shape: tuple[int, ...]) -> object:
    if isinstance(value, u.Quantity):
        if var_name in _CONDUCTANCE_PARAM_NAMES:
            return u.Quantity(jnp.zeros(full_shape, dtype=jnp.asarray(value.mantissa).dtype), value.unit)
        mantissa = jnp.asarray(value.mantissa)
        if mantissa.shape == full_shape:
            return u.Quantity(mantissa, value.unit)
        return u.Quantity(jnp.broadcast_to(mantissa, full_shape), value.unit)

    values = jnp.asarray(value)
    if var_name in _CONDUCTANCE_PARAM_NAMES:
        return jnp.zeros(full_shape, dtype=values.dtype)
    if values.shape == full_shape:
        return values
    return jnp.broadcast_to(values, full_shape)


def _scatter_active_channel_param(target, value, *, cv_mask: np.ndarray, full_shape: tuple[int, ...]):
    if isinstance(value, u.Quantity):
        unit = target.unit
        values = jnp.asarray(value.to_decimal(unit))
        target_mantissa = jnp.asarray(target.mantissa).at[..., cv_mask].set(values[..., cv_mask])
        return u.Quantity(target_mantissa, unit)

    values = jnp.asarray(value)
    return jnp.asarray(target).at[..., cv_mask].set(values[..., cv_mask])


def _owner_channel_executable(node, *, bound_ions, owner_specs, owner_ion):
    component_key, owner_key = owner_specs[0]
    if component_key is None and len(bound_ions) == 1 and bound_ions[0][0] == owner_key:
        return node
    if component_key is None:
        return _BoundIonChannelRuntime(
            node,
            bound_ions=tuple(ion for _, ion in bound_ions),
            owner_ion=owner_ion,
        )
    return _BoundIonChannelCurrentComponentRuntime(
        node,
        bound_ions=tuple(ion for _, ion in bound_ions),
        owner_ion=owner_ion,
        component_key=component_key,
        owns_state=True,
    )


def _unique_ion_channel_key(owner_ion: object, instance_name: str, *, layout_id: int) -> str:
    channels = getattr(owner_ion, "channels", None)
    if not isinstance(channels, dict) or instance_name not in channels:
        return instance_name

    candidate = f"{instance_name}__layout_{int(layout_id)}"
    if candidate not in channels:
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in channels:
        suffix += 1
    return f"{candidate}_{suffix}"


def _resolve_channel_runtime_bindings(
    *,
    runtime_cls: type,
    mechanism: Density,
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
) -> tuple[tuple[tuple[str, object], ...], tuple[tuple[str | None, str], ...]]:
    """Resolve channel ion dependencies and current owner bindings.

    Parameters
    ----------
    runtime_cls : type
        Concrete runtime channel class selected from the mechanism
        registry.
    mechanism : Density
        Density mechanism declaration being lowered.
    ions : dict of str to object
        Runtime ion instances keyed by ion instance name.
    ion_aliases : dict of str to str
        Aliases that resolve user selectors to runtime ion keys.
    ion_family_candidates : dict of str to tuple of str
        Candidate ion instance names grouped by family key such as
        ``"k"``, ``"na"``, ``"ca"``, or ``"no"``.

    Returns
    -------
    bound_ions : tuple of tuple
        ``(ion_key, ion_instance)`` pairs in the order required by the
        channel's ``root_type``.
    current_owner_specs : tuple of tuple
        ``(component_key, ion_key)`` pairs. ``component_key is None``
        denotes legacy single-owner behavior where owner current is the
        channel's total ``current(...)`` return value. Non-``None`` keys
        denote multi-owner component currents retrieved through
        ``current_components(...)``.

    Raises
    ------
    ValueError
        If selectors do not match the channel arity or a mixed-ion
        channel cannot resolve its declared current owners.
    KeyError
        If an explicit ion selector cannot be resolved.

    Notes
    -----
    Existing mixed-ion channels declare ``current_owner_type`` and
    therefore return exactly one legacy owner spec. Channels that write
    more than one ion current may declare ``current_owner_types`` as a
    mapping from component key to owner ion type; these channels must
    implement ``current_components(...)`` on the channel class.
    """
    family_slots = _channel_family_slots(runtime_cls)
    if len(family_slots) == 0:
        if getattr(mechanism, "ion_name", None) is not None or getattr(mechanism, "ion_names", None) is not None:
            raise ValueError(f"Channel {mechanism.class_name!r} does not bind ions but ion selectors were provided.")
        return (), ()

    if len(family_slots) == 1:
        if getattr(mechanism, "ion_names", None) is not None:
            raise ValueError(f"Single-ion channel {mechanism.class_name!r} must use ion_name, not ion_names.")
        family_key = family_slots[0][0]
        ion_key = _resolve_ion_instance_key(
            family_key=family_key,
            selector=getattr(mechanism, "ion_name", None),
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
        )
        return ((ion_key, ions[ion_key]),), ((None, ion_key),)

    if getattr(mechanism, "ion_name", None) is not None:
        raise ValueError(f"Mixed-ion channel {mechanism.class_name!r} must use ion_names, not ion_name.")
    selector_map = dict(getattr(mechanism, "ion_names", ()) or ())
    slot_keys = {family_key for family_key, _ in family_slots}
    unknown_selector_keys = set(selector_map.keys()) - slot_keys
    if unknown_selector_keys:
        raise ValueError(
            f"Mixed-ion channel {mechanism.class_name!r} received unknown ion_names keys "
            f"{sorted(unknown_selector_keys)!r}; expected subset of {sorted(slot_keys)!r}."
        )

    bound_ions: list[tuple[str, object]] = []
    for family_key, _ in family_slots:
        ion_key = _resolve_ion_instance_key(
            family_key=family_key,
            selector=selector_map.get(family_key),
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
        )
        bound_ions.append((ion_key, ions[ion_key]))

    owner_specs = _channel_current_owner_specs(runtime_cls)
    if not owner_specs:
        raise ValueError(f"Mixed-ion channel class {runtime_cls.__name__!r} must define current_owner_type.")
    current_owner_specs: list[tuple[str | None, str]] = []
    for component_key, current_owner_family in owner_specs:
        owner_candidates = [
            ion_key
            for (family_key, _), (ion_key, _) in zip(family_slots, bound_ions)
            if family_key == current_owner_family
        ]
        if len(owner_candidates) != 1:
            raise ValueError(
                f"Mixed-ion channel class {runtime_cls.__name__!r} could not resolve a unique current owner for family "
                f"{current_owner_family!r}."
            )
        current_owner_specs.append((component_key, owner_candidates[0]))
    return tuple(bound_ions), tuple(current_owner_specs)


def _resolve_ion_instance_key(
    *,
    family_key: str,
    selector: str | None,
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
) -> str:
    candidates = ion_family_candidates.get(family_key, ())
    if len(candidates) == 0:
        raise KeyError(f"No ion candidates are registered for family {family_key!r}.")
    if selector is None:
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Ion family {family_key!r} is ambiguous; candidates are {list(candidates)!r}. "
            f"Declare an explicit ion selector for this family."
        )

    ion_key = selector if selector in ions else ion_aliases.get(selector)
    if ion_key is None:
        raise KeyError(f"Ion selector {selector!r} could not be resolved for family {family_key!r}.")
    if ion_key not in candidates:
        raise ValueError(
            f"Ion selector {selector!r} resolved to {ion_key!r}, which is not a candidate for family "
            f"{family_key!r} ({list(candidates)!r})."
        )
    return ion_key


def _channel_family_slots(cls: type) -> tuple[tuple[str, type], ...]:
    root_type = getattr(cls, "root_type", None)
    if root_type is None:
        return ()
    args = getattr(root_type, "__args__", None)
    if args:
        slots = []
        for root in args:
            family = _root_type_to_family(root)
            if family is not None:
                slots.append((family, root))
        return tuple(slots)
    if isinstance(root_type, type):
        family = _root_type_to_family(root_type)
        if family is not None:
            return ((family, root_type),)
    return ()


def _root_type_to_family(root_type: type) -> str | None:
    """Return the runtime ion family key for an ion root type.

    Parameters
    ----------
    root_type : type
        Ion base class or subclass referenced by a channel
        ``root_type`` declaration.

    Returns
    -------
    str or None
        Family key used by mechanism ``ion_name`` / ``ion_names``
        selectors. Returns ``None`` when ``root_type`` is not a known
        runtime ion family.

    Notes
    -----
    ``"no"`` denotes the NEURON-style nonspecific current-owner
    placeholder family. It is used for written currents such as
    ``USEION no WRITE ino`` and does not imply concentration dynamics.
    """
    try:
        if issubclass(root_type, runtime_ion.Sodium):
            return "na"
        if issubclass(root_type, runtime_ion.Potassium):
            return "k"
        if issubclass(root_type, runtime_ion.Calcium):
            return "ca"
        if issubclass(root_type, runtime_ion.NonSpecific):
            return "no"
    except TypeError:
        return None
    return None


def _channel_current_owner_specs(cls: type) -> tuple[tuple[str | None, str], ...]:
    """Return current-owner component specs for a channel class.

    Parameters
    ----------
    cls : type
        Runtime channel class.

    Returns
    -------
    tuple of tuple
        ``(component_key, family_key)`` pairs. ``component_key is None``
        preserves the legacy single-owner path where the owner receives
        ``current(...)``. A string component key requires the channel to
        provide ``current_components(...)`` and lets the owner receive
        only that component.

    Notes
    -----
    For one-ion channels the sole ion family is always the current
    owner. For mixed-ion channels, ``current_owner_type`` keeps existing
    behavior, while ``current_owner_types`` enables mechanisms that
    write more than one ion current, for example an NMODL mechanism with
    both ``WRITE ik`` and ``WRITE ino``.
    """
    family_slots = _channel_family_slots(cls)
    if len(family_slots) == 0:
        return ()
    if len(family_slots) == 1:
        return ((None, family_slots[0][0]),)
    owner_types = getattr(cls, "current_owner_types", None)
    if owner_types is not None:
        specs = []
        for component_key, owner_type in owner_types.items():
            family = _root_type_to_family(owner_type)
            if family is not None:
                specs.append((component_key, family))
        return tuple(specs)
    owner_type = getattr(cls, "current_owner_type", None)
    if owner_type is None:
        return ()
    family = _root_type_to_family(owner_type)
    if family is None:
        return ()
    return ((None, family),)


def _channel_current_owner_family(cls: type) -> str | None:
    """Return the legacy single current-owner family for a channel class.

    Parameters
    ----------
    cls : type
        Runtime channel class.

    Returns
    -------
    str or None
        Family key when the channel has exactly one current owner.
        Returns ``None`` for root-level channels and for multi-owner
        channels declared with ``current_owner_types``.

    Notes
    -----
    This helper exists for backwards-compatible call sites that only
    need to distinguish root-level channels from ion-bound channels. New
    binding code should use :func:`_channel_current_owner_specs`.
    """
    specs = _channel_current_owner_specs(cls)
    if len(specs) != 1:
        return None
    return specs[0][1]


_CONDUCTANCE_PARAM_NAMES = frozenset({"g_max", "g", "gbar", "conductance"})


def _runtime_param_value(
    *,
    layout: MechanismLayout,
    var_name: str,
    state_buffers: dict,
) -> object:
    """Materialize a runtime-facing value for a rectangular param.

    For :class:`u.Quantity` buffers with a density mask over the declared
    ``g*``-class conductance names, zero out inactive points in a
    JAX-traceable way via :func:`jnp.where`. Other buffers pass through.
    """
    buffer = state_buffers[(layout.id, var_name)]
    if isinstance(buffer, RuntimeParameterState):
        return buffer
    if isinstance(buffer, u.Quantity) and layout.cv_mask is not None and var_name in _CONDUCTANCE_PARAM_NAMES:
        mask_bool = np.asarray(layout.cv_mask)
        masked_mantissa = np.where(mask_bool, np.asarray(buffer.mantissa), 0.0)
        return u.Quantity(masked_mantissa, buffer.unit)
    return buffer


def _sync_runtime_node_param(runtime: CellRuntimeState, *, layout_id: int, var_name: str) -> None:
    node = runtime.runtime_nodes.get(int(layout_id))
    if node is None:
        return
    layout = runtime.layouts[int(layout_id)]
    kind = layout.kind
    if kind.startswith("ion:"):
        _sync_runtime_ion(runtime, layout_id=int(layout_id))
        return
    merged_groups = runtime.merged_channel_layout_groups or {}
    merged_layout_ids = merged_groups.get(int(layout_id))
    if merged_layout_ids is not None and kind.startswith("channel:"):
        new_value = _merged_channel_param_value(
            runtime,
            layout_ids=merged_layout_ids,
            var_name=str(var_name),
        )
        setattr(node, var_name, new_value)
        hook = getattr(node, "_on_param_updated", None)
        if callable(hook):
            hook(var_name, new_value)
        return
    new_value = _runtime_param_value(
        layout=layout,
        var_name=var_name,
        state_buffers=runtime.state_buffers,
    )
    setattr(node, var_name, new_value)
    hook = getattr(node, "_on_param_updated", None)
    if callable(hook):
        hook(var_name, new_value)


def _merged_channel_param_value(
    runtime: CellRuntimeState,
    *,
    layout_ids: tuple[int, ...],
    var_name: str,
) -> object:
    items = [
        (
            runtime.layouts[int(layout_id)],
            runtime.layout_mechanisms[int(layout_id)],
        )
        for layout_id in layout_ids
        if (int(layout_id), str(var_name)) in runtime.state_buffers
    ]
    if not items:
        raise KeyError(f"Unknown merged channel parameter {var_name!r} for layouts {layout_ids!r}.")

    full_shape = runtime.pop_size + (runtime.n_cv,)
    first_layout, _first_mechanism = items[0]
    first_value = parameter_state_value(
        _runtime_param_value(
            layout=first_layout,
            var_name=str(var_name),
            state_buffers=runtime.state_buffers,
        )
    )
    merged = _initial_merged_channel_param(
        var_name=str(var_name),
        value=first_value,
        full_shape=full_shape,
    )
    for layout, _mechanism in items:
        value = parameter_state_value(
            _runtime_param_value(
                layout=layout,
                var_name=str(var_name),
                state_buffers=runtime.state_buffers,
            )
        )
        merged = _scatter_active_channel_param(
            merged,
            value,
            cv_mask=np.asarray(layout.cv_mask, dtype=bool),
            full_shape=full_shape,
        )
    return merged


def _runtime_constructor_params(
    *,
    layout: MechanismLayout,
    mechanism: Density,
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> dict[str, object]:
    """Build the kwargs passed to a concrete channel class's ``__init__``.

    Reads each declared parameter from its state buffer (so per-point
    values already live in the buffer, not in the frozen declaration).
    ``coverage_area_fraction`` is a :class:`Density` field, not a param,
    so it does not leak into kwargs.
    """
    if mechanism.category != "channel":
        return {}
    return {
        var_name: _runtime_param_value(layout=layout, var_name=var_name, state_buffers=state_buffers)
        for var_name in density_parameter_names(mechanism)
    }


def _attach_runtime_parameter_states(node: object, params: dict[str, object]) -> None:
    """Restore schema parameter states unwrapped by ``braintools.init.param``."""
    for name, value in params.items():
        if isinstance(value, RuntimeParameterState):
            setattr(node, name, value)


def _is_root_level_runtime_node(kind: str) -> bool:
    """Return True when a channel layout installs at the root level.

    Root-level channels are those whose concrete class has
    ``root_type == HHTypedNeuron`` (i.e. not bound to an ion
    container). The registry is consulted to inspect the class.

    Raises
    ------
    ValueError
        If ``kind`` names a channel class not registered in the
        mechanism registry — previously this was silently treated
        as "not root-level", hiding misspelled channel names.
    """
    if kind.startswith("synapse:"):
        return True
    if not kind.startswith("channel:"):
        return False
    class_name = kind.split(":", 1)[1]
    try:
        cls = get_registry().get("channel", class_name)
    except KeyError as exc:
        raise ValueError(f"Unknown runtime channel class {class_name!r} for layout kind {kind!r}.") from exc
    return not _channel_current_owner_specs(cls)
