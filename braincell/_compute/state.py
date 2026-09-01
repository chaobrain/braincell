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

"""Mutable runtime state bridging a ``Cell`` declaration and its lowering.

This module owns :class:`CellRuntimeState`, the object a ``Cell`` compiles
itself into once and then delegates to for the rest of its life. It is the top
of the ``_compute`` layer stack: it consumes the vocabularies defined by its
siblings and is consumed in turn by ``Cell`` and the multi-compartment
current/probe code.

- :meth:`CellRuntimeState.from_cell` — the lowering entry point. It groups a
  cell's density and point mechanisms by signature into
  :class:`~braincell._compute.layouts.MechanismLayout` records, allocates one
  state buffer per mechanism variable, instantiates the runtime ions and nodes
  through :func:`~braincell._compute.bindings._build_runtime_nodes`, builds the
  clamp routing table, and precomputes CV/point areas and the midpoint mask.
- Layout membership lookup — :meth:`CellRuntimeState.get_point_layouts` and
  :meth:`CellRuntimeState.get_cv_layouts`.
- State inspection and mutation — :meth:`CellRuntimeState.get_state`,
  :meth:`CellRuntimeState.set_state`, :meth:`CellRuntimeState.get_point_state`,
  :meth:`CellRuntimeState.get_cv_state`,
  :meth:`CellRuntimeState.expected_state_shape`,
  :meth:`CellRuntimeState.has_layout_value`,
  :meth:`CellRuntimeState.get_layout_value`. Writes go through
  :func:`~braincell._compute.bindings._sync_runtime_node_param` so the
  installed runtime node stays in step with the buffer.
- Runtime object lookup — :meth:`CellRuntimeState.get_runtime_node`,
  :meth:`CellRuntimeState.get_layout_mechanism`,
  :meth:`CellRuntimeState.get_ion`, :meth:`CellRuntimeState.resolve_ion_key`,
  :meth:`CellRuntimeState.iter_synapse_layouts`.
- Per-step point-current evaluation through
  :meth:`CellRuntimeState.evaluate_point_clamps`.

Because it sits at the top of the stack, this module imports its siblings
directly: :mod:`braincell._compute.layouts` for the layout record and the
buffer allocation / clamp evaluation helpers,
:mod:`braincell._compute.bindings` for runtime node construction and
synchronization, and :mod:`braincell._compute.bridge` for ion geometry
attachment. Ions are reached only indirectly, through
:mod:`braincell._compute.bindings`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import braintools
import brainstate
import brainunit as u
import numpy as np

from braincell._base import Synapse as RuntimeSynapse
from braincell._discretization.base import NodeTree
from braincell.mech import NoEventInput, ScalarEventInput, TriggerEventInput
from braincell.mech import (
    CurrentClamp,
    Density,
    SynapseSpec as SynapsePlacement,
    get_registry,
)
from .bindings import (
    _build_runtime_nodes,
    _configure_runtime_subsolvers,
    _sync_runtime_node_param,
)
from .bridge import attach_runtime_ion_geometry
from .layouts import (
    CLAMP_KINDS,
    MechanismLayout,
    _allocate_clamp_ragged_buffer,
    _allocate_current_clamp_buffer,
    _allocate_current_clamp_delay_buffer,
    _allocate_spatial_density_buffer,
    _allocate_state_buffer,
    _evaluate_clamp_layout,
    _extract_point_value,
    _mechanism_var_names,
    _mechanism_var_value,
    _quantity_sequence_to_decimal_vector,
    _source_cv_ids_for_point,
    _stack_synapse_values,
    _write_state_buffer,
    build_clamp_routing_table,
    choose_layout,
    mechanism_kind,
    mechanism_signature,
)

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

__all__ = [
    "CellRuntimeState",
]


@dataclass
class CellRuntimeState:
    """Lightweight bridge state between ``Cell`` declarations and runtime layout.

    This object is intentionally internal-facing. Users still interact with
    ``Cell``; runtime state simply owns the lowered layouts, state buffers, and
    installed runtime nodes that ``Cell`` delegates to after compilation.

    It stores four kinds of runtime-facing data:

    - topology context: point tree, point count, CV count, voltage shape
    - lowering metadata: :class:`MechanismLayout` records and layout memberships
    - mutable state: per-layout state buffer arrays and expected shapes
    - installed runtime objects: instantiated mechanism nodes and ion objects

    Main method groups:

    - layout membership lookup: :meth:`get_point_layouts`, :meth:`get_cv_layouts`
    - state inspection and mutation: :meth:`get_state`, :meth:`set_state`,
      :meth:`get_point_state`, :meth:`get_cv_state`
    - runtime object lookup: :meth:`get_runtime_node`, :meth:`get_ion`
    - point-level clamp evaluation: :meth:`evaluate_point_clamps`
    - table views: :meth:`mechanism_cv_table`, :meth:`mechanism_point_table`

    The main collaboration is upward: :class:`Cell` compiles and caches one
    ``CellRuntimeState`` instance, then uses it to install runtime nodes, bridge
    between CV-space and point-space arrays, and expose runtime inspection APIs.
    """

    node_tree: NodeTree
    n_point: int
    n_cv: int
    layouts: tuple[MechanismLayout, ...]
    point_to_layout_ids: tuple[tuple[int, ...], ...]
    cv_to_layout_ids: tuple[tuple[int, ...], ...]
    voltage_shape: tuple[int, ...]
    state_shapes: dict[tuple[int, str], tuple[int, ...]]
    state_buffers: dict[tuple[int, str], np.ndarray]
    event_buffers: dict[int, object]
    layout_mechanisms: dict[int, object]
    runtime_nodes: dict[int, object]
    ions: dict[str, object]
    ion_aliases: dict[str, str]
    ion_family_candidates: dict[str, tuple[str, ...]]
    ion_class_candidates: dict[str, tuple[str, ...]]
    bound_ion_keys: dict[int, tuple[str, ...]]
    current_owner_keys: dict[int, str | tuple[str, ...] | None]
    midpoint_mask_np: np.ndarray
    merged_channel_layout_groups: dict[int, tuple[int, ...]] | None = None
    dhs_static_source_np: object | None = None
    dhs_static_cache: object | None = None
    axial_operator_np: np.ndarray | None = None
    axial_operator_cache: object | None = None
    clamp_routing_table: object | None = None
    cv_area: object | None = None  # (n_cv,) brainunit Quantity, cm^2
    point_area: object | None = None  # (n_point,) brainunit Quantity, cm^2
    pop_size: tuple[int, ...] = ()

    @classmethod
    def from_cell(cls, cell: "Cell") -> "CellRuntimeState":
        """Lower one initialized ``Cell`` declaration into runtime state.

        Parameters
        ----------
        cell : Cell
            Source cell declaration.

        Returns
        -------
        CellRuntimeState
            Runtime layout/state bridge for the declaration.

        Notes
        -----
        Dense layout buffers are allocated with shape
        ``cell.pop_size + (n_point,)``; sparse point-layout buffers use
        ``cell.pop_size + (n_active,)``.
        """
        # Compile from immutable CV declarations into runtime layouts. Dense
        # layouts cover all points with masked storage, while sparse layouts keep
        # only the active point rows for point-only mechanisms such as clamps.
        node_tree = cell.node_tree
        n_point = len(node_tree.nodes)
        n_cv = len(cell.cvs)
        cv_contexts = cell.cv_contexts

        grouped: dict[tuple[object, ...], dict[str, object]] = {}
        cv_to_layout_sets: list[set[int]] = [set() for _ in range(n_cv)]
        point_to_layout_sets: list[set[int]] = [set() for _ in range(n_point)]
        layout_id = 0
        pop_size = tuple(cell.pop_size)
        synapse_store = cell._get_synapse_store()

        def register(
            *,
            mechanism: object,
            target: str,
            cv_ids: tuple[int, ...],
            point_id: int,
            placement_id: int | None = None,
            population_index: int | None = None,
        ) -> None:
            nonlocal layout_id
            storage = "packed" if population_index is not None else "broadcast"
            signature = (target, storage) + mechanism_signature(mechanism)
            entry = grouped.get(signature)
            if entry is None:
                entry = {
                    "id": layout_id,
                    "mechanism": mechanism,
                    "target": target,
                    "cv_ids": set(),
                    "point_ids": set() if target == "density" else [],
                    "placement_ids": [],
                    "population_indices": [],
                }
                grouped[signature] = entry
                layout_id += 1
            entry["cv_ids"].update(int(cv_id) for cv_id in cv_ids)
            if target == "density":
                entry["point_ids"].add(int(point_id))
            else:
                entry["point_ids"].append(int(point_id))
                entry["placement_ids"].append(int(placement_id))
                if population_index is not None:
                    entry["population_indices"].append(int(population_index))

        for cv in cell.cvs:
            midpoint_point_id = int(node_tree.cv_to_mid_node_id[cv.id])
            for mechanism in cv.density_mech:
                register(mechanism=mechanism, target="density", cv_ids=(cv.id,), point_id=midpoint_point_id)

        for placement in cell.point_placements:
            if isinstance(placement.mechanism, SynapsePlacement):
                continue
            register(
                mechanism=placement.mechanism,
                target="point",
                cv_ids=(placement.cv_id,),
                point_id=placement.point_id,
                placement_id=placement.id,
                population_index=placement.population_index,
            )

        for synapse_type in dict.fromkeys(synapse_store.synapse_type.tolist()):
            logical_ids = synapse_store.id[synapse_store.synapse_type == synapse_type]
            if logical_ids.size == 0:
                continue
            store_rows = synapse_store.row_indices(logical_ids)
            mechanisms = tuple(synapse_store.mechanism[int(row)] for row in store_rows.tolist())
            grouped[("point", "synapse", str(synapse_type))] = {
                "id": layout_id,
                "mechanism": mechanisms[0],
                "target": "point",
                "cv_ids": set(int(synapse_store.cv_id[row]) for row in store_rows.tolist()),
                "point_ids": [int(synapse_store.point_id[row]) for row in store_rows.tolist()],
                "placement_ids": [int(synapse_store.placement_id[row]) for row in store_rows.tolist()],
                "population_indices": (
                    [int(synapse_store.population_index[row]) for row in store_rows.tolist()]
                    if len(pop_size) > 0
                    else []
                ),
                "synapse_ids": np.asarray(logical_ids, dtype=np.int64),
            }
            layout_id += 1

        layouts: list[MechanismLayout] = []
        state_shapes: dict[tuple[int, str], tuple[int, ...]] = {}
        state_buffers: dict[tuple[int, str], np.ndarray] = {}
        event_buffers: dict[int, object] = {}
        layout_mechanisms: dict[int, object] = {}
        for entry in sorted(grouped.values(), key=lambda item: int(item["id"])):
            mechanism = entry["mechanism"]
            target = str(entry["target"])
            cv_ids = tuple(sorted(int(cv_id) for cv_id in entry["cv_ids"]))
            if target == "density":
                point_ids = np.asarray(sorted(int(point_id) for point_id in entry["point_ids"]), dtype=np.int32)
                placement_index = None
            else:
                point_ids = np.asarray(entry["point_ids"], dtype=np.int32)
                placement_index = np.asarray(entry["placement_ids"], dtype=np.int32)
            population_indices = (
                np.asarray(entry["population_indices"], dtype=np.int32) if entry["population_indices"] else None
            )
            synapse_ids = entry.get("synapse_ids")
            layout = choose_layout(target=target)
            if layout == "dense":
                point_mask = np.zeros(n_point, dtype=bool)
                point_mask[point_ids] = True
                point_index = point_ids
                shape = pop_size + (n_point,)
            elif layout == "sparse":
                point_mask = None
                point_index = point_ids
                shape = (len(point_ids),) if population_indices is not None else pop_size + (len(point_ids),)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported layout {layout!r}.")

            layout_spec = MechanismLayout(
                id=int(entry["id"]),
                kind=mechanism_kind(mechanism),
                target=target,
                layout=layout,
                point_index=point_index,
                point_mask=point_mask,
                n_active=len(point_ids),
                source_cv_ids=cv_ids,
                placement_index=placement_index,
                population_index=population_indices,
                synapse_index=None if synapse_ids is None else np.asarray(synapse_ids, dtype=np.int64),
                source_rule=None,
            )
            layouts.append(layout_spec)
            layout_mechanisms[layout_spec.id] = mechanism

            for point_id in point_ids.tolist():
                point_to_layout_sets[point_id].add(layout_spec.id)
            for cv_id in cv_ids:
                cv_to_layout_sets[cv_id].add(layout_spec.id)

            if synapse_ids is not None:
                runtime_cls = get_registry().get("synapse", mechanism.synapse_type)
                event_input = runtime_cls.event_input
                if isinstance(event_input, ScalarEventInput):
                    event_buffers[layout_spec.id] = brainstate.ShortTermState(
                        u.Quantity(np.zeros((len(point_ids),), dtype=float), event_input.unit)
                    )
                elif isinstance(event_input, TriggerEventInput):
                    event_buffers[layout_spec.id] = brainstate.ShortTermState(
                        np.zeros((len(point_ids),), dtype=np.int32)
                    )
                elif not isinstance(event_input, NoEventInput):
                    raise TypeError(
                        f"Unsupported event input {type(event_input).__name__!r} for {mechanism.synapse_type!r}."
                    )
                for var_name in tuple(runtime_cls.parameters):
                    values = [
                        synapse_store.parameter_value(int(logical_id), var_name)
                        for logical_id in np.asarray(synapse_ids, dtype=np.int64).tolist()
                    ]
                    buffer = _stack_synapse_values(values, parameter=var_name)
                    state_buffers[(layout_spec.id, var_name)] = buffer
                    state_shapes[(layout_spec.id, var_name)] = (len(point_ids),)
                synapse_store.bind_runtime(
                    mechanism.synapse_type,
                    layout_spec.id,
                    np.asarray(synapse_ids, dtype=np.int64),
                )
                continue

            for var_name in _mechanism_var_names(mechanism):
                if isinstance(mechanism, CurrentClamp) and var_name == "delay":
                    quantity = _allocate_current_clamp_delay_buffer(
                        mechanism=mechanism,
                        pop_size=pop_size,
                        n_active=len(point_ids),
                    )
                    state_buffers[(layout_spec.id, var_name)] = quantity
                    state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
                    continue
                if isinstance(mechanism, CurrentClamp) and var_name in ("durations", "amplitudes"):
                    quantity, mask = _allocate_current_clamp_buffer(
                        mechanism=mechanism,
                        var_name=var_name,
                        pop_size=pop_size,
                        n_active=len(point_ids),
                    )
                    state_buffers[(layout_spec.id, var_name)] = quantity
                    state_buffers[(layout_spec.id, f"_mask_{var_name}")] = mask
                    state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
                    continue
                state_shapes[(layout_spec.id, var_name)] = shape
                value = _mechanism_var_value(mechanism, var_name)
                if (
                    isinstance(mechanism, Density)
                    and callable(value)
                    and not isinstance(value, braintools.init.Initialization)
                ):
                    state_buffers[(layout_spec.id, var_name)] = _allocate_spatial_density_buffer(
                        mechanism=mechanism,
                        var_name=var_name,
                        value=value,
                        layout=layout_spec,
                        shape=shape,
                        cv_contexts=cv_contexts,
                        node_tree=node_tree,
                    )
                else:
                    state_buffers[(layout_spec.id, var_name)] = _allocate_state_buffer(
                        mechanism,
                        var_name=var_name,
                        shape=shape,
                    )

        _apply_density_parameter_overrides(
            cell=cell,
            layouts=tuple(layouts),
            layout_mechanisms=layout_mechanisms,
            state_buffers=state_buffers,
            node_tree=node_tree,
            pop_size=pop_size,
        )

        (
            ions,
            ion_aliases,
            ion_family_candidates,
            ion_class_candidates,
            runtime_nodes,
            bound_ion_keys,
            current_owner_keys,
            merged_channel_layout_groups,
        ) = _build_runtime_nodes(
            n_point=n_point,
            layouts=tuple(layouts),
            layout_mechanisms=layout_mechanisms,
            state_buffers=state_buffers,
            pop_size=pop_size,
        )
        attach_runtime_ion_geometry(
            ions=ions,
            cvs=cell.cvs,
            point_ids=node_tree.cv_to_mid_node_id,
            n_point=n_point,
        )

        clamp_routing_table = build_clamp_routing_table(
            layouts=tuple(layouts),
            cvs=cell.cvs,
            node_tree=node_tree,
            n_point=n_point,
        )

        cv_area_decimal = np.asarray(
            [float(np.asarray(cv.area.to_decimal(u.cm**2), dtype=float)) for cv in cell.cvs],
            dtype=float,
        )
        cv_area = u.Quantity(cv_area_decimal, u.cm**2)
        point_area_decimal = np.zeros((n_point,), dtype=float)
        for point in node_tree.nodes:
            roles = tuple(point.roles)
            if len(roles) == 0:
                continue
            cv_id = int(roles[0].cv_id)
            point_area_decimal[int(point.id)] = cv_area_decimal[cv_id]
        point_area = u.Quantity(point_area_decimal, u.cm**2)
        midpoint_mask_np = np.zeros((n_point,), dtype=bool)
        midpoint_mask_np[np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)] = True

        runtime = cls(
            node_tree=node_tree,
            n_point=n_point,
            n_cv=n_cv,
            layouts=tuple(layouts),
            point_to_layout_ids=tuple(tuple(sorted(ids)) for ids in point_to_layout_sets),
            cv_to_layout_ids=tuple(tuple(sorted(ids)) for ids in cv_to_layout_sets),
            voltage_shape=pop_size + (n_point,),
            state_shapes=state_shapes,
            state_buffers=state_buffers,
            event_buffers=event_buffers,
            layout_mechanisms=layout_mechanisms,
            runtime_nodes=runtime_nodes,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
            ion_class_candidates=ion_class_candidates,
            bound_ion_keys=bound_ion_keys,
            current_owner_keys=current_owner_keys,
            midpoint_mask_np=midpoint_mask_np,
            merged_channel_layout_groups=merged_channel_layout_groups,
            dhs_static_source_np=None,
            dhs_static_cache=None,
            axial_operator_np=None,
            axial_operator_cache=None,
            clamp_routing_table=clamp_routing_table,
            cv_area=cv_area,
            point_area=point_area,
            pop_size=pop_size,
        )
        _configure_runtime_subsolvers(
            runtime,
            solver=cell.subsolver,
            substeps=cell.substeps,
        )
        return runtime

    def get_point_layouts(self, point_id: int) -> tuple[MechanismLayout, ...]:
        if not (0 <= int(point_id) < self.n_point):
            raise IndexError(f"point_id out of range: {point_id!r}.")
        ids = self.point_to_layout_ids[int(point_id)]
        return tuple(self.layouts[layout_id] for layout_id in ids)

    def get_cv_layouts(self, cv_id: int) -> tuple[MechanismLayout, ...]:
        if not (0 <= int(cv_id) < self.n_cv):
            raise IndexError(f"cv_id out of range: {cv_id!r}.")
        ids = self.cv_to_layout_ids[int(cv_id)]
        return tuple(self.layouts[layout_id] for layout_id in ids)

    def expected_state_shape(self, layout_id: int, var_name: str) -> tuple[int, ...]:
        key = (int(layout_id), str(var_name))
        if key not in self.state_shapes:
            raise KeyError(f"Unknown state shape for {(layout_id, var_name)!r}.")
        return self.state_shapes[key]

    def get_state(self, layout_id: int, var_name: str) -> np.ndarray:
        key = (int(layout_id), str(var_name))
        if key not in self.state_buffers:
            raise KeyError(f"Unknown state buffer for {(layout_id, var_name)!r}.")
        return self.state_buffers[key]

    def set_state(self, layout_id: int, var_name: str, value: object) -> None:
        key = (int(layout_id), str(var_name))
        if key not in self.state_buffers:
            raise KeyError(f"Unknown state buffer for {(layout_id, var_name)!r}.")
        layout = self.layouts[int(layout_id)]

        mask_key = (int(layout_id), f"_mask_{var_name}")
        if (
            var_name in ("durations", "amplitudes")
            and mask_key in self.state_buffers
            and isinstance(value, (tuple, list))
        ):
            buffer = self.state_buffers[key]
            if isinstance(buffer, u.Quantity):
                unit = buffer.unit
                n_active = buffer.mantissa.shape[0]
                if value and isinstance(value[0], (list, tuple)):
                    sequences = [list(row) for row in value]
                else:
                    if n_active != 1:
                        raise ValueError(
                            f"Flat sequence only valid for n_active=1 ragged clamp buffer; got n_active={n_active}."
                        )
                    sequences = [list(value)]
                if len(sequences) != n_active:
                    raise ValueError(
                        f"Ragged clamp buffer expected {n_active} per-point sequences; got {len(sequences)}."
                    )
                new_q, new_mask = _allocate_clamp_ragged_buffer(per_point_sequences=sequences, unit=unit)
                self.state_buffers[key] = new_q
                self.state_buffers[mask_key] = new_mask
                self.state_shapes[key] = new_q.mantissa.shape
                _sync_runtime_node_param(self, layout_id=int(layout_id), var_name=str(var_name))
                return

        self.state_buffers[key] = _write_state_buffer(layout, self.state_buffers[key], value)
        _sync_runtime_node_param(self, layout_id=int(layout_id), var_name=str(var_name))

    def get_point_state(self, point_id: int) -> dict[int, dict[str, object]]:
        if not (0 <= int(point_id) < self.n_point):
            raise IndexError(f"point_id out of range: {point_id!r}.")

        point_state: dict[int, dict[str, object]] = {}
        for layout in self.get_point_layouts(point_id):
            values: dict[str, object] = {}
            for buffer_key, buffer in self.state_buffers.items():
                layout_id, var_name = buffer_key
                if layout_id != layout.id:
                    continue
                values[var_name] = _extract_point_value(layout, point_id=int(point_id), buffer=buffer)
            point_state[layout.id] = values
        return point_state

    def get_cv_state(self, cv_id: int) -> dict[int, dict[str, object]]:
        if not (0 <= int(cv_id) < self.n_cv):
            raise IndexError(f"cv_id out of range: {cv_id!r}.")
        point_id = int(self.node_tree.cv_to_mid_node_id[int(cv_id)])
        return self.get_point_state(point_id)

    def get_runtime_node(self, layout_id: int) -> object:
        key = int(layout_id)
        if key not in self.runtime_nodes:
            raise KeyError(f"No runtime node is registered for layout {layout_id!r}.")
        return self.runtime_nodes[key]

    def get_layout_mechanism(self, layout_id: int) -> object:
        key = int(layout_id)
        if key not in self.layout_mechanisms:
            raise KeyError(f"No declaration mechanism is registered for layout {layout_id!r}.")
        return self.layout_mechanisms[key]

    def iter_synapse_layouts(self):
        """Yield ``(layout, runtime_synapse)`` pairs for placed synapses."""
        for layout in self.layouts:
            declaration = self.layout_mechanisms[layout.id]
            if not isinstance(declaration, SynapsePlacement):
                continue
            node = self.runtime_nodes.get(layout.id)
            if isinstance(node, RuntimeSynapse):
                yield layout, node

    def get_event_buffer(self, layout_id: int):
        """Return one private per-type event aggregation buffer."""
        try:
            return self.event_buffers[int(layout_id)].value
        except KeyError as exc:
            raise KeyError(f"Synapse layout {layout_id!r} has no discrete event input.") from exc

    def clear_event_buffer(self, layout_id: int) -> None:
        """Clear one synapse layout's pending event payload."""
        buffer = self.get_event_buffer(layout_id)
        self.event_buffers[int(layout_id)].value = u.math.zeros_like(buffer)

    def get_ion(self, name: str) -> object:
        return self.ions[self.resolve_ion_key(name)]

    def resolve_ion_key(self, name: str) -> str:
        key = str(name)
        if key in self.ions:
            return key
        alias = self.ion_aliases.get(key)
        if alias is None:
            family_candidates = self.ion_family_candidates.get(key)
            if family_candidates is not None and len(family_candidates) > 1:
                raise ValueError(
                    f"Ion selector {name!r} is ambiguous; family {key!r} has candidates {list(family_candidates)!r}."
                )
            class_candidates = self.ion_class_candidates.get(key)
            if class_candidates is not None and len(class_candidates) > 1:
                raise ValueError(
                    f"Ion selector {name!r} is ambiguous; class {key!r} has candidates {list(class_candidates)!r}."
                )
            raise KeyError(f"No ion container is registered for {name!r}.")
        return alias

    def has_layout_value(self, layout_id: int, var_name: str) -> bool:
        return (int(layout_id), str(var_name)) in self.state_buffers

    def get_layout_value(self, layout_id: int, *, point_id: int, var_name: str) -> object:
        key = (int(layout_id), str(var_name))
        if key not in self.state_buffers:
            raise KeyError(f"Unknown state buffer for {(layout_id, var_name)!r}.")
        layout = self.layouts[int(layout_id)]
        return _extract_point_value(layout, point_id=int(point_id), buffer=self.state_buffers[key])

    def evaluate_point_clamps(self, *, t, point_ids=None) -> object:
        """Evaluate clamp current on selected point-tree nodes.

        Parameters
        ----------
        t : Quantity[time]
            Absolute simulation time.
        point_ids : array-like of int or None, optional
            Optional point-id filter. When provided, only clamp layouts that
            touch these point ids are evaluated and scattered.

        Returns
        -------
        Quantity
            Total clamp current in ``nA`` with full point-space shape
            ``pop_size + (n_point,)``. Points outside ``point_ids`` are zero
            when a filter is provided.
        """
        point_current_decimal = u.math.zeros(self.pop_size + (self.n_point,), dtype=float)
        point_filter = (
            None if point_ids is None else set(int(pid) for pid in np.asarray(point_ids, dtype=np.int32).tolist())
        )
        for layout in self.layouts:
            if layout.target != "point" or layout.point_index is None:
                continue
            if layout.kind not in CLAMP_KINDS:
                continue
            if point_filter is None:
                local_indices = range(layout.n_active)
                point_index = layout.point_index
            else:
                selected = [
                    local_index
                    for local_index, point_id in enumerate(layout.point_index.tolist())
                    if int(point_id) in point_filter
                ]
                if not selected:
                    continue
                local_indices = selected
                point_index = layout.point_index[np.asarray(selected, dtype=np.int32)]
            local_currents = _evaluate_clamp_layout(
                self,
                layout=layout,
                t=t,
                local_indices=local_indices,
            )
            local_current_decimal = _quantity_sequence_to_decimal_vector(local_currents, unit=u.nA)
            point_current_decimal = point_current_decimal.at[..., point_index].add(local_current_decimal)
        return u.Quantity(point_current_decimal, u.nA)


def _apply_density_parameter_overrides(
    *,
    cell: "Cell",
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
    state_buffers: dict[tuple[int, str], object],
    node_tree: NodeTree,
    pop_size: tuple[int, ...],
) -> None:
    """Scatter declaration-time channel and ion view parameter overrides."""
    overrides = getattr(cell, "_density_parameter_overrides", {})
    if not overrides:
        return
    if len(pop_size) != 1:
        raise ValueError("Density parameter views currently require one-dimensional Cell.pop_size.")
    population_size = int(pop_size[0])

    # One pass over the layouts builds the (category, name, cv) -> layouts index that
    # every override then resolves in constant time; scanning per override is quadratic
    # in the number of selected rows, and an unrestricted view selects every CV.
    layouts_by_owner: dict[tuple[str, str, int], list[MechanismLayout]] = {}
    for layout in layouts:
        mechanism = layout_mechanisms[layout.id]
        if not isinstance(mechanism, Density):
            continue
        for source_cv_id in layout.source_cv_ids:
            owner = (mechanism.category, mechanism.instance_name, int(source_cv_id))
            layouts_by_owner.setdefault(owner, []).append(layout)

    # Resolve and validate every override first, then apply each buffer's writes in one
    # scatter -- writing them one at a time copies the whole buffer per override.
    writes: dict[tuple[int, str], list[tuple[int, int, object]]] = {}
    for (category, name, population_index, cv_id, var_name), value in overrides.items():
        matches = layouts_by_owner.get((category, name, int(cv_id)), ())
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one density layout for {category} {name!r} on CV {cv_id}, got {len(matches)!r}."
            )
        key = (int(matches[0].id), str(var_name))
        if key not in state_buffers:
            raise KeyError(f"{category.title()} {name!r} has no parameter {var_name!r}.")
        buffer = state_buffers[key]
        unit = buffer.unit if isinstance(buffer, u.Quantity) else None
        if unit is not None:
            if not isinstance(value, u.Quantity):
                raise TypeError(f"Density parameter {var_name!r} requires a Quantity compatible with {unit}.")
            decimal = value.to_decimal(unit)
        else:
            if isinstance(value, u.Quantity):
                raise TypeError(f"Density parameter {var_name!r} is dimensionless.")
            decimal = value
        point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
        writes.setdefault(key, []).append((int(population_index), point_id, decimal))

    for key, entries in writes.items():
        buffer = state_buffers[key]
        unit = buffer.unit if isinstance(buffer, u.Quantity) else None
        mantissa = np.array(buffer.to_decimal(unit) if unit is not None else buffer, copy=True)
        populations, points, values = zip(*entries)
        if mantissa.ndim >= 2 and mantissa.shape[0] == population_size:
            mantissa[np.asarray(populations), np.asarray(points)] = values
        else:
            mantissa[np.asarray(points)] = values
        state_buffers[key] = u.Quantity(mantissa, unit) if unit is not None else mantissa
