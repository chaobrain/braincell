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


import inspect
import warnings
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

import brainunit as u
import jax.numpy as jnp
import numpy as np

Target = Literal["density", "point"]
Layout = Literal["dense", "sparse"]

from braincell import ion as runtime_ion
from braincell._base import Channel, IonChannel
from braincell.ion._base import DynamicNernstIon, FixedIon, InitNernstIon, KineticIon
from braincell.mech import (
    CurrentProbe,
    CurrentClamp,
    Density,
    FunctionClamp,
    Junction,
    MechanismProbe,
    Point,
    ProbeMechanism,
    SineClamp,
    StateProbe,
    Synapse,
    get_registry,
)
from braincell.mech._params import _to_hashable
from braincell.ion import build_placeholder_ions
from braincell.morph.morphology import Morphology, clone_morpho
from braincell._multi_compartment.bridge import (
    attach_runtime_ion_geometry,
    cv_value_vector,
    fill_like,
    quantity_vector,
    scatter_midpoint_values,
)
from braincell._discretization.base import NodeTree

__all__ = [
    "CellRuntimeState",
    "ClampActiveTable",
    "MechanismLayout",
    "build_clamp_active_table",
    "build_placeholder_ions",
    "clone_morpho",
    "cv_value_vector",
    "fill_like",
    "mechanism_signature",
    "quantity_vector",
    "scatter_midpoint_values",
]


@dataclass(frozen=True)
class MechanismLayout:
    """Internal layout decision for one mechanism instance lowered onto points.

    Layouts are the runtime bridge format: they describe where one declaration
    ended up in point space, whether it is stored densely or sparsely, and which
    state buffers/runtime node belong to it.

    Important fields:

    - ``kind`` identifies the lowered mechanism family, such as a named channel
      or one of the point clamp kinds
    - ``target`` separates density-like layouts from point-only layouts
    - ``layout`` distinguishes dense storage over all points from sparse storage
      over just the active point ids
    - ``point_index`` records which points are active for this layout
    - ``source_cv_ids`` remembers which CV declarations contributed to it

    ``CellRuntimeState`` uses these records to allocate state buffers, answer
    inspection queries, and instantiate runtime nodes with the correct shapes.
    """

    id: int
    kind: str
    target: Target
    layout: Layout
    point_index: np.ndarray | None
    point_mask: np.ndarray | None
    n_active: int
    source_cv_ids: tuple[int, ...]
    source_rule: str | None = None


#: Clamp layout kinds that contribute point-space current via
#: :meth:`CellRuntimeState.evaluate_point_clamps`.
CLAMP_KINDS = frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"})


@dataclass(frozen=True)
class ClampActiveTable:
    """Active midpoint clamp points and their membrane areas.

    Building this once at compile time replaces the per-step filter
    walk the legacy ``Cell._point_clamp_input`` used, and lets
    ``currents._clamp_density`` run without iterating over layouts.

    Attributes
    ----------
    ids : np.ndarray
        ``(n_active,)`` ``int32`` sorted unique midpoint point ids that carry
        a clamp layout.
    area : np.ndarray
        ``(n_active,)`` ``float64`` membrane area in ``cm^2`` at those
        points.
    """

    ids: np.ndarray
    area: np.ndarray


def build_clamp_active_table(
    *,
    layouts: "tuple[MechanismLayout, ...]",
    cvs,
    node_tree: "NodeTree",
    n_point: int,
) -> ClampActiveTable | None:
    """Return a :class:`ClampActiveTable` or ``None`` if no clamps placed.

    Parameters
    ----------
    layouts : tuple[MechanismLayout, ...]
        All mechanism layouts from :class:`CellRuntimeState`.
    cvs : Sequence[CV]
        The cell's control volumes — source of per-CV membrane area.
    node_tree : NodeTree
        Carries ``cv_to_mid_node_id`` for CV→node mapping.
    n_point : int
        Number of nodes in ``node_tree``.

    Raises
    ------
    ValueError
        If any active midpoint clamp point has non-positive membrane area
        (would produce NaN in ``I_total / area`` division). Endpoint clamps
        are intentionally excluded here because they are total point currents
        consumed by the point-tree solver, not membrane current densities.
    """
    active: set[int] = set()
    for layout in layouts:
        if layout.target != "point" or layout.point_index is None:
            continue
        if layout.kind not in CLAMP_KINDS:
            continue
        active.update(int(pid) for pid in layout.point_index.tolist())

    if not active:
        return None

    point_area = np.zeros((n_point,), dtype=float)
    midpoint_ids: set[int] = set()
    for cv in cvs:
        pid = int(node_tree.cv_to_mid_node_id[cv.id])
        midpoint_ids.add(pid)
        point_area[pid] = float(np.asarray(cv.area.to_decimal(u.cm ** 2), dtype=float))

    active_midpoints = sorted(pid for pid in active if pid in midpoint_ids)
    if not active_midpoints:
        return None

    ids = np.asarray(active_midpoints, dtype=np.int32)
    area = point_area[ids]
    if np.any(area <= 0.0):
        bad = ids[area <= 0.0].tolist()
        raise ValueError(
            "Midpoint clamp active points must have positive membrane area; "
            f"got non-positive area at point ids {bad!r}."
        )
    return ClampActiveTable(ids=ids, area=area.astype(np.float64, copy=False))


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
    layout_mechanisms: dict[int, object]
    runtime_nodes: dict[int, object]
    ions: dict[str, object]
    ion_aliases: dict[str, str]
    ion_family_candidates: dict[str, tuple[str, ...]]
    ion_class_candidates: dict[str, tuple[str, ...]]
    bound_ion_keys: dict[int, tuple[str, ...]]
    current_owner_keys: dict[int, str | None]
    dhs_static_source_np: object | None = None
    dhs_static_cache: object | None = None
    axial_operator_np: np.ndarray | None = None
    axial_operator_cache: object | None = None
    clamp_active_table: object | None = None
    cv_area: object | None = None  # (n_cv,) brainunit Quantity, cm^2

    @classmethod
    def from_cell(cls, cell: "Cell") -> "CellRuntimeState":
        # Compile from immutable CV declarations into runtime layouts. Dense
        # layouts cover all points with masked storage, while sparse layouts keep
        # only the active point rows for point-only mechanisms such as clamps.
        node_tree = cell.node_tree
        n_point = len(node_tree.nodes)
        n_cv = len(cell.cvs)

        grouped: dict[tuple[object, ...], dict[str, object]] = {}
        cv_to_layout_sets: list[set[int]] = [set() for _ in range(n_cv)]
        point_to_layout_sets: list[set[int]] = [set() for _ in range(n_point)]
        layout_id = 0

        def register(*, mechanism: object, target: str, cv_ids: tuple[int, ...], point_id: int) -> None:
            nonlocal layout_id
            signature = (target,) + mechanism_signature(mechanism)
            entry = grouped.get(signature)
            if entry is None:
                entry = {
                    "id": layout_id,
                    "mechanism": mechanism,
                    "target": target,
                    "cv_ids": set(),
                    "point_ids": set(),
                }
                grouped[signature] = entry
                layout_id += 1
            entry["cv_ids"].update(int(cv_id) for cv_id in cv_ids)
            entry["point_ids"].add(int(point_id))

        for cv in cell.cvs:
            midpoint_point_id = int(node_tree.cv_to_mid_node_id[cv.id])
            for mechanism in cv.density_mech:
                register(mechanism=mechanism, target="density", cv_ids=(cv.id,), point_id=midpoint_point_id)

        point_mech = tuple(node.point_mech for node in node_tree.nodes)
        if any(point_mech):
            for point_id, mechanisms in enumerate(point_mech):
                source_cv_ids = _source_cv_ids_for_point(node_tree, point_id=int(point_id))
                for mechanism in mechanisms:
                    register(
                        mechanism=mechanism,
                        target="point",
                        cv_ids=source_cv_ids,
                        point_id=int(point_id),
                    )
        else:
            for cv in cell.cvs:
                midpoint_point_id = int(node_tree.cv_to_mid_node_id[cv.id])
                for mechanism in cv.point_mech:
                    register(mechanism=mechanism, target="point", cv_ids=(cv.id,), point_id=midpoint_point_id)

        layouts: list[MechanismLayout] = []
        state_shapes: dict[tuple[int, str], tuple[int, ...]] = {}
        state_buffers: dict[tuple[int, str], np.ndarray] = {}
        layout_mechanisms: dict[int, object] = {}
        for entry in sorted(grouped.values(), key=lambda item: int(item["id"])):
            mechanism = entry["mechanism"]
            target = str(entry["target"])
            cv_ids = tuple(sorted(int(cv_id) for cv_id in entry["cv_ids"]))
            point_ids = np.asarray(sorted(int(point_id) for point_id in entry["point_ids"]), dtype=np.int32)
            layout = choose_layout(target=target)
            if layout == "dense":
                point_mask = np.zeros(n_point, dtype=bool)
                point_mask[point_ids] = True
                point_index = point_ids
                shape = (n_point,)
            elif layout == "sparse":
                point_mask = None
                point_index = point_ids
                shape = (len(point_ids),)
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
                source_rule=None,
            )
            layouts.append(layout_spec)
            layout_mechanisms[layout_spec.id] = mechanism

            for point_id in point_ids.tolist():
                point_to_layout_sets[point_id].add(layout_spec.id)
            for cv_id in cv_ids:
                cv_to_layout_sets[cv_id].add(layout_spec.id)

            for var_name in _mechanism_var_names(mechanism):
                if isinstance(mechanism, CurrentClamp) and var_name in ("durations", "amplitudes"):
                    seq = mechanism.durations if var_name == "durations" else mechanism.amplitudes
                    unit = u.ms if var_name == "durations" else u.nA
                    quantity, mask = _allocate_clamp_ragged_buffer(
                        per_point_sequences=[seq] * len(point_ids),
                        unit=unit,
                    )
                    state_buffers[(layout_spec.id, var_name)] = quantity
                    state_buffers[(layout_spec.id, f"_mask_{var_name}")] = mask
                    state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
                    continue
                state_shapes[(layout_spec.id, var_name)] = shape
                state_buffers[(layout_spec.id, var_name)] = _allocate_state_buffer(
                    mechanism,
                    var_name=var_name,
                    shape=shape,
                )

        (
            ions,
            ion_aliases,
            ion_family_candidates,
            ion_class_candidates,
            runtime_nodes,
            bound_ion_keys,
            current_owner_keys,
        ) = _build_runtime_nodes(
            n_point=n_point,
            layouts=tuple(layouts),
            layout_mechanisms=layout_mechanisms,
            state_buffers=state_buffers,
        )
        attach_runtime_ion_geometry(
            ions=ions,
            cvs=cell.cvs,
            point_ids=node_tree.cv_to_mid_node_id,
            n_point=n_point,
        )

        clamp_active_table = build_clamp_active_table(
            layouts=tuple(layouts),
            cvs=cell.cvs,
            node_tree=node_tree,
            n_point=n_point,
        )

        cv_area_decimal = np.asarray(
            [float(np.asarray(cv.area.to_decimal(u.cm ** 2), dtype=float)) for cv in cell.cvs],
            dtype=float,
        )
        cv_area = u.Quantity(cv_area_decimal, u.cm ** 2)

        return cls(
            node_tree=node_tree,
            n_point=n_point,
            n_cv=n_cv,
            layouts=tuple(layouts),
            point_to_layout_ids=tuple(tuple(sorted(ids)) for ids in point_to_layout_sets),
            cv_to_layout_ids=tuple(tuple(sorted(ids)) for ids in cv_to_layout_sets),
            voltage_shape=(n_point,),
            state_shapes=state_shapes,
            state_buffers=state_buffers,
            layout_mechanisms=layout_mechanisms,
            runtime_nodes=runtime_nodes,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
            ion_class_candidates=ion_class_candidates,
            bound_ion_keys=bound_ion_keys,
            current_owner_keys=current_owner_keys,
            dhs_static_source_np=None,
            dhs_static_cache=None,
            axial_operator_np=None,
            axial_operator_cache=None,
            clamp_active_table=clamp_active_table,
            cv_area=cv_area,
        )

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
                            f"Flat sequence only valid for n_active=1 ragged clamp buffer; "
                            f"got n_active={n_active}."
                        )
                    sequences = [list(value)]
                if len(sequences) != n_active:
                    raise ValueError(
                        f"Ragged clamp buffer expected {n_active} per-point sequences; got {len(sequences)}."
                    )
                new_q, new_mask = _allocate_clamp_ragged_buffer(
                    per_point_sequences=sequences, unit=unit
                )
                self.state_buffers[key] = new_q
                self.state_buffers[mask_key] = new_mask
                self.state_shapes[key] = new_q.mantissa.shape
                _sync_runtime_node_param(self, layout_id=int(layout_id), var_name=str(var_name))
                return

        self.state_buffers[key] = _write_state_buffer(
            layout, self.state_buffers[key], value
        )
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

    def evaluate_point_clamps(self, *, t) -> object:
        point_current_decimal = u.math.zeros((self.n_point,), dtype=float)
        for layout in self.layouts:
            if layout.target != "point" or layout.point_index is None:
                continue
            if layout.kind not in CLAMP_KINDS:
                continue
            local_currents = _evaluate_clamp_layout(self, layout=layout, t=t)
            if len(local_currents) != len(layout.point_index):
                raise ValueError(
                    f"Clamp layout {layout.id!r} produced {len(local_currents)!r} currents "
                    f"for {len(layout.point_index)!r} active points."
                )
            # Clamp current is accumulated directly in point space so a placed
            # mechanism only affects its target points instead of broadcasting to
            # all CVs/points.
            local_current_decimal = _quantity_sequence_to_decimal_vector(local_currents, unit=u.nA)
            point_current_decimal = point_current_decimal.at[layout.point_index].add(local_current_decimal)
        return u.Quantity(point_current_decimal, u.nA)


## build_placeholder_ions moved to braincell.ion
## clone_morpho moved to braincell.morph.morphology
## scatter/gather/quantity/geometry helpers moved to
##   braincell._multi_compartment.bridge


def _source_cv_ids_for_point(node_tree: NodeTree, *, point_id: int) -> tuple[int, ...]:
    """Return CV ids whose local roles collapsed into ``point_id``."""

    point = node_tree.nodes[int(point_id)]
    cv_ids = sorted({int(role.cv_id) for role in point.roles})
    return tuple(cv_ids)


def choose_layout(*, target: Target) -> Layout:
    if target == "point":
        return "sparse"
    if target == "density":
        return "dense"
    raise ValueError(f"Unsupported target {target!r}.")


def mechanism_kind(mechanism: object) -> str:
    """Return a stable string tag describing the mechanism's type.

    For :class:`Density` mechanisms the tag is
    ``"{category}:{class_name}"``. :class:`Point` mechanisms use their
    class ``__name__`` (``CurrentClamp``, ``SineClamp``,
    ``StateProbe`` / ``MechanismProbe`` / ``CurrentProbe`` / ``ProbeMechanism`` appending
    their selector fields for debuggability.
    """
    if isinstance(mechanism, Density):
        return f"{mechanism.category}:{mechanism.class_name}"
    if isinstance(mechanism, Synapse):
        return f"synapse:{mechanism.synapse_type}"
    if isinstance(mechanism, StateProbe):
        return f"state_probe:{mechanism.field}:{mechanism.name}"
    if isinstance(mechanism, MechanismProbe):
        return f"mechanism_probe:{mechanism.mechanism}:{mechanism.field}:{mechanism.name}"
    if isinstance(mechanism, CurrentProbe):
        target = mechanism.mechanism if mechanism.mechanism is not None else mechanism.ion
        return f"current_probe:{target}:{mechanism.name}"
    if isinstance(mechanism, ProbeMechanism):
        return f"probe:{mechanism.variable}:{mechanism.target}"
    if isinstance(mechanism, Point):
        return type(mechanism).__name__
    return type(mechanism).__name__


_opaque_warned: set = set()


def _fn_fingerprint(fn) -> tuple:
    """Produce a hashable fingerprint for a callable ``fn``.

    Structurally identical lambdas (same bytecode, consts, varnames,
    and closure-cell contents) yield the same fingerprint, so two
    separately-constructed ``lambda`` objects can merge into one
    :class:`MechanismLayout` when used inside :class:`FunctionClamp`.
    Non-hashable / opaque closure cells fall back to ``id(value)``;
    such lambdas will not dedup with textually identical siblings, so
    a :class:`RuntimeWarning` is emitted once per call-site pointing
    the user at the `hoist to module level` fix.
    """
    code = fn.__code__
    closure_cells: list[object] = []
    opaque_hit = False
    for cell in (fn.__closure__ or ()):
        v = cell.cell_contents
        if hasattr(v, "to_decimal") and hasattr(v, "unit"):
            closure_cells.append(("quantity", float(v.to_decimal(v.unit)), str(v.unit)))
        elif isinstance(v, (int, float, str, bytes, bool)) or v is None:
            closure_cells.append(("prim", v))
        else:
            closure_cells.append(("id", id(v)))
            opaque_hit = True
    if opaque_hit:
        site = (code.co_filename, code.co_firstlineno)
        if site not in _opaque_warned:
            _opaque_warned.add(site)
            warnings.warn(
                f"FunctionClamp.fn at {site[0]}:{site[1]} has an opaque closure "
                "cell; two textually identical lambdas will dedup as distinct. "
                "Hoist to module level with a named function to recover dedup.",
                RuntimeWarning,
                stacklevel=2,
            )
    return (code.co_code, code.co_consts, code.co_varnames, tuple(closure_cells))


def mechanism_signature(mechanism: object) -> tuple[object, ...]:
    """Return a hashable signature used to group declarations.

    Most supported mechanism types are frozen dataclasses with
    structural equality, so the signature reduces to
    ``(type_name, hashable_field_view)``. :class:`FunctionClamp` is
    special-cased: its ``fn`` field is compared by identity under the
    dataclass-generated ``__eq__``, so we fingerprint the callable by
    bytecode + normalized closure so structurally identical lambdas merge
    into one layout.
    """
    if isinstance(mechanism, FunctionClamp):
        return (
            "FunctionClamp",
            _fn_fingerprint(mechanism.fn),
            _to_hashable(mechanism.start),
            _to_hashable(mechanism.duration),
        )
    return (type(mechanism).__qualname__, _to_hashable(mechanism))


def _mechanism_var_names(mechanism: object) -> tuple[str, ...]:
    """Return the state-buffer variable names for a mechanism.

    For :class:`Density` this is the declared ``params`` keys. For
    synapses and junctions it is the parameter keys (or a single
    default name when empty). For clamps it is the concrete dataclass
    field names. The v1 probe declarations do not allocate their own
    state buffers; they are read through explicit sampling helpers.
    """
    if isinstance(mechanism, Density):
        return tuple(mechanism.params.keys())
    if isinstance(mechanism, Synapse):
        names = tuple(mechanism.params.keys())
        return names if names else ("g",)
    if isinstance(mechanism, Junction):
        names = tuple(mechanism.params.keys())
        return names if names else ("conductance",)
    if isinstance(mechanism, CurrentClamp):
        return ("start", "durations", "amplitudes")
    if isinstance(mechanism, SineClamp):
        return ("amplitude", "frequency", "phase", "offset", "start", "duration")
    if isinstance(mechanism, FunctionClamp):
        return ("fn", "start", "duration")
    if isinstance(mechanism, (StateProbe, MechanismProbe, CurrentProbe)):
        return ()
    if isinstance(mechanism, ProbeMechanism):
        return (mechanism.variable,)
    if is_dataclass(mechanism):
        return tuple(field.name for field in fields(mechanism))
    return ("value",)


def _mechanism_var_value(mechanism: object, var_name: str) -> object:
    if isinstance(mechanism, Density):
        if var_name not in mechanism.params:
            raise KeyError(f"Mechanism has no parameter {var_name!r}.")
        return mechanism.params[var_name]
    if isinstance(mechanism, (Synapse, Junction)):
        if var_name in mechanism.params:
            return mechanism.params[var_name]
    if hasattr(mechanism, var_name):
        return getattr(mechanism, var_name)
    raise KeyError(
        f"Mechanism {type(mechanism).__name__} has no attribute {var_name!r}."
    )


def _allocate_clamp_ragged_buffer(
    *,
    per_point_sequences: list,
    unit,
) -> tuple:
    """Pack ragged per-point sequences into ``(Quantity 2D, bool mask 2D)``.

    Each row ``i`` is zero-padded up to ``max_steps``; ``mask[i, j]`` is
    ``True`` where the original sequence had a value. :func:`_eval_current_clamp`
    multiplies through the mask so padded slots contribute nothing.
    """
    if not per_point_sequences:
        raise ValueError("Ragged clamp buffer requires at least one sequence.")
    max_steps = max(len(seq) for seq in per_point_sequences)
    n_active = len(per_point_sequences)
    mantissa = np.zeros((n_active, max_steps), dtype=np.float64)
    mask = np.zeros((n_active, max_steps), dtype=bool)
    for i, seq in enumerate(per_point_sequences):
        for j, item in enumerate(seq):
            mantissa[i, j] = float(item.to_decimal(unit))
            mask[i, j] = True
    return u.Quantity(mantissa, unit), mask


def _is_ragged_param(value: object) -> bool:
    """True for callable / tuple / list param values.

    Ragged params include :class:`CurrentClamp` ``durations`` /
    ``amplitudes`` (tuple of Quantity) and :class:`FunctionClamp`
    ``fn`` (callable). These are stored per-point as a Python tuple
    buffer rather than a rectangular :class:`u.Quantity` array.
    """
    if callable(value):
        return True
    if isinstance(value, (tuple, list)):
        return True
    return False


def _allocate_state_buffer(mechanism: object, *, var_name: str, shape: tuple[int, ...]) -> object:
    """Allocate a state buffer for one mechanism parameter.

    Returns a :class:`u.Quantity` whose mantissa is a :class:`jnp.ndarray`
    when the declared value carries a unit. For ragged sequence / callable
    values (``CurrentClamp.durations`` / ``.amplitudes`` / ``FunctionClamp.fn``)
    the buffer is a Python tuple of length ``shape[0]`` (handled in Task 13).
    Plain numeric values (no unit) become a :class:`jnp.ndarray`.
    """
    value = _mechanism_var_value(mechanism, var_name)

    if _is_ragged_param(value):
        n = int(np.prod(shape, dtype=int)) if shape else 1
        return tuple(value for _ in range(n))

    if hasattr(value, "unit") and hasattr(value, "to_decimal"):
        unit = value.unit
        mantissa = np.full(shape, float(value.to_decimal(unit)), dtype=np.float64)
        return u.Quantity(mantissa, unit)

    return np.full(shape, value, dtype=np.float64)


def _write_state_buffer(layout: "MechanismLayout", buffer: object, value: object) -> object:
    """Write ``value`` into ``buffer``; return the possibly-new buffer.

    - Quantity buffer: broadcast scalar Quantity, validate unit and shape.
    - Tuple buffer (ragged): replace whole tuple, or fill every slot with a
      scalar.
    - Plain ``jnp.ndarray`` buffer: broadcast scalar, validate shape.
    """
    if isinstance(buffer, u.Quantity):
        target_shape = buffer.mantissa.shape
        target_unit = buffer.unit

        if isinstance(value, u.Quantity):
            mantissa = np.asarray(value.to_decimal(target_unit), dtype=np.float64)
            if mantissa.ndim == 0:
                mantissa = np.broadcast_to(mantissa, target_shape).copy()
            if mantissa.shape != target_shape:
                raise ValueError(
                    f"State assignment shape mismatch: expected {target_shape!r}, got {mantissa.shape!r}."
                )
            return u.Quantity(mantissa, target_unit)

        if isinstance(value, (list, tuple)):
            if len(target_shape) == 2 and value and isinstance(value[0], (list, tuple)):
                rows = [
                    [float(np.asarray(q.to_decimal(target_unit))) for q in row]
                    for row in value
                ]
                arr = np.asarray(rows, dtype=np.float64)
            else:
                decimals = [float(np.asarray(q.to_decimal(target_unit))) for q in value]
                arr = np.asarray(decimals, dtype=np.float64)
                if (
                    len(target_shape) == 2
                    and arr.ndim == 1
                    and target_shape[0] == 1
                    and arr.shape[0] == target_shape[1]
                ):
                    arr = arr.reshape(target_shape)
            if arr.shape != target_shape:
                raise ValueError(
                    f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}."
                )
            return u.Quantity(arr, target_unit)

        raise TypeError(
            f"State buffer for layout {layout.id!r} expects a Quantity or sequence of Quantities, "
            f"got {type(value).__name__!r}."
        )

    if isinstance(buffer, tuple):
        if isinstance(value, (tuple, list)):
            if len(buffer) == 1:
                return (tuple(value),)
            if len(value) != len(buffer):
                raise ValueError(
                    f"State assignment shape mismatch for ragged buffer: expected length {len(buffer)}, got {len(value)}."
                )
            return tuple(value)
        return tuple(value for _ in buffer)

    arr = np.asarray(value, dtype=np.float64)
    target_shape = np.asarray(buffer).shape
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, target_shape).copy()
    if arr.shape != target_shape:
        raise ValueError(
            f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}."
        )
    return arr


def _extract_point_value(layout: MechanismLayout, *, point_id: int, buffer: object) -> object:
    def _pick(index: int) -> object:
        if isinstance(buffer, u.Quantity):
            return u.Quantity(buffer.mantissa[index], buffer.unit)
        if isinstance(buffer, tuple):
            return buffer[index]
        return buffer[index]

    if layout.layout == "dense":
        return _pick(int(point_id))
    if layout.point_index is None:
        raise ValueError(f"Sparse layout {layout.id!r} is missing point_index.")
    matches = np.flatnonzero(layout.point_index == int(point_id))
    if len(matches) == 0:
        raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
    return _pick(int(matches[0]))


def _evaluate_clamp_layout(runtime: CellRuntimeState, *, layout: MechanismLayout, t) -> tuple[object, ...]:
    if layout.layout != "sparse" or layout.point_index is None:
        raise ValueError(f"Clamp layout {layout.id!r} must be sparse with point_index.")
    local_t = (t - _scalar_state_value(runtime, layout_id=layout.id, var_name="start")).in_unit(u.ms)
    out: list[object] = []
    for local_index in range(layout.n_active):
        if layout.kind == "CurrentClamp":
            out.append(_eval_current_clamp(runtime, layout_id=layout.id, local_index=local_index, local_t=local_t))
            continue
        if layout.kind == "SineClamp":
            out.append(_eval_sine_clamp(runtime, layout_id=layout.id, local_index=local_index, local_t=local_t))
            continue
        if layout.kind == "FunctionClamp":
            out.append(_eval_function_clamp(runtime, layout_id=layout.id, local_index=local_index, local_t=local_t))
            continue
        raise ValueError(f"Unsupported clamp layout kind {layout.kind!r}.")
    return tuple(out)


def _scalar_state_value(runtime: CellRuntimeState, *, layout_id: int, var_name: str, local_index: int = 0) -> object:
    buffer = runtime.state_buffers[(int(layout_id), str(var_name))]
    index = int(local_index)
    if isinstance(buffer, u.Quantity):
        return u.Quantity(buffer.mantissa[index], buffer.unit)
    if isinstance(buffer, tuple):
        return buffer[index]
    return buffer.reshape((-1,))[index]


def _quantity_sequence_to_decimal_vector(values: object, *, unit: object) -> object:
    if hasattr(values, "to_decimal"):
        return u.math.asarray(values.to_decimal(unit))
    decimals = [item.to_decimal(unit) for item in values]
    return u.math.asarray(decimals)


def _eval_current_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    """Evaluate a :class:`CurrentClamp` step protocol at a padded local row.

    Uses the ``(n_active, max_steps)`` :class:`u.Quantity` buffer paired
    with the bool ``_mask_durations`` buffer: padded slots are masked
    out so they never contribute to the accumulated current.
    """
    durations_q = runtime.state_buffers[(int(layout_id), "durations")]
    amplitudes_q = runtime.state_buffers[(int(layout_id), "amplitudes")]
    mask = runtime.state_buffers[(int(layout_id), "_mask_durations")]

    dur_row = jnp.asarray(durations_q.mantissa[int(local_index)])
    amp_row = jnp.asarray(amplitudes_q.mantissa[int(local_index)])
    mask_row = jnp.asarray(mask[int(local_index)])

    local_t_ms = local_t.to_decimal(u.ms)
    ends = jnp.cumsum(dur_row)
    starts = ends - dur_row
    is_active = (local_t_ms >= 0.0) & (local_t_ms >= starts) & (local_t_ms < ends) & mask_row
    current = jnp.sum(jnp.where(is_active, amp_row, 0.0))
    return u.Quantity(current, u.nA)


def _eval_sine_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    duration = _scalar_state_value(runtime, layout_id=layout_id, var_name="duration", local_index=local_index)
    amplitude_decimal = _scalar_state_value(runtime, layout_id=layout_id, var_name="amplitude",
                                            local_index=local_index).to_decimal(u.nA)
    offset_decimal = _scalar_state_value(runtime, layout_id=layout_id, var_name="offset",
                                         local_index=local_index).to_decimal(u.nA)
    frequency = _scalar_state_value(runtime, layout_id=layout_id, var_name="frequency", local_index=local_index)
    phase = u.math.asarray(_scalar_state_value(runtime, layout_id=layout_id, var_name="phase", local_index=local_index))
    local_t_ms = local_t.to_decimal(u.ms)
    active = u.math.logical_and(local_t_ms >= 0.0, local_t < duration)
    angle = 2.0 * np.pi * frequency.to_decimal(u.Hz) * local_t.to_decimal(u.second) + phase
    current_decimal = offset_decimal + (u.math.sin(angle) * amplitude_decimal)
    return u.Quantity(u.math.where(active, current_decimal, 0.0), u.nA)


def _eval_function_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    duration = _scalar_state_value(runtime, layout_id=layout_id, var_name="duration", local_index=local_index)
    fn = _scalar_state_value(runtime, layout_id=layout_id, var_name="fn", local_index=local_index)
    value = fn(local_t)
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"FunctionClamp fn must return a current quantity, got {value!r}.")
    shape = getattr(value, "shape", ())
    if shape not in ((), None):
        raise ValueError(f"FunctionClamp fn must return a scalar current, got shape {shape!r}.")
    active = u.math.logical_and(local_t.to_decimal(u.ms) >= 0.0, local_t < duration)
    return u.Quantity(u.math.where(active, value.to_decimal(u.nA), 0.0), u.nA)


def _build_runtime_nodes(
    *,
    n_point: int,
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[int, object],
    dict[int, tuple[str, ...]],
    dict[int, str | None],
]:
    (
        ions,
        ion_aliases,
        ion_family_candidates,
        ion_class_candidates,
        ion_runtime_nodes,
    ) = _build_runtime_ions(
        n_point=n_point,
        layouts=layouts,
        layout_mechanisms=layout_mechanisms,
        state_buffers=state_buffers,
    )
    runtime_nodes: dict[int, object] = dict(ion_runtime_nodes)
    bound_ion_keys: dict[int, tuple[str, ...]] = {}
    current_owner_keys: dict[int, str | None] = {}
    for layout in layouts:
        mechanism = layout_mechanisms[layout.id]
        node, layout_bound_ion_keys, current_owner_key = _instantiate_runtime_node(
            layout=layout,
            mechanism=mechanism,
            state_buffers=state_buffers,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
        )
        if node is not None:
            runtime_nodes[layout.id] = node
            bound_ion_keys[layout.id] = layout_bound_ion_keys
            current_owner_keys[layout.id] = current_owner_key
    return ions, ion_aliases, ion_family_candidates, ion_class_candidates, runtime_nodes, bound_ion_keys, current_owner_keys


def _build_runtime_ions(
    *,
    n_point: int,
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[int, object],
]:
    ion_instances, ion_family_candidates = _collect_runtime_ion_instances(
        layouts=layouts,
        layout_mechanisms=layout_mechanisms,
    )
    ions: dict[str, object] = {}
    ion_class_candidates: dict[str, list[str]] = {}
    ion_runtime_nodes: dict[int, object] = {}

    for instance_name, record in ion_instances.items():
        runtime_ion = _instantiate_runtime_ion_instance(
            instance_name=instance_name,
            runtime_cls=record["runtime_cls"],
            layouts=tuple(record["layouts"]),
            declarations=tuple(record["declarations"]),
            state_buffers=state_buffers,
            n_point=n_point,
        )
        ions[instance_name] = runtime_ion
        ion_class_candidates.setdefault(record["runtime_cls"].__name__, []).append(instance_name)
        for layout in record["layouts"]:
            ion_runtime_nodes[layout.id] = runtime_ion

    for family_key in ("na", "k", "ca"):
        if family_key in ion_family_candidates:
            continue
        default_ion = _build_default_ions(n_point)[family_key]
        ions[family_key] = default_ion
        ion_family_candidates[family_key] = [family_key]
        ion_class_candidates.setdefault(type(default_ion).__name__, []).append(family_key)

    ion_aliases = _build_ion_alias_map(
        ions=ions,
        ion_family_candidates=ion_family_candidates,
        ion_class_candidates=ion_class_candidates,
    )
    return (
        ions,
        ion_aliases,
        {key: tuple(value) for key, value in ion_family_candidates.items()},
        {key: tuple(value) for key, value in ion_class_candidates.items()},
        ion_runtime_nodes,
    )


def _build_default_ions(n_point: int) -> dict[str, object]:
    return build_placeholder_ions(size=(n_point,))


def _collect_runtime_ion_instances(
    *,
    layouts: tuple[MechanismLayout, ...],
    layout_mechanisms: dict[int, object],
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    instances: dict[str, dict[str, object]] = {}
    family_candidates: dict[str, list[str]] = {}

    for layout in layouts:
        if layout.target != "density":
            continue
        mechanism = layout_mechanisms[layout.id]
        if not isinstance(mechanism, Density) or mechanism.category != "ion":
            continue
        runtime_cls = get_registry().get("ion", mechanism.class_name)
        species_key = _runtime_ion_species_key(runtime_cls)
        family = _runtime_ion_family(runtime_cls)

        instance_name = mechanism.instance_name
        if instance_name in {"na", "k", "ca"} and instance_name != species_key:
            raise ValueError(
                f"Ion instance name {instance_name!r} conflicts with canonical family key for a different ion family."
            )
        record = instances.get(instance_name)
        if record is None:
            record = {
                "runtime_cls": runtime_cls,
                "family": family,
                "layouts": [],
                "declarations": [],
            }
            instances[instance_name] = record
            family_candidates.setdefault(species_key, []).append(instance_name)
        elif record["runtime_cls"] is not runtime_cls:
            raise ValueError(
                f"Ion instance name {instance_name!r} cannot mix classes "
                f"{record['runtime_cls'].__name__!r} and {runtime_cls.__name__!r}."
            )
        elif _runtime_ion_species_key(record["runtime_cls"]) != species_key:
            raise ValueError(
                f"Ion instance name {instance_name!r} cannot be reused across families "
                f"{_runtime_ion_species_key(record['runtime_cls'])!r} and {species_key!r}."
            )

        record["layouts"].append(layout)
        record["declarations"].append(mechanism)

    return instances, family_candidates


def _build_ion_alias_map(
    *,
    ions: dict[str, object],
    ion_family_candidates: dict[str, list[str]],
    ion_class_candidates: dict[str, list[str]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}

    def register(alias: str, canonical: str) -> None:
        existing = aliases.get(alias)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"Ion alias {alias!r} conflicts between species {existing!r} and {canonical!r}."
            )
        aliases[alias] = canonical

    for instance_name in ions:
        register(instance_name, instance_name)

    for family_key, candidates in ion_family_candidates.items():
        if len(candidates) == 1:
            register(family_key, candidates[0])

    for class_name, candidates in ion_class_candidates.items():
        unique_candidates = tuple(dict.fromkeys(candidates))
        if len(unique_candidates) == 1:
            register(class_name, unique_candidates[0])

    return aliases


def _runtime_ion_species_key(cls: type) -> str:
    if issubclass(cls, runtime_ion.Sodium):
        return "na"
    if issubclass(cls, runtime_ion.Potassium):
        return "k"
    if issubclass(cls, runtime_ion.Calcium):
        return "ca"
    raise ValueError(f"Unsupported ion runtime class {cls.__name__!r}: cannot infer species key.")


def _runtime_ion_family(cls: type) -> str:
    if issubclass(cls, KineticIon):
        return "kinetic"
    if issubclass(cls, DynamicNernstIon):
        return "dynamic"
    if issubclass(cls, InitNernstIon):
        return "init_nernst"
    if issubclass(cls, FixedIon):
        return "fixed"
    raise ValueError(f"Unsupported ion runtime class {cls.__name__!r}: unsupported ion template family.")


def _supported_ion_runtime_params(cls: type) -> tuple[str, ...]:
    signature = inspect.signature(cls.__init__)
    supported: list[str] = []
    excluded = {"solver", "substeps", "species_initializers"}
    for name, parameter in signature.parameters.items():
        if name in {"self", "size", "name"}:
            continue
        if name in excluded:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        supported.append(name)
    return tuple(supported)


def _ion_runtime_attr_name(cls: type, param_name: str) -> str:
    if param_name == "Ci_initializer" and issubclass(cls, DynamicNernstIon):
        return "_Ci_initializer"
    return param_name


def _normalize_ion_runtime_param_value(cls: type, param_name: str, value: object) -> object:
    if param_name == "Ci_initializer" and issubclass(cls, DynamicNernstIon):
        inner = getattr(value, "value", None)
        if hasattr(inner, "to_decimal"):
            return inner
    return value


def _instantiate_runtime_ion_instance(
    *,
    instance_name: str,
    runtime_cls: type,
    layouts: tuple[MechanismLayout, ...],
    declarations: tuple[Density, ...],
    state_buffers: dict,
    n_point: int,
) -> object:
    """Build one runtime ion instance from its sparse declaration layouts.

    Start from a baseline ion and replace per-point params where each
    declaration layout requests them. Uses ``.at[point_index].set(...)``
    on the Quantity mantissa — no Python loops on per-point Quantity
    boxes.
    """
    supported_params = _supported_ion_runtime_params(runtime_cls)
    unsupported_params: dict[int, set[str]] = {}
    for layout, declaration in zip(layouts, declarations):
        invalid = set(declaration.params.keys()) - set(supported_params)
        if invalid:
            unsupported_params[layout.id] = invalid
    if unsupported_params:
        layout_id, invalid = next(iter(unsupported_params.items()))
        raise ValueError(
            f"Ion layout {layout_id!r} for instance {instance_name!r} uses unsupported runtime ion params "
            f"{sorted(invalid)!r} on {runtime_cls.__name__!r}."
        )

    baseline_ion = runtime_cls(size=(n_point,))
    full_param_values: dict[str, object] = {}
    for param_name in supported_params:
        baseline_value = _normalize_ion_runtime_param_value(
            runtime_cls,
            param_name,
            getattr(baseline_ion, _ion_runtime_attr_name(runtime_cls, param_name)),
        )
        full_param_values[param_name] = _ion_param_broadcast(baseline_value, shape=(n_point,))

    for layout, declaration in zip(layouts, declarations):
        point_index = layout.point_index
        if point_index is None:
            raise ValueError(f"Ion layout {layout.id!r} is missing point_index.")
        for param_name in declaration.params.keys():
            buffer = state_buffers[(layout.id, param_name)]
            full_param_values[param_name] = _ion_param_scatter(
                runtime_cls=runtime_cls,
                param_name=param_name,
                target=full_param_values[param_name],
                buffer=buffer,
                point_index=point_index,
            )

    return runtime_cls(size=(n_point,), name=instance_name, **full_param_values)


def _ion_param_broadcast(value: object, *, shape: tuple[int, ...]) -> object:
    """Broadcast an ion baseline value onto ``shape``.

    Handles three cases: already-shaped Quantity pass-through, scalar
    Quantity broadcast, and plain numeric / object fallbacks. Returns
    a buffer that :func:`_ion_param_scatter` can update in-place by
    ``.at[...].set(...)``.
    """
    if isinstance(value, u.Quantity):
        raw = value.mantissa if hasattr(value, "mantissa") else value.to_decimal(value.unit)
        mantissa = np.asarray(raw, dtype=np.float64)
        if mantissa.shape == shape:
            return u.Quantity(mantissa.copy(), value.unit)
        if mantissa.ndim == 0 or mantissa.shape == ():
            return u.Quantity(np.full(shape, float(mantissa), dtype=np.float64), value.unit)
        raise ValueError(
            f"Cannot broadcast ion baseline value with shape {mantissa.shape!r} onto shape {shape!r}."
        )
    # Plain numeric baseline (e.g., valence): broadcast as numpy array.
    if isinstance(value, (np.ndarray,)) or isinstance(value, (int, float)):
        arr = np.asarray(value)
        if arr.shape == shape:
            return arr.copy()
        if arr.ndim == 0:
            return np.broadcast_to(arr, shape).copy()
    if hasattr(value, "shape") and not callable(value):
        arr = np.asarray(value)
        if arr.shape == shape:
            return arr.copy()
        if arr.ndim == 0:
            return np.broadcast_to(arr, shape).copy()
    # Callable / opaque baseline: keep as tuple of length shape[0].
    n = int(np.prod(shape, dtype=int)) if shape else 1
    return tuple(value for _ in range(n))


def _ion_param_scatter(
    *,
    runtime_cls: type,
    param_name: str,
    target: object,
    buffer: object,
    point_index: np.ndarray,
) -> object:
    """Scatter the values of one sparse ion-layout buffer into ``target``.

    For ``Ci_initializer`` on :class:`DynamicNernstIon` (which may hold a
    State-wrapped callable), fall back to the Python per-point path on a
    tuple buffer. Rectangular Quantity buffers scatter via
    ``.at[point_index].set(...)`` with unit coercion.
    """
    if isinstance(target, u.Quantity) and isinstance(buffer, u.Quantity):
        target_unit = target.unit
        src_mantissa = np.asarray(buffer.mantissa, dtype=np.float64)
        # Sparse buffer has shape (n_active,) matching point_index; dense has (n_point,).
        src = src_mantissa if src_mantissa.shape == point_index.shape else src_mantissa[point_index]
        incoming = np.asarray(
            u.Quantity(src, buffer.unit).to_decimal(target_unit), dtype=np.float64
        )
        new_mantissa = np.asarray(target.mantissa, dtype=np.float64).copy()
        new_mantissa[point_index] = incoming
        return u.Quantity(new_mantissa, target_unit)

    if isinstance(target, tuple):
        target_list = list(target)
        for local_idx, global_idx in enumerate(point_index.tolist()):
            if isinstance(buffer, u.Quantity):
                target_list[int(global_idx)] = u.Quantity(buffer.mantissa[local_idx], buffer.unit)
            elif isinstance(buffer, tuple):
                target_list[int(global_idx)] = buffer[local_idx]
            else:
                target_list[int(global_idx)] = buffer[local_idx]
        return tuple(target_list)

    if isinstance(target, np.ndarray):
        new_target = target.copy()
        if isinstance(buffer, u.Quantity):
            src = np.asarray(buffer.mantissa)
        elif isinstance(buffer, np.ndarray):
            src = buffer
        else:
            raise TypeError(
                f"Cannot scatter non-array buffer into numpy target for ion param {param_name!r}."
            )
        src = src if src.shape == point_index.shape else src[point_index]
        new_target[point_index] = src
        return new_target

    raise TypeError(
        f"Unsupported target/buffer combination for ion param {param_name!r}: "
        f"target={type(target).__name__}, buffer={type(buffer).__name__}."
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

    def update(self, V, *unused):
        return self._channel.update(V, *self._infos())

    def current(self, V, *unused):
        return self._channel.current(V, *self._infos())


def _instantiate_runtime_node(
    *,
    layout: MechanismLayout,
    mechanism: object,
    state_buffers: dict[tuple[int, str], np.ndarray],
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
) -> tuple[object | None, tuple[str, ...], str | None]:
    if layout.target != "density" or layout.layout != "dense":
        return None, (), None
    if not isinstance(mechanism, Density):
        return None, (), None
    if mechanism.category != "channel":
        return None, (), None

    runtime_cls = get_registry().get("channel", mechanism.class_name)
    params = _runtime_constructor_params(
        layout=layout, mechanism=mechanism, state_buffers=state_buffers
    )
    if len(params) > 0 and hasattr(next(iter(params.values())), "shape"):
        size = next(iter(params.values())).shape
    elif layout.layout == "dense" and layout.point_mask is not None:
        size = layout.point_mask.shape
    else:
        size = (layout.n_active,)
    node = runtime_cls(size=size, **params)
    bound_ions, current_owner_key = _resolve_channel_runtime_bindings(
        runtime_cls=runtime_cls,
        mechanism=mechanism,
        ions=ions,
        ion_aliases=ion_aliases,
        ion_family_candidates=ion_family_candidates,
    )
    if current_owner_key is not None:
        channel_key = mechanism.instance_name
        owner_ion = ions[current_owner_key]
        if len(bound_ions) == 1 and bound_ions[0][0] == current_owner_key:
            owner_ion.add(**{channel_key: node})
        else:
            wrapper = _BoundIonChannelRuntime(
                node,
                bound_ions=tuple(ion for _, ion in bound_ions),
                owner_ion=owner_ion,
            )
            owner_ion.add(**{channel_key: wrapper})
    return node, tuple(ion_key for ion_key, _ in bound_ions), current_owner_key


def _resolve_channel_runtime_bindings(
    *,
    runtime_cls: type,
    mechanism: Density,
    ions: dict[str, object],
    ion_aliases: dict[str, str],
    ion_family_candidates: dict[str, tuple[str, ...]],
) -> tuple[tuple[tuple[str, object], ...], str | None]:
    family_slots = _channel_family_slots(runtime_cls)
    if len(family_slots) == 0:
        if getattr(mechanism, "ion_name", None) is not None or getattr(mechanism, "ion_names", None) is not None:
            raise ValueError(
                f"Channel {mechanism.class_name!r} does not bind ions but ion selectors were provided."
            )
        return (), None

    if len(family_slots) == 1:
        if getattr(mechanism, "ion_names", None) is not None:
            raise ValueError(
                f"Single-ion channel {mechanism.class_name!r} must use ion_name, not ion_names."
            )
        family_key = family_slots[0][0]
        ion_key = _resolve_ion_instance_key(
            family_key=family_key,
            selector=getattr(mechanism, "ion_name", None),
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=ion_family_candidates,
        )
        return ((ion_key, ions[ion_key]),), ion_key

    if getattr(mechanism, "ion_name", None) is not None:
        raise ValueError(
            f"Mixed-ion channel {mechanism.class_name!r} must use ion_names, not ion_name."
        )
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

    current_owner_family = _channel_current_owner_family(runtime_cls)
    if current_owner_family is None:
        raise ValueError(
            f"Mixed-ion channel class {runtime_cls.__name__!r} must define current_owner_type."
        )
    owner_candidates = [ion_key for (family_key, _), (ion_key, _) in zip(family_slots, bound_ions) if
                        family_key == current_owner_family]
    if len(owner_candidates) != 1:
        raise ValueError(
            f"Mixed-ion channel class {runtime_cls.__name__!r} could not resolve a unique current owner for family "
            f"{current_owner_family!r}."
        )
    return tuple(bound_ions), owner_candidates[0]


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
    try:
        if issubclass(root_type, runtime_ion.Sodium):
            return "na"
        if issubclass(root_type, runtime_ion.Potassium):
            return "k"
        if issubclass(root_type, runtime_ion.Calcium):
            return "ca"
    except TypeError:
        return None
    return None


def _channel_current_owner_family(cls: type) -> str | None:
    family_slots = _channel_family_slots(cls)
    if len(family_slots) == 0:
        return None
    if len(family_slots) == 1:
        return family_slots[0][0]
    owner_type = getattr(cls, "current_owner_type", None)
    if owner_type is None:
        return None
    return _root_type_to_family(owner_type)


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
    if (
        isinstance(buffer, u.Quantity)
        and layout.point_mask is not None
        and var_name in {"g_max", "g", "gbar", "conductance"}
    ):
        mask_bool = np.asarray(layout.point_mask)
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
    new_value = _runtime_param_value(
        layout=layout,
        var_name=var_name,
        state_buffers=runtime.state_buffers,
    )
    setattr(node, var_name, new_value)
    hook = getattr(node, "_on_param_updated", None)
    if callable(hook):
        hook(var_name, new_value)


def _sync_runtime_ion(runtime: CellRuntimeState, *, layout_id: int) -> None:
    """Rebuild the runtime ion's per-point params from state buffers.

    Uses vectorised ``.at[point_index].set(...)`` on Quantity mantissas
    instead of Python per-index loops.
    """
    mechanism = runtime.layout_mechanisms[int(layout_id)]
    if not isinstance(mechanism, Density) or mechanism.category != "ion":
        return
    instance_name = mechanism.instance_name
    ion = runtime.ions[instance_name]
    ion_cls = type(ion)
    supported_params = _supported_ion_runtime_params(ion_cls)

    full_values: dict[str, object] = {}
    for param_name in supported_params:
        baseline = _normalize_ion_runtime_param_value(
            ion_cls, param_name,
            getattr(ion, _ion_runtime_attr_name(ion_cls, param_name)),
        )
        full_values[param_name] = _ion_param_broadcast(baseline, shape=(runtime.n_point,))

    for candidate in runtime.layouts:
        candidate_mechanism = runtime.layout_mechanisms[candidate.id]
        if candidate.target != "density":
            continue
        if not isinstance(candidate_mechanism, Density) or candidate_mechanism.category != "ion":
            continue
        if candidate_mechanism.instance_name != instance_name:
            continue
        if candidate.point_index is None:
            raise ValueError(f"Ion layout {candidate.id!r} is missing point_index.")
        for param_name in candidate_mechanism.params.keys():
            buffer = runtime.state_buffers[(candidate.id, param_name)]
            full_values[param_name] = _ion_param_scatter(
                runtime_cls=ion_cls,
                param_name=param_name,
                target=full_values[param_name],
                buffer=buffer,
                point_index=candidate.point_index,
            )

    for param_name, value in full_values.items():
        setattr(ion, _ion_runtime_attr_name(ion_cls, param_name), value)
    if hasattr(ion, "_update_reversal") and callable(getattr(ion, "_update_reversal")):
        ion._update_reversal()


def _quantity_full(shape: tuple[int, ...], value: object) -> object:
    """Broadcast ``value`` to ``shape``, preserving unit if present."""
    if isinstance(value, u.Quantity):
        decimal = float(value.to_decimal(value.unit))
        return u.Quantity(np.full(shape, decimal, dtype=np.float64), value.unit)
    if hasattr(value, "unit") and hasattr(value, "to_decimal"):
        decimal = float(value.to_decimal(value.unit))
        return u.Quantity(np.full(shape, decimal, dtype=np.float64), value.unit)
    return np.full(shape, value, dtype=np.float64)


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
        var_name: _runtime_param_value(
            layout=layout, var_name=var_name, state_buffers=state_buffers
        )
        for var_name in mechanism.params.keys()
    }


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
    if not kind.startswith("channel:"):
        return False
    class_name = kind.split(":", 1)[1]
    try:
        cls = get_registry().get("channel", class_name)
    except KeyError as exc:
        raise ValueError(
            f"Unknown runtime channel class {class_name!r} for layout kind {kind!r}."
        ) from exc
    return _channel_current_owner_family(cls) is None
