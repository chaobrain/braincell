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
from dataclasses import dataclass, fields, is_dataclass

import brainunit as u
import numpy as np

from braincell import ion as runtime_ion
from braincell._base import Channel, IonChannel
from braincell.ion._template import DynamicNernstIon, FixedIon, InitNernstIon
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
from braincell.morph.morphology import Morphology
from ._point_tree import PointTree

__all__ = ["CellRuntimeState"]


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
    target: str
    layout: str
    point_index: np.ndarray | None
    point_mask: np.ndarray | None
    n_active: int
    source_cv_ids: tuple[int, ...]
    source_rule: str | None = None


@dataclass(frozen=True)
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

    point_tree: PointTree
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
    dhs_static_cache: object | None = None

    @classmethod
    def from_cell(cls, cell: "Cell") -> "CellRuntimeState":
        # Compile from immutable CV declarations into runtime layouts. Dense
        # layouts cover all points with masked storage, while sparse layouts keep
        # only the active point rows for point-only mechanisms such as clamps.
        point_tree = cell.point_tree()
        n_point = len(point_tree.points)
        n_cv = len(cell.cvs)

        grouped: dict[tuple[object, ...], dict[str, object]] = {}
        cv_to_layout_sets: list[set[int]] = [set() for _ in range(n_cv)]
        point_to_layout_sets: list[set[int]] = [set() for _ in range(n_point)]
        layout_id = 0

        def register(*, mechanism: object, target: str, cv_id: int, point_id: int) -> None:
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
            entry["cv_ids"].add(int(cv_id))
            entry["point_ids"].add(int(point_id))

        for cv in cell.cvs:
            midpoint_point_id = int(point_tree.cv_midpoint_point_id[cv.id])
            for mechanism in cv.density_mech:
                register(mechanism=mechanism, target="density", cv_id=cv.id, point_id=midpoint_point_id)
            for mechanism in cv.point_mech:
                register(mechanism=mechanism, target="point", cv_id=cv.id, point_id=midpoint_point_id)

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
        _attach_runtime_ion_geometry(
            ions=ions,
            cvs=cell.cvs,
            point_ids=point_tree.cv_midpoint_point_id,
            n_point=n_point,
        )
        return cls(
            point_tree=point_tree,
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
            dhs_static_cache=None,
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
        buffer = self.get_state(layout_id, var_name)
        _write_state_buffer(buffer, value)
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
        point_id = int(self.point_tree.cv_midpoint_point_id[int(cv_id)])
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
            if layout.kind not in {"CurrentClamp", "SineClamp", "FunctionClamp"}:
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


def install_cell_runtime(cell: "Cell", runtime: CellRuntimeState) -> None:
    cell._in_size = (runtime.n_cv,)
    cell._out_size = (runtime.n_cv,)

    root_nodes = dict(runtime.ions)
    for layout in runtime.layouts:
        node = runtime.runtime_nodes.get(layout.id)
        if node is None:
            continue
        if _is_root_level_runtime_node(layout.kind):
            root_nodes[f"layout_{layout.id}"] = node

    cell.ion_channels = cell._format_elements(IonChannel, **root_nodes)
    cell.C = cv_value_vector(cell, attr_name="cm")
    cell.V_th = fill_like(cell.varshape, cell._V_th_value)


def build_placeholder_ions() -> dict[str, object]:
    size = (1,)
    return {
        "na": runtime_ion.SodiumFixed(size=size),
        "k": runtime_ion.PotassiumFixed(size=size),
        "ca": runtime_ion.CalciumFixed(size=size),
    }


def clone_morpho(morpho: Morphology) -> Morphology:
    cloned = Morphology.from_root(morpho.root.branch, name=morpho.root.name)
    for index in range(1, len(morpho.branches)):
        branch = morpho.branch(index=index)
        parent = branch.parent
        if parent is None:
            continue
        cloned.attach(
            parent=parent.name,
            child_branch=branch.branch,
            child_name=branch.name,
            parent_x=float(branch.parent_x),  # type: ignore[arg-type]
            child_x=float(branch.child_x),  # type: ignore[arg-type]
        )
    return cloned


def cv_value_vector(cell: "Cell", *, attr_name: str) -> object:
    return quantity_vector([getattr(cv, attr_name) for cv in cell.cvs])


def fill_like(shape: tuple[int, ...], value: object) -> object:
    values = [value for _ in range(int(np.prod(shape, dtype=int)))]
    return quantity_vector(values, shape=shape)


def quantity_vector(values: list[object], *, shape: tuple[int, ...] | None = None) -> object:
    if len(values) == 0:
        return values
    first = values[0]
    target_shape = (len(values),) if shape is None else shape
    if hasattr(first, "unit"):
        decimals = [item.to_decimal(first.unit) for item in values]
        return u.Quantity(u.math.asarray(decimals).reshape(target_shape), first.unit)
    return u.math.asarray(values).reshape(target_shape)


def scatter_midpoint_values(*, values: object, point_ids: np.ndarray, n_point: int) -> object:
    if hasattr(values, "unit"):
        unit = values.unit
        mantissa = u.math.asarray(values.to_decimal(unit))
        base_shape = mantissa.shape[:-1] + (n_point,)
        out = u.math.zeros(base_shape, dtype=mantissa.dtype)
        out = out.at[..., point_ids].set(mantissa)
        return u.Quantity(out, unit)
    array = u.math.asarray(values)
    base_shape = array.shape[:-1] + (n_point,)
    out = u.math.zeros(base_shape, dtype=array.dtype)
    return out.at[..., point_ids].set(array)


def gather_midpoint_values(values: object, *, point_ids: np.ndarray) -> object:
    return values[..., point_ids]


def _attach_runtime_ion_geometry(
    *,
    ions: dict[str, object],
    cvs: tuple[object, ...],
    point_ids: np.ndarray,
    n_point: int,
) -> None:
    length = _scatter_cv_geometry(cvs=cvs, attr_name="length", point_ids=point_ids, n_point=n_point)
    area = _scatter_cv_geometry(cvs=cvs, attr_name="area", point_ids=point_ids, n_point=n_point)
    diam_mid = _scatter_cv_geometry(cvs=cvs, attr_name="diam_mid", point_ids=point_ids, n_point=n_point)
    radius_prox = _scatter_cv_geometry(cvs=cvs, attr_name="radius_prox", point_ids=point_ids, n_point=n_point)
    radius_dist = _scatter_cv_geometry(cvs=cvs, attr_name="radius_dist", point_ids=point_ids, n_point=n_point)

    for ion in ions.values():
        setattr(ion, "length", length)
        setattr(ion, "area", area)
        setattr(ion, "diam_mid", diam_mid)
        setattr(ion, "radius_prox", radius_prox)
        setattr(ion, "radius_dist", radius_dist)


def _scatter_cv_geometry(
    *,
    cvs: tuple[object, ...],
    attr_name: str,
    point_ids: np.ndarray,
    n_point: int,
) -> object:
    values = quantity_vector([getattr(cv, attr_name) for cv in cvs])
    return scatter_midpoint_values(values=values, point_ids=point_ids, n_point=n_point)


def matches_last_dim(value: object, size: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return False
    return int(shape[-1]) == int(size)


def is_python_zero(value: object) -> bool:
    return isinstance(value, (int, float)) and value == 0


def choose_layout(*, target: str) -> str:
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


def mechanism_signature(mechanism: object) -> tuple[object, ...]:
    """Return a hashable signature used to group declarations.

    Every supported mechanism type is a frozen dataclass with
    structural equality, so the signature reduces to
    ``(type_name, mechanism)``. Two declarations that compare equal
    share a signature and are grouped into one runtime layout.

    Notes
    -----
    For :class:`FunctionClamp`, the ``fn`` field is compared by
    identity under the dataclass-generated ``__eq__`` — two separately
    constructed ``lambda`` closures with identical bodies are treated
    as distinct signatures.
    """
    return (type(mechanism).__qualname__, mechanism)


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


def _allocate_state_buffer(mechanism: object, *, var_name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = _mechanism_var_value(mechanism, var_name)
    array = np.empty(shape, dtype=object)
    array.fill(value)
    return array


def _write_state_buffer(buffer: np.ndarray, value: object) -> None:
    if isinstance(value, np.ndarray):
        incoming = value.astype(object, copy=False)
        if incoming.shape != buffer.shape:
            raise ValueError(f"State assignment shape mismatch: expected {buffer.shape!r}, got {incoming.shape!r}.")
        buffer[...] = incoming
        return

    if isinstance(value, (list, tuple)):
        if buffer.size == 1:
            buffer.flat[0] = tuple(value) if isinstance(value, list) else value
            return
        incoming = np.empty(buffer.shape, dtype=object)
        flat_values = list(value)
        if len(flat_values) != buffer.size:
            raise ValueError(
                f"State assignment shape mismatch: expected {buffer.shape!r}, got sequence length {len(flat_values)!r}.")
        incoming.flat[:] = flat_values
        buffer[...] = incoming
        return

    buffer.fill(value)


def _extract_point_value(layout: MechanismLayout, *, point_id: int, buffer: np.ndarray) -> object:
    if layout.layout == "dense":
        return buffer[point_id]
    if layout.point_index is None:
        raise ValueError(f"Sparse layout {layout.id!r} is missing point_index.")
    matches = np.flatnonzero(layout.point_index == int(point_id))
    if len(matches) == 0:
        raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
    return buffer[int(matches[0])]


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
    return buffer.reshape((-1,))[int(local_index)]


def _quantity_sequence_to_decimal_vector(values: object, *, unit: object) -> object:
    if hasattr(values, "to_decimal"):
        return u.math.asarray(values.to_decimal(unit))
    decimals = [item.to_decimal(unit) for item in values]
    return u.math.asarray(decimals)


def _eval_current_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    durations = _scalar_state_value(runtime, layout_id=layout_id, var_name="durations", local_index=local_index)
    amplitudes = _scalar_state_value(runtime, layout_id=layout_id, var_name="amplitudes", local_index=local_index)
    duration_decimal = _quantity_sequence_to_decimal_vector(durations, unit=u.ms)
    amplitude_decimal = _quantity_sequence_to_decimal_vector(amplitudes, unit=u.nA)
    local_t_decimal = local_t.to_decimal(u.ms)
    end_decimal = u.math.cumsum(duration_decimal)
    start_decimal = end_decimal - duration_decimal
    is_active = u.math.logical_and(local_t_decimal >= 0.0,
                                   u.math.logical_and(local_t_decimal >= start_decimal, local_t_decimal < end_decimal))
    current_decimal = u.math.sum(u.math.where(is_active, amplitude_decimal, 0.0))
    return u.Quantity(current_decimal, u.nA)


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
    size = (n_point,)
    return {
        "na": runtime_ion.SodiumFixed(size=size),
        "k": runtime_ion.PotassiumFixed(size=size),
        "ca": runtime_ion.CalciumFixed(size=size),
    }


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
    for name, parameter in signature.parameters.items():
        if name in {"self", "size", "name"}:
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
    state_buffers: dict[tuple[int, str], np.ndarray],
    n_point: int,
) -> object:
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
    full_param_values: dict[str, np.ndarray] = {}
    for param_name in supported_params:
        baseline_value = _normalize_ion_runtime_param_value(
            runtime_cls,
            param_name,
            getattr(baseline_ion, _ion_runtime_attr_name(runtime_cls, param_name)),
        )
        full_param_values[param_name] = _object_array_from_value(baseline_value, shape=(n_point,))

    for layout, declaration in zip(layouts, declarations):
        point_index = layout.point_index
        if point_index is None:
            raise ValueError(f"Ion layout {layout.id!r} is missing point_index.")
        for param_name in declaration.params.keys():
            values = _object_array_from_buffer(state_buffers[(layout.id, param_name)])
            if param_name == "Ci_initializer" and issubclass(runtime_cls, DynamicNernstIon):
                for index in point_index.tolist():
                    full_param_values[param_name][index] = _normalize_ion_runtime_param_value(
                        runtime_cls,
                        param_name,
                        values[index],
                    )
            else:
                full_param_values[param_name][point_index] = values[point_index]

    runtime_kwargs = {
        param_name: _as_runtime_array(values)
        for param_name, values in full_param_values.items()
    }
    return runtime_cls(size=(n_point,), name=instance_name, **runtime_kwargs)


class _BoundIonChannelRuntime(Channel):
    __module__ = "braincell.compute"

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
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> object:
    buffer = state_buffers[(layout.id, var_name)]
    values = _object_array_from_buffer(buffer)
    if layout.point_mask is not None and var_name in {"g_max", "g", "gbar", "conductance"}:
        values = _mask_quantity_like(values, layout.point_mask)
    return _as_runtime_array(values)


def _sync_runtime_node_param(runtime: CellRuntimeState, *, layout_id: int, var_name: str) -> None:
    node = runtime.runtime_nodes.get(int(layout_id))
    if node is None:
        return
    layout = runtime.layouts[int(layout_id)]
    kind = layout.kind
    if kind.startswith("ion:"):
        _sync_runtime_ion(runtime, layout_id=int(layout_id))
        return
    if kind in {"channel:INa_HH1952", "channel:IK_HH1952"} and var_name == "T":
        setattr(
            node,
            "phi",
            _runtime_temperature_phi(
                _runtime_param_value(layout=layout, var_name=var_name, state_buffers=runtime.state_buffers)
            ),
        )
        return
    setattr(
        node,
        var_name,
        _runtime_param_value(layout=layout, var_name=var_name, state_buffers=runtime.state_buffers),
    )


def _sync_runtime_ion(runtime: CellRuntimeState, *, layout_id: int) -> None:
    layout = runtime.layouts[int(layout_id)]
    mechanism = runtime.layout_mechanisms[int(layout_id)]
    if not isinstance(mechanism, Density) or mechanism.category != "ion":
        return
    instance_name = mechanism.instance_name
    ion = runtime.ions[instance_name]
    supported_params = _supported_ion_runtime_params(type(ion))
    full_values = {
        param_name: _object_array_from_value(
            _normalize_ion_runtime_param_value(
                type(ion),
                param_name,
                getattr(ion, _ion_runtime_attr_name(type(ion), param_name)),
            ),
            shape=(runtime.n_point,),
        )
        for param_name in supported_params
    }

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
            values = _object_array_from_buffer(runtime.state_buffers[(candidate.id, param_name)])
            if param_name == "Ci_initializer" and isinstance(ion, DynamicNernstIon):
                for index in candidate.point_index.tolist():
                    full_values[param_name][index] = _normalize_ion_runtime_param_value(
                        type(ion),
                        param_name,
                        values[index],
                    )
            else:
                full_values[param_name][candidate.point_index] = values[candidate.point_index]

    for param_name, values in full_values.items():
        setattr(ion, _ion_runtime_attr_name(type(ion), param_name), _as_runtime_array(values))
    if hasattr(ion, "_update_reversal") and callable(getattr(ion, "_update_reversal")):
        ion._update_reversal()


def _mask_quantity_like(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = values.copy()
    for index, active in enumerate(mask.tolist()):
        if active:
            continue
        value = masked[index]
        try:
            masked[index] = 0 * value.unit
        except Exception:
            masked[index] = 0
    return masked


def _object_array_from_buffer(buffer: np.ndarray) -> np.ndarray:
    values = np.empty(buffer.shape, dtype=object)
    for index, item in enumerate(buffer.flat):
        values.flat[index] = item
    return values


def _object_array_from_value(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    values = np.empty(shape, dtype=object)
    raw_shape = getattr(value, "shape", ())
    if raw_shape not in (None, (), shape):
        raise ValueError(f"Cannot coerce value with shape {raw_shape!r} into object array shape {shape!r}.")
    if raw_shape == shape:
        for index in np.ndindex(shape):
            values[index] = value[index]
        return values
    values.fill(value)
    return values


def _as_runtime_array(values: np.ndarray) -> object:
    if values.size == 0:
        return values
    first = values.flat[0]
    if hasattr(first, "to_decimal") and callable(getattr(first, "to_decimal")):
        decimals = np.asarray([item.to_decimal(first.unit) for item in values.flat], dtype=float).reshape(values.shape)
        return u.Quantity(decimals, first.unit)
    return np.asarray(values.tolist())


def _runtime_constructor_params(
    *,
    layout: MechanismLayout,
    mechanism: Density,
    state_buffers: dict[tuple[int, str], np.ndarray],
) -> dict[str, object]:
    """Build the kwargs passed to a concrete channel class's ``__init__``.

    Reads each declared parameter from its state buffer (so
    per-point values already live in the buffer, not in the frozen
    declaration). ``coverage_area_fraction`` is a :class:`Density`
    field, not a param, so it does not leak into kwargs.

    Notes
    -----
    HH1952 channels expose a ``T`` parameter in their declaration but
    accept ``phi`` in their constructor (the Q10 scaling factor
    derived from temperature). This translation is applied here.
    """
    if mechanism.category != "channel":
        return {}

    raw: dict[str, object] = {}
    for var_name in mechanism.params.keys():
        raw[var_name] = _runtime_param_value(
            layout=layout, var_name=var_name, state_buffers=state_buffers
        )

    if mechanism.class_name in {"INa_HH1952", "IK_HH1952"}:
        if "T" in raw and "phi" not in raw:
            raw["phi"] = _runtime_temperature_phi(raw.pop("T"))
    return raw


def _runtime_temperature_phi(temperature: object) -> object:
    if hasattr(temperature, "to_decimal"):
        temperature_c = u.kelvin2celsius(temperature)
        return 3 ** ((temperature_c - 36) / 10)
    return temperature


def _is_root_level_runtime_node(kind: str) -> bool:
    """Return True when a channel layout installs at the root level.

    Root-level channels are those whose concrete class has
    ``root_type == HHTypedNeuron`` (i.e. not bound to an ion
    container). The registry is consulted to inspect the class.
    """
    if not kind.startswith("channel:"):
        return False
    class_name = kind.split(":", 1)[1]
    try:
        cls = get_registry().get("channel", class_name)
    except KeyError:
        return False
    return _channel_current_owner_family(cls) is None
