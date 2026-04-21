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


from dataclasses import dataclass, fields, is_dataclass

import brainunit as u
import numpy as np

from braincell import ion as runtime_ion
from braincell._base import IonChannel
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
from braincell.morph._morphology import Morphology
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

        ions, runtime_nodes = _build_runtime_nodes(
            n_point=n_point,
            layouts=tuple(layouts),
            layout_mechanisms=layout_mechanisms,
            state_buffers=state_buffers,
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
        key = str(name)
        if key not in self.ions:
            raise KeyError(f"No ion container is registered for {name!r}.")
        return self.ions[key]

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
) -> tuple[dict[str, object], dict[int, object]]:
    ions = _build_default_ions(n_point)
    runtime_nodes: dict[int, object] = {}
    for layout in layouts:
        mechanism = layout_mechanisms[layout.id]
        node = _instantiate_runtime_node(
            layout=layout,
            mechanism=mechanism,
            state_buffers=state_buffers,
            ions=ions,
        )
        if node is not None:
            runtime_nodes[layout.id] = node
    return ions, runtime_nodes


def _build_default_ions(n_point: int) -> dict[str, object]:
    size = (n_point,)
    return {
        "na": runtime_ion.SodiumFixed(size=size),
        "k": runtime_ion.PotassiumFixed(size=size),
        "ca": runtime_ion.CalciumFixed(size=size),
    }


def _instantiate_runtime_node(
    *,
    layout: MechanismLayout,
    mechanism: object,
    state_buffers: dict[tuple[int, str], np.ndarray],
    ions: dict[str, object],
) -> object | None:
    if layout.target != "density" or layout.layout != "dense":
        return None
    if not isinstance(mechanism, Density):
        return None
    if mechanism.category != "channel":
        return None

    runtime_cls = get_registry().get("channel", mechanism.class_name)
    params = _runtime_constructor_params(
        layout=layout, mechanism=mechanism, state_buffers=state_buffers
    )
    if len(params) > 0 and hasattr(next(iter(params.values())), "shape"):
        size = next(iter(params.values())).shape
    else:
        size = (layout.n_active,)
    node = runtime_cls(size=size, **params)

    ion_key = _channel_ion_key(runtime_cls)
    if ion_key is not None:
        channel_key = mechanism.instance_name
        ions[ion_key].add(**{channel_key: node})
    return node


def _channel_ion_key(cls: type) -> str | None:
    """Infer the ion-container key for a concrete channel class.

    Returns ``"na"``, ``"k"``, or ``"ca"`` when the class's
    ``root_type`` is a subclass of the corresponding ion base, or
    ``None`` when the channel installs at root level (e.g. leak
    channels whose ``root_type`` is :class:`HHTypedNeuron`). Channels
    with compound ``root_type`` (e.g. ``JointTypes[Potassium,
    Calcium]`` for K-Ca channels) also return ``None`` — they need
    per-channel runtime wiring that is not yet implemented.
    """
    root_type = getattr(cls, "root_type", None)
    if root_type is None or not isinstance(root_type, type):
        return None
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


def _as_runtime_array(values: np.ndarray) -> object:
    if values.size == 0:
        return values
    first = values.flat[0]
    if hasattr(first, "unit"):
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
    return _channel_ion_key(cls) is None
