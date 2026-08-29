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

"""Cell-local trainable root registry and binding materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import weakref

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute.parameters import RuntimeParameterState, density_parameter_schema
from braincell.mech import Density
from braincell.trainable._parameters import ParameterSet
from braincell.trainable._sources import DirectSource, ParameterizedSource, ParameterSource, ScaleSource

__all__ = ["ParameterBinding", "TrainableManager"]


@dataclass(frozen=True)
class _TargetRow:
    category: str
    owner: str
    mechanism_type: str
    population_index: int
    cv_id: int
    point_id: int


@dataclass(frozen=True)
class ParameterBinding:
    """Read-only description of one source-to-runtime-field mapping."""

    name: str
    target_owner: str
    target_field: str
    row_keys: tuple[tuple[int, int], ...]
    group_by: str
    root_names: tuple[str, ...]
    unit: object | None
    baseline: object | None = None
    _rows: tuple[_TargetRow, ...] = field(default=(), repr=False)
    _evaluate: object = field(default=None, repr=False, compare=False)


class TrainableManager(brainstate.nn.Module):
    """Own trainable roots and materialize their Cell runtime bindings."""

    def __init__(self, cell) -> None:
        super().__init__()
        self.roots: dict[str, brainstate.nn.Param] = {}
        self._cell_ref = weakref.ref(cell)
        self._binding_list: list[ParameterBinding] = []
        self._owned_targets: set[tuple[str, str, int, int, str]] = set()
        self._root_names_by_id: dict[int, str] = {}
        self._target_axes: dict[tuple[int, str], str] = {}

    def parameters(self) -> ParameterSet:
        """Return a live optimizer-facing view over the original roots."""
        return ParameterSet(self.roots)

    def bindings(self) -> tuple[ParameterBinding, ...]:
        """Return immutable binding metadata in registration order."""
        return tuple(self._binding_list)

    def owns(self, *, category: str, owner: str, population_index: int, cv_id: int, field: str) -> bool:
        """Return whether a logical density row/field is binding-owned."""
        return (category, owner, int(population_index), int(cv_id), field) in self._owned_targets

    def register(self, view, fields: dict[str, ParameterSource]) -> None:
        """Validate and atomically register trainable fields for one density View."""
        cell = self._cell()
        if cell._initialized:
            raise RuntimeError("trainable() must be called before Cell.init_state().")
        if cell.network_owner is not None:
            raise NotImplementedError(
                "Trainable density parameters are Cell-local in P0 and cannot target a Network Cell."
            )
        if not view.rows:
            raise ValueError("trainable() requires a non-empty View.")
        owners = tuple(dict.fromkeys((row.mechanism_type, row.name) for row in view.rows))
        if len(owners) != 1:
            raise TypeError("trainable() requires a View selecting exactly one mechanism owner.")
        if not fields:
            return

        roots_before = dict(self.roots)
        names_before = dict(self._root_names_by_id)
        bindings_before = len(self._binding_list)
        owned_before = set(self._owned_targets)
        try:
            for target_field, source in fields.items():
                self._register_one(view, target_field, source)
        except Exception:
            self.roots.clear()
            self.roots.update(roots_before)
            self._root_names_by_id.clear()
            self._root_names_by_id.update(names_before)
            del self._binding_list[bindings_before:]
            self._owned_targets.clear()
            self._owned_targets.update(owned_before)
            raise

    def materialize(self) -> None:
        """Evaluate every binding and atomically update runtime physical states."""
        cell = self._cell()
        runtime = cell._runtime
        if runtime is None:
            raise RuntimeError("Trainable materialization requires an initialized runtime allocation.")

        evaluated = [(binding, binding._evaluate()) for binding in self._binding_list]
        pending: dict[tuple[int, str], tuple[RuntimeParameterState, object, str]] = {}
        for binding, values in evaluated:
            required_axis = _binding_axis(binding)
            if tuple(getattr(values, "shape", ())) != (len(binding._rows),):
                raise ValueError(
                    f"Binding {binding.name!r} returned shape {getattr(values, 'shape', ())!r}; "
                    f"expected ({len(binding._rows)},)."
                )
            for index, row in enumerate(binding._rows):
                layout = _runtime_layout(runtime, row)
                key = (int(layout.id), binding.target_field)
                if key not in pending:
                    state = runtime.state_buffers.get(key)
                    if not isinstance(state, RuntimeParameterState):
                        raise RuntimeError(
                            f"Runtime target {binding.target_owner!r}.{binding.target_field} is not schema-backed."
                        )
                    pending[key] = (state, state.dense_value(), required_axis)
                state, full, pending_axis = pending[key]
                pending[key] = (
                    state,
                    _set_row(full, row.population_index, row.cv_id, values[index]),
                    _join_axes(pending_axis, required_axis),
                )

        commits = []
        for key, (state, full, required_axis) in pending.items():
            axis = self._target_axes.get(key)
            if axis is None:
                axis = _join_axes(_compact_axis(full, state.point_mask), required_axis)
            else:
                axis = _join_axes(axis, required_axis)
            commits.append((key, state, axis, _project_axis(full, axis, state.point_mask)))
        for key, state, axis, value in commits:
            state.value = value
            state.axis = axis
            self._target_axes[key] = axis
        if commits:
            from braincell._compute.bindings import _sync_runtime_node_param

            for (layout_id, field), _state, _axis, _value in commits:
                _sync_runtime_node_param(runtime, layout_id=layout_id, var_name=field)

    def runtime_reset(self) -> None:
        """Forget runtime layout identities while retaining roots and bindings."""
        self._target_axes.clear()

    def _register_one(self, view, target_field: str, source: ParameterSource) -> None:
        if not isinstance(target_field, str) or not target_field:
            raise ValueError("Trainable target field must be a non-empty string.")
        if not isinstance(source, (DirectSource, ScaleSource, ParameterizedSource)):
            raise TypeError(f"Field {target_field!r} expects a braincell.trainable parameter source.")
        mechanism = view.rows[0].mechanism
        schema = density_parameter_schema(mechanism)
        if not schema:
            raise NotImplementedError(
                f"Channel {mechanism.class_name!r} has no trainable parameter schema in this release."
            )
        if target_field not in schema:
            raise KeyError(f"Channel {mechanism.class_name!r} has no trainable parameter {target_field!r}.")

        rows = tuple(
            _TargetRow(
                row.category,
                row.name,
                row.mechanism_type,
                int(row.population_index),
                int(row.cv_id),
                int(row.point_id),
            )
            for row in view.rows
        )
        target_keys = {(row.category, row.owner, row.population_index, row.cv_id, target_field) for row in rows}
        overlap = target_keys.intersection(self._owned_targets)
        if overlap:
            raise ValueError(f"Trainable target rows are already bound: {tuple(sorted(overlap))!r}.")

        current = tuple(view._row_value(source_row, target_field) for source_row in view.rows)
        base_name = _base_name(rows, target_field, source)
        if isinstance(source, DirectSource):
            evaluate, root_names = self._prepare_direct(source, rows, current, base_name)
            group_by = source.group_by
            baseline = None
        elif isinstance(source, ScaleSource):
            evaluate, root_names, baseline = self._prepare_scale(source, rows, current, base_name)
            group_by = source.group_by
        else:
            evaluate, root_names = self._prepare_parameterized(source, rows, base_name)
            group_by = "parameterized"
            baseline = None

        sample = evaluate()
        if tuple(getattr(sample, "shape", ())) != (len(rows),):
            raise ValueError(
                f"Trainable source for {target_field!r} must produce one scalar per selected row; "
                f"got shape {getattr(sample, 'shape', ())!r}."
            )
        for index in range(len(rows)):
            schema[target_field].validate(sample[index], target_field)
        unit = schema[target_field].default.unit if isinstance(schema[target_field].default, u.Quantity) else None
        binding = ParameterBinding(
            name=base_name,
            target_owner=rows[0].owner,
            target_field=target_field,
            row_keys=tuple((row.population_index, row.cv_id) for row in rows),
            group_by=group_by,
            root_names=root_names,
            unit=unit,
            baseline=baseline,
            _rows=rows,
            _evaluate=evaluate,
        )
        self._binding_list.append(binding)
        self._owned_targets.update(target_keys)

    def _prepare_direct(self, source, rows, current, base_name):
        indices, n_groups = _group_indices(rows, source.group_by)
        initial = (
            _grouped_initial(current, indices, n_groups)
            if source.initial is None
            else _root_initial(source.initial, n_groups)
        )
        root = brainstate.nn.Param(initial, t=source.transform)
        root_name = self._register_root(root, source.name or base_name)
        return lambda: _gather(root.value(), indices), (root_name,)

    def _prepare_scale(self, source, rows, current, base_name):
        indices, n_groups = _group_indices(rows, source.group_by)
        baseline = _stack(current)
        if source.parameter is None:
            initial = 1.0 if n_groups == 1 else jnp.ones((n_groups,))
            transform = brainstate.nn.IdentityT() if source.transform is None else source.transform
            root = brainstate.nn.Param(initial, t=transform)
        else:
            root = source.parameter
            _require_root_shape(root.value(), n_groups)
        root_name = self._register_root(root, source.name or f"{base_name}.factor")
        return lambda: baseline * _gather(root.value(), indices), (root_name,), baseline

    def _prepare_parameterized(self, source, rows, base_name):
        signature = inspect.signature(source.function)
        parameters = tuple(signature.parameters.values())
        if not parameters:
            raise TypeError("parameterized() function must accept CVContext as its first argument.")
        if any(item.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for item in parameters):
            raise TypeError("parameterized() does not support *args or **kwargs signatures.")
        arguments = dict(source.arguments)
        allowed = {item.name for item in parameters[1:]}
        unknown = tuple(sorted(set(arguments).difference(allowed)))
        missing = tuple(
            item.name
            for item in parameters[1:]
            if item.default is inspect.Parameter.empty and item.name not in arguments
        )
        if unknown or missing:
            raise TypeError(
                f"parameterized() arguments differ from the function signature (missing={missing!r}, extra={unknown!r})."
            )

        prepared = {}
        root_names = []
        for name, value in arguments.items():
            if isinstance(value, brainstate.nn.Param):
                root_name = self._register_root(value, f"{base_name}.{name}") if value.fit else None
                if root_name is not None:
                    root_names.append(root_name)
                prepared[name] = ("param", value, None)
            elif isinstance(value, DirectSource):
                if value.initial is None:
                    raise ValueError(f"Nested parameter {name!r} requires an explicit initial value.")
                indices, n_groups = _group_indices(rows, value.group_by)
                root = brainstate.nn.Param(_root_initial(value.initial, n_groups), t=value.transform)
                root_name = self._register_root(root, value.name or f"{base_name}.{name}")
                root_names.append(root_name)
                prepared[name] = ("grouped", root, indices)
            elif isinstance(value, (ScaleSource, ParameterizedSource)):
                raise TypeError("parameterized() arguments only nest parameter(), not scale() or parameterized().")
            else:
                prepared[name] = ("fixed", value, None)

        cell_ref = self._cell_ref

        def evaluate():
            cell = cell_ref()
            values = []
            for row_index, row in enumerate(rows):
                kwargs = {}
                for name, (kind, value, indices) in prepared.items():
                    if kind == "fixed":
                        kwargs[name] = value
                    elif kind == "param":
                        kwargs[name] = value.value()
                    else:
                        kwargs[name] = _gather_one(value.value(), indices[row_index])
                values.append(source.function(cell.cv_contexts[row.cv_id], **kwargs))
            return _stack(tuple(values))

        return evaluate, tuple(root_names)

    def _register_root(self, root: brainstate.nn.Param, name: str) -> str:
        existing_name = self._root_names_by_id.get(id(root))
        if existing_name is not None:
            if name != existing_name:
                raise ValueError(
                    f"Shared nn.Param is already registered as {existing_name!r}; conflicting name {name!r}."
                )
            return existing_name
        existing_root = self.roots.get(name)
        if existing_root is not None and existing_root is not root:
            raise ValueError(f"Trainable root name {name!r} is already used by another nn.Param.")
        self.roots[name] = root
        self._root_names_by_id[id(root)] = name
        return name

    def _cell(self):
        cell = self._cell_ref()
        if cell is None:
            raise RuntimeError("Owning Cell no longer exists.")
        return cell


def _base_name(rows, field: str, source: ParameterSource) -> str:
    fingerprint = hashlib.sha256(repr(tuple((row.population_index, row.cv_id) for row in rows)).encode()).hexdigest()[
        :8
    ]
    role = "direct" if isinstance(source, DirectSource) else "scale" if isinstance(source, ScaleSource) else "function"
    return f"channel.{rows[0].owner}.{field}.{role}.{fingerprint}"


def _group_indices(rows, group_by: str) -> tuple[np.ndarray, int]:
    keys = []
    for row in rows:
        if group_by == "row":
            key = (row.population_index, row.cv_id)
        elif group_by == "population":
            key = row.population_index
        elif group_by == "cv":
            key = row.cv_id
        else:
            key = 0
        keys.append(key)
    positions = {}
    indices = []
    for key in keys:
        if key not in positions:
            positions[key] = len(positions)
        indices.append(positions[key])
    return np.asarray(indices, dtype=np.int32), len(positions)


def _grouped_initial(values, indices: np.ndarray, n_groups: int):
    grouped = []
    for group in range(n_groups):
        items = [values[index] for index in np.flatnonzero(indices == group).tolist()]
        first = items[0]
        if any(not _equal_value(first, item) for item in items[1:]):
            raise ValueError("Direct grouped initial values differ; use scale() to preserve their relative values.")
        grouped.append(first)
    return grouped[0] if n_groups == 1 else _stack(tuple(grouped))


def _root_initial(value: object, n_groups: int):
    shape = tuple(getattr(value, "shape", ()))
    if n_groups == 1:
        if shape not in ((), (1,)):
            raise ValueError(f"Root initial must be scalar for one group, got shape {shape!r}.")
        return value if shape == () else value[0]
    if shape == ():
        return u.math.broadcast_to(value, (n_groups,))
    if shape != (n_groups,):
        raise ValueError(f"Root initial must broadcast to ({n_groups},), got shape {shape!r}.")
    return value


def _require_root_shape(value: object, n_groups: int) -> None:
    expected = () if n_groups == 1 else (n_groups,)
    if tuple(getattr(value, "shape", ())) != expected:
        raise ValueError(f"Existing nn.Param shape must be {expected!r}, got {getattr(value, 'shape', ())!r}.")


def _gather(value: object, indices: np.ndarray):
    if tuple(getattr(value, "shape", ())) == ():
        return u.math.broadcast_to(value, (len(indices),))
    return value[jnp.asarray(indices)]


def _gather_one(value: object, index: int):
    return value if tuple(getattr(value, "shape", ())) == () else value[int(index)]


def _stack(values: tuple[object, ...]):
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.stack([value.to_decimal(unit) for value in values]), unit)
    return u.math.stack(values)


def _equal_value(left: object, right: object) -> bool:
    if isinstance(left, u.Quantity):
        if not isinstance(right, u.Quantity):
            return False
        try:
            return bool(np.array_equal(np.asarray(left.to_decimal(left.unit)), np.asarray(right.to_decimal(left.unit))))
        except Exception:
            return False
    if isinstance(right, u.Quantity):
        return False
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _runtime_layout(runtime, row: _TargetRow):
    matches = []
    for layout in runtime.layouts:
        mechanism = runtime.layout_mechanisms[layout.id]
        if (
            isinstance(mechanism, Density)
            and mechanism.category == row.category
            and mechanism.instance_name == row.owner
            and row.cv_id in layout.source_cv_ids
        ):
            matches.append(layout)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one runtime layout for {row.category} {row.owner!r} CV {row.cv_id}, got {len(matches)}."
        )
    return matches[0]


def _set_row(full: object, population_index: int, point_id: int, value: object):
    if isinstance(full, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Materialized value requires a Quantity compatible with {full.unit}.")
        mantissa = (
            jnp.asarray(full.to_decimal(full.unit)).at[population_index, point_id].set(value.to_decimal(full.unit))
        )
        return u.Quantity(mantissa, full.unit)
    if isinstance(value, u.Quantity):
        raise TypeError("Materialized value must be dimensionless.")
    return jnp.asarray(full).at[population_index, point_id].set(value)


def _compact_axis(value: object, point_mask: object | None) -> str:
    raw = value.mantissa if isinstance(value, u.Quantity) else value
    if isinstance(raw, jax.core.Tracer):
        return "row"
    array = np.asarray(raw)
    active = np.asarray(point_mask, dtype=bool) if point_mask is not None else np.ones(array.shape[-1], dtype=bool)
    active_array = array[..., active]
    if np.all(active_array == active_array.reshape(-1)[0]):
        return "uniform"
    if np.all(active_array == active_array[..., :1]):
        return "population"
    flat_population = active_array.reshape((-1, active_array.shape[-1]))
    if np.all(flat_population == flat_population[:1]):
        return "cv"
    return "row"


def _binding_axis(binding: ParameterBinding) -> str:
    if binding.group_by in ("row", "parameterized"):
        return "row"
    if binding.group_by == "population":
        return "population"
    if binding.group_by == "cv":
        return "cv"
    return "uniform"


def _join_axes(left: str, right: str) -> str:
    if left == right:
        return left
    if left == "uniform":
        return right
    if right == "uniform":
        return left
    if left == "row" or right == "row":
        return "row"
    return "row"


def _project_axis(value: object, axis: str, point_mask: object | None):
    active_indices = np.flatnonzero(np.asarray(point_mask, dtype=bool)) if point_mask is not None else np.asarray([0])
    first_point = int(active_indices[0])
    if axis == "row":
        return value
    if axis == "uniform":
        return value[..., first_point].reshape(-1)[0]
    if axis == "population":
        return value[..., first_point : first_point + 1]
    if axis == "cv":
        flat = value.reshape((-1, value.shape[-1]))
        return flat[0]
    raise ValueError(f"Unknown runtime parameter axis {axis!r}.")
