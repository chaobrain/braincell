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

"""Runtime ion instantiation, parameter normalization, and synchronization.

This module owns everything that turns a cell's ``Density`` ion declarations
into live runtime ion objects, and everything that keeps those objects in step
with the state buffers afterwards:

- :func:`_build_runtime_ions` — the entry point. Collects the ion declarations
  spread across mechanism layouts, instantiates one runtime ion per instance
  name, fills in placeholder ions for any of ``na``/``k``/``ca`` that were never
  declared, and returns a 5-tuple of the ion map, the alias map that lets
  channels look an ion up by instance name, family key, or class name, the
  family and class candidate maps, and the per-layout runtime-ion nodes.
- :func:`_collect_runtime_ion_instances`, :func:`_build_ion_alias_map`,
  :func:`_runtime_ion_species_key`, :func:`_runtime_ion_family` — the grouping
  and naming rules, including the conflict checks that reject an instance name
  reused across two ion classes or two species.
- :func:`_supported_ion_runtime_params`, :func:`_ion_runtime_attr_name`,
  :func:`_normalize_ion_runtime_param_value` — introspection of a runtime ion
  class's constructor and the small amount of renaming/unwrapping needed to
  read a param back off an instance.
- :func:`_ion_param_broadcast` and :func:`_ion_param_scatter` — the rectangular
  buffer algebra. A baseline param is broadcast onto the full point shape once,
  then each sparse declaration layout scatters its own buffer into it, so the
  common rectangular path needs no Python loop over per-point
  :class:`brainunit.Quantity` boxes.
- :func:`_sync_runtime_ion` — the post-compilation counterpart, rebuilding one
  runtime ion's params from the current state buffers when a buffer is written.

Ion construction happens before channels are bound, so this module depends only
on :mod:`braincell._compute.layouts` (for the ``MechanismLayout`` record and the
constant-quantity helper), on ``braincell.mech``, and on ``braincell.ion`` —
including its private :mod:`braincell.ion._base` module, for the runtime ion
base classes. It imports nothing from :mod:`braincell._compute.bindings` or
:mod:`braincell._compute.state`, which both sit above it in the layer stack.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import brainunit as u
import numpy as np

from braincell import ion as runtime_ion
from braincell.ion import build_placeholder_ions
from braincell.ion._base import DynamicNernstIon, FixedIon, InitNernstIon, KineticIon
from braincell.mech import Density, get_registry
from .layouts import MechanismLayout, _constant_quantity_value

if TYPE_CHECKING:
    from .state import CellRuntimeState


def _build_runtime_ions(
    *,
    n_point: int,
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
            pop_size=pop_size,
        )
        ions[instance_name] = runtime_ion
        ion_class_candidates.setdefault(record["runtime_cls"].__name__, []).append(instance_name)
        for layout in record["layouts"]:
            ion_runtime_nodes[layout.id] = runtime_ion

    for family_key in ("na", "k", "ca"):
        if family_key in ion_family_candidates:
            continue
        default_ion = _build_default_ions(pop_size + (n_point,))[family_key]
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
    if isinstance(n_point, tuple):
        return build_placeholder_ions(size=n_point)
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
            raise ValueError(f"Ion alias {alias!r} conflicts between species {existing!r} and {canonical!r}.")
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
    if issubclass(cls, runtime_ion.NonSpecific):
        return "no"
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
        if isinstance(value, u.Quantity):
            return value
        inner = getattr(value, "value", None)
        if isinstance(inner, u.Quantity):
            return inner
    constant_quantity = _constant_quantity_value(value)
    if constant_quantity is not None:
        return constant_quantity
    return value


def _instantiate_runtime_ion_instance(
    *,
    instance_name: str,
    runtime_cls: type,
    layouts: tuple[MechanismLayout, ...],
    declarations: tuple[Density, ...],
    state_buffers: dict,
    n_point: int,
    pop_size: tuple[int, ...] = (),
) -> object:
    """Build one runtime ion instance from its sparse declaration layouts.

    Start from a baseline ion and replace per-point params where each
    declaration layout requests them. Each layout's buffer is scattered
    into the accumulated array by :func:`_ion_param_scatter`, which uses
    ``np.put_along_axis`` on the Quantity mantissa — no Python loops on
    per-point Quantity boxes.
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

    full_size = pop_size + (n_point,)
    baseline_ion = runtime_cls(size=full_size)
    full_param_values: dict[str, object] = {}
    for param_name in supported_params:
        baseline_value = _normalize_ion_runtime_param_value(
            runtime_cls,
            param_name,
            getattr(baseline_ion, _ion_runtime_attr_name(runtime_cls, param_name)),
        )
        full_param_values[param_name] = _ion_param_broadcast(baseline_value, shape=full_size)

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

    runtime_ion_instance = runtime_cls(size=full_size, name=instance_name, **full_param_values)
    _restore_shaped_species_initializers(runtime_ion_instance, full_param_values)
    return runtime_ion_instance


def _restore_shaped_species_initializers(runtime_ion_instance: object, full_param_values: dict[str, object]) -> None:
    if "Ci_initializer" not in full_param_values:
        return
    species_initializers = getattr(runtime_ion_instance, "species_initializers", None)
    if not isinstance(species_initializers, dict) or "Ci" not in species_initializers:
        return
    shaped_ci = full_param_values["Ci_initializer"]
    if not isinstance(shaped_ci, (u.Quantity, np.ndarray)):
        cainull = getattr(runtime_ion_instance, "cainull", None)
        if isinstance(cainull, (u.Quantity, np.ndarray)):
            shaped_ci = cainull
    runtime_ion_instance.Ci_initializer = shaped_ci
    species_initializers["Ci"] = shaped_ci


def _ion_param_broadcast(value: object, *, shape: tuple[int, ...]) -> object:
    """Broadcast an ion baseline value onto ``shape``.

    Handles three cases: already-shaped Quantity pass-through, scalar
    Quantity broadcast, and plain numeric / object fallbacks. Returns
    a buffer that :func:`_ion_param_scatter` copies and updates via
    ``np.put_along_axis``.
    """
    if isinstance(value, u.Quantity):
        raw = value.mantissa if hasattr(value, "mantissa") else value.to_decimal(value.unit)
        mantissa = np.asarray(raw, dtype=np.float64)
        if mantissa.shape == shape:
            return u.Quantity(mantissa.copy(), value.unit)
        if mantissa.ndim == 0 or mantissa.shape == ():
            return u.Quantity(np.full(shape, float(mantissa), dtype=np.float64), value.unit)
        raise ValueError(f"Cannot broadcast ion baseline value with shape {mantissa.shape!r} onto shape {shape!r}.")
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
    tuple buffer. Rectangular Quantity and ndarray buffers scatter via
    ``np.put_along_axis`` onto a copy of ``target``, with unit coercion
    for Quantity buffers.
    """
    if isinstance(target, u.Quantity) and isinstance(buffer, u.Quantity):
        target_unit = target.unit
        src_mantissa = np.asarray(buffer.mantissa, dtype=np.float64)
        target_mantissa = np.asarray(target.mantissa, dtype=np.float64)
        # Sparse buffers end with n_active; dense buffers end with n_point.
        # Any leading axes are homogeneous-population dimensions.
        if src_mantissa.shape[-1:] == point_index.shape:
            src = src_mantissa
        else:
            src = np.take(src_mantissa, point_index, axis=-1)
        incoming = np.asarray(u.Quantity(src, buffer.unit).to_decimal(target_unit), dtype=np.float64)
        new_mantissa = target_mantissa.copy()
        np.put_along_axis(
            new_mantissa,
            np.reshape(point_index, (1,) * (new_mantissa.ndim - 1) + point_index.shape),
            incoming,
            axis=-1,
        )
        return u.Quantity(new_mantissa, target_unit)

    if isinstance(target, tuple):
        if isinstance(buffer, u.Quantity):
            flat = [u.Quantity(value, buffer.unit) for value in np.asarray(buffer.mantissa, dtype=object).reshape(-1)]
            src_arr = np.empty(len(flat), dtype=object)
            for index, value in enumerate(flat):
                src_arr[index] = value
            src_arr = src_arr.reshape(buffer.mantissa.shape)
        elif isinstance(buffer, tuple):
            src_arr = np.asarray(buffer, dtype=object).reshape(np.asarray(buffer, dtype=object).shape)
        else:
            src_arr = np.asarray(buffer, dtype=object)
        src_arr = src_arr if src_arr.shape[-1:] == point_index.shape else np.take(src_arr, point_index, axis=-1)
        leading_shape = src_arr.shape[:-1]
        leading_size = int(np.prod(leading_shape, dtype=int)) if leading_shape else 1
        target_flat = np.empty(len(target), dtype=object)
        for index, value in enumerate(target):
            target_flat[index] = value
        target_arr = target_flat.reshape(leading_shape + (len(target) // leading_size,))
        np.put_along_axis(
            target_arr,
            np.reshape(point_index, (1,) * (target_arr.ndim - 1) + point_index.shape),
            src_arr,
            axis=-1,
        )
        return tuple(target_arr.reshape(-1).tolist())

    if isinstance(target, np.ndarray):
        new_target = target.copy()
        if isinstance(buffer, u.Quantity):
            src = np.asarray(buffer.mantissa)
        elif isinstance(buffer, np.ndarray):
            src = buffer
        else:
            raise TypeError(f"Cannot scatter non-array buffer into numpy target for ion param {param_name!r}.")
        src = src if src.shape[-1:] == point_index.shape else np.take(src, point_index, axis=-1)
        np.put_along_axis(
            new_target,
            np.reshape(point_index, (1,) * (new_target.ndim - 1) + point_index.shape),
            src,
            axis=-1,
        )
        return new_target

    raise TypeError(
        f"Unsupported target/buffer combination for ion param {param_name!r}: "
        f"target={type(target).__name__}, buffer={type(buffer).__name__}."
    )


def _sync_runtime_ion(runtime: CellRuntimeState, *, layout_id: int) -> None:
    """Rebuild the runtime ion's per-point params from state buffers.

    Uses :func:`_ion_param_scatter`, which vectorises via
    ``np.put_along_axis`` on Quantity mantissas instead of Python
    per-index loops.
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
            ion_cls,
            param_name,
            getattr(ion, _ion_runtime_attr_name(ion_cls, param_name)),
        )
        full_values[param_name] = _ion_param_broadcast(baseline, shape=runtime.pop_size + (runtime.n_point,))

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
