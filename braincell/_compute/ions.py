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

"""Build and synchronize runtime ions from signature-derived density parameters.

One runtime ion owns each named pool across its declaration layouts. Numeric
parameters use persistent runtime states and differentiable regional scatters.
Initializers combine explicit regional values with live model defaults; their
states are distinct from the differential species created at initialization.

This module sits below bindings/state and does not import their runtime code.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import brainunit as u
import braintools
import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braincell.ion import build_placeholder_ions
from braincell.ion._base import DynamicNernstIon, InitNernstIon
from braincell.mech import Density, Params, get_registry
from .layouts import MechanismLayout
from .parameters import RuntimeParameterState, _density_signature, parameter_state_value

if TYPE_CHECKING:
    from .state import CellRuntimeState


def _build_runtime_ions(
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
            n_cv=n_cv,
            pop_size=pop_size,
        )
        ions[instance_name] = runtime_ion
        ion_class_candidates.setdefault(record["runtime_cls"].__name__, []).append(instance_name)
        for layout in record["layouts"]:
            ion_runtime_nodes[layout.id] = runtime_ion

    for family_key, default_ion in _build_default_ions(pop_size + (n_cv,)).items():
        if family_key in ion_family_candidates:
            continue
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


@functools.lru_cache(maxsize=1)
def _placeholder_family_keys() -> frozenset[str]:
    """Return the family keys :func:`braincell.ion.build_placeholder_ions` supplies.

    Notes
    -----
    Derived rather than restated. These keys are both the set of families that
    get a placeholder ion when a cell declares none and the set of instance
    names reserved from being reused for a different family; keeping one
    literal list for either would let it drift from the other, which is the
    defect this replaced.
    """
    return frozenset(build_placeholder_ions(size=(1,)))


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

        instance_name = mechanism.instance_name
        if instance_name in _placeholder_family_keys() and instance_name != species_key:
            raise ValueError(
                f"Ion instance name {instance_name!r} conflicts with canonical family key for a different ion family."
            )
        record = instances.get(instance_name)
        if record is None:
            record = {
                "runtime_cls": runtime_cls,
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


def ion_species_key(cls: type) -> str | None:
    """Return the family key of an ion root type, or ``None`` if it is not one.

    Parameters
    ----------
    cls : type
        Any type. Ion roots carry an ``ion_symbol`` class attribute
        (``braincell.ion.Sodium.ion_symbol == 'Na'``); everything else,
        including :class:`~braincell.HHTypedNeuron`, does not.

    Returns
    -------
    str or None
        The lowercased ``ion_symbol`` -- one of ``"na"``, ``"k"``, ``"ca"``,
        ``"no"`` -- or ``None`` for a type that declares no symbol.

    Notes
    -----
    This replaces the four-way ``issubclass`` ladder that both this module and
    :mod:`braincell._compute.bindings` used to carry. The ladder re-derived a
    value ``braincell.ion`` already declares as data, so the family set was
    stated in four places across two modules and nothing forced them to agree
    -- which is how the ``"no"`` family came to be missing from the
    placeholder seed loop. ``getattr`` also subsumes the ``try/except
    TypeError`` the ``bindings`` copy needed for non-class inputs.
    """
    symbol = getattr(cls, "ion_symbol", None)
    return symbol.lower() if isinstance(symbol, str) else None


def _runtime_ion_species_key(cls: type) -> str:
    """Return the family key of a runtime ion class, raising if it has none."""
    species_key = ion_species_key(cls)
    if species_key is None:
        raise ValueError(f"Unsupported ion runtime class {cls.__name__!r}: cannot infer species key.")
    return species_key


@functools.lru_cache(maxsize=None)
def _supported_ion_runtime_params(cls: type) -> tuple[str, ...]:
    return tuple(name for name in _density_signature(cls) if name not in {"size", "name"})


def _ion_runtime_attr_name(cls: type, param_name: str) -> str:
    if param_name == "Ci_initializer" and issubclass(cls, DynamicNernstIon):
        return "_Ci_initializer"
    return param_name


def _instantiate_runtime_ion_instance(
    *,
    instance_name: str,
    runtime_cls: type,
    layouts: tuple[MechanismLayout, ...],
    declarations: tuple[Density, ...],
    state_buffers: dict,
    n_cv: int,
    pop_size: tuple[int, ...] = (),
) -> object:
    """Build one runtime ion instance from its density declaration layouts.

    Merge each layout's numeric buffers with JAX scatters. Initializer
    overrides retain a separate mask so unselected points continue to
    evaluate the model's live defaults.
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

    full_size = pop_size + (n_cv,)
    baseline = runtime_cls(size=full_size)
    fields = tuple(
        dict.fromkeys(field for layout in layouts for layout_id, field in state_buffers if layout_id == layout.id)
    )
    params = {}
    for layout, declaration in zip(layouts, declarations):
        for name, value in declaration.params.items():
            if name not in fields:
                if name in params and Params({name: params[name]}) != Params({name: value}):
                    raise ValueError(f"Ion {instance_name!r} requires uniform constructor configuration {name!r}.")
                params[name] = value
    values = {}
    for name in fields:
        raw = getattr(baseline, _ion_runtime_attr_name(runtime_cls, name))
        if callable(raw):
            raw = braintools.init.param(raw, full_size)
        value = u.math.broadcast_to(raw, full_size)
        for layout in layouts:
            if (layout.id, name) in state_buffers:
                value = _scatter_numeric(value, state_buffers[(layout.id, name)], layout.source_cv_ids)
        values[name] = value
        if not name.endswith("_initializer"):
            if not isinstance(value, u.Quantity):
                concrete = np.asarray(value)
                if np.all(concrete == concrete.flat[0]):
                    value = value.reshape(-1)[0]
            params[name] = value
    node = runtime_cls(size=full_size, name=instance_name, **params)
    node._runtime_ion_parameters = {}
    node._runtime_ion_initial_masks = {}
    for name, value in values.items():
        attr = _ion_runtime_attr_name(runtime_cls, name)
        current = getattr(node, attr)
        state = RuntimeParameterState(value, axis="row", full_shape=full_size)
        node._runtime_ion_parameters[name] = state
        if name.endswith("_initializer"):
            mask = np.zeros(full_size, dtype=bool)
            for layout in layouts:
                buffer = state_buffers.get((layout.id, name))
                if buffer is not None:
                    mask |= buffer.initial_override_mask
            mask_state = brainstate.LongTermState(jnp.asarray(mask))
            node._runtime_ion_initial_masks[name] = mask_state
            setattr(node, attr, state if np.all(mask) else _regional_initializer(current, state, mask_state))
        elif type(current) in (bool, int):
            state.value = u.math.full(full_size, current)
            for layout in layouts:
                buffer = state_buffers.get((layout.id, name))
                if isinstance(buffer, RuntimeParameterState):
                    buffer.value = current
                    buffer.axis = "uniform"
        else:
            state.value = u.math.broadcast_to(current, full_size)
            setattr(node, attr, state)
    return node


def _uniform_scalar(value, name):
    array = u.math.asarray(value)
    if not isinstance(array, jax.core.Tracer):
        concrete = np.asarray(array)
        if not np.all(concrete == concrete.flat[0]):
            raise ValueError(f"Ion constructor configuration {name!r} must be uniform.")
    return array.reshape(-1)[0]


def _scatter_numeric(target, buffer, point_ids):
    value = parameter_state_value(buffer)
    unit = target.unit if isinstance(target, u.Quantity) else None
    raw = value.to_decimal(unit) if unit is not None else value
    old = target.mantissa if unit is not None else target
    ids = jnp.asarray(point_ids, dtype=jnp.int32)
    if raw.shape[-1:] != ids.shape:
        raw = jnp.take(raw, ids, axis=-1)
    result = jnp.asarray(old, dtype=jnp.result_type(old, raw)).at[..., ids].set(raw)
    return u.Quantity(result, unit) if unit is not None else result


def _regional_initializer(default, state, mask):
    def initialize(shape):
        fallback = braintools.init.param(default, shape)
        return u.math.where(mask.value, state.dense_value(), fallback)

    return initialize


def _sync_runtime_ion(runtime: CellRuntimeState, *, layout_id: int, var_name: str | None = None) -> None:
    """Rebuild the runtime ion's per-point params from state buffers.

    Update persistent parameter states so repeated compiled calls observe
    new values without retaining tracers in ordinary Python attributes.
    """
    mechanism = runtime.layout_mechanisms[int(layout_id)]
    if not isinstance(mechanism, Density) or mechanism.category != "ion":
        return
    instance_name = mechanism.instance_name
    ion = runtime.ions[instance_name]
    ion_cls = type(ion)
    states = ion._runtime_ion_parameters
    for param_name in states if var_name is None else (var_name,):
        state = states[param_name]
        value = state.dense_value()
        for candidate in runtime.layouts:
            candidate_mechanism = runtime.layout_mechanisms[candidate.id]
            if not isinstance(candidate_mechanism, Density) or candidate_mechanism.category != "ion":
                continue
            if candidate_mechanism.instance_name != instance_name:
                continue
            buffer = runtime.state_buffers.get((candidate.id, param_name))
            if buffer is not None:
                value = _scatter_numeric(value, buffer, candidate.source_cv_ids)
        attr = _ion_runtime_attr_name(ion_cls, param_name)
        current = vars(ion).get(attr)
        if type(current) in (bool, int):
            setattr(ion, attr, type(current)(_uniform_scalar(value, param_name)))
        state.value = value
    if isinstance(ion, InitNernstIon):
        ion._update_reversal()
