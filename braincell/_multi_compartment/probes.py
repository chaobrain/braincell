"""Probe sampling helpers for :class:`Cell`.

Each helper takes an explicit :class:`Cell` (expected to be in the
``INITIALIZED`` phase — i.e. after :meth:`Cell.init_state`).
"""

from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import numpy as np

from braincell._compute.runtime import CellRuntimeState
from braincell.mech import (
    CurrentProbe,
    Density,
    MechanismProbe,
    StateProbe,
)
from . import bridge

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["sample_probe", "sample_probes"]


def sample_probe(rcell: "Cell", name: str) -> object:
    """Return the current sample for the probe named ``name``."""
    runtime = rcell.runtime
    matches: list[tuple[object, object]] = []
    for layout in runtime.layouts:
        declaration = runtime.get_layout_mechanism(layout.id)
        if isinstance(declaration, (StateProbe, MechanismProbe, CurrentProbe)) \
            and _probe_name(declaration) == name:
            matches.append((layout, declaration))
    if len(matches) == 0:
        raise KeyError(f"No probe is registered with name {name!r}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple probes share the same name {name!r}; probe names must be unique."
        )
    layout, declaration = matches[0]
    return _sample_probe_layout(rcell, runtime, layout=layout, declaration=declaration)


def sample_probes(rcell: "Cell") -> dict[str, object]:
    """Return a ``{probe_name: sample}`` dict for every placed probe."""
    runtime = rcell.runtime
    sampled: dict[str, object] = {}
    for layout in runtime.layouts:
        declaration = runtime.get_layout_mechanism(layout.id)
        if not isinstance(declaration, (StateProbe, MechanismProbe, CurrentProbe)):
            continue
        probe_name = _probe_name(declaration)
        if probe_name in sampled:
            raise ValueError(
                f"Multiple probes share the same name {probe_name!r}; probe names must be unique."
            )
        sampled[probe_name] = _sample_probe_layout(
            rcell, runtime, layout=layout, declaration=declaration,
        )
    return sampled


def _sample_probe_layout(
    rcell: "Cell",
    runtime: CellRuntimeState,
    *,
    layout: object,
    declaration: object,
) -> object:
    point_ids = getattr(layout, "point_index", None)
    if point_ids is None:
        raise ValueError(
            f"Probe layout {getattr(layout, 'id', None)!r} is missing point_index."
        )
    samples = []
    for point_id in point_ids.tolist():
        if isinstance(declaration, StateProbe):
            samples.append(
                _sample_state_probe_point(
                    rcell, runtime, declaration=declaration, point_id=int(point_id),
                )
            )
        elif isinstance(declaration, MechanismProbe):
            samples.append(
                _sample_mechanism_probe_point(
                    rcell, runtime, declaration=declaration, point_id=int(point_id),
                )
            )
        elif isinstance(declaration, CurrentProbe):
            samples.append(
                _sample_current_probe_point(
                    rcell, runtime, declaration=declaration, point_id=int(point_id),
                )
            )
        else:  # pragma: no cover
            raise TypeError(
                f"Unsupported probe declaration type {type(declaration).__name__!s}."
            )
    return _pack_probe_samples(samples)


def _sample_state_probe_point(
    rcell: "Cell",
    runtime: CellRuntimeState,
    *,
    declaration: StateProbe,
    point_id: int,
) -> object:
    if declaration.field != "v":
        raise ValueError(f"Unsupported StateProbe field {declaration.field!r}.")
    cv_id = _representative_cv_id(runtime, point_id=point_id)
    return _select_last_axis(rcell.V.value, cv_id)


def _sample_mechanism_probe_point(
    rcell: "Cell",
    runtime: CellRuntimeState,
    *,
    declaration: MechanismProbe,
    point_id: int,
) -> object:
    matched_layouts = []
    for layout in runtime.get_point_layouts(point_id):
        mechanism = runtime.get_layout_mechanism(layout.id)
        if isinstance(mechanism, Density) and mechanism.instance_name == declaration.mechanism:
            matched_layouts.append(layout)
    if len(matched_layouts) > 1:
        raise ValueError(
            f"Probe {_probe_name(declaration)!r} matched multiple mechanisms named "
            f"{declaration.mechanism!r} at point {point_id!r}."
        )
    if len(matched_layouts) == 1:
        node = runtime.get_runtime_node(matched_layouts[0].id)
        raw = _probe_state_attr(
            node, declaration.field, probe_name=_probe_name(declaration),
        )
        return _select_last_axis(raw.value, point_id)

    try:
        ion = runtime.get_ion(declaration.mechanism)
    except KeyError:
        ion = None
    if ion is not None:
        raw = _probe_state_attr(
            ion, declaration.field, probe_name=_probe_name(declaration),
        )
        return _select_last_axis(raw.value, point_id)

    raise KeyError(
        f"Probe {_probe_name(declaration)!r} could not find a mechanism or ion named "
        f"{declaration.mechanism!r} at point {point_id!r}."
    )


def _sample_current_probe_point(
    rcell: "Cell",
    runtime: CellRuntimeState,
    *,
    declaration: CurrentProbe,
    point_id: int,
) -> object:
    point_V = bridge.cv_to_point(rcell.V.value, runtime)
    if declaration.mechanism is not None:
        matched_layouts = []
        for layout in runtime.get_point_layouts(point_id):
            mechanism = runtime.get_layout_mechanism(layout.id)
            if isinstance(mechanism, Density) and mechanism.instance_name == declaration.mechanism:
                matched_layouts.append(layout)
        if len(matched_layouts) > 1:
            raise ValueError(
                f"Probe {_probe_name(declaration)!r} matched multiple mechanisms named "
                f"{declaration.mechanism!r} at point {point_id!r}."
            )
        if len(matched_layouts) == 0:
            raise KeyError(
                f"Probe {_probe_name(declaration)!r} could not find a mechanism named "
                f"{declaration.mechanism!r} at point {point_id!r}."
            )
        layout_id = matched_layouts[0].id
        node = runtime.get_runtime_node(layout_id)
        bound_ion_keys = runtime.bound_ion_keys.get(layout_id, ())
        if len(bound_ion_keys) > 1:
            current = node.current(
                point_V,
                *tuple(runtime.get_ion(ion_key).pack_info() for ion_key in bound_ion_keys),
            )
        else:
            ion_info = _probe_current_ion_info(
                runtime,
                declaration=declaration,
                mechanism=node,
                layout_id=layout_id,
            )
            current = _probe_current_value(
                node, point_V, ion_info, probe_name=_probe_name(declaration),
            )
        return _select_last_axis(current, point_id)

    if declaration.ion is None:
        raise ValueError(
            f"Probe {_probe_name(declaration)!r} must define 'ion' when 'mechanism' is omitted."
        )
    ion = runtime.get_ion(declaration.ion)
    current = ion.current(point_V, include_external=False)
    return _select_last_axis(current, point_id)


def _midpoint_cv_id(runtime: CellRuntimeState, *, point_id: int) -> int:
    matches = np.flatnonzero(runtime.node_tree.cv_to_mid_node_id == int(point_id))
    if len(matches) != 1:
        raise ValueError(
            f"Point {point_id!r} is not a unique CV midpoint; got CV matches {matches.tolist()!r}."
        )
    return int(matches[0])


def _representative_cv_id(runtime: CellRuntimeState, *, point_id: int) -> int:
    matches = np.flatnonzero(runtime.node_tree.cv_to_mid_node_id == int(point_id))
    if len(matches) == 1:
        return int(matches[0])
    roles = runtime.node_tree.nodes[int(point_id)].roles
    if len(roles) == 0:
        raise ValueError(f"Point {point_id!r} is not associated with any CV.")
    return int(roles[0].cv_id)


def _select_last_axis(value: object, index: int) -> object:
    value = value.value if isinstance(value, brainstate.State) else value
    shape = getattr(value, "shape", ())
    if shape in (None, ()):
        return value
    return value[..., int(index)]


def _probe_state_attr(owner: object, field: str, *, probe_name: str) -> brainstate.State:
    if not hasattr(owner, field):
        raise KeyError(
            f"Probe {probe_name!r} field {field!r} was not found on {type(owner).__name__!s}."
        )
    raw = getattr(owner, field)
    if not isinstance(raw, brainstate.State):
        raise ValueError(
            f"Probe {probe_name!r} field {field!r} on {type(owner).__name__!s} is not a runtime state."
        )
    return raw


def _probe_current_ion_info(
    runtime: CellRuntimeState,
    *,
    declaration: CurrentProbe,
    mechanism: object,
    layout_id: int | None = None,
) -> object | None:
    if declaration.ion is not None:
        return runtime.get_ion(declaration.ion).pack_info()

    if layout_id is not None:
        owner_key = runtime.current_owner_keys.get(int(layout_id))
        if owner_key is not None:
            return runtime.get_ion(owner_key).pack_info()

    mechanism_name = declaration.mechanism
    if mechanism_name is None:
        return None

    for ion in runtime.ions.values():
        channels = getattr(ion, "channels", None)
        if isinstance(channels, dict) and mechanism_name in channels and channels[mechanism_name] is mechanism:
            return ion.pack_info()
    return None


def _probe_current_value(
    owner: object,
    point_V: object,
    ion_info: object | None,
    *,
    probe_name: str,
) -> object:
    if not hasattr(owner, "current"):
        raise KeyError(
            f"Probe {probe_name!r} target {type(owner).__name__!s} has no current(...) method."
        )
    if ion_info is None:
        return owner.current(point_V)
    try:
        return owner.current(point_V, ion_info)
    except TypeError:
        return owner.current(point_V)


def _pack_probe_samples(samples: list[object]) -> object:
    if len(samples) == 0:
        return u.math.asarray([])
    if len(samples) == 1:
        return samples[0]
    first = samples[0]
    if hasattr(first, "unit"):
        unit = first.unit
        decimals = [sample.to_decimal(unit) for sample in samples]
        return u.Quantity(u.math.asarray(decimals), unit)
    return u.math.asarray(samples)


def _probe_name(declaration: "StateProbe | MechanismProbe | CurrentProbe") -> str:
    if declaration.name is None:
        raise ValueError(
            f"Probe declaration {type(declaration).__name__!s} has no resolved name."
        )
    return declaration.name
