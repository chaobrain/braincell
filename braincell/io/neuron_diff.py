from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
import math
from pathlib import Path
from typing import Any

import brainunit as u

from braincell.morpho import Morpho

__all__ = [
    "compare_swc_with_neuron",
    "compute_braincell_metrics",
    "compute_neuron_metrics",
    "load_swc_morphology",
    "supported_metric_names",
]

_LOADED_NEURON_SECTIONS: list[Any] = []
_DEFAULT_METRIC_NAMES = (
    "total_length",
    "total_area",
    "total_volume",
    "mean_radius",
    "n_branches",
    "n_stems",
    "n_bifurcations",
    "max_branch_order",
    "max_path_distance",
)
_ALL_METRIC_NAMES = _DEFAULT_METRIC_NAMES
_SUMMARY_METRIC_NAMES = {
    "total_length",
    "total_area",
    "total_volume",
    "mean_radius",
    "n_branches",
    "n_stems",
    "n_bifurcations",
    "max_branch_order",
}


def _available_metric(value: float | int, *, unit: str | None) -> dict[str, object]:
    return {"available": True, "value": value, "unit": unit}


def _missing_metric(reason: str, *, unit: str | None) -> dict[str, object]:
    return {"available": False, "value": None, "unit": unit, "reason": reason}


def _as_float(quantity, unit) -> float:
    return float(quantity.to_decimal(unit))


def _resolve_metric_names(
    *,
    metric_names: Iterable[str] | None,
    include_optional: bool,
) -> tuple[str, ...]:
    del include_optional
    if metric_names is None:
        return _ALL_METRIC_NAMES

    resolved: list[str] = []
    for name in metric_names:
        if name not in _ALL_METRIC_NAMES:
            raise KeyError(f"Unsupported comparison metric {name!r}.")
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def supported_metric_names(*, include_optional: bool = False) -> tuple[str, ...]:
    del include_optional
    return _ALL_METRIC_NAMES


def _validate_swc_path(swc_filename: str | Path) -> Path:
    path = Path(swc_filename)
    if path.suffix.lower() != ".swc":
        raise ValueError(f"compare_swc_with_neuron only supports .swc files, got {path!s}.")
    return path


def _import_neuron_h():
    try:
        from neuron import h
    except ImportError as exc:
        raise ImportError("neuron_diff requires the `neuron` package to be installed.") from exc
    return h


def _clear_loaded_neuron_sections() -> None:
    if not _LOADED_NEURON_SECTIONS:
        return

    h = _import_neuron_h()
    while _LOADED_NEURON_SECTIONS:
        sec = _LOADED_NEURON_SECTIONS.pop()
        try:
            h.delete_section(sec=sec)
        except Exception:
            continue


def load_swc_morphology(swc_filename: str | Path) -> tuple[Any, ...]:
    """Load a SWC file through NEURON import3d and return its instantiated sections."""

    path = _validate_swc_path(swc_filename)
    h = _import_neuron_h()
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    _clear_loaded_neuron_sections()

    existing_count = sum(1 for _ in h.allsec())
    cell = h.Import3d_SWC_read()
    cell.input(str(path))
    h.Import3d_GUI(cell, 0).instantiate(None)

    all_sections = tuple(h.allsec())
    sections = all_sections[existing_count:]
    if not sections:
        raise RuntimeError(f"NEURON import3d instantiated no sections from {str(path)!r}.")
    _LOADED_NEURON_SECTIONS.extend(sections)
    return sections


def _braincell_metric_record(morpho: Morpho, summary: dict[str, object], metric_name: str) -> dict[str, object]:
    if metric_name in _SUMMARY_METRIC_NAMES:
        value = summary[metric_name]
        if metric_name in {"n_branches", "n_stems", "n_bifurcations", "max_branch_order"}:
            return _available_metric(int(value), unit="count")
        unit = {
            "total_length": ("um", u.um),
            "total_area": ("um^2", u.um ** 2),
            "total_volume": ("um^3", u.um ** 3),
            "mean_radius": ("um", u.um),
        }[metric_name]
        return _available_metric(_as_float(value, unit[1]), unit=unit[0])

    if metric_name == "max_path_distance":
        return _available_metric(_as_float(morpho.metric.max_path_distance, u.um), unit="um")

    raise KeyError(f"Unsupported braincell metric {metric_name!r}.")


def compute_braincell_metrics(
    swc_filename: str | Path,
    *,
    swc_options=None,
    metric_names: Iterable[str] | None = None,
    include_optional: bool = False,
) -> dict[str, dict[str, object]]:
    path = _validate_swc_path(swc_filename)
    selected_metric_names = _resolve_metric_names(metric_names=metric_names, include_optional=include_optional)
    morpho = Morpho.from_swc(path, options=swc_options)
    summary = morpho.summary()
    return {
        metric_name: _braincell_metric_record(morpho, summary, metric_name)
        for metric_name in selected_metric_names
    }


def _segment_length_um(sec: Any) -> float:
    nseg = int(sec.nseg)
    return float(sec.L) / nseg if nseg > 0 else float(sec.L)


def _extract_neuron_geometry(sec: Any) -> dict[str, object]:
    h = _import_neuron_h()
    length_um = float(sec.L)
    segment_length_um = _segment_length_um(sec)
    areas_um2 = [float(h.area(seg.x, sec=sec)) for seg in sec]
    volumes_um3 = [float(seg.volume()) for seg in sec]

    total_length_um = segment_length_um * len(areas_um2)
    mean_radius_um = None
    if total_length_um > 0.0:
        mean_radius_um = sum(segment_length_um * 0.5 * float(seg.diam) for seg in sec) / total_length_um

    return {
        "name": sec.name(),
        "length_um": length_um,
        "area_um2": sum(areas_um2),
        "volume_um3": sum(volumes_um3),
        "mean_radius_um": mean_radius_um,
        "reason": None,
    }


def _build_neuron_state(neuron_sections: Sequence[Any]) -> dict[str, object]:
    h = _import_neuron_h()
    sections = tuple(neuron_sections)
    names = tuple(sec.name() for sec in sections)
    name_set = set(names)
    section_by_name = {sec.name(): sec for sec in sections}
    parent_by_name: dict[str, str | None] = {}
    children_by_name: dict[str, list[str]] = {name: [] for name in names}

    for sec in sections:
        name = sec.name()
        ref = h.SectionRef(sec=sec)
        if ref.has_parent():
            parent_name = ref.parent.name()
            parent_by_name[name] = parent_name if parent_name in name_set else None
            if parent_name in name_set:
                children_by_name[parent_name].append(name)
        else:
            parent_by_name[name] = None

    roots = tuple(name for name in names if parent_by_name[name] is None) or (names[0],)
    geometry_by_name = {sec.name(): _extract_neuron_geometry(sec) for sec in sections}
    depth_by_name = _compute_depths(
        roots=roots,
        children_by_name={name: tuple(children_by_name[name]) for name in names},
    )
    return {
        "sections": sections,
        "section_by_name": section_by_name,
        "names": names,
        "roots": roots,
        "children_by_name": {name: tuple(children_by_name[name]) for name in names},
        "parent_by_name": parent_by_name,
        "geometry_by_name": geometry_by_name,
        "depth_by_name": depth_by_name,
    }


def _compute_depths(
    *,
    roots: Sequence[str],
    children_by_name: dict[str, tuple[str, ...]],
) -> dict[str, int]:
    depths: dict[str, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        name, depth = queue.popleft()
        existing = depths.get(name)
        if existing is not None and existing <= depth:
            continue
        depths[name] = depth
        for child_name in children_by_name[name]:
            queue.append((child_name, depth + 1))
    return depths


def _root_name_for_branch(state: dict[str, object], branch_name: str) -> str:
    current = branch_name
    while state["parent_by_name"][current] is not None:
        current = state["parent_by_name"][current]
    return current


def _neuron_metric_record(state: dict[str, object], metric_name: str) -> dict[str, object]:
    geometry_by_name = state["geometry_by_name"]
    names = state["names"]
    children_by_name = state["children_by_name"]

    if metric_name == "total_length":
        return _available_metric(sum(geometry_by_name[name]["length_um"] for name in names), unit="um")

    if metric_name == "total_area":
        return _available_metric(sum(geometry_by_name[name]["area_um2"] for name in names), unit="um^2")

    if metric_name == "total_volume":
        return _available_metric(sum(geometry_by_name[name]["volume_um3"] for name in names), unit="um^3")

    if metric_name == "mean_radius":
        total_length_um = float(sum(geometry_by_name[name]["length_um"] for name in names))
        if total_length_um <= 0.0:
            return _missing_metric("total section length must be > 0", unit="um")
        weighted_radius_sum = sum(
            geometry_by_name[name]["length_um"] * geometry_by_name[name]["mean_radius_um"]
            for name in names
        )
        return _available_metric(float(weighted_radius_sum / total_length_um), unit="um")

    if metric_name == "n_branches":
        return _available_metric(len(names), unit="count")

    if metric_name == "n_stems":
        return _available_metric(sum(len(children_by_name[root]) for root in state["roots"]), unit="count")

    if metric_name == "n_bifurcations":
        return _available_metric(int(sum(len(children_by_name[name]) >= 2 for name in names)), unit="count")

    if metric_name == "max_branch_order":
        return _available_metric(max(state["depth_by_name"].values(), default=0), unit="count")

    if metric_name == "max_path_distance":
        h = _import_neuron_h()
        section_by_name = state["section_by_name"]
        terminal_names = tuple(name for name in names if len(children_by_name[name]) == 0)
        if not terminal_names:
            return _available_metric(0.0, unit="um")

        max_distance_um = 0.0
        for name in terminal_names:
            root_name = _root_name_for_branch(state, name)
            h.distance(0, section_by_name[root_name](0.0))
            max_distance_um = max(max_distance_um, float(h.distance(section_by_name[name](1.0))))
        return _available_metric(max_distance_um, unit="um")

    raise KeyError(f"Unsupported neuron metric {metric_name!r}.")


def compute_neuron_metrics(
    neuron_sections: Sequence[Any],
    *,
    metric_names: Iterable[str] | None = None,
    include_optional: bool = False,
) -> dict[str, dict[str, object]]:
    selected_metric_names = _resolve_metric_names(metric_names=metric_names, include_optional=include_optional)
    state = _build_neuron_state(neuron_sections)
    return {
        metric_name: _neuron_metric_record(state, metric_name)
        for metric_name in selected_metric_names
    }


def _compare_metric_records(
    braincell_record: dict[str, object],
    neuron_record: dict[str, object],
) -> dict[str, object]:
    unit = braincell_record.get("unit") or neuron_record.get("unit")
    if not braincell_record["available"] or not neuron_record["available"]:
        reasons = []
        if not braincell_record["available"]:
            reasons.append(f"braincell: {braincell_record['reason']}")
        if not neuron_record["available"]:
            reasons.append(f"neuron: {neuron_record['reason']}")
        return {"available": False, "unit": unit, "abs_diff": None, "rel_diff": None, "reason": "; ".join(reasons)}

    braincell_value = float(braincell_record["value"])
    neuron_value = float(neuron_record["value"])
    abs_diff = abs(neuron_value - braincell_value)
    baseline = abs(neuron_value)
    rel_diff = (
        0.0
        if math.isclose(baseline, 0.0) and math.isclose(abs_diff, 0.0)
        else (None if math.isclose(baseline, 0.0) else abs_diff / baseline)
    )
    return {"available": True, "unit": unit, "abs_diff": abs_diff, "rel_diff": rel_diff}


def compare_swc_with_neuron(
    swc_filename: str | Path,
    *,
    swc_options=None,
    metric_names: Iterable[str] | None = None,
    include_optional: bool = False,
) -> dict[str, object]:
    path = _validate_swc_path(swc_filename)
    selected_metric_names = _resolve_metric_names(metric_names=metric_names, include_optional=include_optional)
    braincell_metrics = compute_braincell_metrics(
        path,
        swc_options=swc_options,
        metric_names=selected_metric_names,
        include_optional=include_optional,
    )
    neuron_metrics = compute_neuron_metrics(
        load_swc_morphology(path),
        metric_names=selected_metric_names,
        include_optional=include_optional,
    )
    return {
        "path": str(path),
        "selected_metrics": selected_metric_names,
        "braincell": braincell_metrics,
        "neuron": neuron_metrics,
        "diff": {
            metric_name: _compare_metric_records(braincell_metrics[metric_name], neuron_metrics[metric_name])
            for metric_name in selected_metric_names
        },
    }
