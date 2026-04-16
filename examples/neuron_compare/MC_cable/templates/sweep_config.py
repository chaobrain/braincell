"""Sweep-config schema and Cartesian expansion for multi-compartment cable cases."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from ._shared.schema_common import require_literal, require_mapping, require_str
    from .case_schema import MultiCompartmentCableCase
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    _templates_root = Path(__file__).resolve().parent
    for candidate in (_here, _templates_root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from _shared.schema_common import require_literal, require_mapping, require_str  # type: ignore
    from case_schema import MultiCompartmentCableCase  # type: ignore


_COMMON_SWEEP_PATHS = {
    "swc.path",
    "simulation.dt_ms",
    "simulation.duration_ms",
    "simulation.v_init_mV",
    "cable.ra_ohm_cm",
    "cable.cm_uF_cm2",
    "cv_policy.cv_per_branch",
}
_DC_SWEEP_PATHS = {
    "stimulus.delay_ms",
    "stimulus.dur_ms",
    "stimulus.amp_nA",
}
_PIECEWISE_SWEEP_PATHS = {
    "stimulus.start_ms",
    "stimulus.durations_ms",
    "stimulus.amplitudes_nA",
}
_SINE_SWEEP_PATHS = {
    "stimulus.start_ms",
    "stimulus.duration_ms",
    "stimulus.amplitude_nA",
    "stimulus.frequency_hz",
    "stimulus.phase_rad",
    "stimulus.offset_nA",
}


@dataclass(frozen=True)
class SweepOutputsSpec:
    plot: bool = False


@dataclass(frozen=True)
class SweepCaseGroup:
    group_id: str
    description: str | None
    base_case: dict[str, Any]
    stimulus_kind: str
    sweep_axes: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class SweepConfig:
    template_family: str
    config_id: str
    case_groups: tuple[SweepCaseGroup, ...]
    outputs: SweepOutputsSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepConfig":
        payload = require_mapping(payload, name="config")
        template_family = require_literal(
            payload.get("template_family", "multi_compartment_cable"),
            name="template_family",
            allowed=("multi_compartment_cable",),
        )
        config_id = require_str(payload.get("config_id"), name="config_id")

        case_groups_raw = payload.get("case_groups")
        if not isinstance(case_groups_raw, list) or len(case_groups_raw) == 0:
            raise ValueError("case_groups must be a non-empty list.")

        case_groups: list[SweepCaseGroup] = []
        for index, group_raw in enumerate(case_groups_raw):
            group_data = require_mapping(group_raw, name=f"case_groups[{index}]")
            group_id = require_str(group_data.get("group_id"), name=f"case_groups[{index}].group_id")
            description_raw = group_data.get("description")
            description = None if description_raw is None else require_str(
                description_raw,
                name=f"case_groups[{index}].description",
            )

            base_case_payload = dict(require_mapping(group_data.get("base_case"), name=f"case_groups[{index}].base_case"))
            base_case_payload.setdefault("template_family", template_family)
            base_case_payload.setdefault("case_id", f"{group_id}__BASE")
            base_case = MultiCompartmentCableCase.from_dict(base_case_payload)
            normalized_base_case = json.loads(json.dumps(_case_to_payload(base_case)))
            normalized_base_case.pop("case_id", None)
            stimulus_kind = str(base_case.stimulus.kind)

            sweep_axes_raw = require_mapping(group_data.get("sweep_axes", {}), name=f"case_groups[{index}].sweep_axes")
            allowed_paths = _allowed_sweep_paths(stimulus_kind)
            sweep_axes: dict[str, tuple[Any, ...]] = {}
            for path, values in sweep_axes_raw.items():
                dotted_path = require_str(path, name=f"case_groups[{index}].sweep_axes key")
                if dotted_path not in allowed_paths:
                    raise ValueError(
                        f"Unsupported sweep path {dotted_path!r} for stimulus kind {stimulus_kind!r}."
                    )
                if not isinstance(values, (list, tuple)) or len(values) == 0:
                    raise ValueError(
                        f"case_groups[{index}].sweep_axes[{dotted_path!r}] must be a non-empty list or tuple."
                    )
                sweep_axes[dotted_path] = tuple(values)

            case_groups.append(
                SweepCaseGroup(
                    group_id=group_id,
                    description=description,
                    base_case=normalized_base_case,
                    stimulus_kind=stimulus_kind,
                    sweep_axes=sweep_axes,
                )
            )

        outputs_data = require_mapping(payload.get("outputs", {}), name="outputs")
        outputs = SweepOutputsSpec(plot=bool(outputs_data.get("plot", False)))

        return cls(
            template_family=template_family,
            config_id=config_id,
            case_groups=tuple(case_groups),
            outputs=outputs,
        )


def load_config(config_path: str | Path) -> SweepConfig:
    return SweepConfig.from_dict(json.loads(Path(config_path).read_text()))


def config_to_payload(config: SweepConfig) -> dict[str, Any]:
    return {
        "template_family": config.template_family,
        "config_id": config.config_id,
        "case_groups": [
            {
                "group_id": group.group_id,
                "description": group.description,
                "base_case": group.base_case,
                "sweep_axes": {path: list(values) for path, values in group.sweep_axes.items()},
            }
            for group in config.case_groups
        ],
        "outputs": {"plot": config.outputs.plot},
    }


def expand_cases(config: SweepConfig) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for group in config.case_groups:
        axis_items = list(group.sweep_axes.items())
        products = itertools.product(*(values for _, values in axis_items)) if axis_items else [()]
        for index, combination in enumerate(products):
            payload = json.loads(json.dumps(group.base_case))
            payload["case_id"] = f"{group.group_id}__{index:03d}"
            payload["group_id"] = group.group_id
            for (path, _values), value in zip(axis_items, combination):
                _set_dotted_path(payload, path, value)
            normalized_case = MultiCompartmentCableCase.from_dict(payload)
            normalized_payload = _case_to_payload(normalized_case)
            normalized_payload["group_id"] = group.group_id
            expanded.append(normalized_payload)
    return expanded


def _allowed_sweep_paths(stimulus_kind: str) -> set[str]:
    allowed = set(_COMMON_SWEEP_PATHS)
    if stimulus_kind == "dc_step":
        allowed.update(_DC_SWEEP_PATHS)
    elif stimulus_kind == "piecewise_step":
        allowed.update(_PIECEWISE_SWEEP_PATHS)
    elif stimulus_kind == "sine":
        allowed.update(_SINE_SWEEP_PATHS)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported stimulus kind {stimulus_kind!r}.")
    return allowed


def _case_to_payload(case: MultiCompartmentCableCase) -> dict[str, Any]:
    payload = {
        "template_family": case.template_family,
        "case_id": case.case_id,
        "swc": {
            "path": case.swc.path,
        },
        "simulation": {
            "dt_ms": case.simulation.dt_ms,
            "duration_ms": case.simulation.duration_ms,
            "v_init_mV": case.simulation.v_init_mV,
        },
        "cable": {
            "ra_ohm_cm": case.cable.ra_ohm_cm,
            "cm_uF_cm2": case.cable.cm_uF_cm2,
        },
        "cv_policy": {
            "kind": case.cv_policy.kind,
            "cv_per_branch": case.cv_policy.cv_per_branch,
        },
        "stimulus": _stimulus_to_payload(case.stimulus),
    }
    return payload


def _stimulus_to_payload(stimulus: Any) -> dict[str, Any]:
    payload = {
        "kind": stimulus.kind,
        "target": stimulus.target,
    }
    if stimulus.kind == "dc_step":
        payload.update(
            {
                "delay_ms": stimulus.delay_ms,
                "dur_ms": stimulus.dur_ms,
                "amp_nA": stimulus.amp_nA,
            }
        )
        return payload
    if stimulus.kind == "piecewise_step":
        payload.update(
            {
                "start_ms": stimulus.start_ms,
                "durations_ms": list(stimulus.durations_ms),
                "amplitudes_nA": list(stimulus.amplitudes_nA),
            }
        )
        return payload
    payload.update(
        {
            "start_ms": stimulus.start_ms,
            "duration_ms": stimulus.duration_ms,
            "amplitude_nA": stimulus.amplitude_nA,
            "frequency_hz": stimulus.frequency_hz,
            "phase_rad": stimulus.phase_rad,
            "offset_nA": stimulus.offset_nA,
        }
    )
    return payload


def _set_dotted_path(payload: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = payload
    for key in keys[:-1]:
        next_cursor = cursor.get(key)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[key] = next_cursor
        cursor = next_cursor
    cursor[keys[-1]] = value
