"""Schema and sweep synthesis for multi-compartment cable comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from ._shared.schema_common import require_mapping, require_str, require_submapping
    from .case_schema import MultiCompartmentCableCase
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from _shared.schema_common import require_mapping, require_str, require_submapping  # type: ignore
    from case_schema import MultiCompartmentCableCase  # type: ignore


_COMMON_SWEEP_PATHS = {
    "morphology.kind",
    "morphology.path",
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
_LEGACY_CONFIG_KEYS = {
    "config_id",
    "base_case",
    "case_groups",
    "sweep_axes",
    "template_family",
}
_LEGACY_TEMPLATE_KEYS = {
    "config_id",
    "base_case",
    "case_groups",
    "template_family",
}


@dataclass(frozen=True)
class SweepOutputsSpec:
    plot: bool = False


@dataclass(frozen=True)
class SweepCaseGroup:
    group_id: str
    description: str | None
    sweep_axes: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class ModelConfig:
    config_name: str
    config_path: Path
    meta: dict[str, Any]
    template_paths: tuple[Path, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, config_path: Path) -> "ModelConfig":
        payload = require_mapping(payload, name="config")
        _reject_legacy_config_shape(payload)
        return cls(
            config_name=config_path.stem,
            config_path=config_path,
            meta=_normalize_meta_payload(payload.get("meta"), name="meta"),
            template_paths=_resolve_template_paths(payload.get("templates"), config_path=config_path),
        )


@dataclass(frozen=True)
class ScanTemplate:
    template_name: str
    template_path: Path
    meta: dict[str, Any]
    base: dict[str, Any]
    group_id: str
    group_description: str | None
    raw_sweep_axes: dict[str, tuple[Any, ...]]
    outputs: SweepOutputsSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, template_path: Path) -> "ScanTemplate":
        payload = require_mapping(payload, name="scan_template")
        _reject_legacy_template_shape(payload)
        group_payload = require_submapping(payload, "group")
        outputs_data = require_mapping(payload.get("outputs", {}), name="outputs")
        return cls(
            template_name=template_path.stem,
            template_path=template_path,
            meta=_normalize_meta_payload(payload.get("meta"), name="meta"),
            base=dict(require_submapping(payload, "base")),
            group_id=require_str(group_payload.get("group_id"), name="group.group_id"),
            group_description=(
                None
                if group_payload.get("description") is None
                else require_str(group_payload.get("description"), name="group.description")
            ),
            raw_sweep_axes=_parse_unrestricted_axes(
                require_mapping(group_payload.get("sweep_axes", {}), name="group.sweep_axes"),
                name="group.sweep_axes",
            ),
            outputs=SweepOutputsSpec(plot=bool(outputs_data.get("plot", False))),
        )


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    config_name: str
    template_name: str
    config_path: Path
    template_path: Path
    config_meta: dict[str, Any]
    template_meta: dict[str, Any]
    base_case: dict[str, Any]
    group: SweepCaseGroup
    outputs: SweepOutputsSpec


def load_model_config(config_path: str | Path) -> ModelConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    payload = json.loads(resolved_config_path.read_text())
    return ModelConfig.from_dict(payload, config_path=resolved_config_path)


def load_scan_template(template_path: str | Path) -> ScanTemplate:
    resolved_template_path = Path(template_path).expanduser().resolve()
    payload = json.loads(resolved_template_path.read_text())
    return ScanTemplate.from_dict(payload, template_path=resolved_template_path)


def load_case(case_path: str | Path) -> MultiCompartmentCableCase:
    payload = json.loads(Path(case_path).read_text())
    return MultiCompartmentCableCase.from_dict(payload)


def load_sweep_config(config_path: str | Path, template_path: str | Path) -> SweepConfig:
    model_config = load_model_config(config_path)
    resolved_template_path = _resolve_template_reference(
        template_path=template_path,
        config_dir=model_config.config_path.parent,
    )
    if resolved_template_path not in model_config.template_paths:
        raise ValueError(
            "Template path is not declared in config.templates: "
            f"{resolved_template_path!s}."
        )
    template = load_scan_template(resolved_template_path)
    return build_sweep_config(model_config, template)


def build_sweep_config(model_config: ModelConfig, scan_template: ScanTemplate) -> SweepConfig:
    run_id = f"{model_config.config_name}__{scan_template.template_name}"
    normalized_base_case = _normalize_case_payload(
        dict(scan_template.base),
        case_id=f"{run_id}__BASE",
    )
    normalized_base_case.pop("case_id", None)

    allowed_paths = _allowed_sweep_paths(
        stimulus_kind=str(normalized_base_case["stimulus"]["kind"]),
    )
    sweep_axes = _parse_axes(
        scan_template.raw_sweep_axes,
        name="group.sweep_axes",
        allowed_paths=allowed_paths,
    )

    return SweepConfig(
        config_id=run_id,
        config_name=model_config.config_name,
        template_name=scan_template.template_name,
        config_path=model_config.config_path,
        template_path=scan_template.template_path,
        config_meta=dict(model_config.meta),
        template_meta=dict(scan_template.meta),
        base_case=normalized_base_case,
        group=SweepCaseGroup(
            group_id=scan_template.group_id,
            description=scan_template.group_description,
            sweep_axes=sweep_axes,
        ),
        outputs=scan_template.outputs,
    )


def expand_cases(config: SweepConfig) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    axis_items = list(config.group.sweep_axes.items())
    products = itertools.product(*(values for _, values in axis_items)) if axis_items else [()]
    for index, combination in enumerate(products):
        payload = _deep_merge_dicts(config.base_case, {})
        payload["case_id"] = f"{config.group.group_id}__{index:03d}"
        payload["group_id"] = config.group.group_id
        for (path, _values), value in zip(axis_items, combination):
            _set_dotted_path(payload, path, value)
        normalized_case = MultiCompartmentCableCase.from_dict(payload)
        normalized_payload = case_to_payload(normalized_case)
        normalized_payload["group_id"] = config.group.group_id
        expanded.append(normalized_payload)
    return expanded


def config_to_payload(config: SweepConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "config_name": config.config_name,
        "template_name": config.template_name,
        "config_path": str(config.config_path),
        "template_path": str(config.template_path),
        "meta": {
            "config": dict(config.config_meta),
            "template": dict(config.template_meta),
        },
        "template": {
            "base": _strip_case_runtime_fields(config.base_case),
            "group": {
                "group_id": config.group.group_id,
                "description": config.group.description,
                "sweep_axes": {path: list(values) for path, values in config.group.sweep_axes.items()},
            },
            "outputs": {
                "plot": config.outputs.plot,
            },
        },
    }


def case_to_payload(case: MultiCompartmentCableCase) -> dict[str, Any]:
    return {
        "template_family": case.template_family,
        "case_id": case.case_id,
        "morphology": {
            "kind": case.morphology.kind,
            "path": case.morphology.path,
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


def _normalize_case_payload(
    payload: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    resolved = dict(payload)
    resolved.setdefault("case_id", case_id)
    normalized_case = MultiCompartmentCableCase.from_dict(resolved)
    return case_to_payload(normalized_case)


def _resolve_template_reference(*, template_path: str | Path, config_dir: Path) -> Path:
    raw_path = Path(template_path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (config_dir / raw_path).resolve()


def _allowed_sweep_paths(*, stimulus_kind: str) -> set[str]:
    allowed = set(_COMMON_SWEEP_PATHS)
    if stimulus_kind == "dc_step":
        allowed.update(_DC_SWEEP_PATHS)
        return allowed
    if stimulus_kind == "piecewise_step":
        allowed.update(_PIECEWISE_SWEEP_PATHS)
        return allowed
    if stimulus_kind == "sine":
        allowed.update(_SINE_SWEEP_PATHS)
        return allowed
    raise ValueError(f"Unsupported stimulus kind {stimulus_kind!r}.")


def _parse_axes(
    axes_raw: Mapping[str, Any],
    *,
    name: str,
    allowed_paths: set[str],
) -> dict[str, tuple[Any, ...]]:
    parsed: dict[str, tuple[Any, ...]] = {}
    for path, values in axes_raw.items():
        dotted_path = require_str(path, name=f"{name} key")
        if dotted_path not in allowed_paths:
            raise ValueError(f"Unsupported sweep path {dotted_path!r}.")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"{name}[{dotted_path!r}] must be a non-empty list or tuple.")
        parsed[_normalize_sweep_path(dotted_path)] = tuple(values)
    return parsed


def _parse_unrestricted_axes(
    axes_raw: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, tuple[Any, ...]]:
    parsed: dict[str, tuple[Any, ...]] = {}
    for path, values in axes_raw.items():
        dotted_path = require_str(path, name=f"{name} key")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"{name}[{dotted_path!r}] must be a non-empty list or tuple.")
        parsed[dotted_path] = tuple(values)
    return parsed


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


def _normalize_sweep_path(path: str) -> str:
    if path == "swc.path":
        return "morphology.path"
    return path


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


def _deep_merge_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _strip_case_runtime_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    for key in ("case_id", "group_id"):
        stripped.pop(key, None)
    return stripped


def _normalize_meta_payload(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(require_mapping(value, name=name))


def _resolve_template_paths(templates_raw: Any, *, config_path: Path) -> tuple[Path, ...]:
    if not isinstance(templates_raw, list) or len(templates_raw) == 0:
        raise ValueError("templates must be a non-empty list of JSON paths.")
    resolved_paths: list[Path] = []
    for index, item in enumerate(templates_raw):
        text = require_str(item, name=f"templates[{index}]")
        resolved_path = _resolve_template_reference(template_path=text, config_dir=config_path.parent)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Unknown scan template path: {resolved_path!s}.")
        resolved_paths.append(resolved_path)
    return tuple(resolved_paths)


def _reject_legacy_config_shape(payload: Mapping[str, Any]) -> None:
    if _LEGACY_CONFIG_KEYS & set(payload):
        raise ValueError(
            "Legacy cable sweep config schema is no longer supported; "
            "use a model config with templates."
        )


def _reject_legacy_template_shape(payload: Mapping[str, Any]) -> None:
    if _LEGACY_TEMPLATE_KEYS & set(payload):
        raise ValueError(
            "Legacy cable scan template schema is no longer supported; "
            "use a base/group template JSON."
        )


__all__ = [
    "ModelConfig",
    "MultiCompartmentCableCase",
    "ScanTemplate",
    "SweepConfig",
    "SweepCaseGroup",
    "SweepOutputsSpec",
    "build_sweep_config",
    "case_to_payload",
    "config_to_payload",
    "expand_cases",
    "load_case",
    "load_model_config",
    "load_scan_template",
    "load_sweep_config",
]
