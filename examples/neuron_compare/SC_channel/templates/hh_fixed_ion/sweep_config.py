"""Sweep-config schema and Cartesian expansion for HH + fixed-ion cases."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .._shared.schema_common import require_literal, require_mapping, require_str, require_submapping
    from .case_schema import SingleCompartmentChannelHHFixedIonCase
except ImportError:  # pragma: no cover
    import importlib.util
    import sys

    def _load_local_module(module_name: str, path: Path):
        module_key = f"sc_channel_hh_fixed_ion__{module_name}"
        module = sys.modules.get(module_key)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(module_key, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        return module

    _here = Path(__file__).resolve().parent
    _templates_root = _here.parent
    _schema_common = _load_local_module("schema_common", _templates_root / "_shared" / "schema_common.py")
    require_literal = _schema_common.require_literal
    require_mapping = _schema_common.require_mapping
    require_str = _schema_common.require_str
    require_submapping = _schema_common.require_submapping
    _case_schema = _load_local_module("case_schema", _here / "case_schema.py")
    SingleCompartmentChannelHHFixedIonCase = _case_schema.SingleCompartmentChannelHHFixedIonCase


_COMMON_SWEEP_PATHS = {
    "simulation.v_init_mV",
    "simulation.temperature_celsius",
    "channel_overrides.g_max_S_cm2",
    "channel_overrides.v12_mV",
    "channel_overrides.q",
    "leak.enabled",
    "leak.g_S_cm2",
    "leak.e_mV",
}
_DC_SWEEP_PATHS = {
    "stimulus.amp_nA",
    "stimulus.delay_ms",
    "stimulus.dur_ms",
}
_SINE_SWEEP_PATHS = {
    "stimulus.start_ms",
    "stimulus.duration_ms",
    "stimulus.amplitude_nA",
    "stimulus.frequency_hz",
    "stimulus.phase_rad",
    "stimulus.offset_nA",
}
_ALL_SWEEP_PATHS = _COMMON_SWEEP_PATHS | _DC_SWEEP_PATHS | _SINE_SWEEP_PATHS


@dataclass(frozen=True)
class SweepOutputsSpec:
    plot: bool = False


@dataclass(frozen=True)
class SweepCaseGroup:
    group_id: str
    description: str | None
    overrides: dict[str, Any]
    stimulus_kind: str
    sweep_axes: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class SweepConfig:
    template_family: str
    template_variant: str
    config_id: str
    base_case: dict[str, Any]
    sweep_axes: dict[str, tuple[Any, ...]]
    case_groups: tuple[SweepCaseGroup, ...]
    outputs: SweepOutputsSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepConfig":
        payload = require_mapping(payload, name="config")
        template_family = require_literal(
            payload.get("template_family", "single_compartment_channel"),
            name="template_family",
            allowed=("single_compartment_channel",),
        )
        template_variant = require_literal(
            payload.get("template_variant", "hh_fixed_ion"),
            name="template_variant",
            allowed=("hh_fixed_ion",),
        )
        config_id = require_str(payload.get("config_id"), name="config_id")

        base_case_payload = dict(require_submapping(payload, "base_case"))
        normalized_base_case = _normalize_case_payload(
            base_case_payload,
            template_family=template_family,
            template_variant=template_variant,
            case_id=f"{config_id}__BASE",
        )
        normalized_base_case.pop("case_id", None)

        top_level_axes = _parse_axes(
            require_mapping(payload.get("sweep_axes", {}), name="sweep_axes"),
            name="sweep_axes",
            allowed_paths=_ALL_SWEEP_PATHS,
        )

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
            overrides_raw = require_mapping(group_data.get("overrides", {}), name=f"case_groups[{index}].overrides")
            merged_payload = _deep_merge_dicts(normalized_base_case, dict(overrides_raw))
            normalized_group_payload = _normalize_case_payload(
                merged_payload,
                template_family=template_family,
                template_variant=template_variant,
                case_id=f"{group_id}__BASE",
            )
            normalized_group_payload.pop("case_id", None)
            overrides = _dict_diff(normalized_base_case, normalized_group_payload)
            stimulus_kind = str(normalized_group_payload["stimulus"]["kind"])

            allowed_paths = _allowed_sweep_paths(stimulus_kind)
            for path in top_level_axes:
                if path not in allowed_paths:
                    raise ValueError(
                        f"Top-level sweep path {path!r} is incompatible with group {group_id!r} "
                        f"and stimulus kind {stimulus_kind!r}."
                    )
            local_axes = _parse_axes(
                require_mapping(group_data.get("sweep_axes", {}), name=f"case_groups[{index}].sweep_axes"),
                name=f"case_groups[{index}].sweep_axes",
                allowed_paths=allowed_paths,
            )

            case_groups.append(
                SweepCaseGroup(
                    group_id=group_id,
                    description=description,
                    overrides=overrides,
                    stimulus_kind=stimulus_kind,
                    sweep_axes=local_axes,
                )
            )

        outputs_data = require_mapping(payload.get("outputs", {}), name="outputs")
        outputs = SweepOutputsSpec(
            plot=bool(outputs_data.get("plot", False)),
        )

        return cls(
            template_family=template_family,
            template_variant=template_variant,
            config_id=config_id,
            base_case=normalized_base_case,
            sweep_axes=top_level_axes,
            case_groups=tuple(case_groups),
            outputs=outputs,
        )


def load_config(config_path: str | Path) -> SweepConfig:
    path = Path(config_path)
    payload = json.loads(path.read_text())
    return SweepConfig.from_dict(payload)


def expand_cases(config: SweepConfig) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for group in config.case_groups:
        global_axis_items = list(config.sweep_axes.items())
        global_products = itertools.product(*(values for _, values in global_axis_items)) if global_axis_items else [()]
        local_axis_items = list(group.sweep_axes.items())
        case_index = 0
        for global_combination in global_products:
            local_products = itertools.product(*(values for _, values in local_axis_items)) if local_axis_items else [()]
            for local_combination in local_products:
                payload = _deep_merge_dicts(config.base_case, group.overrides)
                payload["case_id"] = f"{group.group_id}__{case_index:03d}"
                payload["group_id"] = group.group_id
                for (path, _values), value in zip(global_axis_items, global_combination):
                    _set_dotted_path(payload, path, value)
                for (path, _values), value in zip(local_axis_items, local_combination):
                    _set_dotted_path(payload, path, value)
                normalized_case = SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
                normalized_payload = _case_to_payload(normalized_case)
                normalized_payload["group_id"] = group.group_id
                expanded.append(normalized_payload)
                case_index += 1
    return expanded


def config_to_payload(config: SweepConfig) -> dict[str, Any]:
    return {
        "template_family": config.template_family,
        "template_variant": config.template_variant,
        "config_id": config.config_id,
        "base_case": config.base_case,
        "sweep_axes": {
            path: list(values)
            for path, values in config.sweep_axes.items()
        },
        "case_groups": [
            {
                "group_id": group.group_id,
                "description": group.description,
                "overrides": group.overrides,
                "sweep_axes": {
                    path: list(values)
                    for path, values in group.sweep_axes.items()
                },
            }
            for group in config.case_groups
        ],
        "outputs": {
            "plot": config.outputs.plot,
        },
    }


def _normalize_case_payload(
    payload: dict[str, Any],
    *,
    template_family: str,
    template_variant: str,
    case_id: str,
) -> dict[str, Any]:
    resolved = dict(payload)
    resolved.setdefault("template_family", template_family)
    resolved.setdefault("template_variant", template_variant)
    resolved.setdefault("case_id", case_id)
    normalized_case = SingleCompartmentChannelHHFixedIonCase.from_dict(resolved)
    return _case_to_payload(normalized_case)


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
        parsed[dotted_path] = tuple(
            _require_scalar(item, name=f"{dotted_path}[{item_index}]")
            for item_index, item in enumerate(values)
        )
    return parsed


def _allowed_sweep_paths(stimulus_kind: str) -> set[str]:
    allowed = set(_COMMON_SWEEP_PATHS)
    if stimulus_kind == "dc":
        allowed.update(_DC_SWEEP_PATHS)
    elif stimulus_kind == "sine":
        allowed.update(_SINE_SWEEP_PATHS)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported stimulus kind {stimulus_kind!r}.")
    return allowed


def _case_to_payload(case: SingleCompartmentChannelHHFixedIonCase) -> dict[str, Any]:
    payload = {
        "template_family": case.template_family,
        "template_variant": case.template_variant,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "mod_dir": case.mod_dir,
        "morphology": {
            "length_um": case.morphology.length_um,
            "radius_um": case.morphology.radius_um,
            "cm_uF_cm2": case.morphology.cm_uF_cm2,
        },
        "simulation": {
            "dt_ms": case.simulation.dt_ms,
            "duration_ms": case.simulation.duration_ms,
            "v_init_mV": case.simulation.v_init_mV,
            "temperature_celsius": case.simulation.temperature_celsius,
        },
        "stimulus": _stimulus_to_payload(case.stimulus),
        "ion": {
            "mode": case.ion.mode,
            "fixed_E_mV": case.ion.fixed_E_mV,
        },
        "channel_overrides": dict(case.channel_overrides),
        "leak": {
            "enabled": case.leak.enabled,
            "g_S_cm2": case.leak.g_S_cm2,
            "e_mV": case.leak.e_mV,
        },
        "compare": {},
    }
    if case.ion.ion_type is not None:
        payload["ion"]["ion_type"] = case.ion.ion_type
    if case.ion.fixed_Ci_mM is not None:
        payload["ion"]["fixed_Ci_mM"] = case.ion.fixed_Ci_mM
    if case.ion.fixed_Co_mM is not None:
        payload["ion"]["fixed_Co_mM"] = case.ion.fixed_Co_mM
    if case.ion.valence is not None:
        payload["ion"]["valence"] = case.ion.valence
    if case.compare.gate_names is not None:
        payload["compare"]["gate_names"] = list(case.compare.gate_names)
    if case.compare.gate_name_map is not None:
        payload["compare"]["gate_name_map"] = dict(case.compare.gate_name_map)
    if not payload["compare"]:
        payload.pop("compare")
    return payload


def _stimulus_to_payload(stimulus: Any) -> dict[str, Any]:
    if stimulus.kind == "dc":
        return {
            "kind": "dc",
            "delay_ms": stimulus.delay_ms,
            "dur_ms": stimulus.dur_ms,
            "amp_nA": stimulus.amp_nA,
        }
    return {
        "kind": "sine",
        "start_ms": stimulus.start_ms,
        "duration_ms": stimulus.duration_ms,
        "amplitude_nA": stimulus.amplitude_nA,
        "frequency_hz": stimulus.frequency_hz,
        "phase_rad": stimulus.phase_rad,
        "offset_nA": stimulus.offset_nA,
    }


def _set_dotted_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    path_parts = dotted_path.split(".")
    target = payload
    for key in path_parts[:-1]:
        next_value = target.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            target[key] = next_value
        target = next_value
    target[path_parts[-1]] = value


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dict_diff(base: dict[str, Any], updated: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, updated_value in updated.items():
        if key not in base:
            diff[key] = updated_value
            continue
        base_value = base[key]
        if isinstance(base_value, dict) and isinstance(updated_value, dict):
            nested = _dict_diff(base_value, updated_value)
            if nested:
                diff[key] = nested
            continue
        if base_value != updated_value:
            diff[key] = updated_value
    return diff


def _require_scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{name} must be a scalar string/number/bool, got {type(value).__name__!s}.")
