"""Schema and sweep synthesis for single-channel no-concentration comparisons."""



from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .mapping_schema import MappingSpec, normalize_mapping_payload
    from .schema_common import require_mapping, require_number, require_str, require_submapping
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from mapping_schema import MappingSpec, normalize_mapping_payload  # type: ignore
    from schema_common import require_mapping, require_number, require_str, require_submapping  # type: ignore


_DEFAULT_ION_REVERSAL_MV = {
    "na": 50.0,
    "k": -80.0,
    "ca": 120.0,
}


def _require_non_negative_number(value: Any, *, name: str) -> float:
    resolved = require_number(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be >= 0, got {resolved!r}.")
    return resolved


def _require_positive_number(value: Any, *, name: str) -> float:
    resolved = require_number(value, name=name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be > 0, got {resolved!r}.")
    return resolved


def _require_integer_dt_steps(*, duration_ms: float, dt_ms: float) -> None:
    step_count = duration_ms / dt_ms
    rounded_steps = round(step_count)
    if not math.isclose(step_count, rounded_steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "simulation.duration_ms must be an integer multiple of simulation.dt_ms."
        )


@dataclass(frozen=True)
class MorphologySpec:
    length_um: float
    radius_um: float
    cm_uF_cm2: float


@dataclass(frozen=True)
class SimulationSpec:
    dt_ms: float
    duration_ms: float
    v_init_mV: float
    temperature_celsius: float


@dataclass(frozen=True)
class DCStimulusSpec:
    kind: str
    delay_ms: float
    dur_ms: float
    amp_nA: float


@dataclass(frozen=True)
class SineStimulusSpec:
    kind: str
    start_ms: float
    duration_ms: float
    amplitude_nA: float
    frequency_hz: float
    phase_rad: float
    offset_nA: float


StimulusSpec = DCStimulusSpec | SineStimulusSpec


@dataclass(frozen=True)
class IonStateSpec:
    E_mV: float

    @property
    def e_rev_mV(self) -> float:
        return self.E_mV


@dataclass(frozen=True)
class LeakSpec:
    enabled: bool = False
    g_S_cm2: float = 0.0
    e_mV: float = -65.0


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
    mod_dir: str
    defaults: dict[str, Any]
    mapping_payload: dict[str, Any]
    mapping_spec: MappingSpec
    template_paths: tuple[Path, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, config_path: Path) -> "ModelConfig":
        payload = require_mapping(payload, name="config")
        _reject_legacy_config_shape(payload)

        identity = require_submapping(payload, "identity")
        mapping_payload = normalize_mapping_payload(require_submapping(payload, "mapping"))
        mapping_spec = MappingSpec.from_mapping(mapping_payload)
        return cls(
            config_name=config_path.stem,
            config_path=config_path,
            meta=_normalize_meta_payload(payload.get("meta"), name="meta"),
            mod_dir=str(Path(require_str(identity.get("mod_dir"), name="identity.mod_dir")).expanduser()),
            defaults=_normalize_defaults_payload(payload.get("defaults"), name="defaults"),
            mapping_payload=mapping_payload,
            mapping_spec=mapping_spec,
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
    mod_dir: str
    mapping_payload: dict[str, Any]
    mapping_spec: MappingSpec
    base_case: dict[str, Any]
    group: SweepCaseGroup
    outputs: SweepOutputsSpec


@dataclass(frozen=True)
class ChannelNoConcCase:
    case_id: str
    config_name: str
    template_name: str
    run_id: str
    mod_dir: str
    mapping_payload: dict[str, Any]
    mapping_spec: MappingSpec
    morphology: MorphologySpec
    simulation: SimulationSpec
    stimulus: StimulusSpec
    ion_state: IonStateSpec | None
    channel_params: dict[str, Any]
    leak: LeakSpec

    @property
    def channel_state(self) -> IonStateSpec | None:
        return self.ion_state

    @property
    def channel_overrides(self) -> dict[str, Any]:
        return self.channel_params

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChannelNoConcCase":
        payload = require_mapping(payload, name="case")
        if "pair_id" in payload and "mapping" not in payload:
            raise ValueError(
                "Legacy channel_no_conc case schema is no longer supported; "
                "use identity/mapping-based case payloads."
            )

        case_id = require_str(payload.get("case_id"), name="case_id")
        config_name = require_str(payload.get("config_name", "config"), name="config_name")
        template_name = require_str(payload.get("template_name", "template"), name="template_name")
        run_id = require_str(payload.get("run_id", f"{config_name}__{template_name}"), name="run_id")

        identity_data = require_submapping(payload, "identity")
        mod_dir = str(Path(require_str(identity_data.get("mod_dir"), name="identity.mod_dir")).expanduser())

        mapping_payload = normalize_mapping_payload(require_submapping(payload, "mapping"))
        mapping_spec = MappingSpec.from_mapping(mapping_payload)

        morphology_data = require_submapping(payload, "morphology")
        length_um = _require_positive_number(morphology_data.get("length_um"), name="morphology.length_um")
        cm_uF_cm2 = _require_positive_number(morphology_data.get("cm_uF_cm2"), name="morphology.cm_uF_cm2")
        if "radius_um" in morphology_data:
            radius_um = _require_positive_number(morphology_data.get("radius_um"), name="morphology.radius_um")
        elif "diam_um" in morphology_data:
            radius_um = 0.5 * _require_positive_number(morphology_data.get("diam_um"), name="morphology.diam_um")
        else:
            raise KeyError("morphology must provide 'radius_um' or 'diam_um'.")
        morphology = MorphologySpec(length_um=length_um, radius_um=radius_um, cm_uF_cm2=cm_uF_cm2)

        simulation_data = require_submapping(payload, "simulation")
        simulation = SimulationSpec(
            dt_ms=_require_positive_number(simulation_data.get("dt_ms"), name="simulation.dt_ms"),
            duration_ms=_require_positive_number(simulation_data.get("duration_ms"), name="simulation.duration_ms"),
            v_init_mV=require_number(simulation_data.get("v_init_mV"), name="simulation.v_init_mV"),
            temperature_celsius=require_number(
                simulation_data.get("temperature_celsius"),
                name="simulation.temperature_celsius",
            ),
        )
        _require_integer_dt_steps(duration_ms=simulation.duration_ms, dt_ms=simulation.dt_ms)

        stimulus = _parse_stimulus(require_submapping(payload, "stimulus"), simulation=simulation)
        ion_state = _parse_ion_state(payload, mapping_spec=mapping_spec)

        channel_params = _parse_channel_params(payload)
        unsupported_param_keys = sorted(set(channel_params) - set(mapping_spec.parameter_map))
        if unsupported_param_keys:
            raise ValueError(
                "channel_params contains keys not declared in mapping.channel_params: "
                f"{unsupported_param_keys!r}."
            )

        leak_data = require_mapping(payload.get("leak", {}), name="leak")
        leak = LeakSpec(
            enabled=bool(leak_data.get("enabled", False)),
            g_S_cm2=float(leak_data.get("g_S_cm2", 0.0)),
            e_mV=float(leak_data.get("e_mV", -65.0)),
        )

        return cls(
            case_id=case_id,
            config_name=config_name,
            template_name=template_name,
            run_id=run_id,
            mod_dir=mod_dir,
            mapping_payload=mapping_payload,
            mapping_spec=mapping_spec,
            morphology=morphology,
            simulation=simulation,
            stimulus=stimulus,
            ion_state=ion_state,
            channel_params=channel_params,
            leak=leak,
        )


def load_model_config(config_path: str | Path) -> ModelConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    payload = json.loads(resolved_config_path.read_text())
    return ModelConfig.from_dict(payload, config_path=resolved_config_path)


def load_scan_template(template_path: str | Path) -> ScanTemplate:
    resolved_template_path = Path(template_path).expanduser().resolve()
    payload = json.loads(resolved_template_path.read_text())
    return ScanTemplate.from_dict(payload, template_path=resolved_template_path)


def load_case(case_path: str | Path) -> ChannelNoConcCase:
    payload = json.loads(Path(case_path).read_text())
    return ChannelNoConcCase.from_dict(payload)


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
    merged_base = _deep_merge_dicts(model_config.defaults, dict(scan_template.base))
    normalized_base_case = _normalize_case_payload(
        {
            **merged_base,
            "identity": {"mod_dir": model_config.mod_dir},
            "mapping": model_config.mapping_payload,
        },
        case_id=f"{run_id}__BASE",
        config_name=model_config.config_name,
        template_name=scan_template.template_name,
        run_id=run_id,
    )
    normalized_base_case.pop("case_id", None)

    allowed_paths = _allowed_sweep_paths(
        mapping_spec=model_config.mapping_spec,
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
        mod_dir=model_config.mod_dir,
        mapping_payload=model_config.mapping_payload,
        mapping_spec=model_config.mapping_spec,
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
        normalized_case = ChannelNoConcCase.from_dict(payload)
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
        "identity": {
            "mod_dir": config.mod_dir,
        },
        "mapping": dict(config.mapping_payload),
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


def case_to_payload(case: ChannelNoConcCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "config_name": case.config_name,
        "template_name": case.template_name,
        "run_id": case.run_id,
        "identity": {
            "mod_dir": case.mod_dir,
        },
        "mapping": dict(case.mapping_payload),
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
        **(
            {
                "ion_state": {
                    "E_mV": case.ion_state.E_mV,
                },
            }
            if case.ion_state is not None
            else {}
        ),
        "channel_params": dict(case.channel_params),
        "leak": {
            "enabled": case.leak.enabled,
            "g_S_cm2": case.leak.g_S_cm2,
            "e_mV": case.leak.e_mV,
        },
    }


def _normalize_case_payload(
    payload: dict[str, Any],
    *,
    case_id: str,
    config_name: str,
    template_name: str,
    run_id: str,
) -> dict[str, Any]:
    resolved = dict(payload)
    resolved.setdefault("case_id", case_id)
    resolved.setdefault("config_name", config_name)
    resolved.setdefault("template_name", template_name)
    resolved.setdefault("run_id", run_id)
    normalized_case = ChannelNoConcCase.from_dict(resolved)
    return case_to_payload(normalized_case)


def _resolve_template_reference(*, template_path: str | Path, config_dir: Path) -> Path:
    raw_path = Path(template_path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (config_dir / raw_path).resolve()


def _allowed_sweep_paths(*, mapping_spec: MappingSpec, stimulus_kind: str) -> set[str]:
    allowed = {
        "simulation.dt_ms",
        "simulation.v_init_mV",
        "simulation.temperature_celsius",
        *(("ion_state.E_mV",) if mapping_spec.current_source.ion_name is not None else ()),
        "leak.enabled",
        "leak.g_S_cm2",
        "leak.e_mV",
        *(f"channel_params.{name}" for name in mapping_spec.parameter_map),
    }
    if stimulus_kind == "dc":
        allowed.update({
            "stimulus.amp_nA",
            "stimulus.delay_ms",
            "stimulus.dur_ms",
        })
        return allowed
    if stimulus_kind == "sine":
        allowed.update({
            "stimulus.start_ms",
            "stimulus.duration_ms",
            "stimulus.amplitude_nA",
            "stimulus.frequency_hz",
            "stimulus.phase_rad",
            "stimulus.offset_nA",
        })
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
        dotted_path = _canonical_sweep_path(require_str(path, name=f"{name} key"))
        if dotted_path not in allowed_paths:
            raise ValueError(f"Unsupported sweep path {dotted_path!r}.")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"{name}[{dotted_path!r}] must be a non-empty list or tuple.")
        parsed[dotted_path] = tuple(
            _require_scalar(item, name=f"{dotted_path}[{item_index}]")
            for item_index, item in enumerate(values)
        )
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
        parsed[dotted_path] = tuple(
            _require_scalar(item, name=f"{dotted_path}[{item_index}]")
            for item_index, item in enumerate(values)
        )
    return parsed


def _parse_stimulus(payload: Mapping[str, Any], *, simulation: SimulationSpec) -> StimulusSpec:
    kind = require_str(payload.get("kind"), name="stimulus.kind")
    if kind == "dc":
        delay_ms = _require_non_negative_number(payload.get("delay_ms"), name="stimulus.delay_ms")
        dur_ms = _require_positive_number(payload.get("dur_ms"), name="stimulus.dur_ms")
        amp_nA = require_number(payload.get("amp_nA"), name="stimulus.amp_nA")
        if delay_ms + dur_ms > simulation.duration_ms:
            raise ValueError("dc stimulus extends beyond simulation.duration_ms.")
        return DCStimulusSpec(kind=kind, delay_ms=delay_ms, dur_ms=dur_ms, amp_nA=amp_nA)

    if kind == "sine":
        start_ms = _require_non_negative_number(payload.get("start_ms"), name="stimulus.start_ms")
        duration_ms = _require_positive_number(payload.get("duration_ms"), name="stimulus.duration_ms")
        amplitude_nA = require_number(payload.get("amplitude_nA"), name="stimulus.amplitude_nA")
        frequency_hz = _require_positive_number(payload.get("frequency_hz"), name="stimulus.frequency_hz")
        phase_rad = require_number(payload.get("phase_rad", 0.0), name="stimulus.phase_rad")
        offset_nA = require_number(payload.get("offset_nA", 0.0), name="stimulus.offset_nA")
        if start_ms + duration_ms > simulation.duration_ms:
            raise ValueError("sine stimulus extends beyond simulation.duration_ms.")
        return SineStimulusSpec(
            kind=kind,
            start_ms=start_ms,
            duration_ms=duration_ms,
            amplitude_nA=amplitude_nA,
            frequency_hz=frequency_hz,
            phase_rad=phase_rad,
            offset_nA=offset_nA,
        )

    raise ValueError(f"Unsupported stimulus kind {kind!r}.")


def _parse_ion_state(payload: Mapping[str, Any], *, mapping_spec: MappingSpec) -> IonStateSpec | None:
    state_payload: Mapping[str, Any] | None = None
    if "ion_state" in payload:
        state_payload = require_submapping(payload, "ion_state")
    elif "channel_state" in payload:
        state_payload = require_submapping(payload, "channel_state")

    if mapping_spec.current_source.ion_name is None:
        if state_payload is not None:
            raise ValueError("ion_state is only valid when mapping.current resolves to ik/ina/ica.")
        return None

    if state_payload is None:
        ion_name = mapping_spec.current_source.ion_name
        if ion_name is None:
            raise ValueError("mapping.current must resolve to ik/ina/ica for ion_state defaults.")
        return IonStateSpec(E_mV=_default_ion_reversal_mV(ion_name))
    if "E_mV" in state_payload:
        return IonStateSpec(
            E_mV=require_number(state_payload.get("E_mV"), name="ion_state.E_mV"),
        )
    return IonStateSpec(
        E_mV=require_number(state_payload.get("e_rev_mV"), name="channel_state.e_rev_mV"),
    )


def _default_ion_reversal_mV(ion_name: str) -> float:
    try:
        return _DEFAULT_ION_REVERSAL_MV[ion_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported ion_name for ion_state defaults: {ion_name!r}.") from exc


def _parse_channel_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "channel_params" in payload and "channel_overrides" in payload:
        raise ValueError("Case payload cannot define both 'channel_params' and legacy 'channel_overrides'.")
    key = "channel_params" if "channel_params" in payload else "channel_overrides"
    return dict(require_mapping(payload.get(key, {}), name=key))


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


def _canonical_sweep_path(dotted_path: str) -> str:
    if dotted_path == "channel_state.e_rev_mV":
        return "ion_state.E_mV"
    if dotted_path.startswith("channel_overrides."):
        return "channel_params." + dotted_path.removeprefix("channel_overrides.")
    return dotted_path


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _strip_case_runtime_fields(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    for key in ("case_id", "group_id", "config_name", "template_name", "run_id", "identity", "mapping"):
        stripped.pop(key, None)
    return stripped


def _require_scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{name} must be a scalar string/number/bool, got {type(value).__name__!s}.")


def _normalize_meta_payload(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(require_mapping(value, name=name))


def _normalize_defaults_payload(value: Any, *, name: str) -> dict[str, Any]:
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
    legacy_keys = {"config_id", "base_case", "case_groups", "sweep_axes", "preset_id", "pair_id"}
    if legacy_keys & set(payload):
        raise ValueError(
            "Legacy channel_no_conc config schema is no longer supported; "
            "use a model config with identity/mapping/templates."
        )


def _reject_legacy_template_shape(payload: Mapping[str, Any]) -> None:
    legacy_keys = {"config_id", "base_case", "case_groups", "sweep_axes", "preset_id", "pair_id"}
    if legacy_keys & set(payload):
        raise ValueError(
            "Legacy channel_no_conc sweep template schema is no longer supported; "
            "use a base/group template JSON."
        )


__all__ = [
    "ChannelNoConcCase",
    "IonStateSpec",
    "ModelConfig",
    "ScanTemplate",
    "SweepCaseGroup",
    "SweepConfig",
    "build_sweep_config",
    "case_to_payload",
    "config_to_payload",
    "expand_cases",
    "load_case",
    "load_model_config",
    "load_scan_template",
    "load_sweep_config",
]
