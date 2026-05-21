"""Mapping schema for single-channel no-concentration comparisons."""



from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .schema_common import require_mapping, require_str
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from schema_common import require_mapping, require_str  # type: ignore


@dataclass(frozen=True)
class CurrentSourceSpec:
    ion_name: str | None
    neuron_current_var: str


@dataclass(frozen=True)
class SideParameterSpec:
    braincell: str
    neuron: str


@dataclass(frozen=True)
class GateMapSpec:
    canonical_name: str
    braincell: str
    neuron: str


@dataclass(frozen=True)
class NeuronChannelSpec:
    mechanism_name: str
    gate_names: tuple[str, ...]


@dataclass(frozen=True)
class BraincellChannelSpec:
    class_name: str
    gate_names: tuple[str, ...]


@dataclass(frozen=True)
class MappingSpec:
    current_source: CurrentSourceSpec
    neuron: NeuronChannelSpec
    braincell: BraincellChannelSpec
    gate_map: tuple[GateMapSpec, ...]
    parameter_map: dict[str, SideParameterSpec]

    @property
    def current_kind(self) -> str | None:
        return self.current_source.ion_name

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MappingSpec":
        payload = require_mapping(payload, name="mapping")
        current_source = _parse_current_source(payload)

        impl_name_data = require_submapping(payload, "impl_name")
        neuron_impl_name, braincell_impl_name = _resolve_side_names(
            impl_name_data,
            name="mapping.impl_name",
        )

        gate_name_data = require_submapping(payload, "gate_names")
        canonical_gate_names, neuron_gate_names, braincell_gate_names = _resolve_gate_names(
            gate_name_data,
            name="mapping.gate_names",
        )

        params_raw = _resolve_parameter_payload(payload)
        parameter_map: dict[str, SideParameterSpec] = {}
        for ir_key_raw, side_payload in sorted(params_raw.items()):
            ir_key = require_str(ir_key_raw, name="mapping.channel_params key")
            parameter_map[ir_key] = _parse_parameter_spec(
                side_payload,
                name=f"mapping.channel_params.{ir_key}",
            )

        return cls(
            current_source=current_source,
            neuron=NeuronChannelSpec(
                mechanism_name=neuron_impl_name,
                gate_names=neuron_gate_names,
            ),
            braincell=BraincellChannelSpec(
                class_name=braincell_impl_name,
                gate_names=braincell_gate_names,
            ),
            gate_map=tuple(
                GateMapSpec(
                    canonical_name=canonical_name,
                    neuron=neuron_gate,
                    braincell=braincell_gate,
                )
                for canonical_name, neuron_gate, braincell_gate in zip(
                    canonical_gate_names,
                    neuron_gate_names,
                    braincell_gate_names,
                )
            ),
            parameter_map=parameter_map,
        )


def normalize_mapping_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    mapping_spec = MappingSpec.from_mapping(payload)

    if mapping_spec.neuron.mechanism_name == mapping_spec.braincell.class_name:
        impl_name_payload = {"common": mapping_spec.neuron.mechanism_name}
    else:
        impl_name_payload = {
            "neuron": mapping_spec.neuron.mechanism_name,
            "braincell": mapping_spec.braincell.class_name,
        }

    if mapping_spec.neuron.gate_names == mapping_spec.braincell.gate_names:
        gate_names_payload = {"common": list(mapping_spec.neuron.gate_names)}
    else:
        gate_names_payload = {
            "canonical": [item.canonical_name for item in mapping_spec.gate_map],
            "neuron": list(mapping_spec.neuron.gate_names),
            "braincell": list(mapping_spec.braincell.gate_names),
        }

    params_payload: dict[str, dict[str, Any]] = {}
    for ir_key, side_names in sorted(mapping_spec.parameter_map.items()):
        if side_names.neuron == side_names.braincell:
            params_payload[ir_key] = {"common": side_names.neuron}
        else:
            params_payload[ir_key] = {
                "neuron": side_names.neuron,
                "braincell": side_names.braincell,
            }

    return {
        "current": mapping_spec.current_source.neuron_current_var,
        "impl_name": impl_name_payload,
        "gate_names": gate_names_payload,
        "channel_params": params_payload,
    }


def require_submapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload:
        raise KeyError(f"Missing required section {key!r}.")
    return require_mapping(payload[key], name=key)


def _parse_current_source(payload: Mapping[str, Any]) -> CurrentSourceSpec:
    if "current_source" in payload or "current_kind" in payload:
        raise ValueError(
            "Legacy mapping current schema is no longer supported; "
            "use 'mapping.current' with a NEURON current variable name."
        )
    neuron_current_var = require_str(payload.get("current"), name="mapping.current")
    return CurrentSourceSpec(
        ion_name=_infer_ion_name_from_current(neuron_current_var),
        neuron_current_var=neuron_current_var,
    )


def _resolve_parameter_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    has_new = "channel_params" in payload
    has_old = "params" in payload
    if has_new and has_old:
        raise ValueError("mapping cannot define both 'channel_params' and legacy 'params'.")
    key = "channel_params" if has_new else "params"
    return require_mapping(payload.get(key, {}), name=f"mapping.{key}")


def _parse_parameter_spec(payload: Any, *, name: str) -> SideParameterSpec:
    payload = require_mapping(payload, name=name)
    has_common = "common" in payload
    has_neuron = "neuron" in payload
    has_braincell = "braincell" in payload

    if has_common and (has_neuron or has_braincell):
        raise ValueError(f"{name} cannot mix 'common' with side-specific names.")
    if has_common:
        common_name = require_str(payload.get("common"), name=f"{name}.common")
        return SideParameterSpec(
            neuron=common_name,
            braincell=common_name,
        )
    if not (has_neuron and has_braincell):
        raise ValueError(f"{name} must define either 'common' or both 'neuron' and 'braincell'.")

    neuron_name = _resolve_parameter_name(payload.get("neuron"), name=f"{name}.neuron")
    braincell_name = _resolve_braincell_parameter_name(payload.get("braincell"), name=f"{name}.braincell")
    return SideParameterSpec(
        neuron=neuron_name,
        braincell=braincell_name,
    )


def _resolve_side_names(payload: Mapping[str, Any], *, name: str) -> tuple[str, str]:
    payload = require_mapping(payload, name=name)
    has_common = "common" in payload
    has_neuron = "neuron" in payload
    has_braincell = "braincell" in payload

    if has_common and (has_neuron or has_braincell):
        raise ValueError(f"{name} cannot mix 'common' with side-specific names.")
    if has_common:
        common_name = require_str(payload.get("common"), name=f"{name}.common")
        return common_name, common_name
    if not (has_neuron and has_braincell):
        raise ValueError(f"{name} must define either 'common' or both 'neuron' and 'braincell'.")
    return (
        require_str(payload.get("neuron"), name=f"{name}.neuron"),
        require_str(payload.get("braincell"), name=f"{name}.braincell"),
    )


def _resolve_gate_names(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    payload = require_mapping(payload, name=name)
    if "common" in payload and any(key in payload for key in ("canonical", "neuron", "braincell")):
        raise ValueError(f"{name} cannot mix 'common' with canonical/side-specific gate names.")

    if "common" in payload:
        common_gate_names = _require_gate_list(payload.get("common"), name=f"{name}.common")
        return common_gate_names, common_gate_names, common_gate_names

    canonical_gate_names = _require_gate_list(payload.get("canonical"), name=f"{name}.canonical")
    neuron_gate_names = _require_gate_list(payload.get("neuron"), name=f"{name}.neuron")
    braincell_gate_names = _require_gate_list(payload.get("braincell"), name=f"{name}.braincell")
    if not (
        len(canonical_gate_names) == len(neuron_gate_names) == len(braincell_gate_names)
    ):
        raise ValueError(f"{name}.canonical/neuron/braincell must have the same length.")
    return canonical_gate_names, neuron_gate_names, braincell_gate_names


def _resolve_parameter_name(value: Any, *, name: str) -> str:
    if isinstance(value, str):
        return require_str(value, name=name)
    raise ValueError(f"{name} must be a string parameter name.")


def _resolve_braincell_parameter_name(value: Any, *, name: str) -> str:
    return _resolve_parameter_name(value, name=name)


def _require_gate_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list of strings.")
    return tuple(require_str(item, name=f"{name}[]") for item in value)


def _default_neuron_current_var(ion_name: str) -> str:
    return {
        "na": "ina",
        "k": "ik",
        "ca": "ica",
    }[ion_name]


def _infer_ion_name_from_current(current_var: str) -> str | None:
    return {
        "ina": "na",
        "ik": "k",
        "ica": "ca",
        "ical": "cal",
    }.get(current_var)


__all__ = [
    "BraincellChannelSpec",
    "CurrentSourceSpec",
    "GateMapSpec",
    "MappingSpec",
    "NeuronChannelSpec",
    "SideParameterSpec",
    "normalize_mapping_payload",
]
