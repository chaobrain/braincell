"""Case schema for the single-compartment HH + fixed-ion template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .._shared.schema_common import (
        require_literal,
        require_mapping,
        require_number,
        require_str,
        require_submapping,
    )
    from .pair_manifest import get_pair_entry
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
    require_number = _schema_common.require_number
    require_str = _schema_common.require_str
    require_submapping = _schema_common.require_submapping
    _pair_manifest = _load_local_module("pair_manifest", _here / "pair_manifest.py")
    get_pair_entry = _pair_manifest.get_pair_entry


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
class FixedIonSpec:
    ion_type: str | None
    mode: str
    fixed_E_mV: float
    fixed_Ci_mM: float | None = None
    fixed_Co_mM: float | None = None
    valence: int | None = None


@dataclass(frozen=True)
class LeakSpec:
    enabled: bool = False
    g_S_cm2: float = 0.0
    e_mV: float = -65.0


@dataclass(frozen=True)
class CompareSpec:
    gate_names: tuple[str, ...] | None = None
    gate_name_map: dict[str, str] | None = None


@dataclass(frozen=True)
class SingleCompartmentChannelHHFixedIonCase:
    template_family: str
    template_variant: str
    case_id: str
    pair_id: str
    mod_dir: str
    morphology: MorphologySpec
    simulation: SimulationSpec
    stimulus: StimulusSpec
    ion: FixedIonSpec
    channel_overrides: dict[str, Any]
    leak: LeakSpec
    compare: CompareSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SingleCompartmentChannelHHFixedIonCase":
        payload = require_mapping(payload, name="case")

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
        case_id = require_str(payload.get("case_id"), name="case_id")
        pair_id = require_str(payload.get("pair_id"), name="pair_id")
        get_pair_entry(pair_id)
        mod_dir = str(Path(require_str(payload.get("mod_dir"), name="mod_dir")).expanduser())

        morpho_data = require_submapping(payload, "morphology")
        length_um = _require_positive_number(morpho_data.get("length_um"), name="morphology.length_um")
        cm_uF_cm2 = _require_positive_number(morpho_data.get("cm_uF_cm2"), name="morphology.cm_uF_cm2")
        if "radius_um" in morpho_data:
            radius_um = _require_positive_number(morpho_data.get("radius_um"), name="morphology.radius_um")
        elif "diam_um" in morpho_data:
            radius_um = 0.5 * _require_positive_number(morpho_data.get("diam_um"), name="morphology.diam_um")
        else:
            raise KeyError("morphology must provide 'radius_um' or 'diam_um'.")
        morphology = MorphologySpec(length_um=length_um, radius_um=radius_um, cm_uF_cm2=cm_uF_cm2)

        sim_data = require_submapping(payload, "simulation")
        simulation = SimulationSpec(
            dt_ms=_require_positive_number(sim_data.get("dt_ms"), name="simulation.dt_ms"),
            duration_ms=_require_positive_number(sim_data.get("duration_ms"), name="simulation.duration_ms"),
            v_init_mV=require_number(sim_data.get("v_init_mV"), name="simulation.v_init_mV"),
            temperature_celsius=require_number(
                sim_data.get("temperature_celsius"), name="simulation.temperature_celsius"
            ),
        )

        stimulus = _parse_stimulus(
            require_submapping(payload, "stimulus"),
            simulation=simulation,
        )

        ion_data = require_submapping(payload, "ion")
        ion = FixedIonSpec(
            ion_type=(
                require_literal(ion_data.get("ion_type"), name="ion.ion_type", allowed=("na", "k", "ca"))
                if "ion_type" in ion_data else None
            ),
            mode=require_literal(ion_data.get("mode"), name="ion.mode", allowed=("fixed",)),
            fixed_E_mV=require_number(ion_data.get("fixed_E_mV"), name="ion.fixed_E_mV"),
            fixed_Ci_mM=float(ion_data["fixed_Ci_mM"]) if "fixed_Ci_mM" in ion_data else None,
            fixed_Co_mM=float(ion_data["fixed_Co_mM"]) if "fixed_Co_mM" in ion_data else None,
            valence=int(ion_data["valence"]) if "valence" in ion_data else None,
        )

        channel_overrides = dict(require_mapping(payload.get("channel_overrides", {}), name="channel_overrides"))

        leak_data = require_mapping(payload.get("leak", {}), name="leak")
        leak = LeakSpec(
            enabled=bool(leak_data.get("enabled", False)),
            g_S_cm2=float(leak_data.get("g_S_cm2", 0.0)),
            e_mV=float(leak_data.get("e_mV", -65.0)),
        )

        compare_data = require_mapping(payload.get("compare", {}), name="compare")
        gate_names = compare_data.get("gate_names")
        if gate_names is not None:
            gate_names = tuple(require_str(name, name="compare.gate_names[]") for name in gate_names)
        gate_name_map = compare_data.get("gate_name_map")
        if gate_name_map is not None:
            gate_name_map = {
                require_str(src, name="compare.gate_name_map key"): require_str(
                    dst,
                    name=f"compare.gate_name_map[{src!r}]",
                )
                for src, dst in require_mapping(gate_name_map, name="compare.gate_name_map").items()
            }
        compare = CompareSpec(gate_names=gate_names, gate_name_map=gate_name_map)

        return cls(
            template_family=template_family,
            template_variant=template_variant,
            case_id=case_id,
            pair_id=pair_id,
            mod_dir=mod_dir,
            morphology=morphology,
            simulation=simulation,
            stimulus=stimulus,
            ion=ion,
            channel_overrides=channel_overrides,
            leak=leak,
            compare=compare,
        )


def _parse_stimulus(
    payload: Mapping[str, Any],
    *,
    simulation: SimulationSpec,
) -> StimulusSpec:
    kind = require_literal(
        payload.get("kind"),
        name="stimulus.kind",
        allowed=("dc", "sine"),
    )

    if kind == "dc":
        delay_ms = _require_non_negative_number(payload.get("delay_ms"), name="stimulus.delay_ms")
        dur_ms = _require_positive_number(payload.get("dur_ms"), name="stimulus.dur_ms")
        amp_nA = require_number(payload.get("amp_nA"), name="stimulus.amp_nA")
        if delay_ms + dur_ms > simulation.duration_ms:
            raise ValueError("dc stimulus extends beyond simulation.duration_ms.")
        return DCStimulusSpec(
            kind=kind,
            delay_ms=delay_ms,
            dur_ms=dur_ms,
            amp_nA=amp_nA,
        )

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
