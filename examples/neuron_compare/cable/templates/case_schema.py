"""Case schema for the multi-compartment cable voltage-compare template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from ._shared.schema_common import (
        require_literal,
        require_mapping,
        require_number,
        require_str,
        require_submapping,
    )
except ImportError:  # pragma: no cover
    import sys

    _templates_root = Path(__file__).resolve().parent
    if str(_templates_root) not in sys.path:
        sys.path.insert(0, str(_templates_root))
    from _shared.schema_common import (  # type: ignore
        require_literal,
        require_mapping,
        require_number,
        require_str,
        require_submapping,
    )


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return int(value)


def _require_odd_positive_int(value: Any, *, name: str) -> int:
    resolved = _require_positive_int(value, name=name)
    if resolved % 2 == 0:
        raise ValueError(f"{name} must be odd so root soma(0.5) maps to a midpoint, got {resolved!r}.")
    return resolved


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


def _require_number_sequence(value: Any, *, name: str, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__!s}.")
    if len(value) == 0:
        raise ValueError(f"{name} must be non-empty.")
    items = []
    for index, item in enumerate(value):
        item_name = f"{name}[{index}]"
        items.append(_require_positive_number(item, name=item_name) if positive else require_number(item, name=item_name))
    return tuple(items)


@dataclass(frozen=True)
class MorphologySpec:
    kind: str
    path: str


@dataclass(frozen=True)
class SimulationSpec:
    dt_ms: float
    duration_ms: float
    v_init_mV: float


@dataclass(frozen=True)
class CableSpec:
    ra_ohm_cm: float
    cm_uF_cm2: float


@dataclass(frozen=True)
class CVPolicySpec:
    kind: str
    cv_per_branch: int


@dataclass(frozen=True)
class DCStepStimulusSpec:
    kind: str
    target: str
    delay_ms: float
    dur_ms: float
    amp_nA: float


@dataclass(frozen=True)
class PiecewiseStepStimulusSpec:
    kind: str
    target: str
    start_ms: float
    durations_ms: tuple[float, ...]
    amplitudes_nA: tuple[float, ...]


@dataclass(frozen=True)
class SineStimulusSpec:
    kind: str
    target: str
    start_ms: float
    duration_ms: float
    amplitude_nA: float
    frequency_hz: float
    phase_rad: float
    offset_nA: float


StimulusSpec = DCStepStimulusSpec | PiecewiseStepStimulusSpec | SineStimulusSpec


@dataclass(frozen=True)
class MultiCompartmentCableCase:
    template_family: str
    case_id: str
    morphology: MorphologySpec
    simulation: SimulationSpec
    cable: CableSpec
    cv_policy: CVPolicySpec
    stimulus: StimulusSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiCompartmentCableCase":
        payload = require_mapping(payload, name="case")

        template_family = require_literal(
            payload.get("template_family", "multi_compartment_cable"),
            name="template_family",
            allowed=("multi_compartment_cable",),
        )
        case_id = require_str(payload.get("case_id"), name="case_id")

        morphology_data = payload.get("morphology")
        if morphology_data is None and payload.get("swc") is not None:
            legacy_swc = require_submapping(payload, "swc")
            morphology_data = {
                "kind": "swc",
                "path": require_str(legacy_swc.get("path"), name="swc.path"),
            }
        morphology_data = require_mapping(morphology_data, name="morphology")
        morphology = MorphologySpec(
            kind=require_literal(
                morphology_data.get("kind"),
                name="morphology.kind",
                allowed=("swc", "asc", "neuroml2"),
            ),
            path=require_str(morphology_data.get("path"), name="morphology.path"),
        )

        simulation_data = require_submapping(payload, "simulation")
        simulation = SimulationSpec(
            dt_ms=_require_positive_number(simulation_data.get("dt_ms"), name="simulation.dt_ms"),
            duration_ms=_require_positive_number(
                simulation_data.get("duration_ms"),
                name="simulation.duration_ms",
            ),
            v_init_mV=require_number(simulation_data.get("v_init_mV"), name="simulation.v_init_mV"),
        )

        cable_data = require_submapping(payload, "cable")
        cable = CableSpec(
            ra_ohm_cm=_require_positive_number(cable_data.get("ra_ohm_cm"), name="cable.ra_ohm_cm"),
            cm_uF_cm2=_require_positive_number(cable_data.get("cm_uF_cm2"), name="cable.cm_uF_cm2"),
        )

        cv_policy_data = require_submapping(payload, "cv_policy")
        cv_policy = CVPolicySpec(
            kind=require_literal(
                cv_policy_data.get("kind"),
                name="cv_policy.kind",
                allowed=("CVPerBranch",),
            ),
            cv_per_branch=_require_odd_positive_int(
                cv_policy_data.get("cv_per_branch"),
                name="cv_policy.cv_per_branch",
            ),
        )

        stimulus = _parse_stimulus(
            require_submapping(payload, "stimulus"),
            simulation=simulation,
        )

        return cls(
            template_family=template_family,
            case_id=case_id,
            morphology=morphology,
            simulation=simulation,
            cable=cable,
            cv_policy=cv_policy,
            stimulus=stimulus,
        )


def _parse_stimulus(
    payload: Mapping[str, Any],
    *,
    simulation: SimulationSpec,
) -> StimulusSpec:
    kind = require_literal(
        payload.get("kind"),
        name="stimulus.kind",
        allowed=("dc_step", "piecewise_step", "sine"),
    )
    target = require_literal(
        payload.get("target", "root_soma_midpoint"),
        name="stimulus.target",
        allowed=("root_soma_midpoint",),
    )

    if kind == "dc_step":
        delay_ms = _require_non_negative_number(payload.get("delay_ms"), name="stimulus.delay_ms")
        dur_ms = _require_positive_number(payload.get("dur_ms"), name="stimulus.dur_ms")
        amp_nA = require_number(payload.get("amp_nA"), name="stimulus.amp_nA")
        if delay_ms + dur_ms > simulation.duration_ms:
            raise ValueError("dc_step extends beyond simulation.duration_ms.")
        return DCStepStimulusSpec(
            kind=kind,
            target=target,
            delay_ms=delay_ms,
            dur_ms=dur_ms,
            amp_nA=amp_nA,
        )

    if kind == "piecewise_step":
        start_ms = _require_non_negative_number(payload.get("start_ms"), name="stimulus.start_ms")
        durations_ms = _require_number_sequence(
            payload.get("durations_ms"),
            name="stimulus.durations_ms",
            positive=True,
        )
        amplitudes_nA = _require_number_sequence(
            payload.get("amplitudes_nA"),
            name="stimulus.amplitudes_nA",
            positive=False,
        )
        if len(durations_ms) != len(amplitudes_nA):
            raise ValueError(
                "stimulus.durations_ms and stimulus.amplitudes_nA must have the same length."
            )
        if start_ms + sum(durations_ms) > simulation.duration_ms:
            raise ValueError("piecewise_step extends beyond simulation.duration_ms.")
        return PiecewiseStepStimulusSpec(
            kind=kind,
            target=target,
            start_ms=start_ms,
            durations_ms=durations_ms,
            amplitudes_nA=amplitudes_nA,
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
        target=target,
        start_ms=start_ms,
        duration_ms=duration_ms,
        amplitude_nA=amplitude_nA,
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        offset_nA=offset_nA,
    )
