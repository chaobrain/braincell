"""Shared stimulus helpers for the multi-compartment cable template."""

from __future__ import annotations

import math

import brainunit as u
import braincell

try:
    from .case_schema import (
        DCStepStimulusSpec,
        PiecewiseStepStimulusSpec,
        SineStimulusSpec,
        StimulusSpec,
    )
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from case_schema import (  # type: ignore
        DCStepStimulusSpec,
        PiecewiseStepStimulusSpec,
        SineStimulusSpec,
        StimulusSpec,
    )


def current_at_ms(stimulus: StimulusSpec, t_ms: float) -> float:
    kind = getattr(stimulus, "kind", None)
    if kind == "dc_step":
        return stimulus.amp_nA if stimulus.delay_ms <= t_ms < (stimulus.delay_ms + stimulus.dur_ms) else 0.0

    if kind == "piecewise_step":
        if t_ms < stimulus.start_ms:
            return 0.0
        local_t = t_ms - stimulus.start_ms
        start = 0.0
        for duration_ms, amplitude_nA in zip(stimulus.durations_ms, stimulus.amplitudes_nA):
            end = start + duration_ms
            if start <= local_t < end:
                return amplitude_nA
            start = end
        return 0.0

    if kind == "sine":
        if not (stimulus.start_ms <= t_ms < (stimulus.start_ms + stimulus.duration_ms)):
            return 0.0
        local_t_sec = (t_ms - stimulus.start_ms) / 1000.0
        angle = (2.0 * math.pi * stimulus.frequency_hz * local_t_sec) + stimulus.phase_rad
        return stimulus.offset_nA + (stimulus.amplitude_nA * math.sin(angle))

    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")


def build_braincell_stimulus(stimulus: StimulusSpec):
    kind = getattr(stimulus, "kind", None)
    if kind == "dc_step":
        return braincell.CurrentClamp.step(
            stimulus.amp_nA * u.nA,
            stimulus.dur_ms * u.ms,
            delay=stimulus.delay_ms * u.ms,
        )
    if kind == "piecewise_step":
        return braincell.CurrentClamp(
            start=stimulus.start_ms * u.ms,
            durations=tuple(duration * u.ms for duration in stimulus.durations_ms),
            amplitudes=tuple(amplitude * u.nA for amplitude in stimulus.amplitudes_nA),
        )
    if kind == "sine":
        return braincell.SineClamp(
            amplitude=stimulus.amplitude_nA * u.nA,
            frequency=stimulus.frequency_hz * u.Hz,
            phase=stimulus.phase_rad,
            offset=stimulus.offset_nA * u.nA,
            start=stimulus.start_ms * u.ms,
            duration=stimulus.duration_ms * u.ms,
        )
    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")
