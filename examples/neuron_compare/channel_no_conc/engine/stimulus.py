"""Stimulus helpers shared by channel_no_conc."""



import math

import brainunit as u

try:
    from .experiment_schema import StimulusSpec
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from experiment_schema import StimulusSpec  # type: ignore


def current_at_ms(stimulus: StimulusSpec, t_ms: float) -> float:
    kind = getattr(stimulus, "kind", None)
    if kind == "dc":
        return stimulus.amp_nA if stimulus.delay_ms <= t_ms < (stimulus.delay_ms + stimulus.dur_ms) else 0.0

    if kind == "sine":
        if not (stimulus.start_ms <= t_ms < (stimulus.start_ms + stimulus.duration_ms)):
            return 0.0
        local_t_sec = (t_ms - stimulus.start_ms) / 1000.0
        angle = (2.0 * math.pi * stimulus.frequency_hz * local_t_sec) + stimulus.phase_rad
        return stimulus.offset_nA + (stimulus.amplitude_nA * math.sin(angle))

    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")


def build_braincell_stimulus(stimulus: StimulusSpec):
    import braincell

    kind = getattr(stimulus, "kind", None)
    if kind == "dc":
        return braincell.CurrentClamp(
            delay=stimulus.delay_ms * u.ms,
            durations=stimulus.dur_ms * u.ms,
            amplitudes=stimulus.amp_nA * u.nA,
        )

    if kind == "sine":
        return braincell.SineClamp(
            amplitude=stimulus.amplitude_nA * u.nA,
            frequency=stimulus.frequency_hz * u.Hz,
            phase=stimulus.phase_rad,
            offset=stimulus.offset_nA * u.nA,
            delay=stimulus.start_ms * u.ms,
            duration=stimulus.duration_ms * u.ms,
        )

    raise TypeError(f"Unsupported stimulus type {type(stimulus).__name__!s}.")
