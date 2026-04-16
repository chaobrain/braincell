"""Stimulus helpers for the HH + fixed-ion single-compartment template."""

from __future__ import annotations

import math

import braincell
import brainunit as u

try:
    from .case_schema import DCStimulusSpec, SineStimulusSpec, StimulusSpec
except ImportError:  # pragma: no cover
    import importlib.util
    import sys
    from pathlib import Path

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
    _case_schema = _load_local_module("case_schema", _here / "case_schema.py")
    DCStimulusSpec = _case_schema.DCStimulusSpec
    SineStimulusSpec = _case_schema.SineStimulusSpec
    StimulusSpec = _case_schema.StimulusSpec


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
    kind = getattr(stimulus, "kind", None)
    if kind == "dc":
        return braincell.CurrentClamp.step(
            stimulus.amp_nA * u.nA,
            stimulus.dur_ms * u.ms,
            delay=stimulus.delay_ms * u.ms,
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
