# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Experiment protocols and lazy model discovery for whole-cell comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
import math


@dataclass(frozen=True)
class Protocol:
    """Fixed-step soma current-clamp protocol; units are explicit in field names.

    All times are in ms, current in nA, voltage in mV and temperature in Celsius.
    ``precision`` selects BrainCell's floating-point precision (32 or 64 bits).
    """

    dt_ms: float = 0.1
    duration_ms: float = 50.0
    delay_ms: float = 10.0
    stim_dur_ms: float = 30.0
    amp_nA: float = 0.05
    temperature_celsius: float = 36.0
    v_init_mV: float = -65.0
    precision: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.precision not in (32, 64):
            raise ValueError("precision must be 32 or 64.")
        if self.dt_ms <= 0 or self.duration_ms <= 0:
            raise ValueError("dt_ms and duration_ms must be positive.")
        if self.delay_ms < 0 or self.stim_dur_ms < 0:
            raise ValueError("delay_ms and stim_dur_ms must be nonnegative.")
        steps = self.duration_ms / self.dt_ms
        if round(steps) < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
            raise ValueError("duration_ms must be an integer multiple of dt_ms.")
        if self.temperature_celsius <= -273.15:
            raise ValueError("temperature_celsius must be above absolute zero.")


@dataclass(frozen=True)
class ModelSpec:
    """Model-specific constructors and notebook defaults, without eager imports."""

    folder: str
    prefix: str
    class_name: str
    params_module: str
    params_loader: str
    default_protocol: Protocol
    neuron_options: tuple[str, ...] = ()
    temperature_parameters: bool = False

    def parameters(self, protocol: Protocol):
        """Load one parameter object shared by the two model constructors."""
        module = import_module(f"validation.neuron.cell.{self.folder}.{self.params_module}")
        options = {"temperature_celsius": protocol.temperature_celsius} if self.temperature_parameters else {}
        return getattr(module, self.params_loader)(**options)

    def create(self, backend: str, params, protocol: Protocol, *, nrnmech_path=None):
        """Create an unbuilt model; its existing constructor owns morphology defaults."""
        if backend not in ("neuron", "braincell"):
            raise ValueError("backend must be 'neuron' or 'braincell'.")
        module = import_module(f"validation.neuron.cell.{self.folder}.{self.prefix}_{backend}")
        options = {"params": params}
        if backend == "braincell":
            options.update(temperature_celsius=protocol.temperature_celsius, v_init_mV=protocol.v_init_mV)
        else:
            options["nrnmech_path"] = nrnmech_path
            options.update({name: getattr(protocol, name) for name in self.neuron_options})
        return getattr(module, self.class_name)(**options)


# These protocols reproduce the eight regular notebooks, including their precision.
MODELS = {
    "bc_ma2025": ModelSpec(
        "bc_ma2025", "bc", "BC", "parameters", "load_bc25_params", Protocol(dt_ms=0.05, stim_dur_ms=80.0)
    ),
    "dcn_su2015": ModelSpec(
        "dcn_su2015",
        "dcn",
        "DCN",
        "parameters",
        "load_dcn15_params",
        Protocol(amp_nA=0.1, temperature_celsius=32.0),
        ("temperature_celsius", "v_init_mV"),
        True,
    ),
    "goc_ma2020": ModelSpec(
        "goc_ma2020",
        "goc",
        "GoC",
        "parameters",
        "load_goc20_params",
        Protocol(duration_ms=100.0, stim_dur_ms=80.0, amp_nA=0.2, temperature_celsius=34.0, precision=32),
    ),
    "grc_ma2020": ModelSpec(
        "grc_ma2020",
        "grc",
        "GrC",
        "parameters",
        "load_grc20_params",
        Protocol(amp_nA=0.01, temperature_celsius=25.0),
        ("temperature_celsius",),
    ),
    "grc_ma2020_full": ModelSpec(
        "grc_ma2020",
        "grc_full",
        "GrCFull",
        "grc_full_parameters",
        "load_grc20_full_params",
        Protocol(amp_nA=0.01, temperature_celsius=25.0),
        ("temperature_celsius",),
    ),
    "io_zh2019": ModelSpec(
        "io_zh2019", "io", "IO", "parameters", "load_io19_params", Protocol(duration_ms=100.0, stim_dur_ms=80.0)
    ),
    "pc_ma2024": ModelSpec(
        "pc_ma2024",
        "pc",
        "PC",
        "parameters",
        "load_pc24_params",
        Protocol(duration_ms=20.0, delay_ms=5.0, stim_dur_ms=10.0, amp_nA=0.5, precision=32),
    ),
    "sc_ma2021": ModelSpec(
        "sc_ma2021",
        "sc",
        "SC",
        "parameters",
        "load_sc21_params",
        Protocol(duration_ms=100.0, stim_dur_ms=80.0, temperature_celsius=32.0),
        ("temperature_celsius", "v_init_mV"),
    ),
}


def get_model(name: str) -> ModelSpec:
    """Find a regular model or the separately registered GrC full variant."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown cell {name!r}; choose from {', '.join(MODELS)}.") from None
