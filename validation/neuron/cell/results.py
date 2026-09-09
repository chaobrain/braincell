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

"""Portable raw traces and metadata, independent of either simulator."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Store one soma voltage trace.

    Parameters
    ----------
    time_ms : array_like
        Nonempty, strictly increasing sample times, shape ``(n_samples,)``.
    voltage_mV : array_like
        Voltages of the same shape. NaN/Inf values are retained for diagnosis.

    Notes
    -----
    Both arrays are copied into read-only float64 NumPy arrays.
    """

    time_ms: np.ndarray
    voltage_mV: np.ndarray

    def __post_init__(self):
        time = np.array(self.time_ms, dtype=float, copy=True)
        voltage = np.array(self.voltage_mV, dtype=float, copy=True)
        if time.ndim != 1 or voltage.shape != time.shape or time.size == 0:
            raise ValueError("Trace time and voltage must be nonempty matching 1D arrays.")
        if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("Trace times must be finite and strictly increasing.")
        # Preserve nonfinite voltages in raw output so failed simulations remain inspectable.
        time.flags.writeable = False
        voltage.flags.writeable = False
        object.__setattr__(self, "time_ms", time)
        object.__setattr__(self, "voltage_mV", voltage)


@dataclass(frozen=True)
class ComparisonResult:
    """Collect two original traces and the actual run configuration.

    Parameters
    ----------
    neuron, braincell : Trace
        Raw traces on their original time axes.
    metadata : dict
        JSON-compatible configuration and provenance; saved runs use format_version 1.
    output_dir : pathlib.Path, optional
        Location of the persisted run, or None for an in-memory result.
    """

    neuron: Trace
    braincell: Trace
    metadata: dict
    output_dir: Path | None = None


def save_result(result: ComparisonResult, directory: str | Path) -> Path:
    """Write raw arrays and JSON metadata.

    Parameters
    ----------
    result : ComparisonResult
        Raw simulation output, including format_version 1 metadata.
    directory : str or pathlib.Path
        Destination; existing raw result files raise FileExistsError.

    Returns
    -------
    pathlib.Path
        Directory containing traces.npz and metadata.json.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "traces.npz").exists() or (directory / "metadata.json").exists():
        raise FileExistsError(f"Result already exists: {directory}")
    metadata = json.dumps(result.metadata, indent=2, allow_nan=False) + "\n"
    np.savez_compressed(
        directory / "traces.npz",
        neuron_time_ms=result.neuron.time_ms,
        neuron_voltage_mV=result.neuron.voltage_mV,
        braincell_time_ms=result.braincell.time_ms,
        braincell_voltage_mV=result.braincell.voltage_mV,
    )
    (directory / "metadata.json").write_text(metadata)
    return directory


def load_result(directory: str | Path) -> ComparisonResult:
    """Read a saved run without importing BrainCell or NEURON.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing traces.npz and metadata.json.

    Returns
    -------
    ComparisonResult
        Original traces, metadata and their persisted location.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported comparison result format_version.")
    with np.load(directory / "traces.npz", allow_pickle=False) as arrays:
        return ComparisonResult(
            Trace(arrays["neuron_time_ms"], arrays["neuron_voltage_mV"]),
            Trace(arrays["braincell_time_ms"], arrays["braincell_voltage_mV"]),
            metadata,
            directory,
        )
