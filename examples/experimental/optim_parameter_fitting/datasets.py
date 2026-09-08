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

"""Step-only synthetic datasets for modular parameter-fitting experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.models import ModelDefinition

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Protocol:
    """Describe one fixed current-clamp observation."""

    protocol_id: str
    split: str
    feature: str
    amplitude_na: float
    target_spike_count: int


@dataclass(frozen=True)
class DatasetBundle:
    """Store all protocol currents, target voltages, and split metadata."""

    protocols: tuple[Protocol, ...]
    time_ms: np.ndarray
    current_na: np.ndarray
    target_voltage_mv: np.ndarray
    dt_ms: float
    baseline_stop_ms: float
    stimulus_stop_ms: float

    def indices(self, split: str) -> np.ndarray:
        """Return stable indices for one split."""
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}.")
        return np.asarray([index for index, item in enumerate(self.protocols) if item.split == split], dtype=np.int32)

    def subset(self, split: str) -> tuple[np.ndarray, np.ndarray, tuple[Protocol, ...]]:
        """Return current, target, and metadata for one split."""
        indices = self.indices(split)
        return (
            self.current_na[indices],
            self.target_voltage_mv[indices],
            tuple(self.protocols[index] for index in indices),
        )

    def describe(self) -> dict[str, object]:
        """Return an exact, serializable dataset manifest."""
        return {
            "dt_ms": self.dt_ms,
            "duration_ms": float(self.time_ms.size * self.dt_ms),
            "num_steps": int(self.time_ms.size),
            "baseline_stop_ms": self.baseline_stop_ms,
            "stimulus_stop_ms": self.stimulus_stop_ms,
            "current_shape": list(self.current_na.shape),
            "target_voltage_shape": list(self.target_voltage_mv.shape),
            "split_counts": {split: int(self.indices(split).size) for split in SPLITS},
            "protocols": [
                {
                    "protocol_id": item.protocol_id,
                    "split": item.split,
                    "family": "step",
                    "feature": item.feature,
                    "injection_site": "soma",
                    "amplitude_nA": item.amplitude_na,
                    "target_spike_count": item.target_spike_count,
                }
                for item in self.protocols
            ],
        }


@dataclass(frozen=True)
class DatasetDefinition:
    """Calibrate the feature-based Step suite against the selected target model."""

    name: str = "feature_step_1cv_v1"
    target_source: str = "regenerate"
    dt_ms: float = 0.025
    duration_ms: float = 100.0
    baseline_stop_ms: float = 20.0
    stimulus_stop_ms: float = 80.0

    def __post_init__(self) -> None:
        if self.target_source != "regenerate":
            raise ValueError("The baseline dataset requires target_source='regenerate'.")
        if not 0.0 < self.baseline_stop_ms < self.stimulus_stop_ms < self.duration_ms:
            raise ValueError("Require 0 < baseline stop < stimulus stop < duration.")
        for value in (self.duration_ms, self.baseline_stop_ms, self.stimulus_stop_ms):
            steps = value / self.dt_ms
            if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-10):
                raise ValueError("All dataset times must align to dt_ms.")

    @property
    def num_steps(self) -> int:
        """Return the fixed number of simulation steps."""
        return int(round(self.duration_ms / self.dt_ms))

    @property
    def baseline_stop(self) -> int:
        """Return the first stimulus index."""
        return int(round(self.baseline_stop_ms / self.dt_ms))

    @property
    def stimulus_stop(self) -> int:
        """Return the first recovery index."""
        return int(round(self.stimulus_stop_ms / self.dt_ms))

    def generate(self, model: ModelDefinition) -> DatasetBundle:
        """Calibrate eight Step protocols and regenerate synthetic targets."""
        with jax.enable_x64(True), brainstate.environ.context(dt=self.dt_ms * u.ms, precision=64):
            negative_grid = np.linspace(-0.8, 0.0, 321, dtype=np.float64)
            positive_grid = np.linspace(0.0, 0.8, 801, dtype=np.float64)
            negative_voltage = self._simulate_sweep(model, negative_grid)
            positive_voltage = self._simulate_sweep(model, positive_grid)
            amplitudes = self._calibrate_amplitudes(negative_grid, positive_grid, negative_voltage, positive_voltage)
            specifications = (
                ("step_mild_negative", "train", "passive_hyperpolarizing", amplitudes["mild_negative"]),
                ("step_small_positive", "train", "passive_depolarizing", amplitudes["small_positive"]),
                ("step_near_rheobase", "train", "threshold_margin", 0.9 * amplitudes["rheobase"]),
                ("step_1_spike", "train", "firing_rate_1", amplitudes["spike_1"]),
                ("step_3_spike", "train", "firing_rate_3", amplitudes["spike_3"]),
                ("step_moderate_negative", "validation", "passive_holdout", amplitudes["moderate_negative"]),
                ("step_2_spike", "validation", "firing_rate_2", amplitudes["spike_2"]),
                ("step_4_spike", "test", "firing_rate_4", amplitudes["spike_4"]),
            )
            current = np.stack([self._step_current(item[3]) for item in specifications])
            voltage = np.asarray(model.simulate(current, dt_ms=self.dt_ms), dtype=np.float64)
        counts = upward_crossing_counts(voltage[..., 0]).astype(np.int32)
        protocols = tuple(
            Protocol(protocol_id, split, feature, float(amplitude), int(count))
            for (protocol_id, split, feature, amplitude), count in zip(specifications, counts)
        )
        bundle = DatasetBundle(
            protocols=protocols,
            time_ms=np.arange(self.num_steps, dtype=np.float64) * self.dt_ms,
            current_na=current,
            target_voltage_mv=voltage,
            dt_ms=self.dt_ms,
            baseline_stop_ms=self.baseline_stop_ms,
            stimulus_stop_ms=self.stimulus_stop_ms,
        )
        validate_bundle(bundle)
        return bundle

    def describe(self) -> dict[str, object]:
        """Return serializable dataset-generation metadata."""
        return {
            "name": self.name,
            "target_source": self.target_source,
            "family": "step",
            "dt_ms": self.dt_ms,
            "duration_ms": self.duration_ms,
            "baseline_stop_ms": self.baseline_stop_ms,
            "stimulus_stop_ms": self.stimulus_stop_ms,
            "split_features": {
                "train": ["mild_negative", "small_positive", "near_rheobase", "1_spike", "3_spike"],
                "validation": ["moderate_negative", "2_spike"],
                "test": ["4_spike"],
            },
        }

    def _simulate_sweep(self, model: ModelDefinition, amplitudes: np.ndarray) -> np.ndarray:
        current = np.zeros((amplitudes.size, self.num_steps), dtype=np.float64)
        current[:, self.baseline_stop : self.stimulus_stop] = amplitudes[:, None]
        return np.asarray(model.simulate(current, dt_ms=self.dt_ms), dtype=np.float64)

    def _calibrate_amplitudes(self, negative_grid, positive_grid, negative_voltage, positive_voltage):
        baseline_slice = slice(self.baseline_stop - 200, self.baseline_stop)
        steady_slice = slice(self.stimulus_stop - 200, self.stimulus_stop)
        negative_baseline = np.mean(negative_voltage[:, baseline_slice, 0], axis=1)
        positive_baseline = np.mean(positive_voltage[:, baseline_slice, 0], axis=1)
        negative_steady = np.mean(negative_voltage[:, steady_slice, 0], axis=1)
        positive_steady = np.mean(positive_voltage[:, steady_slice, 0], axis=1)
        mild = int(np.argmin(np.abs((negative_steady - negative_baseline) + 5.0)))
        moderate = int(np.argmin(np.abs((negative_steady - negative_baseline) + 10.0)))
        counts = upward_crossing_counts(positive_voltage[..., 0])
        silent = np.flatnonzero(counts == 0)
        spiking = np.flatnonzero(counts > 0)
        if not silent.size or not spiking.size:
            raise RuntimeError("Calibration sweep did not span subthreshold and spiking regimes.")
        small = int(silent[np.argmin(np.abs((positive_steady[silent] - positive_baseline[silent]) - 5.0))])
        result = {
            "mild_negative": float(negative_grid[mild]),
            "moderate_negative": float(negative_grid[moderate]),
            "small_positive": float(positive_grid[small]),
            "rheobase": float(positive_grid[int(spiking[0])]),
        }
        for count in range(1, 5):
            matching = np.flatnonzero(counts == count)
            if not matching.size:
                raise RuntimeError(f"Calibration sweep found no {count}-spike interval.")
            lo, hi = _largest_contiguous_interval(matching)
            result[f"spike_{count}"] = 0.5 * float(positive_grid[lo] + positive_grid[hi])
        return result

    def _step_current(self, amplitude_na: float) -> np.ndarray:
        current = np.zeros((self.num_steps,), dtype=np.float64)
        current[self.baseline_stop : self.stimulus_stop] = amplitude_na
        return current


def feature_step_dataset(**kwargs) -> DatasetDefinition:
    """Return the Step-only one-CV baseline dataset definition."""
    return DatasetDefinition(**kwargs)


def upward_crossing_counts(voltage_mv, *, threshold_mv: float = 0.0):
    """Count upward threshold crossings on the final time axis."""
    values = np.asarray(voltage_mv)
    return np.sum((values[..., :-1] < threshold_mv) & (values[..., 1:] >= threshold_mv), axis=-1)


def validate_bundle(bundle: DatasetBundle) -> None:
    """Validate the exact Step-only baseline split and tensor contract."""
    counts = {split: int(bundle.indices(split).size) for split in SPLITS}
    if counts != {"train": 5, "validation": 2, "test": 1}:
        raise ValueError(f"Expected Step split counts 5/2/1, got {counts!r}.")
    expected_current = (8, bundle.time_ms.size)
    expected_voltage = (8, bundle.time_ms.size, 1)
    if bundle.current_na.shape != expected_current or bundle.target_voltage_mv.shape != expected_voltage:
        raise ValueError(
            f"Unexpected dataset shapes current={bundle.current_na.shape!r}, voltage={bundle.target_voltage_mv.shape!r}."
        )
    if not np.all(np.isfinite(bundle.current_na)) or not np.all(np.isfinite(bundle.target_voltage_mv)):
        raise ValueError("Dataset arrays must be finite.")
    if np.any(bundle.current_na[:, : int(bundle.baseline_stop_ms / bundle.dt_ms)] != 0.0):
        raise ValueError("Baseline current must be zero.")
    if np.any(bundle.current_na[:, int(bundle.stimulus_stop_ms / bundle.dt_ms) :] != 0.0):
        raise ValueError("Recovery current must be zero.")


def write_bundle(directory: Path, bundle: DatasetBundle) -> None:
    """Write reproducible dataset arrays and metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "dataset.npz",
        time_ms=bundle.time_ms,
        current_nA=bundle.current_na,
        target_voltage_mV=bundle.target_voltage_mv,
        protocol_id=np.asarray([item.protocol_id for item in bundle.protocols]),
        split=np.asarray([item.split for item in bundle.protocols]),
        feature=np.asarray([item.feature for item in bundle.protocols]),
        amplitude_nA=np.asarray([item.amplitude_na for item in bundle.protocols]),
        target_spike_count=np.asarray([item.target_spike_count for item in bundle.protocols]),
    )
    (directory / "dataset_manifest.json").write_text(
        json.dumps(bundle.describe(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _largest_contiguous_interval(indices: np.ndarray) -> tuple[int, int]:
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((np.asarray([0]), breaks + 1))
    stops = np.concatenate((breaks + 1, np.asarray([indices.size])))
    winner = int(np.argmax(stops - starts))
    run = indices[starts[winner] : stops[winner]]
    return int(run[0]), int(run[-1])
