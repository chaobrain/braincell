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

"""Calibrated 50 ms DC protocols for the 3-CV/9-conductance experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import AllRegion, BranchSlice, at

DT_MS = 0.025
DURATION_MS = 50.0
N_STEPS = int(round(DURATION_MS / DT_MS))
SITES = ("soma", "dend_a", "dend_b")
CHANNELS = ("leak", "na", "k")
SPLITS = ("train", "validation", "test")
CONDUCTANCE_UNIT = u.mS / u.cm**2
TARGET_CONDUCTANCES = np.asarray(
    (
        (0.60, 120.0, 36.0),
        (0.48, 96.0, 28.8),
        (0.42, 84.0, 25.2),
    ),
    dtype=np.float64,
)
NEGATIVE_TARGETS_MV = (-80.0, -90.0, -100.0, -110.0)
POSITIVE_SPIKE_COUNTS = (0, 1, 2, 3, 4)
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "hybrid_initialization"


@dataclass(frozen=True)
class DcProtocol:
    """One constant-current protocol and its deterministic split."""

    protocol_id: str
    injection_site: str
    split: Literal["train", "validation", "test"]
    response_label: str
    amplitude_na: float

    def __post_init__(self) -> None:
        if self.injection_site not in SITES:
            raise ValueError(f"Unknown injection site {self.injection_site!r}.")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split {self.split!r}.")
        if not np.isfinite(self.amplitude_na):
            raise ValueError("Protocol amplitude must be finite.")


@dataclass(frozen=True)
class DcCalibration:
    """Site-specific currents selected from target response sweeps."""

    negative_amplitudes_na: dict[str, tuple[float, ...]]
    negative_minima_mv: dict[str, tuple[float, ...]]
    positive_intervals_na: dict[str, dict[int, tuple[float, float]]]


@dataclass(frozen=True)
class DcDataset:
    """In-memory target traces and protocol metadata."""

    protocols: tuple[DcProtocol, ...]
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    target_spike_counts: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}.")
        return np.asarray([index for index, protocol in enumerate(self.protocols) if protocol.split == split])

    def subset(self, split: str) -> tuple[tuple[DcProtocol, ...], np.ndarray]:
        indices = self.indices(split)
        return tuple(self.protocols[index] for index in indices), self.voltage_mv[indices]


def parameter_names() -> tuple[str, ...]:
    """Return the stable nine-root order used by search and training."""
    return tuple(f"{site}.{channel}.scale" for site in SITES for channel in CHANNELS)


def build_morphology() -> braincell.Morphology:
    """Build one soma and two asymmetric one-CV dendrites."""
    soma = braincell.Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[12.5, 12.5] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    dend_a = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    dend_b = braincell.Branch.from_lengths(
        lengths=[150.0] * u.um,
        radii=[1.5, 0.75] * u.um,
        type="basal_dendrite",
    )
    morphology.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
    morphology.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)
    return morphology


def build_cell(protocols: tuple[DcProtocol, ...], *, trainable: bool) -> braincell.Cell:
    """Build an x64 experiment Cell independently of caller context."""
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        return _build_cell(protocols, trainable=trainable)


def _build_cell(protocols: tuple[DcProtocol, ...], *, trainable: bool) -> braincell.Cell:
    """Build one protocol-population Cell with shared regional parameters."""
    if not protocols:
        raise ValueError("At least one protocol is required.")
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(len(protocols),),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
    )
    for site_index, site in enumerate(SITES):
        leak, sodium, potassium = TARGET_CONDUCTANCES[site_index]
        region = BranchSlice(branch_index=site_index, prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Channel(
                "IL",
                name=f"{site}_leak",
                g_max=leak * CONDUCTANCE_UNIT,
                E=-54.387 * u.mV,
            ),
            braincell.mech.Channel(
                "Na_HH1952",
                name=f"{site}_na",
                g_max=sodium * CONDUCTANCE_UNIT,
            ),
            braincell.mech.Channel(
                "K_HH1952",
                name=f"{site}_k",
                g_max=potassium * CONDUCTANCE_UNIT,
            ),
        )
    for site in SITES:
        amplitudes = np.asarray(
            [protocol.amplitude_na if protocol.injection_site == site else 0.0 for protocol in protocols],
            dtype=np.float64,
        )
        cell.place(
            at(site, 0.5),
            braincell.mech.CurrentClamp(
                delay=0.0 * u.ms,
                durations=DURATION_MS * u.ms,
                amplitudes=amplitudes * u.nA,
            ),
        )
    if trainable:
        for site in SITES:
            for channel in CHANNELS:
                cell.channels[f"{site}_{channel}"].trainable(
                    g_max=braincell.trainable.scale(
                        group_by="all",
                        transform=brainstate.nn.SigmoidT(0.5, 1.5),
                        name=f"{site}.{channel}.scale",
                    )
                )
    cell.init_state()
    if cell.n_cv != 3:
        raise RuntimeError(f"Expected exactly three CVs, got {cell.n_cv}.")
    return cell


def simulate_voltage(cell: braincell.Cell, *, num_steps: int = N_STEPS) -> object:
    """Return pre-step voltage with shape ``(protocol, time, CV)``."""
    cell.reset_state()

    def step(time_ms):
        voltage = cell.V.value.to_decimal(u.mV)
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return voltage

    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        times_ms = jnp.arange(num_steps, dtype=jnp.float64) * DT_MS
        time_leading = brainstate.transform.for_loop(step, times_ms)
    return jnp.moveaxis(time_leading, 0, 1)


def calibrate_protocols() -> DcCalibration:
    """Run two target sweeps and resolve response-calibrated currents."""
    positive_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    negative_grid = np.linspace(-2.0, 0.0, 401, dtype=np.float64)
    positive_protocols = _calibration_protocols(positive_grid, prefix="positive")
    negative_protocols = _calibration_protocols(negative_grid, prefix="negative")
    positive_voltage = np.asarray(simulate_voltage(build_cell(positive_protocols, trainable=False)))
    negative_voltage = np.asarray(simulate_voltage(build_cell(negative_protocols, trainable=False)))
    positive_counts = upward_crossing_counts(positive_voltage[..., 0])
    negative_minima = np.min(negative_voltage[..., 0], axis=1)

    positive_intervals = {}
    negative_amplitudes = {}
    selected_minima = {}
    for site_index, site in enumerate(SITES):
        p_slice = slice(site_index * positive_grid.size, (site_index + 1) * positive_grid.size)
        n_slice = slice(site_index * negative_grid.size, (site_index + 1) * negative_grid.size)
        site_counts = positive_counts[p_slice]
        intervals = {}
        for count in POSITIVE_SPIKE_COUNTS:
            matching = np.flatnonzero(site_counts == count)
            if matching.size == 0:
                raise RuntimeError(f"Calibration found no {count}-spike interval for site {site!r}.")
            lo, hi = _largest_contiguous_interval(matching)
            intervals[count] = (float(positive_grid[lo]), float(positive_grid[hi]))
        positive_intervals[site] = intervals

        minima = negative_minima[n_slice]
        indices = [int(np.argmin(np.abs(minima - target))) for target in NEGATIVE_TARGETS_MV]
        negative_amplitudes[site] = tuple(float(negative_grid[index]) for index in indices)
        selected_minima[site] = tuple(float(minima[index]) for index in indices)
    return DcCalibration(negative_amplitudes, selected_minima, positive_intervals)


def build_catalog(calibration: DcCalibration) -> tuple[DcProtocol, ...]:
    """Build the fixed 15/6/6 train/validation/test catalog."""
    protocols = []
    negative_splits = ("train", "validation", "test", "train")
    count_splits = {0: "train", 1: "train", 2: "validation", 3: "train", 4: "test"}
    for site in SITES:
        for target, amplitude, split in zip(
            NEGATIVE_TARGETS_MV,
            calibration.negative_amplitudes_na[site],
            negative_splits,
        ):
            protocols.append(
                DcProtocol(
                    protocol_id=f"dc_{site}_negative_{abs(int(target))}mv",
                    injection_site=site,
                    split=split,
                    response_label=f"soma_min_{target:g}mV",
                    amplitude_na=amplitude,
                )
            )
        for count in POSITIVE_SPIKE_COUNTS:
            lo, hi = calibration.positive_intervals_na[site][count]
            protocols.append(
                DcProtocol(
                    protocol_id=f"dc_{site}_{count}spike",
                    injection_site=site,
                    split=count_splits[count],
                    response_label=f"target_{count}_spikes",
                    amplitude_na=0.5 * (lo + hi),
                )
            )
    catalog = tuple(protocols)
    split_counts = {split: sum(protocol.split == split for protocol in catalog) for split in SPLITS}
    if split_counts != {"train": 15, "validation": 6, "test": 6}:
        raise RuntimeError(f"Unexpected split counts {split_counts!r}.")
    return catalog


def generate_dataset(output_dir: Path = ARTIFACT_ROOT / "dataset") -> DcDataset:
    """Generate the target dataset under one continuous x64 lifecycle."""
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        return _generate_dataset(output_dir)


def _generate_dataset(output_dir: Path) -> DcDataset:
    """Calibrate, simulate, validate, and persist the target dataset."""
    calibration = calibrate_protocols()
    protocols = build_catalog(calibration)
    voltage_mv = np.asarray(simulate_voltage(build_cell(protocols, trainable=False)), dtype=np.float64)
    spike_counts = upward_crossing_counts(voltage_mv[..., 0]).astype(np.int16)
    dataset = DcDataset(
        protocols=protocols,
        time_ms=np.arange(N_STEPS, dtype=np.float64) * DT_MS,
        voltage_mv=voltage_mv,
        target_spike_counts=spike_counts,
    )
    _validate_dataset(dataset)
    write_dataset(output_dir, dataset, calibration)
    return dataset


def write_dataset(output_dir: Path, dataset: DcDataset, calibration: DcCalibration) -> None:
    """Write numeric and human-readable dataset artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        time_ms=dataset.time_ms,
        voltage_mV=dataset.voltage_mv,
        target_spike_counts=dataset.target_spike_counts,
        protocol_id=np.asarray([protocol.protocol_id for protocol in dataset.protocols]),
        injection_site=np.asarray([protocol.injection_site for protocol in dataset.protocols]),
        split=np.asarray([protocol.split for protocol in dataset.protocols]),
        response_label=np.asarray([protocol.response_label for protocol in dataset.protocols]),
        amplitude_nA=np.asarray([protocol.amplitude_na for protocol in dataset.protocols]),
        target_conductances_mS_per_cm2=TARGET_CONDUCTANCES,
    )
    rows = [
        {
            "index": index,
            "protocol_id": protocol.protocol_id,
            "site": protocol.injection_site,
            "split": protocol.split,
            "response_label": protocol.response_label,
            "amplitude_nA": protocol.amplitude_na,
            "target_spike_count": int(dataset.target_spike_counts[index]),
        }
        for index, protocol in enumerate(dataset.protocols)
    ]
    with (output_dir / "protocol_catalog.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "num_steps": N_STEPS,
        "protocol_count": len(dataset.protocols),
        "split_counts": {split: int(dataset.indices(split).size) for split in SPLITS},
        "target_conductances_mS_per_cm2": TARGET_CONDUCTANCES.tolist(),
        "target_spike_counts": dataset.target_spike_counts.tolist(),
        "calibration": {
            "negative_amplitudes_nA": calibration.negative_amplitudes_na,
            "negative_minima_mV": calibration.negative_minima_mv,
            "positive_intervals_nA": calibration.positive_intervals_na,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset(output_dir: Path = ARTIFACT_ROOT / "dataset") -> DcDataset:
    """Load a previously generated dataset."""
    with np.load(output_dir / "dataset.npz") as values:
        protocols = tuple(
            DcProtocol(
                protocol_id=str(protocol_id),
                injection_site=str(site),
                split=str(split),
                response_label=str(label),
                amplitude_na=float(amplitude),
            )
            for protocol_id, site, split, label, amplitude in zip(
                values["protocol_id"],
                values["injection_site"],
                values["split"],
                values["response_label"],
                values["amplitude_nA"],
            )
        )
        dataset = DcDataset(
            protocols=protocols,
            time_ms=np.asarray(values["time_ms"]),
            voltage_mv=np.asarray(values["voltage_mV"]),
            target_spike_counts=np.asarray(values["target_spike_counts"]),
        )
    _validate_dataset(dataset)
    return dataset


def upward_crossing_counts(voltage_mv, *, threshold_mv: float = 0.0):
    """Count upward threshold crossings on the final axis."""
    values = np.asarray(voltage_mv)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("Voltage must have a time axis with at least two samples.")
    return np.sum((values[..., :-1] < threshold_mv) & (values[..., 1:] >= threshold_mv), axis=-1)


def _calibration_protocols(amplitudes: np.ndarray, *, prefix: str) -> tuple[DcProtocol, ...]:
    return tuple(
        DcProtocol(
            protocol_id=f"{prefix}_{site}_{index:04d}",
            injection_site=site,
            split="train",
            response_label=prefix,
            amplitude_na=float(amplitude),
        )
        for site in SITES
        for index, amplitude in enumerate(amplitudes)
    )


def _largest_contiguous_interval(indices: np.ndarray) -> tuple[int, int]:
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((np.asarray([0]), breaks + 1))
    stops = np.concatenate((breaks + 1, np.asarray([indices.size])))
    winner = int(np.argmax(stops - starts))
    run = indices[starts[winner] : stops[winner]]
    return int(run[0]), int(run[-1])


def _validate_dataset(dataset: DcDataset) -> None:
    if len(dataset.protocols) != 27:
        raise ValueError(f"Expected 27 protocols, got {len(dataset.protocols)}.")
    if dataset.voltage_mv.shape != (27, N_STEPS, 3):
        raise ValueError(f"Expected voltage shape (27, {N_STEPS}, 3), got {dataset.voltage_mv.shape!r}.")
    if dataset.time_ms.shape != (N_STEPS,):
        raise ValueError(f"Expected time shape ({N_STEPS},), got {dataset.time_ms.shape!r}.")
    if dataset.target_spike_counts.shape != (27,):
        raise ValueError(f"Expected 27 target spike counts, got {dataset.target_spike_counts.shape!r}.")
    if not np.all(np.isfinite(dataset.voltage_mv)):
        raise ValueError("Dataset voltage contains non-finite values.")


__all__ = [
    "ARTIFACT_ROOT",
    "CHANNELS",
    "CONDUCTANCE_UNIT",
    "DT_MS",
    "DURATION_MS",
    "DcCalibration",
    "DcDataset",
    "DcProtocol",
    "N_STEPS",
    "SITES",
    "TARGET_CONDUCTANCES",
    "build_catalog",
    "build_cell",
    "build_morphology",
    "calibrate_protocols",
    "generate_dataset",
    "load_dataset",
    "parameter_names",
    "simulate_voltage",
    "upward_crossing_counts",
]
