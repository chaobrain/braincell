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

"""Seven-CV current-stimulus candidate dataset for robust experiment design."""

from __future__ import annotations

import argparse
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
from braincell.filter import AllRegion, BranchSlice

DT_MS = 0.025
DURATION_MS = 100.0
N_STEPS = int(round(DURATION_MS / DT_MS))
BASELINE_STOP_MS = 20.0
STIMULUS_STOP_MS = 80.0
BASELINE_STOP = int(round(BASELINE_STOP_MS / DT_MS))
STIMULUS_STOP = int(round(STIMULUS_STOP_MS / DT_MS))
SITES = ("soma", "dend_a", "dend_b")
SPLITS = ("train", "validation", "test")
PARAMETER_NAMES = (
    "soma.leak.scale",
    "soma.na.scale",
    "soma.k.scale",
    "dend.leak.scale",
    "dend.na.scale",
    "dend.k.scale",
)
CONDUCTANCE_UNIT = u.mS / u.cm**2
SOMA_CONDUCTANCES = np.asarray((0.60, 120.0, 36.0), dtype=np.float64)
DEND_CONDUCTANCES = np.asarray((0.45, 90.0, 27.0), dtype=np.float64)
PRMLS_CLOCKS_MS = (2.0, 5.0, 10.0)
PRMLS_LEVELS = np.asarray((-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0), dtype=np.float64)
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "stimulus_design_v2"


@dataclass(frozen=True)
class BaseWaveform:
    """One location-independent current waveform."""

    waveform_id: str
    family: Literal["step", "prmls"]
    feature: str
    split: Literal["train", "validation", "test"]
    current_na: np.ndarray
    clock_ms: float | None = None
    seed: int | None = None
    amplitude_na: float | None = None


@dataclass(frozen=True)
class StimulusProtocol:
    """One base waveform applied at one fixed injection location."""

    protocol_id: str
    waveform_id: str
    family: str
    feature: str
    split: str
    injection_site: str
    clock_ms: float | None
    seed: int | None
    amplitude_na: float | None


@dataclass(frozen=True)
class StimulusCalibration:
    """Amplitudes selected once from soma-injection target responses."""

    mild_negative_na: float
    moderate_negative_na: float
    small_positive_na: float
    rheobase_na: float
    spike_intervals_na: dict[int, tuple[float, float]]
    prmls_amplitude_na: float
    prmls_adjustments: int


@dataclass(frozen=True)
class StimulusDataset:
    """Current, target voltage, and metadata for all 60 protocols."""

    protocols: tuple[StimulusProtocol, ...]
    time_ms: np.ndarray
    current_na: np.ndarray
    voltage_mv: np.ndarray
    target_spike_counts: np.ndarray
    calibration: StimulusCalibration

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}.")
        return np.asarray([index for index, item in enumerate(self.protocols) if item.split == split])


def build_morphology() -> braincell.Morphology:
    """Build the unchanged soma and two asymmetric dendritic branches."""
    soma = braincell.Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[12.5, 12.5] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.attach(
        parent="soma",
        child_branch=braincell.Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="basal_dendrite",
        ),
        child_name="dend_a",
        parent_x=1.0,
    )
    morphology.attach(
        parent="soma",
        child_branch=braincell.Branch.from_lengths(
            lengths=[150.0] * u.um,
            radii=[1.5, 0.75] * u.um,
            type="basal_dendrite",
        ),
        child_name="dend_b",
        parent_x=1.0,
    )
    return morphology


def build_cell(current_na, *, trainable: bool) -> braincell.Cell:
    """Build a seven-CV protocol-population Cell with static current playback."""
    current = np.asarray(current_na, dtype=np.float64)
    if current.ndim != 3 or current.shape[1:] != (N_STEPS, len(SITES)):
        raise ValueError(f"current_na must have shape (protocol, {N_STEPS}, 3), got {current.shape!r}.")
    if current.shape[0] < 1 or not np.all(np.isfinite(current)):
        raise ValueError("current_na must contain at least one finite protocol.")
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        cell = _build_cell(current.shape[0], trainable=trainable)
        _attach_current_playback(cell, jnp.asarray(current))
    return cell


def _build_cell(population_size: int, *, trainable: bool) -> braincell.Cell:
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranchList((1, 3, 3)),
        pop_size=(population_size,),
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
    _paint_channels(cell, BranchSlice(branch_index=0, prox=0.0, dist=1.0), "soma", SOMA_CONDUCTANCES)
    _paint_channels(cell, BranchSlice(branch_index=[1, 2], prox=0.0, dist=1.0), "dend", DEND_CONDUCTANCES)
    if trainable:
        for owner in ("soma", "dend"):
            for channel in ("leak", "na", "k"):
                cell.channels[f"{owner}_{channel}"].trainable(
                    g_max=braincell.trainable.scale(
                        group_by="all",
                        transform=brainstate.nn.SigmoidT(0.5, 1.5),
                        name=f"{owner}.{channel}.scale",
                    )
                )
    cell.init_state()
    if cell.n_cv != 7:
        raise RuntimeError(f"Expected seven CVs, got {cell.n_cv}.")
    return cell


def _paint_channels(cell, region, owner: str, values: np.ndarray) -> None:
    leak, sodium, potassium = values
    cell.paint(
        region,
        braincell.mech.Channel(
            "IL",
            name=f"{owner}_leak",
            g_max=leak * CONDUCTANCE_UNIT,
            E=-54.387 * u.mV,
        ),
        braincell.mech.Channel(
            "Na_HH1952",
            name=f"{owner}_na",
            g_max=sodium * CONDUCTANCE_UNIT,
        ),
        braincell.mech.Channel(
            "K_HH1952",
            name=f"{owner}_k",
            g_max=potassium * CONDUCTANCE_UNIT,
        ),
    )


def _attach_current_playback(cell: braincell.Cell, current_na) -> None:
    branch_to_cvs = cell.cv_tree.branch_to_cv_ids
    target_cvs = np.asarray((branch_to_cvs[0][0], branch_to_cvs[1][-1], branch_to_cvs[2][-1]), dtype=np.int32)
    target_points = jnp.asarray(cell.node_tree.cv_to_mid_node_id[target_cvs])
    target_area = cell.runtime.point_area.to_decimal(u.cm**2)[target_points]
    n_point = cell.n_point

    def playback(_point_voltage):
        time_ms = cell._resolve_t().to_decimal(u.ms)
        index = jnp.clip(jnp.rint(time_ms / DT_MS).astype(jnp.int32), 0, N_STEPS - 1)
        density = jnp.zeros((current_na.shape[0], n_point), dtype=current_na.dtype)
        density = density.at[:, target_points].set(current_na[:, index, :] / target_area)
        return density * u.nA / u.cm**2

    cell.add_current_input("stimulus_design_current", playback)


def simulate_voltage(cell: braincell.Cell, *, num_steps: int = N_STEPS) -> object:
    """Return pre-step voltage with shape ``(protocol,time,CV)``."""
    if num_steps < 1 or num_steps > N_STEPS:
        raise ValueError(f"num_steps must lie within [1, {N_STEPS}].")
    cell.reset_state()

    def step(time_ms):
        voltage = cell.V.value.to_decimal(u.mV)
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return voltage

    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        times = jnp.arange(num_steps, dtype=jnp.float64) * DT_MS
        return jnp.moveaxis(brainstate.transform.for_loop(step, times), 0, 1)


def calibrate_step_waveforms() -> StimulusCalibration:
    """Select feature amplitudes from two batched soma-injection sweeps."""
    negative_grid = np.linspace(-0.8, 0.0, 321, dtype=np.float64)
    positive_grid = np.linspace(0.0, 0.8, 801, dtype=np.float64)
    negative_voltage = _simulate_step_sweep(negative_grid)
    positive_voltage = _simulate_step_sweep(positive_grid)
    negative_baseline = np.mean(negative_voltage[:, BASELINE_STOP - 200 : BASELINE_STOP, 0], axis=1)
    positive_baseline = np.mean(positive_voltage[:, BASELINE_STOP - 200 : BASELINE_STOP, 0], axis=1)
    negative_steady = np.mean(negative_voltage[:, STIMULUS_STOP - 200 : STIMULUS_STOP, 0], axis=1)
    positive_steady = np.mean(positive_voltage[:, STIMULUS_STOP - 200 : STIMULUS_STOP, 0], axis=1)
    mild_index = int(np.argmin(np.abs((negative_steady - negative_baseline) + 5.0)))
    moderate_index = int(np.argmin(np.abs((negative_steady - negative_baseline) + 10.0)))
    counts = upward_crossing_counts(positive_voltage[:, BASELINE_STOP:STIMULUS_STOP, 0])
    silent = np.flatnonzero(counts == 0)
    if silent.size == 0:
        raise RuntimeError("Positive calibration sweep did not contain a subthreshold interval.")
    small_positive_index = int(silent[np.argmin(np.abs((positive_steady[silent] - positive_baseline[silent]) - 5.0))])
    spiking = np.flatnonzero(counts > 0)
    if spiking.size == 0:
        raise RuntimeError("Positive calibration sweep did not reach rheobase.")
    rheobase = float(positive_grid[int(spiking[0])])
    intervals = {}
    for count in range(1, 5):
        matching = np.flatnonzero(counts == count)
        if matching.size == 0:
            raise RuntimeError(f"Positive calibration found no {count}-spike interval.")
        lo, hi = _largest_contiguous_interval(matching)
        intervals[count] = (float(positive_grid[lo]), float(positive_grid[hi]))
    return StimulusCalibration(
        mild_negative_na=float(negative_grid[mild_index]),
        moderate_negative_na=float(negative_grid[moderate_index]),
        small_positive_na=float(positive_grid[small_positive_index]),
        rheobase_na=rheobase,
        spike_intervals_na=intervals,
        prmls_amplitude_na=np.nan,
        prmls_adjustments=0,
    )


def _simulate_step_sweep(amplitudes_na: np.ndarray) -> np.ndarray:
    currents = np.zeros((len(amplitudes_na), N_STEPS, len(SITES)), dtype=np.float64)
    currents[:, BASELINE_STOP:STIMULUS_STOP, 0] = amplitudes_na[:, None]
    return np.asarray(simulate_voltage(build_cell(currents, trainable=False)))


def build_base_waveforms(calibration: StimulusCalibration) -> tuple[tuple[BaseWaveform, ...], StimulusCalibration]:
    """Build feature steps and globally subthreshold multilevel sequences."""
    step_specs = (
        ("step_mild_negative", "passive_hyperpolarizing", "train", calibration.mild_negative_na),
        ("step_small_positive", "passive_depolarizing", "train", calibration.small_positive_na),
        ("step_near_rheobase", "threshold_margin", "train", 0.9 * calibration.rheobase_na),
        ("step_1_spike", "firing_rate_1", "train", _interval_midpoint(calibration.spike_intervals_na[1])),
        ("step_3_spike", "firing_rate_3", "train", _interval_midpoint(calibration.spike_intervals_na[3])),
        ("step_moderate_negative", "passive_holdout", "validation", calibration.moderate_negative_na),
        ("step_2_spike", "firing_rate_2", "validation", _interval_midpoint(calibration.spike_intervals_na[2])),
        ("step_4_spike", "firing_rate_4", "test", _interval_midpoint(calibration.spike_intervals_na[4])),
    )
    steps = tuple(
        BaseWaveform(
            waveform_id=waveform_id,
            family="step",
            feature=feature,
            split=split,
            current_na=_step_current(amplitude),
            amplitude_na=amplitude,
        )
        for waveform_id, feature, split, amplitude in step_specs
    )
    normalized = _normalized_prmls_waveforms()
    amplitude = 0.8 * calibration.rheobase_na
    adjustments = 0
    while True:
        candidates = tuple(_scaled_prmls(item, amplitude) for item in normalized)
        protocols, current = cross_locations(candidates)
        voltage = np.asarray(simulate_voltage(build_cell(current, trainable=False)))
        if not np.any(upward_crossing_counts(voltage[..., 0]) > 0):
            break
        amplitude *= 0.9
        adjustments += 1
        if adjustments > 20:
            raise RuntimeError("Unable to find one globally subthreshold PRMLS amplitude.")
    del protocols
    resolved = StimulusCalibration(
        mild_negative_na=calibration.mild_negative_na,
        moderate_negative_na=calibration.moderate_negative_na,
        small_positive_na=calibration.small_positive_na,
        rheobase_na=calibration.rheobase_na,
        spike_intervals_na=calibration.spike_intervals_na,
        prmls_amplitude_na=amplitude,
        prmls_adjustments=adjustments,
    )
    return steps + candidates, resolved


def _normalized_prmls_waveforms() -> tuple[BaseWaveform, ...]:
    specs = tuple(("train", seed) for seed in (0, 1)) + (("validation", 2), ("test", 3))
    outputs = []
    for split, seed in specs:
        for clock_ms in PRMLS_CLOCKS_MS:
            outputs.append(
                BaseWaveform(
                    waveform_id=f"prmls_{split}_seed{seed}_clock{clock_ms:g}ms",
                    family="prmls",
                    feature="broadband_subthreshold",
                    split=split,
                    current_na=_balanced_prmls(seed=seed, clock_ms=clock_ms),
                    clock_ms=clock_ms,
                    seed=seed,
                    amplitude_na=1.0,
                )
            )
    return tuple(outputs)


def _balanced_prmls(*, seed: int, clock_ms: float) -> np.ndarray:
    symbol_steps = int(round(clock_ms / DT_MS))
    if not np.isclose(symbol_steps * DT_MS, clock_ms, rtol=0.0, atol=1e-12):
        raise ValueError(f"clock_ms={clock_ms!r} is not aligned to dt={DT_MS!r}.")
    stimulus_steps = STIMULUS_STOP - BASELINE_STOP
    if stimulus_steps % symbol_steps:
        raise ValueError(f"clock_ms={clock_ms!r} does not divide the stimulus window.")
    n_symbol = stimulus_steps // symbol_steps
    levels = np.resize(PRMLS_LEVELS, n_symbol)
    random = brainstate.random.RandomState(seed * 1000 + int(round(clock_ms * 10)))
    order = np.asarray(random.permutation(n_symbol))
    symbols = levels[order]
    waveform = np.zeros((N_STEPS,), dtype=np.float64)
    waveform[BASELINE_STOP:STIMULUS_STOP] = np.repeat(symbols, symbol_steps)
    return waveform


def _scaled_prmls(waveform: BaseWaveform, amplitude: float) -> BaseWaveform:
    return BaseWaveform(
        waveform_id=waveform.waveform_id,
        family=waveform.family,
        feature=waveform.feature,
        split=waveform.split,
        current_na=waveform.current_na * amplitude,
        clock_ms=waveform.clock_ms,
        seed=waveform.seed,
        amplitude_na=amplitude,
    )


def _step_current(amplitude_na: float) -> np.ndarray:
    waveform = np.zeros((N_STEPS,), dtype=np.float64)
    waveform[BASELINE_STOP:STIMULUS_STOP] = amplitude_na
    return waveform


def cross_locations(waveforms: tuple[BaseWaveform, ...]) -> tuple[tuple[StimulusProtocol, ...], np.ndarray]:
    """Apply every base waveform unchanged at all three locations."""
    protocols = []
    currents = []
    for waveform in waveforms:
        for site_index, site in enumerate(SITES):
            current = np.zeros((N_STEPS, len(SITES)), dtype=np.float64)
            current[:, site_index] = waveform.current_na
            currents.append(current)
            protocols.append(
                StimulusProtocol(
                    protocol_id=f"{waveform.waveform_id}__at_{site}",
                    waveform_id=waveform.waveform_id,
                    family=waveform.family,
                    feature=waveform.feature,
                    split=waveform.split,
                    injection_site=site,
                    clock_ms=waveform.clock_ms,
                    seed=waveform.seed,
                    amplitude_na=waveform.amplitude_na,
                )
            )
    return tuple(protocols), np.stack(currents)


def generate_dataset(output_dir: Path = ARTIFACT_ROOT / "dataset") -> StimulusDataset:
    """Calibrate and write the complete 60-protocol target dataset."""
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        calibration = calibrate_step_waveforms()
        waveforms, calibration = build_base_waveforms(calibration)
        protocols, current = cross_locations(waveforms)
        voltage = np.asarray(simulate_voltage(build_cell(current, trainable=False)), dtype=np.float64)
    spike_counts = upward_crossing_counts(voltage[..., 0]).astype(np.int16)
    result = StimulusDataset(
        protocols=protocols,
        time_ms=np.arange(N_STEPS, dtype=np.float64) * DT_MS,
        current_na=current,
        voltage_mv=voltage,
        target_spike_counts=spike_counts,
        calibration=calibration,
    )
    _validate_dataset(result)
    write_dataset(output_dir, result)
    return result


def write_dataset(output_dir: Path, dataset: StimulusDataset) -> None:
    """Persist numeric truth, protocol catalog, plots, and configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        time_ms=dataset.time_ms,
        current_nA=dataset.current_na,
        voltage_mV=dataset.voltage_mv,
        target_spike_counts=dataset.target_spike_counts,
        protocol_id=np.asarray([item.protocol_id for item in dataset.protocols]),
        waveform_id=np.asarray([item.waveform_id for item in dataset.protocols]),
        family=np.asarray([item.family for item in dataset.protocols]),
        feature=np.asarray([item.feature for item in dataset.protocols]),
        split=np.asarray([item.split for item in dataset.protocols]),
        injection_site=np.asarray([item.injection_site for item in dataset.protocols]),
        clock_ms=np.asarray([np.nan if item.clock_ms is None else item.clock_ms for item in dataset.protocols]),
        seed=np.asarray([-1 if item.seed is None else item.seed for item in dataset.protocols], dtype=np.int32),
        amplitude_nA=np.asarray(
            [np.nan if item.amplitude_na is None else item.amplitude_na for item in dataset.protocols]
        ),
        target_soma_conductances_mS_per_cm2=SOMA_CONDUCTANCES,
        target_dend_conductances_mS_per_cm2=DEND_CONDUCTANCES,
    )
    rows = [
        {
            "index": index,
            "protocol_id": item.protocol_id,
            "waveform_id": item.waveform_id,
            "family": item.family,
            "feature": item.feature,
            "split": item.split,
            "injection_site": item.injection_site,
            "clock_ms": item.clock_ms,
            "seed": item.seed,
            "amplitude_nA": item.amplitude_na,
            "target_soma_spike_count": int(dataset.target_spike_counts[index]),
        }
        for index, item in enumerate(dataset.protocols)
    ]
    with (output_dir / "protocol_catalog.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "num_steps": N_STEPS,
        "n_cv": 7,
        "parameter_names": list(PARAMETER_NAMES),
        "protocol_count": len(dataset.protocols),
        "split_counts": {split: int(dataset.indices(split).size) for split in SPLITS},
        "target_soma_conductances_mS_per_cm2": SOMA_CONDUCTANCES.tolist(),
        "target_dend_conductances_mS_per_cm2": DEND_CONDUCTANCES.tolist(),
        "target_spike_histogram": {
            str(count): int(np.sum(dataset.target_spike_counts == count))
            for count in range(int(dataset.target_spike_counts.max()) + 1)
        },
        "family_counts": {
            family: sum(item.family == family for item in dataset.protocols) for family in ("step", "prmls")
        },
        "site_counts": {site: sum(item.injection_site == site for item in dataset.protocols) for site in SITES},
        "calibration": {
            "mild_negative_nA": dataset.calibration.mild_negative_na,
            "moderate_negative_nA": dataset.calibration.moderate_negative_na,
            "small_positive_nA": dataset.calibration.small_positive_na,
            "rheobase_nA": dataset.calibration.rheobase_na,
            "spike_intervals_nA": dataset.calibration.spike_intervals_na,
            "prmls_amplitude_nA": dataset.calibration.prmls_amplitude_na,
            "prmls_adjustments": dataset.calibration.prmls_adjustments,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_dataset(output_dir / "dataset_overview.png", dataset)
    _plot_currents(output_dir / "current_waveforms.png", dataset)
    _plot_spatial_examples(output_dir / "spatial_examples.png", dataset)


def load_dataset(output_dir: Path = ARTIFACT_ROOT / "dataset") -> StimulusDataset:
    """Load the complete dataset and catalog."""
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "dataset.npz") as values:
        protocols = tuple(
            StimulusProtocol(
                protocol_id=str(protocol_id),
                waveform_id=str(waveform_id),
                family=str(family),
                feature=str(feature),
                split=str(split),
                injection_site=str(site),
                clock_ms=None if np.isnan(clock_ms) else float(clock_ms),
                seed=None if int(seed) < 0 else int(seed),
                amplitude_na=None if np.isnan(amplitude) else float(amplitude),
            )
            for protocol_id, waveform_id, family, feature, split, site, clock_ms, seed, amplitude in zip(
                values["protocol_id"],
                values["waveform_id"],
                values["family"],
                values["feature"],
                values["split"],
                values["injection_site"],
                values["clock_ms"],
                values["seed"],
                values["amplitude_nA"],
            )
        )
        calibration_values = summary["calibration"]
        calibration = StimulusCalibration(
            mild_negative_na=calibration_values["mild_negative_nA"],
            moderate_negative_na=calibration_values["moderate_negative_nA"],
            small_positive_na=calibration_values["small_positive_nA"],
            rheobase_na=calibration_values["rheobase_nA"],
            spike_intervals_na={
                int(key): tuple(value) for key, value in calibration_values["spike_intervals_nA"].items()
            },
            prmls_amplitude_na=calibration_values["prmls_amplitude_nA"],
            prmls_adjustments=calibration_values["prmls_adjustments"],
        )
        dataset = StimulusDataset(
            protocols=protocols,
            time_ms=np.asarray(values["time_ms"]),
            current_na=np.asarray(values["current_nA"]),
            voltage_mv=np.asarray(values["voltage_mV"]),
            target_spike_counts=np.asarray(values["target_spike_counts"]),
            calibration=calibration,
        )
    _validate_dataset(dataset)
    return dataset


def upward_crossing_counts(voltage_mv, *, threshold_mv: float = 0.0):
    values = np.asarray(voltage_mv)
    return np.sum((values[..., :-1] < threshold_mv) & (values[..., 1:] >= threshold_mv), axis=-1)


def _validate_dataset(dataset: StimulusDataset) -> None:
    if len(dataset.protocols) != 60:
        raise ValueError(f"Expected 60 protocols, got {len(dataset.protocols)}.")
    if dataset.current_na.shape != (60, N_STEPS, 3) or dataset.voltage_mv.shape != (60, N_STEPS, 7):
        raise ValueError(
            f"Unexpected dataset shapes current={dataset.current_na.shape!r}, voltage={dataset.voltage_mv.shape!r}."
        )
    expected = {"train": 33, "validation": 15, "test": 12}
    actual = {split: int(dataset.indices(split).size) for split in SPLITS}
    if actual != expected:
        raise ValueError(f"Unexpected split counts {actual!r}.")
    if not np.all(np.isfinite(dataset.current_na)) or not np.all(np.isfinite(dataset.voltage_mv)):
        raise ValueError("Dataset contains non-finite arrays.")
    if np.any(dataset.current_na[:, :BASELINE_STOP] != 0.0) or np.any(dataset.current_na[:, STIMULUS_STOP:] != 0.0):
        raise ValueError("Current must be zero in baseline and recovery windows.")
    for waveform_id in {item.waveform_id for item in dataset.protocols}:
        indices = [index for index, item in enumerate(dataset.protocols) if item.waveform_id == waveform_id]
        active = [
            dataset.current_na[index, :, SITES.index(dataset.protocols[index].injection_site)] for index in indices
        ]
        np.testing.assert_array_equal(active[0], active[1])
        np.testing.assert_array_equal(active[0], active[2])


def _plot_dataset(path: Path, dataset: StimulusDataset) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(15.0, 10.0), sharex=True, constrained_layout=True)
    for row, split in enumerate(SPLITS):
        for col, site in enumerate(SITES):
            axis = axes[row, col]
            indices = [
                index
                for index, item in enumerate(dataset.protocols)
                if item.split == split and item.injection_site == site
            ]
            for index in indices:
                item = dataset.protocols[index]
                axis.plot(dataset.time_ms, dataset.voltage_mv[index, :, 0], linewidth=0.9, label=item.waveform_id)
            axis.set_title(f"{split}: inject {site}")
            axis.grid(True)
            if row == 2:
                axis.set_xlabel("time (ms)")
            if col == 0:
                axis.set_ylabel("soma V (mV)")
            if len(indices) <= 5:
                axis.legend(frameon=False, fontsize=7)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_currents(path: Path, dataset: StimulusDataset) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True, constrained_layout=True)
    for axis, split in zip(axes, SPLITS):
        seen = set()
        for index, item in enumerate(dataset.protocols):
            if item.split != split or item.waveform_id in seen:
                continue
            seen.add(item.waveform_id)
            site_index = SITES.index(item.injection_site)
            axis.plot(dataset.time_ms, dataset.current_na[index, :, site_index], linewidth=1.0, label=item.waveform_id)
        axis.set(title=f"{split} base waveforms", ylabel="current (nA)")
        axis.grid(True)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("time (ms)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_spatial_examples(path: Path, dataset: StimulusDataset) -> None:
    import matplotlib.pyplot as plt

    examples = ("step_mild_negative", "step_near_rheobase", "prmls_train_seed0_clock2ms")
    figure, axes = plt.subplots(3, 3, figsize=(15.0, 10.0), sharex=True, sharey=True, constrained_layout=True)
    for row, waveform_id in enumerate(examples):
        for col, site in enumerate(SITES):
            axis = axes[row, col]
            index = next(
                index
                for index, item in enumerate(dataset.protocols)
                if item.waveform_id == waveform_id and item.injection_site == site
            )
            for cv in range(dataset.voltage_mv.shape[-1]):
                axis.plot(dataset.time_ms, dataset.voltage_mv[index, :, cv], linewidth=0.9, label=f"CV {cv}")
            axis.set_title(f"{waveform_id}: inject {site}")
            axis.grid(True)
            if row == 2:
                axis.set_xlabel("time (ms)")
            if col == 0:
                axis.set_ylabel("V (mV)")
            if row == 0 and col == 2:
                axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _largest_contiguous_interval(indices: np.ndarray) -> tuple[int, int]:
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((np.asarray([0]), breaks + 1))
    stops = np.concatenate((breaks + 1, np.asarray([indices.size])))
    winner = int(np.argmax(stops - starts))
    run = indices[starts[winner] : stops[winner]]
    return int(run[0]), int(run[-1])


def _interval_midpoint(interval: tuple[float, float]) -> float:
    return 0.5 * (interval[0] + interval[1])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT / "dataset")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    result = generate_dataset(args.output_dir)
    print(args.output_dir)
    print({split: int(result.indices(split).size) for split in SPLITS})


if __name__ == "__main__":
    main()
