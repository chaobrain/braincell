#!/usr/bin/env python3
"""Generate the heterogeneous three-compartment protocol dataset.

The model, calibration, batching, and artifact contract is documented in
``docs/specs/2026-08-17-heterogeneous-protocol-dataset.md``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Literal

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import braincell
from braincell import mech
from braincell.filter import AllRegion, BranchSlice, at

import conductance_learning_core as core


DT = 0.025 * u.ms
DURATION = 100.0 * u.ms
DT_MS = float(DT.to_decimal(u.ms))
N_STEPS = int(round(float(DURATION.to_decimal(u.ms)) / DT_MS))
TIME_MS = np.arange(N_STEPS, dtype=float) * DT_MS
SITES = ("soma", "dend_a", "dend_b")
FAMILIES = ("dc", "paired", "sine")
PROBES = SITES
CONDUCTANCE_UNIT = u.mS / u.cm**2
TARGET_PARAMETERS = np.asarray(
    (
        (0.60, 120.0, 36.0),
        (0.48, 96.0, 28.8),
        (0.42, 84.0, 25.2),
    ),
    dtype=float,
)
ARTIFACT_ROOT = Path(__file__).resolve().parent / "plot"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "heterogeneous_protocol_dataset"
EVOKED_START_INDEX = int(round(20.0 / DT_MS))
STIMULUS_STOP_INDEX = int(round(80.0 / DT_MS))
SPIKE_THRESHOLD_MV = 0.0


@dataclass(frozen=True)
class Protocol:
    """Describe one deterministic current-injection protocol."""

    protocol_id: str
    family: Literal["dc", "paired", "sine"]
    injection_site: Literal["soma", "dend_a", "dend_b"]
    split: Literal["train", "validation", "test"]
    amplitudes_na: tuple[float, ...]
    frequency_hz: float = 0.0
    phase_rad: float = 0.0
    calibration_label: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown protocol family {self.family!r}.")
        if self.injection_site not in SITES:
            raise ValueError(f"Unknown injection site {self.injection_site!r}.")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown dataset split {self.split!r}.")
        expected = {"dc": 1, "paired": 2, "sine": 1}[self.family]
        if len(self.amplitudes_na) != expected:
            raise ValueError(f"{self.family} requires {expected} amplitude value(s).")
        values = np.asarray(self.amplitudes_na + (self.frequency_hz, self.phase_rad), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Protocol values must be finite.")
        if self.family == "sine" and self.frequency_hz <= 0.0:
            raise ValueError("Sine protocols require a positive frequency.")


@dataclass(frozen=True)
class Calibration:
    """Store site-specific negative levels and positive spike intervals."""

    negative_amplitudes_na: dict[str, tuple[float, ...]]
    positive_intervals_na: dict[str, dict[int, tuple[float, float]]]
    negative_soma_minima_mv: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class BatchResult:
    """Store one family rollout and warm timing information."""

    protocols: tuple[Protocol, ...]
    voltages_mv: np.ndarray
    currents_na: np.ndarray
    compile_seconds: float
    warm_seconds: float


def build_morphology() -> braincell.Morphology:
    """Build the asymmetric three-compartment morphology."""
    soma = braincell.Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[12.5, 12.5] * u.um,
        type="soma",
    )
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
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend_a = dend_a
    morphology.soma.dend_b = dend_b
    return morphology


def _branch_region(index: int) -> BranchSlice:
    return BranchSlice(branch_index=index, prox=0.0, dist=1.0)


def build_cell(
    protocols: tuple[Protocol, ...],
    *,
    trainable: bool = False,
    probe_names: tuple[str, ...] = PROBES,
) -> braincell.Cell:
    """Build one population cell whose rows receive different protocols."""
    if not protocols:
        raise ValueError("At least one protocol is required.")
    family = protocols[0].family
    if any(protocol.family != family for protocol in protocols):
        raise ValueError("One population batch may contain only one protocol family.")

    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranch(),
        V_init=-65.0 * u.mV,
        solver="staggered",
        pop_size=(len(protocols),),
    )
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
    )
    channel_prefix = "ExplorationTrainable" if trainable else "ExplorationFrozen"
    for index, site in enumerate(SITES):
        leak, sodium, potassium = TARGET_PARAMETERS[index]
        cell.paint(
            _branch_region(index),
            mech.Channel(
                f"{channel_prefix}Na",
                name=f"{site}_na",
                g_max=sodium * CONDUCTANCE_UNIT,
            ),
            mech.Channel(
                f"{channel_prefix}K",
                name=f"{site}_k",
                g_max=potassium * CONDUCTANCE_UNIT,
            ),
            mech.Channel(
                f"{channel_prefix}Leak",
                name=f"{site}_leak",
                g_max=leak * CONDUCTANCE_UNIT,
                E=-54.387 * u.mV,
            ),
        )

    _place_protocol_clamps(cell, protocols)
    for name in probe_names:
        if name not in SITES:
            raise ValueError(f"Unknown probe name {name!r}.")
        cell.place(at(name, 0.5), mech.StateProbe(name=f"{name}_v", field="v"))
    cell.init_state()
    return cell


def _place_protocol_clamps(cell: braincell.Cell, protocols: tuple[Protocol, ...]) -> None:
    family = protocols[0].family
    n_protocol = len(protocols)
    if family == "dc":
        for site in SITES:
            active = np.asarray(
                [protocol.amplitudes_na[0] if protocol.injection_site == site else 0.0 for protocol in protocols]
            )
            amplitudes = np.stack((np.zeros(n_protocol), active, np.zeros(n_protocol)), axis=-1)
            cell.place(
                at(site, 0.5),
                mech.CurrentClamp(
                    durations=np.asarray([20.0, 60.0, 20.0]) * u.ms,
                    amplitudes=amplitudes * u.nA,
                ),
            )
        return
    if family == "paired":
        for site in SITES:
            first = np.asarray(
                [protocol.amplitudes_na[0] if protocol.injection_site == site else 0.0 for protocol in protocols]
            )
            second = np.asarray(
                [protocol.amplitudes_na[1] if protocol.injection_site == site else 0.0 for protocol in protocols]
            )
            amplitudes = np.stack(
                (np.zeros(n_protocol), first, np.zeros(n_protocol), second, np.zeros(n_protocol)),
                axis=-1,
            )
            cell.place(
                at(site, 0.5),
                mech.CurrentClamp(
                    durations=np.asarray([20.0] * 5) * u.ms,
                    amplitudes=amplitudes * u.nA,
                ),
            )
        return

    for site in SITES:
        amplitude = np.asarray(
            [protocol.amplitudes_na[0] if protocol.injection_site == site else 0.0 for protocol in protocols]
        )
        frequency = np.asarray([protocol.frequency_hz for protocol in protocols])
        phase = np.asarray([protocol.phase_rad for protocol in protocols])
        cell.place(
            at(site, 0.5),
            mech.SineClamp(
                amplitude=amplitude * u.nA,
                frequency=frequency * u.Hz,
                phase=phase,
                offset=np.zeros(n_protocol) * u.nA,
                delay=20.0 * u.ms,
                duration=60.0 * u.ms,
            ),
        )


def find_conductance_parameters(cell: braincell.Cell) -> dict[str, object]:
    """Return the nine independent regional conductance parameters."""
    found: dict[str, object] = {}
    expected = {f"{site}_{channel}" for site in SITES for channel in ("leak", "na", "k")}
    for layout in cell.layouts:
        if layout.target != "density":
            continue
        declaration = cell.runtime.get_layout_mechanism(layout.id)
        name = getattr(declaration, "instance_name", None)
        node = cell.get_runtime_node(layout.id)
        if name in expected and hasattr(node, "g_max"):
            found[name] = node.g_max
    missing = expected.difference(found)
    if missing:
        raise LookupError(f"Missing conductance parameters: {sorted(missing)!r}.")
    return {name: found[name] for name in sorted(found)}


def protocol_currents(protocols: tuple[Protocol, ...]) -> np.ndarray:
    """Return exact three-site current waveforms for a protocol batch."""
    currents = np.zeros((len(protocols), N_STEPS, len(SITES)), dtype=np.float32)
    active = (TIME_MS >= 20.0) & (TIME_MS < 80.0)
    first = (TIME_MS >= 20.0) & (TIME_MS < 40.0)
    second = (TIME_MS >= 60.0) & (TIME_MS < 80.0)
    local_seconds = (TIME_MS - 20.0) / 1000.0
    for row, protocol in enumerate(protocols):
        site_index = SITES.index(protocol.injection_site)
        if protocol.family == "dc":
            currents[row, active, site_index] = protocol.amplitudes_na[0]
        elif protocol.family == "paired":
            currents[row, first, site_index] = protocol.amplitudes_na[0]
            currents[row, second, site_index] = protocol.amplitudes_na[1]
        else:
            angle = 2.0 * np.pi * protocol.frequency_hz * local_seconds + protocol.phase_rad
            currents[row, active, site_index] = protocol.amplitudes_na[0] * np.sin(angle[active])
    return currents


def _extract_voltages(result, probe_names: tuple[str, ...]) -> np.ndarray:
    values = [np.asarray(result.traces[f"{name}_v"].to_decimal(u.mV)) for name in probe_names]
    stacked = np.stack(values, axis=-1)
    if stacked.shape[0] != N_STEPS:
        raise ValueError(f"Expected time-leading traces, got shape {stacked.shape!r}.")
    return np.swapaxes(stacked, 0, 1).astype(np.float32, copy=False)


def simulate_batch(
    protocols: tuple[Protocol, ...],
    *,
    probe_names: tuple[str, ...] = PROBES,
    benchmark: bool = False,
) -> BatchResult:
    """Compile and run one population batch, optionally timing a warm rerun."""
    cell = build_cell(protocols, trainable=False, probe_names=probe_names)
    cell.reset_state()
    started = time.perf_counter()
    first_result = cell.run(dt=DT, duration=DURATION)
    first_voltages = _extract_voltages(first_result, probe_names)
    compile_seconds = time.perf_counter() - started

    warm_seconds = compile_seconds
    voltages = first_voltages
    if benchmark:
        cell.reset_state()
        started = time.perf_counter()
        warm_result = cell.run(dt=DT, duration=DURATION)
        voltages = _extract_voltages(warm_result, probe_names)
        warm_seconds = time.perf_counter() - started
        np.testing.assert_allclose(voltages, first_voltages, rtol=2e-5, atol=2e-4)

    return BatchResult(
        protocols=protocols,
        voltages_mv=voltages,
        currents_na=protocol_currents(protocols),
        compile_seconds=compile_seconds,
        warm_seconds=warm_seconds,
    )


def spike_mask(voltages_mv: np.ndarray) -> np.ndarray:
    """Return soma upward zero-crossings for a protocol-by-time array."""
    if voltages_mv.ndim != 2:
        raise ValueError(f"voltages_mv must have shape (protocol, time), got {voltages_mv.shape!r}.")
    crossings = np.zeros(voltages_mv.shape, dtype=bool)
    crossings[:, 1:] = (voltages_mv[:, :-1] < SPIKE_THRESHOLD_MV) & (voltages_mv[:, 1:] >= SPIKE_THRESHOLD_MV)
    return crossings


def _calibration_protocols(amplitudes: np.ndarray, *, prefix: str) -> tuple[Protocol, ...]:
    return tuple(
        Protocol(
            protocol_id=f"{prefix}_{site}_{index:04d}",
            family="dc",
            injection_site=site,
            split="train",
            amplitudes_na=(float(amplitude),),
            calibration_label=prefix,
        )
        for site in SITES
        for index, amplitude in enumerate(amplitudes)
    )


def _largest_contiguous_interval(indices: np.ndarray) -> tuple[int, int]:
    if indices.size == 0:
        raise ValueError("Cannot select an interval from an empty index set.")
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((np.asarray([0]), breaks + 1))
    stops = np.concatenate((breaks + 1, np.asarray([indices.size])))
    lengths = stops - starts
    winner = int(np.argmax(lengths))
    run = indices[starts[winner] : stops[winner]]
    return int(run[0]), int(run[-1])


def calibrate_protocol_levels() -> Calibration:
    """Calibrate DC levels in two compiled population sweeps."""
    positive_grid = np.linspace(0.0, 0.60, 601, dtype=float)
    positive_protocols = _calibration_protocols(positive_grid, prefix="positive")
    positive_result = simulate_batch(positive_protocols, probe_names=("soma",))
    positive_crossings = spike_mask(positive_result.voltages_mv[:, :, 0])
    positive_counts = positive_crossings[:, EVOKED_START_INDEX:].sum(axis=1)

    negative_grid = np.linspace(-2.0, 0.0, 401, dtype=float)
    negative_protocols = _calibration_protocols(negative_grid, prefix="negative")
    negative_result = simulate_batch(negative_protocols, probe_names=("soma",))
    negative_minima = negative_result.voltages_mv[:, EVOKED_START_INDEX:STIMULUS_STOP_INDEX, 0].min(axis=1)

    positive_intervals: dict[str, dict[int, tuple[float, float]]] = {}
    negative_amplitudes: dict[str, tuple[float, ...]] = {}
    selected_minima: dict[str, tuple[float, ...]] = {}
    positive_stride = positive_grid.size
    negative_stride = negative_grid.size
    target_minima = np.asarray([-70.0, -80.0, -90.0, -100.0, -110.0, -120.0])
    for site_index, site in enumerate(SITES):
        site_counts = positive_counts[site_index * positive_stride : (site_index + 1) * positive_stride]
        intervals: dict[int, tuple[float, float]] = {}
        for count in range(6):
            matching = np.flatnonzero(site_counts == count)
            lo_index, hi_index = _largest_contiguous_interval(matching)
            intervals[count] = (float(positive_grid[lo_index]), float(positive_grid[hi_index]))
        positive_intervals[site] = intervals

        minima = negative_minima[site_index * negative_stride : (site_index + 1) * negative_stride]
        selected_indices = [int(np.argmin(np.abs(minima - target))) for target in target_minima]
        negative_amplitudes[site] = tuple(float(negative_grid[index]) for index in selected_indices)
        selected_minima[site] = tuple(float(minima[index]) for index in selected_indices)

    return Calibration(
        negative_amplitudes_na=negative_amplitudes,
        positive_intervals_na=positive_intervals,
        negative_soma_minima_mv=selected_minima,
    )


def _interval_value(interval: tuple[float, float], fraction: float) -> float:
    return float(interval[0] + fraction * (interval[1] - interval[0]))


def build_protocol_catalog(calibration: Calibration) -> tuple[Protocol, ...]:
    """Build the deterministic 144-row protocol catalog."""
    dc_protocols: list[Protocol] = []
    paired_protocols: list[Protocol] = []
    sine_protocols: list[Protocol] = []
    for site in SITES:
        negative = calibration.negative_amplitudes_na[site]
        intervals = calibration.positive_intervals_na[site]
        for index, amplitude in enumerate(negative):
            split = "validation" if index == 1 else "test" if index == 4 else "train"
            dc_protocols.append(
                Protocol(
                    protocol_id=f"dc_{site}_negative_{index}",
                    family="dc",
                    injection_site=site,
                    split=split,
                    amplitudes_na=(amplitude,),
                    calibration_label=f"soma_min_{70 + 10 * index}mV",
                )
            )
        for count in range(6):
            for quantile_index, fraction in enumerate((0.25, 0.50, 0.75)):
                split = "train"
                if (count, quantile_index) in {(1, 1), (4, 1)}:
                    split = "validation"
                elif (count, quantile_index) in {(2, 1), (5, 1)}:
                    split = "test"
                dc_protocols.append(
                    Protocol(
                        protocol_id=f"dc_{site}_positive_{count}spike_q{quantile_index + 1}",
                        family="dc",
                        injection_site=site,
                        split=split,
                        amplitudes_na=(_interval_value(intervals[count], fraction),),
                        calibration_label=f"{count}_spike_interval_q{quantile_index + 1}",
                    )
                )

        negative_levels = (negative[1], negative[3])
        positive_levels = tuple(_interval_value(intervals[count], 0.5) for count in (1, 2, 4, 5))
        for order in ("negative_positive", "positive_negative"):
            pair_index = 0
            for negative_index, negative_amplitude in enumerate(negative_levels):
                for positive_index, positive_amplitude in enumerate(positive_levels):
                    split = "validation" if pair_index == 2 else "test" if pair_index == 5 else "train"
                    amplitudes = (
                        (negative_amplitude, positive_amplitude)
                        if order == "negative_positive"
                        else (positive_amplitude, negative_amplitude)
                    )
                    paired_protocols.append(
                        Protocol(
                            protocol_id=f"paired_{site}_{order}_n{negative_index}_p{positive_index}",
                            family="paired",
                            injection_site=site,
                            split=split,
                            amplitudes_na=amplitudes,
                            calibration_label=order,
                        )
                    )
                    pair_index += 1

        for frequency in (10.0, 40.0):
            for count in (0, 1, 3, 5):
                split = (
                    "validation"
                    if (frequency, count) == (10.0, 3)
                    else "test"
                    if (
                        frequency,
                        count,
                    )
                    == (40.0, 3)
                    else "train"
                )
                phase = -np.pi * frequency * 0.060
                sine_protocols.append(
                    Protocol(
                        protocol_id=f"sine_{site}_{frequency:g}hz_{count}spike_level",
                        family="sine",
                        injection_site=site,
                        split=split,
                        amplitudes_na=(_interval_value(intervals[count], 0.5),),
                        frequency_hz=frequency,
                        phase_rad=float(phase),
                        calibration_label=f"dc_{count}_spike_midpoint",
                    )
                )
    return tuple(dc_protocols + paired_protocols + sine_protocols)


def _scale_positive(protocol: Protocol, factor: float) -> Protocol:
    if protocol.family == "sine":
        return replace(protocol, amplitudes_na=(protocol.amplitudes_na[0] * factor,))
    if protocol.family == "paired":
        return replace(
            protocol,
            amplitudes_na=tuple(value * factor if value > 0.0 else value for value in protocol.amplitudes_na),
        )
    return protocol


def _select_attenuated_protocols(
    protocols: tuple[Protocol, ...],
    counts: np.ndarray,
) -> tuple[Protocol, ...]:
    invalid = [protocol for protocol, count in zip(protocols, counts) if count > 5 and protocol.family != "dc"]
    if not invalid:
        return protocols
    candidates = tuple(_scale_positive(protocol, 0.9**step) for protocol in invalid for step in range(1, 21))
    selected: dict[str, Protocol] = {}
    for family in ("paired", "sine"):
        family_candidates = tuple(protocol for protocol in candidates if protocol.family == family)
        if not family_candidates:
            continue
        result = simulate_batch(family_candidates, probe_names=("soma",))
        family_counts = spike_mask(result.voltages_mv[:, :, 0])[:, EVOKED_START_INDEX:].sum(axis=1)
        for protocol, count in zip(family_candidates, family_counts):
            if count <= 5 and protocol.protocol_id not in selected:
                selected[protocol.protocol_id] = protocol
    missing = {protocol.protocol_id for protocol in invalid}.difference(selected)
    if missing:
        raise RuntimeError(f"Could not attenuate protocols below the five-spike limit: {sorted(missing)!r}.")
    return tuple(selected.get(protocol.protocol_id, protocol) for protocol in protocols)


def _run_catalog(catalog: tuple[Protocol, ...]) -> tuple[tuple[Protocol, ...], tuple[BatchResult, ...]]:
    dc = tuple(protocol for protocol in catalog if protocol.family == "dc")
    paired = tuple(protocol for protocol in catalog if protocol.family == "paired")
    sine = tuple(protocol for protocol in catalog if protocol.family == "sine")
    dc_result = simulate_batch(dc, benchmark=True)
    paired_result = simulate_batch(paired, benchmark=True)
    sine_result = simulate_batch(sine, benchmark=True)
    results = (dc_result, paired_result, sine_result)
    counts = np.concatenate(
        [spike_mask(result.voltages_mv[:, :, 0])[:, EVOKED_START_INDEX:].sum(axis=1) for result in results]
    )
    adjusted = _select_attenuated_protocols(catalog, counts)
    if adjusted != catalog:
        return _run_catalog(adjusted)
    return catalog, results


def _validate_dataset(
    catalog: tuple[Protocol, ...],
    currents_na: np.ndarray,
    voltages_mv: np.ndarray,
    evoked_counts: np.ndarray,
) -> None:
    if len(catalog) != 144:
        raise ValueError(f"Expected 144 protocols, got {len(catalog)}.")
    if currents_na.shape != (144, N_STEPS, 3) or voltages_mv.shape != (144, N_STEPS, 3):
        raise ValueError(f"Unexpected dataset shapes: current={currents_na.shape}, voltage={voltages_mv.shape}.")
    if not np.all(np.isfinite(currents_na)) or not np.all(np.isfinite(voltages_mv)):
        raise ValueError("Dataset contains non-finite values.")
    if np.any(currents_na[:, :EVOKED_START_INDEX] != 0.0) or np.any(currents_na[:, STIMULUS_STOP_INDEX:] != 0.0):
        raise ValueError("Current is nonzero outside the 20--80 ms stimulus window.")
    if np.any(evoked_counts > 5):
        raise ValueError(f"Evoked spike count exceeds five: {evoked_counts.max()}.")
    split_counts = {
        split: sum(protocol.split == split for protocol in catalog) for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 108, "validation": 18, "test": 18}:
        raise ValueError(f"Unexpected split counts {split_counts!r}.")


def _catalog_rows(catalog: tuple[Protocol, ...], evoked_counts: np.ndarray, initial_counts: np.ndarray) -> list[dict]:
    return [
        {
            "index": index,
            "protocol_id": protocol.protocol_id,
            "family": protocol.family,
            "injection_site": protocol.injection_site,
            "split": protocol.split,
            "amplitudes_nA": ";".join(f"{value:.8g}" for value in protocol.amplitudes_na),
            "frequency_Hz": protocol.frequency_hz,
            "phase_rad": protocol.phase_rad,
            "calibration_label": protocol.calibration_label,
            "initial_spike_count": int(initial_counts[index]),
            "evoked_spike_count": int(evoked_counts[index]),
        }
        for index, protocol in enumerate(catalog)
    ]


def _write_artifacts(
    output_dir: Path,
    calibration: Calibration,
    catalog: tuple[Protocol, ...],
    results: tuple[BatchResult, ...],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    voltages_mv = np.concatenate([result.voltages_mv for result in results], axis=0)
    currents_na = np.concatenate([result.currents_na for result in results], axis=0)
    spikes = spike_mask(voltages_mv[:, :, 0])
    initial_counts = spikes[:, :EVOKED_START_INDEX].sum(axis=1)
    evoked_counts = spikes[:, EVOKED_START_INDEX:].sum(axis=1)
    _validate_dataset(catalog, currents_na, voltages_mv, evoked_counts)

    np.savez_compressed(
        output_dir / "dataset.npz",
        time_ms=TIME_MS.astype(np.float32),
        current_nA=currents_na,
        voltage_mV=voltages_mv,
        spike_mask=spikes,
        initial_spike_counts=initial_counts.astype(np.int16),
        evoked_spike_counts=evoked_counts.astype(np.int16),
        protocol_id=np.asarray([protocol.protocol_id for protocol in catalog]),
        family=np.asarray([protocol.family for protocol in catalog]),
        injection_site=np.asarray([protocol.injection_site for protocol in catalog]),
        split=np.asarray([protocol.split for protocol in catalog]),
        target_parameters_mS_per_cm2=TARGET_PARAMETERS.astype(np.float32),
    )

    rows = _catalog_rows(catalog, evoked_counts, initial_counts)
    with (output_dir / "protocol_catalog.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    performance = {
        result.protocols[0].family: {
            "population_size": len(result.protocols),
            "compile_seconds": result.compile_seconds,
            "warm_seconds": result.warm_seconds,
            "warm_ms_per_protocol": 1000.0 * result.warm_seconds / len(result.protocols),
        }
        for result in results
    }
    with (output_dir / "performance.json").open("w", encoding="utf-8") as stream:
        json.dump(performance, stream, indent=2)

    summary = {
        "protocol_count": len(catalog),
        "shape": {"time": N_STEPS, "current_sites": 3, "voltage_probes": 3},
        "split_counts": {
            split: sum(protocol.split == split for protocol in catalog) for split in ("train", "validation", "test")
        },
        "family_counts": {family: sum(protocol.family == family for protocol in catalog) for family in FAMILIES},
        "site_counts": {site: sum(protocol.injection_site == site for protocol in catalog) for site in SITES},
        "evoked_spike_histogram": {
            str(count): int(np.sum(evoked_counts == count)) for count in range(int(evoked_counts.max()) + 1)
        },
        "maximum_evoked_spikes": int(evoked_counts.max()),
        "target_parameters_mS_per_cm2": TARGET_PARAMETERS.tolist(),
        "calibration": {
            "negative_amplitudes_nA": calibration.negative_amplitudes_na,
            "negative_soma_minima_mV": calibration.negative_soma_minima_mv,
            "positive_intervals_nA": {
                site: {str(count): interval for count, interval in values.items()}
                for site, values in calibration.positive_intervals_na.items()
            },
        },
        "performance": performance,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    _plot_trace_atlases(output_dir, catalog, currents_na, voltages_mv, evoked_counts)
    _plot_coverage(output_dir, catalog, voltages_mv, evoked_counts)
    _plot_performance(output_dir, performance)
    return summary


def _plot_trace_atlases(
    output_dir: Path,
    catalog: tuple[Protocol, ...],
    currents_na: np.ndarray,
    voltages_mv: np.ndarray,
    evoked_counts: np.ndarray,
) -> None:
    trace_dir = output_dir / "trace_atlases"
    trace_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    for site in SITES:
        for family in FAMILIES:
            indices = [
                index
                for index, protocol in enumerate(catalog)
                if protocol.injection_site == site and protocol.family == family
            ]
            n_column = 4
            n_row = int(np.ceil(len(indices) / n_column))
            fig, axes = plt.subplots(n_row, n_column, figsize=(16, 2.55 * n_row), sharex=True, sharey=True)
            axes = np.asarray(axes).reshape(-1)
            for axis, index in zip(axes, indices):
                protocol = catalog[index]
                for probe_index, (probe, color) in enumerate(zip(PROBES, colors)):
                    axis.plot(TIME_MS, voltages_mv[index, :, probe_index], color=color, lw=0.8, label=probe)
                current = currents_na[index, :, SITES.index(site)]
                scale = max(float(np.max(np.abs(current))), 1e-6)
                axis.plot(TIME_MS, -112.0 + 12.0 * current / scale, color="0.35", lw=0.7, alpha=0.8)
                axis.axvspan(20.0, 80.0, color="0.92", zorder=-1)
                axis.set_title(f"{protocol.protocol_id}\nspikes={evoked_counts[index]}, {protocol.split}", fontsize=8)
                axis.set_xlim(0.0, 100.0)
                axis.set_ylim(-125.0, 65.0)
                axis.grid(alpha=0.15)
            for axis in axes[len(indices) :]:
                axis.set_visible(False)
            for axis in axes[-n_column:]:
                if axis.get_visible():
                    axis.set_xlabel("Time (ms)")
            for row in range(n_row):
                axes[row * n_column].set_ylabel("Voltage (mV)")
            handles = [plt.Line2D([], [], color=color, label=name) for name, color in zip(PROBES, colors)]
            handles.append(plt.Line2D([], [], color="0.35", label="scaled current"))
            fig.suptitle(f"{site}: {family} protocols", y=0.997)
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.975),
                ncol=4,
                frameon=False,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
            fig.savefig(trace_dir / f"{site}_{family}_traces.png", dpi=150)
            plt.close(fig)


def _plot_coverage(
    output_dir: Path,
    catalog: tuple[Protocol, ...],
    voltages_mv: np.ndarray,
    evoked_counts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(6)
    width = 0.24
    for family_index, family in enumerate(FAMILIES):
        mask = np.asarray([protocol.family == family for protocol in catalog])
        histogram = np.asarray([np.sum(evoked_counts[mask] == count) for count in x])
        axes[0, 0].bar(x + (family_index - 1) * width, histogram, width=width, label=family)
    axes[0, 0].set(title="Evoked soma spike coverage", xlabel="Spike count", ylabel="Protocols")
    axes[0, 0].legend(frameon=False)

    for site in SITES:
        mask = np.asarray([protocol.injection_site == site for protocol in catalog])
        axes[0, 1].scatter(
            voltages_mv[mask, :, 0].min(axis=1),
            voltages_mv[mask, :, 0].max(axis=1),
            s=18,
            alpha=0.7,
            label=site,
        )
    axes[0, 1].set(title="Soma voltage extrema", xlabel="Minimum (mV)", ylabel="Maximum (mV)")
    axes[0, 1].legend(frameon=False)

    split_names = ("train", "validation", "test")
    bottom = np.zeros(3)
    for family in FAMILIES:
        values = np.asarray(
            [
                sum(protocol.split == split and protocol.family == family for protocol in catalog)
                for split in split_names
            ]
        )
        axes[1, 0].bar(split_names, values, bottom=bottom, label=family)
        bottom += values
    axes[1, 0].set(title="Stratified dataset split", ylabel="Protocols")
    axes[1, 0].legend(frameon=False)

    table = np.asarray(
        [
            [
                sum(protocol.injection_site == site and protocol.family == family for protocol in catalog)
                for family in FAMILIES
            ]
            for site in SITES
        ]
    )
    image = axes[1, 1].imshow(table, cmap="Blues", vmin=0)
    axes[1, 1].set_xticks(np.arange(3), FAMILIES)
    axes[1, 1].set_yticks(np.arange(3), SITES)
    axes[1, 1].set_title("Site/family protocol counts")
    for row in range(3):
        for column in range(3):
            axes[1, 1].text(column, row, str(table[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_coverage.png", dpi=170)
    plt.close(fig)


def _plot_performance(output_dir: Path, performance: dict) -> None:
    families = list(FAMILIES)
    values = [performance[family]["warm_ms_per_protocol"] for family in families]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(families, values, color=("#4c78a8", "#f58518", "#54a24b"))
    axis.bar_label(bars, fmt="%.2f ms")
    axis.set(title="Warm CPU population throughput", ylabel="Milliseconds per protocol")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "population_performance.png", dpi=170)
    plt.close(fig)


def generate_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Calibrate, simulate, validate, and save the full dataset."""
    calibration = calibrate_protocol_levels()
    catalog = build_protocol_catalog(calibration)
    catalog, results = _run_catalog(catalog)
    return _write_artifacts(output_dir, calibration, catalog, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with brainstate.environ.context(dt=DT):
        summary = generate_dataset(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
