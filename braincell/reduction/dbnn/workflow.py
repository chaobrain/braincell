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

"""End-to-end workflows for fitting DBNN-GIF reductions."""

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.filter import at
from braincell.reduction.dbnn.dataset import (
    DatasetBatch,
    TeacherSession,
    TeacherSpec,
    rasterize_events,
    save_dataset,
    validate_dataset,
)
from braincell.reduction.dbnn.functional import compute_metrics
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import DBNN, DBNNGIF, SpikeAlignmentEvidence, align_spike_times
from braincell.reduction.dbnn.stimulus import (
    SparseEvents,
    StimulusPlan,
    combine_protocols,
    generate_multichannel_protocol,
    measure_coverage,
)
from braincell.reduction.dbnn.train import DBNNTrainer, compute_spike_metrics


Acceptance = Callable[[DatasetBatch], Any]
CellFactory = Callable[[Any], Any]
_DEFAULT_OUTPUT_LOCATION = at("soma", 0.5)


def fit_dbnn_gif(
    layout: ChannelLayout,
    *,
    cell_factory: CellFactory | None = None,
    dataset_pool: DatasetBatch | None = None,
    rate_hz: float | np.ndarray | None = None,
    output_location: Any = _DEFAULT_OUTPUT_LOCATION,
    acceptance: Acceptance | None = None,
    train_traces: int = 900,
    validation_traces: int = 50,
    test_traces: int = 50,
    duration_ms: float | None = None,
    dt: Any | None = None,
    teacher_batch_size: int = 512,
    max_teacher_batches: int = 100,
    data_seed: int = 20260828,
    split_seed: int = 42,
    training_seed: int = 42,
    source_fingerprint: str | None = None,
    dynamics_fingerprint: str | None = None,
    spike_threshold_mv: float | None = None,
    epochs: int = 500,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    lr_step_size: int = 100,
    lr_gamma: float = 0.5,
    gradient_clip: float | None = 1.0,
    initialization_search: bool = True,
    initialization_options: Mapping[str, Any] | None = None,
    spike_window_pre_ms: float = 3.0,
    spike_window_post_ms: float = 10.0,
    gif_thresholds_mv: Any | None = None,
    gif_threshold_increments_mv: Any = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
    gif_threshold_taus_ms: Any = (10.0, 30.0, 50.0, 100.0),
    gif_match_window_ms: float = 10.0,
    gif_candidate_batch_size: int = 128,
    mode: str = "f",
    input_sign_mode: str = "channel_type",
    output_dir: str | Path | None = None,
    forbidden_trace_seeds: Any = (),
    reserve_trace_seeds: Callable[[tuple[int, ...]], None] | None = None,
) -> tuple[DBNNGIF, float, float]:
    """Fit a DBNN-GIF reduction from a detailed Cell factory or dataset pool.

    Exactly one of ``cell_factory`` and ``dataset_pool`` must be supplied. In
    generation mode, multichannel Bernoulli traces are generated at ``rate_hz``
    and simulated through one reusable detailed-Cell population. As in the
    canonical SC workflow, a missing channel receives one deterministic event
    so every candidate batch covers the complete layout.
    In pool mode, the supplied traces are validated and split directly without
    constructing the detailed cell.

    Parameters
    ----------
    layout : ChannelLayout
        Logical DBNN channels and their detailed-cell synapse mapping.
    cell_factory : callable, optional
        Callable accepting a population shape such as ``(512,)`` and returning
        an uninitialized detailed Cell with the synapses represented by
        ``layout``.
    dataset_pool : DatasetBatch, optional
        Unified accepted trace pool. Its size must equal the sum of the three
        requested split sizes.
    rate_hz : float or numpy.ndarray, optional
        Bernoulli event rate per channel in generation mode. A scalar is
        broadcast; an array must have one value per layout channel.
    output_location : object, optional
        Detailed-cell voltage location. Defaults to ``at("soma", 0.5)``.
    acceptance : callable, optional
        Callback returning unique integer row indices accepted from each
        generated candidate batch. By default, accept every valid trace.
    train_traces : int, optional
        Number of training traces.
    validation_traces : int, optional
        Number of validation traces.
    test_traces : int, optional
        Number of held-out test traces.
    duration_ms : float, optional
        Generated trace duration. Defaults to 6000 ms in generation mode and
        is inferred from ``dataset_pool`` otherwise.
    dt : brainunit.Quantity, optional
        Sampling interval. Defaults to 1 ms in generation mode and is inferred
        from ``dataset_pool`` otherwise.
    teacher_batch_size : int, optional
        Fixed detailed-cell population size used for candidate batches.
    max_teacher_batches : int, optional
        Maximum generated candidate batches before failing for insufficient
        accepted traces.
    data_seed : int, optional
        Root seed for generated candidate traces.
    split_seed : int, optional
        Seed for deterministic spike-count-stratified pool splitting.
    training_seed : int, optional
        DBNN initialization and minibatch seed.
    source_fingerprint : str, optional
        Stable detailed-cell provenance identifier. Required in generation
        mode; in pool mode it defaults to dataset metadata.
    dynamics_fingerprint : str, optional
        Stable source identity excluding spike-readout ``V_th``. Defaults to
        ``source_fingerprint`` for standalone workflows.
    spike_threshold_mv : float, optional
        Threshold used for teacher spike detection and voltage-fit masking.
        Defaults to pool metadata or -20 mV.
    epochs : int, optional
        Maximum DBNN training epochs.
    batch_size : int, optional
        Trace minibatch size.
    learning_rate : float, optional
        Adam learning rate.
    lr_step_size : int, optional
        StepLR epoch interval; zero disables the scheduler.
    lr_gamma : float, optional
        StepLR multiplicative decay.
    gradient_clip : float or None, optional
        Gradient norm clipping threshold.
    initialization_search : bool, optional
        Whether to search shared kernel parameters before gradient training.
    initialization_options : mapping, optional
        Overrides for :class:`DBNNTrainer` initialization search.
    spike_window_pre_ms : float, optional
        Voltage-fit exclusion duration before teacher spikes.
    spike_window_post_ms : float, optional
        Voltage-fit exclusion duration after teacher spikes.
    gif_thresholds_mv : array-like, optional
        GIF threshold candidates. Defaults to percentiles of validation drive.
    gif_threshold_increments_mv : array-like, optional
        GIF adaptive-threshold increment candidates.
    gif_threshold_taus_ms : array-like, optional
        GIF adaptive-threshold time-constant candidates.
    gif_match_window_ms : float, optional
        Spike matching window used during GIF calibration.
    gif_candidate_batch_size : int, optional
        Number of GIF candidates evaluated together.
    mode : {"f", "r"}, optional
        DBNN complete-sequence backend.
    input_sign_mode : {"none", "channel_type"}, optional
        Input polarity encoding used by the DBNN.
    output_dir : str or pathlib.Path, optional
        Directory for the GIF asset and metric JSON. No files are written when
        omitted.
    forbidden_trace_seeds : iterable of int, optional
        Seeds already used by fitting or reuse validation in the current
        Network. Colliding generated roots are replaced before teacher
        simulation.
    reserve_trace_seeds : callable or None, optional
        Callback invoked with concrete trace seeds before their teacher data
        are simulated or consumed.

    Returns
    -------
    tuple
        ``(gif_model, test_masked_mse, test_masked_variance_explained)``.

    Raises
    ------
    ValueError
        If the data source, split sizes, generated acceptance rows, or dataset
        metadata violate the workflow contract.
    TypeError
        If ``layout`` or ``dataset_pool`` has the wrong type.
    RuntimeError
        If generated teacher batches do not yield enough accepted traces.
    """
    if not isinstance(layout, ChannelLayout):
        raise TypeError(f"layout must be a ChannelLayout, got {type(layout).__name__!r}.")
    if (cell_factory is None) == (dataset_pool is None):
        raise ValueError("Provide exactly one of cell_factory and dataset_pool.")
    requested = {
        "train": _positive_int(train_traces, "train_traces"),
        "validation": _positive_int(validation_traces, "validation_traces"),
        "test": _positive_int(test_traces, "test_traces"),
    }
    total_traces = sum(requested.values())

    forbidden_seeds = _seed_set(forbidden_trace_seeds, name="forbidden_trace_seeds")
    if reserve_trace_seeds is not None and not callable(reserve_trace_seeds):
        raise TypeError("reserve_trace_seeds must be callable or None.")
    data_seed_roots: tuple[int, ...] = ()
    consumed_trace_seeds: tuple[int, ...]
    if dataset_pool is not None:
        pool, dt_ms, fingerprint, threshold_mv = _validate_pool(
            dataset_pool,
            layout,
            total_traces=total_traces,
            dt=dt,
            source_fingerprint=source_fingerprint,
            spike_threshold_mv=spike_threshold_mv,
        )
        overlap = forbidden_seeds.intersection(map(int, pool.plan.seeds))
        if overlap:
            raise ValueError(f"dataset_pool trace seeds overlap forbidden seeds: {sorted(overlap)!r}.")
        consumed_trace_seeds = tuple(int(seed) for seed in pool.plan.seeds)
        if reserve_trace_seeds is not None:
            reserve_trace_seeds(consumed_trace_seeds)
    else:
        if not callable(cell_factory):
            raise TypeError("cell_factory must be callable.")
        if rate_hz is None:
            raise ValueError("rate_hz is required when generating teacher data.")
        if not source_fingerprint:
            raise ValueError("source_fingerprint is required when generating teacher data.")
        duration = 6000.0 if duration_ms is None else _positive_float(duration_ms, "duration_ms")
        generation_dt = 1.0 * u.ms if dt is None else dt
        dt_ms = _time_to_ms(generation_dt)
        rates = _validate_rates(rate_hz, layout.n_channels, dt_ms)
        threshold_mv = -20.0 if spike_threshold_mv is None else _finite_float(
            spike_threshold_mv, "spike_threshold_mv"
        )
        fingerprint = source_fingerprint
        pool, data_seed_roots, consumed_trace_seeds = _generate_pool(
            layout,
            cell_factory=cell_factory,
            rate_hz=rates,
            output_location=output_location,
            acceptance=acceptance,
            total_traces=total_traces,
            duration_ms=duration,
            dt=generation_dt,
            dt_ms=dt_ms,
            teacher_batch_size=_positive_int(teacher_batch_size, "teacher_batch_size"),
            max_teacher_batches=_positive_int(max_teacher_batches, "max_teacher_batches"),
            data_seed=int(data_seed),
            source_fingerprint=fingerprint,
            spike_threshold_mv=threshold_mv,
            forbidden_trace_seeds=forbidden_seeds,
            reserve_trace_seeds=reserve_trace_seeds,
        )

    if output_dir is not None and dataset_pool is None:
        raw_pool_path = Path(output_dir) / "data" / "raw_teacher_pool.npz"
        raw_pool_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(raw_pool_path, pool)

    coverage_target = requested["validation"] + requested["test"] + 1
    coverage = _measure_effective_coverage(pool.plan, layout, target=coverage_target)
    if not coverage.meets_target:
        raise ValueError(
            "dataset pool lacks holdout-safe channel or pair coverage; "
            f"required traces per channel/pair={coverage_target}, "
            f"channel coverage={coverage.channel_coverage_fraction:.3f}, "
            f"pair coverage={coverage.pair_coverage_fraction:.3f}."
        )
    split_indices = _stratified_split_indices(pool, requested, seed=int(split_seed))
    split_batches = {
        name: _select_rows(pool, indices, split=name) for name, indices in split_indices.items()
    }
    for split_batch in split_batches.values():
        split_batch.metadata.update(
            {
                "spike_threshold_mv": threshold_mv,
                "spike_window_pre_ms": float(spike_window_pre_ms),
                "spike_window_post_ms": float(spike_window_post_ms),
                "gif_match_window_ms": float(gif_match_window_ms),
            }
        )
    data = {name: _prepare_data(batch, dt_ms) for name, batch in split_batches.items()}

    model = DBNN(layout, mode=mode, input_sign_mode=input_sign_mode, dt=dt_ms * u.ms)
    model.source_fingerprint = fingerprint
    model.input_alignment = pool.metadata["input_alignment"]
    model.dataset_trace_seeds = tuple(int(seed) for seed in pool.plan.seeds)
    model.consumed_trace_seeds = consumed_trace_seeds
    model.data_seed_roots = data_seed_roots
    params = model.get_params()
    params["bias"] = jnp.asarray(float(np.mean(split_batches["train"].voltage_mv[:, 0])))
    model.set_params(params)
    trainer = DBNNTrainer(
        model,
        learning_rate=learning_rate,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        gradient_clip=gradient_clip,
        loss="masked_mse",
        seed=int(training_seed),
        initialization_search=initialization_search,
        initialization_options=initialization_options,
    )
    for split in data.values():
        split["mask"] = trainer.build_fit_mask(
            split["targets"],
            spike_threshold_mv=threshold_mv,
            spike_window_pre_ms=spike_window_pre_ms,
            spike_window_post_ms=spike_window_post_ms,
        )
    if not np.all(np.any(np.asarray(data["train"]["mask"]), axis=1)):
        raise ValueError("Every training trace must retain at least one voltage-fit sample.")
    if not np.any(np.asarray(data["validation"]["mask"])):
        raise ValueError("Validation data must retain at least one voltage-fit sample.")
    if not any(len(times) for times in data["validation"]["spike_times_ms"]):
        raise ValueError("DBNN-GIF calibration requires at least one validation spike.")
    if initialization_search:
        trainer.search_initialization(data["train"])
    trainer.fit(
        data["train"],
        validation_data=data["validation"],
        epochs=_positive_int(epochs, "epochs"),
        batch_size=_positive_int(batch_size, "batch_size"),
        patience=None,
    )
    gif_model, gif_report = trainer.calibrate_gif(
        data["validation"],
        thresholds_mv=gif_thresholds_mv,
        threshold_increments_mv=gif_threshold_increments_mv,
        threshold_taus_ms=gif_threshold_taus_ms,
        match_window_ms=gif_match_window_ms,
        candidate_batch_size=gif_candidate_batch_size,
    )
    test_prediction = gif_model.predict(data["test"]["inputs"])
    metrics = compute_metrics(test_prediction["voltage"], data["test"]["targets"], data["test"]["mask"])
    mse = float(np.asarray(metrics["mse"]))
    variance_explained = float(np.asarray(metrics["variance_explained"]))
    raw_test_spikes = tuple(
        np.flatnonzero(row) * dt_ms for row in np.asarray(test_prediction["spike"])
    )
    validation_prediction = gif_model.predict(data["validation"]["inputs"])
    raw_validation_spikes = tuple(
        np.flatnonzero(row).astype(float) * dt_ms for row in np.asarray(validation_prediction["spike"])
    )
    gif_model.spike_alignment_evidence = SpikeAlignmentEvidence(
        teacher_voltage_mv=split_batches["validation"].voltage_mv,
        raw_spike_times_ms=raw_validation_spikes,
        dt_ms=dt_ms,
        match_window_ms=gif_match_window_ms,
        layout_fingerprint=layout.fingerprint,
        dynamics_fingerprint=dynamics_fingerprint or fingerprint,
        validation_seeds=tuple(split_batches["validation"].plan.seeds),
    )
    aligned_test_spikes = align_spike_times(
        raw_test_spikes,
        offset_ms=gif_model.spike_time_offset_ms,
        duration_ms=(test_prediction["spike"].shape[-1] - 1) * dt_ms,
    )
    test_spike_metrics = {
        "raw": compute_spike_metrics(
            raw_test_spikes,
            split_batches["test"].spike_times_ms,
            window_ms=gif_match_window_ms,
        ),
        "aligned": compute_spike_metrics(
            aligned_test_spikes,
            split_batches["test"].spike_times_ms,
            window_ms=gif_match_window_ms,
        ),
    }
    gif_model.calibration_report = gif_report["gif"]
    gif_model.test_spike_metrics = test_spike_metrics

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        data_destination = destination / "data"
        data_destination.mkdir(parents=True, exist_ok=True)
        gif_model.save(
            destination / "dbnn_gif_model.npz",
            spike_alignment_path=data_destination / "spike_alignment_validation.npz",
        )
        report = {
            "test_masked_mse": _json_metric(mse),
            "test_masked_variance_explained": _json_metric(variance_explained),
            "split_sizes": requested,
            "split_seed": int(split_seed),
            "training_seed": int(training_seed),
            "source_fingerprint": fingerprint,
            "data_seed_roots": data_seed_roots,
            "dataset_trace_seeds": model.dataset_trace_seeds,
            "consumed_trace_seeds": model.consumed_trace_seeds,
            "spike_time_offset_ms": gif_model.spike_time_offset_ms,
            "gif_calibration": _json_value(gif_report["gif"]),
            "test_spikes": _json_value(test_spike_metrics),
        }
        (destination / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    return gif_model, mse, variance_explained


def _validate_pool(
    pool: DatasetBatch,
    layout: ChannelLayout,
    *,
    total_traces: int,
    dt: Any | None,
    source_fingerprint: str | None,
    spike_threshold_mv: float | None,
) -> tuple[DatasetBatch, float, str, float]:
    if not isinstance(pool, DatasetBatch):
        raise TypeError(f"dataset_pool must be a DatasetBatch, got {type(pool).__name__!r}.")
    if pool.plan.n_traces != total_traces:
        raise ValueError(
            f"dataset_pool must contain exactly {total_traces} traces, got {pool.plan.n_traces}."
        )
    if len(set(map(int, pool.plan.seeds))) != pool.plan.n_traces:
        raise ValueError("dataset_pool trace seeds must be unique.")
    inferred_dt_ms = float(np.diff(np.asarray(pool.time_ms, dtype=float))[0])
    dt_ms = inferred_dt_ms if dt is None else _time_to_ms(dt)
    if not np.isclose(dt_ms, inferred_dt_ms):
        raise ValueError(f"dt={dt_ms} ms is incompatible with dataset_pool dt={inferred_dt_ms} ms.")
    metadata_fingerprint = str(pool.metadata.get("source_fingerprint") or "")
    fingerprint = source_fingerprint or metadata_fingerprint
    if not fingerprint:
        raise ValueError("dataset_pool requires source_fingerprint metadata or an explicit override.")
    validate_dataset(pool, layout, dt_ms=dt_ms, source_fingerprint=source_fingerprint)
    metadata_threshold = pool.metadata.get("spike_threshold_mv")
    if metadata_threshold is None:
        raise ValueError("dataset_pool metadata must declare spike_threshold_mv for its spike_times_ms.")
    if spike_threshold_mv is not None and metadata_threshold is not None:
        if not np.isclose(float(spike_threshold_mv), float(metadata_threshold)):
            raise ValueError("spike_threshold_mv is incompatible with dataset_pool metadata.")
    threshold = metadata_threshold if spike_threshold_mv is None else spike_threshold_mv
    if threshold is None:
        threshold = -20.0
    return pool, dt_ms, fingerprint, _finite_float(threshold, "spike_threshold_mv")


def _generate_pool(
    layout: ChannelLayout,
    *,
    cell_factory: CellFactory,
    rate_hz: float | np.ndarray,
    output_location: Any,
    acceptance: Acceptance | None,
    total_traces: int,
    duration_ms: float,
    dt: Any,
    dt_ms: float,
    teacher_batch_size: int,
    max_teacher_batches: int,
    data_seed: int,
    source_fingerprint: str,
    spike_threshold_mv: float,
    forbidden_trace_seeds: set[int],
    reserve_trace_seeds: Callable[[tuple[int, ...]], None] | None,
) -> tuple[DatasetBatch, tuple[int, ...], tuple[int, ...]]:
    teacher = TeacherSpec(cell_factory, output_location, source_fingerprint, dt, spike_threshold_mv)
    amplitudes = _training_amplitudes(layout)
    session = TeacherSession(teacher, layout, n_traces=teacher_batch_size)
    accepted = []
    count = 0
    used_trace_seeds = set(forbidden_trace_seeds)
    consumed_trace_seeds = []
    used_data_roots = []
    for attempt in range(max_teacher_batches):
        split = f"pool_candidate_{attempt:03d}"
        candidate_root = int(data_seed)
        for nonce in range(10_000):
            plan = generate_multichannel_protocol(
                layout,
                n_traces=teacher_batch_size,
                duration_ms=duration_ms,
                dt_ms=dt_ms,
                rate_hz=rate_hz,
                amplitude=amplitudes,
                seed=candidate_root,
                split=split,
            )
            if used_trace_seeds.isdisjoint(map(int, plan.seeds)):
                break
            candidate_root = _derived_data_seed(data_seed, attempt, nonce)
        else:
            raise RuntimeError("Could not derive fitting trace seeds disjoint from reserved data.")
        used_data_roots.append(candidate_root)
        used_trace_seeds.update(map(int, plan.seeds))
        consumed_trace_seeds.extend(map(int, plan.seeds))
        if reserve_trace_seeds is not None:
            reserve_trace_seeds(tuple(map(int, plan.seeds)))
        candidate = session.run(plan)
        rows = np.arange(candidate.plan.n_traces, dtype=np.int64) if acceptance is None else acceptance(candidate)
        rows = _validate_acceptance_rows(rows, candidate.plan.n_traces)
        rows = rows[: total_traces - count]
        if len(rows):
            accepted.append(_select_rows(candidate, rows, split="accepted_pool"))
            count += len(rows)
        if count == total_traces:
            break
    if count != total_traces:
        raise RuntimeError(
            f"Accepted only {count}/{total_traces} traces after {max_teacher_batches} teacher batches."
        )
    return _combine_batches(accepted), tuple(used_data_roots), tuple(consumed_trace_seeds)


def _training_amplitudes(layout: ChannelLayout) -> np.ndarray:
    amplitudes = np.empty(layout.n_channels, dtype=float)
    for channel in range(layout.n_channels):
        low, high = layout.training_range(channel)
        if high <= 0:
            raise ValueError(f"Channel {channel} training range has no positive event amplitude.")
        amplitude = max(low, min(1.0, high))
        amplitudes[channel] = high / 2.0 if amplitude == 0 else amplitude
    return amplitudes


def _seed_set(values: Any, *, name: str) -> set[int]:
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers.") from exc
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in supplied):
        raise TypeError(f"{name} must contain only integers.")
    seeds = tuple(int(value) for value in supplied)
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**31 - 1 for seed in seeds):
        raise ValueError(f"{name} must contain unique seeds in [0, {2**31 - 1}).")
    return set(seeds)


def _derived_data_seed(root: int, attempt: int, nonce: int) -> int:
    payload = f"dbnn-fit:{int(root)}:{attempt}:{nonce}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def _prepare_data(batch: DatasetBatch, dt_ms: float) -> dict[str, Any]:
    alignment = batch.metadata.get("input_alignment")
    return {
        "inputs": rasterize_events(batch.plan, dt_ms=dt_ms, input_alignment=alignment),
        "targets": jnp.asarray(batch.voltage_mv),
        "spike_times_ms": batch.spike_times_ms,
    }


def _select_rows(batch: DatasetBatch, rows: np.ndarray, *, split: str) -> DatasetBatch:
    rows = np.asarray(rows, dtype=np.int64)
    remap = np.full(batch.plan.n_traces, -1, dtype=np.int64)
    remap[rows] = np.arange(len(rows), dtype=np.int64)
    keep = remap[batch.plan.events.trace_id] >= 0
    plan = StimulusPlan(
        batch.plan.layout_fingerprint,
        len(rows),
        SparseEvents(
            remap[batch.plan.events.trace_id[keep]],
            batch.plan.events.channel_id[keep],
            batch.plan.events.time_ms[keep],
        ),
        batch.plan.channel_weights[rows],
        batch.plan.duration_ms,
        tuple(batch.plan.seeds[int(row)] for row in rows),
        tuple(batch.plan.protocol_labels[int(row)] for row in rows),
        split,
    )
    return DatasetBatch(
        plan,
        batch.voltage_mv[rows],
        tuple(batch.spike_times_ms[int(row)] for row in rows),
        batch.time_ms,
        dict(batch.metadata),
    )


def _combine_batches(batches: list[DatasetBatch]) -> DatasetBatch:
    first = batches[0]
    for batch in batches[1:]:
        if not np.array_equal(batch.time_ms, first.time_ms) or batch.metadata != first.metadata:
            raise ValueError("Generated accepted batches must share time and metadata contracts.")
    return DatasetBatch(
        combine_protocols(*(batch.plan for batch in batches)),
        np.concatenate([batch.voltage_mv for batch in batches]),
        tuple(spikes for batch in batches for spikes in batch.spike_times_ms),
        first.time_ms,
        dict(first.metadata),
    )


def _stratified_split_indices(
    pool: DatasetBatch,
    requested: dict[str, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    spike_counts = np.asarray([len(times) for times in pool.spike_times_ms], dtype=np.int64)
    names = tuple(requested)
    assigned = {name: [] for name in names}
    capacities = np.asarray([requested[name] for name in names], dtype=np.int64)
    remaining_total = pool.plan.n_traces
    rng = brainstate.random.RandomState(seed)
    for spike_count in np.unique(spike_counts):
        group = np.asarray(rng.permutation(np.flatnonzero(spike_counts == spike_count)), dtype=np.int64)
        ideal = len(group) * capacities / remaining_total
        allocation = np.minimum(np.floor(ideal).astype(np.int64), capacities)
        unassigned = len(group) - int(np.sum(allocation))
        fractions = ideal - np.floor(ideal)
        while unassigned:
            available = np.flatnonzero(allocation < capacities)
            destination = int(available[np.argmax(fractions[available])])
            allocation[destination] += 1
            fractions[destination] = -1.0
            unassigned -= 1
        start = 0
        for name, size in zip(names, allocation):
            stop = start + int(size)
            assigned[name].extend(map(int, group[start:stop]))
            start = stop
        capacities -= allocation
        remaining_total -= len(group)
    _ensure_validation_spike(assigned, spike_counts)
    result = {name: np.asarray(indices, dtype=np.int64) for name, indices in assigned.items()}
    combined = np.concatenate(tuple(result.values()))
    if len(np.unique(combined)) != pool.plan.n_traces:
        raise AssertionError("Dataset split indices must be complete and disjoint.")
    return result


def _ensure_validation_spike(
    assigned: dict[str, list[int]],
    spike_counts: np.ndarray,
) -> None:
    if any(spike_counts[index] > 0 for index in assigned["validation"]):
        return
    validation_zero = next(
        (index for index in assigned["validation"] if spike_counts[index] == 0),
        None,
    )
    if validation_zero is None:
        return
    for source in ("train", "test"):
        source_spike = next(
            (
                index
                for index in assigned[source]
                if spike_counts[index] > 0
            ),
            None,
        )
        if source_spike is not None:
            assigned["validation"].remove(validation_zero)
            assigned["validation"].append(source_spike)
            assigned[source].remove(source_spike)
            assigned[source].append(validation_zero)
            return


def _measure_effective_coverage(
    plan: StimulusPlan,
    layout: ChannelLayout,
    *,
    target: int,
):
    event_weights = plan.channel_weights[plan.events.trace_id, plan.events.channel_id]
    keep = event_weights > 0
    effective_plan = StimulusPlan(
        plan.layout_fingerprint,
        plan.n_traces,
        SparseEvents(
            plan.events.trace_id[keep],
            plan.events.channel_id[keep],
            plan.events.time_ms[keep],
        ),
        plan.channel_weights,
        plan.duration_ms,
        plan.seeds,
        plan.protocol_labels,
        plan.split,
    )
    return measure_coverage(effective_plan, layout, target=target)


def _validate_acceptance_rows(rows: Any, n_traces: int) -> np.ndarray:
    result = np.asarray(rows)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise TypeError("acceptance must return a one-dimensional integer row array.")
    result = result.astype(np.int64, copy=False)
    if np.any((result < 0) | (result >= n_traces)):
        raise ValueError("acceptance returned a row outside the candidate batch.")
    if len(np.unique(result)) != len(result):
        raise ValueError("acceptance returned duplicate rows.")
    return result


def _time_to_ms(dt: Any) -> float:
    if not hasattr(dt, "to_decimal"):
        raise TypeError("dt must be a brainunit time quantity.")
    try:
        value = float(np.asarray(dt.to_decimal(u.ms)).reshape(()))
    except Exception as error:
        raise TypeError("dt must be convertible to milliseconds.") from error
    return _positive_float(value, "dt")


def _validate_rates(rate_hz: Any, n_channels: int, dt_ms: float) -> np.ndarray:
    try:
        rates = np.broadcast_to(np.asarray(rate_hz, dtype=float), (n_channels,))
    except ValueError as error:
        raise ValueError(f"rate_hz must be scalar or have shape ({n_channels},).") from error
    probabilities = rates * dt_ms / 1000.0
    if not np.isfinite(rates).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("rate_hz must be finite and define Bernoulli probabilities in [0, 1].")
    return rates


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _json_metric(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


__all__ = ["fit_dbnn_gif"]
