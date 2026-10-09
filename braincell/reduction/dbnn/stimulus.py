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

"""Coverage-aware sparse DBNN stimulus strategies."""

from dataclasses import dataclass
import hashlib
from typing import Any

import brainstate
import numpy as np

from braincell.reduction.dbnn.layout import ChannelLayout, packed_pair_index


DEFAULT_TRAINING_TRACES = 1000


def _split_seed(seed: int, split: str) -> int:
    digest = hashlib.sha256(split.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:4], "little")) % (2**31 - 1)


@dataclass(frozen=True)
class SparseEvents:
    """Store sparse event identities and times in milliseconds."""

    trace_id: np.ndarray
    channel_id: np.ndarray
    time_ms: np.ndarray

    def __post_init__(self):
        trace_id = np.asarray(self.trace_id)
        channel_id = np.asarray(self.channel_id)
        time_ms = np.asarray(self.time_ms)
        if trace_id.ndim != 1 or channel_id.shape != trace_id.shape or time_ms.shape != trace_id.shape:
            raise ValueError("Sparse event columns must be one-dimensional and share a shape.")
        if not np.issubdtype(trace_id.dtype, np.integer) or not np.issubdtype(channel_id.dtype, np.integer):
            raise TypeError("Sparse event trace_id and channel_id must use integer dtypes.")
        if np.any(trace_id < 0) or np.any(channel_id < 0) or not np.isfinite(time_ms).all() or np.any(time_ms < 0):
            raise ValueError("Sparse event identities and times must be finite and non-negative.")


@dataclass(frozen=True)
class StimulusPlan:
    """Describe independently generated sparse DBNN stimulus traces."""

    layout_fingerprint: str
    n_traces: int
    events: SparseEvents
    channel_weights: np.ndarray
    duration_ms: float
    seeds: tuple[int, ...]
    protocol_labels: tuple[str, ...]
    split: str

    def __post_init__(self):
        if not isinstance(self.layout_fingerprint, str) or not self.layout_fingerprint:
            raise ValueError("StimulusPlan requires a non-empty channel layout fingerprint.")
        if not isinstance(self.n_traces, int) or isinstance(self.n_traces, bool) or self.n_traces <= 0:
            raise ValueError("Stimulus n_traces must be a positive integer.")
        if self.duration_ms <= 0:
            raise ValueError("Stimulus dimensions and duration must be positive.")
        weights = np.asarray(self.channel_weights)
        if weights.ndim != 2 or weights.shape[1] <= 0:
            raise ValueError("channel_weights must be a two-dimensional array with at least one channel.")
        if weights.shape != (self.n_traces, self.n_channels):
            raise ValueError(
                f"channel_weights must have shape {(self.n_traces, self.n_channels)}, got {weights.shape}."
            )
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("DBNN channel weights must be finite and non-negative.")
        if len(self.seeds) != self.n_traces or len(self.protocol_labels) != self.n_traces:
            raise ValueError("Each stimulus trace requires one seed and protocol label.")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in self.seeds):
            raise TypeError("Stimulus trace seeds must be integers.")
        seeds = tuple(int(seed) for seed in self.seeds)
        if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**31 - 1 for seed in seeds):
            raise ValueError("Stimulus trace seeds must be unique and lie in [0, 2**31 - 1).")
        if np.any(self.events.trace_id >= self.n_traces) or np.any(self.events.channel_id >= self.n_channels):
            raise ValueError("Sparse events reference a trace or channel outside the plan.")
        if np.any(self.events.time_ms >= self.duration_ms):
            raise ValueError("Sparse event times must lie in the half-open stimulus duration.")

    @property
    def n_channels(self) -> int:
        """Return the channel count derived from the aligned weight matrix."""
        weights = np.asarray(self.channel_weights)
        return int(weights.shape[1]) if weights.ndim == 2 else 0


@dataclass(frozen=True)
class CoverageReport:
    """Report channel and packed pair activation coverage."""

    channel_count: np.ndarray
    pair_count: np.ndarray
    pair_class_count: dict[str, int]
    undercovered_pairs: tuple[tuple[int, int], ...]
    target: int
    n_traces: int
    channel_coverage_fraction: float
    pair_coverage_fraction: float
    pair_class_coverage: dict[str, float]
    protocol_channel_count: dict[str, np.ndarray]
    protocol_channel_coverage: dict[str, float]
    meets_target: bool


def measure_coverage(plan: StimulusPlan, layout: ChannelLayout, *, target: int = 1) -> CoverageReport:
    """Measure per-channel, per-pair, and EE/EI/II trace coverage."""
    if plan.layout_fingerprint != layout.fingerprint:
        raise ValueError("Stimulus plan belongs to a different channel layout.")
    if plan.n_channels != layout.n_channels:
        raise ValueError("Stimulus and channel layout sizes do not match.")
    if target < 0:
        raise ValueError(f"Coverage target must be non-negative, got {target}.")
    channel_count = np.zeros(layout.n_channels, dtype=np.int64)
    pair_count = np.zeros(layout.n_channels * (layout.n_channels - 1) // 2, dtype=np.int64)
    pair_classes = {"EE": 0, "EI": 0, "II": 0}
    pair_class_indices = {"EE": [], "EI": [], "II": []}
    for i in range(layout.n_channels):
        for j in range(i + 1, layout.n_channels):
            pair_class = "".join(sorted((layout.polarity(i), layout.polarity(j))))
            pair_class_indices[pair_class].append(packed_pair_index(i, j, layout.n_channels))
    active_by_trace = [set() for _ in range(plan.n_traces)]
    for trace_id, channel_id in zip(plan.events.trace_id, plan.events.channel_id):
        active_by_trace[int(trace_id)].add(int(channel_id))
    for active in active_by_trace:
        ordered = sorted(active)
        channel_count[ordered] += 1
        for offset, i in enumerate(ordered):
            for j in ordered[offset + 1 :]:
                pair_count[packed_pair_index(i, j, layout.n_channels)] += 1
                pair_classes["".join(sorted((layout.polarity(i), layout.polarity(j))))] += 1
    protocol_channel_count = {
        label: np.zeros(layout.n_channels, dtype=np.int64) for label in sorted(set(plan.protocol_labels))
    }
    for trace_id, active in enumerate(active_by_trace):
        if active:
            protocol_channel_count[plan.protocol_labels[trace_id]][sorted(active)] += 1
    protocol_channel_coverage = {label: float(np.mean(counts > 0)) for label, counts in protocol_channel_count.items()}
    undercovered = []
    for i in range(layout.n_channels):
        for j in range(i + 1, layout.n_channels):
            if pair_count[packed_pair_index(i, j, layout.n_channels)] < target:
                undercovered.append((i, j))
    channel_covered = channel_count >= target
    pair_covered = pair_count >= target
    pair_class_coverage = {
        name: (float(np.mean(pair_covered[indices])) if indices else float("nan"))
        for name, indices in pair_class_indices.items()
    }
    return CoverageReport(
        channel_count=channel_count,
        pair_count=pair_count,
        pair_class_count=pair_classes,
        undercovered_pairs=tuple(undercovered),
        target=target,
        n_traces=plan.n_traces,
        channel_coverage_fraction=float(np.mean(channel_covered)),
        pair_coverage_fraction=float(np.mean(pair_covered)) if pair_covered.size else 1.0,
        pair_class_coverage=pair_class_coverage,
        protocol_channel_count=protocol_channel_count,
        protocol_channel_coverage=protocol_channel_coverage,
        meets_target=bool(np.all(channel_covered) and np.all(pair_covered)),
    )


def generate_pair_protocol(
    layout: ChannelLayout,
    *,
    repetitions: int = 1,
    event_times_ms: tuple[float, float] = (10.0, 10.0),
    amplitude: float | np.ndarray = 1.0,
    duration_ms: float = 50.0,
    seed: int = 0,
    split: str = "train",
    n_traces: int | None = None,
) -> StimulusPlan:
    """Generate pair coverage within a fixed trace budget.

    Distinct EE, EI, and II pairs are visited in round-robin order before a
    second repetition is generated. When complete coverage exceeds the budget,
    :func:`measure_coverage` reports the remaining undercovered coefficients.
    """
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}.")
    if layout.n_channels < 2:
        raise ValueError("Pair stimulation requires at least two DBNN channels.")
    if len(event_times_ms) != 2 or min(event_times_ms) < 0 or max(event_times_ms) >= duration_ms:
        raise ValueError("Pair event times must lie within the half-open duration.")
    if amplitude <= 0 or not np.isfinite(amplitude):
        raise ValueError("Pair amplitude must be finite and positive.")
    if n_traces is not None and (not isinstance(n_traces, int) or isinstance(n_traces, bool) or n_traces <= 0):
        raise ValueError(f"n_traces must be a positive integer or None, got {n_traces!r}.")
    if n_traces is not None and repetitions != 1:
        raise ValueError("Specify either n_traces or repetitions, not both.")
    minimum_channel_coverage = (layout.n_channels + 1) // 2
    pair_count = layout.n_channels * (layout.n_channels - 1) // 2
    requested_traces = pair_count * repetitions if n_traces is None else n_traces
    if requested_traces < minimum_channel_coverage:
        raise ValueError(
            f"Covering all {layout.n_channels} channels with pair stimuli requires at least "
            f"{minimum_channel_coverage} traces, but n_traces={requested_traces}."
        )
    effective_seed = _split_seed(seed, split)
    selected_pairs = _balanced_pair_order(layout, n_traces=requested_traces, seed=effective_seed)
    trace_ids = []
    channel_ids = []
    times = []
    weights = np.zeros((requested_traces, layout.n_channels), dtype=np.float32)
    labels = []
    seeds = []
    for trace, (repetition, i, j) in enumerate(selected_pairs):
        trace_ids.extend((trace, trace))
        channel_ids.extend((i, j))
        times.extend(event_times_ms)
        weights[trace, [i, j]] = amplitude
        labels.append("pair")
        seeds.append(
            int(
                (effective_seed + repetition * (layout.n_channels**2) + packed_pair_index(i, j, layout.n_channels))
                % (2**31 - 1)
            )
        )
    events = SparseEvents(
        np.asarray(trace_ids, dtype=np.int64),
        np.asarray(channel_ids, dtype=np.int64),
        np.asarray(times, dtype=np.float32),
    )
    return StimulusPlan(
        layout_fingerprint=layout.fingerprint,
        n_traces=requested_traces,
        events=events,
        channel_weights=weights,
        duration_ms=duration_ms,
        seeds=tuple(seeds),
        protocol_labels=tuple(labels),
        split=split,
    )


def generate_multichannel_protocol(
    layout: ChannelLayout,
    *,
    n_traces: int,
    duration_ms: float,
    dt_ms: float,
    rate_hz: float | np.ndarray,
    amplitude: float = 1.0,
    seed: int = 0,
    split: str = "train",
    ensure_channel_coverage: bool = True,
) -> StimulusPlan:
    """Generate reproducible independent Bernoulli multichannel traces."""
    if n_traces <= 0 or duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("Trace count, duration, and dt must be positive.")
    rates = np.broadcast_to(np.asarray(rate_hz, dtype=float), (layout.n_channels,))
    probabilities = rates * dt_ms / 1000.0
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("rate_hz * dt_ms must define probabilities in [0, 1].")
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms):
        raise ValueError("duration_ms must be an integer multiple of dt_ms.")
    if n_steps < 1:
        raise ValueError("duration_ms must contain at least one complete dt_ms interval.")
    amplitudes = np.asarray(amplitude, dtype=float)
    try:
        amplitudes = np.broadcast_to(amplitudes, (layout.n_channels,))
    except ValueError as exc:
        raise ValueError(f"amplitude must be scalar or have shape {(layout.n_channels,)}.") from exc
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("Multichannel amplitudes must be finite and positive.")
    if not isinstance(ensure_channel_coverage, bool):
        raise TypeError("ensure_channel_coverage must be a Boolean.")
    trace_ids = []
    channel_ids = []
    times = []
    effective_seed = _split_seed(seed, split)
    trace_seeds = tuple((effective_seed + trace) % (2**31 - 1) for trace in range(n_traces))
    for trace in range(n_traces):
        rng = brainstate.random.RandomState(trace_seeds[trace])
        draws = np.asarray(rng.random((layout.n_channels, n_steps)))
        channels, steps = np.nonzero(draws < probabilities[:, None])
        trace_ids.extend([trace] * len(channels))
        channel_ids.extend(channels.tolist())
        times.extend((steps * dt_ms).tolist())
    if ensure_channel_coverage:
        covered = np.zeros(layout.n_channels, dtype=bool)
        if channel_ids:
            covered[np.asarray(channel_ids, dtype=np.int64)] = True
        for channel in np.flatnonzero(~covered):
            trace_ids.append(int(channel % n_traces))
            channel_ids.append(int(channel))
            times.append(float((channel % n_steps) * dt_ms))
    weights = np.broadcast_to(amplitudes, (n_traces, layout.n_channels)).astype(np.float32, copy=True)
    events = SparseEvents(
        np.asarray(trace_ids, dtype=np.int64),
        np.asarray(channel_ids, dtype=np.int64),
        np.asarray(times, dtype=np.float32),
    )
    return StimulusPlan(
        layout_fingerprint=layout.fingerprint,
        n_traces=n_traces,
        events=events,
        channel_weights=weights,
        duration_ms=duration_ms,
        seeds=trace_seeds,
        protocol_labels=tuple("multichannel" for _ in range(n_traces)),
        split=split,
    )


def combine_protocols(*plans: StimulusPlan) -> StimulusPlan:
    """Combine compatible independently generated stimulus plans."""
    if not plans:
        raise ValueError("At least one stimulus plan is required.")
    first = plans[0]
    for plan in plans[1:]:
        if plan.layout_fingerprint != first.layout_fingerprint:
            raise ValueError("Combined plans must belong to the same channel layout.")
        if (plan.n_channels, plan.duration_ms, plan.split) != (first.n_channels, first.duration_ms, first.split):
            raise ValueError("Combined plans must share channel count, duration, and split identity.")
    total_traces = sum(plan.n_traces for plan in plans)
    offsets = np.cumsum([0] + [plan.n_traces for plan in plans[:-1]])
    events = SparseEvents(
        np.concatenate([plan.events.trace_id + offset for plan, offset in zip(plans, offsets)]),
        np.concatenate([plan.events.channel_id for plan in plans]),
        np.concatenate([plan.events.time_ms for plan in plans]),
    )
    return StimulusPlan(
        layout_fingerprint=first.layout_fingerprint,
        n_traces=total_traces,
        events=events,
        channel_weights=np.concatenate([plan.channel_weights for plan in plans]),
        duration_ms=first.duration_ms,
        seeds=sum((plan.seeds for plan in plans), ()),
        protocol_labels=sum((plan.protocol_labels for plan in plans), ()),
        split=first.split,
    )


def generate_default_protocol(
    layout: ChannelLayout,
    *,
    n_traces: int = DEFAULT_TRAINING_TRACES,
    seed: int = 0,
    split: str = "train",
    pair_fraction: float = 0.2,
) -> StimulusPlan:
    """Split a requested training size into pair and multichannel traces.

    The pair allocation is increased above ``pair_fraction`` when necessary to
    guarantee that every channel appears in at least one pair trace.
    """
    if not isinstance(n_traces, int) or isinstance(n_traces, bool) or n_traces < 2:
        raise ValueError(f"n_traces must be an integer of at least 2, got {n_traces!r}.")
    if not 0 < pair_fraction < 1:
        raise ValueError("pair_fraction must lie strictly between zero and one.")
    minimum_pair_budget = (layout.n_channels + 1) // 2
    if minimum_pair_budget + 1 > n_traces:
        raise ValueError(
            f"Covering all {layout.n_channels} channels requires {minimum_pair_budget} pair traces; "
            f"at least one additional multichannel trace is required, exceeding n_traces={n_traces}."
        )
    pair_budget = min(
        n_traces - 1,
        max(minimum_pair_budget, int(round(n_traces * pair_fraction))),
    )
    pair_plan = generate_pair_protocol(
        layout,
        seed=seed,
        split=split,
        n_traces=pair_budget,
    )
    multichannel_budget = n_traces - pair_plan.n_traces
    multichannel_seed = seed + pair_plan.n_traces
    for _ in range(10_000):
        multichannel_plan = generate_multichannel_protocol(
            layout,
            n_traces=multichannel_budget,
            duration_ms=pair_plan.duration_ms,
            dt_ms=1.0,
            rate_hz=50.0,
            seed=multichannel_seed,
            split=split,
        )
        if set(pair_plan.seeds).isdisjoint(multichannel_plan.seeds):
            break
        multichannel_seed += n_traces
    else:
        raise RuntimeError("Could not derive disjoint pair and multichannel trace seeds.")
    return combine_protocols(pair_plan, multichannel_plan)


def _balanced_pair_order(
    layout: ChannelLayout,
    *,
    n_traces: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Generate only the requested number of pairs without materializing O(N^2)."""
    channel_cover = _channel_cover_pairs(layout.n_channels)
    unique_order = list(channel_cover)
    used = set(unique_order)
    total_pairs = layout.n_channels * (layout.n_channels - 1) // 2
    unique_target = min(n_traces, total_pairs)
    polarity_channels = {
        polarity: [channel_id for channel_id in range(layout.n_channels) if layout.polarity(channel_id) == polarity]
        for polarity in ("E", "I")
    }
    representatives = []
    if len(polarity_channels["E"]) >= 2:
        representatives.append(tuple(sorted(polarity_channels["E"][:2])))
    if polarity_channels["E"] and polarity_channels["I"]:
        representatives.append(tuple(sorted((polarity_channels["E"][0], polarity_channels["I"][0]))))
    if len(polarity_channels["I"]) >= 2:
        representatives.append(tuple(sorted(polarity_channels["I"][:2])))
    for pair in representatives:
        if len(unique_order) == unique_target:
            break
        if pair not in used:
            unique_order.append(pair)
            used.add(pair)

    start = seed % layout.n_channels
    if len(unique_order) < unique_target:
        for offset in range(1, layout.n_channels):
            for step in range(layout.n_channels):
                i = (start + step) % layout.n_channels
                j = (i + offset) % layout.n_channels
                pair = (i, j) if i < j else (j, i)
                if pair not in used:
                    unique_order.append(pair)
                    used.add(pair)
                    if len(unique_order) == unique_target:
                        break
            if len(unique_order) == unique_target:
                break

    result = [(0, i, j) for i, j in unique_order]
    repetitions = int(np.ceil(n_traces / len(unique_order)))
    for repetition in range(1, repetitions):
        rng = brainstate.random.RandomState(seed + repetition)
        order = np.asarray(rng.permutation(len(unique_order)), dtype=np.int64)
        for index in order:
            i, j = unique_order[int(index)]
            result.append((repetition, i, j))
            if len(result) == n_traces:
                break
    return result


def _channel_cover_pairs(n_channels: int) -> list[tuple[int, int]]:
    pairs = [(channel, channel + 1) for channel in range(0, n_channels - 1, 2)]
    if n_channels % 2:
        pairs.append((0, n_channels - 1))
    return pairs


__all__ = [
    "CoverageReport",
    "DEFAULT_TRAINING_TRACES",
    "SparseEvents",
    "StimulusPlan",
    "combine_protocols",
    "generate_default_protocol",
    "generate_multichannel_protocol",
    "generate_pair_protocol",
    "measure_coverage",
]
