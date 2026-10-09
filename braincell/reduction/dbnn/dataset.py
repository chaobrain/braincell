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

"""Detailed-cell simulation and sparse DBNN dataset contracts."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.layout import validate_channel_coverage
from braincell.reduction.dbnn.stimulus import SparseEvents, StimulusPlan


DATASET_FORMAT_VERSION = 4


@dataclass(frozen=True)
class TeacherSpec:
    """Configure reproducible detailed-Cell teacher batches."""

    cell_factory: Any
    output_location: Any
    source_fingerprint: str
    dt: Any
    spike_threshold_mv: float

    def __post_init__(self):
        if not callable(self.cell_factory):
            raise TypeError("cell_factory must create an uninitialized detailed Cell population.")
        if not self.source_fingerprint:
            raise ValueError("TeacherSpec requires a non-empty source fingerprint.")
        if not hasattr(self.dt, "to_decimal"):
            raise TypeError("TeacherSpec.dt must be a brainunit time quantity.")


@dataclass(frozen=True)
class DatasetBatch:
    """Store one independently generated DBNN teacher batch."""

    plan: StimulusPlan
    voltage_mv: np.ndarray
    spike_times_ms: tuple[np.ndarray, ...]
    time_ms: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self):
        voltage = np.asarray(self.voltage_mv)
        time = np.asarray(self.time_ms)
        if voltage.shape != (self.plan.n_traces, time.size):
            raise ValueError(f"voltage_mv must have shape {(self.plan.n_traces, time.size)}, got {voltage.shape}.")
        if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("time_ms must be a finite, strictly increasing one-dimensional axis.")
        if not np.isfinite(voltage).all():
            raise ValueError("Teacher voltage contains non-finite values.")
        if len(self.spike_times_ms) != self.plan.n_traces:
            raise ValueError("Each teacher trace requires one spike-time array.")
        for spike_times in self.spike_times_ms:
            spikes = np.asarray(spike_times)
            if spikes.ndim != 1 or not np.isfinite(spikes).all():
                raise ValueError("Teacher spike times must be finite one-dimensional arrays.")
            if np.any(spikes < 0) or np.any(spikes > self.plan.duration_ms):
                raise ValueError("Teacher spike times must lie within the recorded duration.")

    @property
    def layout_fingerprint(self) -> str:
        """Return the authoritative layout identity carried by the stimulus plan."""
        return self.plan.layout_fingerprint


class TeacherSession:
    """Reuse one detailed-Cell population and compiled dense-input rollout.

    Parameters
    ----------
    teacher : TeacherSpec
        Detailed-cell factory, output location, timing, and provenance.
    layout : ChannelLayout
        Logical-synapse channels routed into the detailed cell.
    n_traces : int
        Fixed population size shared by every plan passed to :meth:`run`.
    """

    def __init__(self, teacher: TeacherSpec, layout: ChannelLayout, *, n_traces: int):
        if not isinstance(n_traces, int) or isinstance(n_traces, bool) or n_traces < 1:
            raise ValueError("n_traces must be a positive integer.")
        self.teacher = teacher
        self.layout = layout
        self.n_traces = n_traces
        self.dt_ms = float(np.asarray(teacher.dt.to_decimal(u.ms)).reshape(()))
        self.cell = teacher.cell_factory((n_traces,))
        self.cell.init_state()
        self._output_cv_id = _resolve_output_cv(self.cell, teacher.output_location)
        self._routes = _build_dense_routes(self.cell, layout)
        self._initial_state = tuple(
            (state, state.value) for state in brainstate.graph.states(self.cell).values()
        )
        self._runner = self._build_runner()

    def _build_runner(self):
        cell = self.cell
        output_cv_id = self._output_cv_id
        routes = self._routes
        dt = self.teacher.dt

        def run_dense(events):
            initial_voltage = cell.V.value[:, output_cv_id]

            def step(event):
                for synapse, buffer_size, entries in routes:
                    payload = jnp.zeros((buffer_size,), dtype=event.dtype)
                    for runtime_row, population_index, channel_id, reference_weight_us in entries:
                        contribution = event[population_index, channel_id] * reference_weight_us
                        payload = payload.at[runtime_row].add(contribution)
                    synapse.apply_events(payload * u.uS)
                cell._update_dynamics()
                return cell.V.value[:, output_cv_id]

            with brainstate.environ.context(dt=dt):
                voltage = brainstate.transform.for_loop(step, events)
            return initial_voltage, voltage

        return brainstate.transform.jit(run_dense)

    def run(self, plan: StimulusPlan) -> DatasetBatch:
        """Simulate one fixed-shape plan using the reusable compiled rollout.

        Parameters
        ----------
        plan : StimulusPlan
            Sparse event plan with the session's fixed trace and channel shape.

        Returns
        -------
        DatasetBatch
            Teacher voltage and spike times aligned to target-step inputs.
        """
        _validate_teacher_plan(self.layout, plan)
        if plan.n_traces != self.n_traces:
            raise ValueError(
                f"TeacherSession requires {self.n_traces} traces, got {plan.n_traces}."
            )
        for state, value in self._initial_state:
            state.value = value
        raster = rasterize_events(plan, dt_ms=self.dt_ms, input_alignment="interval-start")
        events = jnp.transpose(raster[:, :, :-1], (2, 0, 1))
        initial_voltage, voltage = self._runner(events)
        voltage_mv = np.asarray(voltage.to_decimal(u.mV), dtype=np.float32).T
        initial_voltage_mv = np.asarray(
            initial_voltage.to_decimal(u.mV), dtype=np.float32
        )[:, None]
        voltage_mv = np.concatenate((initial_voltage_mv, voltage_mv), axis=1)
        return _make_teacher_batch(self.teacher, plan, voltage_mv, self.dt_ms)


def rasterize_events(plan: StimulusPlan, *, dt_ms: float, input_alignment: str) -> jnp.ndarray:
    """Rasterize sparse events into ``(trace, channel, time)`` input bins."""
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}.")
    if input_alignment not in {"target-step", "interval-start"}:
        raise ValueError(f"Unsupported input alignment: {input_alignment!r}.")
    n_intervals = int(round(plan.duration_ms / dt_ms))
    if not np.isclose(n_intervals * dt_ms, plan.duration_ms):
        raise ValueError("Stimulus duration must be an integer multiple of dt_ms.")
    result = np.zeros((plan.n_traces, plan.n_channels, n_intervals + 1), dtype=np.float32)
    bins = np.floor(np.asarray(plan.events.time_ms) / dt_ms + 1e-9).astype(np.int64)
    if input_alignment == "target-step":
        bins = bins + 1
    amplitudes = plan.channel_weights[plan.events.trace_id, plan.events.channel_id]
    np.add.at(result, (plan.events.trace_id, plan.events.channel_id, bins), amplitudes)
    return jnp.asarray(result)


def validate_dataset(
    batch: DatasetBatch,
    layout: ChannelLayout,
    *,
    dt_ms: float,
    source_fingerprint: str | None = None,
) -> None:
    """Validate layout, split, timing, events, and padding contracts."""
    if batch.plan.layout_fingerprint != layout.fingerprint:
        raise ValueError("Dataset channel layout fingerprint does not match the requested layout.")
    if batch.plan.n_channels != layout.n_channels:
        raise ValueError("Dataset channel count does not match the requested layout.")
    if source_fingerprint is not None and batch.metadata.get("source_fingerprint") != source_fingerprint:
        raise ValueError("Dataset source fingerprint does not match the requested teacher.")
    actual_dt = float(np.diff(batch.time_ms)[0])
    if not np.allclose(np.diff(batch.time_ms), actual_dt) or not np.isclose(actual_dt, dt_ms):
        raise ValueError(f"Dataset dt={actual_dt} ms is incompatible with model dt={dt_ms} ms.")
    alignment = batch.metadata.get("input_alignment")
    if alignment not in {"target-step", "interval-start"}:
        raise ValueError("Dataset metadata must declare input_alignment.")
    raster = rasterize_events(batch.plan, dt_ms=dt_ms, input_alignment=alignment)
    if raster.shape[-1] != batch.voltage_mv.shape[-1]:
        raise ValueError(
            f"Dataset time length {batch.voltage_mv.shape[-1]} does not match "
            f"rasterized input length {raster.shape[-1]}."
        )
    expected_time = np.arange(raster.shape[-1], dtype=float) * dt_ms
    if not np.allclose(batch.time_ms, expected_time):
        raise ValueError("Dataset time_ms must start at zero and cover the complete stimulus duration.")


def save_dataset(path: str | Path, batch: DatasetBatch) -> None:
    """Save a sparse DBNN dataset batch as a versioned NPZ file."""
    spike_trace = np.concatenate(
        [np.full(len(times), trace, dtype=np.int64) for trace, times in enumerate(batch.spike_times_ms)]
    )
    spike_time = np.concatenate(batch.spike_times_ms).astype(np.float32, copy=False)
    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "layout_fingerprint": batch.plan.layout_fingerprint,
        "n_traces": batch.plan.n_traces,
        "n_channels": batch.plan.n_channels,
        "duration_ms": batch.plan.duration_ms,
        "seeds": batch.plan.seeds,
        "protocol_labels": batch.plan.protocol_labels,
        "split": batch.plan.split,
        "metadata": batch.metadata,
    }
    np.savez_compressed(
        path,
        event_trace_id=batch.plan.events.trace_id,
        event_channel_id=batch.plan.events.channel_id,
        event_time_ms=batch.plan.events.time_ms,
        channel_weights=batch.plan.channel_weights,
        voltage_mv=batch.voltage_mv,
        time_ms=batch.time_ms,
        spike_trace_id=spike_trace,
        spike_time_ms=spike_time,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False)),
    )


def load_dataset(path: str | Path) -> DatasetBatch:
    """Load and structurally validate a sparse DBNN dataset batch."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "event_trace_id",
            "event_channel_id",
            "event_time_ms",
            "channel_weights",
            "voltage_mv",
            "time_ms",
            "spike_trace_id",
            "spike_time_ms",
            "manifest_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Dataset is missing fields: {sorted(missing)}.")
        manifest = json.loads(str(data["manifest_json"]))
        if manifest.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError(f"Unsupported DBNN dataset format version: {manifest.get('format_version')!r}.")
        events = SparseEvents(data["event_trace_id"], data["event_channel_id"], data["event_time_ms"])
        plan = StimulusPlan(
            layout_fingerprint=manifest["layout_fingerprint"],
            n_traces=int(manifest["n_traces"]),
            events=events,
            channel_weights=np.array(data["channel_weights"], copy=True),
            duration_ms=float(manifest["duration_ms"]),
            seeds=tuple(manifest["seeds"]),
            protocol_labels=tuple(manifest["protocol_labels"]),
            split=manifest["split"],
        )
        if plan.n_channels != int(manifest["n_channels"]):
            raise ValueError("Dataset manifest channel count does not match channel_weights.")
        spike_trace = np.asarray(data["spike_trace_id"])
        spike_time = np.asarray(data["spike_time_ms"])
        spikes = tuple(np.array(spike_time[spike_trace == trace], copy=True) for trace in range(plan.n_traces))
        return DatasetBatch(
            plan,
            np.array(data["voltage_mv"], copy=True),
            spikes,
            np.array(data["time_ms"], copy=True),
            dict(manifest["metadata"]),
        )


def run_teacher_batch(teacher: TeacherSpec, layout: ChannelLayout, plan: StimulusPlan) -> DatasetBatch:
    """Run a small detailed-Cell teacher batch through existing Network APIs."""
    import braincell

    _validate_teacher_plan(layout, plan)
    dt_ms = float(np.asarray(teacher.dt.to_decimal(u.ms)).reshape(()))
    cell = teacher.cell_factory((plan.n_traces,))

    synapse_views = []
    channel_id_columns = []
    for spec_index, spec in enumerate(layout.specs):
        view = cell.synapses.by_name(spec.instance_name).by_type(spec.synapse_type)
        channel_ids = validate_channel_coverage(layout, spec_index, view)
        synapse_views.append(view)
        channel_id_columns.append(channel_ids)
    cell.loc(teacher.output_location).record(
        "dbnn_teacher_voltage",
        braincell.observe.state("v"),
        period=teacher.dt,
    )
    source_index = plan.events.trace_id * layout.n_channels + plan.events.channel_id
    sequence = braincell.EventSequence(
        size=plan.n_traces * layout.n_channels,
        events=braincell.EventTable(source_index=source_index, time=plan.events.time_ms * u.ms),
        name="dbnn_teacher_inputs",
    )
    network = braincell.Network(name="dbnn_teacher_batch")
    network.add_population("input", sequence)
    network.add_population("teacher", cell)
    for spec_index, (spec, view, channel_ids) in enumerate(zip(layout.specs, synapse_views, channel_id_columns)):
        if len(view) == 0:
            continue
        populations = np.asarray(view.population_index, dtype=np.int64)
        sources = sequence[populations * layout.n_channels + channel_ids]
        weights = plan.channel_weights[populations, channel_ids] * layout.reference_weights_us[spec_index] * u.uS
        network.connect(
            f"dbnn_teacher_{spec_index}",
            source=sources,
            synapse=view,
            weight=weights,
        )
    result = network.run(dt=teacher.dt, duration=plan.duration_ms * u.ms)
    sample = result.samples["teacher"]["dbnn_teacher_voltage"]
    voltage = np.asarray(sample.values.to_decimal(u.mV), dtype=np.float32)
    if voltage.shape[1] != plan.n_traces:
        raise ValueError("Teacher output location must resolve exactly once per population member.")
    final_voltage = []
    for row in sample.schema.rows:
        final_voltage.append(cell.V.value[int(row.population_index), int(row.cv_id)].to_decimal(u.mV))
    voltage = np.concatenate((voltage, np.asarray(final_voltage, dtype=np.float32)[None, :]), axis=0).T
    return _make_teacher_batch(teacher, plan, voltage, dt_ms)


def _validate_teacher_plan(layout: ChannelLayout, plan: StimulusPlan) -> None:
    if plan.layout_fingerprint != layout.fingerprint:
        raise ValueError("Teacher stimulus plan belongs to a different channel layout.")
    if plan.n_channels != layout.n_channels:
        raise ValueError("Teacher stimulus channel count does not match the channel layout.")


def _resolve_output_cv(cell, output_location) -> int:
    from braincell._discretization.base import locate_cv_on_branch

    mask = output_location.evaluate(cell.morpho)
    if len(mask) != 1:
        raise ValueError("Teacher output location must resolve exactly once per population member.")
    branch_id = int(mask.branch_id[0])
    return int(
        locate_cv_on_branch(
            cell.cv_tree.branch_to_cv_ids[branch_id],
            cell.cvs,
            x=float(mask.branch_x[0]),
        )
    )


def _build_dense_routes(cell, layout: ChannelLayout) -> tuple:
    entries_by_layout = {}
    synapse_store = cell._get_synapse_store()
    for spec_index, spec in enumerate(layout.specs):
        view = cell.synapses.by_name(spec.instance_name).by_type(spec.synapse_type)
        channel_ids = validate_channel_coverage(layout, spec_index, view)
        if len(view) == 0:
            continue
        layout_id = synapse_store.layout_id(spec.synapse_type)
        entries_by_layout.setdefault(layout_id, []).append(
            (
                synapse_store.runtime_rows(view.id).astype(np.int32),
                np.asarray(view.population_index, dtype=np.int32),
                np.asarray(channel_ids, dtype=np.int32),
                float(layout.reference_weights_us[spec_index]),
            )
        )
    routes = []
    for layout_id, entries in entries_by_layout.items():
        buffer_size = int(np.prod(cell.runtime.get_event_buffer(layout_id).shape))
        routes.append(
            (
                cell.runtime.get_runtime_node(layout_id),
                buffer_size,
                tuple(entries),
            )
        )
    return tuple(routes)


def _make_teacher_batch(
    teacher: TeacherSpec,
    plan: StimulusPlan,
    voltage: np.ndarray,
    dt_ms: float,
) -> DatasetBatch:
    # Keep unstable candidate traces representable so the caller's acceptance
    # predicate can reject them without discarding finite traces in the batch.
    voltage = np.nan_to_num(np.asarray(voltage), nan=1e6, posinf=1e6, neginf=-1e6)
    time_ms = np.arange(voltage.shape[1], dtype=np.float32) * dt_ms
    above = voltage >= teacher.spike_threshold_mv
    spikes = tuple(
        (np.flatnonzero(above[trace, 1:] & ~above[trace, :-1]) + 1).astype(np.float32) * dt_ms
        for trace in range(plan.n_traces)
    )
    metadata = {
        "input_alignment": "target-step",
        "source_fingerprint": teacher.source_fingerprint,
        "dt_ms": dt_ms,
        "spike_threshold_mv": teacher.spike_threshold_mv,
    }
    return DatasetBatch(plan, voltage, spikes, time_ms, metadata)


__all__ = [
    "DATASET_FORMAT_VERSION",
    "DatasetBatch",
    "TeacherSpec",
    "TeacherSession",
    "load_dataset",
    "rasterize_events",
    "run_teacher_batch",
    "save_dataset",
    "validate_dataset",
]
