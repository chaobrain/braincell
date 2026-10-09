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
# ============================================================================

"""Train a DBNN-GIF replacement for the final layer of a small HH network."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import brainunit as u
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import braincell
from braincell import mech
from braincell.filter import AllRegion, at
from braincell.reduction.dbnn import DBNN, DBNNGIF, DBNNReduction, DBNNTrainer
from braincell.reduction.dbnn.dataset import rasterize_events
from braincell.reduction.dbnn.layout import build_channel_layout
from braincell.reduction.dbnn.stimulus import SparseEvents, StimulusPlan, generate_multichannel_protocol


DT = 1.0 * u.ms
DURATION_MS = 100.0
N_TRACES = 120
INPUT_NAMES = tuple(f"input_{index}" for index in range(10))
INPUT_WEIGHTS = np.asarray((0.08, 0.05) * 5, dtype=np.float32)
INPUT_RATES_HZ = np.asarray((45.0, 30.0) * 5, dtype=np.float32)
OUTPUT_DIR = REPO_ROOT / "artifacts" / "reduction" / "DBNN_test_network"
TIME_LIMIT_SECONDS = 9.0 * 60.0


def build_target_cell(pop_size, name: str) -> tuple[braincell.Cell, tuple[mech.Synapse, ...]]:
    """Build one two-input detailed HH target population."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        cv_policy=braincell.CVPerBranch(),
        pop_size=pop_size,
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        name=name,
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-54.3 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
    )
    specs = tuple(
        mech.Synapse("Exp2Syn", name=name, tau1=0.5 * u.ms, tau2=5.0 * u.ms, e=0.0 * u.mV)
        for name in INPUT_NAMES
    )
    cell.place(at("soma", 0.5), *specs)
    cell.record("voltage", braincell.observe.output("voltage"))
    return cell, specs


def make_network(plan: StimulusPlan, *, target: braincell.Cell, name: str) -> braincell.Network:
    """Connect one independent event source to each DBNN channel."""
    network = braincell.Network(name)
    target_population = network.add_population("target", target)
    positions = np.arange(plan.n_traces, dtype=np.int64)
    for channel, input_name in enumerate(INPUT_NAMES):
        selected = plan.events.channel_id == channel
        source = braincell.EventSequence(
            size=plan.n_traces,
            events=braincell.EventTable(
                source_index=plan.events.trace_id[selected],
                time=plan.events.time_ms[selected] * u.ms,
            ),
            name=input_name,
        )
        network.add_population(input_name, source)
        network.connect(
            f"{input_name}_to_target",
            source=source[positions],
            synapse=target_population.synapses[input_name][positions],
            weight=plan.channel_weights[:, channel] * INPUT_WEIGHTS[channel] * u.uS,
        )
    return network


def slice_trace(plan: StimulusPlan, trace_id: int) -> StimulusPlan:
    """Return one trace with source IDs remapped to a one-cell population."""
    selected = plan.events.trace_id == trace_id
    return StimulusPlan(
        layout_fingerprint=plan.layout_fingerprint,
        n_traces=1,
        events=SparseEvents(
            trace_id=np.zeros(np.count_nonzero(selected), dtype=np.int64),
            channel_id=plan.events.channel_id[selected],
            time_ms=plan.events.time_ms[selected],
        ),
        channel_weights=plan.channel_weights[trace_id : trace_id + 1],
        duration_ms=plan.duration_ms,
        seeds=(plan.seeds[trace_id],),
        protocol_labels=(plan.protocol_labels[trace_id],),
        split=plan.split,
    )


def spike_times(voltage_mv: np.ndarray, time_ms: np.ndarray) -> tuple[np.ndarray, ...]:
    """Detect detailed HH threshold crossings for GIF calibration."""
    return tuple(time_ms[1:][(row[1:] >= 0.0) & (row[:-1] < 0.0)] for row in voltage_mv)


def require_time_budget(start_time: float) -> None:
    """Fail before publishing results once the bounded runtime budget is exceeded."""
    if time.monotonic() - start_time > TIME_LIMIT_SECONDS:
        raise TimeoutError("DBNN_test_network exceeded its nine-minute internal runtime budget.")


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, float | int]:
    """Generate teacher data, train DBNN-GIF, and run the replacement network."""
    start_time = time.monotonic()
    template, specs = build_target_cell((1,), "template")
    layout = build_channel_layout(
        template,
        specs,
        reference_weights_us=tuple(float(value) for value in INPUT_WEIGHTS),
        training_ranges=((0.0, 1.0),) * len(INPUT_NAMES),
    )
    plan = generate_multichannel_protocol(
        layout,
        n_traces=N_TRACES,
        duration_ms=DURATION_MS,
        dt_ms=float(DT.to_decimal(u.ms)),
        rate_hz=INPUT_RATES_HZ,
        seed=20260908,
    )
    teacher, _ = build_target_cell((N_TRACES,), "teacher")
    teacher_result = make_network(plan, target=teacher, name="detailed_teacher").run(dt=DT, duration=DURATION_MS * u.ms)
    time_ms = np.asarray(teacher_result.time.to_decimal(u.ms), dtype=np.float32)
    voltage_mv = np.asarray(teacher_result.samples["target"]["voltage"].values.to_decimal(u.mV), dtype=np.float32).T
    inputs = rasterize_events(plan, dt_ms=float(DT.to_decimal(u.ms)), input_alignment="interval-start")[:, :, :-1]
    if voltage_mv.shape != (N_TRACES, inputs.shape[-1]):
        raise RuntimeError(f"Teacher voltage shape {voltage_mv.shape} does not match inputs {inputs.shape}.")
    data = {
        "inputs": inputs,
        "targets": voltage_mv,
        "spike_times_ms": spike_times(voltage_mv, time_ms),
    }
    train_data = {key: value[:80] if hasattr(value, "shape") else value[:80] for key, value in data.items()}
    validation_data = {key: value[80:100] if hasattr(value, "shape") else value[80:100] for key, value in data.items()}
    test_data = {key: value[100:] if hasattr(value, "shape") else value[100:] for key, value in data.items()}
    model = DBNN(layout, mode="f", dt=DT)
    model.source_fingerprint = "dbnn-test-network-hh-v1"
    model.input_alignment = "interval-start"
    params = model.get_params()
    params["bias"] = np.asarray(np.mean(train_data["targets"][:, 0]), dtype=np.float32)
    model.set_params(params)
    trainer = DBNNTrainer(
        model,
        learning_rate=0.01,
        lr_step_size=40,
        lr_gamma=0.5,
        loss="mse",
        seed=42,
        initialization_options={"maxiter": 2, "popsize": 2, "max_pairs": 1, "samples": 8},
    )
    trainer.search_initialization(train_data)
    trainer.fit(train_data, validation_data=validation_data, epochs=120, batch_size=8, patience=20)
    gif_model, gif_report = trainer.calibrate_gif(
        validation_data,
        thresholds_mv=np.linspace(-55.0, 20.0, 16),
        threshold_increments_mv=(0.0, 2.0, 5.0, 8.0),
        threshold_taus_ms=(10.0, 30.0, 80.0),
        match_window_ms=4.0,
    )
    require_time_budget(start_time)
    gif_model.set_mode("r")
    replacement, _ = build_target_cell((1,), "dbnn_target")
    replacement.add_reduction("dbnn", DBNNReduction(gif_model))
    replacement.use_model("dbnn")
    selected_trace = 20
    runtime_plan = slice_trace(plan, selected_trace)
    runtime_result = make_network(runtime_plan, target=replacement, name="dbnn_replacement").run(
        dt=DT,
        duration=DURATION_MS * u.ms,
    )
    runtime_voltage_mv = np.asarray(runtime_result.samples["target"]["voltage"].values.to_decimal(u.mV), dtype=np.float32)[:, 0]
    runtime_spikes = runtime_result.events["target"]["spike"]
    test_metrics = trainer.evaluate(test_data)
    elapsed_seconds = time.monotonic() - start_time
    require_time_budget(start_time)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "data" / "teacher_traces.npz",
        inputs=np.asarray(inputs),
        voltage_mv=voltage_mv,
        time_ms=time_ms,
    )
    gif_model.save(output_dir / "models" / "dbnn_gif_model.npz")
    report = {
        "channels": layout.n_channels,
        "teacher_traces": N_TRACES,
        "test_mse": float(np.asarray(test_metrics["mse"])),
        "gif_threshold_mv": float(gif_report["gif"]["threshold_mv"]),
        "runtime_samples": int(runtime_voltage_mv.size),
        "runtime_spike_records": int(len(runtime_spikes.time)),
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "reports" / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
