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

"""BrainCell backend for the simulator comparison."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranch
from braincell.filter import AllRegion, at

from common import (
    DT_MS,
    MORPHOLOGY_PATH,
    N_BRANCHES,
    N_CV,
    N_CV_PER_BRANCH,
    N_STEPS,
    PROBE_BRANCHES,
    PROBE_X,
    STIM_DELAY_MS,
    STIM_DURATION_MS,
    TEMPERATURE_C,
    TOTAL_LENGTH_UM,
    TSTOP_MS,
    V_INIT_MV,
    assert_morphology_asset,
    add_throughput,
    base_metadata,
    current_amplitudes,
    git_commit,
    timing_summary,
    write_json,
)


def build_cell(batch_size: int, *, linearizer: str = "point") -> tuple[Cell, dict]:
    brainstate.environ.set(precision=32)
    morph = Morphology.from_swc(MORPHOLOGY_PATH, mode="neuron")
    cell = Cell(
        morph,
        pop_size=(batch_size,),
        cv_policy=CVPerBranch(N_CV_PER_BRANCH),
        V_init=V_INIT_MV * u.mV,
        solver="staggered",
        membrane_linearizer=linearizer,
        name=f"hh_n144_n{batch_size}",
    )
    temperature = u.celsius2kelvin(TEMPERATURE_C)
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-54.3 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=1000.0 * (u.ohm * u.cm),
            temperature=temperature,
        ),
        mech.Ion("SodiumFixed", name="na", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", name="k", E=-77.0 * u.mV),
        mech.Channel(
            "Na_HH1952",
            name="na_hh",
            ion_name="na",
            g_max=120.0 * (u.mS / u.cm**2),
            temp=temperature,
            temp_ref=temperature,
        ),
        mech.Channel(
            "K_HH1952",
            name="k_hh",
            ion_name="k",
            g_max=36.0 * (u.mS / u.cm**2),
            temp=temperature,
            temp_ref=temperature,
        ),
        mech.Channel("IL", name="leak", g_max=0.3 * (u.mS / u.cm**2), E=-54.3 * u.mV),
    )
    for branch in PROBE_BRANCHES:
        cell.place(at(branch, PROBE_X), mech.StateProbe(name=f"v_b{branch}"))
    amplitudes = u.Quantity(jnp.asarray(current_amplitudes(batch_size)), u.nA)
    cell.place(
        at(0, PROBE_X),
        mech.CurrentClamp(
            delay=STIM_DELAY_MS * u.ms,
            durations=STIM_DURATION_MS * u.ms,
            amplitudes=amplitudes,
        ),
    )
    cell.init_state()
    length_um = sum(float(branch.length.to_decimal(u.um)) for branch in morph.branches)
    area_um2 = sum(float(cv.area.to_decimal(u.um**2)) for cv in cell.cvs)
    morphology = {
        "n_branches": len(morph.branches),
        "n_cv": cell.n_cv,
        "total_length_um": length_um,
        "total_area_um2": area_um2,
    }
    if morphology["n_branches"] != N_BRANCHES or morphology["n_cv"] != N_CV:
        raise RuntimeError(f"unexpected BrainCell discretization: {morphology}")
    if not np.isclose(length_um, TOTAL_LENGTH_UM, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"unexpected BrainCell morphology length: {length_um}")
    return cell, morphology


def block(result) -> None:
    jax.block_until_ready(result)


def _memory_metadata() -> dict:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "peak_mib_in_use": float(stats.get("peak_bytes_in_use", 0)) / (1024**2),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }


def run_backend(
    batch_size: int,
    warmup: int,
    repeat: int,
    include_trace: bool,
    transfer_repeat: int,
    linearizer: str = "point",
) -> dict:
    assert_morphology_asset()
    build_start = time.perf_counter()
    cell, morphology = build_cell(batch_size, linearizer=linearizer)
    build_seconds = time.perf_counter() - build_start

    def simulate():
        cell.reset_state()
        return cell.run(dt=DT_MS * u.ms, duration=TSTOP_MS * u.ms)

    compile_start = time.perf_counter()
    result = simulate()
    block(result)
    compilation_seconds = time.perf_counter() - compile_start
    for _ in range(warmup):
        result = simulate()
        block(result)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = simulate()
        block(result)
        samples.append(time.perf_counter() - start)

    transfer_samples = []
    host_result = None
    for _ in range(transfer_repeat):
        result = simulate()
        block(result)
        start = time.perf_counter()
        host_result = jax.device_get(result)
        transfer_samples.append(time.perf_counter() - start)
    if host_result is None:
        host_result = jax.device_get(result)

    host_traces = [
        np.asarray(host_result.traces[f"v_b{branch}"].to_decimal(u.mV))
        for branch in PROBE_BRANCHES
    ]
    traces = np.stack([trace[:, 0] for trace in host_traces])
    all_finite = all(bool(np.isfinite(trace).all()) for trace in host_traces)
    trace_output_bytes = sum(trace.nbytes for trace in host_traces)
    if not all_finite:
        raise RuntimeError("BrainCell produced non-finite values")
    payload = base_metadata("braincell", batch_size)
    payload.update(
        {
            "timing": add_throughput(timing_summary(samples), batch_size=batch_size),
            "host_transfer": {
                **timing_summary(transfer_samples),
                "included_in_primary_timing": False,
            }
            if transfer_samples
            else None,
            "build_seconds": build_seconds,
            "compilation_seconds": compilation_seconds,
            "morphology": morphology,
            "output_shape": [batch_size, len(PROBE_BRANCHES), N_STEPS],
            "software": {
                "python": platform.python_version(),
                "braincell": braincell.__version__,
                "jax": jax.__version__,
                "jaxlib": jax.lib.__version__,
                "braincell_git_commit": git_commit(Path(braincell.__file__).resolve().parents[1]),
            },
            "device": str(jax.devices()[0]),
            "device_memory": _memory_metadata(),
            "execution": {
                "jitted": True,
                "jit_entrypoint": "brainstate.transform.jit(Cell.run loop)",
                "membrane_linearizer": linearizer,
                "synchronization": "jax.block_until_ready(all RunResult pytree leaves)",
                "host_transfer_included_in_primary_timing": False,
            },
            "output_validation": {
                "all_finite": all_finite,
                "trace_output_bytes": trace_output_bytes,
                "first_amplitude_na": current_amplitudes(batch_size)[0],
                "last_amplitude_na": current_amplitudes(batch_size)[-1],
            },
        }
    )
    if include_trace:
        payload["trace_mv"] = traces.tolist()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--transfer-repeat", type=int, default=3)
    parser.add_argument("--linearizer", choices=("point", "generic"), default="point")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.repeat <= 0 or args.transfer_repeat < 0:
        parser.error("batch-size/repeat must be positive and warmup/transfer-repeat non-negative")
    write_json(
        args.output,
        run_backend(
            args.batch_size,
            args.warmup,
            args.repeat,
            args.include_trace,
            args.transfer_repeat,
            args.linearizer,
        ),
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
