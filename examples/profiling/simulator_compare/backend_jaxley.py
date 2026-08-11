"""Jaxley backend for the simulator comparison."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
import jaxley.channels.hh as hh_module
import jaxley.solver_gate as gate_module

from common import (
    DT_MS,
    MORPHOLOGY_PATH,
    N_BRANCHES,
    N_CV,
    N_CV_PER_BRANCH,
    N_STEPS,
    PROBE_BRANCHES,
    PROBE_X,
    STIM_END_STEP,
    STIM_START_STEP,
    TOTAL_LENGTH_UM,
    assert_morphology_asset,
    add_throughput,
    base_metadata,
    current_amplitudes,
    git_commit,
    timing_summary,
    write_json,
)


def _save_exp_jax_010_compatible(x, max_value: float = 20.0):
    """Use JAX 0.10's renamed ``clip`` bounds with unchanged HH math."""
    return jnp.exp(jnp.clip(x, max=max_value))


def _vtrap_stable(x, y):
    """Evaluate ``x / expm1(x/y)`` at its removable zero singularity."""
    ratio = x / y
    return jnp.where(
        jnp.abs(ratio) < 1e-6,
        y * (1.0 - ratio / 2.0),
        x / jnp.expm1(ratio),
    )


# Jaxley 0.13 uses the removed ``a_max=`` spelling. HH methods resolve this
# module global at call time, so the compatibility fix does not touch Jaxley.
hh_module.save_exp = _save_exp_jax_010_compatible
gate_module.save_exp = _save_exp_jax_010_compatible
hh_module._vtrap = _vtrap_stable


def build_cell():
    cell = jx.read_swc(str(MORPHOLOGY_PATH), N_CV_PER_BRANCH, max_branch_len=2000.0)
    cell.insert(HH())
    for branch in PROBE_BRANCHES:
        cell.branch(branch).loc(PROBE_X).record(verbose=False)
    cell.set("axial_resistivity", 1000.0)
    cell.set("capacitance", 1.0)
    cell.set("v", -62.0)
    cell.init_states(delta_t=DT_MS)
    branch_lengths = cell.nodes.groupby("global_branch_index").length.sum().to_numpy()
    morphology = {
        "n_branches": int(len(branch_lengths)),
        "n_cv": int(len(cell.nodes)),
        "total_length_um": float(np.sum(branch_lengths)),
        "total_area_um2": float(np.sum(cell.nodes.area.to_numpy())),
    }
    if morphology["n_branches"] != N_BRANCHES or morphology["n_cv"] != N_CV:
        raise RuntimeError(f"unexpected Jaxley discretization: {morphology}")
    if not np.isclose(morphology["total_length_um"], TOTAL_LENGTH_UM, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"unexpected Jaxley morphology length: {morphology['total_length_um']}")
    return cell, morphology


def make_currents(batch_size: int):
    amplitudes = jnp.asarray(current_amplitudes(batch_size), dtype=jnp.float32)
    currents = jnp.zeros((batch_size, N_STEPS), dtype=jnp.float32)
    return currents.at[:, STIM_START_STEP:STIM_END_STEP].set(amplitudes[:, None])


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
) -> dict:
    assert_morphology_asset()
    build_start = time.perf_counter()
    cell, morphology = build_cell()
    currents = make_currents(batch_size)

    def simulate_one(current):
        stimuli = cell.branch(0).loc(PROBE_X).data_stimulate(current, None)
        return jx.integrate(cell, delta_t=DT_MS, data_stimuli=stimuli)[:, 1:]

    simulate = jax.jit(jax.vmap(simulate_one))
    build_seconds = time.perf_counter() - build_start
    compile_start = time.perf_counter()
    result = simulate(currents)
    jax.block_until_ready(result)
    compilation_seconds = time.perf_counter() - compile_start
    for _ in range(warmup):
        result = simulate(currents)
        jax.block_until_ready(result)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = simulate(currents)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)

    transfer_samples = []
    host_result = None
    for _ in range(transfer_repeat):
        result = simulate(currents)
        jax.block_until_ready(result)
        start = time.perf_counter()
        host_result = jax.device_get(result)
        transfer_samples.append(time.perf_counter() - start)
    if host_result is None:
        host_result = jax.device_get(result)

    all_finite = bool(np.isfinite(host_result).all())
    if not all_finite:
        raise RuntimeError("Jaxley produced non-finite values")
    trace = np.asarray(host_result[0])
    if trace.shape != (len(PROBE_BRANCHES), N_STEPS):
        raise RuntimeError(f"unexpected Jaxley trace shape: {trace.shape}")
    payload = base_metadata("jaxley", batch_size)
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
                "jaxley": jx.__version__,
                "jax": jax.__version__,
                "jaxlib": jax.lib.__version__,
                "jaxley_git_commit": git_commit(Path(jx.__file__).resolve().parents[1]),
                "compatibility": [
                    "HH save_exp uses jnp.clip(max=...) for JAX >= 0.10",
                    "HH vtrap uses its analytic zero limit to avoid intermittent 0/0",
                ],
            },
            "device": str(jax.devices()[0]),
            "device_memory": _memory_metadata(),
            "execution": {
                "jitted": True,
                "jit_entrypoint": "jax.jit(jax.vmap(jaxley.integrate))",
                "synchronization": "jax.block_until_ready(result)",
                "host_transfer_included_in_primary_timing": False,
            },
            "output_validation": {
                "all_finite": all_finite,
                "trace_output_bytes": int(host_result.nbytes),
                "first_amplitude_na": current_amplitudes(batch_size)[0],
                "last_amplitude_na": current_amplitudes(batch_size)[-1],
            },
        }
    )
    if include_trace:
        payload["trace_mv"] = trace.tolist()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--transfer-repeat", type=int, default=3)
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
        ),
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
