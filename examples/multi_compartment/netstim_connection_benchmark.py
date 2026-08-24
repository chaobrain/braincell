"""Scaling benchmark for heterogeneous synapse placement and NetStim connections."""

from __future__ import annotations

import argparse
import json
import time

import brainunit as u
import jax
import numpy as np

import braincell
from braincell.filter import AllRegion, at


def build_morphology():
    """Build the three-branch morphology shared with the example notebook."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = braincell.Branch.from_lengths(
        lengths=[120.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    dend_b = braincell.Branch.from_lengths(
        lengths=[160.0] * u.um,
        radii=[2.5, 0.8] * u.um,
        type="basal_dendrite",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend_a = dend_a
    morphology.soma.dend_b = dend_b
    return morphology


def build_model(size: int, *, heterogeneous: bool = True):
    """Build a real HH population with two synapses and sources per cell."""
    size = int(size)
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranchList([1, 2, 2]),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        braincell.mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        braincell.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.387 * u.mV),
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    exp = braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
    if heterogeneous:
        patterns = np.asarray([[0, 1], [0, 3], [1, 4], [2, 3]], dtype=np.int32)
        cell.place(cell.cv_midpoints[patterns[np.arange(size) % len(patterns)]], exp)
    else:
        cell.place(cell.cv_midpoints[[0, 1]], exp)

    targets = cell.synapses[exp]
    if heterogeneous:
        targets.set(
            tau=(1.0 + 0.02 * np.arange(len(targets))) * u.ms,
            e=(-10.0 - 0.1 * np.arange(len(targets))) * u.mV,
        )
    sources = braincell.NetStim(
        size=len(targets),
        start=(1.0 + 0.01 * (np.arange(len(targets)) % 5)) * u.ms,
        number=1,
        interval=10.0 * u.ms,
        noise=0.0,
    )
    connection = braincell.connect(
        "benchmark_input",
        source=sources,
        synapse=targets,
        weight=(0.02 + 0.0001 * np.arange(len(targets))) * u.uS,
        delay=(0.05 * (np.arange(len(targets)) % 3)) * u.ms,
    )
    return cell, sources, connection


def _block(result) -> None:
    for value in result.traces.values():
        jax.block_until_ready(value)


def _nbytes(value) -> int:
    array = value.mantissa if isinstance(value, u.Quantity) else value
    return int(np.asarray(array).nbytes)


def storage_summary(cell, sources, connection) -> dict[str, int]:
    """Return persistent source/connection/synapse storage in bytes."""
    synapse_layout_ids = {layout.id for layout in cell.runtime.layouts if layout.kind.startswith("synapse:")}
    synapse_bytes = sum(
        _nbytes(value)
        for (layout_id, _), value in cell.runtime.state_buffers.items()
        if layout_id in synapse_layout_ids
    )
    source_bytes = sources._event_times_ms.nbytes + sources._event_mask.nbytes
    connection_bytes = sum(
        _nbytes(value)
        for value in (
            connection.id,
            connection.source_index,
            connection.target_index,
            connection.weight,
            connection.delay,
        )
    )
    return {
        "source_bytes": int(source_bytes),
        "connection_bytes": int(connection_bytes),
        "synapse_bytes": int(synapse_bytes),
        "total_bytes": int(source_bytes + connection_bytes + synapse_bytes),
    }


def benchmark(size: int, *, heterogeneous: bool, steps: int = 5, repeats: int = 3):
    """Measure declaration, initialization, compilation, warm run, and storage."""
    started = time.perf_counter()
    cell, sources, connection = build_model(size, heterogeneous=heterogeneous)
    declaration_s = time.perf_counter() - started

    started = time.perf_counter()
    cell.init_state()
    init_s = time.perf_counter() - started
    storage = storage_summary(cell, sources, connection)

    duration = int(steps) * 0.05 * u.ms
    started = time.perf_counter()
    _block(cell.run(dt=0.05 * u.ms, duration=duration))
    first_run_s = time.perf_counter() - started
    warm_times = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        _block(cell.run(dt=0.05 * u.ms, duration=duration))
        warm_times.append(time.perf_counter() - started)

    return {
        "platform": jax.default_backend(),
        "mode": "heterogeneous" if heterogeneous else "broadcast",
        "population_size": int(size),
        "synapse_instances": len(cell.synapses),
        "connections": len(connection),
        "declaration_s": declaration_s,
        "init_s": init_s,
        "first_run_s": first_run_s,
        "warm_run_ms": 1000.0 * float(np.median(warm_times)),
        **storage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--mode", choices=("broadcast", "heterogeneous"), required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.size,
                heterogeneous=args.mode == "heterogeneous",
                steps=args.steps,
                repeats=args.repeats,
            )
        )
    )


if __name__ == "__main__":
    main()
