"""CPU benchmark used by heterogeneous_synapses.ipynb.

Run one configuration per process so JAX compilation caches do not leak across
storage layouts. The script prints one JSON record to stdout.
"""

from __future__ import annotations

import argparse
import json
import time

import brainunit as u
import jax
import numpy as np

import braincell
from braincell.filter import at


def build_morphology():
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
        lengths=[180.0] * u.um,
        radii=[2.5, 0.8] * u.um,
        type="basal_dendrite",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend_a = dend_a
    morphology.soma.dend_b = dend_b
    return morphology


def hold_voltage(cell):
    cell.V.value = cell.V.value


def synapse_declaration():
    return braincell.mech.Synapse(
        "ExpSyn",
        tau=2.0 * u.ms,
        e=0.0 * u.mV,
        weight=0.1 * u.uS,
        name="bench_exp",
    )


def locations(number: int):
    names = ("soma", "dend_a", "dend_b")
    return tuple(at(names[index % 3], 0.15 + 0.7 * ((index * 7) % 17) / 16.0) for index in range(number))


def counts_for(mode: str, size: int) -> np.ndarray:
    if mode in {"broadcast", "packed_uniform"}:
        return np.full(size, 4, dtype=np.int32)
    if mode == "packed_heterogeneous":
        return np.resize(np.asarray([0, 1, 2, 13], dtype=np.int32), size)
    raise ValueError(f"Unknown benchmark mode {mode!r}.")


def build_cell(mode: str, size: int):
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=hold_voltage,
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    synapse = synapse_declaration()
    counts = counts_for(mode, size)
    if mode == "broadcast":
        locset = locations(4)[0]
        for location in locations(4)[1:]:
            locset = locset | location
        cell.place(locset, synapse)
    else:
        available = locations(int(np.max(counts, initial=0)))
        for post_index, count in enumerate(counts.tolist()):
            for location in available[:count]:
                cell[post_index].place(location, synapse)
    return cell, counts


def block(result) -> None:
    for value in result.traces.values():
        jax.block_until_ready(value)


def synapse_storage(cell) -> tuple[int, int, tuple[int, ...]]:
    layout, _ = next(
        (layout, node)
        for layout, node in cell.runtime.iter_synapse_layouts()
        if cell.runtime.get_layout_mechanism(layout.id).instance_name == "bench_exp"
    )
    elements = 0
    nbytes = 0
    for (layout_id, _), value in cell.runtime.state_buffers.items():
        if layout_id != layout.id:
            continue
        array = np.asarray(value.mantissa if isinstance(value, u.Quantity) else value)
        elements += int(array.size)
        nbytes += int(array.nbytes)
    shape = tuple(int(value) for value in cell.runtime.get_state(layout.id, "pre_spike").shape)
    return elements, nbytes, shape


def benchmark(mode: str, size: int, *, repeats: int, steps: int) -> dict[str, object]:
    started = time.perf_counter()
    cell, counts = build_cell(mode, size)
    declaration_s = time.perf_counter() - started

    started = time.perf_counter()
    cell.init_state()
    init_s = time.perf_counter() - started
    elements, nbytes, state_shape = synapse_storage(cell)

    duration = steps * 0.1 * u.ms
    started = time.perf_counter()
    block(cell.run(dt=0.1 * u.ms, duration=duration))
    first_run_s = time.perf_counter() - started

    warm = []
    for _ in range(repeats):
        started = time.perf_counter()
        block(cell.run(dt=0.1 * u.ms, duration=duration))
        warm.append(time.perf_counter() - started)

    return {
        "mode": mode,
        "pop_size": size,
        "logical_instances": int(np.sum(counts)),
        "max_count": int(np.max(counts, initial=0)),
        "padded_slots": int(size * np.max(counts, initial=0)),
        "pre_spike_shape": state_shape,
        "synapse_buffer_elements": elements,
        "synapse_buffer_bytes": nbytes,
        "declaration_s": declaration_s,
        "init_s": init_s,
        "first_run_s": first_run_s,
        "warm_run_ms": 1000.0 * float(np.median(warm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("broadcast", "packed_uniform", "packed_heterogeneous"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.mode, args.size, repeats=args.repeats, steps=args.steps)))


if __name__ == "__main__":
    main()
