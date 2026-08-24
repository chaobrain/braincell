"""Non-gating CPU benchmark for endpoint pairing materialization."""

from __future__ import annotations

import argparse
import json
import time

import brainunit as u
import numpy as np

import braincell
from braincell.filter import at


def build_target(size: int):
    """Build one target population with one logical synapse per cell."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(size,),
    )
    synapse = braincell.mech.SynapseSpec(
        "ExpSyn",
        name="bench_ampa",
        tau=2.0 * u.ms,
        e=0.0 * u.mV,
    )
    cell.place(at("soma", 0.5), synapse)
    return cell, synapse


def benchmark(strategy: str, *, source_size: int, target_size: int, rows: int, seed: int):
    """Build one connection and report declaration-time cost."""
    cell, synapse = build_target(target_size)
    source = braincell.NetStim(size=source_size, start=1.0 * u.ms)
    if strategy == "independent":
        pairing = braincell.connection.independent(rows, seed=seed)
    elif strategy == "source_first":
        pairing = braincell.connection.source_first(rows, seed=seed)
    elif strategy == "by_source":
        degree = np.full(source_size, rows // source_size, dtype=np.int64)
        degree[: rows % source_size] += 1
        pairing = braincell.connection.by_source(degree, seed=seed)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}.")

    started = time.perf_counter()
    connection = braincell.connect(
        "benchmark",
        source=source,
        synapse=cell.synapses[synapse],
        pairing=pairing,
    )
    elapsed = time.perf_counter() - started
    return {
        "strategy": strategy,
        "source_size": source_size,
        "target_size": target_size,
        "rows": len(connection),
        "seconds": elapsed,
        "rows_per_second": len(connection) / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("independent", "source_first", "by_source"), required=True)
    parser.add_argument("--source-size", type=int, default=1_000)
    parser.add_argument("--target-size", type=int, default=1_000)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.strategy,
                source_size=args.source_size,
                target_size=args.target_size,
                rows=args.rows,
                seed=args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()
