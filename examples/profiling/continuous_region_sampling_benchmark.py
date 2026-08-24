"""Non-gating benchmark for continuous morphology-region sampling.

The default grid covers 1, 100, and 1,000 morphology components and 1,000,
100,000, and 1,000,000 output locations. Results are emitted as JSON lines so
they can be compared without imposing unstable wall-time limits in CI.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

import brainunit as u
import numpy as np

import braincell
from braincell.filter import AllRegion, sample


def build_morphology(components: int) -> braincell.Morphology:
    """Build one branch divided into equal constant-radius segments."""
    if components <= 0:
        raise ValueError("components must be positive.")
    branch = braincell.Branch.from_lengths(
        lengths=np.ones(components) * u.um,
        radii=np.ones(components + 1) * u.um,
        type="dendrite",
    )
    return braincell.Morphology.from_root(branch, name="dend")


def benchmark(components: int, samples: int, *, custom_density: bool) -> dict[str, object]:
    """Measure public-API setup, inversion, output construction, and peak memory."""
    morphology = build_morphology(components)
    preference = (lambda context: 1.0 + 0.25 * context.branch_x) if custom_density else None
    expression = sample(
        AllRegion(),
        number=samples,
        seed=1,
        measure="length",
        density=preference,
    )

    tracemalloc.start()
    started = time.perf_counter()
    result = expression.evaluate(morphology)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "components": components,
        "samples": samples,
        "density": "custom_linear" if custom_density else "uniform",
        "elapsed_s": elapsed,
        "peak_bytes": peak_bytes,
        "result_rows": len(result),
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=_csv_ints, default=(1, 100, 1000))
    parser.add_argument("--samples", type=_csv_ints, default=(1_000, 100_000, 1_000_000))
    parser.add_argument("--custom-density", action="store_true")
    args = parser.parse_args()
    for components in args.components:
        for samples in args.samples:
            print(json.dumps(benchmark(components, samples, custom_density=args.custom_density)))


if __name__ == "__main__":
    main()
