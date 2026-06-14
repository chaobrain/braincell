"""Child process for the network event backend benchmark notebook.

The benchmark launches this script in a fresh Python interpreter for each
platform/backend case so ``JAX_PLATFORMS`` is set before importing JAX.
Configuration is read from stdin as JSON and the final status is written to
stdout as JSON.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))


def main() -> int:
    cfg = json.loads(sys.stdin.read())

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "error": f"JAX import failed: {type(exc).__name__}: {exc}",
            }
        )
        return 0

    try:
        default_backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
    except Exception as exc:
        _emit(
            {
                "status": "skipped",
                "error": f"JAX device initialization failed: {type(exc).__name__}: {exc}",
            }
        )
        return 0

    if cfg["platform"] == "cuda" and default_backend != "gpu":
        _emit(
            {
                "status": "skipped",
                "error": f"Requested CUDA but JAX default backend is {default_backend!r}; devices={devices!r}",
                "default_backend": default_backend,
                "devices": devices,
            }
        )
        return 0

    n_pre = int(cfg["n_pre"])
    n_post = int(cfg["n_post"])
    n_active = int(cfg["n_active"])
    edges = int(cfg["edges"])
    target_size = n_post * n_active
    rng = np.random.default_rng(int(cfg["seed"]))

    pre_index = jnp.asarray(rng.integers(0, n_pre, size=edges, dtype=np.int32))
    target_index = jnp.asarray(rng.integers(0, target_size, size=edges, dtype=np.int32))
    weight = jnp.asarray(rng.uniform(0.1, 1.0, size=edges).astype(np.float32))
    pre_spike = jnp.asarray((rng.random(n_pre) < 0.1).astype(np.float32))

    event_backend = cfg["event_backend"]
    brainevent_backend = cfg.get("brainevent_backend")

    if event_backend == "scatter":

        def op(pre_spike, weight, pre_index, target_index):
            edge_event = pre_spike[pre_index] * weight
            return jnp.zeros((target_size,), dtype=edge_event.dtype).at[target_index].add(edge_event)

    elif event_backend == "brainevent":
        try:
            import brainevent
        except Exception as exc:
            _emit({"status": "skipped", "error": f"brainevent import failed: {type(exc).__name__}: {exc}"})
            return 0
        if not hasattr(brainevent, "coomv"):
            _emit({"status": "skipped", "error": "brainevent.coomv is unavailable"})
            return 0

        def op(pre_spike, weight, pre_index, target_index):
            return brainevent.coomv(
                weight,
                pre_index,
                target_index,
                pre_spike,
                shape=(n_pre, target_size),
                transpose=True,
                backend=brainevent_backend,
            )

    else:
        _emit({"status": "error", "error": f"unknown event_backend={event_backend!r}"})
        return 0

    jit_op = jax.jit(op)

    try:
        for _ in range(int(cfg["warmup"])):
            out = jit_op(pre_spike, weight, pre_index, target_index)
            out.block_until_ready()

        times = []
        for _ in range(int(cfg["repeats"])):
            start = time.perf_counter()
            out = jit_op(pre_spike, weight, pre_index, target_index)
            out.block_until_ready()
            times.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "error": f"benchmark failed: {type(exc).__name__}: {exc}",
                "default_backend": default_backend,
                "devices": devices,
            }
        )
        return 0

    times_arr = np.asarray(times, dtype=float)
    _emit(
        {
            "status": "ok",
            "median_ms": float(np.median(times_arr)),
            "min_ms": float(np.min(times_arr)),
            "std_ms": float(np.std(times_arr)),
            "output_sum": float(np.asarray(out).sum()),
            "default_backend": default_backend,
            "devices": devices,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
