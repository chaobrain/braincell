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


def _benchmark_delay_queue(cfg, *, jax, jnp, np, default_backend, devices) -> int:
    """Benchmark canonical contact scatter against per-delay grouped rings."""
    n_pre = int(cfg["n_pre"])
    target_size = int(cfg["target_size"])
    edges = int(cfg["edges"])
    batch_size = int(cfg.get("batch_size", 1))
    unique_delays = int(cfg["unique_delays"])
    max_delay_steps = int(cfg["max_delay_steps"])
    activity = float(cfg["activity"])
    rng = np.random.default_rng(int(cfg["seed"]))
    delay_values = (
        np.asarray([max_delay_steps], dtype=np.int32)
        if unique_delays == 1
        else np.unique(np.rint(np.linspace(1, max_delay_steps, unique_delays)).astype(np.int32))
    )
    delay_steps_np = np.resize(delay_values, edges)
    rng.shuffle(delay_steps_np)
    pre_index_np = rng.integers(0, n_pre, size=edges, dtype=np.int32)
    target_index_np = rng.integers(0, target_size, size=edges, dtype=np.int32)
    weight_np = rng.uniform(0.1, 1.0, size=edges).astype(np.float32)
    pre_spike_np = (rng.random((batch_size, n_pre)) < activity).astype(np.float32)

    pre_index = jnp.asarray(pre_index_np)
    target_index = jnp.asarray(target_index_np)
    delay_steps = jnp.asarray(delay_steps_np)
    weight = jnp.asarray(weight_np)
    pre_spike = jnp.asarray(pre_spike_np)
    mode = cfg["delay_layout"]

    if mode == "shared_contact_scatter":
        ring_depth = int(np.max(delay_values)) + 1
        ring = jnp.zeros((batch_size, ring_depth, target_size), dtype=jnp.float32)
        batch_index = jnp.arange(batch_size, dtype=jnp.int32)[:, None]

        def op(pre_spike, ring):
            events = pre_spike[:, pre_index] * weight[None, :]
            return ring.at[batch_index, delay_steps[None, :], target_index[None, :]].add(events)

        args = (pre_spike, ring)
        queue_bytes = int(batch_size * ring_depth * target_size * 4)
    elif mode == "per_delay_group_rings":
        groups = tuple(
            (
                int(delay),
                jnp.asarray(np.flatnonzero(delay_steps_np == int(delay)), dtype=jnp.int32),
            )
            for delay in delay_values.tolist()
        )
        rings = tuple(jnp.zeros((batch_size, delay + 1, target_size), dtype=jnp.float32) for delay, _ in groups)

        def op(pre_spike, rings):
            outputs = []
            for ring, (delay, rows) in zip(rings, groups):
                events = pre_spike[:, pre_index[rows]] * weight[rows][None, :]
                target = target_index[rows]
                arrival = jnp.zeros((batch_size, target_size), dtype=events.dtype)
                arrival = arrival.at[:, target].add(events)
                outputs.append(ring.at[:, delay, :].add(arrival))
            return tuple(outputs)

        args = (pre_spike, rings)
        queue_bytes = int(batch_size * sum((delay + 1) * target_size * 4 for delay, _ in groups))
    else:
        _emit({"status": "error", "error": f"unknown delay_layout={mode!r}"})
        return 0

    jit_op = jax.jit(op)
    try:
        start = time.perf_counter()
        out = jit_op(*args)
        jax.block_until_ready(out)
        compile_ms = (time.perf_counter() - start) * 1000.0
        for _ in range(int(cfg["warmup"])):
            out = jit_op(*args)
            jax.block_until_ready(out)
        times = []
        for _ in range(int(cfg["repeats"])):
            start = time.perf_counter()
            out = jit_op(*args)
            jax.block_until_ready(out)
            times.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "error": f"delay benchmark failed: {type(exc).__name__}: {exc}",
                "default_backend": default_backend,
                "devices": devices,
            }
        )
        return 0

    leaves = jax.tree.leaves(out)
    output_sum = sum(float(np.asarray(leaf).sum()) for leaf in leaves)
    times_arr = np.asarray(times, dtype=float)
    _emit(
        {
            "status": "ok",
            "compile_ms": compile_ms,
            "median_ms": float(np.median(times_arr)),
            "min_ms": float(np.min(times_arr)),
            "std_ms": float(np.std(times_arr)),
            "queue_bytes": queue_bytes,
            "output_sum": output_sum,
            "realized_unique_delays": int(delay_values.size),
            "default_backend": default_backend,
            "devices": devices,
        }
    )
    return 0


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

    if cfg.get("benchmark") == "delay_queue":
        return _benchmark_delay_queue(
            cfg,
            jax=jax,
            jnp=jnp,
            np=np,
            default_backend=default_backend,
            devices=devices,
        )

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
