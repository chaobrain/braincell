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

"""Profiling adapter for the block-exact RTRL/BPTT gradient benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np


def add_case_args(parser) -> None:
    """Add RTRL/BPTT gradient profiling arguments to ``parser``."""
    parser.add_argument("--method", choices=("bptt", "rtrl"), default="bptt")
    parser.add_argument("--n-cv", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-seed", type=int, default=16)
    parser.add_argument(
        "--execution-seed-count",
        type=int,
        default=None,
        help="Static seed extent used for execution; extra outputs are discarded.",
    )
    parser.add_argument("--backsub", choices=("recursive", "ordinary"), default="recursive")
    parser.add_argument("--hlo-out", default=None)


def create_workload(args):
    """Create the RTRL/BPTT gradient profiling workload."""
    return RTRLGradientWorkload(args)


class RTRLGradientWorkload:
    """Compile and profile one fixed-shape exact-gradient executable."""

    def __init__(self, args) -> None:
        self.args = args
        self.duration_ms = 40.0 if args.duration_ms is None else float(args.duration_ms)
        self.dt_ms = 0.025 if args.dt_ms is None else float(args.dt_ms)
        if not np.isclose(self.dt_ms, 0.025, rtol=0.0, atol=1e-12):
            raise ValueError("The scaling benchmark requires --dt-ms=0.025.")
        self.requested_seed_count = int(args.n_seed)
        self.execution_seed_count = (
            self.requested_seed_count if args.execution_seed_count is None else int(args.execution_seed_count)
        )
        if self.requested_seed_count < 1:
            raise ValueError("--n-seed must be positive.")
        if self.execution_seed_count < self.requested_seed_count:
            raise ValueError("--execution-seed-count must be at least --n-seed.")
        self.config = None
        self.prepared = None
        self.execution_roots = None
        self.compiled = None
        self.compile_seconds = None
        self.memory_analysis = None
        self.cost_analysis = None

    def build_phases(self):
        """Return separate preparation and compilation phases."""
        return (("prepare_gradient", self.prepare_gradient), ("compile_gradient", self.compile_gradient))

    def build(self) -> None:
        """Prepare and compile the gradient workload."""
        self.prepare_gradient()
        self.compile_gradient()

    def prepare_gradient(self) -> None:
        """Build the stateful gradient engine and padded seed roots."""
        import brainstate
        import brainunit as u
        import jax
        from examples.experimental.optim_gradient_scaling.benchmark import (
            BenchmarkConfig,
            prepare_benchmark,
        )

        jax.config.update("jax_enable_x64", True)
        brainstate.environ.set(dt=self.dt_ms * u.ms, precision=64)
        os.environ["BRAINCELL_DHS_BACKSUB"] = self.args.backsub
        self.config = BenchmarkConfig(
            n_cv=int(self.args.n_cv),
            duration_ms=self.duration_ms,
            batch_size=int(self.args.batch_size),
            n_seed=self.requested_seed_count,
        )
        self.prepared = prepare_benchmark(self.config, self.args.method)
        self.execution_roots = _pad_seed_roots(
            self.prepared.seed_roots,
            execution_seed_count=self.execution_seed_count,
        )

    def compile_gradient(self) -> None:
        """Compile the padded executable and capture compiler metadata."""
        import jax

        self._require_prepared()
        requested_seed_count = self.requested_seed_count
        function = self.prepared.function

        def requested_outputs(roots):
            loss, losses, gradient = function(roots)
            return (
                loss[:requested_seed_count],
                losses[:requested_seed_count],
                gradient[:requested_seed_count],
            )

        started = time.perf_counter()
        self.compiled = jax.jit(requested_outputs).lower(self.execution_roots).compile()
        self.compile_seconds = time.perf_counter() - started
        memory = self.compiled.memory_analysis()
        self.memory_analysis = {
            "argument_bytes": int(memory.argument_size_in_bytes),
            "output_bytes": int(memory.output_size_in_bytes),
            "temporary_bytes": int(memory.temp_size_in_bytes),
            "alias_bytes": int(memory.alias_size_in_bytes),
        }
        self.cost_analysis = _jsonable(self.compiled.cost_analysis())
        if self.args.hlo_out:
            path = Path(self.args.hlo_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.compiled.as_text(), encoding="utf-8")

    def init_reset(self) -> None:
        """Keep the harness phase contract; compiled inputs are immutable."""

    def reset_for_run(self) -> None:
        """Keep the harness phase contract; each call receives the same roots."""

    def run(self):
        """Execute one complete requested-seed gradient call."""
        if self.compiled is None:
            raise RuntimeError("compile_gradient() must be called before run().")
        return self.compiled(self.execution_roots)

    def block(self, result) -> None:
        """Synchronize every returned device array."""
        import jax

        for leaf in jax.tree.leaves(result):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    def materialize(self, result) -> dict[str, object]:
        """Return small correctness metadata without persisting full gradients."""
        loss, losses, gradient = (np.asarray(value) for value in result)
        return {
            "loss_shape": list(loss.shape),
            "losses_shape": list(losses.shape),
            "gradient_shape": list(gradient.shape),
            "loss_mean": float(np.mean(loss)),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient": gradient.tolist(),
        }

    def metadata(self) -> dict[str, object]:
        """Return the static configuration and compiler measurements."""
        config = self.config
        return {
            "method": self.args.method,
            "n_cv": int(self.args.n_cv),
            "duration_ms": self.duration_ms,
            "dt_ms": self.dt_ms,
            "batch_size": int(self.args.batch_size),
            "requested_seed_count": self.requested_seed_count,
            "execution_seed_count": self.execution_seed_count,
            "backsub": self.args.backsub,
            "num_steps": None if config is None else config.num_steps,
            "compile_seconds": self.compile_seconds,
            "memory_analysis": self.memory_analysis,
            "cost_analysis": self.cost_analysis,
            "hlo_out": self.args.hlo_out,
        }

    def _require_prepared(self) -> None:
        if self.prepared is None or self.execution_roots is None:
            raise RuntimeError("prepare_gradient() must be called before compilation.")


def _pad_seed_roots(roots, *, execution_seed_count: int):
    """Repeat seed roots to a static extent while preserving the original prefix."""
    import jax.numpy as jnp

    if not roots:
        raise ValueError("At least one seed root is required.")
    requested_seed_count = int(roots[0].shape[0])
    if requested_seed_count < 1 or any(int(root.shape[0]) != requested_seed_count for root in roots):
        raise ValueError("Every root must have the same non-empty seed axis.")
    if execution_seed_count < requested_seed_count:
        raise ValueError("execution_seed_count must preserve every requested seed.")
    indices = jnp.arange(execution_seed_count, dtype=jnp.int32) % requested_seed_count
    return tuple(root[indices] for root in roots)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
