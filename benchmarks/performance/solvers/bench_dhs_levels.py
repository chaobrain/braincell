#!/usr/bin/env python3
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

"""Benchmark level-wise toy DHS elimination on JAX devices."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class DHSLevelProblem:
    """Static arrays for a complete-binary-tree toy DHS solve."""

    widths: tuple[int, ...]
    effective_n_cv: int
    diag_levels: tuple
    rhs_levels: tuple
    child_to_parent_levels: tuple
    parent_to_child_levels: tuple


def main(argv: list[str] | None = None) -> int:
    """Run the toy DHS level benchmark."""
    args = _parse_args(argv)
    if args.platform:
        os.environ["JAX_PLATFORMS"] = _normalize_jax_platform(args.platform)
    elif os.environ.get("JAX_PLATFORMS") == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"

    import jax

    problem = make_problem(
        n_cv=args.n_cv,
        popsize=args.popsize,
        dtype=args.dtype,
        seed=args.seed,
    )
    metadata = {
        "n_cv_requested": args.n_cv,
        "effective_n_cv": problem.effective_n_cv,
        "popsize": args.popsize,
        "widths": list(problem.widths),
        "profile_barrier": args.profile_barrier,
        "execution_mode": args.execution_mode,
        "dtype": args.dtype,
        "platform": str(jax.default_backend()),
        "devices": [str(device) for device in jax.devices()],
    }

    if args.execution_mode == "full-jit":
        solver = make_solver(
            problem.widths,
            popsize=args.popsize,
            profile_barrier=args.profile_barrier,
        )
        compiled = jax.jit(solver)
        timed_call = lambda scale: _time_full_jit_call(compiled, problem, scale=scale)
    else:
        runner = make_level_jit_runner(
            problem.widths,
            popsize=args.popsize,
            profile_barrier=args.profile_barrier,
        )
        timed_call = lambda scale: _time_level_jit_call(runner, problem, scale=scale)

    warmup_times = [timed_call(1.0 + 0.001 * i) for i in range(args.warmup)]

    with _maybe_jax_trace(jax, args.trace_dir):
        repeat_times = [
            timed_call(1.0 + 0.001 * (args.warmup + i))
            for i in range(args.repeat)
        ]

    result = {
        **metadata,
        "warmup_s": warmup_times,
        "repeat_s": repeat_times,
        "repeat_mean_s": float(np.mean(repeat_times)) if repeat_times else 0.0,
        "repeat_min_s": float(np.min(repeat_times)) if repeat_times else 0.0,
        "repeat_max_s": float(np.max(repeat_times)) if repeat_times else 0.0,
        "level_work_items": [
            {
                "level": int(level),
                "width": int(width),
                "work_items": int(width * args.popsize),
            }
            for level, width in enumerate(problem.widths)
        ],
    }
    _print_result(result)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {out_path}")
    return 0


def make_problem(
    *,
    n_cv: int,
    popsize: int,
    dtype: str = "float32",
    seed: int = 0,
) -> DHSLevelProblem:
    """Create deterministic toy DHS arrays.

    Parameters
    ----------
    n_cv : int
        Requested control-volume count. The benchmark uses the largest
        complete binary tree with no more than this many nodes.
    popsize : int
        Number of independent cells solved in one batched JAX call.
    dtype : {'float32', 'float64'}
        Floating-point dtype for the synthetic linear systems.
    seed : int
        Random seed used for deterministic perturbations.

    Returns
    -------
    DHSLevelProblem
        Tuple-structured arrays accepted by the jitted solver.
    """
    if n_cv < 1:
        raise ValueError("n_cv must be positive.")
    if popsize < 1:
        raise ValueError("popsize must be positive.")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'.")

    import jax.numpy as jnp

    widths = _complete_binary_widths(n_cv)
    effective_n_cv = sum(widths)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    rng = np.random.default_rng(seed)

    diag_levels = []
    rhs_levels = []
    for level, width in enumerate(widths):
        base = np.linspace(0.0, 1.0, width, dtype=np_dtype)[None, :]
        pop = np.linspace(0.0, 0.1, popsize, dtype=np_dtype)[:, None]
        noise = rng.normal(0.0, 0.002, size=(popsize, width)).astype(np_dtype)
        diag_levels.append(jnp.asarray(2.0 + 0.05 * level + base + pop + noise))
        rhs_levels.append(jnp.asarray(0.5 + 0.01 * level + base - pop + noise))

    child_to_parent_levels = []
    parent_to_child_levels = []
    for level in range(1, len(widths)):
        width = widths[level]
        edge_base = np.linspace(0.0, 0.02, width, dtype=np_dtype)
        child_to_parent_levels.append(jnp.asarray(-0.05 - edge_base))
        parent_to_child_levels.append(jnp.asarray(-0.04 - edge_base))

    return DHSLevelProblem(
        widths=widths,
        effective_n_cv=effective_n_cv,
        diag_levels=tuple(diag_levels),
        rhs_levels=tuple(rhs_levels),
        child_to_parent_levels=tuple(child_to_parent_levels),
        parent_to_child_levels=tuple(parent_to_child_levels),
    )


def make_solver(
    widths: tuple[int, ...],
    *,
    popsize: int,
    profile_barrier: bool = False,
):
    """Build a JIT-compatible complete-tree DHS toy solver.

    Parameters
    ----------
    widths : tuple of int
        Complete-tree widths from root to leaves.
    popsize : int
        Batched cell count, embedded in profiler scope names.
    profile_barrier : bool
        If ``True``, insert optimization barriers at level boundaries so JAX
        profiler traces keep late narrow levels visible.

    Returns
    -------
    Callable
        Function accepting problem arrays and a scalar scale, returning a
        scalar checksum of the solved states.
    """
    import jax
    import jax.numpy as jnp

    def _maybe_barrier(value):
        return jax.lax.optimization_barrier(value) if profile_barrier else value

    def _forward_level(parent_diag, parent_rhs, child_diag, child_rhs, c2p, p2c, *, parent_width):
        child_diag = _maybe_barrier(child_diag)
        child_rhs = _maybe_barrier(child_rhs)
        inv_child = 1.0 / child_diag
        diag_delta = (c2p * p2c * inv_child).reshape((-1, parent_width, 2)).sum(axis=-1)
        rhs_delta = (c2p * child_rhs * inv_child).reshape((-1, parent_width, 2)).sum(axis=-1)
        return _maybe_barrier(parent_diag - diag_delta), _maybe_barrier(parent_rhs - rhs_delta)

    def _root_solve(root_rhs, root_diag):
        return _maybe_barrier(root_rhs / root_diag)

    def _backward_level(parent_solve, child_diag, child_rhs, p2c):
        parent_x = jnp.repeat(parent_solve, 2, axis=-1)
        return _maybe_barrier((child_rhs - p2c * parent_x) / child_diag)

    def solve(diag_levels, rhs_levels, child_to_parent_levels, parent_to_child_levels, scale):
        diags = [diag * scale for diag in diag_levels]
        rhs = [value * scale for value in rhs_levels]

        for level in range(len(widths) - 1, 0, -1):
            width = widths[level]
            parent_width = widths[level - 1]
            scope = _level_scope("forward", level, width, popsize)
            with jax.named_scope(scope):
                c2p = child_to_parent_levels[level - 1][None, :]
                p2c = parent_to_child_levels[level - 1][None, :]
                level_fn = lambda parent_diag, parent_rhs, child_diag, child_rhs, c2p, p2c: _forward_level(
                    parent_diag,
                    parent_rhs,
                    child_diag,
                    child_rhs,
                    c2p,
                    p2c,
                    parent_width=parent_width,
                )
                diags[level - 1], rhs[level - 1] = jax.named_call(
                    level_fn,
                    name=scope,
                )(
                    diags[level - 1],
                    rhs[level - 1],
                    diags[level],
                    rhs[level],
                    c2p,
                    p2c,
                )

        solves = [None] * len(widths)
        root_scope = _level_scope("root", 0, widths[0], popsize)
        with jax.named_scope(root_scope):
            solves[0] = jax.named_call(_root_solve, name=root_scope)(rhs[0], diags[0])

        for level in range(1, len(widths)):
            width = widths[level]
            scope = _level_scope("backward", level, width, popsize)
            with jax.named_scope(scope):
                p2c = parent_to_child_levels[level - 1][None, :]
                solves[level] = jax.named_call(
                    _backward_level,
                    name=scope,
                )(solves[level - 1], diags[level], rhs[level], p2c)

        return jnp.sum(solves[-1]) + jnp.sum(solves[0])

    return solve


def make_level_jit_runner(
    widths: tuple[int, ...],
    *,
    popsize: int,
    profile_barrier: bool = False,
):
    """Build a runner with one jitted function per toy DHS level.

    Parameters
    ----------
    widths : tuple of int
        Complete-tree widths from root to leaves.
    popsize : int
        Batched cell count, embedded in profiler scope names.
    profile_barrier : bool
        If ``True``, insert optimization barriers inside each level function.

    Returns
    -------
    Callable
        Python runner that launches one compiled JAX function per level. This
        mode is intended for profiler attribution rather than end-to-end DHS
        performance measurement.
    """
    import jax
    import jax.numpy as jnp

    def _maybe_barrier(value):
        return jax.lax.optimization_barrier(value) if profile_barrier else value

    forward_fns = []
    for level in range(len(widths) - 1, 0, -1):
        width = widths[level]
        parent_width = widths[level - 1]
        scope = _level_scope("forward", level, width, popsize)

        def _make_forward_level(_scope, _parent_width):
            def _forward(parent_diag, parent_rhs, child_diag, child_rhs, c2p, p2c):
                with jax.named_scope(_scope):
                    child_diag = _maybe_barrier(child_diag)
                    child_rhs = _maybe_barrier(child_rhs)
                    inv_child = 1.0 / child_diag
                    diag_delta = (c2p * p2c * inv_child).reshape((-1, _parent_width, 2)).sum(axis=-1)
                    rhs_delta = (c2p * child_rhs * inv_child).reshape((-1, _parent_width, 2)).sum(axis=-1)
                    return _maybe_barrier(parent_diag - diag_delta), _maybe_barrier(parent_rhs - rhs_delta)

            return _forward

        forward_fn = _make_forward_level(scope, parent_width)
        forward_fns.append((level, jax.jit(jax.named_call(forward_fn, name=scope))))

    root_scope = _level_scope("root", 0, widths[0], popsize)

    def _root(root_rhs, root_diag):
        with jax.named_scope(root_scope):
            return _maybe_barrier(root_rhs / root_diag)

    root_fn = jax.jit(jax.named_call(_root, name=root_scope))

    backward_fns = []
    for level in range(1, len(widths)):
        width = widths[level]
        scope = _level_scope("backward", level, width, popsize)

        def _make_backward_level(_scope):
            def _backward(parent_solve, child_diag, child_rhs, p2c):
                with jax.named_scope(_scope):
                    parent_x = jnp.repeat(parent_solve, 2, axis=-1)
                    numerator = _maybe_barrier(child_rhs - p2c * parent_x)
                    solved = numerator / child_diag
                    return jax.named_call(lambda value: value + 0.0, name=_scope)(
                        _maybe_barrier(solved)
                    )

            return _backward

        backward_fn = _make_backward_level(scope)
        backward_fns.append((level, jax.jit(jax.named_call(backward_fn, name=scope))))

    def run(problem: DHSLevelProblem, scale: float):
        diags = [diag * scale for diag in problem.diag_levels]
        rhs = [value * scale for value in problem.rhs_levels]

        for level, fn in forward_fns:
            c2p = problem.child_to_parent_levels[level - 1][None, :]
            p2c = problem.parent_to_child_levels[level - 1][None, :]
            diags[level - 1], rhs[level - 1] = fn(
                diags[level - 1],
                rhs[level - 1],
                diags[level],
                rhs[level],
                c2p,
                p2c,
            )

        solves = [None] * len(widths)
        solves[0] = root_fn(rhs[0], diags[0])

        for level, fn in backward_fns:
            p2c = problem.parent_to_child_levels[level - 1][None, :]
            solves[level] = fn(solves[level - 1], diags[level], rhs[level], p2c)

        return jnp.sum(solves[-1]) + jnp.sum(solves[0])

    return run


def _complete_binary_widths(n_cv: int) -> tuple[int, ...]:
    leaf_width = 1 << max(0, (int(n_cv).bit_length() - 2))
    widths = []
    width = 1
    while width <= leaf_width:
        widths.append(width)
        width *= 2
    return tuple(widths)


def _level_scope(phase: str, level: int, width: int, popsize: int) -> str:
    return f"braincell:dhs_toy:{phase}:level={level:02d}:width={width:06d}:pop={popsize}"


def _time_full_jit_call(compiled, problem: DHSLevelProblem, *, scale: float) -> float:
    start = time.perf_counter()
    out = compiled(
        problem.diag_levels,
        problem.rhs_levels,
        problem.child_to_parent_levels,
        problem.parent_to_child_levels,
        scale,
    )
    out.block_until_ready()
    return time.perf_counter() - start


def _time_level_jit_call(runner, problem: DHSLevelProblem, *, scale: float) -> float:
    start = time.perf_counter()
    out = runner(problem, scale)
    out.block_until_ready()
    return time.perf_counter() - start


def _maybe_jax_trace(jax, trace_dir: str | None):
    if not trace_dir:
        return nullcontext()
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    return jax.profiler.trace(
        str(path),
        create_perfetto_link=False,
        create_perfetto_trace=True,
    )


def _print_result(result: dict) -> None:
    print(f"platform: {result['platform']}")
    print(f"n_cv: requested={result['n_cv_requested']} effective={result['effective_n_cv']}")
    print(f"popsize: {result['popsize']}")
    print(f"widths: {result['widths']}")
    print(f"execution_mode: {result['execution_mode']}")
    print(f"profile_barrier: {result['profile_barrier']}")
    print(f"warmup_s: {_format_times(result['warmup_s'])}")
    print(f"repeat_s: {_format_times(result['repeat_s'])}")
    print(f"repeat_mean_s: {result['repeat_mean_s']:.6f}")


def _format_times(values: list[float]) -> str:
    if not values:
        return "<none>"
    return ", ".join(f"{value:.6f}" for value in values)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-cv", type=int, default=1024)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--platform", choices=("cpu", "gpu", "cuda"), default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--profile-barrier", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("full-jit", "level-jit"),
        default="full-jit",
        help="'full-jit' measures fused end-to-end code; 'level-jit' exposes one profiler scope per level.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if args.repeat < 1:
        raise ValueError("repeat must be positive.")
    return args


def _normalize_jax_platform(platform: str) -> str:
    """Return the concrete JAX platform name for user-facing aliases."""
    return "cuda" if platform == "gpu" else platform


if __name__ == "__main__":
    sys.exit(main())
