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

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from examples.experimental.optim_gradient_scaling.benchmark import BenchmarkConfig
from examples.experimental.optim_training_comparison.rtrl_bptt import (
    PARAMETER_NAMES,
    _seed_padded_function,
    _unflatten_gradient,
    compare_training_runs,
    run_training_worker,
)


def test_unflatten_gradient_restores_named_seed_roots() -> None:
    roots = tuple(jnp.zeros((2, width)) for width in (1, 2, 3))
    flat = jnp.arange(12.0).reshape(2, 6)

    result = _unflatten_gradient(flat, roots)

    assert tuple(result) == PARAMETER_NAMES
    assert tuple(result[name].shape for name in PARAMETER_NAMES) == tuple(root.shape for root in roots)
    np.testing.assert_array_equal(np.concatenate(tuple(result.values()), axis=1), flat)


def test_seed_padded_function_preserves_requested_prefix() -> None:
    def function(roots):
        values = roots[0]
        return values[:, 0], values, values * 2.0

    roots = (jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),)
    execute = _seed_padded_function(function, requested_seed_count=2, execution_seed_count=5)

    loss, losses, gradient = execute(roots)

    np.testing.assert_array_equal(loss, np.asarray([1.0, 3.0]))
    np.testing.assert_array_equal(losses, roots[0])
    np.testing.assert_array_equal(gradient, roots[0] * 2.0)


def test_execution_seed_count_cannot_drop_requested_seeds(tmp_path) -> None:
    config = BenchmarkConfig(n_cv=1, duration_ms=0.1, batch_size=1, n_seed=2)

    with np.testing.assert_raises_regex(ValueError, "at least config.n_seed"):
        run_training_worker(
            config,
            "bptt",
            epochs=1,
            learning_rate=0.01,
            output_json=tmp_path / "bptt.json",
            execution_seed_count=1,
        )


def test_tiny_training_histories_match_and_generate_comparison(tmp_path) -> None:
    config = BenchmarkConfig(n_cv=1, duration_ms=0.1, batch_size=2, n_seed=2)
    with jax.default_device(jax.devices("cpu")[0]):
        for method in ("bptt", "rtrl"):
            run_training_worker(
                config,
                method,
                epochs=2,
                learning_rate=0.01,
                output_json=tmp_path / f"{method}.json",
                execution_seed_count=4,
            )

    comparison = compare_training_runs(tmp_path)

    assert comparison["status"] == "ok"
    assert comparison["gradient_epoch0_max_abs_difference"] < 1e-8
    assert comparison["gradient_epoch0_max_relative_difference"] < 1e-8
    assert comparison["parameter_final_max_abs_difference"] < 1e-8
    assert comparison["bptt_temporary_bytes"] > 0
    assert comparison["rtrl_temporary_bytes"] > 0
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.png").exists()
    for method in ("bptt", "rtrl"):
        metadata = json.loads((tmp_path / f"{method}.json").read_text(encoding="utf-8"))
        assert metadata["history_shapes"]["loss"] == [2, 2]
        assert metadata["history_shapes"]["gradient"] == [2, 2, 3, 1]
        assert metadata["execution_seed_count"] == 4
