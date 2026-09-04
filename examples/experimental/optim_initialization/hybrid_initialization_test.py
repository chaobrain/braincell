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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.experimental.optim_initialization.dc_protocol_dataset import (
    DcProtocol,
    build_cell,
    simulate_voltage,
)
from examples.experimental.optim_initialization.hybrid_initialization import (
    PARAMETER_NAMES,
    CandidateEvaluator,
    CandidateMetrics,
    ExperimentConfig,
    RtrlTrainer,
    clip_gradient_rows,
    coordinates_to_scales,
    optimizer_roots_to_scales,
    run_search,
    scales_to_optimizer_roots,
)


class _QuadraticEvaluator:
    def evaluate_scales(self, scales) -> CandidateMetrics:
        values = np.asarray(scales)
        loss = np.sum((values - 1.0) ** 2, axis=1)
        return CandidateMetrics(
            total_mse=loss,
            protocol_mse=loss[:, None],
            spike_counts=np.zeros((values.shape[0], 1), dtype=np.int32),
            spike_timing_error_ms=np.zeros((values.shape[0], 1)),
        )


def _protocols() -> tuple[DcProtocol, ...]:
    return (
        DcProtocol("soma", "soma", "train", "smoke", 0.04),
        DcProtocol("dend", "dend_a", "train", "smoke", -0.04),
    )


def test_scale_optimizer_transform_round_trip() -> None:
    scales = jnp.linspace(0.55, 1.45, 18).reshape(2, 9)

    restored = optimizer_roots_to_scales(scales_to_optimizer_roots(scales))

    np.testing.assert_allclose(restored, scales, rtol=1e-12, atol=1e-12)
    assert coordinates_to_scales(jnp.zeros((1, 9))).shape == (1, 9)


def test_gradient_clipping_is_independent_per_start() -> None:
    gradient = jnp.asarray([[3.0, 4.0], [0.3, 0.4]])

    clipped, norms = clip_gradient_rows(gradient, 1.0)

    np.testing.assert_allclose(norms, [5.0, 0.5])
    np.testing.assert_allclose(clipped, [[0.6, 0.8], [0.3, 0.4]])


def test_random_and_sobol_search_obey_budget_bounds_and_selection() -> None:
    config = ExperimentConfig(
        screen_candidates=64, screen_batch_size=16, selected_starts=4, updates=2, checkpoint_every=1
    )
    evaluator = _QuadraticEvaluator()

    direct = run_search("direct_random", 3, evaluator, config)
    random = run_search("random_screen", 3, evaluator, config)
    sobol = run_search("sobol", 3, evaluator, config)

    assert direct.scales.shape == (4, 9)
    assert random.scales.shape == (64, 9)
    assert sobol.scales.shape == (64, 9)
    assert np.all((sobol.scales > 0.5) & (sobol.scales < 1.5))
    np.testing.assert_array_equal(
        random.selected_indices,
        np.argsort(random.metrics.total_mse, kind="stable")[:4],
    )


def test_nevergrad_search_obeys_same_batched_contract_when_installed() -> None:
    pytest.importorskip("nevergrad")
    config = ExperimentConfig(
        screen_candidates=16,
        screen_batch_size=4,
        selected_starts=2,
        updates=2,
        checkpoint_every=1,
    )

    result = run_search("two_points_de", 2, _QuadraticEvaluator(), config)

    assert result.scales.shape == (16, 9)
    assert result.selected_indices.shape == (2,)
    assert np.isfinite(result.metrics.total_mse).all()


def test_target_scale_has_zero_forward_mse_and_finite_rtrl_gradient() -> None:
    protocols = _protocols()
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        target = np.asarray(simulate_voltage(build_cell(protocols, trainable=False), num_steps=3))
        evaluator = CandidateEvaluator(protocols, target, batch_size=2, num_steps=3)
        metrics = evaluator.evaluate_scales(np.ones((2, len(PARAMETER_NAMES))))
        trainer = RtrlTrainer(protocols, target, num_starts=2, num_steps=3)
        loss, gradient = trainer.gradient(np.ones((2, len(PARAMETER_NAMES))))

    np.testing.assert_allclose(metrics.total_mse, 0.0, atol=1e-20)
    np.testing.assert_allclose(loss, 0.0, atol=1e-20)
    assert gradient.shape == (2, 9)
    assert np.isfinite(gradient).all()


def test_three_cv_bptt_and_rtrl_gradients_match_away_from_target() -> None:
    protocols = _protocols()
    scales = np.full((2, len(PARAMETER_NAMES)), 0.9)
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        target = np.asarray(simulate_voltage(build_cell(protocols, trainable=False), num_steps=3))
        bptt = RtrlTrainer(protocols, target, num_starts=2, num_steps=3, method="bptt")
        rtrl = RtrlTrainer(protocols, target, num_starts=2, num_steps=3, method="rtrl")
        bptt_loss, bptt_gradient = bptt.gradient(scales)
        rtrl_loss, rtrl_gradient = rtrl.gradient(scales)

    np.testing.assert_allclose(rtrl_loss, bptt_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtrl_gradient, bptt_gradient, rtol=1e-9, atol=1e-10)
