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
import numpy as np

from examples.experimental.optim_stimulus_design.dataset import (
    N_STEPS,
    StimulusCalibration,
    StimulusDataset,
    StimulusProtocol,
    build_cell,
    simulate_voltage,
)
from examples.experimental.optim_stimulus_design.global_ensemble import (
    ForwardEnsembleEvaluator,
    aggregate_scores,
    direction_interval,
    ensemble_scales,
    parameter_geometry,
    protocol_normalizers,
    top_indices,
)


class _Dataset:
    def __init__(self):
        self.protocols = tuple(range(6))
        self.target_spike_counts = np.asarray((0, 1, 2, 0, 1, 2))

    def indices(self, split):
        return {
            "train": np.asarray((0, 1)),
            "validation": np.asarray((2, 3)),
            "test": np.asarray((4, 5)),
        }[split]


def test_sobol_ensemble_is_reproducible_and_strictly_bounded() -> None:
    first = ensemble_scales(16)
    second = ensemble_scales(16)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (16, 6)
    assert np.all((first > 0.5) & (first < 1.5))
    assert not np.any(np.all(first == 1.0, axis=1))


def test_protocol_normalizer_uses_prior_median_and_floor() -> None:
    values = np.ones((3, 60)) * np.asarray((0.0, 2.0, 10.0))[:, None]
    values[:, 0] = (0.0, 0.2, 0.4)

    result = protocol_normalizers(values)

    assert result[0] == 1.0
    np.testing.assert_allclose(result[1:], 2.0)


def test_aggregate_scores_uses_train_only_for_train_score_and_preserves_spikes() -> None:
    dataset = _Dataset()
    scales = np.ones((3, 6))
    mse = np.asarray(
        (
            (1, 2, 100, 100, 1000, 1000),
            (2, 1, 1, 1, 1, 1),
            (3, 3, 0, 0, 0, 0),
        ),
        dtype=float,
    )
    counts = np.tile(dataset.target_spike_counts, (3, 1))
    counts[0, 2] += 1

    scores = aggregate_scores(dataset, scales, mse, counts, np.ones((6,)))
    selected = top_indices(scores.raw_train, size=2)
    changed = mse.copy()
    changed[:, 2:] = changed[::-1, 2:]
    changed_scores = aggregate_scores(dataset, scales, changed, counts, np.ones((6,)))

    np.testing.assert_array_equal(selected, top_indices(changed_scores.raw_train, size=2))
    np.testing.assert_allclose(scores.raw_train, [1.5, 1.5, 3.0])
    assert not scores.spike_exact_validation[0]
    assert scores.spike_distance_validation[0] == 1


def test_parameter_geometry_aligns_pc1_with_known_weak_direction() -> None:
    weak = np.asarray((1.0, -1.0, 0.0, 0.0, 0.0, 0.0)) / np.sqrt(2.0)
    alpha = np.linspace(-0.3, 0.3, 32)
    log_scales = alpha[:, None] * weak
    vectors = np.eye(6)
    vectors[:, 0] = weak
    vectors[:, 1] = np.asarray((1.0, 1.0, 0.0, 0.0, 0.0, 0.0)) / np.sqrt(2.0)

    result = parameter_geometry(log_scales, np.arange(32), vectors)

    assert result["pc1_weak_cosine"] > 0.999999
    assert result["fim_variance"][0] > 1000.0 * max(result["fim_variance"][1], 1e-30)


def test_direction_interval_respects_all_log_scale_bounds() -> None:
    direction = np.asarray((1.0, -2.0, 0.5, 0.0, 0.0, 0.0))
    lower, upper = direction_interval(direction)
    points = np.linspace(lower, upper, 11)[:, None] * direction

    assert np.all(points >= np.log(0.5) - 1e-12)
    assert np.all(points <= np.log(1.5) + 1e-12)


def test_forward_evaluator_reproduces_target_at_unit_scales() -> None:
    current = np.zeros((1, N_STEPS, 3))
    current[:, 1:3, 0] = 0.02
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        target = np.asarray(simulate_voltage(build_cell(current, trainable=False), num_steps=3))
        protocol = StimulusProtocol("smoke", "smoke", "step", "smoke", "train", "soma", None, None, 0.02)
        calibration = StimulusCalibration(-0.1, -0.2, 0.01, 0.1, {i: (0.1, 0.2) for i in range(1, 5)}, 0.02, 0)
        dataset = StimulusDataset(
            (protocol,),
            np.arange(3) * 0.025,
            current,
            target,
            np.zeros((1,), dtype=np.int16),
            calibration,
        )
        evaluator = ForwardEnsembleEvaluator(dataset, batch_size=2, num_steps=3)
        mse, counts = evaluator.evaluate(np.ones((2, 6)))

    np.testing.assert_allclose(mse, 0.0, atol=1e-20)
    np.testing.assert_array_equal(counts, 0)
