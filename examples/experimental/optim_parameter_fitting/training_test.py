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

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.config import (
    ExperimentConfig,
    InitializationConfig,
)
from examples.experimental.optim_parameter_fitting.datasets import DatasetBundle, Protocol, feature_step_dataset
from examples.experimental.optim_parameter_fitting.losses import raw_voltage_mse
from examples.experimental.optim_parameter_fitting.models import hh_1cv_classic_bounded_direct
from examples.experimental.optim_parameter_fitting.optimizers import adam
from examples.experimental.optim_parameter_fitting.search import ForwardSelectionStage
from examples.experimental.optim_parameter_fitting.training import (
    GradientEngine,
    initialize_candidates,
    run_pipeline,
)


def _config(*, starts=2, epochs=2, method="rtrl") -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        model=hh_1cv_classic_bounded_direct(),
        dataset=feature_step_dataset(),
        loss=raw_voltage_mse(),
        initialization=InitializationConfig(seed=0, num_candidates=starts),
        stages=(adam(epochs=epochs, checkpoint_every=1, gradient_method=method),),
    )


def _short_bundle(config: ExperimentConfig) -> DatasetBundle:
    splits = ("train",) * 5 + ("validation",) * 2 + ("test",)
    protocols = tuple(Protocol(f"step_{index}", split, "smoke", 0.0, 0) for index, split in enumerate(splits))
    current = np.zeros((8, 3))
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        target = np.asarray(config.model.simulate(current, dt_ms=0.025))
    return DatasetBundle(protocols, np.arange(3) * 0.025, current, target, 0.025, 0.025, 0.05)


def test_initialization_generates_all_64_candidates_in_one_matrix() -> None:
    candidates = initialize_candidates(_config(starts=64))

    assert candidates.physical.shape == (64, 3)
    assert candidates.candidate_id.tolist() == list(range(64))


def test_explicit_initial_support_is_independent_of_wider_transform_bounds() -> None:
    baseline = _config(starts=64)
    wide = ExperimentConfig(
        name="wide",
        model=hh_1cv_classic_bounded_direct(bound_multipliers=(0.1, 2.0)),
        dataset=baseline.dataset,
        loss=baseline.loss,
        initialization=InitializationConfig(
            seed=0,
            num_candidates=64,
            target_relative_range=(0.5, 1.5),
        ),
        stages=baseline.stages,
    )

    with jax.enable_x64(True):
        baseline_candidates = initialize_candidates(baseline)
        wide_candidates = initialize_candidates(wide)

    np.testing.assert_array_equal(wide_candidates.physical, baseline_candidates.physical)


def test_bptt_and_rtrl_match_for_bounded_direct_parameters() -> None:
    rtrl_config = _config(method="rtrl")
    bptt_config = _config(method="bptt")
    bundle = _short_bundle(rtrl_config)
    current, target, _protocols = bundle.subset("train")
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        physical = initialize_candidates(rtrl_config).physical
        roots = rtrl_config.model.parameter_space.z_roots(physical)
        rtrl = GradientEngine(rtrl_config, current, target, method="rtrl")
        bptt = GradientEngine(bptt_config, current, target, method="bptt")
        rtrl_loss, rtrl_gradient = rtrl(roots)
        bptt_loss, bptt_gradient = bptt(roots)

    np.testing.assert_allclose(rtrl_loss, bptt_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtrl_gradient, bptt_gradient, rtol=1e-9, atol=1e-10)


def test_two_epoch_pipeline_keeps_test_final_only_and_all_lanes_together() -> None:
    config = _config()
    bundle = _short_bundle(config)
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        result = run_pipeline(config, bundle)
    stage = result.stages[0]

    assert stage.optimizer_z.shape == (3, 2, 3)
    assert stage.physical_parameters.shape == (3, 2, 3)
    assert stage.train_loss.shape == (3, 2)
    assert stage.validation_epoch.tolist() == [0, 1, 2]
    assert stage.test_epoch.tolist() == [2]
    assert stage.test.total_loss.shape == (1, 2)


def test_gradient_search_gradient_pipeline_hands_off_physical_values() -> None:
    config = ExperimentConfig(
        name="hybrid_smoke",
        model=hh_1cv_classic_bounded_direct(),
        dataset=feature_step_dataset(),
        loss=raw_voltage_mse(),
        initialization=InitializationConfig(seed=0, num_candidates=2),
        stages=(
            adam(epochs=1, checkpoint_every=1),
            ForwardSelectionStage(),
            adam(epochs=1, checkpoint_every=1, name="adam_after_search"),
        ),
    )
    bundle = _short_bundle(config)
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        result = run_pipeline(config, bundle)

    first, search, second = result.stages
    np.testing.assert_allclose(search.candidates.physical, first.output_candidates.physical)
    np.testing.assert_allclose(second.input_candidates.physical, search.candidates.physical)
    np.testing.assert_allclose(second.physical_parameters[0], search.candidates.physical)
    assert second.name == "adam_after_search"
