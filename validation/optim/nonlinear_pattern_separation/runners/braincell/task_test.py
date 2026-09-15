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

"""Tests for the exact-RTRL nonlinear single-neuron experiment."""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from validation.optim.nonlinear_pattern_separation.runners.braincell.task import (
    CONDUCTANCE_UNIT,
    CURRENT_SCALE_NA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOSS_KIND,
    DT_MS,
    N_CV,
    NUM_STEPS,
    PARAMETER_BOUNDS,
    READOUT_INDEX,
    STIMULUS_START_MS,
    STIMULUS_STOP_MS,
    NonlinearPatternExperiment,
    build_training_batches,
    generate_sweep_grid,
    generate_dataset,
    spike_counts,
    spike_presence_accuracy,
    soft_spike_probability,
)


def test_dataset_is_reproducible_balanced_and_collinear() -> None:
    with jax.enable_x64(True):
        first = generate_dataset(seed=3, n_train=32, n_test=16)
        second = generate_dataset(seed=3, n_train=32, n_test=16)
    np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
    np.testing.assert_array_equal(np.bincount(np.asarray(first.train_classes)), (16, 16))
    np.testing.assert_allclose(np.mean(np.asarray(first.train_inputs).sum(axis=1)), 5.0, atol=0.1)
    assert set(np.asarray(first.train_targets_mv)) == {-70.0, 35.0}


def test_model_has_twelve_cvs_and_36_bounded_x64_parameters() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=0)
        physical = experiment.parameters.physical_values()
    assert experiment.cell.n_cv == N_CV
    assert tuple(physical) == ("na.g_max", "k.g_max", "leak.g_max")
    for name, values in physical.items():
        numeric = np.asarray(values.to_decimal(CONDUCTANCE_UNIT))
        lower, upper = PARAMETER_BOUNDS[name]
        assert numeric.shape == (N_CV,)
        assert numeric.dtype == np.float64
        assert np.all((numeric >= lower) & (numeric <= upper))


def test_initial_conductances_are_uniform_across_cvs_like_jaxley() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=0)
        physical = experiment.parameters.physical_values()
    for values in physical.values():
        numeric = np.asarray(values.to_decimal(CONDUCTANCE_UNIT))
        np.testing.assert_array_equal(numeric, np.full((N_CV,), numeric[0]))


def test_zero_current_initialization_is_quiescent_over_observation_window() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=0)
        trace = experiment.simulate(jnp.zeros((2,), dtype=jnp.float64))
        count = spike_counts(trace[None, :])
    assert int(count[0]) == 0


def test_step_data_matches_jaxley_stimulus_and_readout_timing() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=0)
        times, currents, targets = experiment.step_data(jnp.asarray((1.5, 3.5)), 35.0)
    active = (np.asarray(times) >= STIMULUS_START_MS) & (np.asarray(times) < STIMULUS_STOP_MS)
    assert times.shape == (NUM_STEPS,)
    assert READOUT_INDEX * DT_MS == 3.0
    np.testing.assert_allclose(np.asarray(currents)[active][0], CURRENT_SCALE_NA * np.asarray((1.5, 3.5)))
    assert np.all(np.asarray(currents)[~active] == 0.0)
    assert np.all(np.asarray(targets) == 35.0)


def test_default_sweep_grid_is_inclusive_with_point_one_spacing() -> None:
    with jax.enable_x64(True):
        x_values, y_values, inputs = generate_sweep_grid(minimum=0.0, maximum=5.0, step=0.1)
    assert x_values.shape == y_values.shape == (51,)
    assert inputs.shape == (2601, 2)
    np.testing.assert_allclose(np.diff(np.asarray(x_values)), 0.1)
    np.testing.assert_array_equal(np.asarray(inputs[0]), (0.0, 0.0))
    np.testing.assert_array_equal(np.asarray(inputs[-1]), (5.0, 5.0))


def test_chunked_sweep_matches_single_chunk_and_preserves_parameters() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        inputs = jnp.asarray(((0.0, 0.0), (1.5, 3.5), (2.5, 2.5)), dtype=jnp.float64)
        experiment = NonlinearPatternExperiment(seed=0)
        before = experiment.parameters.physical_values()
        chunked = experiment.simulate_sweep(inputs, chunk_size=2)
        single = experiment.simulate_sweep(inputs, chunk_size=3)
        after = experiment.parameters.physical_values()
    for chunked_value, single_value in zip(chunked, single):
        np.testing.assert_allclose(chunked_value, single_value, equal_nan=True)
    for name in before:
        np.testing.assert_array_equal(
            before[name].to_decimal(CONDUCTANCE_UNIT),
            after[name].to_decimal(CONDUCTANCE_UNIT),
        )


def test_exact_rtrl_returns_finite_nonzero_gradient() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=1)
        data = experiment.step_data(jnp.asarray((1.5, 3.5)), 35.0)
        result = experiment.engine(data)
    assert np.isfinite(np.asarray(result.loss))
    flat = np.concatenate([np.ravel(np.asarray(value)) for value in result.gradients.values()])
    assert flat.shape == (36,)
    assert np.isfinite(flat).all()
    assert np.linalg.norm(flat) > 0.0


def test_soft_spike_probability_detects_peak_independent_of_time() -> None:
    with jax.enable_x64(True):
        no_spike = jnp.full((NUM_STEPS,), -60.0, dtype=jnp.float64)
        early = no_spike.at[70:90].set(35.0)
        late = no_spike.at[170:190].set(35.0)
        no_spike_probability = soft_spike_probability(no_spike)
        early_probability = soft_spike_probability(early)
        late_probability = soft_spike_probability(late)
    assert float(no_spike_probability) < 0.01
    assert float(early_probability) > 0.99
    np.testing.assert_allclose(early_probability, late_probability)


def test_peak_margin_is_default_and_previous_losses_remain_available() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        margin_experiment = NonlinearPatternExperiment(seed=0)
        soft_experiment = NonlinearPatternExperiment(seed=0, loss_kind="soft_spike")
        voltage_experiment = NonlinearPatternExperiment(seed=0, loss_kind="voltage_at_3ms")
        data = voltage_experiment.step_data(jnp.asarray((1.5, 3.5)), 35.0)
        trace = voltage_experiment.simulate(jnp.asarray((1.5, 3.5)))
        voltage_loss = voltage_experiment._loss(trace, data)
    assert margin_experiment.loss_kind == DEFAULT_LOSS_KIND == "peak_margin"
    assert soft_experiment.loss_kind == "soft_spike"
    np.testing.assert_allclose(voltage_loss, np.abs(np.asarray(trace)[READOUT_INDEX] - 35.0))


def test_peak_margin_rewards_high_peaks_and_low_non_spiking_traces_at_any_time() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        experiment = NonlinearPatternExperiment(seed=0)
        low_data = experiment.step_data(jnp.asarray((2.5, 2.5)), -70.0)
        high_data = experiment.step_data(jnp.asarray((1.5, 3.5)), 35.0)
        baseline = jnp.full((NUM_STEPS,), -60.0, dtype=jnp.float64)
        early_peak = baseline.at[70:90].set(35.0)
        late_peak = baseline.at[170:190].set(35.0)
        low_quiet_loss = experiment._loss(baseline, low_data)
        low_spike_loss = experiment._loss(early_peak, low_data)
        high_quiet_loss = experiment._loss(baseline, high_data)
        high_early_loss = experiment._loss(early_peak, high_data)
        high_late_loss = experiment._loss(late_peak, high_data)
    assert float(low_quiet_loss) < float(low_spike_loss)
    assert float(high_early_loss) < float(high_quiet_loss)
    np.testing.assert_allclose(high_early_loss, high_late_loss)


def test_training_batches_preserve_samples_and_stratify_batch_eight() -> None:
    with jax.enable_x64(True):
        dataset = generate_dataset(seed=0, n_train=32, n_test=16)
        batches = build_training_batches(
            dataset,
            epochs=2,
            batch_size=8,
            random=brainstate.random.RandomState(2_000),
        )
    assert batches.shape == (8, 8)
    inputs = np.asarray(dataset.train_inputs)
    classes = np.asarray(dataset.train_classes)
    for epoch_batches in np.asarray(batches).reshape((2, 4, 8)):
        np.testing.assert_array_equal(np.sort(epoch_batches.reshape((-1,))), np.arange(32))
        for batch in epoch_batches:
            left = (classes[batch] == 1) & (inputs[batch, 0] < inputs[batch, 1])
            middle = classes[batch] == 0
            right = (classes[batch] == 1) & (inputs[batch, 0] > inputs[batch, 1])
            np.testing.assert_array_equal((left.sum(), middle.sum(), right.sum()), (2, 4, 2))


def test_batch_one_remains_one_sample_per_optimizer_update() -> None:
    with jax.enable_x64(True):
        dataset = generate_dataset(seed=0, n_train=32, n_test=16)
        batches = build_training_batches(
            dataset,
            epochs=3,
            batch_size=DEFAULT_BATCH_SIZE,
            random=brainstate.random.RandomState(2_000),
        )
    assert batches.shape == (96, 1)


def test_stratified_batch_training_averages_four_rtrl_trajectories_per_update() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        dataset = generate_dataset(seed=0, n_train=4, n_test=2)
        experiment = NonlinearPatternExperiment(seed=0)
        batch_losses, sample_losses = experiment.train(dataset, epochs=1, batch_size=4)
        jax.block_until_ready((batch_losses, sample_losses))
    assert batch_losses.shape == (1,)
    assert sample_losses.shape == (4,)
    assert np.isfinite(np.asarray(batch_losses)).all()
    assert np.isfinite(np.asarray(sample_losses)).all()


def test_hard_spike_metrics_distinguish_presence_from_voltage_training_target() -> None:
    traces = jnp.asarray(((-70.0, 10.0, -60.0), (-70.0, -60.0, -50.0)))
    targets = jnp.asarray((35.0, -70.0))
    np.testing.assert_array_equal(spike_counts(traces), (1, 0))
    np.testing.assert_allclose(spike_presence_accuracy(traces, targets), 1.0)
