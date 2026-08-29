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

"""Tests for the experimental rollout gradient engines."""

import unittest

import braincell
import brainstate
import brainunit as u
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute._testing import _build_tree
from braincell.filter import AllRegion
from examples.experimental.online_learning.rollout_gradients import (
    TrajectoryGradientResult,
    build_rollout_value_and_grad,
    build_trajectory_value_and_grad,
)


def _cell(*, population=1):
    cell = braincell.Cell(_build_tree(), pop_size=(population,))
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "IL",
            name="leak",
            g_max=0.3 * u.mS / u.cm**2,
            E=-54.3 * u.mV,
        ),
    )
    cell.channels["leak"].trainable(g_max=braincell.trainable.scale(group_by="all", name="leak.factor"))
    cell.init_state()
    return cell


def _engine(cell, method):
    def rollout_step(target_mv):
        cell.update()
        error = cell.V.value.to_decimal(u.mV) - target_mv
        return jnp.mean(error * error)

    return build_rollout_value_and_grad(cell, step=rollout_step, method=method)


class RolloutGradientEngineTest(unittest.TestCase):
    def test_bptt_and_full_rtrl_share_losses_and_optimizer_gradient_mapping(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            bptt_cell = _cell()
            rtrl_cell = _cell()
            targets = jnp.full((8,), -60.0, dtype=jnp.float64)
            bptt = _engine(bptt_cell, "bptt")(targets)
            rtrl = _engine(rtrl_cell, "rtrl")(targets)

        self.assertEqual(tuple(bptt.gradients), ("leak.factor",))
        self.assertEqual(tuple(rtrl.gradients), ("leak.factor",))
        np.testing.assert_allclose(rtrl.losses, bptt.losses, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rtrl.loss, bptt.loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            rtrl.gradients["leak.factor"],
            bptt.gradients["leak.factor"],
            rtol=1e-9,
            atol=1e-10,
        )

    def test_explicit_parameter_subset_uses_stable_names(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _cell()
            selected = {"fit.leak": cell.trainables.parameters().states()["leak.factor"]}

            def rollout_step(target_mv):
                cell.update()
                return jnp.mean((cell.V.value.to_decimal(u.mV) - target_mv) ** 2)

            result = build_rollout_value_and_grad(
                cell,
                step=rollout_step,
                method="rtrl",
                parameters=selected,
            )(jnp.full((3,), -60.0))

        self.assertEqual(tuple(result.gradients), ("fit.leak",))
        self.assertEqual(result.gradients["fit.leak"].shape, ())

    def test_gradient_mapping_updates_the_existing_optimizer_states(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _cell()
            parameter_states = cell.trainables.parameters().states()
            optimizer = braintools.optim.Adam(lr=0.01)
            optimizer.register_trainable_weights(parameter_states)
            before = parameter_states["leak.factor"].value
            targets = jnp.full((3,), -60.0)
            engine = _engine(cell, "rtrl")
            engine.prepare(targets[0])

            def train_step(_):
                result = engine(targets)
                optimizer.update(result.gradients)
                return result.loss

            losses = brainstate.transform.for_loop(train_step, jnp.arange(2))
            after = parameter_states["leak.factor"].value

        self.assertFalse(bool(jnp.allclose(before, after)))
        self.assertLess(float(losses[-1]), float(losses[0]))

    def test_default_initializer_differentiates_parameter_dependent_reset(self) -> None:
        def make_cell():
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel("Na_HH1952", name="na", g_max=12.0 * u.mS / u.cm**2),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            cell.channels["na"].trainable(V_sh=braincell.trainable.scale(group_by="all", name="na.vsh.factor"))
            cell.init_state()
            return cell

        def make_engine(cell, method):
            def rollout_step(target_mv):
                cell.update()
                return jnp.mean((cell.V.value.to_decimal(u.mV) - target_mv) ** 2)

            return build_rollout_value_and_grad(cell, step=rollout_step, method=method)

        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            targets = jnp.full((3,), -62.0, dtype=jnp.float64)
            bptt = make_engine(make_cell(), "bptt")(targets)
            rtrl = make_engine(make_cell(), "rtrl")(targets)

        np.testing.assert_allclose(
            rtrl.gradients["na.vsh.factor"],
            bptt.gradients["na.vsh.factor"],
            rtol=1e-9,
            atol=1e-10,
        )

    def test_diagnostic_decomposes_local_and_prefix_gradients(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _cell(population=2)
            targets = jnp.full((6, 2, 1), -60.0, dtype=jnp.float64)
            engine = _engine(cell, "rtrl")
            normal = engine(targets)
            diagnostic = jax.jit(lambda data: engine.diagnose(data, at=(0, 2, 5)))(targets)
            all_steps = jax.jit(lambda data: engine.diagnose(data))(targets)

        np.testing.assert_array_equal(diagnostic.at, np.asarray([0, 2, 5], dtype=np.int32))
        np.testing.assert_allclose(diagnostic.decomposition_residual, 0.0, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            diagnostic.local_gradients,
            diagnostic.eligibility_gradients + diagnostic.direct_gradients,
            rtol=1e-9,
            atol=1e-10,
        )
        np.testing.assert_allclose(diagnostic.loss, normal.loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            diagnostic.gradients["leak.factor"],
            normal.gradients["leak.factor"],
            rtol=1e-9,
            atol=1e-10,
        )
        sensitivity_leaves = jax.tree.leaves(diagnostic.sensitivity)
        learning_signal_leaves = jax.tree.leaves(diagnostic.learning_signal)
        self.assertTrue(any(leaf.shape[:2] == (3, 1) for leaf in sensitivity_leaves))
        self.assertTrue(any(leaf.shape[:1] == (3,) and 2 in leaf.shape for leaf in learning_signal_leaves))
        np.testing.assert_array_equal(all_steps.at, np.arange(6, dtype=np.int32))
        for sampled_leaf, all_leaf in zip(
            jax.tree.leaves(diagnostic.sensitivity),
            jax.tree.leaves(all_steps.sensitivity),
        ):
            np.testing.assert_allclose(sampled_leaf, np.asarray(all_leaf)[[0, 2, 5]], rtol=1e-10, atol=1e-10)
        for sampled_leaf, all_leaf in zip(
            jax.tree.leaves(diagnostic.learning_signal),
            jax.tree.leaves(all_steps.learning_signal),
        ):
            np.testing.assert_allclose(sampled_leaf, np.asarray(all_leaf)[[0, 2, 5]], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            diagnostic.local_gradients,
            np.asarray(all_steps.local_gradients)[[0, 2, 5]],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_diagnostic_rejects_bptt_and_invalid_indices(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _cell()
            engine = _engine(cell, "bptt")
            targets = jnp.full((3,), -60.0)
            with self.assertRaisesRegex(ValueError, "method='rtrl'"):
                engine.diagnose(targets, at=(0,))

            rtrl = _engine(_cell(), "rtrl")
            with self.assertRaisesRegex(ValueError, "sorted unique"):
                rtrl.diagnose(targets, at=(1, 0))
            with self.assertRaises(IndexError):
                rtrl.diagnose(targets, at=(3,))


class TrajectoryGradientEngineTest(unittest.TestCase):
    @staticmethod
    def _engine(cell, method, *, pytree=False):
        def observation_step(data):
            time_ms, _target = data
            with brainstate.environ.context(t=time_ms * u.ms):
                cell.update()
            voltage = cell.V.value.to_decimal(u.mV)
            return {"v": voltage, "v_squared": voltage**2} if pytree else voltage

        def trajectory_loss(observations, data):
            _times, target = data
            voltage = observations["v"] if pytree else observations
            mse = jnp.mean((voltage - target) ** 2)
            derivative = jnp.mean((jnp.diff(voltage, axis=0) - jnp.diff(target, axis=0)) ** 2)
            if pytree:
                mse = mse + 1e-5 * jnp.mean(observations["v_squared"])
            return mse + 0.1 * derivative

        return build_trajectory_value_and_grad(
            cell,
            step=observation_step,
            loss=trajectory_loss,
            method=method,
        )

    def test_two_pass_matches_bptt_and_finite_difference(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            targets = jnp.full((8, 1, 1), -60.0, dtype=jnp.float64)
            data = (jnp.arange(8, dtype=jnp.float64) * 0.025, targets)
            bptt = self._engine(_cell(), "bptt")
            rtrl = self._engine(_cell(), "rtrl")
            bptt_result = bptt(data)
            rtrl_result = rtrl(data)

            rtrl.prepare((data[0][0], data[1][0]))
            root = tuple(state.value for state in rtrl.parameter_states.values())
            epsilon = 1e-4

            def loss_at(value):
                observations = rtrl._observation_rollout((value,), data)
                return rtrl.loss(observations, data)

            finite_difference = (loss_at(root[0] + epsilon) - loss_at(root[0] - epsilon)) / (2.0 * epsilon)

        self.assertIsInstance(rtrl_result, TrajectoryGradientResult)
        self.assertEqual(rtrl_result._fields, ("loss", "gradients"))
        np.testing.assert_allclose(rtrl_result.loss, bptt_result.loss, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            rtrl_result.gradients["leak.factor"],
            bptt_result.gradients["leak.factor"],
            rtol=1e-9,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            rtrl_result.gradients["leak.factor"],
            finite_difference,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_two_pass_supports_weighted_observation_pytree(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            targets = jnp.full((5, 1, 1), -61.0, dtype=jnp.float64)
            data = (jnp.arange(5, dtype=jnp.float64) * 0.025, targets)
            bptt = self._engine(_cell(), "bptt", pytree=True)(data)
            rtrl = self._engine(_cell(), "rtrl", pytree=True)(data)

        np.testing.assert_allclose(rtrl.loss, bptt.loss, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            rtrl.gradients["leak.factor"],
            bptt.gradients["leak.factor"],
            rtol=1e-9,
            atol=1e-10,
        )

    def test_trajectory_loss_must_be_scalar(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _cell()

            def step(target):
                cell.update()
                return cell.V.value.to_decimal(u.mV)

            engine = build_trajectory_value_and_grad(
                cell,
                step=step,
                loss=lambda observations, _data: observations,
                method="rtrl",
            )
            with self.assertRaisesRegex(ValueError, "must return a scalar"):
                engine(jnp.full((2,), -60.0))

    def test_two_pass_includes_parameter_dependent_reset(self) -> None:
        def make_cell():
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel("Na_HH1952", name="na", g_max=12.0 * u.mS / u.cm**2),
                braincell.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
            )
            cell.channels["na"].trainable(V_sh=braincell.trainable.scale(group_by="all", name="na.vsh.factor"))
            cell.init_state()
            return cell

        def make_engine(cell, method):
            def step(target):
                cell.update()
                return cell.V.value.to_decimal(u.mV)

            return build_trajectory_value_and_grad(
                cell,
                step=step,
                loss=lambda trace, target: jnp.mean((trace - target[:, None, None]) ** 2),
                method=method,
            )

        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            targets = jnp.full((3,), -62.0, dtype=jnp.float64)
            bptt = make_engine(make_cell(), "bptt")(targets)
            rtrl = make_engine(make_cell(), "rtrl")(targets)

        np.testing.assert_allclose(
            rtrl.gradients["na.vsh.factor"],
            bptt.gradients["na.vsh.factor"],
            rtol=1e-9,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
