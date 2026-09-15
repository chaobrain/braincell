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
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import braincell
import brainstate
import brainunit as u
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute._testing import _build_tree
from braincell._base_channel import IonChannel
from braincell._parameter_schema import RuntimeParameterState
from braincell.trainable._manager import ParameterBinding, TrainableManager, _TargetRow
from braincell.filter import AllRegion
from braincell.experimental.optim.gradients import (
    TrajectoryGradientResult,
    build_rollout_value_and_grad,
    build_trajectory_value_and_grad,
)


def _cell(*, population=1, source=None):
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
    if source is None:
        source = braincell.trainable.scale(group_by="all", name="leak.factor")
    cell.channels["leak"].trainable(g_max=source)
    cell.init_state()
    return cell


def _engine(cell, method):
    def rollout_step(target_mv):
        cell.update()
        error = cell.V.value.to_decimal(u.mV) - target_mv
        return jnp.mean(error * error)

    return build_rollout_value_and_grad(cell, step=rollout_step, method=method)


class _MappedRecurrence:
    """Two-coordinate recurrence with real bindings, without a neuron solver."""

    def __init__(self, mapping="direct", *, mutation=None):
        self.root = brainstate.nn.Param(jnp.asarray([0.4, 0.7]))
        self.x = brainstate.ShortTermState(jnp.zeros((1, 2)))
        self.g = RuntimeParameterState(jnp.zeros((1, 2)), axis="row", full_shape=(1, 2))
        self.mapping = mapping
        self.mutation = mutation
        node = IonChannel(size=(2,))
        node.g_max = self.g
        layout = SimpleNamespace(id=0, source_cv_ids=(0, 1), kind="channel:IL")
        self._runtime = SimpleNamespace(
            layouts=(layout,), state_buffers={(0, "g_max"): self.g},
            layout_mechanisms={0: braincell.mech.Channel("IL", name="leak")},
            runtime_nodes={0: node}, merged_channel_layout_groups={},
        )
        self.trainables = TrainableManager(self)
        self.trainables.roots["theta"] = self.root
        rows = tuple(_TargetRow("channel", "leak", "IL", 0, cv, cv) for cv in range(2))
        self.trainables._binding_list.append(ParameterBinding(
            name="theta", target_owner="leak", target_field="g_max",
            row_keys=((0, 0), (0, 1)), group_by="row", root_names=("theta",), unit=None,
            _rows=rows, _evaluate=self.evaluate,
            _direct_root_name="theta" if mapping == "direct" else None,
        ))
        self.trainables._target_axes[(0, "g_max")] = "row"
        self.trainables.materialize()

    def evaluate(self):
        value = self.root.value()
        if self.mapping == "scale":
            value = value * jnp.asarray([2., 3.])
        elif self.mapping == "distribution":
            value = jnp.exp(value[0]) + value[1] * jnp.asarray([0., 2.])
        elif self.mapping == "dynamic":
            value = value + self.x.value.reshape(2)
        return value

    def reset_state(self):
        self.x.value = self.g.value * 0.2

    def step(self, data):
        self.x.value = 0.8 * self.x.value + self.g.value * data
        if self.mutation == "root":
            self.root.val.value = self.root.val.value * 0.9
        elif self.mutation == "physical":
            self.g.value = self.g.value * 0.9
        return jnp.sum(self.x.value ** 2) + 0.1 * jnp.sum(self.root.value() ** 2)


class MaterializationScheduleTest(unittest.TestCase):
    def test_full_hh_static_scales_match_per_step_bptt_and_rtrl(self):
        # A short correctness check, without target generation or performance
        # timing. Both population rows share the same per-CV optimizer roots.
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(2,))
            for mechanism, name, conductance in (("IL", "leak", 0.3), ("K_HH1952", "k", 3.6),
                                                ("Na_HH1952", "na", 12.0)):
                cell.paint(AllRegion(), braincell.mech.Channel(
                    mechanism, name=name, g_max=conductance * u.mS / u.cm**2,
                ))
                cell.channels[name].trainable(g_max=braincell.trainable.scale(group_by="cv", name=name))
            cell.init_state()
            data = jnp.full((3,), -60.0)
            reference = _engine(cell, "bptt")
            with patch.object(cell.trainables, "_rollout_materialization_states", return_value=None):
                reference.prepare(data[0])
            expected = reference(data)
            for method in ("bptt", "rtrl"):
                engine = _engine(cell, method)
                actual = engine(data)
                self.assertEqual(engine.materialization_mode, "rollout")
                np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-11, atol=1e-11)
                for name in ("leak", "k", "na"):
                    np.testing.assert_allclose(actual.gradients[name], expected.gradients[name],
                                               rtol=1e-8, atol=1e-10)

    def test_physical_parameters_transforms_and_parameterized_api_use_rollout_schedule(self):
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            sources = (
                braincell.trainable.parameter(group_by="cv", name="g"),
                braincell.trainable.parameter(
                    group_by="cv", name="g", transform=brainstate.nn.SoftplusT(0.0 * u.mS / u.cm**2),
                ),
                braincell.trainable.parameterized(
                    lambda context, factor: factor * (0.3 * u.mS / u.cm**2),
                    factor=braincell.trainable.parameter(1.0, group_by="all", name="g"),
                ),
            )
            for source in sources:
                for method in ("bptt", "rtrl"):
                    with self.subTest(source=type(source).__name__, method=method):
                        cell = _cell(population=2, source=source)
                        actual = _engine(cell, method)
                        actual.prepare(jnp.asarray(-60.0))
                        self.assertEqual(actual.materialization_mode, "rollout")
                        expected_direct = int(
                            hasattr(source, "transform") and type(source.transform) is brainstate.nn.IdentityT
                        )
                        self.assertEqual(len(cell.trainables._rollout_direct_reads()), expected_direct)
                        reference = _engine(cell, method)
                        with patch.object(cell.trainables, "_rollout_materialization_states", return_value=None):
                            reference.prepare(jnp.asarray(-60.0))
                        data = jnp.full((3,), -60.0)
                        a, b = actual(data), reference(data)
                        np.testing.assert_allclose(a.loss, b.loss, rtol=1e-11, atol=1e-11)
                        self.assertEqual(jax.tree.structure(a.gradients), jax.tree.structure(b.gradients))
                        for ag, bg in zip(jax.tree.leaves(a.gradients), jax.tree.leaves(b.gradients)):
                            np.testing.assert_allclose(ag, bg, rtol=1e-9, atol=1e-10)

    def test_direct_physical_gradient_tree_matches_bptt_in_all_rtrl_outputs(self):
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _cell(source=braincell.trainable.parameter(group_by="cv", name="g"))
            data = jnp.full((3,), -60.0)
            bptt = _engine(cell, "bptt")(data)
            rtrl = _engine(cell, "rtrl")
            outputs = [rtrl(data), rtrl.diagnose(data, at=(1,)), rtrl.diagnose(data)]
            trajectory = build_trajectory_value_and_grad(
                cell, step=rtrl.step, loss=lambda observations, _: jnp.sum(observations), method="rtrl",
            )
            outputs.append(trajectory(data))
            expected_tree = jax.tree.structure(bptt.gradients)
            for result in outputs:
                self.assertEqual(jax.tree.structure(result.gradients), expected_tree)
                for actual, expected in zip(jax.tree.leaves(result.gradients), jax.tree.leaves(bptt.gradients)):
                    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)

    def test_static_mappings_match_step_schedule_and_refresh_updated_roots(self):
        data = jnp.asarray([0.1, 0.3, -0.2])
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for mapping in ("direct", "scale", "distribution"):
                for method in ("bptt", "rtrl"):
                    with self.subTest(mapping=mapping, method=method):
                        target = _MappedRecurrence(mapping)
                        engine = build_rollout_value_and_grad(target, step=target.step, method=method)
                        self.assertIsNone(engine.materialization_mode)
                        engine.prepare(data[0])
                        self.assertEqual(engine.materialization_mode, "rollout")
                        physical_written = id(target.g) in {
                            id(state) for state in engine._functional_step.state_trace.get_write_states()
                        }
                        self.assertEqual(physical_written, mapping == "direct")
                        reference = build_rollout_value_and_grad(target, step=target.step, method=method)
                        with patch.object(target.trainables, "_rollout_materialization_states", return_value=None):
                            reference.prepare(data[0])
                        self.assertEqual(reference.materialization_mode, "step")
                        if mapping == "distribution":
                            optimized_graph = engine._functional_step.function.get_jaxpr(data[0]).jaxpr
                            reference_graph = reference._functional_step.function.get_jaxpr(data[0]).jaxpr
                            self.assertNotIn("exp", {eqn.primitive.name for eqn in optimized_graph.eqns})
                            self.assertIn("exp", {eqn.primitive.name for eqn in reference_graph.eqns})
                        actual = brainstate.transform.jit(lambda: engine(data))
                        expected = brainstate.transform.jit(lambda: reference(data))
                        for theta in ([0.4, 0.7], [0.8, 0.5]):
                            target.root.val.value = jnp.asarray(theta)
                            a, b = actual(), expected()
                            np.testing.assert_allclose(a.loss, b.loss, rtol=1e-11, atol=1e-11)
                            np.testing.assert_allclose(a.gradients["theta"], b.gradients["theta"],
                                                       rtol=1e-10, atol=1e-11)

    def test_mutations_and_state_dependent_mappings_retain_step_schedule(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for mapping, mutation in (("dynamic", None), ("direct", "root"), ("scale", "physical")):
                with self.subTest(mapping=mapping, mutation=mutation):
                    target = _MappedRecurrence(mapping, mutation=mutation)
                    engine = build_rollout_value_and_grad(target, step=target.step)
                    engine.prepare(jnp.asarray(0.1))
                    self.assertEqual(engine.materialization_mode, "step")
                    reference = build_rollout_value_and_grad(target, step=target.step, method="bptt")
                    data = jnp.asarray([0.1, 0.3])
                    a, b = engine(data), reference(data)
                    np.testing.assert_allclose(a.loss, b.loss, rtol=1e-11, atol=1e-11)
                    np.testing.assert_allclose(a.gradients["theta"], b.gradients["theta"],
                                               rtol=1e-10, atol=1e-11)

    def test_non_state_backed_nodes_and_source_side_effects_are_not_hoisted(self):
        target = _MappedRecurrence()
        node = target._runtime.runtime_nodes[0]
        node.g_max = target.g.value
        self.assertIsNone(target.trainables._rollout_materialization_states())
        node.g_max = target.g

        class RefreshingChannel(IonChannel):
            def _on_param_updated(self, name, value):
                self.cached = value * 2

        custom = RefreshingChannel(size=(2,))
        custom.g_max = target.g
        target._runtime.runtime_nodes[0] = custom
        self.assertIsNone(target.trainables._rollout_materialization_states())
        target._runtime.runtime_nodes[0] = node

        original = target.trainables._binding_list[0]
        def evaluate():
            target.x.value = target.x.value + 1
            return target.root.value()

        target.trainables._binding_list[0] = replace(original, _evaluate=evaluate)
        before = target.x.value
        self.assertIsNone(target.trainables._rollout_materialization_states())
        np.testing.assert_array_equal(target.x.value, before)

    def test_partial_and_reordered_coordinates_use_entry_mapping(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for partial in (True, False):
                target = _MappedRecurrence()
                binding = target.trainables._binding_list[0]
                rows = binding._rows[:1] if partial else tuple(reversed(binding._rows))
                target.trainables._binding_list[0] = replace(
                    binding, _rows=rows, row_keys=tuple((row.population_index, row.cv_id) for row in rows),
                    _evaluate=(lambda: target.root.value()[:1]) if partial else target.evaluate,
                )
                self.assertEqual(target.trainables._rollout_direct_reads(), ())
                optimized = build_rollout_value_and_grad(target, step=target.step, method="rtrl")
                optimized.prepare(jnp.asarray(0.1))
                self.assertEqual(optimized.materialization_mode, "rollout")
                reference = build_rollout_value_and_grad(target, step=target.step, method="bptt")
                with patch.object(target.trainables, "_rollout_materialization_states", return_value=None):
                    reference.prepare(jnp.asarray(0.1))
                data = jnp.asarray([0.1, 0.3])
                a, b = optimized(data), reference(data)
                np.testing.assert_allclose(a.loss, b.loss, rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(a.gradients["theta"], b.gradients["theta"], rtol=1e-10, atol=1e-11)

    def test_custom_reset_parameter_changes_preserve_first_step_refresh(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for method in ("bptt", "rtrl"):
                target = _MappedRecurrence("scale")

                def initialize():
                    target.reset_state()
                    target.root.val.value = target.root.val.value * 1.5
                    target.g.value = jnp.ones((1, 2)) * 9.

                engine = build_rollout_value_and_grad(
                    target, step=target.step, initializer=initialize, method=method,
                )
                engine.prepare(jnp.asarray(0.1))
                reference = build_rollout_value_and_grad(
                    target, step=target.step, initializer=initialize, method=method,
                )
                with patch.object(target.trainables, "_rollout_materialization_states", return_value=None):
                    reference.prepare(jnp.asarray(0.1))
                data = jnp.asarray([0.1, 0.3])
                a, b = engine(data), reference(data)
                np.testing.assert_allclose(a.loss, b.loss, rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(a.gradients["theta"], b.gradients["theta"],
                                           rtol=1e-10, atol=1e-11)

    def test_trajectory_schedule_matches_additive_loss_and_rtrl(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            target = _MappedRecurrence("distribution")
            data = jnp.asarray([0.1, -0.2, 0.4])
            reference = build_rollout_value_and_grad(target, step=target.step, method="bptt")(data)
            for method in ("bptt", "rtrl"):
                engine = build_trajectory_value_and_grad(
                    target, step=target.step, loss=lambda observations, _: jnp.sum(observations), method=method,
                )
                result = engine(data)
                self.assertEqual(engine.materialization_mode, "rollout")
                np.testing.assert_allclose(result.loss, reference.loss, rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(result.gradients["theta"], reference.gradients["theta"],
                                           rtol=1e-10, atol=1e-11)


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
