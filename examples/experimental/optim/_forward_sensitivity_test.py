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

"""Tests for the experimental exact forward-sensitivity core."""

import unittest

import braincell
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute._testing import _build_tree
from braincell.filter import AllRegion, RootLocation

from examples.experimental.optim._forward_sensitivity import (
    bptt_reference_loss,
    build_stateful_step,
    forward_sensitivity_rollout,
    initialize_forward_sensitivity,
    initialized_bptt_reference_loss,
    seed_scalar_parameter_directions,
    select_parameter_derivatives,
)


def _leak_cell():
    cell = braincell.Cell(_build_tree(), pop_size=(1,))
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


def _hh_cell():
    cell = braincell.Cell(_build_tree(), pop_size=(1,))
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "Na_HH1952",
            name="na",
            g_max=12.0 * u.mS / u.cm**2,
        ),
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "K_HH1952",
            name="k",
            g_max=3.6 * u.mS / u.cm**2,
        ),
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "IL",
            name="leak",
            g_max=0.3 * u.mS / u.cm**2,
            E=-54.3 * u.mV,
        ),
    )
    cell.channels["na"].trainable(g_max=braincell.trainable.scale(group_by="all", name="na.factor"))
    cell.channels["k"].trainable(g_max=braincell.trainable.scale(group_by="all", name="k.factor"))
    cell.init_state()
    return cell


def _functional_voltage_loss(cell, *, example_target):
    def local_loss(target_mv):
        cell.trainables.materialize()
        cell.update()
        error = cell.V.value.to_decimal(u.mV) - target_mv
        return jnp.mean(error * error)

    return build_stateful_step(
        local_loss,
        example_target,
        cell.trainables.parameters().states(),
    )


class ForwardSensitivityCoreTest(unittest.TestCase):
    def test_leak_prefix_gradients_match_reverse_mode_and_finite_difference(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _leak_cell()
            target = jnp.asarray(-60.0, dtype=jnp.float64)

            def weighted_local_loss(data):
                target_mv, loss_weight = data
                cell.trainables.materialize()
                cell.update()
                error = cell.V.value.to_decimal(u.mV) - target_mv
                return loss_weight * jnp.mean(error * error)

            step = build_stateful_step(
                weighted_local_loss,
                (target, jnp.asarray(1.0, dtype=jnp.float64)),
                cell.trainables.parameters().states(),
            )
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            targets = jnp.full((8,), target)
            step_data = (targets, jnp.ones((8,), dtype=jnp.float64))

            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, step_data)
            reverse_state_gradient = jax.grad(lambda values: bptt_reference_loss(step, values, step_data))(
                initial_values
            )
            reverse = select_parameter_derivatives(step, reverse_state_gradient)["leak.factor"]

            def reverse_prefix(loss_weights):
                derivative = jax.grad(lambda values: bptt_reference_loss(step, values, (targets, loss_weights)))(
                    initial_values
                )
                return select_parameter_derivatives(step, derivative)["leak.factor"]

            reverse_prefixes = jax.vmap(reverse_prefix)(jnp.tril(jnp.ones((8, 8), dtype=jnp.float64)))

            epsilon = 1e-4
            parameter_index = step.parameter_indices[0]

            def shifted_loss(delta):
                shifted = list(initial_values)
                shifted[parameter_index] = shifted[parameter_index] + delta
                return bptt_reference_loss(step, tuple(shifted), step_data)

            finite_difference = (shifted_loss(epsilon) - shifted_loss(-epsilon)) / (2.0 * epsilon)

        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward.prefix_gradients[:, 0], reverse_prefixes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward.final_gradient[0], finite_difference, rtol=1e-6, atol=1e-8)
        self.assertEqual(forward.final_gradient.shape, (1,))
        self.assertEqual(forward.prefix_gradients.shape, (8, 1))
        self.assertEqual(forward.local_gradients.shape, (8, 1))
        for tangent, value in zip(
            jax.tree.leaves(forward.final_state_tangents),
            jax.tree.leaves(forward.final_state_values),
        ):
            self.assertEqual(np.shape(tangent), (1,) + np.shape(value))
        np.testing.assert_allclose(
            forward.prefix_gradients,
            np.cumsum(np.asarray(forward.local_gradients), axis=0),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_hh_multiple_parameter_directions_match_reverse_mode(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _hh_cell()
            target = jnp.asarray(-62.0, dtype=jnp.float64)
            step = _functional_voltage_loss(cell, example_target=target)
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            targets = jnp.full((4,), target)

            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, targets)
            reverse_state_gradient = jax.grad(lambda values: bptt_reference_loss(step, values, targets))(initial_values)
            selected = select_parameter_derivatives(step, reverse_state_gradient)
            reverse = jnp.stack([selected[name] for name in step.parameter_names])

        np.testing.assert_allclose(forward.final_gradient, reverse, rtol=1e-9, atol=1e-10)
        self.assertEqual(forward.final_gradient.shape, (2,))
        self.assertEqual(forward.prefix_gradients.shape, (4, 2))

    def test_parameter_dependent_reset_is_included_in_initial_sensitivity(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel("Na_HH1952", name="na", g_max=12.0 * u.mS / u.cm**2),
            )
            cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            cell.channels["na"].trainable(V_sh=braincell.trainable.scale(group_by="all", name="na.vsh.factor"))
            cell.init_state()
            parameters = cell.trainables.parameters().states()

            def reset_and_zero(_):
                cell.reset_state()
                return jnp.asarray(0.0, dtype=jnp.float64)

            initializer = build_stateful_step(reset_and_zero, None, parameters)
            target = jnp.asarray(-62.0, dtype=jnp.float64)
            step = _functional_voltage_loss(cell, example_target=target)
            initial_values, initial_tangents = initialize_forward_sensitivity(initializer, step, None)
            targets = jnp.full((3,), target)
            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, targets)

            initializer_values = initializer.state_values()
            reverse_initializer_gradient = jax.grad(
                lambda values: initialized_bptt_reference_loss(
                    initializer,
                    step,
                    values,
                    None,
                    targets,
                )
            )(initializer_values)
            reverse = select_parameter_derivatives(initializer, reverse_initializer_gradient)["na.vsh.factor"]

            naive_values = step.state_values()
            naive_tangents = seed_scalar_parameter_directions(step, naive_values)
            naive = forward_sensitivity_rollout(step, naive_values, naive_tangents, targets)

        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-9, atol=1e-10)
        self.assertGreater(float(jnp.abs(forward.final_gradient[0] - naive.final_gradient[0])), 1e-6)

    def test_fixed_delayed_external_clamps_remain_exact_scan_inputs(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            cell.place(
                RootLocation(x=0.0),
                braincell.mech.CurrentClamp(
                    delay=0.025 * u.ms,
                    durations=0.05 * u.ms,
                    amplitudes=0.2 * u.nA,
                ),
            )
            cell.place(
                RootLocation(x=1.0),
                braincell.mech.CurrentClamp(
                    delay=0.075 * u.ms,
                    durations=0.05 * u.ms,
                    amplitudes=0.1 * u.nA,
                ),
            )
            cell.channels["leak"].trainable(g_max=braincell.trainable.scale(group_by="all", name="leak.factor"))
            cell.init_state()
            target = jnp.asarray(-60.0, dtype=jnp.float64)

            def local_loss(data):
                time_ms, target_mv = data
                with brainstate.environ.context(t=time_ms * u.ms):
                    cell.trainables.materialize()
                    cell.update()
                error = cell.V.value.to_decimal(u.mV) - target_mv
                return jnp.mean(error * error)

            example_data = (jnp.asarray(0.0, dtype=jnp.float64), target)
            step = build_stateful_step(
                local_loss,
                example_data,
                cell.trainables.parameters().states(),
            )
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            step_data = (
                jnp.arange(6, dtype=jnp.float64) * 0.025,
                jnp.full((6,), target),
            )
            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, step_data)
            reverse_state_gradient = jax.grad(lambda values: bptt_reference_loss(step, values, step_data))(
                initial_values
            )
            reverse = select_parameter_derivatives(step, reverse_state_gradient)["leak.factor"]

            clamp_delays = sorted(
                float(delay)
                for layout in cell.layouts
                if layout.kind == "CurrentClamp"
                for delay in np.asarray(cell.get_state(layout.id, "delay").to_decimal(u.ms)).reshape(-1)
            )

        np.testing.assert_allclose(clamp_delays, [0.025, 0.075], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-9, atol=1e-10)
        self.assertTrue(bool(jnp.all(jnp.isfinite(forward.local_gradients))))

    def test_requires_scalar_local_loss_and_scalar_coordinate_roots(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _leak_cell()
            parameter_states = cell.trainables.parameters().states()

            def vector_loss(_):
                cell.trainables.materialize()
                return jnp.ones((2,))

            with self.assertRaisesRegex(ValueError, "scalar local loss"):
                build_stateful_step(vector_loss, None, parameter_states)

            vector_cell = braincell.Cell(_build_tree(), pop_size=(2,))
            vector_cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            vector_cell.channels["leak"].trainable(
                g_max=braincell.trainable.scale(group_by="population", name="leak.population.factor")
            )
            vector_cell.init_state()
            vector_step = _functional_voltage_loss(vector_cell, example_target=jnp.asarray(-60.0))
            with self.assertRaisesRegex(ValueError, "not scalar"):
                seed_scalar_parameter_directions(vector_step, vector_step.state_values())


if __name__ == "__main__":
    unittest.main()
