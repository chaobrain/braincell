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

"""Tests for exact compact RTRL on configurable multi-CV HH cells."""

import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from examples.experimental.online_learning.multicv_hh_rtrl import (
    FIVE_CV_TARGET_ROW_SCALES,
    THREE_CV_TARGET_ROW_SCALES,
    bptt_loss,
    build_bifurcating_morphology,
    build_multicv_problem,
    compare_gradients,
)


class _X64TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._x64_context = jax.enable_x64(True)
        self._x64_context.__enter__()
        self.addCleanup(self._x64_context.__exit__, None, None, None)


class ThreeCVRegressionTest(_X64TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            cls.problem = build_multicv_problem(
                dendrite_segments=(1, 1),
                target_row_scales=THREE_CV_TARGET_ROW_SCALES,
                num_steps=8,
            )
            cls.comparison = compare_gradients(cls.problem)

    def test_three_cv_shape_and_gradient_regression(self) -> None:
        self.assertEqual(self.problem.cell.n_compartment, 3)
        self.assertEqual(self.problem.projection.size, 12)
        self.assertEqual(self.problem.parameter_coordinates.size, 9)
        self.assertEqual(self.comparison.compact_sensitivity.shape, (9, 12))
        np.testing.assert_allclose(
            self.comparison.compact_gradient,
            self.comparison.full_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.comparison.compact_gradient,
            self.comparison.bptt_gradient,
            rtol=1e-10,
            atol=1e-12,
        )


class FiveCVHHRTRLTest(_X64TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            cls.problem = build_multicv_problem(
                dendrite_segments=(2, 2),
                target_row_scales=FIVE_CV_TARGET_ROW_SCALES,
                num_steps=8,
            )
            cls.comparison = compare_gradients(cls.problem)

            directions = (
                jnp.asarray([1.0, -0.5, 0.25, -0.2, 0.4]),
                jnp.asarray([-0.3, 0.8, 0.4, -0.6, 0.2]),
                jnp.asarray([0.6, -0.2, -0.7, 0.3, 0.5]),
            )
            direction_vector = cls.problem.parameter_coordinates.flatten(
                dict(zip(cls.problem.parameter_coordinates.names, directions))
            )
            epsilon = 1e-4
            plus = tuple(
                value + epsilon * direction for value, direction in zip(cls.problem.parameter_values, directions)
            )
            minus = tuple(
                value - epsilon * direction for value, direction in zip(cls.problem.parameter_values, directions)
            )
            plus_loss = bptt_loss(cls.problem, plus, cls.problem.step_data)
            minus_loss = bptt_loss(cls.problem, minus, cls.problem.step_data)
            cls.finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon)
            cls.directional_derivative = jnp.dot(cls.comparison.compact_gradient, direction_vector)

    def test_morphology_builder_validates_and_counts_segments(self) -> None:
        morphology = build_bifurcating_morphology((2, 2))
        self.assertEqual(len(morphology.branches), 5)
        self.assertEqual(
            tuple(branch.name for branch in morphology.branches),
            ("soma", "dend_a_0", "dend_a_1", "dend_b_0", "dend_b_1"),
        )
        for invalid in ((0, 2), (2, 0), (1,), [2, 2], (True, 2)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_bifurcating_morphology(invalid)

    def test_problem_has_twenty_active_states_and_fifteen_row_parameters(self) -> None:
        problem = self.problem
        self.assertEqual(problem.cell.n_compartment, 5)
        self.assertEqual(problem.projection.names, ("V", "Na.m", "Na.h", "K.n"))
        self.assertEqual(problem.projection.size, 20)
        self.assertEqual(problem.parameter_coordinates.names, ("leak.scale", "na.scale", "k.scale"))
        self.assertEqual(problem.parameter_coordinates.shapes, ((5,), (5,), (5,)))
        self.assertEqual(problem.parameter_coordinates.size, 15)
        self.assertEqual(problem.initial_active_state_tangents.shape, (15, 20))
        self.assertEqual(
            problem.initial_active_state_tangents.size * problem.initial_active_state_tangents.dtype.itemsize,
            2400,
        )
        np.testing.assert_array_equal(problem.initial_active_state_tangents, 0.0)

        na_layout = next(layout for layout in problem.cell.layouts if layout.kind == "channel:Na_HH1952")
        k_layout = next(layout for layout in problem.cell.layouts if layout.kind == "channel:K_HH1952")
        na_node = problem.cell.runtime.get_runtime_node(na_layout.id)
        k_node = problem.cell.runtime.get_runtime_node(k_layout.id)
        self.assertEqual(na_node.p.value.shape, (1, 5))
        self.assertEqual(na_node.q.value.shape, (1, 5))
        self.assertEqual(k_node.p.value.shape, (1, 5))
        self.assertIsNone(na_layout.point_index)
        np.testing.assert_array_equal(na_layout.cv_mask, np.ones(5, dtype=bool))

        channel_layouts = tuple(layout for layout in problem.cell.layouts if layout.kind.startswith("channel:"))
        for layout in channel_layouts:
            runtime_state = problem.cell.runtime.state_buffers[(layout.id, "g_max")]
            self.assertEqual(runtime_state.axis, "row")

    def test_vector_parameter_coordinate_basis_is_block_identity(self) -> None:
        problem = self.problem
        coordinates = problem.parameter_coordinates
        tangents = coordinates.seed(problem.initial_state_values)
        for root_number, (state_index, coordinate_slice) in enumerate(
            zip(coordinates.state_indices, coordinates.slices)
        ):
            actual = np.asarray(jax.tree.leaves(tangents[state_index])[0])
            expected = np.zeros((15, 5))
            expected[coordinate_slice, :] = np.eye(5)
            np.testing.assert_array_equal(actual, expected, err_msg=f"root {root_number}")

    def test_compact_full_and_bptt_gradients_match(self) -> None:
        comparison = self.comparison
        np.testing.assert_allclose(comparison.compact_loss, comparison.full_loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(comparison.compact_loss, comparison.bptt_loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            comparison.compact_gradient,
            comparison.full_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            comparison.compact_gradient,
            comparison.bptt_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
        self.assertEqual(comparison.compact_gradient.shape, (15,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(comparison.compact_gradient))))
        self.assertTrue(bool(jnp.all(comparison.compact_gradient != 0.0)))

        restored = self.problem.parameter_coordinates.unflatten(comparison.compact_gradient)
        self.assertEqual(tuple(restored), ("leak.scale", "na.scale", "k.scale"))
        self.assertTrue(all(value.shape == (5,) for value in restored.values()))

    def test_compact_gradient_matches_central_directional_difference(self) -> None:
        np.testing.assert_allclose(
            self.directional_derivative,
            self.finite_difference,
            rtol=1e-6,
            atol=1e-9,
        )

    def test_row_parameters_propagate_between_distal_arms(self) -> None:
        sensitivity = np.asarray(self.comparison.compact_sensitivity)
        # Coordinates: leak[0:5], Na[5:10], K[10:15]. Active state starts V[0:5].
        self.assertGreater(abs(sensitivity[7, 4]), 1e-8)  # distal dend_a Na -> distal dend_b V
        self.assertGreater(abs(sensitivity[14, 0]), 1e-8)  # distal dend_b K -> soma V
        self.assertGreater(abs(sensitivity[1, 3]), 1e-8)  # proximal dend_a leak -> proximal dend_b V


if __name__ == "__main__":
    unittest.main()
