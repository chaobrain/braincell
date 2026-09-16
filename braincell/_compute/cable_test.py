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

"""Independent cable assembly and finite-difference checks in float64."""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell import Branch, Cell, CVPerBranch, Morphology
from braincell._compute.cable import CableArrays, CableTopology, axial_matrix
from braincell._compute.scheduling import build_node_scheduling


def _fork():
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um, type="soma")
    morph = Morphology.from_root(soma, name="soma")
    for name, radii in (("a", [2.0, 1.5, 1.0]), ("b", [1.5, 1.0, 0.8])):
        branch = Branch.from_lengths(lengths=[15.0, 25.0] * u.um, radii=radii * u.um, type="dendrite")
        morph.attach(parent="soma", child_branch=branch, child_name=name)
    return Cell(morph, cv_policy=CVPerBranch(2))


def _host_reference(cell, point_to_row):
    """Original scalar physical assembly, independent of the array kernel."""
    cvs, tree = cell.cvs, cell.node_tree
    capacitance = np.ones(len(tree.nodes))
    for cv in cvs:
        row = point_to_row[tree.cv_to_mid_node_id[cv.id]]
        capacitance[row] = float((cv.area * cv.cm).to_decimal(u.uF))
    matrix = np.zeros((len(tree.nodes), len(tree.nodes)))
    for edge in tree.edges:
        resistances = [
            float(
                (cvs[role.cv_id].r_axial_prox if role.half == "prox" else cvs[role.cv_id].r_axial_dist).to_decimal(
                    u.ohm
                )
            )
            for role in edge.roles
        ]
        branches = {cvs[role.cv_id].branch_id for role in edge.roles}
        g = 1.0 / sum(resistances) if len(resistances) > 1 and len(branches) == 1 else sum(1.0 / r for r in resistances)
        p, c = point_to_row[edge.parent_node_id], point_to_row[edge.child_node_id]
        matrix[p, p] += 1e3 * g / capacitance[p]
        matrix[p, c] -= 1e3 * g / capacitance[p]
        matrix[c, c] += 1e3 * g / capacitance[c]
        matrix[c, p] -= 1e3 * g / capacitance[c]
    return matrix, capacitance


class CableArraysTest(unittest.TestCase):
    def test_population_axes_match_independent_operators_and_gradients(self):
        with brainstate.environ.context(precision=64):
            cell = _fork()
            mapping = build_node_scheduling(cell.node_tree).point_id_to_row
            topology = CableTopology.build(cvs=cell.cvs, node_tree=cell.node_tree, point_id_to_row=mapping)
            reference = CableArrays.from_cvs(cell.cvs)
            factors = jnp.linspace(0.7, 1.6, 6 * cell.n_cv).reshape(2, 3, cell.n_cv)

            def operators(scales):
                # A shared area/cm paired with batched resistance exercises
                # broadcasting independently of the geometry parameterization.
                cable = reference._replace(resistance_prox=reference.resistance_prox * scales)
                matrix, capacitance = axial_matrix(cable, topology)
                return matrix.to_decimal(u.ms**-1), capacitance.to_decimal(u.uF)

            compiled = jax.jit(operators)
            matrices, capacitances = compiled(factors)
            self.assertEqual(matrices.shape, (2, 3, topology.n_point, topology.n_point))
            for i in range(2):
                for j in range(3):
                    expected, expected_c = operators(factors[i, j])
                    np.testing.assert_allclose(matrices[i, j], expected, rtol=1e-13)
                    np.testing.assert_allclose(capacitances[i, j], expected_c, rtol=1e-13)

            def response(scales):
                matrices, _ = operators(scales)
                rhs = jnp.linspace(-65.0, -30.0, topology.n_point)
                states = jnp.linalg.solve(
                    jnp.eye(topology.n_point) + 0.025 * matrices,
                    jnp.broadcast_to(rhs, scales.shape[:-1] + rhs.shape)[..., None],
                )[..., 0]
                return states[..., topology.cv_rows[-1]].sum()

            gradient = jax.jit(jax.grad(response))(factors)
            direction = jnp.zeros_like(factors).at[1, 2, -1].set(1.0)
            epsilon = 1e-4
            finite = (response(factors + epsilon * direction) - response(factors - epsilon * direction)) / (2 * epsilon)
            self.assertGreater(abs(float(gradient[1, 2, -1])), 1e-7)
            np.testing.assert_allclose(gradient[1, 2, -1], finite, rtol=2e-5, atol=1e-7)
            changed, _ = compiled(factors.at[1, 2, -1].multiply(1.2))
            np.testing.assert_array_equal(changed[0], matrices[0])
            self.assertGreater(float(jnp.max(jnp.abs(changed[1, 2] - matrices[1, 2]))), 1e-7)

    def test_fork_taper_and_interior_series_match_scalar_reference(self):
        with brainstate.environ.context(precision=64):
            cell = _fork()
            mapping = build_node_scheduling(cell.node_tree).point_id_to_row
            topo = CableTopology.build(cvs=cell.cvs, node_tree=cell.node_tree, point_id_to_row=mapping)
            cable = CableArrays.from_cvs(cell.cvs)
            actual, capacitance = axial_matrix(cable, topo)
            expected, expected_c = _host_reference(cell, mapping)
            self.assertTrue(np.any(topo.series_edges))
            self.assertTrue(np.any(~topo.series_edges))
            self.assertEqual(topo.cv_rows.dtype, np.int32)
            for value in cable:
                self.assertIsInstance(value.mantissa, jax.Array)
                self.assertEqual(value.dtype, jnp.float64)
            np.testing.assert_allclose(actual.to_decimal(u.ms**-1), expected, rtol=2e-14, atol=1e-12)
            np.testing.assert_allclose(capacitance.to_decimal(u.uF), expected_c, rtol=2e-14)
            np.testing.assert_allclose(np.asarray(actual.mantissa).sum(axis=1), 0.0, atol=1e-10)

    def test_array_input_gradients_match_finite_difference_through_voltage_solve(self):
        with brainstate.environ.context(precision=64):
            cell = _fork()
            mapping = build_node_scheduling(cell.node_tree).point_id_to_row
            topo = CableTopology.build(cvs=cell.cvs, node_tree=cell.node_tree, point_id_to_row=mapping)
            reference = CableArrays.from_cvs(cell.cvs)
            n = len(cell.cvs)
            storage = jnp.zeros(topo.n_point).at[topo.cv_rows].set(1.0)
            initial = jnp.linspace(-65.0, -30.0, n)
            rhs = jnp.zeros(topo.n_point).at[topo.cv_rows].set(initial)

            def response(scales):
                values = CableArrays(*(field * scales[i] for i, field in enumerate(reference)))
                operator, _ = axial_matrix(values, topo)
                solution = jnp.linalg.solve(jnp.diag(storage) + 0.025 * operator.to_decimal(u.ms**-1), rhs)
                return solution[topo.cv_rows[-1]]

            value_and_grad = jax.jit(jax.value_and_grad(response))
            scales = jnp.ones((4, n))
            voltage, gradient = value_and_grad(scales)
            self.assertTrue(np.isfinite(voltage))
            self.assertTrue(np.all(np.isfinite(gradient)))
            # Four physical array families, testing nonzero directional paths.
            for family in range(4):
                direction = jnp.zeros_like(scales).at[family].set(jnp.linspace(0.2, 1.0, n))
                epsilon = 1e-4
                finite = (response(scales + epsilon * direction) - response(scales - epsilon * direction)) / (
                    2 * epsilon
                )
                automatic = jnp.sum(gradient * direction)
                self.assertGreater(abs(float(automatic)), 1e-7)
                np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-7)
            changed, _ = value_and_grad(scales.at[2, -1].set(1.5))
            self.assertGreater(abs(float(changed - voltage)), 1e-7)


if __name__ == "__main__":
    unittest.main()
