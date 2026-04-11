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

import unittest
from unittest import mock

import brainstate
import brainunit as u

import braincell
from braincell import Branch, Cell, Morphology
from braincell.filter import BranchSlice, RootLocation, Terminals


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def _build_soma_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    return Morphology.from_root(soma, name="soma")


class CellExecutionTest(unittest.TestCase):
    def test_cell_is_runtime_object_after_init_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm**2)),
        )

        cell.init_state()

        self.assertIsInstance(cell, braincell.HHTypedNeuron)
        self.assertEqual(cell.varshape, (2,))
        self.assertIs(cell.ion_channels["na"], cell.get_ion("na"))
        self.assertIs(cell.ion_channels["k"], cell.get_ion("k"))
        self.assertIs(cell.ion_channels["ca"], cell.get_ion("ca"))
        self.assertTrue(any(key.startswith("layout_") for key in cell.ion_channels))

    def test_reset_state_reinitializes_cell_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertEqual(cell.spike.value.shape, (2,))

        original = cell.V.value
        cell.V.value = cell.V.value + 1.0 * u.mV
        cell.reset_state()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertFalse(u.math.all(cell.V.value == original + 1.0 * u.mV))

    def test_compute_derivative_and_update_work_on_cell(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        cell.compute_derivative()
        self.assertEqual(cell.V.derivative.shape, (2,))

        with brainstate.environ.context(dt=0.01 * u.ms):
            spike = cell.update()
        self.assertEqual(spike.shape, (2,))

    def test_staggered_solver_updates_cv_sized_voltage(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        with brainstate.environ.context(dt=0.01 * u.ms):
            spike = cell.update()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertEqual(spike.shape, (2,))

    def test_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="exp_euler")
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()

        with mock.patch("braincell.quad._exp_euler.apply_standard_solver_step") as step_mock:
            with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
                cell.update()

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "concat")

    def test_splitting_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="splitting")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                braincell.quad.splitting_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)

    def test_cn_rk4_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="cn_rk4")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.rk4_step") as rk4_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap", side_effect=RuntimeError("bridge reached")):
                    braincell.quad.cn_rk4_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(rk4_mock.called)

    def test_cn_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="cn_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap", side_effect=RuntimeError("bridge reached")):
                    braincell.quad.cn_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_implicit_rk4_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="implicit_rk4")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.rk4_step") as rk4_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap", side_effect=RuntimeError("bridge reached")):
                    braincell.quad.implicit_rk4_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(rk4_mock.called)

    def test_implicit_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="implicit_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap", side_effect=RuntimeError("bridge reached")):
                    braincell.quad.implicit_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_exp_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="exp_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap", side_effect=RuntimeError("bridge reached")):
                    braincell.quad.exp_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_staggered_solver_single_cv_leak_moves_toward_reversal(self) -> None:
        cell = Cell(_build_soma_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        initial = cell.V.value
        with brainstate.environ.context(dt=0.01 * u.ms):
            cell.update()
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), float(initial[0].to_decimal(u.mV)))
        self.assertGreater(float(cell.V.value[0].to_decimal(u.mV)), -68.0)

    def test_total_current_input_matches_current_density_input(self) -> None:
        cell_total = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell_density = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        for cell in (cell_total, cell_density):
            cell.paint(
                BranchSlice(branch_index=0, prox=0.0, dist=1.0),
                braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
            )
            cell.init_state()

        initial = -65.0 * u.mV
        cell_total.V.value = u.math.asarray([initial.to_decimal(u.mV)]) * u.mV
        cell_density.V.value = u.math.asarray([initial.to_decimal(u.mV)]) * u.mV
        area = cell_total.cvs[0].area
        total_current = 0.05 * u.nA
        current_density = total_current / area

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            cell_total.update(total_current)
            cell_density.update(current_density)

        self.assertAlmostEqual(
            float(cell_total.V.value[0].to_decimal(u.mV)),
            float(cell_density.V.value[0].to_decimal(u.mV)),
            places=6,
        )

    def test_compute_axial_derivative_is_zero_for_single_cv(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.init_state()

        derivative = cell.compute_axial_derivative(u.math.asarray([-65.0]) * u.mV)
        self.assertEqual(derivative.shape, (1,))
        self.assertAlmostEqual(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_compute_axial_derivative_couples_two_cv_cable(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.init_state()

        derivative = cell.compute_axial_derivative(u.math.asarray([-60.0, -80.0]) * u.mV)
        self.assertEqual(derivative.shape, (2,))
        self.assertLess(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertGreater(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0)

    def test_explicit_solver_uses_axial_coupling(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.init_state()
        cell.V.value = u.math.asarray([-60.0, -80.0]) * u.mV

        with brainstate.environ.context(dt=1e-6 * u.ms, t=0.0 * u.ms):
            cell.update()

        self.assertGreater(float(cell.V.value[0].to_decimal(u.mV)), -80.0)
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), -60.0)
        self.assertGreater(float(cell.V.value[1].to_decimal(u.mV)), -80.0)
        self.assertLess(float(cell.V.value[1].to_decimal(u.mV)), -60.0)

    def test_staggered_solver_two_cv_passive_cable_stays_symmetric(self) -> None:
        tree = _build_soma_tree()
        cell = Cell(tree, solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        initial = cell.V.value
        with brainstate.environ.context(dt=0.01 * u.ms):
            cell.update()

        self.assertEqual(cell.V.value.shape, (2,))
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), float(initial[0].to_decimal(u.mV)))
        self.assertLess(float(cell.V.value[1].to_decimal(u.mV)), float(initial[1].to_decimal(u.mV)))
        self.assertAlmostEqual(
            float(cell.V.value[0].to_decimal(u.mV)),
            float(cell.V.value[1].to_decimal(u.mV)),
            places=6,
        )

    def test_staggered_solver_branched_passive_cell_matches_explicit_direction(self) -> None:
        tree = _build_tree()
        explicit = Cell(tree, solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        staggered = Cell(tree, solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        for cell in (explicit, staggered):
            cell.paint(
                BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
                braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
            )
            cell.init_state()

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            explicit.update()
            staggered.update()

        self.assertEqual(staggered.V.value.shape, explicit.V.value.shape)
        for index in range(staggered.V.value.shape[0]):
            explicit_v = float(explicit.V.value[index].to_decimal(u.mV))
            staggered_v = float(staggered.V.value[index].to_decimal(u.mV))
            self.assertLess(staggered_v, -65.0)
            self.assertGreater(staggered_v, -68.0)
            self.assertAlmostEqual(staggered_v, explicit_v, places=3)

    def test_update_requires_init_state_after_declaration_change(self) -> None:
        cell = Cell(_build_tree())
        cell.init_state()
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        with self.assertRaisesRegex(ValueError, "Cell.init_state"):
            cell.compute_derivative()

    def test_placed_current_clamp_removes_manual_step_current_logic(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp(
                start=1.0 * u.ms,
                durations=(2.0 * u.ms, 2.0 * u.ms),
                amplitudes=(0.0 * u.nA, 0.2 * u.nA),
            ),
        )
        cell.init_state()

        with brainstate.environ.context(t=0.5 * u.ms):
            early = cell.compute_membrane_derivative(cell.V.value)
        with brainstate.environ.context(t=3.5 * u.ms):
            late = cell.compute_membrane_derivative(cell.V.value)
        with brainstate.environ.context(t=5.5 * u.ms):
            after = cell.compute_membrane_derivative(cell.V.value)

        self.assertAlmostEqual(float(early[0].to_decimal(u.mV / u.ms)), 0.0, places=8)
        self.assertGreater(float(late[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertAlmostEqual(float(after[0].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_placed_current_clamp_targets_only_selected_point(self) -> None:
        cell = Cell(_build_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp(amplitude=0.2 * u.nA, delay=0.0 * u.ms, duration=10.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(t=1.0 * u.ms):
            derivative = cell.compute_membrane_derivative(cell.V.value)

        self.assertGreater(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertAlmostEqual(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_multiple_terminal_clamps_do_not_broadcast_to_all_points(self) -> None:
        cell = Cell(_build_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            Terminals(),
            braincell.CurrentClamp(amplitude=0.15 * u.nA, delay=0.0 * u.ms, duration=10.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(t=1.0 * u.ms):
            derivative = cell.compute_membrane_derivative(cell.V.value)

        self.assertAlmostEqual(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0, places=8)
        self.assertGreater(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0)
