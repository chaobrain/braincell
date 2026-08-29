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

"""Integration tests for Cell-local trainable parameter bindings."""

import unittest

import brainstate
import brainunit as u

import braincell
from braincell._compute._testing import _build_tree
from braincell.filter import AllRegion


def _leak_cell(*, pop_size=(2,)):
    cell = braincell.Cell(_build_tree(), pop_size=pop_size)
    cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak"))
    return cell


class TrainableManagerTest(unittest.TestCase):
    def test_grouping_produces_expected_degrees_of_freedom(self) -> None:
        expected = {"row": (4,), "population": (2,), "cv": (2,), "all": ()}
        for group_by, shape in expected.items():
            with self.subTest(group_by=group_by):
                cell = _leak_cell()
                cell.channels["leak"].trainable(
                    g_max=braincell.trainable.parameter(group_by=group_by, name=f"g.{group_by}")
                )
                value = next(iter(cell.trainables.parameters().physical_values().values()))
                self.assertEqual(value.shape, shape)

    def test_direct_materialization_and_binding_ownership(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        leak = cell.channels["leak"]
        leak.trainable(g_max=braincell.trainable.parameter(group_by="all", name="g"))
        with self.assertRaises(RuntimeError):
            leak.set(g_max=0.3 * u.mS / u.cm**2)
        cell.init_state()
        cell.trainables.parameters().set_physical_values({"g": 0.2 * u.mS / u.cm**2})
        cell.trainables.materialize()
        self.assertTrue(u.math.allclose(leak.g_max, 0.2 * u.mS / u.cm**2))
        runtime_state = vars(cell.runtime.get_runtime_node(0))["g_max"]
        self.assertEqual(runtime_state.axis, "uniform")
        self.assertEqual(runtime_state.value.shape, ())

    def test_scale_gradient_flows_through_materialized_state(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(0)
        voltage = cell.V.value

        def loss():
            cell.trainables.materialize()
            return node.current(voltage).to_decimal(u.nA / u.cm**2).sum()

        gradients = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()
        self.assertNotEqual(float(gradients["theta"]), 0.0)

    def test_equal_row_scale_initial_values_keep_independent_runtime_gradients(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(group_by="row", name="theta.row"))
        cell.init_state()
        runtime_state = cell.runtime.state_buffers[(0, "g_max")]
        self.assertEqual(runtime_state.axis, "row")

        node = cell.runtime.get_runtime_node(0)
        voltage = cell.V.value

        def loss():
            cell.trainables.materialize()
            return node.current(voltage).to_decimal(u.nA / u.cm**2).sum()

        gradient = brainstate.transform.grad(
            loss,
            grad_states=cell.trainables.parameters().states(),
        )()["theta.row"]
        self.assertEqual(gradient.shape, (2,))
        self.assertTrue(bool(u.math.all(gradient != 0.0)))

    def test_equal_scale_initial_values_preserve_declared_group_axes(self) -> None:
        expected = {
            "all": "uniform",
            "population": "population",
            "cv": "cv",
            "row": "row",
        }
        for group_by, axis in expected.items():
            with self.subTest(group_by=group_by):
                cell = _leak_cell(pop_size=(2,))
                cell.channels["leak"].trainable(
                    g_max=braincell.trainable.scale(group_by=group_by, name=f"theta.{group_by}")
                )
                cell.init_state()
                self.assertEqual(cell.runtime.state_buffers[(0, "g_max")].axis, axis)

    def test_direct_and_scale_obey_chain_rule(self) -> None:
        direct_cell = _leak_cell(pop_size=(1,))
        scale_cell = _leak_cell(pop_size=(1,))
        direct_cell.channels["leak"].trainable(g_max=braincell.trainable.parameter(group_by="all", name="g"))
        scale_cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        direct_cell.init_state()
        scale_cell.init_state()
        direct_node = direct_cell.runtime.get_runtime_node(0)
        scale_node = scale_cell.runtime.get_runtime_node(0)
        direct_voltage = direct_cell.V.value
        scale_voltage = scale_cell.V.value

        def direct_loss():
            direct_cell.trainables.materialize()
            return direct_node.current(direct_voltage).to_decimal(u.nA / u.cm**2).sum()

        def scale_loss():
            scale_cell.trainables.materialize()
            return scale_node.current(scale_voltage).to_decimal(u.nA / u.cm**2).sum()

        direct_gradient = brainstate.transform.grad(
            direct_loss, grad_states=direct_cell.trainables.parameters().states()
        )()["g"]
        scale_gradient = brainstate.transform.grad(
            scale_loss, grad_states=scale_cell.trainables.parameters().states()
        )()["theta"]
        expected = direct_gradient.to_decimal(u.mS / u.cm**2) * 0.1
        self.assertTrue(u.math.allclose(scale_gradient, expected))

    def test_cached_cell_run_remains_differentiable(self) -> None:
        cell = braincell.Cell(_build_tree(), V_init=-65.0 * u.mV)
        cell.paint(
            AllRegion(),
            braincell.mech.CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
            braincell.mech.Channel("IL", name="leak", E=-70.0 * u.mV),
        )
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        cell.soma.record("v", braincell.observe.state("v"), period=0.02 * u.ms)
        cell.init_state()

        def simulate():
            cell.reset_state()
            result = cell.run(dt=0.01 * u.ms, duration=0.04 * u.ms)
            return result.samples["v"].values.to_decimal(u.mV)

        compiled_simulate = brainstate.transform.jit(simulate)
        first_trace = compiled_simulate()
        second_trace = compiled_simulate()
        self.assertEqual(first_trace.shape[0], 2)
        self.assertTrue(u.math.allclose(first_trace, second_trace))

        def loss():
            cell.reset_state()
            result = cell.run(dt=0.01 * u.ms, duration=0.04 * u.ms)
            return result.samples["v"].values.to_decimal(u.mV).sum()

        first = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()["theta"]
        second = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()["theta"]
        self.assertNotEqual(float(first), 0.0)
        self.assertTrue(u.math.allclose(first, second))

    def test_differentiable_recording_rejects_variable_length_schedule(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        cell.soma.record("v", braincell.observe.state("v"), period=0.02 * u.ms)
        cell.init_state()

        def loss():
            cell.reset_state()
            result = cell.run(dt=0.01 * u.ms, duration=0.03 * u.ms)
            return result.samples["v"].values.to_decimal(u.mV).sum()

        with self.assertRaisesRegex(ValueError, "integer multiple of every recording period"):
            brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()

    def test_differentiable_recording_rejects_nonzero_start(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        cell.soma.record(
            "v",
            braincell.observe.state("v"),
            period=0.02 * u.ms,
            start=0.01 * u.ms,
        )
        cell.init_state()

        def loss():
            cell.reset_state()
            result = cell.run(dt=0.01 * u.ms, duration=0.04 * u.ms)
            return result.samples["v"].values.to_decimal(u.mV).sum()

        with self.assertRaisesRegex(ValueError, "requires start=0"):
            brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()

    def test_full_reset_keeps_roots_and_rebuilds_runtime_binding(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="theta"))
        root = cell.trainables.parameters().states()["theta"]
        cell.init_state()
        cell.reset()
        cell.init_state()
        self.assertIs(cell.trainables.parameters().states()["theta"], root)
        self.assertTrue(u.math.allclose(cell.channels["leak"].g_max, 0.1 * u.mS / u.cm**2))

    def test_sodium_and_potassium_share_one_factor(self) -> None:
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Channel("Na_HH1952", name="na"),
            braincell.mech.Channel("K_HH1952", name="k"),
        )
        factor = brainstate.nn.Param(1.0)
        cell.channels["na"].trainable(g_max=braincell.trainable.scale(factor, name="shared.factor"))
        cell.channels["k"].trainable(g_max=braincell.trainable.scale(factor, name="shared.factor"))
        cell.init_state()
        states = brainstate.graph.states(cell, brainstate.ParamState)
        self.assertEqual(len(states), 1)
        self.assertTrue(u.math.allclose(cell.channels["na"].g_max, 120.0 * u.mS / u.cm**2))
        self.assertTrue(u.math.allclose(cell.channels["k"].g_max, 10.0 * u.mS / u.cm**2))

    def test_parameterized_scalar_coefficients_are_two_roots(self) -> None:
        cell = _leak_cell(pop_size=(1,))
        a = brainstate.nn.Param(0.01 * u.mS / u.cm**2)
        b = brainstate.nn.Param(0.1 * u.mS / u.cm**2)

        def profile(ctx, a, b):
            return ctx.cv_id * a + b

        cell.channels["leak"].trainable(g_max=braincell.trainable.parameterized(profile, a=a, b=b))
        cell.init_state()
        self.assertEqual(len(cell.trainables.parameters().states()), 2)
        self.assertTrue(
            u.math.allclose(
                cell.channels["leak"].g_max,
                u.math.asarray([0.1, 0.11]) * u.mS / u.cm**2,
            )
        )

    def test_legacy_channel_is_rejected_without_affecting_its_runtime(self) -> None:
        cell = braincell.Cell(_build_tree())
        cell.paint(AllRegion(), braincell.mech.Channel("K_Leak", name="legacy"))
        with self.assertRaises(NotImplementedError):
            cell.channels["legacy"].trainable(g_max=braincell.trainable.parameter())
        self.assertEqual(cell.trainables.bindings(), ())
