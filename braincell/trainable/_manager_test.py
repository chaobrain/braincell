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
import jax
import numpy as np

import braincell
from braincell._compute._testing import _build_tree
from braincell.filter import AllRegion, BranchSlice


@braincell.mech.register_channel("_SignatureRequiredLeak")
class _SignatureRequiredLeak(braincell.channel.IL):
    def __init__(self, size, g_max, E=None, name=None):
        super().__init__(size, g_max=g_max, E=-70.0 * u.mV if E is None else E, name=name)


def _leak_cell(*, pop_size=(2,)):
    cell = braincell.Cell(_build_tree(), pop_size=pop_size)
    cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak"))
    return cell


class TrainableManagerTest(unittest.TestCase):
    def test_missing_and_none_defaults_can_be_supplied_before_initialization(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(AllRegion(), braincell.mech.Channel("_SignatureRequiredLeak", name="leak"))
        with self.assertRaises(KeyError):
            cell.channels["leak"].get("g_max")
        cell.channels["leak"].set(g_max=0.2 * u.mS / u.cm**2)
        cell.channels["leak"].trainable(E=braincell.trainable.parameter(-60.0 * u.mV, group_by="all", name="E"))
        cell.init_state()
        self.assertTrue(u.math.allclose(cell.channels["leak"].g_max, 0.2 * u.mS / u.cm**2))
        self.assertTrue(u.math.allclose(cell.channels["leak"].E, -60.0 * u.mV))

    def test_floating_override_does_not_truncate_integer_default(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", Ci=0.01 * u.mM),
            braincell.mech.Channel("AHP_De1994", name="ahp"),
        )
        cell.channels["ahp"].set(n=2.5)
        cell.init_state()
        self.assertTrue(u.math.allclose(cell.channels["ahp"].n, 2.5))
        cell.channels["ahp"].set(n=2.75)
        self.assertTrue(u.math.allclose(cell.channels["ahp"].n, 2.75))

    def test_missing_numeric_default_requires_values_on_all_active_rows(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(AllRegion(), braincell.mech.Channel("_SignatureRequiredLeak", name="leak"))
        cell.soma.channels["leak"].set(g_max=0.2 * u.mS / u.cm**2)
        with self.assertRaisesRegex(ValueError, "every active row"):
            cell.init_state()

    def test_required_parameter_accepts_an_explicit_trainable_initial(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(AllRegion(), braincell.mech.Channel("_SignatureRequiredLeak", name="leak"))
        cell.channels["leak"].trainable(
            g_max=braincell.trainable.parameter(0.2 * u.mS / u.cm**2, group_by="all", name="g"),
        )
        cell.init_state()
        self.assertTrue(u.math.allclose(cell.channels["leak"].g_max, 0.2 * u.mS / u.cm**2))

    def test_static_boolean_keeps_constructor_semantics(self):
        for class_name, freeze, expected in (
            ("Ca_ZH2019_IO", False, False),
            ("Ca_ZH2019_IO", True, True),
            ("Ca_ZH2019_IO_Frozen", False, True),
        ):
            with self.subTest(channel=class_name, freeze=freeze):
                cell = braincell.Cell(_build_tree())
                cell.paint(AllRegion(), braincell.mech.Channel(class_name, name="ca", freeze_m_inf=freeze))
                cell.init_state()
                node = cell.runtime.get_runtime_node(0)
                self.assertIs(node.freeze_m_inf, expected)
                self.assertTrue(bool(u.math.all(cell.channels["ca"].get("freeze_m_inf") == expected)))

    def test_frozen_gate_has_zero_gradient_but_unfrozen_gate_does_not(self):
        for freeze in (False, True):
            with self.subTest(freeze=freeze):
                cell = braincell.Cell(_build_tree())
                cell.paint(AllRegion(), braincell.mech.Channel("Ca_ZH2019_IO", name="ca", freeze_m_inf=freeze))
                cell.channels["ca"].trainable(mMidV=braincell.trainable.parameter(group_by="all", name="midpoint"))
                cell.init_state()
                node = cell.runtime.get_runtime_node(0)

                def loss():
                    cell.trainables.materialize()
                    return node.current(-50.0 * u.mV).to_decimal(u.mA / u.cm**2).sum()

                gradient = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()
                value = float(u.get_mantissa(gradient["midpoint"]))
                self.assertEqual(value == 0.0, freeze)

    def test_new_signature_parameter_has_independent_regional_gradients(self):
        self._check_regional_gradients(split_layouts=False)
        self._check_regional_gradients(split_layouts=True)

    def _check_regional_gradients(self, *, split_layouts):
        cell = braincell.Cell(_build_tree(), pop_size=(2,))
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        )
        if split_layouts:
            cell.paint(
                BranchSlice(0, 0.0, 1.0), braincell.mech.Channel("Na_TM1991", name="na", g_max=100.0 * u.mS / u.cm**2)
            )
            cell.paint(
                BranchSlice(1, 0.0, 1.0), braincell.mech.Channel("Na_TM1991", name="na", g_max=120.0 * u.mS / u.cm**2)
            )
        else:
            cell.paint(AllRegion(), braincell.mech.Channel("Na_TM1991", name="na"))
        original = cell[1].channels["na"].V_sh
        cell[0].soma.channels["na"].trainable(V_sh=braincell.trainable.parameter(group_by="all", name="soma"))
        cell[0].branch[1].channels["na"].trainable(V_sh=braincell.trainable.parameter(group_by="all", name="dendrite"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(1)
        if split_layouts:
            self.assertIs(node, cell.runtime.get_runtime_node(2))

        def loss():
            cell.trainables.materialize()
            return node.f_p_alpha(-30.0 * u.mV, None)[0, 0]

        gradients = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()
        self.assertNotEqual(float(u.get_mantissa(gradients["soma"])), 0.0)
        self.assertEqual(float(u.get_mantissa(gradients["dendrite"])), 0.0)
        values = cell.trainables.parameters().physical_values()
        cell.trainables.parameters().set_physical_values(
            {
                "soma": values["soma"] + 1.0 * u.mV,
                "dendrite": values["dendrite"] - 2.0 * u.mV,
            }
        )
        cell.trainables.materialize()
        self.assertTrue(u.math.allclose(cell[1].channels["na"].V_sh, original))
        self.assertTrue(u.math.allclose(cell[0].soma.channels["na"].V_sh, values["soma"] + 1.0 * u.mV))
        self.assertTrue(u.math.allclose(cell[0].branch[1].channels["na"].V_sh, values["dendrite"] - 2.0 * u.mV))

    def test_gate_current_switch_has_zero_gradient_at_both_values(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Channel("Kv1p1_MA2025_BC", name="k"),
        )
        cell.channels["k"].trainable(gateCurrent=braincell.trainable.parameter(group_by="all", name="switch"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(1)
        ion = braincell.IonInfo(E=-77.0 * u.mV, Ci=140.0 * u.mM, Co=5.0 * u.mM, valence=1)

        def loss():
            cell.trainables.materialize()
            return node.current(-30.0 * u.mV, ion).to_decimal(u.mA / u.cm**2).sum()

        gradient = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())
        self.assertEqual(float(gradient()["switch"]), 0.0)
        disabled = brainstate.transform.jit(loss)()
        cell.trainables.parameters().set_physical_values({"switch": 1.0})
        self.assertEqual(float(gradient()["switch"]), 0.0)
        enabled = brainstate.transform.jit(loss)()
        self.assertNotEqual(float(disabled), float(enabled))

    def test_q10_gradient_is_zero_only_at_zero_temperature_offset(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
            braincell.mech.Channel("Na_TM1991", name="na"),
        )
        view = cell.channels["na"]
        view.set(temp=view.parameter_info()["temp_ref"].default)
        view.trainable(q10=braincell.trainable.parameter(group_by="all", name="q10"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(1)

        def loss():
            cell.trainables.materialize()
            return node.gate_phi(node._iter_gates()[0]).sum()

        gradient = brainstate.transform.jit(
            brainstate.transform.grad(
                loss,
                grad_states=cell.trainables.parameters().states(),
            )
        )
        self.assertEqual(float(gradient()["q10"]), 0.0)
        view.set(temp=view.parameter_info()["temp_ref"].default + 10.0 * u.kelvin)
        self.assertGreater(float(gradient()["q10"]), 0.0)

    def test_integer_default_exponent_and_independent_phi_are_learnable(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", Ci=0.01 * u.mM),
            braincell.mech.Channel("AHP_De1994", name="ahp"),
        )
        cell.channels["ahp"].trainable(
            n=braincell.trainable.parameter(group_by="all", name="exponent"),
            phi=braincell.trainable.parameter(group_by="all", name="phi"),
        )
        cell.init_state()
        node = cell.runtime.get_runtime_node(2)
        calcium = braincell.IonInfo(Ci=0.01 * u.mM, Co=2.0 * u.mM, E=120.0 * u.mV, valence=2)

        def loss():
            cell.trainables.materialize()
            return (node.phi * node.f_p_alpha(-30.0 * u.mV, None, calcium)).sum()

        gradients = brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()
        self.assertLess(float(gradients["exponent"]), 0.0)
        self.assertGreater(float(gradients["phi"]), 0.0)

    def test_invalid_sources_use_existing_validation_and_rollback(self):
        cell = _leak_cell(pop_size=(1,))
        for field, initial, errors in (
            ("name", "not-a-number", (TypeError, ValueError)),
            ("g_max", 1.0, (TypeError,)),
            ("E", 1.0 * u.kelvin, (ValueError,)),
            ("g_max", np.ones((3, 4)) * u.mS / u.cm**2, (ValueError,)),
            ("not_in_signature", 1.0, (KeyError,)),
        ):
            with self.subTest(field=field), self.assertRaises(errors):
                cell.channels["leak"].trainable(
                    **{
                        field: braincell.trainable.parameter(initial, group_by="all"),
                    }
                )
            self.assertEqual(cell.trainables.bindings(), ())
            self.assertEqual(cell.trainables.parameters().states(), {})

    def test_training_python_boolean_control_flow_errors_naturally(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(AllRegion(), braincell.mech.Channel("Ca_ZH2019_IO", name="ca"))
        cell.channels["ca"].trainable(
            freeze_m_inf=braincell.trainable.parameter(group_by="all", name="freeze"),
        )
        cell.init_state()
        node = cell.runtime.get_runtime_node(0)

        def loss():
            cell.trainables.materialize()
            return node.current(-50.0 * u.mV).to_decimal(u.mA / u.cm**2).sum()

        with self.assertRaises((ValueError, jax.errors.TracerBoolConversionError)):
            brainstate.transform.grad(loss, grad_states=cell.trainables.parameters().states())()

    def test_structural_signature_parameters_are_not_blacklisted(self):
        for field in ("size", "solver", "substeps"):
            with self.subTest(field=field):
                cell = braincell.Cell(_build_tree())
                cell.paint(
                    AllRegion(),
                    braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
                    braincell.mech.Channel("Nav1p6_MA2020_GoC", name="na"),
                )
                cell.channels["na"].trainable(
                    **{
                        field: braincell.trainable.parameter(2.0, group_by="all", name="configuration"),
                    }
                )
                self.assertEqual(len(cell.trainables.bindings()), 1)
                with self.assertRaises((TypeError, ValueError)):
                    cell.init_state()

    def test_derived_phi_updates_through_runtime_temperature_binding(self):
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
            braincell.mech.Channel("Nav1p6_MA2020_GoC", name="na"),
        )
        cell.channels["na"].trainable(temp=braincell.trainable.parameter(group_by="all", name="temp"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(1)

        def loss():
            cell.trainables.materialize()
            return node.f01(-30.0 * u.mV).sum()

        evaluate = brainstate.transform.jit(loss)
        before = evaluate()
        parameters = cell.trainables.parameters()
        parameters.set_physical_values({"temp": u.celsius2kelvin(32.0)})
        after = evaluate()
        self.assertTrue(u.math.allclose(after, 3.0 * before))
        gradient = brainstate.transform.grad(loss, grad_states=parameters.states())()["temp"]
        self.assertTrue(u.math.allclose(u.get_mantissa(gradient), after * np.log(3.0) / 10.0))

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

    def test_signature_default_can_be_set_and_trained_without_a_dictionary(self) -> None:
        cell = braincell.Cell(_build_tree())
        cell.paint(
            AllRegion(),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Channel("K_Leak", name="legacy"),
        )
        view = cell.channels["legacy"]
        self.assertIn("g_max", view.parameter_info())
        view.set(g_max=0.2 * u.mS / u.cm**2)
        view.trainable(g_max=braincell.trainable.scale(name="factor"))
        cell.init_state()
        cell.trainables.parameters().set_physical_values({"factor": 2.0})
        cell.trainables.materialize()
        self.assertTrue(u.math.allclose(view.g_max, 0.4 * u.mS / u.cm**2))
