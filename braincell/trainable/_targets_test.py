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

"""Logical point parameter binding regressions."""

import unittest

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell.filter import at
from braincell.network._testing import make_soma_tree
from braincell.network.event import VoltageCrossingSource


def _cell():
    cell = braincell.Cell(make_soma_tree(), pop_size=(1,))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms))
    return cell


class PointTargetsTest(unittest.TestCase):
    def test_exp2_factor_uses_current_tau_and_matches_finite_difference(self):
        cell = braincell.Cell(make_soma_tree(), pop_size=(1,))
        cell.place(at("soma", 0.5), braincell.mech.Synapse("Exp2Syn", name="syn"))
        syn = cell.synapses["syn"]
        syn.trainable(tau1=braincell.trainable.scale(name="factor"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(syn._store.layout_id("Exp2Syn"))
        parameters = cell.trainables.parameters()

        def observe():
            cell.reset_state()
            node.apply_events(0.01 * u.uS)
            return node.A.value.to_decimal(u.uS).sum()

        run = brainstate.transform.jit(observe)
        derivative = brainstate.transform.jit(brainstate.transform.grad(observe, grad_states=parameters.states()))()[
            "factor"
        ]
        eps = 0.01
        parameters.set_physical_values({"factor": 1.0 + eps})
        plus = run()
        parameters.set_physical_values({"factor": 1.0 - eps})
        minus = run()
        self.assertNotEqual(float(derivative), 0.0)
        np.testing.assert_allclose(derivative, (plus - minus) / (2 * eps), rtol=0.01)

    def test_failed_materialization_does_not_partially_write(self):
        cell = _cell()
        invalid = [False]
        syn = cell.synapses["syn"]
        syn.trainable(
            tau=braincell.trainable.parameter(2.0 * u.ms, name="tau"),
            e=braincell.trainable.parameterized(lambda ctx: 1.0 * (u.ms if invalid[0] else u.mV)),
        )
        cell.init_state()
        cell.trainables.parameters().set_physical_values({"tau": 4.0 * u.ms})
        invalid[0] = True
        with self.assertRaises(u.UnitMismatchError):
            cell.trainables.materialize()
        np.testing.assert_allclose(syn.tau.to_decimal(u.ms), [2.0])

    def test_grouping_parameterized_and_unselected_rows(self):
        for group, shape in (("row", (4,)), ("population", (2,)), ("cv", ()), ("all", ())):
            with self.subTest(group=group):
                cell = braincell.Cell(make_soma_tree(), pop_size=(2,))
                for name in ("a", "b"):
                    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name=name, tau=2.0 * u.ms))
                cell.synapses.by_type("ExpSyn").trainable(tau=braincell.trainable.parameter(group_by=group, name="tau"))
                self.assertEqual(cell.trainables.parameters().physical_values()["tau"].shape, shape)
                cell.init_state()
                np.testing.assert_allclose(cell.synapses.by_type("ExpSyn").tau.to_decimal(u.ms), 2.0)
        cell = _cell()
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="other", tau=3.0 * u.ms))
        cell.synapses["syn"].trainable(
            tau=braincell.trainable.parameterized(lambda ctx, factor: factor * u.ms, factor=brainstate.nn.Param(4.0))
        )
        cell.init_state()
        np.testing.assert_allclose(cell.synapses["syn"].tau.to_decimal(u.ms), [4.0])
        np.testing.assert_allclose(cell.synapses["other"].tau.to_decimal(u.ms), [3.0])

    def test_disjoint_contact_bindings_and_shared_scale(self):
        cell = _cell()
        first = braincell.connect("a", source=braincell.NetStim(), synapse=cell.synapses["syn"], weight=0.01 * u.uS)
        second = braincell.connect("b", source=braincell.NetStim(), synapse=cell.synapses["syn"], weight=0.02 * u.uS)
        root = brainstate.nn.Param(2.0)
        for view in (first, second):
            view.trainable(weight=braincell.trainable.scale(root, name="factor"))
        self.assertEqual(len(cell.trainables.roots), 1)
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            first.set(weight=0.03 * u.uS)
        cell.init_state()
        np.testing.assert_allclose(first.weight.to_decimal(u.uS), [0.02])
        np.testing.assert_allclose(second.weight.to_decimal(u.uS), [0.04])

    def test_threshold_selection_and_alias_ownership(self):
        cell = braincell.Cell(make_soma_tree(), pop_size=(2,))
        source = VoltageCrossingSource(cell)
        source[0].trainable(threshold=braincell.trainable.parameter(-40.0 * u.mV, name="vth"))
        with self.assertRaisesRegex(ValueError, "already bound"):
            cell.event_outputs["spike"][0].trainable(threshold=braincell.trainable.parameter())
        cell[1].set(V_th=-15.0 * u.mV)
        cell.init_state()
        np.testing.assert_allclose(cell.V_th.to_decimal(u.mV), [[-40.0], [-15.0]])
        with self.assertRaises(RuntimeError):
            source[1].trainable(threshold=braincell.trainable.parameter())

    def test_detector_and_connection_invalid_fields(self):
        cell = _cell()
        source = VoltageCrossingSource(cell, threshold=-30.0 * u.mV)
        for action, error in (
            (lambda: source.trainable(direction=braincell.trainable.parameter(1.0)), KeyError),
            (lambda: source[[0, 0]].trainable(threshold=braincell.trainable.parameter()), ValueError),
            (lambda: braincell.NetStim().trainable(threshold=braincell.trainable.parameter()), TypeError),
        ):
            with self.assertRaises(error):
                action()
        connection = braincell.connect(
            "a", source=braincell.NetStim(), synapse=cell.synapses["syn"], weight=0.01 * u.uS
        )
        with self.assertRaises(KeyError):
            connection.trainable(threshold=braincell.trainable.parameter())
        with self.assertRaises((TypeError, ValueError)):
            connection.trainable(weight=braincell.trainable.parameter(1.0 * u.ms))

    def test_synapse_and_weight_survive_compiled_reset(self):
        cell = _cell()
        syn = cell.synapses["syn"]
        connection = braincell.connect("input", source=braincell.NetStim(), synapse=syn, weight=0.02 * u.uS)
        syn.trainable(tau=braincell.trainable.parameter(name="tau"))
        connection.trainable(weight=braincell.trainable.parameter(name="weight"))
        cell.init_state()
        node = cell.runtime.get_runtime_node(syn._store.layout_id("ExpSyn"))

        def observe():
            cell.reset_state()
            node.apply_events(connection.weight)
            node.compute_derivative()
            return node.g.derivative.to_decimal(u.uS / u.ms).sum()

        run = brainstate.transform.jit(observe)
        grad = brainstate.transform.jit(
            brainstate.transform.grad(observe, grad_states=cell.trainables.parameters().states())
        )
        np.testing.assert_allclose(run(), -0.01)
        self.assertNotEqual(float(u.get_mantissa(grad()["tau"])), 0.0)
        cell.trainables.parameters().set_physical_values({"tau": 4.0 * u.ms, "weight": 0.08 * u.uS})
        np.testing.assert_allclose(run(), -0.02)
        self.assertNotEqual(float(u.get_mantissa(grad()["weight"])), 0.0)

    def test_colocated_synapses_have_independent_rows(self):
        cell = _cell()
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="other", tau=3.0 * u.ms))
        cell.synapses.by_type("ExpSyn").trainable(tau=braincell.trainable.parameter(name="tau"))
        cell.init_state()
        self.assertEqual(cell.trainables.parameters().physical_values()["tau"].shape, (2,))
        cell.trainables.parameters().set_physical_values({"tau": np.array([5.0, 7.0]) * u.ms})
        cell.trainables.materialize()
        np.testing.assert_allclose(cell.synapses.by_type("ExpSyn").tau.to_decimal(u.ms), [5.0, 7.0])

    def test_detector_threshold_binding_and_gradients(self):
        cell = _cell()
        detector = VoltageCrossingSource(cell, threshold=-40.0 * u.mV)
        detector.trainable(threshold=braincell.trainable.parameter(name="threshold"))
        cell.init_state()

        def event():
            cell.reset_state()
            cell._event_previous_V.value = np.array([[-45.0]]) * u.mV
            cell.V.value = np.array([[-38.0]]) * u.mV
            return detector.current_event_count([0]).sum()

        grad = brainstate.transform.jit(
            brainstate.transform.grad(event, grad_states=cell.trainables.parameters().states())
        )
        self.assertNotEqual(float(u.get_mantissa(grad()["threshold"])), 0.0)

    def test_default_detector_binds_cell_threshold(self):
        cell = _cell()
        cell.event_outputs["spike"].trainable(threshold=braincell.trainable.parameter(-40.0 * u.mV, name="threshold"))
        cell.init_state()
        np.testing.assert_allclose(cell.V_th.to_decimal(u.mV), [[-40.0]])
        cell.trainables.parameters().set_physical_values({"threshold": -30.0 * u.mV})
        cell.reset_state()
        np.testing.assert_allclose(cell.V_th.to_decimal(u.mV), [[-30.0]])

    def test_static_delay_and_owned_set_are_rejected(self):
        cell = _cell()
        syn = cell.synapses["syn"]
        connection = braincell.connect("input", source=braincell.NetStim(), synapse=syn, weight=0.02 * u.uS)
        with self.assertRaisesRegex(NotImplementedError, "delay"):
            connection.trainable(delay=braincell.trainable.parameter())
        syn.trainable(tau=braincell.trainable.parameter())
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            syn.set(tau=3.0 * u.ms)

    def test_invalid_synapse_registration_is_atomic(self):
        cell = _cell()
        with self.assertRaisesRegex(ValueError, "> 0"):
            cell.synapses["syn"].trainable(tau=braincell.trainable.parameter(-1.0 * u.ms))
        self.assertEqual(len(cell.trainables.roots), 0)
        cell.place(at("soma", 0.5), braincell.mech.Synapse("Exp2Syn", name="double"))
        syn = cell.synapses["double"]
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            syn.trainable(tau1=braincell.trainable.parameter(20.0 * u.ms))
        self.assertEqual(len(cell.trainables.roots), 0)
        syn.trainable(tau1=braincell.trainable.parameter(20.0 * u.ms), tau2=braincell.trainable.parameter(30.0 * u.ms))

    def test_owned_cell_threshold_cannot_be_overwritten(self):
        cell = _cell()
        cell.event_outputs["spike"].trainable(threshold=braincell.trainable.parameter())
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            cell.V_th = -30.0 * u.mV
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            cell[0].set(V_th=-30.0 * u.mV)
