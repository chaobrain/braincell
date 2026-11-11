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

"""Declaration, discretization, parameter ownership and deinitialization."""

import re
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import numpy as np

import braincell as bc
from braincell.filter import AllRegion, BranchSlice, RootLocation, at
from braincell._multi_compartment import cell as cell_module


def make_cell(policy=None, pop_size=1):
    branch = bc.Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 2.0] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=policy or bc.CVPerBranch(2), pop_size=pop_size)
    cell.paint(AllRegion(), bc.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65 * u.mV))
    return cell


def make_clamp():
    return bc.CurrentClamp(delay=0.0 * u.ms, durations=[1.0] * u.ms, amplitudes=[0.1] * u.nA)


def make_clamped_cell(policy=None):
    cell = make_cell(policy)
    cell.place(RootLocation(0.5), make_clamp())
    return cell


def final_voltage(cell):
    return np.asarray(cell.V.value.to_decimal(u.mV)).copy()


#: A ``Network.<method>()`` a lifecycle message tells the user to call.
_NETWORK_METHOD = re.compile(r"Network\.(\w+)\(\)")


class LifecycleTest(unittest.TestCase):
    def test_reset_state_cannot_change_dynamic_state_shape(self):
        cell = make_cell()
        cell.init_state(batch_size=2)
        original_shape = cell.V.value.shape
        cell.reset_state(batch_size=2)
        for size in (None, 3):
            with self.assertRaisesRegex(ValueError, "preserve"):
                cell.reset_state(batch_size=size)
            self.assertEqual(cell.V.value.shape, original_shape)

    def test_declaration_changes_refresh_discretize_and_init_consumes_cache(self):
        with patch.object(cell_module, "build_discretization", wraps=cell_module.build_discretization) as build:
            cell = make_cell()
            self.assertEqual(build.call_count, 2)
            _ = cell.cvs
            self.assertEqual(build.call_count, 2)
            _ = cell.cv_tree
            self.assertEqual(build.call_count, 2)
            cell.init_state()
            self.assertEqual(build.call_count, 2)
            cell.reset_state()
            _ = cell.cvs
            self.assertEqual(build.call_count, 2)

    def test_final_policy_uses_latest_cable_declaration(self):
        for policy in (bc.DLambda(0.05), bc.MaxCVLen(10 * u.um), bc.CVPerBranch(2)):
            with self.subTest(policy=policy):
                cell = make_cell(policy)
                before = cell.n_cv
                cell.paint(
                    AllRegion(),
                    bc.mech.CableProperty(
                        resting_potential=-65 * u.mV,
                        axial_resistivity=1000 * u.ohm * u.cm,
                        membrane_capacitance=10 * u.uF / u.cm**2,
                    ),
                )
                expected = sum(
                    len(bounds) for bounds in policy.resolve_cv_bounds(cell.morpho, paint_rules=cell.paint_rules)
                )
                cell.init_state()
                self.assertEqual(cell.n_cv, expected)
                if isinstance(policy, bc.DLambda):
                    self.assertGreater(expected, before)
                else:
                    self.assertEqual(expected, before)

    def test_continuous_and_cv_selection_address_same_runtime_rows(self):
        cell = make_cell()
        cell.init_state()
        layout = cell.runtime.layouts
        cell.loc(at(0, 0.25)).channels["leak"].set(g_max=0.2 * u.mS / u.cm**2)
        np.testing.assert_allclose(cell.cv[0].channels["leak"].g_max.to_decimal(u.mS / u.cm**2), [0.2])
        cell.cv[0].set(V_init=-50 * u.mV)
        np.testing.assert_allclose(cell.loc(at(0, 0.25)).V_init.to_decimal(u.mV), [-50])
        self.assertIs(cell.runtime.layouts, layout)
        with self.assertRaises(RuntimeError):
            cell.paint(AllRegion(), bc.mech.Channel("IL", name="extra"))

    def test_connection_and_detector_restore_declarations_after_reset(self):
        cell = make_cell()
        cell.place(at(0, 0.5), bc.mech.Synapse("ExpSyn", name="syn"))
        detector = bc.VoltageCrossingSource(cell, threshold=-30 * u.mV)
        bc.connect("input", source=detector, synapse=cell.synapses["syn"], weight=0.01 * u.uS)
        cell.init_state()
        cell.connections["input"].set(weight=0.02 * u.uS)
        detector.trainable(threshold=bc.trainable.parameter(-20 * u.mV, name="threshold"))
        params = cell.trainables.parameters()
        cell.reset_state()
        np.testing.assert_allclose(detector.threshold.to_decimal(u.mV), -20)
        cell.reset()
        cell.init_state()
        np.testing.assert_allclose(cell.connections["input"].weight.to_decimal(u.uS), 0.01)
        np.testing.assert_allclose(detector.threshold.to_decimal(u.mV), -30)
        detector.trainable(threshold=bc.trainable.parameter(name="new_threshold"))
        self.assertEqual(params.states(), {})

    def test_detector_remaps_continuous_location_after_policy_change(self):
        cell = make_cell()
        detector = bc.VoltageCrossingSource(cell, location=at(0, 0.8))
        old_selector = cell.cv
        cell.cv_policy = bc.CVPerBranch(4)
        cell.init_state()
        np.testing.assert_array_equal(detector.cv_id, [3])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            _ = old_selector[0]

    def test_discretization_refresh_and_atomic_policy_assignment(self):
        cell = make_cell()
        first = cell.cvs
        with patch.object(cell_module, "build_discretization", wraps=cell_module.build_discretization) as build:
            self.assertIs(cell.cvs, first)
            self.assertEqual(build.call_count, 0)
            cell.cv_policy = bc.CVPerBranch(4)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(cell.n_cv, 4)
            preview = cell.cvs
            with self.assertRaises(ValueError):
                cell.cv_policy = bc.CVPerBranch(0)
            self.assertEqual(cell.cv_policy, bc.CVPerBranch(4))
            self.assertEqual(cell.n_cv, 4)
            self.assertIs(cell.cvs, preview)
            self.assertEqual(build.call_count, 2)
        cell.init_state()
        with self.assertRaises(RuntimeError):
            cell.discretize()
        with self.assertRaises(RuntimeError):
            cell.cv_policy = bc.CVPerBranch(1)

    def test_paint_declaration_refreshes_grid_before_init(self):
        cell = make_cell()
        first = cell.cvs
        with patch.object(cell_module, "build_discretization", wraps=cell_module.build_discretization) as build:
            cell.paint(
                AllRegion(),
                bc.mech.CableProperty(
                    resting_potential=-65 * u.mV,
                    axial_resistivity=1200 * u.ohm * u.cm,
                    membrane_capacitance=10 * u.uF / u.cm**2,
                ),
            )
            self.assertEqual(build.call_count, 1)
            self.assertIsNot(cell.cvs, first)
            cell.init_state()
            self.assertEqual(build.call_count, 1)

    def test_morphology_replacement_refreshes_grid_before_init(self):
        cell = make_cell(policy=bc.CVPerBranch(2))
        with patch.object(cell_module, "build_discretization", wraps=cell_module.build_discretization) as build:
            branch = bc.Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 2.0] * u.um)
            cell.morphology = bc.Morphology.from_root(branch)
            self.assertEqual(build.call_count, 1)
            cell.init_state()
            self.assertEqual(build.call_count, 1)
            with self.assertRaises(RuntimeError):
                cell.morphology = bc.Morphology.from_root(branch)

    def test_views_are_readonly_before_init_and_expire_on_rebuild(self):
        cell = make_cell()
        view = cell.channels["leak"]
        self.assertEqual(len(view.g_max), 2)
        with self.assertRaisesRegex(RuntimeError, "init_state"):
            view.set(g_max=0.2 * u.mS / u.cm**2)
        with self.assertRaisesRegex(RuntimeError, "init_state"):
            view.trainable(g_max=bc.trainable.parameter())
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "stale|expired"):
            view.get("g_max")
        view = cell.channels["leak"]
        view.trainable(g_max=bc.trainable.parameter(group_by="cv", name="g"))
        np.testing.assert_allclose(
            cell.trainables.parameters().physical_values()["g"].to_decimal(u.mS / u.cm**2), [0.1, 0.1]
        )

    def test_reset_state_preserves_parameters_and_reset_restores_declaration(self):
        cell = make_cell()
        cell.init_state()
        cell.on(BranchSlice(0, 0, 0.5)).channels["leak"].set(g_max=0.2 * u.mS / u.cm**2)
        np.testing.assert_allclose(cell.channels["leak"].g_max.to_decimal(u.mS / u.cm**2), [0.2, 0.1])
        cell.channels["leak"].trainable(g_max=bc.trainable.scale(name="scale"))
        params = cell.trainables.parameters()
        params.set_physical_values({"scale": 2.0})
        cell.reset_state()
        np.testing.assert_allclose(cell.channels["leak"].g_max.to_decimal(u.mS / u.cm**2), [0.4, 0.2])
        old = cell.channels["leak"]
        cell.reset()
        self.assertEqual(cell.trainables.parameters().states(), {})
        with self.assertRaisesRegex(RuntimeError, "stale|expired"):
            old.get("g_max")
        cell.init_state()
        np.testing.assert_allclose(cell.channels["leak"].g_max.to_decimal(u.mS / u.cm**2), [0.1, 0.1])

    def test_runtime_voltage_initial_overrides_have_cv_scope(self):
        cell = make_cell(pop_size=2)
        cell.V_init = -65 * u.mV
        cell.init_state()
        cell[1].cv[0].set(V_init=-50 * u.mV, V_th=-10 * u.mV)
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-65, -65], [-65, -65]])
        cell.reset_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-65, -65], [-50, -65]])
        cell.reset()
        cell.init_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-65, -65], [-65, -65]])

    def test_runtime_voltage_override_after_compiled_reset(self):
        cell = make_cell()
        cell.init_state()
        reset = brainstate.transform.jit(cell.reset_state)
        threshold = brainstate.transform.jit(lambda: cell.V_th)
        threshold()
        reset()
        cell.cv[0].set(V_init=-50 * u.mV, V_th=-10 * u.mV)
        reset()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), [[-50, -65]])
        np.testing.assert_allclose(cell.cv[0].V_init.to_decimal(u.mV), [-50])
        np.testing.assert_allclose(threshold().to_decimal(u.mV)[0, 0], -10)

    def test_runtime_inspection_views_expire_after_deinit(self):
        cell = make_cell()
        cell.init_state()
        views = (cell.runtime_cvs[0], cell.runtime_nodes[0], cell[0].get_runtime_node(0))
        cell.reset()
        for view in views:
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = view.id if hasattr(type(view), "__dataclass_fields__") else view.g_max

    def test_bad_registration_is_atomic(self):
        cell = make_cell()
        cell.init_state()
        before = cell.channels["leak"].g_max
        with self.assertRaises((TypeError, ValueError)):
            cell.channels["leak"].trainable(
                g_max=bc.trainable.parameter(0.3 * u.mS / u.cm**2),
                E=bc.trainable.parameter(1 * u.ms),
            )
        self.assertFalse(cell.trainables.bindings())
        self.assertFalse(cell.trainables.roots)
        np.testing.assert_array_equal(
            cell.channels["leak"].g_max.to_decimal(u.mS / u.cm**2), before.to_decimal(u.mS / u.cm**2)
        )

    def test_repeated_reset_and_init_compile_the_current_declarations(self):
        # Each reset() + init_state() builds and compiles the runtime again from
        # the declarations; nothing compiled for an earlier grid is reused.
        cell = make_clamped_cell()

        def cycle(policy):
            cell.cv_policy = policy
            cell.init_state()
            cell.run(dt=0.025 * u.ms, duration=0.5 * u.ms)
            final = final_voltage(cell)
            cell.reset()
            return final

        first = cycle(bc.CVPerBranch(2))
        self.assertEqual(first.shape, (1, 2))
        self.assertFalse(np.allclose(first, -65.0))
        self.assertEqual(cycle(bc.CVPerBranch(4)).shape, (1, 4))
        for _ in range(2):
            np.testing.assert_array_equal(cycle(bc.CVPerBranch(2)), first)


class NetworkOwnedLifecycleTest(unittest.TestCase):
    """A Cell that a Network runs cannot reset(): its messages say what works instead."""

    def assert_names_existing_network_methods(self, message):
        named = _NETWORK_METHOD.findall(message)
        for name in named:
            self.assertTrue(callable(getattr(bc.Network, name, None)), f"{message!r} names a missing Network.{name}()")
        return named

    def run_network(self, net):
        net.run(dt=0.025 * u.ms, duration=0.5 * u.ms)

    def test_lifecycle_messages_name_only_network_methods_that_exist(self):
        self.assertFalse(hasattr(bc.Network, "reset"))
        cell = make_cell()
        net = bc.Network("n")
        net.add_population("p", cell)
        for initialized in (False, True):
            if initialized:
                net.init_state()
            for call, suggested in (
                (cell.init_state, ["init_state"]),
                (cell.reset_state, ["reset_state"]),
                (cell.reset, ["reset_state"]),
            ):
                with self.subTest(call=call.__name__, initialized=initialized):
                    with self.assertRaisesRegex(RuntimeError, "Cell belongs to Network 'n'") as caught:
                        call()
                    self.assertEqual(self.assert_names_existing_network_methods(str(caught.exception)), suggested)

    def test_the_recoveries_the_reset_message_names_work(self):
        cell = make_clamped_cell()
        net = bc.Network("n")
        net.add_population("p", cell)
        net.init_state()
        initial = final_voltage(cell)
        self.run_network(net)
        first = final_voltage(cell)
        with self.assertRaisesRegex(
            RuntimeError, r"no reset\(\).*Network\.reset_state\(\).*declare a new Cell and a new Network"
        ):
            cell.reset()
        self.assertTrue(cell._initialized)

        # Network.reset_state() starts the run again from the initial state.
        net.reset_state()
        self.assertEqual(cell.current_time, 0.0 * u.ms)
        np.testing.assert_array_equal(final_voltage(cell), initial)
        self.run_network(net)
        np.testing.assert_array_equal(final_voltage(cell), first)

        # A new Cell in a new Network takes changed declarations.
        new_cell = make_clamped_cell(bc.CVPerBranch(4))
        new_net = bc.Network("n")
        new_net.add_population("p", new_cell)
        new_net.init_state()
        self.run_network(new_net)
        self.assertEqual(final_voltage(new_cell).shape, (1, 4))

    def test_an_initialized_cell_in_a_network_is_not_told_to_reset(self):
        # A Cell initialized before it joined the Network (or one that a failed
        # Network.init_state left initialized) cannot reset() either.
        cell = make_cell()
        cell.init_state()
        net = bc.Network("n")
        net.add_population("p", cell)
        cable = bc.mech.CableProperty(
            resting_potential=-65 * u.mV,
            axial_resistivity=1200 * u.ohm * u.cm,
            membrane_capacitance=10 * u.uF / u.cm**2,
        )
        for action, call in (
            ("paint()", lambda: cell.paint(AllRegion(), cable)),
            ("place()", lambda: cell.place(RootLocation(0.5), make_clamp())),
            ("discretize()", cell.discretize),
            ("assign cv_policy", lambda: setattr(cell, "cv_policy", bc.CVPerBranch(4))),
        ):
            with self.subTest(action):
                with self.assertRaises(RuntimeError) as caught:
                    call()
                message = str(caught.exception)
                self.assertIn(f"Cannot {action} after init_state()", message)
                self.assertIn("belongs to Network 'n'", message)
                self.assertIn("declare a new Cell and a new Network", message)
                self.assertNotIn("call reset()", message)
                self.assert_names_existing_network_methods(message)

        # The Network runs the Cell as it was initialized; a new Cell in a new
        # Network takes the change.
        net.init_state()
        self.run_network(net)
        self.assertEqual(final_voltage(cell).shape, (1, 2))
        with self.assertRaisesRegex(RuntimeError, "owning Network 'n' has been initialized"):
            cell.paint(AllRegion(), cable)
        new_cell = make_cell()
        new_cell.paint(AllRegion(), cable)
        new_net = bc.Network("n")
        new_net.add_population("p", new_cell)
        new_net.init_state()
        self.run_network(new_net)


if __name__ == "__main__":
    unittest.main()
