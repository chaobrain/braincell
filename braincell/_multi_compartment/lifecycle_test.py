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

import contextlib
import logging
import re
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
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

    def test_runtime_access_before_init_state_names_what_needs_it(self):
        cell = make_cell(pop_size=2)
        cell.paint(AllRegion(), bc.mech.Ion("PotassiumFixed", name="k"))
        cell.place(at(0, 0.5), bc.mech.Synapse("ExpSyn", name="syn"))
        threshold = bc.trainable.parameter(-20 * u.mV, name="threshold")
        accesses = {
            "geometry": (lambda: cell.geometry, lambda: cell[0].geometry),
            "length": (lambda: cell.length,),
            "radius_scale": (lambda: cell.radius_scale,),
            "Ra": (lambda: cell[1].Ra,),
            "cm": (lambda: cell.cm,),
            "radius_mid": (lambda: cell.radius_mid,),
            "diam_arc_mean": (lambda: cell.diam_arc_mean,),
            "CV geometry": (lambda: cell.cvs.length, lambda: cell.cvs[0:1].area),
            "CellView.set()": (lambda: cell[0].set(V_init=-60 * u.mV),),
            "ChannelView.set()": (lambda: cell.channels["leak"].set(g_max=0.2 * u.mS / u.cm**2),),
            "IonView.set()": (lambda: cell.ions["k"].set(E=-80 * u.mV),),
            "SynapseView.set()": (lambda: cell.synapses["syn"].set(tau=3 * u.ms),),
            "trainable()": (
                lambda: cell.channels["leak"].trainable(g_max=bc.trainable.parameter()),
                lambda: cell.event_outputs["spike"].trainable(threshold=threshold),
            ),
        }
        paint = " Before init_state(), give the values in the paint() declaration."
        hints = {
            "CellView.set()": " Before init_state(), assign Cell.V_init and Cell.V_th instead.",
            "ChannelView.set()": paint,
            "IonView.set()": paint,
            "SynapseView.set()": " Before init_state(), give the values in the place() declaration.",
        }
        for name, reads in accesses.items():
            for index, read in enumerate(reads):
                with self.subTest(name=name, index=index):
                    with self.assertRaises(RuntimeError) as caught:
                        read()
                    self.assertEqual(str(caught.exception), f"{name} requires init_state() first.{hints.get(name, '')}")

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
        # A Cell initialized before it joined the Network cannot reset() either.
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


DT = 0.025 * u.ms


@contextlib.contextmanager
def loop_compiles():
    """The names of the XLA compiles in the block, from ``jax_log_compiles`` (not printed).

    ``brainstate.transform.jit`` compiles a loop as ``jit(jaxpr_call)``.
    """
    names = []

    class Collect(logging.Handler):
        def emit(self, record):
            match = re.match(r"Finished XLA compilation of (\S+)", record.getMessage())
            if match:
                names.append(match.group(1))

    logger = logging.getLogger("jax")
    handler, propagate = Collect(), logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with jax.log_compiles(True):
            yield names
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate


def make_hh_cell(kind="cable", g_na=120.0):
    """Two members with HH channels and a 0.1 nA clamp from 0.5 to 4.5 ms; V_init -65 mV.

    ``"cable"``: one 60 um x 5 um soma in 3 CVs; ``"lumped"``:
    ``Cell.point`` 20 um x 10 um. Parameters given as Python floats make
    weak-typed initial arrays (V, spike, the gates).
    """
    if kind == "lumped":
        cell = bc.Cell.point(2, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
    else:
        branch = bc.Branch.from_lengths(lengths=[60.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        cell = bc.Cell(bc.Morphology.from_root(branch), pop_size=2, cv_policy=bc.CVPerBranch(3), V_init=-65.0 * u.mV)
    cell.paint(
        AllRegion(),
        bc.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        bc.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        bc.mech.Channel("Na_HH1952", name="na", g_max=g_na * u.mS / u.cm**2),
        bc.mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        bc.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(RootLocation(0.5), bc.CurrentClamp(delay=0.5 * u.ms, durations=[4.0] * u.ms, amplitudes=[0.1] * u.nA))
    return cell


def weak_short_term_states(cell):
    """Paths of the short-term States of ``cell`` that hold a weak-typed array."""
    names = []
    for path, state in brainstate.graph.states(cell, brainstate.ShortTermState).items():
        leaves = jax.tree.leaves(state.value)
        if any(isinstance(leaf, jax.Array) and leaf.aval.weak_type for leaf in leaves):
            names.append(".".join(map(str, path)))
    return names


def state_shapes(cell):
    """``{path: shape}`` of every State value of ``cell``."""
    return {
        ".".join(map(str, path)): tuple(np.shape(leaf) for leaf in jax.tree.leaves(state.value))
        for path, state in brainstate.graph.states(cell).items()
    }


def strong_by_hand(cell):
    """Replace each weak-typed array of the short-term States of ``cell`` by a strong one, through NumPy."""

    def strong(leaf):
        return jnp.asarray(np.asarray(leaf)) if isinstance(leaf, jax.Array) and leaf.aval.weak_type else leaf

    for state in brainstate.graph.states(cell, brainstate.ShortTermState).values():
        state.value = jax.tree.map(strong, state.value)


def no_strong_state_values():
    """The init_state() and reset_state() of before CELL-54: States are left as built."""
    return patch.object(cell_module, "strong_state_values", lambda module: None, create=True)


class SecondRunCompileTest(unittest.TestCase):
    """CELL-54: init_state() and reset_state() leave strong-typed States, so a second run reuses the loop."""

    def test_a_second_cell_run_does_not_compile_the_loop_again(self):
        # On main 29d6828 V, spike and the gates were weak-typed after
        # init_state() and reset_state(), and strong after a run, which jit
        # keys on: the loop compiled at runs 1 and 2, [1, 1, 0, 0, 0].
        for kind in ("cable", "lumped"):
            with self.subTest(kind=kind):
                cell = make_hh_cell(kind)
                cell.init_state()
                counts = []
                for index in range(5):
                    if index == 3:
                        cell.reset_state()
                    with loop_compiles() as names:
                        cell.run(dt=DT, duration=40 * DT)
                    counts.append(names.count("jit(jaxpr_call)"))
                self.assertGreaterEqual(counts[0], 1)
                self.assertEqual(counts[1:], [0, 0, 0, 0])

    def test_a_jitted_update_loop_compiles_once(self):
        # A loop of the user's own over update(), compiled with
        # brainstate.transform.jit, called twice after init_state() and twice
        # after reset_state() (main 29d6828: [1, 1, 0, 0]).
        for kind in ("cable", "lumped"):
            with self.subTest(kind=kind):
                cell = make_hh_cell(kind)
                cell.init_state()

                def step(index, cell=cell):
                    with brainstate.environ.context(t=index * DT, dt=DT):
                        cell.update()
                    return cell.V.value

                loop = brainstate.transform.jit(lambda step=step: brainstate.transform.for_loop(step, np.arange(40)))
                counts, voltages = [], []
                for index in range(4):
                    if index == 2:
                        cell.reset_state()
                    with loop_compiles() as names:
                        voltages.append(np.asarray(loop().to_decimal(u.mV)))
                    counts.append(names.count("jit(jaxpr_call)"))
                self.assertGreaterEqual(counts[0], 1)
                self.assertEqual(counts[1:], [0, 0, 0])
                self.assertTrue(np.isfinite(voltages).all())
                # The clamp (from 0.5 ms) depolarizes the cell; the loop after
                # reset_state() repeats the first.
                self.assertGreater(float(np.max(voltages[0])), -64.0)
                np.testing.assert_array_equal(voltages[2], voltages[0])

    def test_init_and_reset_with_a_batch_keep_the_axis_and_leave_no_weak_state(self):
        for kind in ("cable", "lumped"):
            with self.subTest(kind=kind):
                before = make_hh_cell(kind)
                with no_strong_state_values():
                    before.init_state(batch_size=2)
                cell = make_hh_cell(kind)
                cell.init_state(batch_size=2)
                self.assertEqual(cell.V.value.shape, (2, 2, 3) if kind == "cable" else (2, 2))
                self.assertEqual(state_shapes(cell), state_shapes(before))
                self.assertIn("V", weak_short_term_states(before))
                self.assertEqual(weak_short_term_states(cell), [])
                cell.run(dt=DT, duration=4 * DT)
                cell.reset_state(batch_size=2)
                self.assertEqual(state_shapes(cell), state_shapes(before))
                self.assertEqual(weak_short_term_states(cell), [])

    def test_a_float32_parameter_at_float64_computes_the_first_run_in_float64(self):
        # On main 29d6828 the gates were weak-typed float64 after init_state()
        # and reset_state(); met by the float32 g_max, the first run after
        # each computed the Na conductance in float32, the runs after it in
        # float64: V after run 1 was off by up to 8.4e-7 mV.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            finals = {}
            for by_hand in (False, True):
                cell = make_hh_cell(g_na=np.float32(120.0))
                with no_strong_state_values() if by_hand else contextlib.nullcontext():
                    cell.init_state()
                    (g_max,) = [
                        state.value
                        for path, state in brainstate.graph.states(cell).items()
                        if path[-2:] == ("na", "g_max")
                    ]
                    self.assertEqual(np.asarray(u.get_mantissa(g_max)).dtype, np.float32)
                    finals[by_hand] = []
                    for index in range(2):
                        if index == 1:
                            cell.reset_state()
                        if by_hand:
                            strong_by_hand(cell)
                        cell.run(dt=DT, duration=200 * DT)
                        finals[by_hand].append(final_voltage(cell))
            for run, (voltage, expected) in enumerate(zip(finals[False], finals[True])):
                with self.subTest(run=run):
                    self.assertEqual(voltage.dtype, np.float64)
                    self.assertTrue(np.isfinite(voltage).all())
                    self.assertGreater(float(np.max(np.abs(voltage + 65.0))), 1.0)
                    np.testing.assert_allclose(voltage, expected, rtol=1e-12, atol=0.0)

    def test_python_float_parameters_give_the_numbers_of_before(self):
        # With parameters given as Python floats, strong-typed States change
        # only the cache key: the runs are those of before CELL-54.
        for precision in (32, 64):
            for kind in ("cable", "lumped"):
                with (
                    self.subTest(precision=precision, kind=kind),
                    jax.enable_x64(precision == 64),
                    brainstate.environ.context(precision=precision),
                ):
                    finals = {}
                    for before in (False, True):
                        cell = make_hh_cell(kind)
                        with no_strong_state_values() if before else contextlib.nullcontext():
                            cell.init_state()
                            finals[before] = []
                            for index in range(3):
                                if index == 2:
                                    cell.reset_state()
                                cell.run(dt=DT, duration=120 * DT)
                                finals[before].append(final_voltage(cell))
                    self.assertTrue(np.isfinite(finals[False]).all())
                    self.assertGreater(float(np.max(np.abs(np.asarray(finals[False]) + 65.0))), 1.0)
                    np.testing.assert_allclose(finals[False], finals[True], rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
