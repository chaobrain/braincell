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
import warnings
from types import ModuleType
from unittest.mock import patch

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell
from braincell._random_test import default_key_data, drawn_seeds
from braincell._multi_compartment.run_test import (
    DT,
    WEIGHT,
    _float64,
    assert_same_trajectory,
    make_event_cell,
    split_steps,
)
from braincell.filter import AllRegion, at
from braincell.filter import RandomSamples
from braincell.network import Network
from braincell.network.connection import independent
from braincell.network._testing import make_runtime_network, make_threshold_cell


class NetworkRuntimeTest(unittest.TestCase):
    def test_preparation_requires_cell_and_keeps_configuration_static(self):
        net = Network("source_only")
        net.add_population("input", braincell.NetStim())
        with self.assertRaisesRegex(ValueError, "at least one Cell"):
            net.prepare_run(dt=0.1 * u.ms)
        net = make_runtime_network()
        net.prepare_run(dt=0.1 * u.ms, event_backend="scatter")
        with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
            net.prepare_run(dt=0.2 * u.ms, event_backend="scatter")

    def test_cell_with_trainable_bindings_is_aggregated_without_copying(self) -> None:
        cell = make_threshold_cell()
        cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak"))
        network = Network("network")
        network.add_population("cell", cell)
        network.init_state()
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="factor"))
        self.assertIs(
            network.trainables.parameters().states()["cell.factor"], cell.trainables.parameters().states()["factor"]
        )

    def test_prepared_network_keeps_weight_and_threshold_gradients(self):
        self._check_prepared_network_gradients("scatter")

    def test_prepared_network_keeps_weight_and_threshold_gradients_brainevent(self):
        self._check_prepared_network_gradients("brainevent")

    def test_event_backends_agree_elementwise(self):
        traces = {}
        for backend in ("scatter", "brainevent", "auto"):
            result = make_runtime_network(delay=0.2 * u.ms).run(
                dt=0.1 * u.ms, duration=1.0 * u.ms, event_backend=backend
            )
            traces[backend] = result.samples["post"]["g"].values.to_decimal(u.uS)
        self.assertGreater(float(np.max(traces["scatter"])), 0.0)
        np.testing.assert_allclose(traces["brainevent"], traces["scatter"], rtol=1e-6)
        # "auto" resolves to the scatter path, so it is bit-identical.
        np.testing.assert_array_equal(traces["auto"], traces["scatter"])

    def test_brainevent_without_csrmv_is_rejected_at_prepare_run(self):
        network = make_runtime_network(delay=0.2 * u.ms)
        with patch.dict("sys.modules", {"brainevent": ModuleType("brainevent")}):
            with self.assertRaisesRegex(RuntimeError, "event_backend='scatter'"):
                network.prepare_run(dt=0.1 * u.ms, event_backend="brainevent")

    def _check_prepared_network_gradients(self, backend):
        network = make_runtime_network(delay=0.2 * u.ms)
        network.init_state()
        post = network.populations["post"].cell
        pre = network.populations["pre"].cell
        post.connections["drive"].trainable(weight=braincell.trainable.scale(name="w"))
        pre.event_outputs["spike"].trainable(threshold=braincell.trainable.parameter(group_by="all", name="th"))
        network.prepare_run(dt=0.1 * u.ms, event_backend=backend)
        synapse = post.runtime.get_runtime_node(post.synapses["exp"]._store.layout_id("ExpSyn"))

        def observe():
            network.reset_state()
            brainstate.transform.for_loop(lambda _: network.update(), np.arange(5))
            return synapse.g.value.to_decimal(u.uS).sum()

        run = brainstate.transform.jit(observe)
        grad = brainstate.transform.jit(
            brainstate.transform.grad(observe, grad_states=network.trainables.parameters().states())
        )
        first = run()
        gradients = grad()
        self.assertGreater(float(first), 0.0)
        self.assertNotEqual(float(gradients["post.w"]), 0.0)
        self.assertNotEqual(float(u.get_mantissa(gradients["pre.th"])), 0.0)
        values = network.trainables.parameters().physical_values()
        values["post.w"] = 2.0
        network.trainables.parameters().set_physical_values(values)
        np.testing.assert_allclose(run(), 2 * first, rtol=1e-6)

    def test_prepare_required_and_update_matches_run(self):
        network = make_runtime_network(delay=0.2 * u.ms)
        with self.assertRaisesRegex(RuntimeError, "prepare_run"):
            network.update()
        network.prepare_run(dt=0.1 * u.ms, event_backend="scatter")
        post = network.populations["post"].cell
        synapse = post.runtime.get_runtime_node(post.synapses["exp"]._store.layout_id("ExpSyn"))

        def step(_):
            network.update()
            return synapse.g.value

        actual = brainstate.transform.for_loop(step, np.arange(5))
        reference = make_runtime_network(delay=0.2 * u.ms).run(
            dt=0.1 * u.ms, duration=0.5 * u.ms, event_backend="scatter"
        )
        # This fixture records before the step; update() exposes post-step state.
        np.testing.assert_allclose(
            actual.to_decimal(u.uS)[:-1], reference.samples["post"]["g"].values.to_decimal(u.uS)[1:], rtol=1e-6
        )

    def test_cell_has_one_network_execution_owner(self) -> None:
        cell = make_threshold_cell()
        first = Network("first")
        second = Network("second")
        first.add_population("pre", cell)
        with self.assertRaisesRegex(ValueError, "more than one Network population"):
            first.add_population("alias", cell)
        with self.assertRaisesRegex(RuntimeError, "already belongs"):
            second.add_population("pre", cell)

    def test_initialization_freezes_topology(self) -> None:
        network = make_runtime_network()
        network.init_state()
        self.assertIs(network.init_state(), network)
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.add_population("extra", braincell.NetStim())
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.connect(
                "late",
                source=network.populations["pre"],
                synapse=network.populations["post"].synapses["exp"],
            )

    def test_batch_size_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "batch execution"):
            make_runtime_network().init_state(batch_size=2)

    def test_zero_delay_delivers_on_source_boundary(self) -> None:
        result = make_runtime_network().run(dt=0.1 * u.ms, duration=0.4 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        first_nonzero = int(np.flatnonzero(conductance[:, 0] > 0.0)[0])
        self.assertEqual(first_nonzero, 1)

    def test_heterogeneous_delay_routes_rows_independently(self) -> None:
        network = make_runtime_network(delay=[0.0, 0.3] * u.ms)
        result = network.run(dt=0.1 * u.ms, duration=0.6 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        self.assertEqual(int(np.flatnonzero(conductance[:, 0] > 0.0)[0]), 1)
        self.assertEqual(int(np.flatnonzero(conductance[:, 1] > 0.0)[0]), 4)

    def test_split_run_preserves_events_in_flight(self) -> None:
        continuous = make_runtime_network(delay=0.3 * u.ms).run(dt=0.1 * u.ms, duration=0.7 * u.ms)
        split_network = make_runtime_network(delay=0.3 * u.ms)
        split = braincell.NetworkResult.concat(
            (
                split_network.run(dt=0.1 * u.ms, duration=0.2 * u.ms),
                split_network.run(dt=0.1 * u.ms, duration=0.5 * u.ms),
            )
        )
        np.testing.assert_allclose(
            split.samples["post"]["g"].values.to_decimal(u.uS),
            continuous.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_reset_state_restarts_and_discards_events_in_flight(self) -> None:
        network = make_runtime_network(delay=0.3 * u.ms)
        first = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        network.reset_state()
        second = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        np.testing.assert_allclose(
            first.samples["post"]["g"].values.to_decimal(u.uS),
            second.samples["post"]["g"].values.to_decimal(u.uS),
        )

    def test_event_backend_auto_matches_scatter(self) -> None:
        auto = make_runtime_network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="auto",
        )
        scatter = make_runtime_network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="scatter",
        )
        np.testing.assert_allclose(
            auto.samples["post"]["g"].values.to_decimal(u.uS),
            scatter.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_run_setup_and_compiled_loop_are_reused(self) -> None:
        network = make_runtime_network()
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        setup_count = len(network._run_setup_cache)
        loop_count = len(network._network_run_loop_cache)
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(network._run_setup_cache), setup_count)
        self.assertEqual(len(network._network_run_loop_cache), loop_count)


class SplitNetworkRunTest(unittest.TestCase):
    """CELL-34: ``Network.run`` split at any step, event times included, matches one run."""

    def setUp(self):
        self.enterContext(_float64())

    @staticmethod
    def _run(solver, source_kwargs, steps, *, delay=0.0 * u.ms):
        cell, _ = make_event_cell(solver)
        network = Network("split")
        source = network.add_population("stim", braincell.NetStim(**source_kwargs))
        target = network.add_population("post", cell)
        network.connect("drive", source=source, synapse=target.synapses["syn"], weight=WEIGHT, delay=delay)
        parts = tuple(network.run(dt=DT, duration=n * DT) for n in steps)
        return braincell.NetworkResult.concat(parts).samples["post"]

    def _check(self, source_kwargs, steps, **kwargs):
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver, steps=steps, **kwargs):
                whole = self._run(solver, source_kwargs, (sum(steps),), **kwargs)
                split = self._run(solver, source_kwargs, steps, **kwargs)
                self.assertGreater(float(np.max(split["g"].values.to_decimal(u.uS))), 0.0)
                assert_same_trajectory(split, whole)

    def test_fixed_interval_events_inside_and_on_segment_ends(self):
        # Events every 0.5 ms from 0.5 ms; 1 ms segments put every other one
        # on a segment end. A 0.5 ms delay moves the arrivals by 20 steps.
        source = dict(start=0.5 * u.ms, interval=0.5 * u.ms, number=9)
        for delay in (0.0, 0.5):
            self._check(source, (40,) * 5, delay=delay * u.ms)

    def test_noisy_events_on_segment_ends(self):
        source = dict(start=0.5 * u.ms, interval=0.5 * u.ms, number=9, noise=0.5, seed=34)
        times = np.asarray(braincell.NetStim(**source).event_times.to_decimal(u.ms))[0]
        event_steps = np.unique(np.floor(times / 0.025 + 0.5).astype(int))
        # Split on the first three event steps, plus one step inside a segment.
        points = (*event_steps[:3], event_steps[0] + 7)
        self._check(source, split_steps(200, points))

    def test_random_split_points_match_one_run(self):
        rng = np.random.default_rng(34)
        event_steps = np.arange(10, 200, 10)
        points = (*rng.choice(event_steps, 2, replace=False), rng.choice(np.setdiff1d(np.arange(1, 200), event_steps)))
        self._check(dict(start=0.25 * u.ms, interval=0.25 * u.ms, number=19), split_steps(200, points))


def _resting_one_cv_cell(kind, solver, pop_size=2):
    """A soma 20 um x 10 um with a leak at its reversal, -65 mV."""
    if kind == "lumped":
        cell = braincell.Cell.point(
            pop_size, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver=solver
        )
    else:
        morphology = braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
        cell = braincell.Cell(
            morphology, pop_size=pop_size, cv_policy=braincell.CVPerBranch(), V_init=-65.0 * u.mV, solver=solver
        )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV))
    return cell


_DELTA_PATHS = (("lumped", "rk4"), ("cable", "rk4"), ("cable", "staggered"))


class NetworkDeltaInputTest(unittest.TestCase):
    """CELL-33: Network-managed cells add delta inputs; a later input drops the compiled loops."""

    def test_delta_projection_drives_update_in_successive_loops(self):
        # On main the lumped and "staggered" cells never consumed a kick (the
        # second loop raised ValueError: key already defined), and the second
        # loop of the "rk4" cable cell raised UnexpectedTracerError: update()
        # reused the loop traced with the first loop's input.
        import brainpy

        kicks = np.asarray([[1.0, 0.5], [2.0, 0.0], [0.0, 1.5]])
        for kind, solver in _DELTA_PATHS:
            with self.subTest(kind=kind, solver=solver), _float64():
                finals = []
                for managed in (True, False):
                    cell = _resting_one_cv_cell(kind, solver)
                    shape = (2,) if kind == "lumped" else (2, 1)
                    proj = brainpy.state.DeltaProj(comm=lambda x, shape=shape: x.reshape(shape) * u.mV, post=cell)
                    if managed:
                        network = Network("kicked")
                        network.add_population("cells", cell)
                        network.prepare_run(dt=DT)
                        advance = network.update
                    else:
                        cell.init_state()
                        advance = cell.update

                    def step(index, kick, proj=proj, advance=advance):
                        with brainstate.environ.context(t=index * DT):
                            proj(kick)
                            advance()
                        return index

                    with brainstate.environ.context(dt=DT):
                        for first in (0, 3):
                            brainstate.transform.for_loop(step, np.arange(first, first + 3), kicks)
                    finals.append(cell.V.value.to_decimal(u.mV).reshape(-1))
                # Six kicks: 3 + 2 mV per member, the leak relaxing them a little.
                np.testing.assert_allclose(finals[0], [-59.0, -61.0], atol=0.2)
                np.testing.assert_allclose(finals[0], finals[1], rtol=0.0, atol=1e-12)

    def test_run_includes_inputs_added_after_a_run(self):
        # On main the second Network.run reused the loop traced without them.
        def kick():
            return 0.25 * u.mV

        def drive(V):
            return u.math.full_like(u.get_mantissa(V), 10.0) * u.uA / u.cm**2

        for kind, solver in _DELTA_PATHS:
            for entry in ("delta", "current"):
                with self.subTest(kind=kind, solver=solver, input=entry):
                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("late_input")
                    network.add_population("cells", cell)
                    network.run(dt=DT, duration=0.5 * u.ms)
                    np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -65.0)
                    if entry == "delta":
                        cell.add_delta_input("kick", kick)
                    else:
                        cell.add_current_input("drive", drive)
                    network.run(dt=DT, duration=0.5 * u.ms)
                    # 20 kicks of 0.25 mV, or 10 uA/cm^2 for 0.5 ms, less the leak.
                    self.assertGreater(float(np.min(cell.V.value.to_decimal(u.mV))), -61.0)


def _two_branch_tree() -> braincell.Morphology:
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
    tree = braincell.Morphology.from_root(soma, name="soma")
    tree.soma.dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"
    )
    return tree


class NetworkInitSnapshotTest(unittest.TestCase):
    """MORPH-14: Network cells run on the morphology as it was at ``init_state``."""

    def test_editing_a_shared_morphology_after_init_warns_once_and_runs(self):
        tree = _two_branch_tree()
        first, second = braincell.Cell(tree), braincell.Cell(tree)
        network = Network("shared")
        network.add_population("first", first)
        network.add_population("second", second)
        network.init_state()
        self.assertIsNot(first.morpho, tree)
        self.assertIsNot(second.morpho, first.morpho)

        tree.soma.extra = braincell.Branch.from_lengths(
            lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"
        )

        def snapshot_warnings():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                network.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
            return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

        (message,) = snapshot_warnings()
        self.assertRegex(message, r"populations 'first', 'second' run on the snapshot.*new Network")
        self.assertEqual(snapshot_warnings(), [])
        self.assertEqual((first.n_cv, second.n_cv), (2, 2))


def _leak(name="leak"):
    return braincell.mech.Channel("IL", name=name, g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV)


def _leak_cell(kind="cable"):
    """A 40 um branch named soma, radius 2 um, two CVs (one on a lumped Cell), with a leak."""
    branch = braincell.Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 2.0] * u.um)
    policy = braincell.Lumped() if kind == "lumped" else braincell.CVPerBranch(2)
    cell = braincell.Cell(braincell.Morphology.from_root(branch, name="soma"), cv_policy=policy)
    cell.paint(AllRegion(), _leak())
    return cell


def _failing_cell():
    """A Cell whose init_state raises: a sodium ion named after the potassium family."""
    cell = _leak_cell()
    cell.paint(AllRegion(), braincell.mech.Ion("SodiumFixed", name="k"))
    return cell


_INIT_ERROR = "conflicts with canonical family key"


class NetworkInitRollbackTest(unittest.TestCase):
    """CELL-37: when a Cell fails init_state, the Cells this call initialized go back to declaring."""

    @staticmethod
    def _declaration(cell):
        # What the Cell is before Network.init_state: its declaration tree,
        # rules and the grid built from them.
        return {
            "morpho": cell.morpho,
            "paint rules": cell._paint_rules,
            "place rules": cell._place_rules,
            "grid": cell._discretization,
        }

    def test_the_cells_before_a_failing_cell_are_declaring_as_before(self):
        for kind in ("cable", "lumped"):
            with self.subTest(kind=kind):
                a, b = _leak_cell(kind), _failing_cell()
                net = Network("n")
                net.add_population("a", a)
                net.add_population("b", b)
                declared = a.morpho
                before = self._declaration(a)
                generation = a._view_generation
                leak = a.channels["leak"]

                with self.assertRaisesRegex(ValueError, _INIT_ERROR):
                    net.init_state()

                self.assertEqual((a._initialized, b._initialized, net._initialized), (False, False, False))
                self.assertIs(a.morpho, declared)
                for (name, now), then in zip(self._declaration(a).items(), before.values()):
                    self.assertTrue(now is then, f"{name} changed")
                self.assertEqual(a._view_generation, generation)
                self.assertEqual(leak.names, ("leak",))  # a selection made before the call is still live
                # The recovery: a declares again, and the Network takes new connections onto it.
                a.paint(AllRegion(), _leak("extra"))
                net.connect(
                    "c",
                    source=net.populations["b"].event_outputs["spike"],
                    target=net.populations["a"],
                    synapse=braincell.mech.Synapse("ExpSyn", name="s"),
                    locations=at("soma", 0.5),
                )
                self.assertFalse(a._initialized)

    def test_a_retry_after_the_failure_initializes_and_runs(self):
        a, b = _leak_cell(), _leak_cell()
        net = Network("n")
        net.add_population("a", a)
        net.add_population("b", b)
        boom = RuntimeError("boom")
        init_runtime = braincell.Cell._init_runtime
        failed = []

        def fail_once(cell, batch_size):
            if cell is b and not failed:
                failed.append(cell)
                raise boom
            return init_runtime(cell, batch_size)

        with patch.object(braincell.Cell, "_init_runtime", autospec=True, side_effect=fail_once):
            with self.assertRaises(RuntimeError) as caught:
                net.init_state()
            self.assertIs(caught.exception, boom)  # the original error, not wrapped
            self.assertEqual((a._initialized, b._initialized, net._initialized), (False, False, False))
            net.init_state()
        self.assertEqual((a._initialized, b._initialized, net._initialized), (True, True, True))
        result = net.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(_voltages(a), -65.0, atol=1e-6)

    def test_rolls_back_in_reverse_order_and_leaves_cells_initialized_earlier(self):
        early = _leak_cell()
        early.init_state()
        early_snapshot = early.morpho
        first, second, failing = _leak_cell(), _leak_cell(), _failing_cell()
        net = Network("n")
        for name, cell in (("first", first), ("early", early), ("second", second), ("failing", failing)):
            net.add_population(name, cell)
        reset = braincell.Cell.reset
        order = []

        def tracked(cell):
            order.append(cell)
            return reset(cell)

        with patch.object(braincell.Cell, "reset", autospec=True, side_effect=tracked):
            with self.assertRaisesRegex(ValueError, _INIT_ERROR):
                net.init_state()

        self.assertEqual([id(cell) for cell in order], [id(second), id(first)])
        self.assertTrue(early._initialized)  # not initialized by this call: left as it was
        self.assertIs(early.morpho, early_snapshot)
        self.assertEqual((first._initialized, second._initialized), (False, False))


def _add_value(cell, entry):
    if entry == "delta":
        cell.add_delta_input("kick", 5.0 * u.mV)
    else:
        cell.add_current_input("pulse", 10.0 * u.uA / u.cm**2)


def _voltages(cell):
    return np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)


class NetworkValueInputTest(unittest.TestCase):
    """CELL-36: a value input on a population Cell acts on one step of Network.run and Network.update."""

    def _update_loop(self, kind, solver, entry):
        cell = _resting_one_cv_cell(kind, solver)
        cell.init_state()
        _add_value(cell, entry)
        finals = []
        with brainstate.environ.context(dt=DT):
            for step in range(8):
                with brainstate.environ.context(t=step * DT):
                    cell.update()
                if step in (3, 7):
                    finals.append(_voltages(cell))
        return finals

    def test_run_and_update_match_the_cell_update_loop(self):
        # Acceptance 4. On main both loops added the value at every step (a
        # constant of their trace), and so did every later call that reused
        # them: Network.run gave -45.22 mV after four steps of a 5 mV delta.
        for kind, solver in _DELTA_PATHS:
            for entry in ("delta", "current"):
                with self.subTest(kind=kind, solver=solver, input=entry), _float64():
                    expected = self._update_loop(kind, solver, entry)
                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("valued")
                    network.add_population("cells", cell)
                    network.init_state()
                    _add_value(cell, entry)
                    actual = []
                    for _ in range(2):
                        network.run(dt=DT, duration=4 * DT)
                        actual.append(_voltages(cell))
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("stepped")
                    network.add_population("cells", cell)
                    network.prepare_run(dt=DT)
                    _add_value(cell, entry)
                    stepped = []
                    with brainstate.environ.context(dt=DT):
                        for step in range(8):
                            network.update()
                            if step in (3, 7):
                                stepped.append(_voltages(cell))
                    np.testing.assert_allclose(stepped, expected, rtol=1e-12, atol=0.0)
                    if entry == "delta":
                        np.testing.assert_allclose(actual[0], -60.11, atol=0.01)

    def test_value_registered_between_runs_does_not_retrace(self):
        with _float64():
            cell = _resting_one_cv_cell("cable", "rk4")
            network = Network("late_value")
            network.add_population("cells", cell)
            network.run(dt=DT, duration=4 * DT)
            loops = dict(network._network_run_loop_cache)
            _add_value(cell, "delta")
            self.assertEqual(dict(network._network_run_loop_cache), loops)
            network.run(dt=DT, duration=4 * DT)
            self.assertGreater(float(np.min(_voltages(cell))), -60.2)
            self.assertNotIn("kick", cell.delta_inputs)

    def test_a_run_that_fails_keeps_its_value_inputs(self):
        cell = _resting_one_cv_cell("lumped", "rk4")
        network = Network("failing")
        network.add_population("cells", cell)
        network.prepare_run(dt=DT)
        cell.add_current_input("pulse", np.ones((3,)) * 10.0 * u.uA / u.cm**2)
        for advance in (lambda: network.update(), lambda: network.run(dt=DT, duration=4 * DT)):
            with brainstate.environ.context(dt=DT), self.assertRaisesRegex(ValueError, "pulse"):
                advance()
            self.assertIn("pulse", cell.current_inputs)


def _event_times(stim) -> np.ndarray:
    model = stim.model if hasattr(stim, "model") else stim
    return np.asarray(model.event_times.to_decimal(u.ms))


def _soma_dend_cell(name, *, random):
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    morpho = braincell.Morphology.from_root(soma, name="soma")
    morpho.soma.dend = braincell.Branch.from_lengths(lengths=[200.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
    v_init = braintools.init.Uniform(-70.0 * u.mV, -60.0 * u.mV) if random else -65.0 * u.mV
    cell = braincell.Cell(morpho, pop_size=3, cv_policy=braincell.CVPerBranch(4), V_init=v_init, name=name)
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.loc(at("soma", 0.5)).record("v", braincell.observe.state("v"))
    if not random:
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=1.0 * u.ms, durations=5.0 * u.ms, amplitudes=0.05 * u.nA),
        )
        return cell, None
    sites = braincell.filter.sample(AllRegion(), number=3)
    cell.place(sites, braincell.mech.Synapse("ExpSyn", name="dend", tau=3.0 * u.ms, e=0.0 * u.mV))
    # A user callback drawing from the default stream at every step.
    cell.place(
        at("soma", 0.5),
        braincell.mech.FunctionClamp(fn=lambda t: brainstate.random.uniform(0.0, 0.02) * u.nA),
    )
    return cell, sites


def _context_experiment() -> dict:
    """Every source of randomness of a Network experiment, none given a seed (CELL-11a acceptance 2)."""
    net = Network("demo", seed=None)
    stim = net.add_population(
        "stim", braincell.NetStim(size=4, number=5, noise=1.0, start=1.0 * u.ms, interval=2.0 * u.ms)
    )
    cell, sites = _soma_dend_cell("post", random=True)
    det, _ = _soma_dend_cell("det", random=False)
    post = net.add_population("post", cell)
    net.add_population("det", det)
    rows = net.connect("drive", source=stim, synapse=post.synapses["ampa"], weight=0.002 * u.uS, pairing=independent(6))
    out = {
        "root": np.asarray(net.seed),
        "netstim": _event_times(stim),
        "pairing": np.stack([np.asarray(rows.source_index), np.asarray(rows.target_index)]),
        "sample_seed": np.asarray(sites.seed),
        "sites": np.asarray(sites.evaluate(cell.morpho).points, dtype=float),
    }
    result = net.run(dt=DT, duration=6.0 * u.ms)
    out["v_post"] = np.asarray(result.samples["post"]["v"].values.to_decimal(u.mV))
    out["v_det"] = np.asarray(result.samples["det"]["v"].values.to_decimal(u.mV))
    return out


def _two_stim_network(order) -> dict:
    net = Network("ordered", seed=None)
    made = {name: net.add_population(name, braincell.NetStim(size=2, number=4, noise=1.0)) for name in order}
    return {name: _event_times(stim) for name, stim in made.items()}


def _explicit_seed_draws() -> dict:
    """Explicit seeds and independent generators: local streams that no context controls."""
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    morpho = braincell.Morphology.from_root(soma, name="soma")
    network_stim = Network(seed=7).add_population("s", braincell.NetStim(size=3, number=4, noise=1.0))
    return {
        "NetStim(seed=9)": _event_times(braincell.NetStim(size=3, number=4, noise=1.0, seed=9)),
        "Network(seed=7)": _event_times(network_stim),
        "sample(seed=3)": np.asarray(
            braincell.filter.sample(AllRegion(), number=4, seed=3).evaluate(morpho).points, dtype=float
        ),
        "RandomSamples(seed=123)": np.asarray(RandomSamples(AllRegion(), 4, 123).evaluate(morpho).points, dtype=float),
        "RandomState(5)": np.asarray(brainstate.random.RandomState(5).normal(size=3)),
        "jax.random.key(3)": np.asarray(jax.random.normal(jax.random.key(3), (3,))),
        "default_rng(1)": np.random.default_rng(1).random(3),
    }


class NetworkContextSeedTest(unittest.TestCase):
    """CELL-11a: ``Network(seed=None)`` draws its root from the BrainState default stream."""

    def test_seed_none_draws_the_root_by_the_definition(self) -> None:
        # Acceptance 1. On main ``Network(seed=None)`` raised TypeError.
        for seed in (0, 42):
            with self.subTest(seed=seed):
                with brainstate.random.seed_context(seed):
                    roots = [Network(seed=None).seed, Network("second", seed=None).seed]
                self.assertEqual(roots, drawn_seeds(seed, 2))
                self.assertNotEqual(roots[0], roots[1])
                self.assertIs(type(roots[0]), int)
        roots = []
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision), brainstate.random.seed_context(42):
                roots.append(Network(seed=None).seed)
        self.assertEqual(roots[0], roots[1])

    def test_one_context_seed_controls_every_random_stream_of_an_experiment(self) -> None:
        # Acceptance 2: root, NetStim noise, pairing, sampled locations,
        # initial values and a callback; the Cell without randomness keeps its V.
        with brainstate.random.seed_context(42):
            first = _context_experiment()
        with brainstate.random.seed_context(42):
            again = _context_experiment()
        with brainstate.random.seed_context(43):
            other = _context_experiment()
        for key, value in first.items():
            with self.subTest(stream=key):
                np.testing.assert_array_equal(again[key], value)
                if key == "v_det":
                    np.testing.assert_allclose(other[key], value, rtol=1e-12, atol=0.0)
                else:
                    self.assertFalse(np.array_equal(other[key], value), f"{key} did not change with the seed")

    def test_the_recorded_root_reproduces_the_network_streams(self) -> None:
        # Acceptance 3: record ``net.seed``, pass it back as an explicit root.
        with brainstate.random.seed_context(42):
            drawn = Network(seed=None)
            stim = drawn.add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
        replayed = Network(seed=drawn.seed).add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
        np.testing.assert_array_equal(_event_times(replayed), _event_times(stim))

    def test_registration_order_does_not_change_the_streams_under_a_drawn_root(self) -> None:
        # Acceptance 4: a Network takes one draw, then derives by path.
        with brainstate.random.seed_context(5):
            forward = _two_stim_network(("a", "b"))
        with brainstate.random.seed_context(5):
            backward = _two_stim_network(("b", "a"))
        for name in ("a", "b"):
            np.testing.assert_array_equal(backward[name], forward[name])
        self.assertFalse(np.array_equal(forward["a"], forward["b"]))

    def test_reset_keeps_the_drawn_streams_and_outside_a_context_the_global_stream_is_used(self) -> None:
        # Acceptance 5. No persistent RNG: the schedule and the rows are concrete.
        with brainstate.random.seed_context(3):
            net = Network("reset", seed=None)
            stim = net.add_population("stim", braincell.NetStim(size=3, number=2, noise=1.0))
            soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"), pop_size=2, V_init=-65.0 * u.mV)
            cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
            post = net.add_population("c", cell)
            rows = net.connect("d", source=stim, synapse=post.synapses["ampa"], weight=WEIGHT, pairing=independent(5))
            events = _event_times(stim)
            pairs = (np.asarray(rows.source_index).copy(), np.asarray(rows.target_index).copy())
            net.run(dt=DT, duration=1.0 * u.ms)
            net.reset_state()
        fresh = net.connections["c", "d"]
        np.testing.assert_array_equal(_event_times(stim), events)
        np.testing.assert_array_equal(np.asarray(fresh.source_index), pairs[0])
        np.testing.assert_array_equal(np.asarray(fresh.target_index), pairs[1])

        with brainstate.random.seed_context(4):
            inside = Network(seed=None).seed
        brainstate.random.seed(4)
        self.assertEqual(Network(seed=None).seed, inside)

    def test_invalid_seeds_still_raise_at_construction(self) -> None:
        # Acceptance 6; main raised the same TypeError.
        for seed in (True, 1.5, "3"):
            with self.subTest(seed=seed), self.assertRaisesRegex(TypeError, "Network seed must be an integer"):
                Network(seed=seed)

    def test_seed_none_under_a_transformation_raises_and_leaves_the_default_stream(self) -> None:
        # Acceptance 6: inside a trace the default stream has no concrete key.
        def in_loop(fn):
            return lambda: brainstate.transform.for_loop(lambda i: fn(), np.arange(2))

        for name, transform in (("jit", brainstate.transform.jit), ("for_loop", in_loop)):
            with self.subTest(transform=name):
                brainstate.random.seed(1)
                expected = float(brainstate.random.uniform())
                brainstate.random.seed(1)
                before = default_key_data()
                with self.assertRaisesRegex(TypeError, "traced"):
                    transform(lambda: Network(seed=None).seed)()
                np.testing.assert_array_equal(default_key_data(), before)
                self.assertEqual(float(brainstate.random.uniform()), expected)
        # The suggested fix, an integer seed, builds inside jit.
        self.assertEqual(int(brainstate.transform.jit(lambda: Network(seed=3).seed)()), 3)

    def test_explicit_seeds_and_independent_generators_ignore_the_context(self) -> None:
        # Acceptance 7.
        with brainstate.random.seed_context(1):
            one = _explicit_seed_draws()
        with brainstate.random.seed_context(2):
            two = _explicit_seed_draws()
        for key, value in one.items():
            with self.subTest(stream=key):
                np.testing.assert_array_equal(two[key], value)


if __name__ == "__main__":
    unittest.main()
