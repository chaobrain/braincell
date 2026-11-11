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

"""Tests for :mod:`braincell._multi_compartment.run`: helpers and split runs."""

import contextlib
import unittest
import warnings

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment.run import (
    RunResult,
    _validate_time_quantity,
)
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import AllRegion, RootLocation, at
from braincell.synapse import ExpSyn

DT = 0.025 * u.ms
WEIGHT = 0.005 * u.uS


class TestValidateTimeQuantity(unittest.TestCase):
    def test_accepts_scalar_time(self):
        _validate_time_quantity(0.1 * u.ms, name="dt")

    def test_rejects_plain_float(self):
        with self.assertRaises(TypeError):
            _validate_time_quantity(0.1, name="dt")

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            _validate_time_quantity(0.0 * u.ms, name="dt")

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            _validate_time_quantity(-0.1 * u.ms, name="dt")


class TestRunResult(unittest.TestCase):
    def test_is_frozen_dataclass(self):
        result = RunResult(time=u.math.arange(0.0, 3.0, 1.0) * u.ms, traces={"v": 0})
        self.assertEqual(result.traces, {"v": 0})


@contextlib.contextmanager
def _float64():
    with jax.enable_x64(True), brainstate.environ.context(precision=64):
        yield


def make_event_cell(solver="staggered", cv_per_branch=3):
    """One 600 um x 0.5 um passive dendrite (3 CVs) with an ExpSyn at x = 0.

    Leak 0.1 mS/cm^2 at -65 mV, cm 1 uF/cm^2, Ra 100 ohm cm; ExpSyn tau 2 ms,
    e 0 mV. Returns ``(cell, synapse view)``; the cell is not initialized.
    """
    branch = braincell.Branch.from_lengths(lengths=[600.0] * u.um, radii=[0.5, 0.5] * u.um, type="dendrite")
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch, name="dend"),
        cv_policy=braincell.CVPerBranch(cv_per_branch=cv_per_branch),
        V_init=-65.0 * u.mV,
        solver=solver,
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
    )
    declaration = mech.Synapse(ExpSyn, name="syn", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(RootLocation(0.0), declaration)
    cell.record("v", braincell.observe.state("v"))
    cell.loc(RootLocation(0.0)).record("g", braincell.observe.synapse(name="syn").state("g"))
    return cell, cell.synapses[declaration]


def split_steps(total_steps, points):
    """Step counts of the segments ``[0, p1), [p1, p2), ..., [pk, total)``."""
    edges = (0, *sorted(int(step) for step in points), int(total_steps))
    return tuple(stop - start for start, stop in zip(edges, edges[1:]))


def traces(samples):
    """``(g in uS, V in mV)`` at each step start, from one cell's samples."""
    return (
        np.asarray(samples["g"].values.to_decimal(u.uS)),
        np.asarray(samples["v"].values.to_decimal(u.mV)),
    )


def assert_same_trajectory(split, whole):
    """Compare two cells' samples of ``make_event_cell``'s recordings."""
    for name, actual, expected in zip(("g", "V"), traces(split), traces(whole)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0, err_msg=name)


class SplitCellRunTest(unittest.TestCase):
    """CELL-34: ``Cell.run`` split at any step, event times included, matches one run.

    Each step prepares the synaptic input of the boundary that follows it, so
    a run continuing an earlier run or ``update()`` must not prepare its first
    boundary again: events landing on it would be delivered twice.
    """

    def setUp(self):
        self.enterContext(_float64())

    @staticmethod
    def _cell(solver, *sources):
        cell, synapse = make_event_cell(solver)
        for index, source in enumerate(sources):
            braincell.connect(f"drive{index}", source=source, synapse=synapse, weight=WEIGHT)
        cell.init_state()
        return cell

    @staticmethod
    def _run(cell, steps):
        return RunResult.concat(tuple(cell.run(dt=DT, duration=n * DT) for n in steps))

    def test_split_at_the_event_time_delivers_it_once(self):
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver):
                whole = self._run(self._cell(solver, braincell.NetStim(start=1.0 * u.ms, number=1)), (200,))
                split = self._run(self._cell(solver, braincell.NetStim(start=1.0 * u.ms, number=1)), (40,) * 5)
                assert_same_trajectory(split.samples, whole.samples)
                g = traces(whole.samples)[0][:, 0]
                # Sampled at each step start: the event at 1 ms acts in step 40.
                self.assertEqual(int(np.flatnonzero(g > 0.0)[0]), 41)
                np.testing.assert_allclose(g[41], 0.005 * np.exp(-0.025 / 2.0), rtol=1e-9)

    def test_random_split_points_match_one_run(self):
        # Events every 0.25 ms (10 steps) and a noisy source; two split points
        # on event steps and one elsewhere, drawn with a fixed seed.
        def sources():
            return (
                braincell.NetStim(start=0.25 * u.ms, interval=0.25 * u.ms, number=19),
                braincell.NetStim(start=0.1 * u.ms, interval=0.4 * u.ms, number=12, noise=0.5, seed=34),
            )

        rng = np.random.default_rng(34)
        event_steps = np.arange(10, 200, 10)
        points = (*rng.choice(event_steps, 2, replace=False), rng.choice(np.setdiff1d(np.arange(1, 200), event_steps)))
        steps = split_steps(200, points)
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver, steps=steps):
                whole = self._run(self._cell(solver, *sources()), (200,))
                split = self._run(self._cell(solver, *sources()), steps)
                assert_same_trajectory(split.samples, whole.samples)

    def test_update_then_run_and_run_then_update_deliver_once(self):
        def cell():
            return self._cell("staggered", braincell.NetStim(start=1.0 * u.ms, number=1))

        def g(model):
            node = next(iter(model.runtime_objects(ExpSyn, allowed_hierarchy=(1, 1)).values()))
            return np.asarray(node.g.value.to_decimal(u.uS))

        def updates(model, first, last):
            with brainstate.environ.context(dt=DT):
                for step in range(first, last):
                    with brainstate.environ.context(t=step * DT):
                        model.update()

        whole = cell()
        whole.run(dt=DT, duration=80 * DT)
        then_run = cell()
        updates(then_run, 0, 40)
        then_run._set_current_time(40 * DT)
        then_run.run(dt=DT, duration=40 * DT)
        then_update = cell()
        then_update.run(dt=DT, duration=40 * DT)
        updates(then_update, 40, 80)
        np.testing.assert_allclose(g(then_run), g(whole), rtol=1e-12)
        np.testing.assert_allclose(g(then_update), g(whole), rtol=1e-12)

    def test_first_run_after_init_or_reset_delivers_a_start_event(self):
        cell = self._cell("staggered", braincell.NetStim(start=0.0 * u.ms, number=1))
        first = traces(cell.run(dt=DT, duration=4 * DT).samples)
        cell.reset_state()
        again = traces(cell.run(dt=DT, duration=4 * DT).samples)
        self.assertGreater(float(first[0][1, 0]), 0.0)
        for actual, expected in zip(again, first):
            np.testing.assert_array_equal(actual, expected)

    def test_bound_inputs_are_added_once_per_boundary(self):
        def cell():
            model, _ = make_event_cell("staggered")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model.bind_synapse_input("syn", 1e-4)
            model.init_state()
            return model

        whole = self._run(cell(), (80,))
        split = self._run(cell(), (40, 40))
        assert_same_trajectory(split.samples, whole.samples)

    def test_reduced_cell_receives_a_boundary_event_once(self):
        def run(steps):
            cell = _cell(1)
            cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="input"))
            cell.add_reduction("payload", braincell.PayloadAccumulatorReduction(threshold=10.0 * u.uS))
            cell.use_model("payload")
            cell.record("value", braincell.observe.output("value"))
            braincell.connect(
                "drive",
                source=braincell.NetStim(start=0.3 * u.ms, number=1),
                synapse=cell.synapses["input"],
                weight=0.2 * u.uS,
            )
            cell.init_state()
            result = RunResult.concat(tuple(cell.run(dt=0.1 * u.ms, duration=n * 0.1 * u.ms) for n in steps))
            return np.asarray(result.samples["value"].values.to_decimal(u.uS))[:, 0]

        np.testing.assert_allclose(run((6,)), [0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(run((3, 3)), run((6,)))

    def test_traced_runs_continue_without_redelivery(self):
        cell = self._cell("staggered", braincell.NetStim(start=1.0 * u.ms, number=1))
        node = next(iter(cell.runtime_objects(ExpSyn, allowed_hierarchy=(1, 1)).values()))
        segment = brainstate.transform.jit(lambda: cell.run(dt=DT, duration=40 * DT).stop_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(3):
                segment()
        whole = self._cell("staggered", braincell.NetStim(start=1.0 * u.ms, number=1))
        whole.run(dt=DT, duration=120 * DT)
        reference = next(iter(whole.runtime_objects(ExpSyn, allowed_hierarchy=(1, 1)).values()))
        np.testing.assert_allclose(node.g.value.to_decimal(u.uS), reference.g.value.to_decimal(u.uS), rtol=1e-12)


class UpdateLoopTest(unittest.TestCase):
    """CELL-35: an ``update()`` loop matches ``run``, events on the first step included.

    ``run`` prepares its first boundary when no step has; ``update()`` must do
    the same, or an event at the start time after ``init_state`` or
    ``reset_state`` is never delivered.
    """

    N = 80

    def setUp(self):
        self.enterContext(_float64())

    @staticmethod
    def _cell(solver, start):
        """The one-CV event cell, driven by ``NetStim(start, number=1)`` or, for
        ``start=None``, by a bound input of 1e-4 uS on every boundary."""
        cell, synapse = make_event_cell(solver, cv_per_branch=1)
        if start is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cell.bind_synapse_input("syn", 1e-4)
        else:
            braincell.connect("drive", source=braincell.NetStim(start=start, number=1), synapse=synapse, weight=WEIGHT)
        cell.init_state()
        return cell

    @staticmethod
    def _updates(cell, first, last, *, traced=True):
        """``(g in uS, V in mV)`` before each ``update()`` of steps ``[first, last)``.

        ``traced=False`` calls ``update()`` eagerly, one step at a time.
        """

        def step(index):
            with brainstate.environ.context(t=index * DT):
                state = (_synapse(cell).g.value.to_decimal(u.uS), cell.V.value.to_decimal(u.mV))
                cell.update()
            return state

        with brainstate.environ.context(dt=DT):
            if traced:
                g, v = brainstate.transform.for_loop(step, np.arange(first, last))
            else:
                g, v = zip(*(step(index) for index in range(first, last)))
        return np.asarray(g), np.asarray(v)

    def test_update_loop_matches_run_with_an_event_at_the_start(self):
        for solver in ("staggered", "rk4"):
            for start in (0.0 * u.ms, 1.0 * u.ms, None):
                with self.subTest(solver=solver, start=start):
                    looped = self._cell(solver, start)
                    trajectory = self._updates(looped, 0, self.N)
                    ran = self._cell(solver, start)
                    expected = traces(ran.run(dt=DT, duration=self.N * DT).samples)
                    for name, actual, reference in zip(("g", "V"), trajectory, expected):
                        np.testing.assert_allclose(
                            actual.reshape(reference.shape), reference, rtol=1e-12, atol=0.0, err_msg=name
                        )
                    for actual, expected in ((looped, ran), (_synapse(looped), _synapse(ran))):
                        np.testing.assert_allclose(
                            _state(actual), _state(expected), rtol=1e-12, atol=0.0, err_msg=type(actual).__name__
                        )
                    if start == 0.0 * u.ms:
                        # The event acts in step 0: g is non-zero from the second sample on.
                        self.assertEqual(float(trajectory[0][0, 0]), 0.0)
                        np.testing.assert_allclose(trajectory[0][1, 0], 0.005 * np.exp(-0.025 / 2.0), rtol=1e-9)

    def test_first_update_after_reset_delivers_the_start_event_again(self):
        cell = self._cell("rk4", 0.0 * u.ms)
        first = self._updates(cell, 0, 4)
        cell.reset_state()
        again = self._updates(cell, 0, 4)
        self.assertGreater(float(first[0][1, 0]), 0.0)
        for actual, expected in zip(again, first):
            np.testing.assert_array_equal(actual, expected)
        # Eager calls decide outside a trace and deliver the event once too.
        cell.reset_state()
        for actual, expected in zip(self._updates(cell, 0, 4, traced=False), first):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    def test_update_and_run_continue_each_other_without_redelivery(self):
        # Events at 0 ms and 1 ms (step 40); the first segment starts with an
        # event, the second at one.
        def cell():
            model, synapse = make_event_cell("rk4", cv_per_branch=1)
            for name, start in (("first", 0.0 * u.ms), ("split", 1.0 * u.ms)):
                braincell.connect(name, source=braincell.NetStim(start=start, number=1), synapse=synapse, weight=WEIGHT)
            model.init_state()
            return model

        whole = cell()
        whole.run(dt=DT, duration=self.N * DT)
        then_run = cell()
        self._updates(then_run, 0, 40)
        then_run._set_current_time(40 * DT)
        then_run.run(dt=DT, duration=40 * DT)
        then_update = cell()
        then_update.run(dt=DT, duration=40 * DT)
        self._updates(then_update, 40, self.N)
        for model in (then_run, then_update):
            np.testing.assert_allclose(_state(_synapse(model)), _state(_synapse(whole)), rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(_state(model), _state(whole), rtol=1e-12, atol=0.0)
        self.assertGreater(float(_state(_synapse(whole))[0]), 0.0)


def _synapse(cell):
    return next(iter(cell.runtime_objects(ExpSyn, allowed_hierarchy=(1, 1)).values()))


def _state(model):
    """A cell's ``V`` in mV or a synapse's ``g`` in uS, as a numpy array."""
    if isinstance(model, ExpSyn):
        return np.asarray(model.g.value.to_decimal(u.uS))
    return np.asarray(model.V.value.to_decimal(u.mV))


if __name__ == "__main__":
    unittest.main()
