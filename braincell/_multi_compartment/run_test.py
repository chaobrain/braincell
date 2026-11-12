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
import re
import unittest
import warnings
from unittest import mock

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment import run as run_module
from braincell._multi_compartment.run import (
    RunResult,
    _validate_time_quantity,
)
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import AllRegion, RootLocation, at
from braincell.quad import _staggered
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
    cell.record("v", braincell.observe.state("V"))
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


class CellRunClockTest(unittest.TestCase):
    """CELL-45: ``Cell.run`` steps at ``origin + step·dt`` of an integer step count, not at an accumulated time."""

    N = 32010

    @staticmethod
    def _cell():
        """``make_event_cell`` with one CV, driven at 100, 409.6 and 800 ms (steps 4000, 16384, 32000)."""
        cell, synapse = make_event_cell("staggered", cv_per_branch=1)
        source = braincell.EventSequence.from_times([np.asarray([4000, 16384, 32000]) * DT])
        braincell.connect("drive", source=source, synapse=synapse, weight=WEIGHT)
        return cell

    def test_split_runs_step_at_the_times_of_one_run(self):
        # main: float32 800-step segments put 6338 of the 32010 steps at
        # other times than one run (float64: 10841).
        for precision, float_type, rtol in ((32, np.float32, 1e-6), (64, np.float64, 1e-12)):
            context = _float64() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                whole = self._cell().run(dt=DT, duration=self.N * DT)
                cell = self._cell()
                steps = (800,) * (self.N // 800) + (self.N % 800,)
                split = RunResult.concat(tuple(cell.run(dt=DT, duration=n * DT) for n in steps))
                expected = np.arange(self.N).astype(float_type) * float_type(0.025)
                np.testing.assert_array_equal(whole.time.to_decimal(u.ms), expected)
                np.testing.assert_array_equal(split.time.to_decimal(u.ms), expected)
                np.testing.assert_array_equal(
                    cell.current_time.to_decimal(u.ms), float_type(self.N) * float_type(0.025)
                )
                for name, actual, expected in zip(("g", "V"), traces(split.samples), traces(whole.samples)):
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0, err_msg=name)

    def test_a_change_of_dt_counts_on_from_the_time_reached(self):
        cell = self._cell()
        cell.run(dt=DT, duration=10 * DT)
        result = cell.run(dt=2 * DT, duration=20 * DT)
        np.testing.assert_allclose(result.time.to_decimal(u.ms), 0.25 + 0.05 * np.arange(10), rtol=1e-6)
        np.testing.assert_allclose(cell.current_time.to_decimal(u.ms), 0.75, rtol=1e-6)
        np.testing.assert_array_equal(cell._clock_step.value, 10)
        # A compiled function holds the choice in its trace: called again
        # after a run at another dt, it still counts from the time reached.
        traced = []

        def run():
            traced.append(True)
            return cell.run(dt=DT, duration=4 * DT).time

        jitted = brainstate.transform.jit(run)
        np.testing.assert_allclose(jitted().to_decimal(u.ms), 0.75 + 0.025 * np.arange(4), rtol=1e-6)
        cell.run(dt=2 * DT, duration=4 * DT)
        np.testing.assert_allclose(jitted().to_decimal(u.ms), 0.95 + 0.025 * np.arange(4), rtol=1e-6)
        self.assertEqual(len(traced), 1)
        np.testing.assert_allclose(cell.current_time.to_decimal(u.ms), 1.05, rtol=1e-6)
        cell.reset_state()
        np.testing.assert_array_equal(cell.current_time.to_decimal(u.ms), 0.0)
        np.testing.assert_array_equal(cell.run(dt=DT, duration=2 * DT).time.to_decimal(u.ms), [0.0, np.float32(0.025)])

    def test_the_step_count_stops_at_its_int32_limit(self):
        cell = self._cell()
        cell.run(dt=DT, duration=DT)
        cell._clock_step.value = np.int32(2**31 - 10)
        with self.assertRaisesRegex(ValueError, r"Cell\.run.*2147483647 steps"):
            cell.run(dt=DT, duration=20 * DT)
        np.testing.assert_array_equal(cell._clock_step.value, 2**31 - 10)
        # A run at another dt counts from step 0.
        cell.run(dt=2 * DT, duration=40 * DT)
        np.testing.assert_array_equal(cell._clock_step.value, 20)

    def test_segments_reuse_the_compiled_loop(self):
        cell = self._cell()
        cell.init_state()
        calls = [0]
        begin_step = cell._begin_step

        def counting(*args, **kwargs):
            calls[0] += 1
            return begin_step(*args, **kwargs)

        cell._begin_step = counting
        for _ in range(3):
            cell.run(dt=DT, duration=20 * DT)
        self.assertEqual(calls[0], 1)
        self.assertEqual(len(cell._run_loop_cache), 1)


class CellArrivalStepTest(unittest.TestCase):
    """CELL-46, CELL-48: arrivals in ``Cell.run`` after a change of dt, and in a ``Cell.update()`` loop with the caller's t."""

    TIMES = (0.3, 0.5125, 0.6, 0.8375, 1.0, 1.0125)

    @staticmethod
    def _counting_cell(times=TIMES):
        """A passive 1-CV cell at rest; its ExpSyn (tau 1e9 ms, e = rest) gains 1 uS per arrival."""
        cell = braincell.Cell(
            braincell.Morphology.from_root(
                braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
            ),
            cv_policy=braincell.CVPerBranch(1),
            V_init=-65.0 * u.mV,
        )
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        cell.place(at(0, 0.5), mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV))
        cell.record("g", braincell.observe.synapse(name="s").state("g"))
        source = braincell.EventSequence.from_times([np.asarray(times) * u.ms])
        braincell.connect("c", source=source, synapse=cell.synapses["s"], weight=1.0 * u.uS)
        return cell

    @staticmethod
    def _jumps(results):
        """``(sample times, arrivals)`` at which g steps up over ``results``; the jump at ``t_k + dt`` is step ``t_k``'s."""
        t = np.concatenate([np.asarray(r.time.to_decimal(u.ms), dtype=np.float64) for r in results])
        g = np.concatenate(
            [np.asarray(r.samples["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(-1) for r in results]
        )
        increments = np.diff(g, prepend=0.0)
        jumps = np.flatnonzero(increments > 0.5)
        return t[jumps], np.rint(increments[jumps])

    def test_a_change_of_dt_delivers_on_the_grid_of_the_time_reached(self):
        # After 3 steps of 0.025 ms the clock counts 0.05 ms steps from
        # 0.075 ms, 1.5 steps of 0.05 ms. The arrivals count from there too,
        # rounded half up at float64 (CELL-46): 0.3 ms is (0.3 - 0.075)/0.05 =
        # 4.5 steps, a tie, so it acts on step 5 (0.325 ms) and shows in the
        # sample at 0.375 ms; 0.5125 ms is 8.75 -> 9 (0.525 ms), 0.6 ms 10.5
        # -> 11 (0.625 ms), 0.8375 ms 15.25 -> 15 (0.825 ms), 1.0 and 1.0125 ms
        # 18.5 and 18.75 -> 19 (1.025 ms). main counted them from time 0, so
        # the boundary at 0.275 ms took 0.3 ms, half a new step early, and
        # 0.475 ms took 0.5125 ms (CELL-48). An event at 0.06 ms acted on the
        # first segment's step at 0.05 ms (2.4 steps of 0.025 ms); 0.3 steps
        # before the new origin, it is not delivered again.
        # After 10 steps the clock counts from 0.25 ms, on the 0.05 ms grid:
        # the steps are those of the grid from time 0, as on main.
        # Steps of 0.025, 0.05 and 0.1 ms (3, 5, 13) count the last from
        # 0.325 ms, 3.25 steps of 0.1 ms: 0.425 ms is step 1 (0.425 ms),
        # 0.455 ms 1.3 -> 1, 0.475 ms 1.5 -> 2 (0.525 ms), 0.6 ms 2.75 -> 3
        # (0.625 ms), 1.06 ms 7.35 -> 7 (1.025 ms), 1.375 ms 10.5 -> 11
        # (1.425 ms); main put 0.455 and 1.06 ms one step later (rounded from
        # time 0, 4.55 -> 5 and 10.6 -> 11 steps of 0.1 ms).
        later = (0.425, 0.455, 0.475, 0.6, 1.06, 1.375)
        cases = {
            "off the new grid": (
                (0.06,) + self.TIMES,
                (3, 22),
                [0.075, 0.375, 0.575, 0.675, 0.875, 1.075],
                [1, 1, 1, 1, 1, 2],
            ),
            "on the new grid": (self.TIMES, (10, 20), [0.35, 0.55, 0.65, 0.9, 1.05], [1, 1, 1, 1, 2]),
            "two changes": (later, (3, 5, 13), [0.525, 0.625, 0.725, 1.125, 1.525], [2, 1, 1, 1, 1]),
        }
        for precision in (32, 64):
            for name, (times, steps, expected_times, counts) in cases.items():
                context = _float64() if precision == 64 else brainstate.environ.context(precision=32)
                with self.subTest(precision=precision, case=name), context:
                    cell = self._counting_cell(times)
                    for _ in range(2):  # again after reset_state(), which forgets the clock
                        results = tuple(cell.run(dt=(2**i) * DT, duration=n * (2**i) * DT) for i, n in enumerate(steps))
                        actual_times, actual_counts = self._jumps(results)
                        np.testing.assert_allclose(actual_times, expected_times, rtol=1e-6)
                        np.testing.assert_array_equal(actual_counts, counts)
                        cell.reset_state()

    def test_segments_after_a_change_of_dt_match_one_run(self):
        # The arrival steps count from the clock's origin, which stays where
        # dt changed while the segments run at the new dt: they deliver at
        # the steps of one run, and reuse one compiled loop.
        for precision in (32, 64):
            context = _float64() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                whole = self._counting_cell()
                whole.run(dt=DT, duration=3 * DT)
                expected = whole.run(dt=2 * DT, duration=25 * 2 * DT)
                cell = self._counting_cell()
                cell.run(dt=DT, duration=3 * DT)
                parts = RunResult.concat(tuple(cell.run(dt=2 * DT, duration=5 * 2 * DT) for _ in range(5)))
                np.testing.assert_array_equal(parts.time.to_decimal(u.ms), expected.time.to_decimal(u.ms))
                np.testing.assert_array_equal(
                    parts.samples["g"].values.to_decimal(u.uS), expected.samples["g"].values.to_decimal(u.uS)
                )
                np.testing.assert_allclose(self._jumps((parts,))[0], [0.375, 0.575, 0.675, 0.875, 1.075], rtol=1e-6)
                self.assertEqual(len(cell._run_loop_cache), 2)

    def test_a_traced_run_after_a_change_of_dt_counts_arrivals_from_time_zero(self):
        # Inside jax.jit the clock has no host value: the arrivals keep the
        # dt grid from time 0 (main's steps), each delivered once.
        cell = self._counting_cell()
        cell.run(dt=DT, duration=3 * DT)
        g = brainstate.transform.jit(
            lambda: cell.run(dt=2 * DT, duration=22 * 2 * DT).samples["g"].values.to_decimal(u.uS).reshape(-1)
        )()
        increments = np.diff(np.asarray(g, dtype=np.float64), prepend=0.0)
        jumps = np.flatnonzero(increments > 0.5)
        np.testing.assert_allclose(0.075 + 0.05 * jumps, [0.325, 0.525, 0.625, 0.875, 1.025], rtol=1e-6)
        np.testing.assert_array_equal(np.rint(increments[jumps]), [1, 1, 1, 1, 2])

    def test_an_update_loop_with_the_callers_t_delivers_at_the_host_steps(self):
        # A Cell.update() loop sets its own t = k*dt, so the arrivals are
        # compared with t / dt rounded in float32 (32010 steps, L77).
        with brainstate.environ.context(precision=32):
            cell = CellRunClockTest._cell()
            cell.init_state()
            (synapse,) = cell.runtime_objects(ExpSyn).values()

            def step(k):
                with brainstate.environ.context(t=k * DT):
                    cell.update()
                return synapse.g.value.to_decimal(u.uS).reshape(-1)[0]

            with brainstate.environ.context(dt=DT):
                g = brainstate.transform.jit(
                    lambda: brainstate.transform.for_loop(step, np.arange(CellRunClockTest.N))
                )()
        jumps = np.flatnonzero(np.diff(np.asarray(g), prepend=0.0) > 0.5 * WEIGHT.to_decimal(u.uS))
        np.testing.assert_array_equal(jumps, [4000, 16384, 32000])


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
        """The one-CV event cell, driven by ``NetStim(start, number=1)``."""
        cell, synapse = make_event_cell(solver, cv_per_branch=1)
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
            for start in (0.0 * u.ms, 1.0 * u.ms):
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

    def test_lumped_cell_update_loop_delivers_the_start_event(self):
        def cell():
            model = braincell.Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
            model.paint(AllRegion(), mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
            model.place(mech.Synapse(ExpSyn, name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
            braincell.connect(
                "drive",
                source=braincell.NetStim(start=0.0 * u.ms, number=1),
                synapse=model.synapses["syn"],
                weight=WEIGHT,
            )
            model.init_state()
            return model

        looped = cell()
        self._updates(looped, 0, self.N)
        ran = cell()
        ran.run(dt=DT, duration=self.N * DT)
        for actual, expected in ((looped, ran), (_synapse(looped), _synapse(ran))):
            np.testing.assert_allclose(_state(actual), _state(expected), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(_state(_synapse(ran)), 0.005 * np.exp(-2.0 / 2.0), rtol=1e-9)

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


AMPLITUDES = np.asarray([0.0, 0.05, 0.1]) * u.nA


def _hh_cell(kind, n=3, *, amplitudes=AMPLITUDES, V_th=0.0 * u.mV):
    """A Hodgkin-Huxley population that spikes under a 1-19 ms clamp (CELL-26).

    ``kind="lumped"``: ``Cell.point(n, length=20 um, radius=10 um)`` on rk4;
    ``kind="cable"``: one 60 um x 5 um branch, ``CVPerBranch(3)``, staggered.
    SodiumFixed E 50 mV, PotassiumFixed E -77 mV, Na_HH1952 120 mS/cm^2,
    K_HH1952 36 mS/cm^2, IL 0.3 mS/cm^2 at -54.3 mV, V_init -65 mV.
    ``amplitudes=None`` places no clamp.
    """
    if kind == "lumped":
        cell = braincell.Cell.point(
            n, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, V_th=V_th, solver="rk4"
        )
    else:
        branch = braincell.Branch.from_lengths(lengths=[60.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        cell = braincell.Cell(
            braincell.Morphology.from_root(branch),
            pop_size=n,
            cv_policy=braincell.CVPerBranch(3),
            V_init=-65.0 * u.mV,
            V_th=V_th,
        )
    _paint_hh(cell)
    if amplitudes is not None:
        cell.place(
            RootLocation(0.5),
            mech.CurrentClamp(delay=1.0 * u.ms, durations=18.0 * u.ms, amplitudes=amplitudes[:n]),
        )
    return cell


def _paint_hh(cell):
    cell.paint(
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )


def _reduced_cell():
    """A PayloadAccumulatorReduction (threshold 0.1 uS) with an ``input`` ExpSyn, not connected."""
    cell = _cell(1)
    cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="input"))
    cell.add_reduction("payload", braincell.PayloadAccumulatorReduction(threshold=0.1 * u.uS))
    cell.use_model("payload")
    return cell


def _netstim():
    return braincell.NetStim(start=0.3 * u.ms, interval=1.0 * u.ms, number=3)


def _ms(series):
    return np.asarray(series.time.to_decimal(u.ms))


def assert_same_events(actual, expected):
    """Event rows equal: integer columns exactly, times to rtol 1e-12."""
    np.testing.assert_array_equal(actual.source_id, expected.source_id)
    np.testing.assert_array_equal(actual.count, expected.count)
    np.testing.assert_allclose(_ms(actual), _ms(expected), rtol=1e-12, atol=0.0)


class RunSpikesTest(unittest.TestCase):
    """CELL-26: ``RunResult.spikes`` carries the Cell's spikes as ``NetworkResult.events`` would."""

    DURATION = 20.0 * u.ms

    def setUp(self):
        self.enterContext(_float64())

    def _alone_and_in_network(self, build, *, dt=DT, duration=DURATION, stim=None):
        cell = build()
        if stim is not None:
            braincell.connect("drive", source=stim(), synapse=cell.synapses["input"], weight=0.2 * u.uS)
        alone = cell.run(dt=dt, duration=duration).spikes
        network = braincell.Network()
        post = network.add_population("post", build())
        if stim is not None:
            source = network.add_population("stim", stim())
            network.connect("drive", source=source, synapse=post.synapses["input"], weight=0.2 * u.uS)
        events = network.run(dt=dt, duration=duration).events["post"]["spike"]
        return alone, events

    def test_spikes_match_network_events(self):
        cases = {
            "lumped": (lambda: _hh_cell("lumped"), {}, ([3.175, 4.525, 19.425], [2, 1, 2])),
            "cable": (lambda: _hh_cell("cable"), {}, ([3.875, 6.325], [2, 1])),
            "reduction": (
                _reduced_cell,
                dict(dt=0.1 * u.ms, duration=4.0 * u.ms, stim=_netstim),
                ([0.3, 1.3, 2.3], [0, 0, 0]),
            ),
        }
        for name, (build, options, (times, ids)) in cases.items():
            with self.subTest(name):
                alone, events = self._alone_and_in_network(build, **options)
                assert_same_events(alone, events)
                self.assertEqual(dict(alone.metadata).keys(), {"port", "cv_id"})
                self.assertEqual(alone.metadata["port"], events.metadata["port"])
                np.testing.assert_array_equal(alone.metadata["cv_id"], events.metadata["cv_id"])
                # This machine's values (2b02a98); the comparison above is the contract.
                np.testing.assert_allclose(_ms(alone), times, rtol=1e-9)
                np.testing.assert_array_equal(alone.source_id, ids)
                np.testing.assert_array_equal(alone.count, np.ones(len(ids)))

    def test_events_sit_at_the_step_whose_end_crosses_the_threshold(self):
        for kind, V_th in (("lumped", 0.0), ("cable", 0.0), ("lumped", -20.0)):
            with self.subTest(kind=kind, V_th=V_th):
                cell = _hh_cell(kind, V_th=V_th * u.mV)
                cell.loc(RootLocation(0.5)).record("v", "V")
                result = cell.run(dt=DT, duration=self.DURATION)
                v = np.asarray(result.samples["v"].values.to_decimal(u.mV))
                t = np.asarray(result.time.to_decimal(u.ms))
                rows = result.samples["v"].schema.rows
                for member in range(3):
                    (column,) = [i for i, row in enumerate(rows) if row.population_index == member]
                    crossed = np.flatnonzero((v[:-1, column] < V_th) & (v[1:, column] >= V_th))
                    spikes = result.spikes
                    np.testing.assert_allclose(_ms(spikes)[spikes.source_id == member], t[crossed], rtol=1e-12)
                    self.assertLessEqual(len(crossed), 3)
                self.assertGreater(len(result.spikes.source_id), 0)

    def test_members_beyond_the_first_byte_report_their_own_crossings(self):
        # Each step packs eight members to a byte; 20 members span three bytes.
        drive = np.linspace(0.0, 0.2, 20) * u.nA
        cell = _hh_cell("lumped", n=20, amplitudes=None)
        cell.add_current_input("drive", lambda V: drive)
        cell.loc(RootLocation(0.5)).record("v", "V")
        result = cell.run(dt=DT, duration=self.DURATION)
        v = np.asarray(result.samples["v"].values.to_decimal(u.mV))
        crossed = (v[:-1] < 0.0) & (v[1:] >= 0.0)
        rows = [row.population_index for row in result.samples["v"].schema.rows]
        step, column = np.nonzero(crossed)
        order = np.lexsort((np.asarray(rows)[column], step))
        spikes = result.spikes
        np.testing.assert_array_equal(spikes.source_id, np.asarray(rows)[column][order])
        np.testing.assert_allclose(_ms(spikes), np.asarray(result.time.to_decimal(u.ms))[step[order]], rtol=1e-12)
        self.assertGreater(len(set(spikes.source_id.tolist()) & set(range(16, 20))), 0)

    def test_a_lower_threshold_fires_earlier(self):
        at_zero = _hh_cell("lumped").run(dt=DT, duration=self.DURATION).spikes
        at_minus_20 = _hh_cell("lumped", V_th=-20.0 * u.mV).run(dt=DT, duration=self.DURATION).spikes
        np.testing.assert_array_equal(at_minus_20.source_id, at_zero.source_id)
        self.assertTrue(np.all(_ms(at_minus_20) < _ms(at_zero)))

    def test_batch_rows_are_batch_major_with_their_member_and_batch(self):
        drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
        cell = _hh_cell("lumped", amplitudes=None)
        cell.add_current_input("drive", lambda V: drive)
        cell.init_state(batch_size=2)
        batched = cell.run(dt=DT, duration=self.DURATION).spikes
        batch_index = batched.metadata["batch_index"]
        population_index = batched.metadata["population_index"]
        np.testing.assert_array_equal(batch_index, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(population_index, [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(batched.metadata["cv_id"], np.zeros(6))
        for b in range(2):
            single = _hh_cell("lumped", amplitudes=None)
            single.add_current_input("drive", lambda V, b=b: drive[b])
            expected = single.run(dt=DT, duration=self.DURATION).spikes
            rows = batch_index[batched.source_id] == b
            np.testing.assert_array_equal(population_index[batched.source_id][rows], expected.source_id)
            np.testing.assert_array_equal(batched.source_id[rows], 3 * b + expected.source_id)
            np.testing.assert_allclose(_ms(batched)[rows], _ms(expected), rtol=1e-12)
            self.assertGreater(len(expected.source_id), 0)

    def test_a_batched_cable_cell_reports_each_batch(self):
        # One member: staggered with pop_size > 1 and a batch fails on main (a separate fix).
        cell = _hh_cell("cable", n=1, amplitudes=np.asarray([0.3]) * u.nA)
        cell.init_state(batch_size=2)
        batched = cell.run(dt=DT, duration=self.DURATION).spikes
        expected = _hh_cell("cable", n=1, amplitudes=np.asarray([0.3]) * u.nA).run(dt=DT, duration=self.DURATION).spikes
        self.assertGreater(len(expected.source_id), 0)
        for b in range(2):
            rows = batched.metadata["batch_index"][batched.source_id] == b
            np.testing.assert_array_equal(batched.source_id[rows], np.full(len(expected.source_id), b))
            np.testing.assert_allclose(_ms(batched)[rows], _ms(expected), rtol=1e-12)

    def test_a_run_split_at_a_spike_step_concatenates_to_one_run(self):
        for kind in ("lumped", "cable"):
            with self.subTest(kind):
                whole = _hh_cell(kind).run(dt=DT, duration=self.DURATION)
                first_step = int(round(_ms(whole.spikes)[0] / 0.025))
                cell = _hh_cell(kind)
                parts = (cell.run(dt=DT, duration=first_step * DT), cell.run(dt=DT, duration=(800 - first_step) * DT))
                self.assertEqual(len(parts[0].spikes.source_id), 0)
                np.testing.assert_allclose(_ms(parts[1].spikes)[0], first_step * 0.025, rtol=1e-12)
                assert_same_events(RunResult.concat(parts).spikes, whole.spikes)

    def test_concat_without_spikes_in_a_part_has_none(self):
        cell = _hh_cell("lumped")
        first = cell.run(dt=DT, duration=1.0 * u.ms)
        second = cell.run(dt=DT, duration=1.0 * u.ms)
        stripped = RunResult(
            time=second.time,
            traces=second.traces,
            samples=second.samples,
            start_time=second.start_time,
            stop_time=second.stop_time,
            dt=second.dt,
        )
        self.assertIsNotNone(RunResult.concat((first, second)).spikes)
        self.assertIsNone(RunResult.concat((first, stripped)).spikes)

    def test_reset_state_gives_the_same_spikes_again(self):
        cell = _hh_cell("lumped")
        first = cell.run(dt=DT, duration=self.DURATION).spikes
        cell.reset_state()
        assert_same_events(cell.run(dt=DT, duration=self.DURATION).spikes, first)

    def test_a_new_grid_after_reset_moves_the_port(self):
        cell = _hh_cell("cable")
        self.assertEqual(cell.run(dt=DT, duration=1.0 * u.ms).spikes.metadata["cv_id"].tolist(), [1, 1, 1])
        cell.reset()
        cell.cv_policy = braincell.CVPerBranch(5)
        cell.init_state()
        spikes = cell.run(dt=DT, duration=self.DURATION).spikes
        self.assertEqual(spikes.metadata["cv_id"].tolist(), [2, 2, 2])
        network = braincell.Network()
        twin = _hh_cell("cable")
        twin.cv_policy = braincell.CVPerBranch(5)
        network.add_population("post", twin)
        assert_same_events(spikes, network.run(dt=DT, duration=self.DURATION).events["post"]["spike"])

    def test_a_traced_run_has_no_spikes(self):
        cell = _hh_cell("lumped")
        cell.record("v", "V")
        cell.init_state()
        seen = {}

        @brainstate.transform.jit
        def traced():
            result = cell.run(dt=DT, duration=1.0 * u.ms)
            seen["spikes"] = result.spikes
            return result.samples["v"].values

        self.assertEqual(traced().shape, (40, 3))
        self.assertIsNone(seen["spikes"])
        self.assertIsNotNone(cell.run(dt=DT, duration=1.0 * u.ms).spikes)

    def test_a_two_dimensional_population_flattens_in_c_order(self):
        drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
        grid = braincell.Cell.point((2, 3), length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
        flat = _hh_cell("lumped", n=6, amplitudes=None)
        _paint_hh(grid)
        grid.add_current_input("drive", lambda V: drive)
        flat.add_current_input("drive", lambda V: drive.reshape(6))
        on_grid = grid.run(dt=DT, duration=self.DURATION).spikes
        assert_same_events(on_grid, flat.run(dt=DT, duration=self.DURATION).spikes)
        self.assertEqual(on_grid.metadata["cv_id"].shape, (6,))
        self.assertGreater(len(on_grid.source_id), 0)


class RunResultPytreeTest(unittest.TestCase):
    """CELL-26: a RunResult is a JAX pytree whose leaves are its arrays."""

    def _result(self):
        cell = _hh_cell("lumped")
        cell.record("v", "V")
        return cell, cell.run(dt=DT, duration=5.0 * u.ms)

    def test_leaves_hold_the_arrays_and_rebuild_the_result(self):
        cell, result = self._result()
        leaves, treedef = jax.tree.flatten(result)
        self.assertTrue(any(leaf is u.get_mantissa(result.samples["v"].values) for leaf in leaves))
        self.assertTrue(any(leaf is result.spikes.source_id for leaf in leaves))
        rebuilt = jax.tree.unflatten(treedef, leaves)
        self.assertIsInstance(rebuilt, RunResult)
        self.assertEqual(tuple(rebuilt.samples), ("v",))
        self.assertEqual(rebuilt.samples["v"].schema, result.samples["v"].schema)
        with self.assertRaises(TypeError):
            rebuilt.samples["w"] = rebuilt.samples["v"]
        assert_same_events(rebuilt.spikes, result.spikes)
        joined = RunResult.concat((rebuilt, cell.run(dt=DT, duration=5.0 * u.ms)))
        self.assertEqual(joined.time.shape, (400,))

    def test_tree_utilities_accept_results(self):
        _, result = self._result()
        self.assertIsInstance(jax.block_until_ready(result), RunResult)
        copied = jax.tree.map(lambda x: x, result)
        np.testing.assert_array_equal(
            copied.samples["v"].values.to_decimal(u.mV), result.samples["v"].values.to_decimal(u.mV)
        )
        self.assertEqual(np.shape(jax.tree.leaves(result.samples["v"])[0]), (200, 3))
        self.assertIsInstance(jax.tree.map(lambda a, b: a, result, result), RunResult)

    def test_a_jitted_function_returns_a_result(self):
        cell = _hh_cell("lumped")
        cell.record("v", "V")
        cell.init_state()
        result = brainstate.transform.jit(lambda: cell.run(dt=DT, duration=1.0 * u.ms))()
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.samples["v"].values.shape, (40, 3))
        self.assertIsNone(result.spikes)


def _indent(line):
    return len(line) - len(line.lstrip())


def _step_loop_rewrites(cell):
    """Carries that ``cell``'s cached Cell.run loop sets to a constant or to another carry's input on every step.

    Reads the lowered StableHLO of the loop: in the body of its outer
    ``stablehlo.while``, a returned value that is neither the carry itself
    nor computed by a non-constant op of the body is a value every step
    writes over the carry; on GPU each such carry costs a copy per step
    (CELL-39). Returns ``(position, value)`` pairs.
    """
    (plan,) = cell._run_loop_cache.values()
    with brainstate.environ.context(dt=DT):
        lines = plan.loop.lower(cell._take_value_inputs()).as_text().splitlines()
    start = max(
        (index for index, line in enumerate(lines) if "stablehlo.while(" in line),
        key=lambda index: lines[index].count("%iterArg"),
    )
    indent = _indent(lines[start])
    carries = re.findall(r"(%iterArg\w*) = ", lines[start])
    body = next(i for i in range(start, len(lines)) if lines[i].strip() == "} do {" and _indent(lines[i]) == indent)
    computed = set()
    for line in lines[body + 1 :]:
        if _indent(line) != indent + 2:
            continue
        op = line.strip()
        if op.startswith("stablehlo.return "):
            returned = [value.strip() for value in op[len("stablehlo.return ") :].split(" : ")[0].split(",")]
            break
        match = re.match(r"(%\w+)(?::\d+)? = (\S+)", op)
        if match and match.group(2) != "stablehlo.constant":
            computed.add(match.group(1))
    return [
        (position, value)
        for position, value in enumerate(returned)
        if value != carries[position] and value.split("#")[0] not in computed
    ]


def _detector_cell():
    """``_hh_cell("lumped")`` whose -20 mV detector drives its own ExpSyn (weight 1e-6 uS, tau 1e6 ms)."""
    cell = _hh_cell("lumped")
    cell.place(RootLocation(0.5), mech.Synapse("ExpSyn", name="self", tau=1e6 * u.ms, e=-65.0 * u.mV))
    detector = braincell.VoltageCrossingSource(cell, threshold=-20.0 * u.mV)
    braincell.connect("self", source=detector, synapse=cell.synapses["self"], weight=1e-6 * u.uS)
    cell.record("V", "V")
    cell.record("g", braincell.observe.synapse(name="self").state("g"))
    return cell


class RunLoopCarryTest(unittest.TestCase):
    """CELL-39: the run loop carries no state that every step rewrites before reading it."""

    def test_no_carry_is_rewritten_by_every_step(self):
        cells = {"lumped": _hh_cell("lumped"), "cable": _hh_cell("cable"), "reduced": _reduced_cell()}
        cells["detector"] = _detector_cell()
        for name, cell in cells.items():
            with self.subTest(cell=name):
                cell.init_state()
                cell.run(dt=DT, duration=4 * DT)
                self.assertEqual(_step_loop_rewrites(cell), [])
        with self.subTest(cell="lumped with a value input"):
            # A value input is an argument of the loop, which then wraps each step to apply it on the first.
            cell = cells["lumped"]
            cell.add_current_input("pulse", 0.1 * u.nA)
            self.assertEqual(_step_loop_rewrites(cell), [])

    def test_a_detector_reads_the_voltage_before_each_step(self):
        with _float64():
            cell = _detector_cell()
            cell.init_state()
            result = cell.run(dt=DT, duration=20.0 * u.ms)
        V = np.asarray(result.samples["V"].values.to_decimal(u.mV))
        g = np.asarray(result.samples["g"].values.to_decimal(u.uS))
        # A crossing in the step from t_k is delivered at t_k + dt: g jumps between samples k and k + 1.
        crossings = (V[:-1] < -20.0) & (V[1:] >= -20.0)
        np.testing.assert_array_equal(np.diff(g, axis=0) > 0.0, crossings)
        np.testing.assert_array_equal(crossings.sum(axis=0), [0, 1, 2])

    def test_the_voltage_before_the_last_step_outlives_the_run(self):
        for kind in ("lumped", "cable"):
            with self.subTest(kind=kind), _float64():
                cell = _hh_cell(kind)
                cell.record("V", "V")
                cell.init_state()
                result = cell.run(dt=DT, duration=5.0 * u.ms)
                last = result.samples["V"].values[-1]
                np.testing.assert_array_equal(
                    np.asarray(cell._event_previous_V.value.to_decimal(u.mV)).reshape(-1),
                    np.asarray(last.to_decimal(u.mV)).reshape(-1),
                )


#: The spike output paths of ``Cell.run`` the tests run on this machine's
#: backend, by the module switches each sets (CELL-49). The first packs
#: the spike bits at every step; each other path must give its results.
SPIKE_PATHS = {"per step": {}}


@contextlib.contextmanager
def _spike_path(name):
    """Plans built inside take the spike output path ``name`` of :data:`SPIKE_PATHS`."""
    with contextlib.ExitStack() as stack:
        for switch, value in SPIKE_PATHS[name].items():
            stack.enter_context(mock.patch.object(run_module, switch, value))
        yield


def _high_firing_cell(n=100, solver="rk4"):
    """A population a quarter of whose rows spike at every step from step 2 (CELL-49).

    ``Cell.point(n, length=20 um, radius=10 um, V_init=-65 mV, V_th=-65 mV)``
    on ``solver`` with only ``IL`` (0.3 mS/cm^2, E -65 mV), driven by 1 nA
    sin(2 pi t / 0.1 ms + phi_m), phi_m = (m mod 4) pi / 2 + pi / 4: at
    dt = 0.025 ms a period is 4 steps, and on rk4 each member's V crosses
    -65 mV upward once in each. ``V`` is recorded as ``"V"``.
    """
    cell = braincell.Cell.point(
        n, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, V_th=-65.0 * u.mV, solver=solver
    )
    cell.paint(AllRegion(), mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV))
    phase = (np.arange(n) % 4) * (np.pi / 2) + np.pi / 4
    cell.add_current_input(
        "drive", lambda V: 1.0 * u.nA * u.math.sin(2 * np.pi * (brainstate.environ.get("t") / (0.1 * u.ms)) + phase)
    )
    cell.record("V", "V")
    return cell


def _clamped_hh_cell(n=100):
    """CELL-39's population: ``_hh_cell``'s lumped HH (rk4) under a 10 uA/cm^2 clamp from 5 to 45 ms."""
    area = 2.0 * u.math.pi * (10.0 * u.um) * (20.0 * u.um)
    cell = _hh_cell("lumped", n=n, amplitudes=None)
    cell.place(
        mech.CurrentClamp(
            delay=5.0 * u.ms, durations=40.0 * u.ms, amplitudes=(10.0 * u.uA / u.cm**2 * area).in_unit(u.nA)
        )
    )
    cell.record("V", "V")
    return cell


def _segments(*segments):
    """A ``run(cell)`` of the ``(steps, dt)`` (or ``steps`` at ``DT``) segments, one after another."""
    segments = tuple(segment if isinstance(segment, tuple) else (segment, DT) for segment in segments)

    def run(cell):
        return tuple(cell.run(dt=dt, duration=n * dt) for n, dt in segments)

    return run


def _outcome(build, run):
    """``(results, clock)``: ``run(cell)``'s results of a fresh ``build()`` and its clock after them."""
    cell = build()
    results = run(cell)
    return results, cell._clock_host()


def assert_same_runs(actual, expected):
    """Runs pairwise equal: spikes (time, source_id, count, metadata), times, recordings; floats to rtol 1e-12."""
    np.testing.assert_equal(len(actual), len(expected))
    for one, other in zip(actual, expected):
        _assert_same_run(one, other)


def _assert_same_run(actual, expected):
    assert_same_events(actual.spikes, expected.spikes)
    np.testing.assert_equal(dict(actual.spikes.metadata), dict(expected.spikes.metadata))
    np.testing.assert_allclose(_ms(actual), _ms(expected), rtol=1e-12, atol=0.0)
    np.testing.assert_equal(tuple(actual.samples), tuple(expected.samples))
    for name, block in expected.samples.items():
        values = actual.samples[name].values
        np.testing.assert_equal(u.get_unit(values), u.get_unit(block.values))
        np.testing.assert_allclose(u.get_mantissa(values), u.get_mantissa(block.values), rtol=1e-12, atol=0.0)


class SpikeOutputPathTest(unittest.TestCase):
    """CELL-49: each spike output path of ``Cell.run`` gives the result of packing the bits at every step."""

    def setUp(self):
        self.enterContext(_float64())

    def assert_paths_match(self, build, run):
        """Each path's ``run`` of a fresh ``build()`` gives the per-step path's results and clock; returns those."""
        with _spike_path("per step"):
            expected, clock = _outcome(build, run)
        for name in tuple(SPIKE_PATHS)[1:]:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock = _outcome(build, run)
                assert_same_runs(actual, expected)
                self.assertEqual(actual_clock, clock)
        return expected

    def test_a_quarter_of_the_high_firing_rows_spike_at_every_step(self):
        with _spike_path("per step"):
            spikes = _high_firing_cell().run(dt=DT, duration=64 * DT).spikes
        per_step = np.bincount(np.rint(_ms(spikes) / 0.025).astype(int), minlength=64)
        np.testing.assert_array_equal(per_step[2:], np.full(62, 25))

    def test_step_counts_with_and_without_a_block_size(self):
        # Blocks of 25, 32, 31 and 23 steps; 37 and 7 steps have no divisor in [8, 32].
        for n_steps in (2000, 64, 31, 69, 37, 7):
            with self.subTest(n_steps=n_steps):
                (expected,) = self.assert_paths_match(_high_firing_cell, _segments(n_steps))
                self.assertGreater(len(expected.spikes.source_id), 0)

    def test_the_clamped_population(self):
        for precision in (32, 64):
            context = contextlib.nullcontext() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                (expected,) = self.assert_paths_match(_clamped_hh_cell, _segments(2000))
                self.assertGreater(len(expected.spikes.source_id), 100)

    def test_a_run_split_in_blocks_of_other_sizes_matches_one_run(self):
        # 31 steps make one block of 31, 69 three of 23; one run of 100 makes four of 25.
        with _spike_path("per step"):
            expected, clock = _outcome(_high_firing_cell, _segments(100))
        for name in SPIKE_PATHS:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock = _outcome(_high_firing_cell, _segments(31, 69))
                assert_same_runs((RunResult.concat(actual),), expected)
                self.assertEqual(actual_clock, clock)

    def test_a_cable_and_a_batch(self):
        def batched():
            drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
            cell = _hh_cell("lumped", amplitudes=None)
            cell.add_current_input("drive", lambda V: drive)
            cell.init_state(batch_size=2)
            return cell

        def cable():
            cell = _hh_cell("cable")
            cell.loc(RootLocation(0.5)).record("v", "V")
            return cell

        for name, build in (("cable", cable), ("batch", batched)):
            with self.subTest(name):
                (expected,) = self.assert_paths_match(build, _segments(800))
                self.assertGreater(len(expected.spikes.source_id), 0)

    def test_a_reduction_model(self):
        def reduced():
            cell = _reduced_cell()
            braincell.connect("drive", source=_netstim(), synapse=cell.synapses["input"], weight=0.2 * u.uS)
            return cell

        (expected,) = self.assert_paths_match(reduced, _segments((40, 0.1 * u.ms)))
        np.testing.assert_allclose(_ms(expected.spikes), [0.3, 1.3, 2.3], rtol=1e-9)

    def test_a_value_input_acts_on_the_first_step(self):
        # A 20 nA pulse for one step of 0.025 ms lifts the soma (12.6 pF) by
        # about 40 mV: every member fires at once, in the first run only.
        def pulsed(cell):
            cell.add_current_input("pulse", 20.0 * u.nA)
            return (cell.run(dt=DT, duration=64 * DT), cell.run(dt=DT, duration=64 * DT))

        pulsed_run, later = self.assert_paths_match(lambda: _hh_cell("lumped", amplitudes=None), pulsed)
        np.testing.assert_array_equal(pulsed_run.spikes.source_id, [0, 1, 2])
        self.assertEqual(len(later.spikes.source_id), 0)

    def test_a_detector_reads_the_voltage_before_each_step(self):
        (expected,) = self.assert_paths_match(_detector_cell, _segments(800))
        g = np.asarray(expected.samples["g"].values.to_decimal(u.uS))
        self.assertEqual(int((np.diff(g, axis=0) > 0.0).sum()), 3)

    def test_the_arrivals_after_a_change_of_dt(self):
        # CELL-48's runs (CellArrivalStepTest): segments of 3 then 22 steps
        # (no block, then blocks of 22), of 10 then 20 (blocks of 10 and 20),
        # and of 3, 5 and 13 steps (no block).
        cases = {
            "off the new grid": ((0.06,) + CellArrivalStepTest.TIMES, ((3, DT), (22, 2 * DT))),
            "on the new grid": (CellArrivalStepTest.TIMES, ((10, DT), (20, 2 * DT))),
            "two changes": ((0.425, 0.455, 0.475, 0.6, 1.06, 1.375), ((3, DT), (5, 2 * DT), (13, 4 * DT))),
        }
        for name, (times, segments) in cases.items():
            with self.subTest(name):
                *_, last = self.assert_paths_match(
                    lambda times=times: CellArrivalStepTest._counting_cell(times), _segments(*segments)
                )
                self.assertGreater(float(np.asarray(last.samples["g"].values.to_decimal(u.uS)).max()), 4.5)


def _synapse(cell):
    """The one runtime ExpSyn of a cable or lumped cell."""
    (synapse,) = cell.runtime_objects(ExpSyn).values()
    return synapse


def _state(model):
    """A cell's ``V`` in mV or a synapse's ``g`` in uS, as a numpy array."""
    if isinstance(model, ExpSyn):
        return np.asarray(model.g.value.to_decimal(u.uS))
    return np.asarray(model.V.value.to_decimal(u.mV))


#: The solvers whose compiled run steps a cable through the axial operator.
AXIAL_SOLVERS = ("staggered", "rk4", "ralston2", "exp_euler")


def hh_y_cable(solver, pop_size=2):
    """A 3-branch Y cable, one CV per branch, firing within 1 ms.

    A 100 um trunk (radius 1 um) and two 100 um children (radius 0.5 um);
    HH sodium (120 mS/cm^2, E 50 mV), potassium (36 mS/cm^2, E -77 mV) and
    leak (0.3 mS/cm^2, E -54.3 mV) in every CV; 0.5 nA at the trunk midpoint
    from 0.1 ms. ``V`` is recorded as ``"v"``; the cell is not initialized.
    """
    trunk = braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
    morpho = braincell.Morphology.from_root(trunk, name="trunk")
    for name in ("left", "right"):
        child = braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[0.5, 0.5] * u.um, type="dendrite")
        morpho.attach(parent="trunk", child_branch=child, child_name=name)
    cell = braincell.Cell(
        morpho, cv_policy=braincell.CVPerBranch(), pop_size=pop_size, V_init=-65.0 * u.mV, solver=solver
    )
    cell.paint(
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(RootLocation(0.5), mech.CurrentClamp(delay=0.1 * u.ms, durations=10.0 * u.ms, amplitudes=0.5 * u.nA))
    cell.record("v", "V")
    return cell


def update_loop_voltage(cell, n_steps):
    """``V`` in mV before each of ``n_steps`` ``cell.update()`` calls in one compiled loop.

    A loop of the caller's own builds the axial operator inside its step,
    from the geometry it traces: the per-step reference for ``Cell.run``.
    """

    def step(index):
        with brainstate.environ.context(t=index * DT):
            voltage = cell.V.value.to_decimal(u.mV)
            cell.update()
        return voltage

    with brainstate.environ.context(dt=DT):
        return brainstate.transform.for_loop(step, np.arange(n_steps))


def axial_builds(run):
    """``(in a step, before the steps)``: the axial matrices ``run()`` assembles while tracing its loop.

    Every solver's operator starts from this node-tree matrix (the CV
    operator of the generic solvers, the DHS coefficients of
    ``"staggered"``); one assembled in the step body of a compiled loop is
    assembled again at every step.
    """
    counts = [0, 0]
    in_step = [0]
    for_loop = brainstate.transform.for_loop
    assemble = _staggered._build_node_tree_axial_matrix

    def counting_for_loop(*args, **kwargs):
        in_step[0] += 1
        try:
            return for_loop(*args, **kwargs)
        finally:
            in_step[0] -= 1

    def counting_assemble(*args, **kwargs):
        counts[0 if in_step[0] else 1] += 1
        return assemble(*args, **kwargs)

    with (
        mock.patch.object(brainstate.transform, "for_loop", counting_for_loop),
        mock.patch.object(_staggered, "_build_node_tree_axial_matrix", counting_assemble),
    ):
        run()
    return tuple(counts)


class AxialOperatorPerRunTest(unittest.TestCase):
    """CELL-50: a compiled run builds the axial operator once, before its steps.

    The geometry is the same at every step of a run, so the operator built
    from it before the loop gives every step the values that building it in
    the step gave: the run matches a loop of ``update()`` calls, which
    builds it in each step, values and gradients alike.
    """

    N = 40

    def setUp(self):
        self.enterContext(_float64())

    def test_run_matches_the_update_loop(self):
        for solver in AXIAL_SOLVERS:
            with self.subTest(solver=solver):
                ran = hh_y_cable(solver)
                ran.init_state()
                expected = np.asarray(ran.run(dt=DT, duration=self.N * DT).traces["v"].to_decimal(u.mV))
                looped = hh_y_cable(solver)
                looped.init_state()
                actual = np.asarray(update_loop_voltage(looped, self.N)).reshape(expected.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
                # The run fires, and the CVs differ: the axial currents matter.
                self.assertGreater(float(expected[-1].max()), 20.0)
                self.assertGreater(float(np.ptp(expected[-1])), 30.0)

    def test_the_operator_is_built_once_before_the_steps(self):
        # On main 75b6104 every step built it: once (staggered, exp_euler),
        # or at each stage (rk4 four times, ralston2 twice).
        for solver in AXIAL_SOLVERS:
            with self.subTest(solver=solver):
                cell = hh_y_cable(solver)
                cell.init_state()
                self.assertEqual(axial_builds(lambda: cell.run(dt=DT, duration=4 * DT)), (0, 1))
                # A later run reuses the compiled loop and builds nothing on the host.
                self.assertEqual(axial_builds(lambda: cell.run(dt=DT, duration=4 * DT)), (0, 0))
        with self.subTest(solver="rk4", run="traced"):
            cell = hh_y_cable("rk4")
            cell.init_state()
            run = brainstate.transform.jit(lambda: cell.run(dt=DT, duration=4 * DT).traces["v"])
            self.assertEqual(axial_builds(run), (0, 1))
        with self.subTest(cell="lumped"):
            # A lumped Cell has no axial operator.
            cell = _hh_cell("lumped")
            cell.init_state()
            self.assertEqual(axial_builds(lambda: cell.run(dt=DT, duration=4 * DT)), (0, 0))

    def test_a_run_that_fails_in_its_trace_leaves_no_operator_behind(self):
        cell = hh_y_cable("rk4")
        cell.init_state()
        with mock.patch.object(braincell.Cell, "compute_axial_derivative", side_effect=RuntimeError("stage failed")):
            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                cell.run(dt=DT, duration=4 * DT)
        self.assertIsNone(cell.runtime.run_axial_operator)
        cell.reset_state()
        expected = hh_y_cable("rk4")
        expected.init_state()
        np.testing.assert_allclose(
            cell.run(dt=DT, duration=4 * DT).traces["v"].to_decimal(u.mV),
            expected.run(dt=DT, duration=4 * DT).traces["v"].to_decimal(u.mV),
            rtol=1e-12,
            atol=0.0,
        )

    def test_gradients_through_run_match_the_update_loop(self):
        # Sodium g_max does not reach the operator; Ra does, through the
        # operator built once in the run and in every step of the loop.
        def gradient(solver, through, field):
            cell = hh_y_cable(solver)
            cell.init_state()
            source = braincell.trainable.parameter(group_by="all", name=field)
            if field == "g_max":
                cell.channels["na"].trainable(g_max=source)
            else:
                cell.geometry.Ra.trainable(source)

            def final_voltage():
                cell.reset_state()
                if through == "run":
                    cell.run(dt=DT, duration=self.N * DT)
                else:
                    update_loop_voltage(cell, self.N)
                return u.math.sum(cell.V.value.to_decimal(u.mV))

            states = cell.trainables.parameters().states()
            (value,) = brainstate.transform.grad(final_voltage, grad_states=states)().values()
            return np.asarray(u.get_mantissa(value)).reshape(-1)

        for solver in ("staggered", "rk4"):
            for field in ("g_max", "Ra"):
                with self.subTest(solver=solver, field=field):
                    expected = gradient(solver, "loop", field)
                    np.testing.assert_allclose(gradient(solver, "run", field), expected, rtol=1e-10, atol=0.0)
                    self.assertTrue(np.all(expected != 0.0))


if __name__ == "__main__":
    unittest.main()
