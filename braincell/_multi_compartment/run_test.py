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
from fractions import Fraction
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


def _boundary_increments(cell, segments, read):
    """Run ``segments`` (``(dt in ms, steps)``, ...) one step per ``Cell.run``; ``{boundary in ms: increase of read}``.

    ``read(cell)`` is read after every step (L93), so each step's increase is
    the boundary it started at.
    """
    increments, before = {}, read(cell)
    for dt_ms, n_steps in segments:
        for _ in range(n_steps):
            t = round(float(np.asarray(cell.current_time.to_decimal(u.ms), dtype=np.float64)), 6)
            cell.run(dt=dt_ms * u.ms, duration=dt_ms * u.ms)
            after = read(cell)
            if not np.isclose(after, before, rtol=0.0, atol=1e-3):
                increments[t] = increments.get(t, 0.0) + after - before
            before = after
    return {t: int(np.rint(value)) for t, value in increments.items()}


def _synapse_g(cell):
    """The ExpSyn's g in uS (1 per arrival on the counting cells)."""
    (synapse,) = cell.runtime_objects(ExpSyn).values()
    return float(np.asarray(synapse.g.value.to_decimal(u.uS), dtype=np.float64).reshape(-1)[0])


@contextlib.contextmanager
def _rule_updates():
    """Record ``(boundary in ms, pre arrivals, post events)`` of every PairSTDP update that runs."""
    from braincell.plasticity import PairSTDP

    records, update = [], PairSTDP.update

    def recording(self, state, *, pre, post, dt):
        t = brainstate.environ.get("t").to_decimal(u.ms)
        jax.debug.callback(
            lambda *values: records.append(tuple(float(np.asarray(value)) for value in values)),
            t,
            u.math.sum(pre),
            u.math.sum(post),
            ordered=True,
        )
        return update(self, state, pre=pre, post=post, dt=dt)

    with mock.patch.object(PairSTDP, "update", recording):
        yield records


def _rule_cell(times):
    """The counting cell's synapse fed through a PairSTDP rule (no decay) whose post events are ``times`` too.

    Returns ``(cell, (source, post events))``.
    """
    cell = braincell.Cell(
        braincell.Morphology.from_root(braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)),
        cv_policy=braincell.CVPerBranch(1),
        V_init=-65.0 * u.mV,
    )
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at(0, 0.5), mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV))
    events = braincell.EventSequence.from_times([np.asarray(times) * u.ms])
    post = braincell.EventSequence.from_times([np.asarray(times) * u.ms])
    rule = mech.ConnectionRule(
        "PairSTDP",
        A_plus=1e-6 * u.uS,
        A_minus=1e-6 * u.uS,
        tau_plus=1e9 * u.ms,
        tau_minus=1e9 * u.ms,
        signals={"post": post},
    )
    braincell.connect("learn", source=events, synapse=cell.synapses["s"], weight=1.0 * u.uS, plasticity=rule)
    return cell, (events, post)


class CellDtChangeBoundaryTest(unittest.TestCase):
    """CELL-55: where ``Cell.run`` changes dt, each event arrives once, on the grid of the run that delivers it.

    Each step prepares the boundary at its start, so the run at the old dt
    took the events of its boundaries' windows, up to half an old step
    before the time reached, t_c. The run at the new dt prepares t_c and
    delivers the others on its own grid (CELL-48), none before t_c. The
    expected boundaries are worked out by hand below; each comment also
    gives main bbb7b0a's counts, where the old run prepared t_c at its end
    and the new one kept that boundary (CELL-57).
    """

    #: name -> (event times in ms, segments ((dt in ms, steps), ...), {boundary in ms: events}).
    CASES = {
        # dt grows at 0.25 ms. The 0.025 ms run's last boundary, 0.225 ms,
        # takes [0.2125, 0.2375); 0.2375 ms is a tie, 9.5 steps, so step 10.
        # The 0.05 ms run prepares 0.25 ms with [0.225, 0.275): 0.2375, 0.26,
        # 0.265 and 0.27 ms, as on that grid from the start; 0.28 and 0.3 ms
        # act at 0.3 ms. main: {0.25: 2, 0.3: 4}.
        "dt grows": ([0.2375, 0.26, 0.265, 0.27, 0.28, 0.3], ((0.025, 10), (0.05, 10)), {0.25: 4, 0.3: 2}),
        # dt shrinks at 0.25 ms. The 0.05 ms run's last boundary, 0.2 ms, takes
        # [0.175, 0.225). The 0.025 ms run's first boundary, 0.25 ms, takes
        # [0.2375, 0.2625) and the earlier arrivals no run took: 0.23 and 0.26
        # ms. 0.2625 ms is a tie, 10.5 steps: with 0.27 and 0.28 ms it acts at
        # 0.275 ms; 0.3 ms at 0.3 ms. main: {0.25: 4, 0.275: 1, 0.3: 1}.
        "dt shrinks": ([0.23, 0.26, 0.27, 0.2625, 0.28, 0.3], ((0.05, 5), (0.025, 10)), {0.25: 2, 0.275: 3, 0.3: 1}),
        # As above with 0.245 ms, in the half new step before the change
        # (L94): 9.8 steps of 0.025 ms, so 0.25 ms. main: {0.25: 5, 0.275: 1,
        # 0.3: 1}.
        "dt shrinks, an event just before": (
            [0.23, 0.245, 0.26, 0.27, 0.2625, 0.28, 0.3],
            ((0.05, 5), (0.025, 10)),
            {0.25: 3, 0.275: 3, 0.3: 1},
        ),
        # dt grows at 0.075 ms, off the 0.05 ms grid: the new steps count from
        # 0.075 ms (CELL-48). The 0.025 ms run's last boundary, 0.05 ms, takes
        # [0.0375, 0.0625). The 0.05 ms run's first boundary, 0.075 ms, takes
        # [0.05, 0.1): 0.065, 0.08, 0.09 and 0.095 ms. 0.1 ms is a tie, 0.5
        # steps on: with 0.13 ms (1.1 steps) it acts at 0.125 ms. main:
        # {0.075: 2, 0.125: 4}.
        "dt grows off the grid": (
            [0.065, 0.08, 0.09, 0.095, 0.1, 0.13],
            ((0.025, 3), (0.05, 6)),
            {0.075: 4, 0.125: 2},
        ),
        # Two changes. The 0.05 ms run's last boundary, 0.2 ms, takes
        # [0.175, 0.225). One 0.0125 ms step prepares 0.25 ms with [0.24375,
        # 0.25625) and the earlier arrivals: 0.24 and 0.255 ms. From 0.2625
        # ms, off the 0.025 ms grid, its first boundary takes [0.25625,
        # 0.275): 0.265 and 0.27 ms; 0.276 and 0.28 ms are 0.54 and 0.7 steps
        # on (0.2875 ms); 0.3 ms is a tie at 1.5 steps (0.3125 ms). main:
        # {0.25: 4, 0.2875: 2, 0.3125: 1}.
        "two changes": (
            [0.24, 0.255, 0.265, 0.27, 0.276, 0.28, 0.3],
            ((0.05, 5), (0.0125, 1), (0.025, 6)),
            {0.25: 2, 0.2625: 2, 0.2875: 2, 0.3125: 1},
        ),
    }

    def _contexts(self):
        return ((32, brainstate.environ.context(precision=32)), (64, _float64()))

    def test_each_event_arrives_once_at_the_boundary_of_the_rule(self):
        for name, (times, segments, expected) in self.CASES.items():
            for precision, context in self._contexts():
                with self.subTest(case=name, precision=precision), context:
                    cell = CellArrivalStepTest._counting_cell(times)
                    cell.init_state()
                    got = _boundary_increments(cell, segments, _synapse_g)
                    self.assertEqual(got, expected)
                    self.assertAlmostEqual(_synapse_g(cell), len(times), places=4)

    def test_one_run_per_segment_delivers_as_one_step_per_run(self):
        # The second segment as one run gives what one step per run gives,
        # bitwise; in both the new run prepares the boundary at the change.
        for name, (times, segments, expected) in self.CASES.items():
            for precision, context in self._contexts():
                with self.subTest(case=name, precision=precision), context:
                    stepped = CellArrivalStepTest._counting_cell(times)
                    stepped.init_state()
                    _boundary_increments(stepped, segments, lambda c: 0.0)
                    whole = CellArrivalStepTest._counting_cell(times)
                    for dt_ms, n_steps in segments:
                        whole.run(dt=dt_ms * u.ms, duration=n_steps * dt_ms * u.ms)
                    self.assertEqual(_synapse_g(whole), _synapse_g(stepped))
                    np.testing.assert_array_equal(whole.V.value.mantissa, stepped.V.value.mantissa)

    def test_a_rule_updates_once_per_boundary_and_receives_each_event_once(self):
        # Scheduled rule rows and the rule's post events follow the same rule
        # as the other rows. Each boundary the runs prepare updates the rule
        # once; its arrivals are those of the boundary.
        for name in ("dt grows", "dt shrinks"):
            times, segments, expected = self.CASES[name]
            for precision, context in self._contexts():
                with self.subTest(case=name, precision=precision), context, _rule_updates() as updates:
                    cell, _ = _rule_cell(times)
                    for dt_ms, n_steps in segments:
                        cell.run(dt=dt_ms * u.ms, duration=n_steps * dt_ms * u.ms)
                    jax.effects_barrier()
                    # Each run prepares the boundaries its steps start at, 0
                    # to n - 1; the time it reaches is the next run's.
                    prepared, t_ms = [], 0.0
                    for dt_ms, n_steps in segments:
                        prepared += [round(t_ms + k * dt_ms, 6) for k in range(n_steps)]
                        t_ms += n_steps * dt_ms
                    self.assertEqual([round(t, 6) for t, _, _ in updates], prepared)
                    pre = {round(t, 6): int(n) for t, n, _ in updates if n}
                    post = {round(t, 6): int(n) for t, _, n in updates if n}
                    self.assertEqual(pre, expected)
                    self.assertEqual(post, expected)
                    x_pre = u.get_mantissa(cell.connections["learn"].rule_state("x_pre"))
                    self.assertAlmostEqual(float(np.asarray(x_pre).reshape(-1)[0]), len(times), places=4)

    def test_a_sum_count_synapse_counts_each_event_once(self):
        # ExpSynSTD ("sum_count", U = 0.5, no recovery) keeps x = 0.5**K
        # after K events, so each step's events are log2(x before / x after).
        for name in ("dt grows", "dt shrinks"):
            times, segments, expected = self.CASES[name]
            for precision, context in self._contexts():
                with self.subTest(case=name, precision=precision), context:
                    cell = CellArrivalStepTest._counting_cell(times)
                    cell.place(
                        at(0, 0.5),
                        mech.Synapse("ExpSynSTD", name="d", tau=1e9 * u.ms, U=0.5, tau_rec=1e9 * u.ms),
                    )
                    events = braincell.EventSequence.from_times([np.asarray(times) * u.ms])
                    braincell.connect("d", source=events, synapse=cell.synapses["d"], weight=1.0 * u.uS)
                    cell.init_state()
                    (depressing,) = cell.runtime_objects(braincell.synapse.ExpSynSTD).values()

                    def events_so_far(c):
                        x = np.asarray(u.get_mantissa(depressing.x.value), dtype=np.float64).reshape(-1)[0]
                        return -np.log2(float(x))

                    got = _boundary_increments(cell, segments, events_so_far)
                    self.assertEqual(got, expected)
                    self.assertAlmostEqual(events_so_far(cell), len(times), places=4)


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
#: "blocked" packs them every K steps where the step count has a divisor
#: K in [8, 32] (and at every step otherwise); "compacted" also copies the
#: grid back compacted, with room for all of it.
SPIKE_PATHS = {
    "per step": {"_BLOCKED_SPIKE_OUTPUT": False, "_COMPACT_SPIKE_COPY": False},
    "blocked": {"_BLOCKED_SPIKE_OUTPUT": True, "_COMPACT_SPIKE_COPY": False},
    "compacted": {
        "_BLOCKED_SPIKE_OUTPUT": True,
        "_COMPACT_SPIKE_COPY": True,
        "_COMPACT_MIN_BYTES": 0,
        "_COMPACT_CAPACITY_DIVISOR": 1,
    },
}


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
    """``(results, clock, ports)``: ``run(cell)``'s results of a fresh ``build()``, its clock, its plans' spike ports."""
    cell = build()
    with mock.patch.object(run_module, "_grid_events", wraps=run_module._grid_events) as whole:
        results = run(cell)
    ports = tuple(plan.spike_port for plan in cell._run_loop_cache.values())
    if any(port.capacity for port in ports):
        # A compacted copy with room for all of the grid is decoded as it is.
        np.testing.assert_equal(whole.call_count, 0)
    return results, cell._clock_host(), ports


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
        """Each path's ``run`` of a fresh ``build()`` gives the per-step path's results and clock.

        Returns those results and the spike ports of each path's plans.
        """
        ports = {}
        with _spike_path("per step"):
            expected, clock, ports["per step"] = _outcome(build, run)
        for name in tuple(SPIKE_PATHS)[1:]:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock, ports[name] = _outcome(build, run)
                assert_same_runs(actual, expected)
                self.assertEqual(actual_clock, clock)
        self.assertTrue(all(port.block_steps is None and port.capacity is None for port in ports["per step"]))
        return expected, ports

    def test_a_quarter_of_the_high_firing_rows_spike_at_every_step(self):
        with _spike_path("per step"):
            spikes = _high_firing_cell().run(dt=DT, duration=64 * DT).spikes
        per_step = np.bincount(np.rint(_ms(spikes) / 0.025).astype(int), minlength=64)
        np.testing.assert_array_equal(per_step[2:], np.full(62, 25))

    def test_step_counts_with_and_without_a_block_size(self):
        # Blocks of 25, 32, 31 and 23 steps; 37 and 7 steps have no divisor in [8, 32].
        for n_steps, block_steps in ((2000, 25), (64, 32), (31, 31), (69, 23), (37, None), (7, None)):
            with self.subTest(n_steps=n_steps):
                (expected,), ports = self.assert_paths_match(_high_firing_cell, _segments(n_steps))
                self.assertGreater(len(expected.spikes.source_id), 0)
                for name in ("blocked", "compacted"):
                    (port,) = ports[name]
                    self.assertEqual(port.block_steps, block_steps)
                self.assertEqual(ports["compacted"][0].capacity, -(-n_steps * 13 // 4))

    def test_the_clamped_population(self):
        for precision in (32, 64):
            context = contextlib.nullcontext() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                (expected,), _ = self.assert_paths_match(_clamped_hh_cell, _segments(2000))
                self.assertGreater(len(expected.spikes.source_id), 100)

    def test_a_run_split_in_blocks_of_other_sizes_matches_one_run(self):
        # 31 steps make one block of 31, 69 three of 23; one run of 100 makes four of 25.
        with _spike_path("per step"):
            expected, clock, _ = _outcome(_high_firing_cell, _segments(100))
        for name in SPIKE_PATHS:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock, _ = _outcome(_high_firing_cell, _segments(31, 69))
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
                (expected,), _ = self.assert_paths_match(build, _segments(800))
                self.assertGreater(len(expected.spikes.source_id), 0)

    def test_a_reduction_model(self):
        def reduced():
            cell = _reduced_cell()
            braincell.connect("drive", source=_netstim(), synapse=cell.synapses["input"], weight=0.2 * u.uS)
            return cell

        (expected,), ports = self.assert_paths_match(reduced, _segments((40, 0.1 * u.ms)))
        np.testing.assert_allclose(_ms(expected.spikes), [0.3, 1.3, 2.3], rtol=1e-9)
        # A reduction model's event counts are not packed: each step stores them.
        for name in SPIKE_PATHS:
            (port,) = ports[name]
            self.assertEqual((port.packed, port.block_steps, port.capacity), (False, None, None))

    def test_a_value_input_acts_on_the_first_step(self):
        # A 20 nA pulse for one step of 0.025 ms lifts the soma (12.6 pF) by
        # about 40 mV: every member fires at once, in the first run only.
        def pulsed(cell):
            cell.add_current_input("pulse", 20.0 * u.nA)
            return (cell.run(dt=DT, duration=64 * DT), cell.run(dt=DT, duration=64 * DT))

        (pulsed_run, later), _ = self.assert_paths_match(lambda: _hh_cell("lumped", amplitudes=None), pulsed)
        np.testing.assert_array_equal(pulsed_run.spikes.source_id, [0, 1, 2])
        self.assertEqual(len(later.spikes.source_id), 0)

    def test_a_detector_reads_the_voltage_before_each_step(self):
        (expected,), _ = self.assert_paths_match(_detector_cell, _segments(800))
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
                (*_, last), _ = self.assert_paths_match(
                    lambda times=times: CellArrivalStepTest._counting_cell(times), _segments(*segments)
                )
                self.assertGreater(float(np.asarray(last.samples["g"].values.to_decimal(u.uS)).max()), 4.5)

    def test_a_count_over_the_capacity_copies_the_whole_grid(self):
        # Room for one of the 4-byte groups: 64 steps of the high-firing grid have about 200 nonzero.
        with _spike_path("per step"):
            expected, _, _ = _outcome(_high_firing_cell, _segments(64))
        with (
            _spike_path("compacted"),
            mock.patch.object(run_module, "_COMPACT_CAPACITY_DIVISOR", 10**9),
            mock.patch.object(run_module, "_grid_events", wraps=run_module._grid_events) as whole,
        ):
            cell = _high_firing_cell()
            actual = cell.run(dt=DT, duration=64 * DT)
        (plan,) = cell._run_loop_cache.values()
        port = plan.spike_port
        self.assertEqual((port.block_steps, port.capacity), (32, 1))
        self.assertEqual(whole.call_count, 1)
        assert_same_runs((actual,), expected)

    def test_a_grid_of_1_mib_is_compacted(self):
        # 100 rows pack into 13 bytes a step; 80660 steps make 1048580 bytes.
        cell = _high_firing_cell()
        cell.init_state()
        for n_steps, capacity in ((80659, None), (80660, 1048580 // 4 // 64)):
            with self.subTest(n_steps=n_steps):
                port = run_module._SpikePort.of(cell, n_steps=n_steps, path=(False, True))
                self.assertEqual((port.block_steps, port.capacity), (None, capacity))
        self.assertIsNone(run_module._SpikePort.of(cell, n_steps=80660, path=(True, False)).capacity)
        self.assertEqual(run_module._SpikePort.of(cell, n_steps=80660, path=(True, False)).block_steps, 20)

    def test_the_paths_follow_the_platform_of_the_spike_state(self):
        cell = _high_firing_cell()
        cell.init_state()
        reduced = _reduced_cell()
        reduced.init_state()
        for on_gpu in (False, True):
            with self.subTest(on_gpu=on_gpu), mock.patch.object(run_module, "_on_gpu", return_value=on_gpu):
                self.assertEqual(run_module._spike_output_path(cell), (on_gpu, on_gpu))
                self.assertEqual(run_module._spike_output_path(reduced), (False, False))
                with _spike_path("blocked"):
                    self.assertEqual(run_module._spike_output_path(cell), (True, False))
        self.assertFalse(run_module._on_gpu(jax.device_put(np.zeros(3), jax.devices("cpu")[0])))
        self.assertFalse(run_module._on_gpu(np.zeros(3) * u.mV))


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


def ca1_cable(solver):
    """The CA1 fixture (``CVPerBranch()``, 681 CVs) with the channels of :func:`hh_y_cable`: the CELL-50 cost model.

    The 0.1 nA clamp at the root starts at 1 ms, after the 400 steps of
    2e-5 ms the tests run (with ``CVPerBranch(2)``, rk4 at that dt diverges
    under it); from ``V_init`` -65 mV the CVs still move apart, so the axial
    currents matter. ``solver=None`` keeps the Cell's default solver. ``V``
    is recorded as ``"v"``; the cell is not initialized.
    """
    from braincell.io._testing import FIXTURE_DIR

    morpho = braincell.Morphology.from_swc(FIXTURE_DIR / "CA1.swc")
    kwargs = {} if solver is None else {"solver": solver}
    cell = braincell.Cell(morpho, cv_policy=braincell.CVPerBranch(), V_init=-65.0 * u.mV, **kwargs)
    cell.paint(
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(RootLocation(0.5), mech.CurrentClamp(delay=1.0 * u.ms, durations=49.0 * u.ms, amplitudes=0.1 * u.nA))
    cell.record("v", "V")
    return cell


#: The models of the CELL-52 tests: ``(name, build, dt)``; CA1 at the 2e-5 ms of the CELL-50 cost model.
PER_GEOMETRY_MODELS = (
    ("CA1 rk4", lambda: ca1_cable("rk4"), 2e-5 * u.ms),
    ("CA1 default", lambda: ca1_cable(None), 2e-5 * u.ms),
    ("Y rk4", lambda: hh_y_cable("rk4", pop_size=3), DT),
    ("Y staggered", lambda: hh_y_cable("staggered", pop_size=3), DT),
)


def loop_operator_ops(cell, run):
    """``(lu, assembly)``: the ``lu`` and the ``(n_point, n_point)`` scatters in the jaxpr of the loop ``run()`` calls.

    Counted over the whole program, inside its steps and before them; the
    loop is the one ``run()`` makes for ``cell`` (the first it makes).
    """
    calls = []
    make = run_module._make_run_loop

    def recording(*args, **kwargs):
        loop = make(*args, **kwargs)

        def called(*loop_args):
            calls.append((loop, loop_args))
            return loop(*loop_args)

        return called

    with mock.patch.object(run_module, "_make_run_loop", recording):
        run()
    loop, loop_args = calls[0]
    jaxpr = loop.stateful_fun.get_jaxpr(*loop_args)
    n_point = cell.runtime.n_point
    counts = [0, 0]

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "lu":
                counts[0] += 1
            if eqn.primitive.name.startswith("scatter") and tuple(eqn.outvars[0].aval.shape[-2:]) == (n_point, n_point):
                counts[1] += 1
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(sub, "eqns"):  # a Jaxpr
                        walk(sub)
                    elif hasattr(getattr(sub, "jaxpr", None), "eqns"):  # a ClosedJaxpr
                        walk(sub.jaxpr)

    walk(jaxpr)
    return tuple(counts)


@contextlib.contextmanager
def operator_built_in_the_program():
    """``Cell.run`` as before CELL-52: the loop builds the axial operator from the traced geometry."""
    with mock.patch.object(run_module, "host_axial_operator", return_value=None):
        yield


class AxialOperatorPerGeometryTest(unittest.TestCase):
    """CELL-52: of a concrete geometry, ``Cell.run`` builds the axial operator once, on the host, not once per run.

    The loop takes the operator's arrays as an argument (``operator_given``
    in its key); the runtime caches keep it until the geometry changes, and
    a run with traced geometry builds it in its program as before. Every
    run equals the one that builds it in the program, bit for bit.
    """

    N = 400

    def setUp(self):
        self.enterContext(_float64())

    def test_a_concrete_geometry_builds_no_operator_in_the_loop(self):
        # On main b182bf0 the loop of every model assembled the matrix in
        # its program before the steps (four scatters), and rk4 factored
        # its Schur complement: (1, 4) for rk4 and (0, 4) for DHS.
        for name, build, dt in PER_GEOMETRY_MODELS:
            with self.subTest(model=name):
                cell = build()
                cell.init_state()
                self.assertEqual(loop_operator_ops(cell, lambda: cell.run(dt=dt, duration=4 * dt)), (0, 0))
                with (
                    mock.patch.object(_staggered, "_reduce_cv_axial_operator") as reduce,
                    mock.patch.object(_staggered, "_build_dhs_static_source") as dhs,
                ):
                    cell.reset_state()
                    cell.run(dt=dt, duration=4 * dt)
                # The runtime caches served the second run.
                self.assertEqual((reduce.call_count, dhs.call_count), (0, 0))
                with operator_built_in_the_program():
                    traced = build()
                    traced.init_state()
                    expected = (1, 4) if "rk4" in name else (0, 4)
                    self.assertEqual(loop_operator_ops(traced, lambda: traced.run(dt=dt, duration=4 * dt)), expected)

    def test_runs_equal_those_that_build_the_operator_in_the_program(self):
        # Each edit compiles the CA1 loops again (7-8 s for the default
        # solver), so CA1 runs the first three cases here; the PR checks
        # all six of CA1 against main in separate processes.
        def sequence(build, dt, full):
            cell = build()
            cell.init_state()

            def run():
                return np.asarray(cell.run(dt=dt, duration=self.N * dt).traces["v"].to_decimal(u.mV))

            out = {"first": run()}
            cell.reset_state()
            out["hit"] = run()
            cell.geometry.set(Ra=(cell.geometry.get("Ra") * 1.3).reshape(-1))
            out["set"] = run()
            if not full:
                return cell, out
            # A run of another length in between keeps the first loop, whose key does not change.
            cell.run(dt=dt, duration=4 * dt)
            cell.geometry.set(length=(cell.geometry.get("length") * 0.9).reshape(-1))
            cell.reset_state()
            out["set_between"] = run()
            cell.reset()
            cell.cv_policy = braincell.CVPerBranch(2)
            cell.init_state()
            out["cv_policy"] = run()
            cell.reset()
            cell.init_state()
            out["reset_init"] = run()
            return cell, out

        for name, build, dt in PER_GEOMETRY_MODELS:
            with self.subTest(model=name):
                full = name.startswith("Y")
                cell, given = sequence(build, dt, full)
                with operator_built_in_the_program():
                    _, expected = sequence(build, dt, full)
                self.assertEqual(len(expected), 6 if full else 3)
                # One column per CV of each member; CVPerBranch(2) doubles the CVs.
                width = int(np.prod(cell.pop_size)) * cell.n_cv // (2 if full else 1)
                for case, values in expected.items():
                    self.assertTrue(np.all(np.isfinite(values)), case)
                    doubled = case in ("cv_policy", "reset_init")
                    self.assertEqual(values.shape, (self.N, 2 * width if doubled else width), case)
                    np.testing.assert_array_equal(given[case], values, err_msg=f"{name}: {case}")
                self.assertFalse(np.array_equal(expected["set"], expected["hit"]))

    def test_gradients_build_the_operator_in_the_program(self):
        def gradient(solver, *, given):
            cell = hh_y_cable(solver, pop_size=3)
            cell.init_state()
            cell.geometry.trainable(Ra=braincell.trainable.scale(name="ra"))
            states = cell.trainables.parameters().states()
            seen = []
            host = run_module.host_axial_operator

            def spy(target):
                operator = host(target)
                seen.append(operator)
                return operator

            def loss():
                cell.reset_state()
                return u.math.sum(cell.run(dt=DT, duration=self.N * DT).traces["v"].to_decimal(u.mV))

            with mock.patch.object(run_module, "host_axial_operator", spy if given else (lambda target: None)):
                grads, value = brainstate.transform.grad(loss, grad_states=states, return_value=True)()
                keys = list(cell._run_loop_cache)
                # A geometry trainable changed by an optimizer: the eager run sees it.
                for state in states.values():
                    state.value = jax.tree.map(lambda x: x * 1.1, state.value)
                cell.reset_state()
                after = np.asarray(cell.run(dt=DT, duration=self.N * DT).traces["v"].to_decimal(u.mV))
            leaf = np.asarray(jax.tree.leaves(grads)[0]).reshape(-1)
            return leaf, np.asarray(value), seen, keys, after

        for solver in ("rk4", "staggered"):
            with self.subTest(solver=solver):
                grad, value, seen, keys, after = gradient(solver, given=True)
                expected_grad, expected_value, _, _, expected_after = gradient(solver, given=False)
                self.assertTrue(np.all(np.isfinite(grad)) and np.all(grad != 0.0))
                np.testing.assert_array_equal(grad, expected_grad)
                np.testing.assert_array_equal(value, expected_value)
                np.testing.assert_array_equal(after, expected_after)
                # Traced geometry: no host operator, and the loop without one.
                self.assertIsNone(seen[0])
                self.assertEqual([key[-1] for key in keys], [False])

    def test_the_loop_compiles_as_before(self):
        # Runs of 40 steps: run, reset_state + run, set + run, reset_state +
        # run make loops [1, 0, 1, 0], as on main b182bf0 (a geometry edit
        # drops the compiled loops). A gradient run and an eager run compile
        # one loop each, and alternating them compiles nothing more (sodium
        # g_max trains: a geometry trainable refreshes the geometry at every
        # eager run, which drops the loops on main as well).
        for solver in ("rk4", "staggered"):
            with self.subTest(solver=solver):
                cell = hh_y_cable(solver)
                cell.init_state()
                made = []
                make = run_module._make_run_loop

                def counting(*args, **kwargs):
                    made.append(kwargs.get("operator_given"))
                    return make(*args, **kwargs)

                counts = []
                with mock.patch.object(run_module, "_make_run_loop", counting):
                    for action in ("run", "reset", "set", "reset"):
                        if action == "reset":
                            cell.reset_state()
                        elif action == "set":
                            cell.geometry.set(Ra=(cell.geometry.get("Ra") * 1.3).reshape(-1))
                        before = len(made)
                        cell.run(dt=DT, duration=40 * DT)
                        counts.append(len(made) - before)
                    self.assertEqual(counts, [1, 0, 1, 0])
                    cell.channels["na"].trainable(g_max=braincell.trainable.scale(name="na"))
                    states = cell.trainables.parameters().states()

                    def loss():
                        cell.reset_state()
                        return u.math.sum(cell.run(dt=DT, duration=40 * DT).traces["v"].to_decimal(u.mV))

                    grad = brainstate.transform.grad(loss, grad_states=states)
                    made.clear()
                    for _ in range(2):
                        grad()
                        cell.reset_state()
                        cell.run(dt=DT, duration=40 * DT)
                self.assertEqual(sorted(made), [False, True])


#: The dt of the history before a dt change, 0.025 ms, as an exact fraction of a millisecond.
_DT_OLD = Fraction(1, 40)


def _schedules(dt):
    """The recording schedules of the off-grid tests at ``dt`` (0.05 or 0.1 ms): ``{name: (start, period, kwargs)}``."""
    period = Fraction(1, 10) if dt == Fraction(1, 20) else Fraction(1, 5)
    kilohertz = int(1 / period)
    out = {
        "every_dt": (Fraction(0), dt, {}),
        "period": (Fraction(0), period, {"period": float(period) * u.ms}),
        "frequency": (Fraction(0), period, {"frequency": kilohertz * u.kHz}),
        "start_0.2": (Fraction(1, 5), dt, {"start": 0.2 * u.ms}),
        "start_0.2_period": (Fraction(1, 5), period, {"start": 0.2 * u.ms, "period": float(period) * u.ms}),
    }
    if dt == Fraction(1, 20):
        out["start_0.05"] = (Fraction(1, 20), dt, {"start": 0.05 * u.ms})
    return out


def _rising_cell(schedules):
    """One CV (20 um x radius 10 um) with a leak at 0 mV from -65 mV: V rises at every step, so a sample's value names its step.

    Records V every step as ``"all"`` and on each schedule of ``schedules`` (:func:`_schedules`).
    """
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=0.0 * u.mV))
    cell.record("all", "V")
    for name, (_, _, kwargs) in schedules.items():
        cell.record(name, "V", **kwargs)
    cell.init_state()
    return cell


def _hand_rows(start, period, origin, dt, n_steps, since=None):
    """Rows of a run of ``n_steps`` steps of ``dt`` from ``origin`` (exact fractions) that sample ``start + j·period``.

    Each time ``s`` goes to row ``floor((s - origin) / dt + 1/2)``, the step at which an event at ``s`` acts;
    ``since`` keeps the times ``s >= since`` only.
    """
    rows, j = [], 0
    while True:
        time = start + j * period
        row = int(((time - origin) / dt + Fraction(1, 2)).__floor__())
        if row >= n_steps:
            return rows
        if row >= 0 and (since is None or time >= since):
            rows.append(row)
        j += 1


def _sampled_rows(block, every):
    """The rows of ``every`` (the ``"all"`` block) whose values ``block`` holds."""
    reference = np.asarray(every.values.to_decimal(u.mV)).reshape(-1)
    rows = []
    for value in np.asarray(block.values.to_decimal(u.mV)).reshape(-1):
        (hit,) = np.flatnonzero(reference == value)
        rows.append(int(hit))
    return rows


def _run_after(cell, history, dt, n_steps):
    """``history`` steps of 0.025 ms, then ``n_steps`` of ``dt`` (fractions of a ms); the result of the second run."""
    if history:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a history shorter than a period
            cell.run(dt=float(_DT_OLD) * u.ms, duration=float(history * _DT_OLD) * u.ms)
    return cell.run(dt=float(dt) * u.ms, duration=float(n_steps * dt) * u.ms)


def long_run_cell():
    """One CV with a leak at rest, recording V every 0.05 ms (``"p2"``) and every 0.1 ms (``"p4"``)."""
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.record("p2", "V", period=0.05 * u.ms)
    cell.record("p4", "V", period=0.1 * u.ms)
    return cell


class OffGridScheduleTest(unittest.TestCase):
    """CELL-59a: a run off the ``dt`` grid samples where a scheduled event at the sample time acts."""

    # (history steps of 0.025 ms, dt in ms): the origins 1.5, 2.5 and 0.75 steps of dt.
    HISTORIES = ((3, Fraction(1, 20)), (5, Fraction(1, 20)), (3, Fraction(1, 10)))

    def _check_rows(self, history, dt, n_steps):
        schedules = _schedules(dt)
        cell = _rising_cell(schedules)
        result = _run_after(cell, history, dt, n_steps)
        origin_ms, step, _ = cell._clock_host()
        self.assertEqual(step, n_steps)  # the clock counts from the dt change (L93)
        self.assertAlmostEqual(origin_ms, float(history * _DT_OLD), places=6)
        every = result.samples["all"]
        self.assertEqual(every.values.shape[0], n_steps)
        rows = {}
        for name, (start, period, _) in schedules.items():
            block = result.samples[name]
            rows[name] = _sampled_rows(block, every)
            self.assertEqual(rows[name], _hand_rows(start, period, history * _DT_OLD, dt, n_steps), name)
            if rows[name]:
                times = np.asarray(every.time.to_decimal(u.ms))[rows[name]]
                np.testing.assert_allclose(np.asarray(block.time.to_decimal(u.ms)), times, rtol=0, atol=1e-6)
            else:
                self.assertIsNone(block.first_time)
        return rows

    def test_off_grid_runs_sample_the_steps_of_events_at_the_sample_times(self):
        # On main 73136f6, after 3 steps of 0.025 ms, steps of 0.05 ms took
        # no "period=0.1 ms" sample in 21 steps (the float time mask) and
        # rows 0, 2, ..., 20 (0.075 ms on, rint of 1.5 steps) in 22, and no
        # "start=0.2 ms" sample in either.
        for precision in (32, 64):
            for history, dt in self.HISTORIES:
                for n_steps in (21, 22):
                    with (
                        self.subTest(precision=precision, history=history, dt=float(dt), n_steps=n_steps),
                        brainstate.environ.context(precision=precision),
                        warnings.catch_warnings(),
                    ):
                        warnings.simplefilter("ignore", RuntimeWarning)  # 21 steps of a 2-step period
                        rows = self._check_rows(history, dt, n_steps)
                        if (history, dt) == (3, Fraction(1, 20)):
                            # 0.1, 0.2, ... ms at 0.125, 0.225, ... ms; 0.2 ms on at 0.225 ms.
                            self.assertEqual(rows["period"], list(range(1, n_steps, 2)))
                            self.assertEqual(rows["start_0.2"], list(range(3, n_steps)))

    def test_on_the_grid_samples_are_the_schedule_times(self):
        for history, dt in ((0, Fraction(1, 20)), (10, Fraction(1, 20)), (4, Fraction(1, 10))):
            for n_steps in (21, 22):
                with self.subTest(history=history, dt=float(dt), n_steps=n_steps), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self._check_rows(history, dt, n_steps)

    def test_two_segments_sample_as_one_run(self):
        # On main 73136f6 each 21-step segment after 3 steps of 0.025 ms took
        # no "period=0.1 ms" sample; one 42-step run took 21 (rint of 1.5).
        dt = Fraction(1, 20)
        schedules = _schedules(dt)
        for precision in (32, 64):
            with (
                self.subTest(precision=precision),
                brainstate.environ.context(precision=precision),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                whole_cell = _rising_cell(schedules)
                whole = _run_after(whole_cell, 3, dt, 42)
                split_cell = _rising_cell(schedules)
                first = _run_after(split_cell, 3, dt, 21)
                split = RunResult.concat((first, split_cell.run(dt=0.05 * u.ms, duration=21 * 0.05 * u.ms)))
                for name in schedules:
                    joined, expected = split.samples[name], whole.samples[name]
                    self.assertEqual(joined.values.shape, expected.values.shape, name)
                    self.assertGreater(joined.values.shape[0], 0, name)
                    np.testing.assert_allclose(
                        joined.values.to_decimal(u.mV), expected.values.to_decimal(u.mV), rtol=1e-12, atol=0.0
                    )
                    np.testing.assert_allclose(
                        joined.time.to_decimal(u.ms), expected.time.to_decimal(u.ms), rtol=0, atol=1e-6
                    )
                    rows = _sampled_rows(expected, whole.samples["all"])
                    np.testing.assert_allclose(
                        joined.time.to_decimal(u.ms),
                        np.asarray(whole.samples["all"].time.to_decimal(u.ms))[rows],
                        rtol=0,
                        atol=1e-6,
                    )

    def test_samples_fall_on_the_boundaries_of_events_at_their_times(self):
        # One event at each schedule time (weight 1 uS, no delay) into an
        # ExpSyn that keeps its conductance: the boundary of row k adds
        # 1 uS between rows k and k + 1, the last one after the run.
        for precision in (32, 64):
            for history, dt in ((3, Fraction(1, 20)), (3, Fraction(1, 10))):
                for name, (start, period, kwargs) in _schedules(dt).items():
                    if name == "frequency":
                        continue
                    with (
                        self.subTest(precision=precision, history=history, dt=float(dt), schedule=name),
                        brainstate.environ.context(precision=precision),
                        warnings.catch_warnings(),
                    ):
                        warnings.simplefilter("ignore", RuntimeWarning)
                        self._check_event_rows(history, dt, start, period, kwargs)

    def _check_event_rows(self, history, dt, start, period, kwargs):
        n_steps = 21
        origin = history * _DT_OLD
        end = origin + n_steps * dt
        times = []
        while start + len(times) * period < end:
            times.append(float(start + len(times) * period))
        branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
        cell = braincell.Cell(
            braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
        )
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        cell.place(at(0, 0.5), mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV))
        cell.record("g", braincell.observe.synapse(name="s").state("g"))
        cell.record("sampled", braincell.observe.synapse(name="s").state("g"), **kwargs)
        braincell.connect(
            "c",
            source=braincell.EventSequence.from_times([np.asarray(times) * u.ms]),
            synapse=cell.synapses["s"],
            weight=1.0 * u.uS,
        )
        cell.init_state()
        result = _run_after(cell, history, dt, n_steps)
        g = np.asarray(result.samples["g"].values.to_decimal(u.uS), dtype=float).reshape(-1)
        self.assertEqual(g.shape, (n_steps,))
        after = float(np.asarray(cell.synapses["s"].get("g").to_decimal(u.uS)).reshape(-1)[0])  # L93
        increments = np.diff(np.r_[g, after])
        event_rows = np.flatnonzero(increments > 0.5).tolist()
        self.assertTrue(np.allclose(increments[event_rows], 1.0))
        sampled = result.samples["sampled"]
        step_times = np.asarray(result.samples["g"].time.to_decimal(u.ms))
        sample_rows = [int(np.argmin(np.abs(step_times - t))) for t in np.asarray(sampled.time.to_decimal(u.ms))]
        self.assertEqual(sample_rows, _hand_rows(start, period, origin, dt, n_steps))
        # The events from the dt change on act in this run, the earlier ones
        # in the history's (CELL-55); each sample time from it on is sampled
        # where its event acts.
        self.assertEqual(event_rows, _hand_rows(start, period, origin, dt, n_steps, since=origin))

    def test_a_long_run_takes_each_sample_once(self):
        # On main 73136f6 the float time mask took extra samples from about
        # 1e6 steps on: 21 steps after 1e6 took 21 and 16 samples (float64),
        # 21 and 12 (float32), for 11 and 6.
        for precision in (32, 64):
            with self.subTest(precision=precision), brainstate.environ.context(precision=precision):
                cell = long_run_cell()
                cell.init_state()
                long = cell.run(dt=DT, duration=1_000_000 * DT)
                self.assertEqual(cell._clock_host()[1], 1_000_000)
                self.assertEqual([long.samples[n].values.shape[0] for n in ("p2", "p4")], [500_000, 250_000])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    short = cell.run(dt=DT, duration=21 * DT)
                self.assertEqual(cell._clock_host()[1], 1_000_021)
                self.assertEqual([short.samples[n].values.shape[0] for n in ("p2", "p4")], [11, 6])

    def test_a_cell_reset_to_another_origin_samples_for_it(self):
        # One Cell, 1.5 steps off the 0.05 ms grid and, after reset_state(),
        # 2.5 steps off it: the second run is compiled for its own origin (L40).
        dt = Fraction(1, 20)
        schedules = _schedules(dt)
        cell = _rising_cell(schedules)
        rows = {}
        for history in (3, 5):
            if history == 5:
                cell.reset_state()
            result = _run_after(cell, history, dt, 22)
            rows[history] = {name: _sampled_rows(result.samples[name], result.samples["all"]) for name in schedules}
            for name, (start, period, _) in schedules.items():
                self.assertEqual(rows[history][name], _hand_rows(start, period, history * _DT_OLD, dt, 22), name)
        self.assertNotEqual(rows[3]["period"], rows[5]["period"])


if __name__ == "__main__":
    unittest.main()
