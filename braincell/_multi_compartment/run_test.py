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
from braincell.filter import AllRegion, RootLocation
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


def make_event_cell(solver="staggered"):
    """One 600 um x 0.5 um passive dendrite (3 CVs) with an ExpSyn at x = 0.

    Leak 0.1 mS/cm^2 at -65 mV, cm 1 uF/cm^2, Ra 100 ohm cm; ExpSyn tau 2 ms,
    e 0 mV. Returns ``(cell, synapse view)``; the cell is not initialized.
    """
    branch = braincell.Branch.from_lengths(lengths=[600.0] * u.um, radii=[0.5, 0.5] * u.um, type="dendrite")
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch, name="dend"),
        cv_policy=braincell.CVPerBranch(cv_per_branch=3),
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


def split_steps(total_steps, split_steps_):
    """Step counts of the segments ``[0, s1), [s1, s2), ..., [sk, total)``."""
    edges = (0, *sorted(int(step) for step in split_steps_), int(total_steps))
    return tuple(stop - start for start, stop in zip(edges, edges[1:]))


def traces(result):
    """``(g in uS, V in mV)`` per step of one RunResult."""
    return (
        np.asarray(result.samples["g"].values.to_decimal(u.uS)),
        np.asarray(result.samples["v"].values.to_decimal(u.mV)),
    )


def assert_same_trajectory(case, split, whole):
    for name, actual, expected in zip(("g", "V"), traces(split), traces(whole)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0, err_msg=name)


class SplitCellRunTest(unittest.TestCase):
    """CELL-34: ``Cell.run`` split at any step, event times included, matches one run.

    Each step prepares the synaptic input of the boundary that follows it, so
    a run continuing an earlier run or ``update()`` must not prepare its first
    boundary again: events landing on it would be delivered twice.
    """

    def setUp(self):
        self._float64 = _float64()
        self._float64.__enter__()
        self.addCleanup(self._float64.__exit__, None, None, None)

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
                assert_same_trajectory(self, split, whole)
                g = traces(whole)[0][:, 0]
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
                assert_same_trajectory(self, split, whole)

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
        first = traces(cell.run(dt=DT, duration=4 * DT))
        cell.reset_state()
        again = traces(cell.run(dt=DT, duration=4 * DT))
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
        assert_same_trajectory(self, split, whole)

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
        np.testing.assert_allclose(
            node.g.value.to_decimal(u.uS), reference.g.value.to_decimal(u.uS), rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
