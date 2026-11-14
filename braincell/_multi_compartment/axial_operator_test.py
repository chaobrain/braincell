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

"""Tests for :mod:`braincell._multi_compartment.axial_operator`: ``Cell.run`` builds the axial operator once."""

import contextlib
import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment import run as run_module
from braincell._multi_compartment.run_test import DT, _float64, _hh_cell
from braincell.filter import RootLocation
from braincell.quad import _staggered


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
