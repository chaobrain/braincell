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

"""Lumped (point) Cells: ``Cell.point`` and ``cv_policy=Lumped()``."""

import functools
import re
import unittest
import warnings
from unittest import mock

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell as bc
from braincell._multi_compartment import cell as cell_module, lumped
from braincell._single_compartment.point import PointNeuron
from braincell.filter import AllRegion, RootLocation, at
from braincell.quad._util import split_diffeq_states
from braincell.ion._testing import run_float64

LENGTH = 20.0 * u.um
RADIUS = 10.0 * u.um
N = 3
DT = 0.025 * u.ms
DURATION = 20.0 * u.ms
#: 10 uA/cm^2 on the 2 pi r L soma, as total current.
CLAMP_DENSITY = 10.0 * u.uA / u.cm**2


def _area():
    return 2.0 * np.pi * RADIUS * LENGTH


def _mechanisms(hh: bool):
    leak = bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV)
    if not hh:
        return (leak,)
    return (
        bc.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        bc.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        bc.mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
        bc.mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
        leak,
    )


def _clamp():
    return bc.mech.CurrentClamp(
        delay=2.0 * u.ms,
        durations=5.0 * u.ms,
        amplitudes=(CLAMP_DENSITY * _area()).in_unit(u.nA),
    )


def _late_current(t):
    return u.math.where(t > 2.0 * u.ms, 0.02 * u.nA, 0.0 * u.nA)


def _mixed_clamps():
    return (
        bc.mech.FunctionClamp(fn=_late_current),
        bc.mech.CurrentClamp(durations=1.0 * u.ms, amplitudes=0.03 * u.nA),
        bc.mech.SineClamp(
            amplitude=0.05 * u.nA,
            frequency=100.0 * u.Hz,
            phase=0.3,
            offset=0.01 * u.nA,
            delay=1.0 * u.ms,
            duration=3.0 * u.ms,
        ),
        bc.mech.CurrentClamp(delay=0.5 * u.ms, durations=2.0 * u.ms, amplitudes=0.04 * u.nA),
    )


class _NoopReduction:
    def init_state(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def reset_state(self, *args, **kwargs):
        pass

    def reset(self):
        pass


def _point_cell(*, hh=True, clamp=False, solver=None, pop_size=N):
    cell = bc.Cell.point(pop_size, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV, solver=solver)
    cell.paint(AllRegion(), *_mechanisms(hh))
    if clamp:
        cell.place(_clamp())
    cell.record("v", bc.observe.state("V"))
    return cell


def _point_neuron(*, hh, solver):
    """The point-neuron math built by hand: the model a lumped Cell must reproduce."""

    class Model(PointNeuron):
        def __init__(self):
            super().__init__(N, V_initializer=braintools.init.Constant(-65.0 * u.mV), solver=solver)
            self.area = _area()
            if hh:
                self.na = bc.ion.SodiumFixed(N, E=50.0 * u.mV)
                self.na.add(INa=bc.channel.Na_HH1952(N, g_max=120.0 * u.mS / u.cm**2))
                self.k = bc.ion.PotassiumFixed(N, E=-77.0 * u.mV)
                self.k.add(IK=bc.channel.K_HH1952(N, g_max=36.0 * u.mS / u.cm**2))
            self.IL = bc.channel.IL(N, g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV)

    return Model()


def _run_point_neuron(model, *, clamp):
    """Voltage before each step; I_ext is the clamp density at t + dt/2."""
    model.init_state()
    model.reset_state()
    n_steps = int(round(float(DURATION / DT)))

    def step(index):
        t = index * DT
        with brainstate.environ.context(t=t, dt=DT):
            mid = t + 0.5 * DT
            on = (mid >= 2.0 * u.ms) & (mid < 7.0 * u.ms)
            current = u.math.where(on, 10.0, 0.0) * u.uA / u.cm**2 if clamp else 0.0 * u.uA / u.cm**2
            before = model.V.value
            model.update(current)
            return before

    with brainstate.environ.context(dt=DT):
        return brainstate.transform.for_loop(step, np.arange(n_steps))


class LumpedEquivalenceTest(unittest.TestCase):
    """Acceptance 1: a lumped Cell reproduces a hand-built PointNeuron in float64."""

    def test_voltage_matches_point_neuron(self) -> None:
        bounds = {"rk4": 1e-10, "euler": 1e-10, "ralston2": 1e-10, "exp_euler": 1e-9}
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for hh in (False, True):
                for clamp in (False, True):
                    for solver, bound in bounds.items():
                        with self.subTest(hh=hh, clamp=clamp, solver=solver):
                            expected = _run_point_neuron(_point_neuron(hh=hh, solver=solver), clamp=clamp)
                            cell = _point_cell(hh=hh, clamp=clamp, solver=solver)
                            cell.init_state()
                            values = cell.run(dt=DT, duration=DURATION).samples["v"].values
                            self.assertEqual(values.shape, (800, N))
                            error = float(u.math.max(u.math.abs(values - expected)).to_decimal(u.mV))
                            self.assertLessEqual(error, bound)

    def test_initial_derivative_matches_point_neuron(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for hh in (False, True):
                with self.subTest(hh=hh):
                    model = _point_neuron(hh=hh, solver="rk4")
                    model.init_state()
                    model.reset_state()
                    cell = _point_cell(hh=hh, solver="rk4")
                    cell.init_state()
                    with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                        model.compute_derivative(0.0 * u.uA / u.cm**2)
                        cell.compute_derivative()
                    expected = model.V.derivative
                    relative = u.math.abs(cell.V.derivative - expected) / u.math.abs(expected)
                    self.assertLessEqual(float(u.math.max(relative)), 1e-12)


class LumpedShapeLifecycleTest(unittest.TestCase):
    """Acceptance 2: no CV axis, batch, continued runs, reset()."""

    def test_state_has_no_cv_axis(self) -> None:
        cell = _point_cell()
        cell.init_state()
        self.assertEqual(cell.V.value.shape, (N,))
        self.assertEqual(cell.spike.value.shape, (N,))
        self.assertEqual(cell.voltage_shape, (N,))
        np.testing.assert_allclose(cell._lumped_runtime.area.to_decimal(u.um**2), _area().to_decimal(u.um**2))
        self.assertIn("lumped=True", repr(cell))

    def test_integration_hooks_match_compute_derivative(self) -> None:
        reference = _point_cell(clamp=True, solver="rk4")
        hooked = _point_cell(clamp=True, solver="rk4")
        for cell in (reference, hooked):
            cell.init_state()
        with brainstate.environ.context(t=3.0 * u.ms, dt=DT):
            # pre_integral samples the clamp for a directly called step
            # (CELL-2c); the reference samples it as update() does.
            reference._prepare_step_clamps()
            reference.compute_derivative()
            hooked.pre_integral()
            hooked.compute_derivative()
            hooked.post_integral()
        np.testing.assert_array_equal(
            hooked.V.derivative.to_decimal(u.mV / u.ms), reference.V.derivative.to_decimal(u.mV / u.ms)
        )

    def test_point_tree_views_raise(self) -> None:
        cell = _point_cell()
        cell.init_state()
        for name in ("n_point", "layouts"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "lumped Cell"):
                    getattr(cell, name)
        # CELL-24: the lumped nodes run on the layouts of the one-CV twin.
        self.assertEqual(cell.runtime.n_cv, 1)
        self.assertEqual(cell.runtime.voltage_shape, (N, 1))
        self.assertTrue(cell.runtime.is_lumped)

    def test_batch_axis_and_reset_state(self) -> None:
        cell = _point_cell(clamp=True)
        cell.init_state(batch_size=2)
        self.assertEqual(cell.V.value.shape, (2, N))
        self.assertEqual(cell.spike.value.shape, (2, N))
        cell.run(dt=DT, duration=5.0 * u.ms)
        self.assertFalse(np.allclose(cell.V.value.to_decimal(u.mV), -65.0))
        cell.reset_state(batch_size=2)
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), np.full((2, N), -65.0))
        self.assertEqual(float(cell.current_time.to_decimal(u.ms)), 0.0)

    def test_consecutive_runs_continue(self) -> None:
        whole = _point_cell(clamp=True)
        whole.init_state()
        expected = whole.run(dt=DT, duration=10.0 * u.ms).samples["v"].values
        split = _point_cell(clamp=True)
        split.init_state()
        first = split.run(dt=DT, duration=5.0 * u.ms).samples["v"].values
        second = split.run(dt=DT, duration=5.0 * u.ms).samples["v"].values
        self.assertEqual(first.shape, (200, N))
        joined = u.math.concatenate([first, second], axis=0)
        np.testing.assert_allclose(joined.to_decimal(u.mV), expected.to_decimal(u.mV), rtol=1e-6, atol=1e-5)

    def test_update_steps_one_dt(self) -> None:
        cell = _point_cell(clamp=True)
        cell.init_state()
        with brainstate.environ.context(dt=DT, t=0.0 * u.ms):
            spike = cell.update()
        self.assertEqual(spike.shape, (N,))

    def test_clamp_current_is_recorded_at_step_midpoints(self) -> None:
        cell = bc.Cell.point(N)
        cell.paint(AllRegion(), *_mechanisms(False))
        cell.place(bc.mech.CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA))
        cell.record("total", bc.observe.clamp_current())
        cell.record("each", bc.observe.clamp_current(reduce="none"))
        cell.init_state()
        result = cell.run(dt=0.5 * u.ms, duration=4.0 * u.ms)
        expected = np.asarray([0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
        for name in ("total", "each"):
            values = result.samples[name].values.to_decimal(u.nA)
            self.assertEqual(values.shape, (8, N))
            np.testing.assert_allclose(values, np.repeat(expected[:, None], N, axis=1), atol=1e-7)

    def test_every_clamp_type_matches_a_one_cv_cable_cell(self) -> None:
        results = []
        for cell, site in (
            (bc.Cell.point(2), ()),
            (bc.Cell(bc.Morphology.from_soma(), pop_size=2), (RootLocation(0.5),)),
        ):
            cell.paint(AllRegion(), *_mechanisms(False))
            cell.place(*site, *_mixed_clamps())
            cell.record("each", bc.observe.clamp_current(reduce="none"))
            cell.init_state()
            results.append(cell.run(dt=DT, duration=5.0 * u.ms).samples["each"].values.to_decimal(u.nA))
        self.assertEqual(results[0].shape, (200, 8))
        self.assertGreater(float(np.max(np.abs(results[0]))), 0.05)
        np.testing.assert_allclose(results[0], results[1], rtol=0.0, atol=1e-12)

    def test_spatial_parameters_are_evaluated_at_the_cv(self) -> None:
        def run(g_max):
            cell = bc.Cell.point(N, V_init=-65.0 * u.mV)
            cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=g_max, E=-54.3 * u.mV))
            cell.record("v", bc.observe.state("V"))
            cell.init_state()
            return cell.run(dt=DT, duration=2.0 * u.ms).samples["v"].values.to_decimal(u.mV)

        constant = run(0.3 * u.mS / u.cm**2)
        np.testing.assert_array_equal(run(lambda cv: 0.3 * u.mS / u.cm**2), constant)

    def test_channel_named_like_an_ion_keeps_both(self) -> None:
        traces = []
        for name in (None, "na"):
            cell = bc.Cell.point(N, V_init=-65.0 * u.mV)
            cell.paint(
                AllRegion(),
                bc.mech.Ion("SodiumFixed", name="na"),
                bc.mech.Channel("IL", name=name, g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
            )
            cell.record("v", bc.observe.state("V"))
            cell.init_state()
            traces.append(cell.run(dt=DT, duration=2.0 * u.ms).samples["v"].values.to_decimal(u.mV))
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_declared_ci_initializer_matches_the_class_default(self) -> None:
        # MECH-38: a declared ``Ci_initializer`` reached the ion as a bare
        # scalar, ``Ci`` started rank-0, and the first step's ``(1,)`` update
        # broke the ``scan`` carry. The class default is the same Constant.
        def run(**declared):
            cell = bc.Cell(
                bc.Morphology.from_soma(length=15.0 * u.um, radius=7.5 * u.um),
                cv_policy=bc.Lumped(),
                V_init=-60.0 * u.mV,
            )
            cell.paint(AllRegion(), bc.mech.Ion("CalciumDetailed", name="ca", **declared))
            cell.paint(AllRegion(), bc.mech.Channel("CaT_HM1992", name="cat"))
            cell.record("v", bc.observe.state("V"))
            cell.init_state()
            ion = cell._lumped_runtime.ion_channels["ca"]
            initial_shape = ion.Ci.value.shape
            v = cell.run(dt=DT, duration=1.0 * u.ms).samples["v"].values.to_decimal(u.mV)
            return initial_shape, v, np.asarray(ion.Ci.value.to_decimal(u.mM))

        shape, v, ci = run()
        self.assertEqual(shape, (1,))
        self.assertEqual(v.shape, (40, 1))
        for declared in (2.4e-4 * u.mM, braintools.init.Constant(2.4e-4 * u.mM)):
            with self.subTest(Ci_initializer=declared):
                got_shape, got_v, got_ci = run(Ci_initializer=declared)
                self.assertEqual(got_shape, shape)
                np.testing.assert_array_equal(got_v, v)
                np.testing.assert_array_equal(got_ci, ci)

    def test_declared_ci_initializer_keeps_the_batch_axis(self) -> None:
        cell = bc.Cell.point(N, V_init=-60.0 * u.mV)
        cell.paint(AllRegion(), bc.mech.Ion("CalciumDetailed", name="ca", Ci_initializer=2.4e-4 * u.mM))
        cell.paint(AllRegion(), bc.mech.Channel("CaT_HM1992", name="cat"))
        cell.init_state(batch_size=2)
        ion = cell._lumped_runtime.ion_channels["ca"]
        self.assertEqual(ion.Ci.value.shape, (2, N))
        cell.run(dt=DT, duration=0.5 * u.ms)
        cell.reset_state(batch_size=2)
        ion = cell._lumped_runtime.ion_channels["ca"]
        self.assertEqual(ion.Ci.value.shape, (2, N))
        np.testing.assert_allclose(ion.Ci.value.to_decimal(u.mM), np.full((2, N), 2.4e-4), rtol=1e-6)

    def test_reset_returns_to_declaring(self) -> None:
        cell = _point_cell(hh=False)
        cell.init_state()
        cell.reset()
        cell.paint(AllRegion(), *_mechanisms(True)[:-1])
        cell.place(_clamp())
        cell.init_state()
        result = cell.run(dt=DT, duration=10.0 * u.ms)
        self.assertGreater(float(u.math.max(result.samples["v"].values).to_decimal(u.mV)), 0.0)


class LumpedDefaultsTest(unittest.TestCase):
    """Acceptance 3: solver by mode, resting V_init, gates at steady state."""

    def test_solver_resolves_by_mode(self) -> None:
        self.assertEqual(bc.Cell.point(5).solver_name, "ralston2")
        self.assertEqual(bc.Cell(bc.Morphology.from_soma()).solver_name, "staggered")
        self.assertEqual(bc.Cell.point(5, solver="rk4").solver_name, "rk4")
        cell = bc.Cell(bc.Morphology.from_soma())
        cell.cv_policy = bc.Lumped()
        self.assertEqual(cell.solver_name, "ralston2")
        cell.cv_policy = bc.CVPerBranch()
        self.assertEqual(cell.solver_name, "staggered")

    def test_default_V_init_is_resting_potential(self) -> None:
        cell = bc.Cell.point(N)
        cell.paint(AllRegion(), *_mechanisms(False))
        cell.init_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), np.full((N,), -65.0))

    def test_gates_start_at_steady_state(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            cell = _point_cell()
            cell.init_state()
            sodium = cell._lumped_runtime.ion_channels["SodiumFixed"]
            channel = sodium.channels["Na_HH1952"]
            V = -65.0 * u.mV
            info = sodium.pack_info()
            m_inf = channel.f_p_alpha(V, info) / (channel.f_p_alpha(V, info) + channel.f_p_beta(V, info))
            h_inf = channel.f_q_alpha(V, info) / (channel.f_q_alpha(V, info) + channel.f_q_beta(V, info))
            np.testing.assert_allclose(np.asarray(channel.p.value), np.broadcast_to(m_inf, (N,)), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(channel.q.value), np.broadcast_to(h_inf, (N,)), rtol=1e-12)
            potassium = cell._lumped_runtime.ion_channels["PotassiumFixed"]
            k_channel = potassium.channels["K_HH1952"]
            k_info = potassium.pack_info()
            n_inf = k_channel.f_p_alpha(V, k_info) / (k_channel.f_p_alpha(V, k_info) + k_channel.f_p_beta(V, k_info))
            np.testing.assert_allclose(np.asarray(k_channel.p.value), np.broadcast_to(n_inf, (N,)), rtol=1e-12)


def _kinetic_pool_cell(solver):
    """A GoC calcium pool with its HV calcium channel: lumped, or the one-CV cable twin under staggered."""
    if solver == "staggered":
        cell = bc.Cell(bc.Morphology.from_soma(length=LENGTH, radius=RADIUS), V_init=-65.0 * u.mV)
    else:
        cell = bc.Cell.point(1, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV, solver=solver)
    cell.paint(
        AllRegion(),
        bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        bc.mech.Ion("CdpStC_MA2020_GoC", name="ca"),
        bc.mech.Channel("CaHVA_MA2020_GoC", g_max=0.5 * u.mS / u.cm**2, ion_name="ca"),
    )
    return cell


def _final_values(cell, dt_ms):
    """``V`` (mV) at 1 ms and the other DiffEqStates, keyed by their path below the runtime."""
    cell.init_state()
    cell.run(dt=dt_ms * u.ms, duration=1.0 * u.ms)
    states = {
        ".".join(map(str, path)).removeprefix("_lumped_runtime."): np.asarray(u.get_mantissa(st.value)).ravel()
        for path, st in brainstate.graph.states(cell).items()
        if isinstance(st, bc.quad.DiffEqState) and path[-1] != "V"
    }
    return np.asarray(cell.V.value.to_decimal(u.mV)).ravel(), states


class LumpedKineticIonTest(unittest.TestCase):
    """CELL-19: a kinetic ion on a lumped Cell runs under every solver it accepts.

    The explicit methods (``ralston2`` is the default) used to raise
    ``ValueError: State ... must be written by compute_derivative()`` in
    ``run``, and the block solvers an ``UnexpectedTracerError`` once the pool
    had a channel of its own.
    """

    def test_every_solver_runs(self) -> None:
        for name in [n for n in bc.quad.list_integrators() if n not in ("staggered", "dhs_voltage")]:
            with self.subTest(solver=name):
                cell = _kinetic_pool_cell(name)
                cell.init_state()
                cell.run(dt=DT, duration=2 * DT)
                self.assertTrue(np.all(np.isfinite(cell.V.value.to_decimal(u.mV))))

    def test_matches_the_cable_twin_under_staggered(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            reference = {dt: _final_values(_kinetic_pool_cell("staggered"), dt) for dt in (0.005, 0.001)}
            for name in (None, "rk4", "ind_exp_euler", "exp_euler", "backward_euler"):
                with self.subTest(solver=name):
                    gap = {}
                    for dt in (0.005, 0.001):
                        V, states = _final_values(_kinetic_pool_cell(name), dt)
                        ref_V, ref_states = reference[dt]
                        self.assertEqual(set(states), set(ref_states))
                        self.assertGreater(len(states), 10)
                        gap[dt] = float(np.max(np.abs(V - ref_V)))
                        for key in states:
                            np.testing.assert_allclose(states[key], ref_states[key], rtol=0, atol=1e-3, err_msg=key)
                    self.assertLessEqual(gap[0.005], 1e-3)
                    self.assertGreaterEqual(gap[0.005] / gap[0.001], 3.0)


#: Explicit Runge-Kutta names: the ``'ode'`` kind of braintools, wrapped by braincell.
EXPLICIT_SOLVERS = tuple(n for n in bc.quad.list_integrators() if bc.quad.integrator_info(n).kind == "ode")
STIFF_WARNING = "stiff for an explicit solver"


def _stiff_gate_cell(*channels, solver=None, pop_size=1, ion=("CalciumFixed", "ca"), leak=True):
    """A lumped soma with a leak, one ion and ``channels`` (declarations), at -65 mV."""
    cell = bc.Cell.point(pop_size, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV, solver=solver)
    if leak:
        cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    if ion is not None:
        cell.paint(AllRegion(), bc.mech.Ion(ion[0], name=ion[1]))
    cell.paint(AllRegion(), *channels)
    cell.record("v", bc.observe.state("V"))
    return cell


def _cav1p2(**kwargs):
    return bc.mech.Channel("Cav1p2_MA2020_GoC", ion_name="ca", **kwargs)


def _stiff_warnings(cell, batch_size=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cell.init_state(batch_size)
    return [w for w in caught if issubclass(w.category, UserWarning) and STIFF_WARNING in str(w.message)]


class LumpedStiffGateTest(unittest.TestCase):
    """CELL-29: an HH gate too fast for an explicit solver warns at init_state.

    ``Cav1p2_MA2020_GoC``'s m gate relaxes at 2.27e6 /ms at -65 mV, so the
    default ``ralston2`` is stable only for dt below 2 / 2.27e6 = 8.8e-7 ms;
    every practical dt gives NaN within a few steps (dt 0.025 ms: step 4).
    """

    def test_default_solver_names_the_channel_and_the_stable_solvers(self) -> None:
        cell = _stiff_gate_cell(_cav1p2())
        (warning,) = _stiff_warnings(cell)
        message = str(warning.message)
        for part in (
            "'ralston2'",
            "Cav1p2_MA2020_GoC",
            "gate 'm'",
            "3.66e+06 /ms at -100 mV",
            "5.5e-07 ms",
            "'ind_exp_euler'",
            "'exp_euler'",
        ):
            self.assertIn(part, message)
        self.assertEqual(warning.filename, __file__)

    def test_the_suggested_solvers_run_without_a_warning(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for solver in ("ind_exp_euler", "exp_euler"):
                with self.subTest(solver=solver):
                    cell = _stiff_gate_cell(_cav1p2(), solver=solver)
                    self.assertEqual(_stiff_warnings(cell), [])
                    v = cell.run(dt=DT, duration=DURATION).samples["v"].values.to_decimal(u.mV)
                    self.assertAlmostEqual(float(v[-1, 0]), -64.9349, places=3)

    def test_every_explicit_solver_warns_and_no_other_does(self) -> None:
        self.assertGreater(len(EXPLICIT_SOLVERS), 15)
        for name in bc.quad.list_integrators():
            if name in ("staggered", "dhs_voltage"):
                continue
            with self.subTest(solver=name):
                found = _stiff_warnings(_stiff_gate_cell(_cav1p2(), solver=name))
                self.assertEqual(len(found), 1 if name in EXPLICIT_SOLVERS else 0)
        with self.subTest(solver="callable"):
            # A user step owns the whole step; the Cell cannot tell how it integrates.
            self.assertEqual(_stiff_warnings(_stiff_gate_cell(_cav1p2(), solver=lambda cell: None)), [])
        with self.subTest(solver="rk4_step"):
            self.assertEqual(len(_stiff_warnings(_stiff_gate_cell(_cav1p2(), solver=bc.quad.rk4_step))), 1)

    def test_the_rate_is_scanned_over_the_voltage_range(self) -> None:
        # HCN1's h gate (class defaults) relaxes at 2.4e-3 /ms at -65 mV and 8.7e4 /ms at +60 mV: a
        # lumped cell at rest stays finite under ralston2, and diverges once it spikes.
        cases = (
            ("HCN1_MA2024_PC", None, "8.68e+04 /ms at 60 mV"),
            ("Na_ZH2019_IO", ("SodiumFixed", "na"), "1e+03 /ms at"),
            ("Cav1p3_MA2020_GoC", ("CalciumFixed", "ca"), "1.47e+06 /ms at -100 mV"),
        )
        for name, ion, expected in cases:
            with self.subTest(channel=name):
                (warning,) = _stiff_warnings(_stiff_gate_cell(bc.mech.Channel(name), ion=ion))
                self.assertIn(name, str(warning.message))
                self.assertIn(expected, str(warning.message))

    def test_one_warning_names_every_stiff_channel_once(self) -> None:
        cell = _stiff_gate_cell(
            _cav1p2(name="cav12"),
            bc.mech.Channel("Cav1p3_MA2020_GoC", ion_name="ca"),
            bc.mech.Channel("Kv3p4_MA2020_GoC"),
        )
        (warning,) = _stiff_warnings(cell)
        message = str(warning.message)
        self.assertIn("'cav12' (Cav1p2_MA2020_GoC)", message)
        self.assertIn("Cav1p3_MA2020_GoC", message)
        self.assertNotIn("Kv3p4", message)
        # reset_state does not repeat the check.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cell.reset_state()
        self.assertEqual([w for w in caught if STIFF_WARNING in str(w.message)], [])

    def test_population_batch_and_kinetic_ion(self) -> None:
        with self.subTest(case="pop_size (2, 3), batch 2"):
            self.assertEqual(len(_stiff_warnings(_stiff_gate_cell(_cav1p2(), pop_size=(2, 3)), batch_size=2)), 1)
        with self.subTest(case="pop_size 2000, scanned in rounds"):
            (warning,) = _stiff_warnings(_stiff_gate_cell(_cav1p2(), pop_size=2000))
            self.assertIn("3.66e+06 /ms at -100 mV", str(warning.message))
        with self.subTest(case="dependent channel of a kinetic ion"):
            cell = _stiff_gate_cell(_cav1p2(), ion=("CdpStC_MA2020_GoC", "ca"))
            self.assertEqual(len(_stiff_warnings(cell)), 1)

    def test_non_stiff_channels_do_not_warn_and_are_unchanged(self) -> None:
        # CaHVA_MA2020_GoC's s gate reaches 78.4 /ms (at +60 mV): 78.4 * 0.025 = 1.96, just inside the bound.
        cases = (
            ("Kv3p4_MA2020_GoC", (bc.mech.Channel("Kv3p4_MA2020_GoC"),), ("PotassiumFixed", "k"), True),
            ("CaHVA_MA2020_GoC", (bc.mech.Channel("CaHVA_MA2020_GoC", ion_name="ca"),), ("CalciumFixed", "ca"), True),
            ("HH1952", _mechanisms(True), None, False),
        )
        for name, channels, ion, leak in cases:
            with self.subTest(channel=name):
                checked = _stiff_gate_cell(*channels, ion=ion, leak=leak)
                self.assertEqual(_stiff_warnings(checked), [])
                unchecked = _stiff_gate_cell(*channels, ion=ion, leak=leak)
                with mock.patch.object(cell_module, "warn_stiff_gates") as check:
                    unchecked.init_state()
                check.assert_called_once()
                np.testing.assert_array_equal(
                    checked.run(dt=DT, duration=DURATION).samples["v"].values.to_decimal(u.mV),
                    unchecked.run(dt=DT, duration=DURATION).samples["v"].values.to_decimal(u.mV),
                )

    def test_large_populations_cover_the_grid_in_rounds(self) -> None:
        grid = lumped.STIFFNESS_GRID_MV
        self.assertEqual(lumped._voltage_grid((817,)).shape, (grid.size, 1))
        self.assertEqual(lumped._voltage_grid((2, 3)).shape, (grid.size, 1, 1))
        for shape, rounds in (((818,), 160), ((2, 1000), 65), ((200_000,), 1)):
            with self.subTest(shape=shape):
                voltages = lumped._voltage_grid(shape)
                self.assertEqual(voltages.shape, (rounds,) + shape)
                for layout in voltages:
                    # Every round covers the whole grid ...
                    np.testing.assert_array_equal(np.unique(layout), grid)
                # ... and each member meets one distinct voltage per round.
                per_member = voltages.reshape(rounds, -1)
                self.assertEqual(
                    min(len(np.unique(per_member[:, i])) for i in range(0, per_member.shape[1], 97)), rounds
                )

    def test_a_rate_that_cannot_be_scanned_is_skipped(self) -> None:
        # The scan evaluates the rates on a voltage grid; a class whose rates cannot be
        # evaluated there must not break init_state, so its gates are left out of the check.
        cav1p2 = bc.channel.Cav1p2_MA2020_GoC
        with mock.patch.object(cav1p2, "_gate_relaxation_rates", side_effect=ValueError("no grid")) as rates:
            cell = _stiff_gate_cell(_cav1p2(), bc.mech.Channel("Cav1p3_MA2020_GoC", ion_name="ca"))
            (warning,) = _stiff_warnings(cell)
        rates.assert_called_once()
        self.assertNotIn("Cav1p2", str(warning.message))
        self.assertIn("Cav1p3_MA2020_GoC", str(warning.message))


class LumpedSiteTest(unittest.TestCase):
    """Acceptance 4: the one site, and Cell subclasses."""

    def test_place_without_locset_targets_the_midpoint(self) -> None:
        implicit = _point_cell(clamp=True)
        explicit = bc.Cell.point(N, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV)
        explicit.paint(AllRegion(), *_mechanisms(True))
        explicit.place(RootLocation(0.5), _clamp())
        explicit.record("v", bc.observe.state("V"))
        midpoints = bc.Cell.point(N, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV)
        midpoints.paint(AllRegion(), *_mechanisms(True))
        midpoints.place(midpoints.cv_midpoints, _clamp())
        midpoints.record("v", bc.observe.state("V"))
        traces = []
        for cell in (implicit, explicit, midpoints):
            cell.init_state()
            traces.append(cell.run(dt=DT, duration=10.0 * u.ms).samples["v"].values.to_decimal(u.mV))
        np.testing.assert_array_equal(traces[0], traces[1])
        np.testing.assert_array_equal(traces[0], traces[2])

    def test_cell_subclass_matches_point_cell(self) -> None:
        class HH(bc.Cell):
            def __init__(self, n):
                super().__init__(
                    bc.Morphology.from_soma(length=LENGTH, radius=RADIUS),
                    pop_size=n,
                    cv_policy=bc.Lumped(),
                    V_init=-65.0 * u.mV,
                )
                self.paint(AllRegion(), *_mechanisms(True))
                self.place(_clamp())
                self.record("v", bc.observe.state("V"))

        subclass = HH(N)
        reference = _point_cell(clamp=True)
        results = []
        for cell in (subclass, reference):
            cell.init_state()
            results.append(cell.run(dt=DT, duration=10.0 * u.ms).samples["v"].values.to_decimal(u.mV))
        np.testing.assert_array_equal(results[0], results[1])


class LumpedCounterexampleTest(unittest.TestCase):
    """Acceptance 5: each failure names the step that raises and its type."""

    def test_place_rejects_locations_other_than_the_midpoint(self) -> None:
        for x in (0.0, 1.0, 0.3):
            with self.subTest(x=x):
                cell = bc.Cell.point(N)
                with self.assertRaisesRegex(ValueError, "single site"):
                    cell.place(RootLocation(x), _clamp())
                self.assertEqual(cell.place_rules, ())

    def test_assigning_lumped_checks_the_placed_sites(self) -> None:
        # CELL-43 (D19): assigning Lumped() runs the check place runs on a
        # lumped Cell, and a failure leaves the cell as it was.
        sites = {
            "ExpSyn": lambda: bc.mech.Synapse("ExpSyn", name="syn"),
            "CurrentClamp": lambda: bc.mech.CurrentClamp(delay=0.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.1 * u.nA),
            "Junction": lambda: bc.mech.Junction(params={"g": 1.0}),
        }

        def cable_cell(mechanism):
            cell = bc.Cell(
                bc.Morphology.from_soma(length=LENGTH, radius=RADIUS),
                cv_policy=bc.CVPerBranch(),
                V_init=-65.0 * u.mV,
            )
            cell.paint(AllRegion(), *_mechanisms(True))
            cell.place(RootLocation(0.2), mechanism)
            return cell

        def voltage(cell):
            cell.init_state()
            cell.run(dt=DT, duration=10 * DT)
            return cell.V.value.to_decimal(u.mV)

        for name, make in sites.items():
            with self.subTest(mechanism=name):
                with self.assertRaises(ValueError) as expected:
                    bc.Cell.point(1, length=LENGTH, radius=RADIUS).place(RootLocation(0.2), make())
                cell = cable_cell(make())
                policy = cell.cv_policy
                with self.assertRaises(ValueError) as caught:
                    cell.cv_policy = bc.Lumped()
                self.assertEqual(str(caught.exception), str(expected.exception))
                self.assertIs(cell.cv_policy, policy)
                self.assertEqual(cell.n_cv, 1)
                self.assertFalse(cell._is_lumped)
                np.testing.assert_allclose(voltage(cell), voltage(cable_cell(make())), rtol=1e-12)

    def test_assigning_lumped_keeps_midpoint_sites(self) -> None:
        # CELL-43: a site at the CV midpoint passes, however it was written.
        for name, place in {
            "RootLocation(0.5)": lambda cell: cell.place(RootLocation(0.5), _clamp()),
            "cv_midpoints": lambda cell: cell.place(cell.cv_midpoints, _clamp()),
        }.items():
            with self.subTest(locset=name):
                cell = bc.Cell(bc.Morphology.from_soma(length=LENGTH, radius=RADIUS), cv_policy=bc.CVPerBranch())
                place(cell)
                cell.cv_policy = bc.Lumped()
                self.assertIsInstance(cell.cv_policy, bc.Lumped)
                self.assertEqual(len(cell.point_placements), 1)

    def test_place_needs_a_locset_on_a_cable_cell(self) -> None:
        cell = bc.Cell(bc.Morphology.from_soma())
        with self.assertRaisesRegex(TypeError, "needs a locset"):
            cell.place(_clamp())

    def test_lumped_rejects_branched_morphology(self) -> None:
        morpho = bc.Morphology.from_soma()
        morpho.soma.dend = bc.Branch.from_lengths(lengths=[50.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
        with self.assertRaisesRegex(ValueError, "single-branch"):
            bc.Cell(morpho, cv_policy=bc.Lumped())
        cell = bc.Cell(morpho)
        with self.assertRaisesRegex(ValueError, "single-branch"):
            cell.cv_policy = bc.Lumped()
        self.assertIsInstance(cell.cv_policy, bc.CVPerBranch)
        self.assertEqual(cell.solver_name, "staggered")
        self.assertEqual(cell.n_cv, 2)

    def test_lumped_rejects_cable_only_solvers(self) -> None:
        for solver in ("staggered", "stagger", "dhs_voltage"):
            with self.subTest(solver=solver):
                with self.assertRaisesRegex(ValueError, "lumped Cell"):
                    bc.Cell.point(N, solver=solver)
        cell = bc.Cell.point(N)
        with self.assertRaisesRegex(ValueError, "lumped Cell"):
            cell.solver = "staggered"
        self.assertEqual(cell.solver_name, "ralston2")
        cable = bc.Cell(bc.Morphology.from_soma(), solver="staggered")
        with self.assertRaisesRegex(ValueError, "lumped Cell"):
            cable.cv_policy = bc.Lumped()
        self.assertIsInstance(cable.cv_policy, bc.CVPerBranch)

    def test_cable_uses_work(self) -> None:
        # CELL-21b raised NotImplementedError for each; the values are checked
        # against the one-CV twin in lumped_twin_test.py.
        cell = _point_cell()
        cell.init_state()
        cases = {
            "channels": lambda: cell.channels,
            "ions": lambda: cell.ions,
            "synapses": lambda: cell.synapses,
            "geometry (trainable geometry)": lambda: cell.geometry,
            "on": lambda: cell.on(AllRegion()),
            "cv": lambda: cell.cv[0],
            "soma": lambda: cell.soma,
            "population view": lambda: cell[0],
        }
        for name, use in cases.items():
            with self.subTest(use=name):
                self.assertIsNotNone(use())

    def test_cable_declarations_are_accepted(self) -> None:
        cell = bc.Cell.point(N)
        cell.paint(AllRegion(), *_mechanisms(False))
        cell.place(bc.mech.Synapse("Exp2Syn", name="syn"))
        cell.add_reduction("reduced", _NoopReduction())
        cell.record("i", bc.observe.membrane_current())
        cell.init_state()
        result = cell.run(dt=DT, duration=2 * DT)
        self.assertEqual(result.samples["i"].values.shape, (2, N))
        self.assertEqual(len(cell.synapses["syn"]), N)
        cable = bc.Cell(bc.Morphology.from_soma())
        cable.place(RootLocation(0.5), bc.mech.Synapse("Exp2Syn", name="syn"))
        cable.cv_policy = bc.Lumped()
        self.assertIsInstance(cable.cv_policy, bc.Lumped)

    def test_bad_spatial_parameters_raise_at_init_state(self) -> None:
        cases = (
            (ValueError, "raised ZeroDivisionError", lambda cv: 1 / 0),
            (TypeError, "must return a scalar", lambda cv: np.ones(2) * u.mS / u.cm**2),
            (TypeError, "numeric scalar or a Quantity", lambda cv: "x"),
        )
        for error, message, g_max in cases:
            with self.subTest(message=message):
                cell = bc.Cell.point(N)
                cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=g_max, E=-54.3 * u.mV))
                with self.assertRaisesRegex(error, message):
                    cell.init_state()

    def test_declared_parameter_units_raise_at_init_state_as_on_a_cable_cell(self) -> None:
        # MECH-41: the lumped build passed declared values to the runtime
        # classes unchecked. A wrong dimension failed on the first step with
        # a UnitMismatchError naming no parameter, and a bare reversal
        # potential (``SodiumFixed`` ``E=50.0``) ran without any error.
        def init_error(declaration, cv_policy):
            cell = bc.Cell(bc.Morphology.from_soma(length=15.0 * u.um, radius=7.5 * u.um), cv_policy=cv_policy)
            cell.paint(AllRegion(), declaration)
            with self.assertRaises((TypeError, ValueError)) as caught:
                cell.init_state()
            return caught.exception

        with warnings.catch_warnings():
            # MECH-26 warns at each literal declaration; init_state must raise.
            warnings.simplefilter("ignore", FutureWarning)
            cases = (
                ("d", bc.mech.Ion("SodiumDetailed", name="na", d=1.0)),
                ("d", bc.mech.Ion("SodiumDetailed", name="na", d=1.0 * u.mV)),
                ("tau", bc.mech.Ion("CalciumDetailed", name="ca", tau=5.0 * u.mV)),
                ("E", bc.mech.Ion("SodiumFixed", name="na", E=50.0)),
                ("g_max", bc.mech.Channel("IL", name="leak", g_max=0.1 * u.mV)),
                ("g_max", bc.mech.Channel("IL", name="leak", g_max=0.1)),
                ("g_max", bc.mech.Channel("IL", name="leak", g_max=lambda ctx: 0.1)),
            )
        for index, (field, declaration) in enumerate(cases):
            with self.subTest(case=index, parameter=field):
                cable = init_error(declaration, bc.CVPerBranch(1))
                lumped = init_error(declaration, bc.Lumped())
                self.assertIs(type(lumped), type(cable))
                self.assertEqual(str(lumped), str(cable))
                self.assertIn(f"parameter '{field}'", str(lumped))
                # "Painted on", or "painted on" after the spatial callable's sentence.
                self.assertIn("ainted on AllRegion() at lumped_test.py:", str(lumped))

    def test_channel_initializer_is_still_sampled_by_the_channel(self) -> None:
        # MECH-41 checks values, not braintools initializers; the lumped build
        # hands a Channel initializer to the class, which samples it.
        traces = []
        for g_max in (0.3 * u.mS / u.cm**2, braintools.init.Constant(0.3 * u.mS / u.cm**2)):
            cell = bc.Cell.point(N, V_init=-60.0 * u.mV)
            cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=g_max, E=-54.3 * u.mV))
            cell.record("v", bc.observe.state("V"))
            cell.init_state()
            traces.append(cell.run(dt=DT, duration=1.0 * u.ms).samples["v"].values.to_decimal(u.mV))
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_bad_ion_declarations_raise_at_init_state(self) -> None:
        # MECH-26: the unknown parameter warns at the declaration first.
        with self.assertWarnsRegex(FutureWarning, "Ion 'SodiumFixed' got an unexpected parameter 'foo'"):
            unknown_param = bc.mech.Ion("SodiumFixed", foo=1.0)
        cases = (
            ("canonical family key", bc.mech.Ion("SodiumFixed", name="k")),
            ("unsupported runtime ion params", unknown_param),
        )
        for message, ion in cases:
            with self.subTest(message=message):
                cell = bc.Cell.point(N)
                cell.paint(AllRegion(), ion)
                with self.assertRaisesRegex(ValueError, message):
                    cell.init_state()

    def test_switching_to_lumped_keeps_cable_declarations(self) -> None:
        with_reduction = bc.Cell(bc.Morphology.from_soma())
        with_reduction.add_reduction("reduced", _NoopReduction())
        with_junction = bc.Cell(bc.Morphology.from_soma())
        with_junction.place(RootLocation(0.5), bc.mech.Junction(params={"g": 1.0}))
        for name, cell in (("reduction", with_reduction), ("junction", with_junction)):
            with self.subTest(declaration=name):
                cell.cv_policy = bc.Lumped()
                self.assertIsInstance(cell.cv_policy, bc.Lumped)
                cell.init_state()

    def test_network_accepts_lumped_cell(self) -> None:
        network = bc.Network("lumped")
        population = network.add_population("points", bc.Cell.point(N))
        self.assertEqual(population.size, N)


# ---------------------------------------------------------------------------
# CELL-23: external input -- update(I_ext) and brainpy projections.
# ---------------------------------------------------------------------------

#: CELL-23 acceptance 1: the caller's drive, a current per member.
_DRIVE_SCALE = np.asarray([0.5, 1.0, 1.5])
_BOUNDS = {"rk4": 1e-10, "euler": 1e-10, "ralston2": 1e-10, "exp_euler": 1e-9}
_STEPS = int(round(float(DURATION / DT)))


def _drive(t):
    """0.1 nA * (1 + sin(2 pi t / 5 ms)) * [0.5, 1, 1.5]."""
    return 0.1 * u.nA * (1.0 + u.math.sin(2.0 * np.pi * t / (5.0 * u.ms))) * _DRIVE_SCALE


def _input_clamp():
    return bc.mech.CurrentClamp(delay=2.0 * u.ms, durations=5.0 * u.ms, amplitudes=0.05 * u.nA)


def _input_clamp_density(t, area):
    """The clamp above at the step midpoint, as a density over ``area``."""
    mid = t + 0.5 * DT
    on = (mid >= 2.0 * u.ms) & (mid < 7.0 * u.ms)
    return u.math.where(on, 0.05, 0.0) * u.nA / area


def _voltage_before_each_step(model, step):
    """Run ``step(i, t)`` for every step under ``for_loop``; return V before each."""

    def body(index):
        t = index * DT
        with brainstate.environ.context(t=t, dt=DT):
            before = model.V.value
            step(index, t)
            return before

    with brainstate.environ.context(dt=DT):
        return brainstate.transform.for_loop(body, np.arange(_STEPS))


def _max_dv(a, b) -> float:
    return float(u.math.max(u.math.abs(a - b)).to_decimal(u.mV))


def _raster(seed):
    """Spikes of 20 sources, each firing with p = 0.05 per step."""
    return jax.numpy.asarray((np.random.default_rng(seed).random((_STEPS, 20)) < 0.05).astype(float))


def _weights(seed):
    return np.random.default_rng(seed).uniform(0.5, 1.5, size=(20, N))


class _Dense(brainstate.nn.Module):
    """``spikes @ W``, the communication of the projection tests."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def update(self, spikes):
        return u.math.matmul(spikes, self.weight)


def _run_projection(post, proj, raster):
    """Initialise through the projection (it reaches ``post`` once) and drive it."""
    brainstate.nn.init_all_states(proj)
    if isinstance(post, PointNeuron):
        post.reset_state()

    def step(index, t):
        proj(raster[index])
        post()

    return _voltage_before_each_step(post, step)


def _align_post(post, weight, g_unit):
    return brainpy.state.AlignPostProj(
        comm=_Dense(weight),
        syn=brainpy.state.Expon.desc(N, tau=5.0 * u.ms, g_initializer=braintools.init.Constant(0.0 * g_unit)),
        out=brainpy.state.COBA.desc(E=0.0 * u.mV),
        post=post,
    )


class ExternalInputEquivalenceTest(unittest.TestCase):
    """CELL-23 acceptance 1: every drive matches a hand-built PointNeuron step by step in float64."""

    def _check(self, drive):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for solver, bound in _BOUNDS.items():
                with self.subTest(solver=solver):
                    expected, actual = drive(solver)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertLessEqual(_max_dv(actual, expected), bound)

    def test_total_current(self) -> None:
        def drive(solver):
            model = _point_neuron(hh=True, solver=solver)
            model.init_state()
            model.reset_state()
            area = model.area
            expected = _voltage_before_each_step(
                model,
                lambda i, t: model.update(_drive(t + 0.5 * DT) / area + _input_clamp_density(t, area)),
            )
            cell = _point_cell(solver=solver)
            cell.place(_input_clamp())
            cell.init_state()
            return expected, _voltage_before_each_step(cell, lambda i, t: cell.update(_drive(t + 0.5 * DT)))

        self._check(drive)

    def test_current_density(self) -> None:
        def drive(solver):
            model = _point_neuron(hh=True, solver=solver)
            model.init_state()
            model.reset_state()
            area = model.area

            def density(t):
                return (_drive(t + 0.5 * DT) / area).in_unit(u.uA / u.cm**2)

            expected = _voltage_before_each_step(
                model, lambda i, t: model.update(density(t) + _input_clamp_density(t, area))
            )
            cell = _point_cell(solver=solver)
            cell.place(_input_clamp())
            cell.init_state()
            return expected, _voltage_before_each_step(cell, lambda i, t: cell.update(density(t)))

        self._check(drive)

    def test_batch_axis(self) -> None:
        rows = np.asarray([[1.0], [2.0]])

        def drive(solver):
            model = _point_neuron(hh=True, solver=solver)
            model.init_state(batch_size=2)
            model.reset_state(batch_size=2)
            area = model.area
            expected = _voltage_before_each_step(model, lambda i, t: model.update(_drive(t + 0.5 * DT) * rows / area))
            cell = _point_cell(solver=solver)
            cell.init_state(batch_size=2)
            actual = _voltage_before_each_step(cell, lambda i, t: cell.update(_drive(t + 0.5 * DT) * rows))
            self.assertEqual(actual.shape, (_STEPS, 2, N))
            return expected, actual

        self._check(drive)

    def test_align_post_projection_with_coba(self) -> None:
        raster = _raster(7)
        weight = _weights(11) * 2.0 * u.nS
        for kind in ("current", "density"):

            def drive(solver, kind=kind):
                model = _point_neuron(hh=True, solver=solver)
                density = (weight / model.area).in_unit(u.mS / u.cm**2)
                expected = _run_projection(model, _align_post(model, density, u.mS / u.cm**2), raster)
                cell = _point_cell(solver=solver)
                if kind == "current":
                    proj = _align_post(cell, weight, u.nS)
                else:
                    proj = _align_post(cell, (weight / _area()).in_unit(u.mS / u.cm**2), u.mS / u.cm**2)
                # The input makes the cell fire, so the comparison is not trivial.
                self.assertGreater(float(u.math.max(expected).to_decimal(u.mV)), 0.0)
                return expected, _run_projection(cell, proj, raster)

            with self.subTest(conductance=kind):
                self._check(drive)

    def test_current_projection(self) -> None:
        raster = _raster(3)
        outputs = {
            "COBA": (_weights(5) * 3.0 * u.nS, u.mS / u.cm**2, lambda: brainpy.state.COBA(E=0.0 * u.mV)),
            "CUBA": (_weights(5) * 0.3 * u.nA, u.uA / u.cm**2, lambda: brainpy.state.CUBA(scale=1.0)),
        }
        for name, (weight, density_unit, make_out) in outputs.items():

            def drive(solver, weight=weight, density_unit=density_unit, make_out=make_out):
                model = _point_neuron(hh=True, solver=solver)
                density = (weight / model.area).in_unit(density_unit)
                expected = _run_projection(
                    model, brainpy.state.CurrentProj(comm=_Dense(density), out=make_out(), post=model), raster
                )
                cell = _point_cell(solver=solver)
                proj = brainpy.state.CurrentProj(comm=_Dense(weight), out=make_out(), post=cell)
                return expected, _run_projection(cell, proj, raster)

            with self.subTest(out=name):
                self._check(drive)


class ExternalInputTest(unittest.TestCase):
    """CELL-23 acceptance 2-4, 6: shapes, stages, units and the other entry points."""

    def _cell(self, *, batch_size=None, solver=None, hh=False):
        cell = _point_cell(hh=hh, solver=solver)
        cell.init_state(batch_size=batch_size)
        return cell

    def test_broadcast_over_population_and_batch(self) -> None:
        cases = (
            (None, 0.1 * u.nA, (N,)),
            (None, np.asarray([0.1, 0.2, 0.3]) * u.nA, (N,)),
            (2, np.asarray([0.1, 0.2, 0.3]) * u.nA, (2, N)),
            (2, np.ones((2, N)) * u.uA / u.cm**2, (2, N)),
        )
        for batch_size, current, shape in cases:
            with self.subTest(batch_size=batch_size, shape=u.math.shape(current)):
                cell = self._cell(batch_size=batch_size)
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    spike = cell.update(current)
                self.assertEqual(spike.shape, shape)
                self.assertEqual(cell.V.value.shape, shape)
        # A scalar drives every member alike.
        cell = self._cell()
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            cell.update(0.1 * u.nA)
        voltage = cell.V.value.to_decimal(u.mV)
        np.testing.assert_array_equal(voltage, np.full((N,), voltage[0]))

    def test_shape_that_does_not_broadcast_raises(self) -> None:
        cases = ((None, (2,)), (None, (N, 1)), (2, (2,)))
        for batch_size, shape in cases:
            with self.subTest(batch_size=batch_size, shape=shape):
                cell = self._cell(batch_size=batch_size)
                before = cell.V.value
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    with self.assertRaisesRegex(ValueError, r"voltage shape \(" + str(cell.V.value.shape[0])):
                        cell.update(np.ones(shape) * u.nA)
                np.testing.assert_array_equal(cell.V.value.to_decimal(u.mV), before.to_decimal(u.mV))

    def test_input_is_held_over_the_stages_of_a_step(self) -> None:
        cell = self._cell(solver="rk4")
        stage_voltages = []

        def probe(V):
            stage_voltages.append(np.asarray(V.to_decimal(u.mV)))
            return u.math.zeros_like(u.get_mantissa(V)) * u.uA / u.cm**2

        cell.add_current_input("probe", probe)
        runtime = cell._lumped_runtime
        with mock.patch.object(runtime, "compute_derivative", wraps=runtime.compute_derivative) as spy:
            with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                cell.update(0.1 * u.nA)
        received = [call.args[0] for call in spy.call_args_list]
        self.assertEqual(len(received), 4)
        expected = float((0.1 * u.nA / _area()).to_decimal(u.uA / u.cm**2))
        for value in received:
            np.testing.assert_allclose(value.to_decimal(u.uA / u.cm**2), np.full((N,), expected), rtol=1e-6)
        self.assertEqual(len(stage_voltages), 4)
        self.assertGreater(len({float(v[0]) for v in stage_voltages}), 1)

    def test_wrong_unit_raises_unit_mismatch_at_update(self) -> None:
        for value in (1.0 * u.mV, 0.1):
            with self.subTest(value=value):
                cell = self._cell()
                before = cell.V.value
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    with self.assertRaisesRegex(u.UnitMismatchError, r"current \(e\.g\. nA\) or a current density"):
                        cell.update(value)
                np.testing.assert_array_equal(cell.V.value.to_decimal(u.mV), before.to_decimal(u.mV))

    def test_current_input_with_wrong_unit_raises_naming_its_key(self) -> None:
        cell = self._cell()
        cell.add_current_input("bad_input", lambda V: u.math.zeros_like(u.get_mantissa(V)) * u.mV)
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            with self.assertRaisesRegex(u.UnitMismatchError, "bad_input"):
                cell.update()

    def test_current_input_was_ignored_before(self) -> None:
        # On main before CELL-23 a lumped Cell summed its runtime's (empty)
        # input table, so a current input left V unchanged.
        traces = []
        for value in (None, 10.0 * u.uA / u.cm**2, (10.0 * u.uA / u.cm**2 * _area()).in_unit(u.nA)):
            cell = self._cell()
            if value is not None:
                cell.add_current_input("drive", lambda V, value=value: value)
            with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                cell.update()
            traces.append(cell.V.value.to_decimal(u.mV))
        self.assertGreater(float(np.min(traces[1] - traces[0])), 1e-3)
        np.testing.assert_allclose(traces[2], traces[1], rtol=0.0, atol=1e-5)

    def test_value_input_is_added_once_as_on_point_neuron(self) -> None:
        # brainpy's rule: a non-callable input is added at the first
        # evaluation (the first solver stage) and then removed.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            model = _point_neuron(hh=True, solver="rk4")
            model.init_state()
            model.reset_state()
            model.add_current_input("kick", (0.2 * u.nA / model.area).in_unit(u.uA / u.cm**2))
            cell = _point_cell(solver="rk4")
            cell.init_state()
            cell.add_current_input("kick", 0.2 * u.nA)
            voltages = []
            for target in (model, cell):
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    target.update()
                    target.update()
                voltages.append(target.V.value.to_decimal(u.mV))
                self.assertNotIn("kick", target.current_inputs)
            self.assertGreater(float(np.min(voltages[1] + 65.0)), 1e-3)
            np.testing.assert_allclose(voltages[1], voltages[0], rtol=0.0, atol=1e-10)

    def test_direct_solver_call_and_call_operator_match_update(self) -> None:
        raster = _raster(3)
        weight = _weights(5) * 3.0 * u.nS
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            results = {}
            for entry in ("update", "rk4_step"):
                cell = _point_cell(solver="rk4")
                proj = brainpy.state.CurrentProj(comm=_Dense(weight), out=brainpy.state.COBA(E=0.0 * u.mV), post=cell)
                brainstate.nn.init_all_states(proj)
                voltages = []
                for index in range(2):
                    with brainstate.environ.context(t=index * DT, dt=DT):
                        proj(raster[2 * index + 30])
                        if entry == "update":
                            cell.update()
                        else:
                            bc.quad.rk4_step(cell)
                    voltages.append(cell.V.value.to_decimal(u.mV))
                results[entry] = np.stack(voltages)
            self.assertGreater(float(np.max(np.abs(results["update"] + 65.0))), 1e-6)
            np.testing.assert_array_equal(results["rk4_step"], results["update"])

            called = []
            for entry in ("update", "call"):
                cell = _point_cell(solver="rk4")
                cell.init_state()
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    if entry == "update":
                        cell.update(0.2 * u.nA)
                    else:
                        cell(0.2 * u.nA)
                called.append(cell.V.value.to_decimal(u.mV))
            np.testing.assert_array_equal(called[1], called[0])

    def test_run_includes_a_current_input_added_after_a_run(self) -> None:
        def drive(V):
            return u.math.full_like(u.get_mantissa(V), 10.0) * u.uA / u.cm**2

        finals = []
        for late in (False, True):
            cell = self._cell()
            if not late:
                cell.add_current_input("drive", drive)
            cell.run(dt=DT, duration=1.0 * u.ms)
            if late:
                cell.add_current_input("drive", drive)
            cell.run(dt=DT, duration=1.0 * u.ms)
            finals.append(cell.V.value.to_decimal(u.mV))
        idle = self._cell()
        idle.run(dt=DT, duration=2.0 * u.ms)
        # Added after the first run, the input still moves the second one.
        self.assertGreater(float(np.min(finals[1] - idle.V.value.to_decimal(u.mV))), 1.0)
        self.assertGreater(float(np.min(finals[0] - finals[1])), 0.0)


class ExternalInputOnCableCellTest(unittest.TestCase):
    """CELL-23 acceptance 5 and 6: a cable Cell rejects I_ext; run() sees late current inputs."""

    @staticmethod
    def _cable_cell(cv_count=3, g_max=0.1 * u.mS / u.cm**2):
        branch = bc.Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 2.0] * u.um, type="dendrite")
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(cv_count), V_init=-65.0 * u.mV)
        cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=g_max, E=-65.0 * u.mV))
        return cell

    def test_update_with_input_raises_type_error_pointing_to_clamps(self) -> None:
        cell = self._cable_cell()
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            with self.assertRaisesRegex(TypeError, r"CurrentClamp.*FunctionClamp"):
                cell.update(0.1 * u.nA)
            cell.update(None)
        after_none = cell.V.value.to_decimal(u.mV)
        reference = self._cable_cell()
        reference.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            reference.update()
        np.testing.assert_array_equal(after_none, reference.V.value.to_decimal(u.mV))

        # The recovery the message gives: reset(), place a clamp, init again.
        cell.reset()
        cell.place(RootLocation(0.5), bc.mech.FunctionClamp(fn=lambda t: 0.01 * u.nA))
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            cell.update()
        middle = cell.V.value.to_decimal(u.mV)[0, 1]
        self.assertGreater(float(middle), float(reference.V.value.to_decimal(u.mV)[0, 1]) + 1e-4)

    def test_run_includes_a_current_input_added_after_a_run(self) -> None:
        def drive(V):
            return u.math.full_like(u.get_mantissa(V), 10.0) * u.uA / u.cm**2

        cell = self._cable_cell(cv_count=1, g_max=0.3 * u.mS / u.cm**2)
        cell.init_state()
        cell.run(dt=DT, duration=1.0 * u.ms)
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -65.0)
        cell.add_current_input("drive", drive)
        cell.run(dt=DT, duration=1.0 * u.ms)
        self.assertGreater(float(np.min(cell.V.value.to_decimal(u.mV))), -60.0)


# ---------------------------------------------------------------------------
# CELL-33: delta inputs -- add_delta_input and brainpy DeltaProj.
# ---------------------------------------------------------------------------

#: The 5 mV kick of CELL-33 acceptance 1.
_KICK = 5.0 * u.mV


def _resting_leak():
    """Leak at its reversal: without input V stays at -65 mV on every solver."""
    return bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV)


def _one_cv_cell(kind, solver, *, pop_size=1, hh=False):
    if kind == "lumped":
        cell = bc.Cell.point(pop_size, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV, solver=solver)
    else:
        morphology = bc.Morphology.from_soma(length=LENGTH, radius=RADIUS)
        cell = bc.Cell(morphology, pop_size=pop_size, cv_policy=bc.CVPerBranch(), V_init=-65.0 * u.mV, solver=solver)
    cell.paint(AllRegion(), *(_mechanisms(True) if hh else (_resting_leak(),)))
    return cell


def _runnable_one_cv_cells(kind, **kwargs):
    """``(solver, cell)`` for every registered integrator that can step ``kind``."""
    for solver in bc.quad.list_integrators():
        try:
            cell = _one_cv_cell(kind, solver, **kwargs)
        except ValueError:
            # Only the cable-tree steps cannot advance a lumped cell.
            if kind != "lumped" or solver not in {"staggered", "dhs_voltage"}:
                raise
            continue
        yield solver, cell


def _resting_point_neuron(solver):
    class Model(PointNeuron):
        def __init__(self):
            super().__init__(1, V_initializer=braintools.init.Constant(-65.0 * u.mV), solver=solver)
            self.IL = bc.channel.IL(1, g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV)

    model = Model()
    model.init_state()
    model.reset_state()
    return model


def _gates(cell):
    """Every channel state of a one-CV cell, as plain arrays keyed by path."""
    return {
        str(path): np.asarray(u.get_mantissa(state.value))
        for path, state in cell.states(brainstate.HiddenState).items()
        if state is not cell.V
    }


def _kick_projection(post, weight):
    return brainpy.state.DeltaProj(comm=_Dense(weight), post=post)


class DeltaInputTest(unittest.TestCase):
    """CELL-33: a delta input moves V on every path, at the end of the step as on PointNeuron."""

    def test_kick_matches_point_neuron_on_every_solver(self) -> None:
        # Acceptance 1. On main the lumped cell (every solver) and the cable
        # cell under "staggered" and "dhs_voltage" stayed at -65 mV.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            model = _resting_point_neuron("rk4")
            model.add_delta_input("kick", _KICK)
            with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                model.update()
            expected = float(model.V.value.to_decimal(u.mV)[0])
            self.assertAlmostEqual(expected, -60.0, places=12)
            checked = {}
            for kind in ("lumped", "cable"):
                for solver, cell in _runnable_one_cv_cells(kind):
                    with self.subTest(kind=kind, solver=solver):
                        cell.init_state()
                        cell.add_delta_input("kick", _KICK)
                        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                            cell.update()
                        self.assertLessEqual(abs(float(cell.V.value.to_decimal(u.mV).reshape(-1)[0]) - expected), 1e-6)
                        # A value is applied once and removed, as brainpy does.
                        self.assertNotIn("kick", cell.delta_inputs)
                        checked.setdefault(kind, []).append(solver)
            self.assertEqual(len(checked["cable"]), len(bc.quad.list_integrators()))
            self.assertEqual(len(checked["lumped"]), len(bc.quad.list_integrators()) - 2)

    def test_kick_is_added_after_the_step_dynamics(self) -> None:
        # The delta does not take part in the step it is applied in: V ends
        # exactly 5 mV above the unkicked step and every gate is unchanged,
        # on PointNeuron and on every path.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind in ("lumped", "cable"):
                for solver, plain in _runnable_one_cv_cells(kind, hh=True):
                    with self.subTest(kind=kind, solver=solver):
                        kicked = _one_cv_cell(kind, solver, hh=True)
                        for cell in (plain, kicked):
                            cell.init_state()
                        kicked.add_delta_input("kick", _KICK)
                        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                            plain.update()
                            kicked.update()
                        np.testing.assert_allclose(
                            kicked.V.value.to_decimal(u.mV) - plain.V.value.to_decimal(u.mV), 5.0, rtol=0, atol=1e-9
                        )
                        gates = _gates(plain)
                        self.assertTrue(gates)
                        for path, value in _gates(kicked).items():
                            np.testing.assert_array_equal(value, gates[path], err_msg=path)

    def test_delta_projection_matches_point_neuron(self) -> None:
        # Acceptance 2: brainpy DeltaProj drives both models with the same raster.
        raster = _raster(13)
        weight = _weights(17) * 4.0 * u.mV

        def drive(solver):
            model = _point_neuron(hh=True, solver=solver)
            expected = _run_projection(model, _kick_projection(model, weight), raster)
            # The kicks make the cell fire, so the comparison is not trivial.
            self.assertGreater(float(u.math.max(expected).to_decimal(u.mV)), 0.0)
            cell = _point_cell(solver=solver)
            return expected, _run_projection(cell, _kick_projection(cell, weight), raster)

        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for solver, bound in _BOUNDS.items():
                with self.subTest(solver=solver):
                    expected, actual = drive(solver)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertLessEqual(_max_dv(actual, expected), bound)

    def test_run_includes_a_delta_input_added_after_a_run(self) -> None:
        # Acceptance 2: run() compiles its loop once; a delta input registered
        # later must drop it (on main the cable "rk4" cell ignored it too).
        def kick():
            return 0.25 * u.mV

        for kind, solver in (("lumped", "rk4"), ("cable", "rk4"), ("cable", "staggered")):
            with self.subTest(kind=kind, solver=solver):
                cell = _one_cv_cell(kind, solver)
                cell.init_state()
                cell.run(dt=DT, duration=1.0 * u.ms)
                np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -65.0)
                cell.add_delta_input("kick", kick)
                cell.run(dt=DT, duration=1.0 * u.ms)
                # 40 kicks of 0.25 mV, partly relaxed by the leak.
                self.assertGreater(float(np.min(cell.V.value.to_decimal(u.mV))), -57.0)

    def test_delta_that_would_reshape_V_raises(self) -> None:
        # On main a cable cell's V took the broadcast shape: (3,) + (3, 1) -> (3, 3).
        cases = (("lumped", "rk4", (N, 1)), ("cable", "rk4", (N,)), ("cable", "staggered", (N,)))
        for kind, solver, shape in cases:
            with self.subTest(kind=kind, solver=solver, shape=shape):
                cell = _one_cv_cell(kind, solver, pop_size=N)
                cell.init_state()
                voltage_shape = cell.V.value.shape
                cell.add_delta_input("kick", np.ones(shape) * u.mV)
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    with self.assertRaisesRegex(ValueError, r"kick.*" + re.escape(str(voltage_shape))):
                        cell.update()
                self.assertEqual(cell.V.value.shape, voltage_shape)
        # Shapes that broadcast to V are accepted: one value per member, or
        # per batch row and member.
        members = np.arange(N, dtype=float)
        accepted = (
            ("lumped", "rk4", None, members * u.mV, -65.0 + members),
            ("cable", "rk4", None, members[:, None] * u.mV, -65.0 + members[:, None]),
            ("cable", "staggered", None, members[:, None] * u.mV, -65.0 + members[:, None]),
            ("lumped", "rk4", 2, np.stack([members, 2 * members]) * u.mV, -65.0 + np.stack([members, 2 * members])),
        )
        for kind, solver, batch_size, value, expected in accepted:
            with self.subTest(kind=kind, solver=solver, batch_size=batch_size, accepted=u.math.shape(value)):
                cell = _one_cv_cell(kind, solver, pop_size=N)
                cell.init_state(batch_size=batch_size)
                cell.add_delta_input("kick", value)
                with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                    cell.update()
                np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), expected, rtol=0.0, atol=1e-5)

    def test_direct_solver_calls_and_the_call_operator_add_the_kick(self) -> None:
        # The solver step itself adds the delta, not only update().
        for kind, solver in (("lumped", "rk4"), ("cable", "rk4"), ("cable", "staggered"), ("cable", "dhs_voltage")):
            for entry in ("solver", "call"):
                with self.subTest(kind=kind, solver=solver, entry=entry):
                    cell = _one_cv_cell(kind, solver)
                    cell.init_state()
                    cell.add_delta_input("kick", _KICK)
                    with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                        if entry == "solver":
                            bc.quad.get_integrator(solver)(cell)
                        else:
                            cell()
                    np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -60.0, rtol=0.0, atol=1e-5)

    def test_runtime_without_its_cell_reads_its_own_inputs(self) -> None:
        # The weak reference is dead once the Cell is gone; the runtime then
        # behaves as the PointNeuron it is.
        cell = _one_cv_cell("lumped", "rk4")
        cell.init_state()
        runtime = cell._lumped_runtime
        runtime._owner_ref = None
        cell.add_delta_input("ignored", _KICK)
        runtime.add_delta_input("own", 2.0 * u.mV)
        runtime.post_integral()
        np.testing.assert_allclose(runtime.V.value.to_decimal(u.mV), -63.0)
        self.assertNotIn("own", runtime.delta_inputs)
        self.assertIn("ignored", cell.delta_inputs)

    def test_reduction_model_rejects_delta_inputs(self) -> None:
        # A reduction has no membrane voltage for the delta to move.
        cell = bc.Cell(bc.Morphology.from_soma())
        cell.add_reduction("reduced", bc.EventAccumulatorReduction(alpha=0.5, threshold=0.05))
        cell.use_model("reduced")
        cell.add_delta_input("kick", _KICK)
        with self.assertRaisesRegex(ValueError, r"'reduced'.*kick"):
            cell.init_state()
        cell.delta_inputs.clear()
        cell.init_state()
        cell.add_delta_input("later", _KICK)
        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
            with self.assertRaisesRegex(ValueError, r"'reduced'.*later"):
                cell.update()


_PULSE = 10.0 * u.uA / u.cm**2
# The leak relaxes at g / C = 0.3 / ms; one 0.025 ms step of the pulse moves
# V by I / g * (1 - exp(-z)), z = g dt / C = 0.0075.
_PULSE_Z = 0.0075
_PULSE_DV = 100.0 / 3.0 * -np.expm1(-_PULSE_Z)


def _v_mV(cell):
    return float(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)[0])


def _add_value(cell, entry):
    if entry == "delta":
        cell.add_delta_input("kick", _KICK)
    else:
        cell.add_current_input("pulse", _PULSE)


def _update_loop(cell, n_steps, start=0, register=None):
    """``n_steps`` eager ``update()`` calls; ``register(step)`` runs before each."""
    with brainstate.environ.context(dt=DT):
        for step in range(start, start + n_steps):
            if register is not None:
                register(step)
            with brainstate.environ.context(t=step * DT):
                cell.update()
    return _v_mV(cell)


class ValueInputStepTest(unittest.TestCase):
    """CELL-36: a value (non-callable) input acts on one step, in every stage and in every loop."""

    def test_one_step_pulse_charge_does_not_depend_on_the_solver(self) -> None:
        # Acceptance 1. On main brainpy removed the value at its first sum,
        # so a multi-stage solver counted it in its first stage only:
        # ralston2 0.061562, rk4 0.041355 mV instead of 0.249065.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            checked = {}
            for kind in ("lumped", "cable"):
                for solver, cell in _runnable_one_cv_cells(kind):
                    with self.subTest(kind=kind, solver=solver):
                        cell.init_state()
                        cell.add_current_input("pulse", _PULSE)
                        with brainstate.environ.context(t=0.0 * u.ms, dt=DT):
                            cell.update()
                        dv = _v_mV(cell) + 65.0
                        self.assertLessEqual(abs(dv - _PULSE_DV), 1e-3)
                        if solver in ("rk4", "exp_euler"):
                            self.assertLessEqual(abs(dv - _PULSE_DV), 1e-6)
                        if solver == "ralston2":
                            # Both stages count the pulse: the two-stage
                            # Taylor value; its distance to the exact
                            # 0.2490648 mV, 2.3e-6 mV, is the method's error.
                            self.assertAlmostEqual(dv, 0.25 * (1.0 - _PULSE_Z / 2.0), places=12)
                        self.assertNotIn("pulse", cell.current_inputs)
                        checked.setdefault(kind, []).append(solver)
            self.assertEqual(len(checked["cable"]), len(bc.quad.list_integrators()))
            self.assertEqual(len(checked["lumped"]), len(bc.quad.list_integrators()) - 2)

    def test_run_matches_an_update_loop(self) -> None:
        # Acceptance 2. On main the loop of run() added the value at every
        # step, as a constant of its trace, and so did the next run that
        # reused the loop: the delta gave -45.22 and -26.03 mV.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind, solver in (("cable", "rk4"), ("cable", "staggered"), ("lumped", "rk4")):
                for entry in ("delta", "current"):
                    with self.subTest(kind=kind, solver=solver, input=entry):
                        expected = []
                        looped = _one_cv_cell(kind, solver)
                        looped.init_state()
                        _add_value(looped, entry)
                        expected.append(_update_loop(looped, 4))
                        expected.append(_update_loop(looped, 4, start=4))
                        cell = _one_cv_cell(kind, solver)
                        cell.init_state()
                        _add_value(cell, entry)
                        actual = []
                        for _ in range(2):
                            cell.run(dt=DT, duration=4 * DT)
                            actual.append(_v_mV(cell))
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
                        if entry == "delta":
                            self.assertAlmostEqual(actual[0], -60.11, places=1)

    def test_value_registered_between_runs_acts_on_the_first_step_of_the_next(self) -> None:
        # Acceptance 3: the value is an argument of the compiled loop, not a
        # constant of its trace, so registering one does not retrace it.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind, solver in (("cable", "rk4"), ("cable", "staggered"), ("lumped", "rk4")):
                with self.subTest(kind=kind, solver=solver):
                    looped = _one_cv_cell(kind, solver)
                    looped.init_state()
                    _update_loop(looped, 4)
                    looped.add_current_input("pulse", _PULSE)
                    expected = _update_loop(looped, 4, start=4)
                    cell = _one_cv_cell(kind, solver)
                    cell.init_state()
                    cell.run(dt=DT, duration=4 * DT)
                    plans = dict(cell._run_loop_cache)
                    cell.add_current_input("pulse", _PULSE)
                    self.assertEqual(dict(cell._run_loop_cache), plans)
                    cell.run(dt=DT, duration=4 * DT)
                    self.assertAlmostEqual(_v_mV(cell), expected, delta=1e-12 * abs(expected))
                    self.assertGreater(_v_mV(cell), -65.0 + 0.2)
                    # Consumed: a third run relaxes back towards rest.
                    cell.run(dt=DT, duration=4 * DT)
                    self.assertLess(_v_mV(cell), expected)

    def test_callable_input_is_still_evaluated_at_every_step(self) -> None:
        # Acceptance 3: a callable acts on every step of every run.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind, solver in (("cable", "rk4"), ("lumped", "rk4")):
                with self.subTest(kind=kind, solver=solver):
                    looped = _one_cv_cell(kind, solver)
                    looped.init_state()
                    looped.add_current_input("drive", lambda V: _PULSE)
                    expected = _update_loop(looped, 8)
                    cell = _one_cv_cell(kind, solver)
                    cell.init_state()
                    cell.add_current_input("drive", lambda V: _PULSE)
                    cell.run(dt=DT, duration=4 * DT)
                    cell.run(dt=DT, duration=4 * DT)
                    self.assertAlmostEqual(_v_mV(cell), expected, delta=1e-12 * abs(expected))
                    # Eight steps of 0.249 mV, less the leak.
                    self.assertGreater(_v_mV(cell), -65.0 + 1.9)

    def test_value_registered_in_a_loop_body_acts_on_every_iteration(self) -> None:
        # A projection that registers a value at every step (brainpy's
        # DeltaProj, gap junctions) inside a traced loop: each step gets its
        # own value, as in an eager loop.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind, solver in (("cable", "rk4"), ("lumped", "rk4")):
                with self.subTest(kind=kind, solver=solver):
                    looped = _one_cv_cell(kind, solver)
                    looped.init_state()
                    expected = _update_loop(looped, 8, register=lambda step: looped.add_current_input("p", _PULSE))
                    cell = _one_cv_cell(kind, solver)
                    cell.init_state()

                    def body(t, cell=cell):
                        cell.add_current_input("p", _PULSE)
                        with brainstate.environ.context(t=t):
                            cell.update()

                    with brainstate.environ.context(dt=DT):
                        brainstate.transform.for_loop(body, np.arange(8) * DT)
                    self.assertAlmostEqual(_v_mV(cell), expected, delta=1e-12 * abs(expected))
                    self.assertNotIn("p", cell.current_inputs)

    def test_membrane_current_recording_reads_a_value_without_consuming_it(self) -> None:
        # On main compiling the recording evaluated it once, which removed
        # the value: run() never applied it (V stayed at -65 mV).
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for kind, solver in (("cable", "rk4"), ("cable", "staggered"), ("lumped", "rk4")):
                with self.subTest(kind=kind, solver=solver):
                    plain = _one_cv_cell(kind, solver)
                    plain.init_state()
                    plain.add_current_input("pulse", _PULSE)
                    plain.run(dt=DT, duration=4 * DT)
                    cell = _one_cv_cell(kind, solver)
                    cell.record("im", bc.observe.membrane_current())
                    cell.init_state()
                    cell.add_current_input("pulse", _PULSE)
                    result = cell.run(dt=DT, duration=4 * DT)
                    self.assertEqual(_v_mV(cell), _v_mV(plain))
                    self.assertGreater(_v_mV(cell), -65.0 + 0.2)
                    current = np.asarray(result.samples["im"].values.to_decimal(u.uA / u.cm**2)).reshape(4)
                    # Sampled before each step: the first holds the pulse
                    # (the leak is zero at rest), the others only the leak.
                    self.assertAlmostEqual(current[0], 10.0, places=9)
                    self.assertLess(np.max(np.abs(current[1:])), 0.1)

    def test_a_run_that_fails_keeps_its_value_inputs(self) -> None:
        cell = _one_cv_cell("lumped", "rk4", pop_size=N)
        cell.init_state()
        cell.add_current_input("pulse", np.ones((N + 1,)) * _PULSE)
        with self.assertRaisesRegex(ValueError, "pulse"):
            cell.run(dt=DT, duration=4 * DT)
        self.assertIn("pulse", cell.current_inputs)
        del cell.current_inputs["pulse"]
        cell.run(dt=DT, duration=4 * DT)
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -65.0)

    def test_reduction_model_leaves_value_inputs_registered(self) -> None:
        # A reduction has no membrane: its run neither applies nor takes a
        # current value, as before.
        cell = bc.Cell(bc.Morphology.from_soma())
        cell.add_reduction("reduced", bc.EventAccumulatorReduction(alpha=0.5, threshold=0.05))
        cell.use_model("reduced")
        cell.init_state()
        cell.add_current_input("pulse", _PULSE)
        cell.run(dt=DT, duration=4 * DT)
        self.assertIn("pulse", cell.current_inputs)


_EXP2SYN = dict(tau1=0.5 * u.ms, tau2=5.0 * u.ms, e=0.0 * u.mV)
_BLOCK_SOLVER_NAMES = ("exp_euler", "crank_nicolson", "backward_euler")


def _lumped_synapse_cell(solver, model, *, lumped_cell=True):
    """The CELL-31c models.

    - ``"P"``: two 20 um x r 10 um members (``IL`` 0.1 mS/cm^2 at -70 mV), an
      Exp2Syn on both and an ExpSyn on member 0 only: two synapse rows in
      member 0, one in member 1. ``lumped_cell=False`` builds the one-CV
      cable twin.
    - ``"G"``: one 20 um x r 0.5 um member, ``GabaA_JE2011`` writing
      ``Cldif_JE2011`` (the ion-current snapshot path).
    """
    if model == "G":
        cell = bc.Cell.point(1, length=20.0 * u.um, radius=0.5 * u.um, V_init=-70.0 * u.mV, solver=solver)
        cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
        cell.paint(AllRegion(), bc.mech.Ion("Cldif_JE2011", name="cl"))
        cell.place(at("soma", 0.5), bc.mech.Synapse("GabaA_JE2011", name="s"))
        return cell
    if lumped_cell:
        cell = bc.Cell.point(2, length=LENGTH, radius=RADIUS, solver=solver)
    else:
        morphology = bc.Morphology.from_soma(length=LENGTH, radius=RADIUS)
        cell = bc.Cell(morphology, pop_size=2, cv_policy=bc.CVPerBranch(), solver=solver)
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.place(at("soma", 0.5), bc.mech.Synapse("Exp2Syn", name="a", **_EXP2SYN))
    cell[0:1].place(at("soma", 0.5), bc.mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
    return cell


def _lumped_synapse_network(solver, model, **kwargs):
    """A model of :func:`_lumped_synapse_cell` driven by NetStims (events at 0.1, 0.4, 0.7 and 0.25, 0.45, 0.65 ms)."""
    cell = _lumped_synapse_cell(solver, model, **kwargs)
    net = bc.Network(name=f"cell31c_{model.lower()}")
    stim = net.add_population("stim", bc.NetStim(size=1, start=0.1 * u.ms, interval=0.3 * u.ms, number=3))
    post = net.add_population("post", cell)
    if model == "G":
        net.connect("s", source=stim, synapse=post.synapses["s"], weight=0.0005 * u.uS)
        return net, cell
    stim2 = net.add_population("stim2", bc.NetStim(size=1, start=0.25 * u.ms, interval=0.2 * u.ms, number=3))
    net.connect("a", source=stim, synapse=post.synapses["a"], weight=0.01 * u.uS)
    net.connect("b", source=stim2, synapse=post.synapses["b"], weight=0.02 * u.uS)
    return net, cell


def _network_values(cell):
    states = {
        ".".join(map(str, path)).removeprefix("_lumped_runtime."): np.asarray(u.get_mantissa(st.value)).ravel()
        for path, st in brainstate.graph.states(cell).items()
        if isinstance(st, bc.quad.DiffEqState) and path[-1] != "V"
    }
    return np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).ravel(), states


@functools.lru_cache(maxsize=None)
def _lumped_synapse_run(model, solver, dt_ms, lumped_cell=True):
    """``(V per member, other DiffEqStates)`` at 1 ms; call under float64."""
    net, cell = _lumped_synapse_network(solver, model, lumped_cell=lumped_cell)
    net.init_state()
    net.run(dt=dt_ms * u.ms, duration=1.0 * u.ms)
    return _network_values(cell)


def _lumped_gaba_report() -> dict:
    """CELL-31c on G; :func:`_lumped_gaba_numbers` runs it at float64.

    ``{"<solver>/<dt>": (max |V - reference| in mV, max |Ci - reference| in mM)}``
    against rk4 at dt = 0.00025 ms.
    """
    with brainstate.environ.context(precision=64):
        ref_V, ref_states = _lumped_synapse_run("G", "rk4", 0.00025)
        report = {}
        for name in _BLOCK_SOLVER_NAMES:
            for dt_ms in (0.005, 0.001):
                V, states = _lumped_synapse_run("G", name, dt_ms)
                ci = "ion_channels.cl.Ci"
                report[f"{name}/{dt_ms}"] = (
                    float(np.max(np.abs(V - ref_V))),
                    float(np.max(np.abs(states[ci] - ref_states[ci]))),
                )
    return report


@functools.cache
def _lumped_gaba_numbers() -> dict:
    return run_float64(_lumped_gaba_report)


class LumpedBlockSolverPointSynapseTest(unittest.TestCase):
    """CELL-31c: the block solvers advance point synapses on a lumped Cell, each row in its member's block."""

    #: Per solver: the bound on |V - reference| at dt 0.005 and 0.001 ms (mV),
    #: about 1.5 times the measured worst member, and the window of their ratio.
    BOUNDS = {
        "exp_euler": (6e-4, 2.5e-5, (20.0, 30.0)),
        "crank_nicolson": (6e-4, 2.4e-5, (20.0, 30.0)),
        "backward_euler": (1.6e-1, 3.2e-2, (4.0, 6.0)),
    }

    def test_block_solvers_keep_their_order_with_point_synapses(self):
        # Measured (member 0; member 1) at dt 0.005 -> 0.001 ms against rk4 at
        # 0.001 ms: exp_euler 4.32e-5 -> 1.71e-6; 4.14e-4 -> 1.65e-5.
        # crank_nicolson 3.91e-4 -> 1.56e-5; 3.62e-4 -> 1.45e-5.
        # backward_euler 1.06e-1 -> 2.12e-2; 1.84e-2 -> 3.86e-3. They used to
        # raise ValueError at init_state.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            reference = _lumped_synapse_run("P", "rk4", 0.001)[0]
            for name, (coarse, fine, (low, high)) in self.BOUNDS.items():
                with self.subTest(solver=name):
                    error = {dt: np.abs(_lumped_synapse_run("P", name, dt)[0] - reference) for dt in (0.005, 0.001)}
                    self.assertLessEqual(float(error[0.005].max()), coarse)
                    self.assertLessEqual(float(error[0.001].max()), fine)
                    ratio = error[0.005] / error[0.001]
                    self.assertTrue(np.all((low <= ratio) & (ratio <= high)), error)

    def test_gaba_writing_a_chloride_pool_converges(self):
        # Measured V: exp_euler 2.62e-5 -> 1.15e-6, crank_nicolson 2.06e-5 ->
        # 7.25e-7, backward_euler 6.42e-4 -> 1.39e-4 mV; Ci at most 3.86e-5 ->
        # 6.75e-6 mM (backward_euler).
        bounds = {
            "exp_euler": (4e-5, 1.8e-6),
            "crank_nicolson": (3.1e-5, 1.1e-6),
            "backward_euler": (9.6e-4, 2.1e-4),
        }
        report = _lumped_gaba_numbers()
        for name, (coarse, fine) in bounds.items():
            with self.subTest(solver=name):
                for dt, bound, ci_bound in ((0.005, coarse, 6e-5), (0.001, fine, 1e-5)):
                    gap_V, gap_ci = report[f"{name}/{dt}"]
                    self.assertLessEqual(gap_V, bound)
                    self.assertLessEqual(gap_ci, ci_bound)

    def test_matches_the_one_cv_cable_cell(self):
        # The same model as a one-CV cable Cell (CELL-31b): measured 6.5e-16 relative.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in _BLOCK_SOLVER_NAMES:
                with self.subTest(solver=name):
                    point = _lumped_synapse_run("P", name, 0.005)[0]
                    cable = _lumped_synapse_run("P", name, 0.005, lumped_cell=False)[0]
                    np.testing.assert_allclose(point, cable, rtol=1e-10, atol=0)

    def test_segments_reset_state_and_batch_repeat_one_run(self):
        def batched_V(name, batch_size):
            # A Network has no batch axis (CELL-15): the Cell alone, B = 0.01 uS and A = 0.
            cell = _lumped_synapse_cell(name, "P")
            cell.init_state(batch_size=batch_size)
            for _, node in cell._lumped_runtime._synapses():
                if hasattr(node, "B"):
                    node.A.value = u.math.zeros_like(node.A.value)
                    node.B.value = u.math.zeros_like(node.B.value) + 0.01 * u.uS
            cell.run(dt=0.005 * u.ms, duration=1.0 * u.ms)
            return np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).reshape(batch_size or 1, -1)

        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in _BLOCK_SOLVER_NAMES:
                with self.subTest(solver=name):
                    once = _lumped_synapse_run("P", name, 0.005)[0]
                    net, cell = _lumped_synapse_network(name, "P")
                    net.init_state()
                    for _ in range(10):
                        net.run(dt=0.005 * u.ms, duration=0.1 * u.ms)
                    np.testing.assert_allclose(_network_values(cell)[0], once, rtol=1e-12, atol=0)
                    net.reset_state()
                    net.run(dt=0.005 * u.ms, duration=1.0 * u.ms)
                    np.testing.assert_allclose(_network_values(cell)[0], once, rtol=1e-12, atol=0)
                    alone = batched_V(name, None)
                    self.assertGreater(float(alone[0, 0]), -69.0)  # the conductance moved V
                    for member in batched_V(name, 2):
                        np.testing.assert_allclose(member, alone[0], rtol=1e-12, atol=0)

    def test_cells_without_synapses_keep_the_merged_layout(self):
        # No synapse state: the hook returns None and the step merges as
        # before; the PointNeuron equivalence tests above hold unchanged.
        returned = []
        declare = lumped.LumpedRuntime._diffeq_block_layout

        def spy(self, diffeq_states, merging):
            layout = declare(self, diffeq_states, merging)
            returned.append(layout)
            return layout

        with mock.patch.object(lumped.LumpedRuntime, "_diffeq_block_layout", spy):
            for name in _BLOCK_SOLVER_NAMES:
                cell = _point_cell(hh=True, solver=name)
                cell.init_state()
                cell.run(dt=DT, duration=2 * DT)
        self.assertGreaterEqual(len(returned), 3)
        self.assertTrue(all(layout is None for layout in returned))
        cell = _lumped_synapse_cell("exp_euler", "P")
        cell.init_state()
        diffeq = split_diffeq_states(cell._lumped_runtime)[1]
        layout = cell._lumped_runtime._diffeq_block_layout(diffeq, "stack")
        self.assertEqual(layout.pop_shape, (2,))
        self.assertEqual(len(layout.slots), 3)  # Exp2Syn A, B and ExpSyn g


if __name__ == "__main__":
    unittest.main()
