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

import unittest
import warnings
from unittest import mock

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell as bc
from braincell._multi_compartment import cell as cell_module, lumped
from braincell.filter import AllRegion, RootLocation

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
    cell.record("v", bc.observe.state("v"))
    return cell


def _single_compartment(*, hh, solver):
    class Model(bc.SingleCompartment):
        def __init__(self):
            super().__init__(
                N,
                length=LENGTH,
                radius=RADIUS,
                V_initializer=braintools.init.Constant(-65.0 * u.mV),
                solver=solver,
            )
            if hh:
                self.na = bc.ion.SodiumFixed(N, E=50.0 * u.mV)
                self.na.add(INa=bc.channel.Na_HH1952(N, g_max=120.0 * u.mS / u.cm**2))
                self.k = bc.ion.PotassiumFixed(N, E=-77.0 * u.mV)
                self.k.add(IK=bc.channel.K_HH1952(N, g_max=36.0 * u.mS / u.cm**2))
            self.IL = bc.channel.IL(N, g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV)

    return Model()


def _run_single_compartment(model, *, clamp):
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
    """Acceptance 1: a lumped Cell reproduces SingleCompartment in float64."""

    def test_voltage_matches_single_compartment(self) -> None:
        bounds = {"rk4": 1e-10, "euler": 1e-10, "ralston2": 1e-10, "exp_euler": 1e-9}
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for hh in (False, True):
                for clamp in (False, True):
                    for solver, bound in bounds.items():
                        with self.subTest(hh=hh, clamp=clamp, solver=solver):
                            expected = _run_single_compartment(_single_compartment(hh=hh, solver=solver), clamp=clamp)
                            cell = _point_cell(hh=hh, clamp=clamp, solver=solver)
                            cell.init_state()
                            values = cell.run(dt=DT, duration=DURATION).samples["v"].values
                            self.assertEqual(values.shape, (800, N))
                            error = float(u.math.max(u.math.abs(values - expected)).to_decimal(u.mV))
                            self.assertLessEqual(error, bound)

    def test_initial_derivative_matches_single_compartment(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for hh in (False, True):
                with self.subTest(hh=hh):
                    model = _single_compartment(hh=hh, solver="rk4")
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
            reference.compute_derivative()
            hooked.pre_integral()
            hooked.compute_derivative()
            hooked.post_integral()
        np.testing.assert_array_equal(
            hooked.V.derivative.to_decimal(u.mV / u.ms), reference.V.derivative.to_decimal(u.mV / u.ms)
        )

    def test_cable_runtime_views_raise(self) -> None:
        cell = _point_cell()
        cell.init_state()
        for name in ("runtime", "n_point", "layouts"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "lumped Cell"):
                    getattr(cell, name)

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
            cell.record("v", bc.observe.state("v"))
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
            cell.record("v", bc.observe.state("v"))
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
            cell.record("v", bc.observe.state("v"))
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
    if np.ndim(pop_size) == 0:
        # Recording needs a one-dimensional population.
        cell.record("v", bc.observe.state("v"))
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
        for part in ("'ralston2'", "Cav1p2_MA2020_GoC", "gate 'm'", "3.66e+06 /ms at -100 mV", "5.5e-07 ms",
                     "'ind_exp_euler'", "'exp_euler'"):
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
        cell = _stiff_gate_cell(_cav1p2(name="cav12"), bc.mech.Channel("Cav1p3_MA2020_GoC", ion_name="ca"),
                                bc.mech.Channel("Kv3p4_MA2020_GoC"))
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
            ("Kv3p4_MA2020_GoC", (bc.mech.Channel("Kv3p4_MA2020_GoC"),), None, True),
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
                self.assertEqual(min(len(np.unique(per_member[:, i])) for i in range(0, per_member.shape[1], 97)),
                                 rounds)

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
        explicit.record("v", bc.observe.state("v"))
        midpoints = bc.Cell.point(N, length=LENGTH, radius=RADIUS, V_init=-65.0 * u.mV)
        midpoints.paint(AllRegion(), *_mechanisms(True))
        midpoints.place(midpoints.cv_midpoints, _clamp())
        midpoints.record("v", bc.observe.state("v"))
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
                self.record("v", bc.observe.state("v"))

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

    def test_unsupported_uses_raise_not_implemented(self) -> None:
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
                with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
                    use()

    def test_unsupported_declarations_raise_not_implemented(self) -> None:
        cell = bc.Cell.point(N)
        with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
            cell.place(bc.mech.Synapse("Exp2Syn", name="syn"))
        with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
            cell.add_reduction("reduced", object())
        with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
            cell.record("i", bc.observe.membrane_current())
            cell.init_state()
            cell.run(dt=DT, duration=DT)
        cable = bc.Cell(bc.Morphology.from_soma())
        cable.place(RootLocation(0.5), bc.mech.Synapse("Exp2Syn", name="syn"))
        with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
            cable.cv_policy = bc.Lumped()
        self.assertIsInstance(cable.cv_policy, bc.CVPerBranch)

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
            cell.record("v", bc.observe.state("v"))
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

    def test_switching_to_lumped_rejects_cable_only_declarations(self) -> None:
        with_reduction = bc.Cell(bc.Morphology.from_soma())
        with_reduction.add_reduction("reduced", _NoopReduction())
        with_junction = bc.Cell(bc.Morphology.from_soma())
        with_junction.place(RootLocation(0.5), bc.mech.Junction())
        for name, cell in (("reduction", with_reduction), ("junction", with_junction)):
            with self.subTest(declaration=name):
                with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
                    cell.cv_policy = bc.Lumped()
                self.assertIsInstance(cell.cv_policy, bc.CVPerBranch)

    def test_network_rejects_lumped_cell(self) -> None:
        network = bc.Network("lumped")
        with self.assertRaisesRegex(NotImplementedError, "lumped Cell"):
            network.add_population("points", bc.Cell.point(N))


if __name__ == "__main__":
    unittest.main()
