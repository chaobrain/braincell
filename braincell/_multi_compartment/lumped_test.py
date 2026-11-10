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

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell as bc
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
