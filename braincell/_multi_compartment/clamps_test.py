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

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell.filter import AllRegion, RootLocation, at
from braincell.quad import get_integrator, list_integrators


def _clamped_cell():
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"))
    first = braincell.CurrentClamp(durations=1.0 * u.ms, amplitudes=0.2 * u.nA)
    second = braincell.CurrentClamp(delay=1.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.3 * u.nA)
    sine = braincell.SineClamp(
        amplitude=0.0 * u.nA,
        frequency=1.0 * u.kHz,
        offset=0.4 * u.nA,
        duration=2.0 * u.ms,
    )
    cell.place(at("soma", 0.5), first, second, sine)
    return cell, first


DT = 0.025 * u.ms


def _cable_cell(solver):
    """One 600 um dendrite, 3 CVs, a clamp at x = 0.5 switching on at 0.02 ms."""
    dend = braincell.Branch.from_lengths(lengths=[600.0] * u.um, radii=[0.5, 0.5] * u.um, type="dendrite")
    cell = braincell.Cell(
        braincell.Morphology.from_root(dend, name="dend"),
        cv_policy=braincell.CVPerBranch(cv_per_branch=3),
        V_init=-65.0 * u.mV,
        solver=solver,
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
    )
    cell.place(
        RootLocation(x=0.5), braincell.CurrentClamp(delay=0.02 * u.ms, durations=1.0 * u.ms, amplitudes=0.05 * u.nA)
    )
    cell.init_state()
    return cell


def _lumped_cell(solver):
    """A 20 um x 10 um lumped soma with a 0.05 nA clamp from t = 0."""
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        cv_policy=braincell.Lumped(),
        V_init=-65.0 * u.mV,
        solver=solver,
    )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(
        RootLocation(x=0.5), braincell.CurrentClamp(delay=0.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.05 * u.nA)
    )
    cell.init_state()
    return cell


def _direct_and_update_voltages(make_cell, name, n_step=2):
    """Advance one cell with ``update()`` and a twin with the bare solver step."""
    stepped, direct = make_cell(name), make_cell(name)
    step = get_integrator(name)
    updated, called = [], []
    for i in range(n_step):
        with brainstate.environ.context(dt=DT, t=i * DT):
            stepped.update()
            step(direct)
        updated.append(np.asarray(stepped.V.value.to_decimal(u.mV)))
        called.append(np.asarray(direct.V.value.to_decimal(u.mV)))
    return updated, called


class ClampViewTest(unittest.TestCase):
    def test_type_declaration_and_chained_position_selection(self) -> None:
        cell, declaration = _clamped_cell()

        self.assertEqual(cell.clamps.clamp_type.tolist(), ["CurrentClamp", "CurrentClamp", "SineClamp"])
        self.assertEqual(cell.clamps["CurrentClamp"].id.tolist(), [0, 1])
        self.assertEqual(cell.clamps.by_type(braincell.CurrentClamp)[1].id.tolist(), [1])
        self.assertEqual(cell.clamps[declaration].id.tolist(), [0])
        self.assertEqual(cell.clamps.by_id([2, 0]).id.tolist(), [2, 0])

    def test_prepared_point_current_is_held_across_solver_stage_times(self) -> None:
        cell, _ = _clamped_cell()
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=1.0 * u.ms):
            cell._prepare_step_clamps(t=0.0 * u.ms, dt=1.0 * u.ms)
            with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                at_start = cell._solver_clamp_point_current(t=0.0 * u.ms)
                at_late_stage = cell._solver_clamp_point_current(t=100.0 * u.ms)

        np.testing.assert_array_equal(at_start.to_decimal(u.nA), at_late_stage.to_decimal(u.nA))
        self.assertAlmostEqual(float(at_start[0, 1].to_decimal(u.nA)), 0.6, places=6)

    def test_direct_step_holds_point_current_from_pre_integral_to_post_integral(self) -> None:
        # A solver step called directly has no prepared-step flag: pre_integral
        # samples the clamps at t + dt / 2, every stage reads that value, and
        # post_integral hands derivative evaluations back to the fallback.
        cell, _ = _clamped_cell()
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=1.0 * u.ms):
            cell.pre_integral()
            at_start = cell._solver_clamp_point_current(t=0.0 * u.ms)
            at_late_stage = cell._solver_clamp_point_current(t=100.0 * u.ms)
            cell.post_integral()
            after_step = cell._solver_clamp_point_current(t=100.0 * u.ms)

        np.testing.assert_array_equal(at_start.to_decimal(u.nA), at_late_stage.to_decimal(u.nA))
        self.assertAlmostEqual(float(at_start[0, 1].to_decimal(u.nA)), 0.6, places=6)
        self.assertEqual(float(after_step[0, 1].to_decimal(u.nA)), 0.0)

    def test_reset_state_drops_an_unfinished_direct_step(self) -> None:
        cell, _ = _clamped_cell()
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=1.0 * u.ms):
            cell.pre_integral()
        cell.reset_state()
        with brainstate.environ.context(t=100.0 * u.ms, dt=1.0 * u.ms):
            sampled = cell._solver_clamp_point_current(t=100.0 * u.ms)
        self.assertEqual(float(sampled[0, 1].to_decimal(u.nA)), 0.0)

    def test_delay_is_sampled_at_main_step_midpoint(self) -> None:
        soma = braincell.Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )

        def sample(delay_ms):
            cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"))
            cell.place(
                at("soma", 0.5),
                braincell.CurrentClamp(
                    delay=delay_ms * u.ms,
                    durations=2.0 * u.ms,
                    amplitudes=1.0 * u.nA,
                ),
            )
            cell.init_state()
            with brainstate.environ.context(t=4.0 * u.ms, dt=1.0 * u.ms):
                cell._prepare_step_clamps(t=4.0 * u.ms, dt=1.0 * u.ms)
            return float(cell._step_clamp_components.value[0].to_decimal(u.nA))

        self.assertEqual(sample(4.4), 1.0)
        self.assertEqual(sample(4.5), 1.0)
        self.assertEqual(sample(4.6), 0.0)


class DirectSolverCallTest(unittest.TestCase):
    """CELL-2c: a bare solver step samples the clamps as ``update()`` does."""

    def test_cable_direct_step_matches_update_for_every_solver(self) -> None:
        names = list_integrators()
        self.assertGreaterEqual(len(names), 25)
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in names:
                with self.subTest(solver=name):
                    updated, called = _direct_and_update_voltages(_cable_cell, name)
                    for step_values, direct_values in zip(updated, called):
                        np.testing.assert_array_equal(direct_values, step_values)
            # The clamp switches on at 0.02 ms: the first step samples it off
            # at 0.0125 ms, the second on at 0.0375 ms.
            updated, called = _direct_and_update_voltages(_cable_cell, "rk4")
        np.testing.assert_array_equal(called[0], np.full((1, 3), -65.0))
        np.testing.assert_allclose(
            called[1],
            [[-64.99847229548408, -64.80436020266207, -64.99847229548408]],
            rtol=1e-13,
        )

    def test_lumped_direct_step_applies_the_clamp(self) -> None:
        names = [name for name in list_integrators() if name not in {"staggered", "dhs_voltage"}]
        self.assertGreaterEqual(len(names), 23)
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in names:
                with self.subTest(solver=name):
                    updated, called = _direct_and_update_voltages(_lumped_cell, name)
                    for step_values, direct_values in zip(updated, called):
                        np.testing.assert_array_equal(direct_values, step_values)
            updated, called = _direct_and_update_voltages(_lumped_cell, "rk4")
        np.testing.assert_allclose(
            np.concatenate(called),
            [-64.90065239681512, -64.80155285243549],
            rtol=1e-13,
        )


if __name__ == "__main__":
    unittest.main()
