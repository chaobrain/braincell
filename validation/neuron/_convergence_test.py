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

"""``dt`` convergence and stiffness: two representative channels, static scans, coverage.

Needs no NEURON. ``Kv3p4_MA2020_GoC`` (HH) is exact under voltage steps;
``Nav1p6_MA2020_GoC`` (Markov, 13 states) converges at first order on the
default ``backward_euler`` x 1 and is stiff.
"""

import dataclasses
import json
import math
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from validation.neuron import _clamp
from validation.neuron import _convergence as cv
from validation.neuron._suite_loader import use_double_precision

os.environ.setdefault("JAX_PLATFORMS", "cpu")
use_double_precision()

from validation.neuron.channel_no_conc import convergence as table  # noqa: E402

KV = "ma20_goc/kv3p4_ma20_goc"
NAV = "ma20_goc/nav1p6_ma20_goc"
TEMPLATES = Path(__file__).resolve().parent / "channel_no_conc" / "templates"


def _spec(name: str) -> dict:
    return json.loads((TEMPLATES / f"{name}.json").read_text())["voltage_clamp"]


def _config_entry(config: str) -> table.ConfigEntry:
    return next(entry for entry in table.CONFIGS if entry.config == config)


def _class(name: str):
    from braincell.mech import get_registry

    return get_registry().get("channel", name)


class CommandAndReplayTableTest(unittest.TestCase):
    def test_commands_on_any_time_grid(self):
        for name in ("vc_steps", "vc_ramp"):
            spec = _spec(name)
            for dt in (0.1, 0.025, 0.00625):
                n = round(_clamp.protocol_duration_ms(spec) / dt)
                values = _clamp.command_values(spec, np.arange(n) * dt)
                np.testing.assert_array_equal(values, np.stack(_clamp.protocol_commands(spec, dt)))
        steps = _spec("vc_steps")
        values = _clamp.command_values(steps, [0.0, 4.9, 5.0, 24.99, 25.0, 34.0])
        np.testing.assert_array_equal(values[-1], [-80, -80, 40, 40, -80, -80])
        np.testing.assert_array_equal(_clamp.command_values(steps, [5.0, 25.0], side="left")[-1], [-80, 40])
        np.testing.assert_allclose(_clamp.command_values(_spec("vc_ramp"), [0.0, 5.0, 85.0, 165.0, 170.0])[0], [-100, -100, -20, 60, 60])
        with self.assertRaises(ValueError):
            _clamp.command_values(steps, [0.0], side="middle")
        with self.assertRaises(ValueError):
            _clamp.protocol_duration_ms({"kind": "sine"})

    def test_replay_table_writes_V_n_plus_1(self):
        # Steps: the value held over [t_n, t_{n+1}]; ramp: V(t_{n+1}).
        table_steps = _clamp.replay_table(_spec("vc_steps"), 0.1)
        self.assertEqual(table_steps.shape, (351, 8))
        np.testing.assert_array_equal(table_steps[:, -1][[0, 50, 51, 250, 251, 350]], [-80, -80, 40, 40, -80, -80])
        table_ramp = _clamp.replay_table(_spec("vc_ramp"), 0.1)
        self.assertEqual(table_ramp.shape, (1701, 1))
        np.testing.assert_allclose(table_ramp[[0, 50, 51, 52], 0], [-100, -100, -99.9, -99.8])

    def test_each_population_replays_its_column(self):
        import brainunit as u

        import braincell

        table_ = np.random.default_rng(1).uniform(-100.0, 40.0, (21, 3))
        table_[0] = -65.0
        cell = braincell.Cell(
            braincell.Morphology.from_soma(), pop_size=3, V_init=-65.0 * u.mV, solver=_clamp.voltage_replay_solver(table_)
        )
        cell.record("v", braincell.observe.state("v"))
        result = cell.run(dt=0.025 * u.ms, duration=20 * 0.025 * u.ms)
        recorded = np.asarray(result.samples["v"].values.to_decimal(u.mV)).reshape(20, 3)
        np.testing.assert_array_equal(recorded, table_[:-1])
        np.testing.assert_array_equal(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(3), table_[-1])
        with self.assertRaises(ValueError):
            _clamp.voltage_replay_solver(np.zeros((2, 2, 2)))


class ClassificationTest(unittest.TestCase):
    @staticmethod
    def _errors(e_I, e_s=None):
        e_s = e_I if e_s is None else e_s
        return {dt: {"e_I": a, "e_s": b} for dt, a, b in zip(cv.DTS_MS, e_I, e_s)}

    def test_classes(self):
        first = [0.2 * dt for dt in cv.DTS_MS]
        self.assertEqual(cv.classify(self._errors([1e-13] * 5))[0], cv.EXACT)
        self.assertEqual(cv.classify(self._errors(first))[0], cv.CONVERGENT)
        self.assertEqual(cv.classify(self._errors([e * 10 for e in first]))[0], cv.ABOVE_GUIDE_LINE)
        self.assertEqual(cv.classify(self._errors([0.1 * dt**2 for dt in cv.DTS_MS]))[0], cv.NOT_CONVERGENT)
        self.assertEqual(cv.classify(self._errors([math.nan] + first[1:]))[0], cv.NOT_CONVERGENT)
        self.assertEqual(cv.classify({0.025: {"e_I": 1e-3, "e_s": 1e-3}}), (None, {}))
        # A current that is 0 throughout is not compared.
        self.assertEqual(cv.classify(self._errors([None] * 5, first)), (cv.CONVERGENT, {"e_s": cv.classify(self._errors(first))[1]["e_s"]}))
        self.assertAlmostEqual(cv.fitted_order([0.1, 0.05, 0.025], [4e-2, 1e-2, 2.5e-3]), 2.0, places=12)
        self.assertTrue(math.isnan(cv.fitted_order([0.1, 0.05], [1e-3, 0.0])))

    def test_rounded_limit(self):
        # 1.5 times the measurement, rounded up to 1, 2 or 5 times a power of ten.
        for measured, limit in ((7.84e-2, 2e-1), (3.24e-2, 5e-2), (1.2e-4, 2e-4), (6.6e-5, 1e-4), (4e-3, 1e-2), (2e-3, 5e-3)):
            self.assertEqual(cv.rounded_limit(measured), limit, measured)
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                cv.rounded_limit(bad)

    def test_assert_matches(self):
        first = self._errors([0.2 * dt for dt in cv.DTS_MS])
        classification, orders = cv.classify(first)
        report = cv.ConvergenceReport("c", "C", "p", cv.STEPS, first, orders, classification)
        entry = table.ConfigEntry("c", cv.CONVERGENT, limits={"e_I": 1e-2})
        table.assert_matches(report, entry)
        for wrong in (
            dataclasses.replace(entry, protocol=cv.RELAXATION, limits={"e_s": 1e-2}),
            dataclasses.replace(entry, classification=cv.ABOVE_GUIDE_LINE),
            dataclasses.replace(entry, limits={"e_I": 4e-3}),
        ):
            with self.assertRaises(AssertionError):
                table.assert_matches(report, wrong)
        with self.assertRaisesRegex(AssertionError, "order"):
            table.assert_matches(dataclasses.replace(report, orders={"e_I": 1.5}), entry)
        exact = cv.ConvergenceReport("c", "C", "p", cv.STEPS, self._errors([1e-13] * 4 + [2e-10]), {}, cv.EXACT)
        with self.assertRaisesRegex(AssertionError, "dt = 0.00625"):
            table.assert_matches(exact, table.ConfigEntry("c", cv.EXACT))


class RepresentativeChannelTest(unittest.TestCase):
    """Acceptance of the two representative configurations on the Cell's default path."""

    @classmethod
    def setUpClass(cls):
        cls.kv = cv.check_config(KV)
        cls.nav = cv.check_config(NAV)

    def test_kv3p4_is_exact_and_its_ramp_converges_at_first_order(self):
        table.assert_matches(self.kv, _config_entry(KV))
        self.assertEqual(self.kv.path, "ind_exp_euler")
        self.assertTrue(self.kv.ramp_ok, self.kv.ramp_orders)
        self.assertTrue(0.8 <= self.kv.ramp_orders["e_s"] <= 1.2)
        self.assertTrue(0.7 <= self.kv.ramp_orders["e_I"] <= 1.3)

    def test_kv3p4_reference_agrees_with_rk4_at_a_fine_step(self):
        case, spec = cv.load_case(KV, cv.STEPS)
        runs = [cv.simulate(case, spec, 0.1 / 256, gate_solver=solver) for solver in ("rk4", "exp_euler")]
        self.assertLessEqual(cv.errors(*runs)["e_I"], 1e-10)

    def test_the_gate_solver_reaches_the_hh_gates(self):
        case, spec = cv.load_case(KV, cv.STEPS)
        default, backward = (cv.simulate(case, spec, 0.025, gate_solver=solver) for solver in (None, "backward_euler"))
        self.assertGreater(cv.errors(backward, default)["e_I"], 1e-2)

    def test_kv3p4_relaxation_is_exact(self):
        report = cv.check_config(KV, protocol=cv.RELAXATION, ramp=False)
        self.assertEqual(report.classification, cv.EXACT)
        self.assertTrue(all(errors["e_I"] is None for errors in report.errors.values()))

    def test_nav1p6_default_path_converges_at_first_order_above_the_guide_line(self):
        table.assert_matches(self.nav, _config_entry(NAV))
        self.assertEqual(self.nav.path, "backward_euler x 1")
        self.assertLessEqual(self.nav.error("e_I"), 2e-1)
        self.assertLessEqual(self.nav.error("e_s"), 5e-2)
        self.assertGreater(self.nav.error("e_I"), cv.GUIDE_LINE)
        self.assertTrue(self.nav.ramp_ok, self.nav.ramp_orders)

    def test_nav1p6_other_subsolvers(self):
        exp_euler = cv.check_config(NAV, subsolver="exp_euler", substeps=1, ramp=False)
        self.assertEqual(exp_euler.classification, cv.EXACT, exp_euler.errors)
        self.assertLessEqual(max(exp_euler.errors[0.025].values()), 1e-10)
        backward_x4 = cv.check_config(NAV, subsolver="backward_euler", substeps=4, dts_ms=(0.025,), ramp=False)
        self.assertLessEqual(backward_x4.error("e_I"), 5e-2)
        # Euler is unstable on this stiff channel: the run completes and returns NaN.
        euler = cv.check_config(NAV, subsolver="euler", substeps=1, ramp=False)
        self.assertEqual(euler.classification, cv.NOT_CONVERGENT)
        self.assertTrue(math.isnan(euler.error("e_I")))

    def test_replaying_V_n_instead_of_V_n_plus_1_breaks_exactness(self):
        replay_table = _clamp.replay_table

        def step_start(spec, dt_ms):
            values = replay_table(spec, dt_ms)
            return np.concatenate([values[:1], values[:-1]])

        with mock.patch.object(_clamp, "replay_table", step_start):
            report = cv.check_config(KV, ramp=False)
        with self.assertRaises(AssertionError):
            table.assert_matches(report, _config_entry(KV))

    def test_a_limit_below_the_measurement_fails(self):
        entry = _config_entry(NAV)
        tightened = dataclasses.replace(entry, limits={**entry.limits, "e_I": 0.9 * self.nav.error("e_I")})
        with self.assertRaises(AssertionError):
            table.assert_matches(self.nav, tightened)


class StiffnessAndSingularityTest(unittest.TestCase):
    def test_representative_classes_match_the_table(self):
        nav = cv.stiffness(_class("Nav1p6_MA2020_GoC"))
        # 1.9635e5 /ms to five figures.
        self.assertLessEqual(abs(nav.rho_max / 1.963516e5 - 1.0), 1e-6)
        self.assertEqual(nav.V_at_max_mV, 60.0)
        self.assertTrue(nav.stiff())
        kv = cv.stiffness(_class("Kv3p4_MA2020_GoC"))
        self.assertLessEqual(abs(kv.rho_max - 6.88), 0.01)
        self.assertEqual(kv.gate, "m")
        self.assertFalse(kv.stiff())
        for entry in table.CLASSES:
            report = cv.stiffness(_class(entry.name))
            self.assertEqual((report.decade, report.stiff()), (entry.decade, entry.stiff), entry.name)

    def test_vectorized_rates_match_a_pointwise_loop(self):
        V = np.linspace(-100.0, 60.0, 17)
        for name in ("Nav1p6_MA2020_GoC", "Kv3p4_MA2020_GoC"):
            vector = cv.stiffness(_class(name), V_mV=V).rho
            loop = np.array([cv.stiffness(_class(name), V_mV=[v]).rho[0] for v in V])
            np.testing.assert_allclose(vector, loop, rtol=1e-12, atol=0.0, err_msg=name)

    def test_registry_singularities_are_the_known_ones(self):
        names = table.class_names()
        found = {}
        for name in names:
            singular = cv.singularities(_class(name))
            if singular:
                found[name] = {quantity: tuple(V.tolist()) for quantity, V in singular.items()}
        print(f"singularity scan: {len(names)} channel classes, {len(cv.SINGULARITY_GRID_MV)} voltages each")
        self.assertGreaterEqual(len(names), 112)
        self.assertEqual(found, table.KNOWN_SINGULARITIES)

    def test_a_removable_singularity_is_reported(self):
        import brainunit as u

        from braincell.channel import OhmicHH

        class Singular(OhmicHH):
            gates = (("m", 1),)

            def __init__(self, size, g_max=1.0 * (u.mS / u.cm**2), E=-80.0 * u.mV):
                super().__init__(size=size)
                self.g_max, self.E = g_max, E

            def f_m_alpha(self, V):
                return ((V + 30.0 * u.mV) / u.mV) / (u.math.exp((V + 30.0 * u.mV) / (5.0 * u.mV)) - 1.0)

            def f_m_beta(self, V):
                return 0.5 * u.math.ones_like(u.get_magnitude(V))

        found = cv.singularities(Singular)
        self.assertEqual(set(found), {"m.value", "m.derivative", "current"})
        for voltages in found.values():
            np.testing.assert_array_equal(voltages, [-30.0])

    def test_ion_infos_match_the_cell(self):
        from validation.neuron.channel_no_conc.engine import braincell_runner

        checked = 0
        for config in ("ma20_goc/cav1p2_ma20_goc", "ma20_grc/kv1p5_ma20_grc", KV):
            case, _ = cv.load_case(config, cv.STEPS)
            cell = braincell_runner.build_cell(case)
            for path, node in cell._family_channel_nodes():
                if type(getattr(node, "_channel", node)).__name__ != case.mapping_spec.braincell.class_name:
                    continue
                _, (_, *expected) = cell._channel_integration_target_and_args(path, node, cell.V.value)
                actual = cv.channel_ion_infos(cv.channel_class(case), case)
                self.assertEqual(len(actual), len(expected), config)
                for got, want in zip(actual, expected):
                    for name in ("Ci", "Co", "E", "valence"):
                        np.testing.assert_allclose(
                            np.ravel(cv._magnitude(getattr(got, name))),
                            np.ravel(cv._magnitude(getattr(want, name))),
                            rtol=1e-12,
                            err_msg=f"{config} {name}",
                        )
                checked += 1
        self.assertEqual(checked, 3)


class StatelessChannelTest(unittest.TestCase):
    CL = "testing/cl_leak_testing"

    def test_a_stateless_channel_is_exact_with_no_state_error(self):
        report = cv.check_config(self.CL)
        table.assert_matches(report, _config_entry(self.CL))
        self.assertTrue(report.ramp_ok, report.ramp_differences)
        for values in (*report.errors.values(), *report.ramp_differences):
            self.assertIsNone(values["e_s"])
            self.assertIsNotNone(values["e_I"])

    def test_exp_euler_rejects_an_empty_state_set(self):
        # Why the gate solver is skipped for a channel without states.
        case, spec = cv.load_case(self.CL, cv.STEPS)
        with mock.patch.object(cv, "_channel_states", lambda cell, cls: [None]):
            with self.assertRaisesRegex(ValueError, "at least one array"):
                cv.simulate(case, spec, 0.025, gate_solver="exp_euler")


class CoverageTest(unittest.TestCase):
    def test_the_table_covers_its_scope(self):
        table.assert_covered()
        # Every configuration and every registered channel class.
        self.assertEqual(table.COVERAGE_SCOPE, {"configs": None, "classes": None})
        self.assertEqual({entry.config for entry in table.CONFIGS}, set(table.config_names()))
        self.assertEqual({entry.name for entry in table.CLASSES}, set(table.class_names()))
        self.assertGreaterEqual(len(table.CLASSES), 113)
        self.assertGreaterEqual(len(table.CONFIGS), 97)

    def test_a_new_channel_class_or_configuration_fails(self):
        from braincell.mech import get_registry, register_channel

        name = "ConvergenceCoverageProbe"
        register_channel(name)(type(name, (_class("K_Kv_test"),), {}))
        try:
            problems = table.coverage_problems()
        finally:
            get_registry().unregister("channel", name)
        self.assertEqual(problems, [f"classes: {name} is not in the table"])
        names = table.config_names()
        with mock.patch.object(table, "config_names", lambda: (*names, "testing/new_probe")):
            self.assertEqual(table.coverage_problems(), ["configs: testing/new_probe is not in the table"])

    def test_a_missing_or_unknown_name_fails(self):
        without_nav = [entry for entry in table.CONFIGS if entry.config != NAV]
        with self.assertRaises(AssertionError):
            table.assert_covered(configs=without_nav)
        unknown = [*table.CLASSES, table.ClassEntry("No_Such_Channel", 0, False)]
        with self.assertRaises(AssertionError):
            table.assert_covered(classes=unknown)
        twice = [*table.CONFIGS, table.CONFIGS[0]]
        with self.assertRaisesRegex(AssertionError, "listed twice"):
            table.assert_covered(configs=twice)

    def test_entries_are_validated(self):
        with self.assertRaises(ValueError):
            table.ConfigEntry(KV, cv.NOT_CONVERGENT)
        with self.assertRaises(ValueError):
            table.ConfigEntry(KV, cv.EXACT, limits={"e_I": 1e-3})
        with self.assertRaises(ValueError):
            table.ConfigEntry(NAV, cv.CONVERGENT)
        with self.assertRaises(ValueError):
            table.ConfigEntry(NAV, cv.CONVERGENT, limits={"e_V": 1e-3})
        with self.assertRaises(ValueError):
            table.ConfigEntry(NAV, cv.CONVERGENT, limits={"e_I": 1e-11})
        with self.assertRaises(ValueError):
            table.ConfigEntry(NAV, cv.CONVERGENT, protocol="vc_ramp", limits={"e_I": 1e-3})
        with self.assertRaises(ValueError):
            table.ConfigEntry(KV, cv.EXACT, ramp_dts_ms=(0.1, 0.05, 0.025))
        with self.assertRaisesRegex(ValueError, "each half the last"):
            cv.ramp_convergence(KV, dts_ms=(0.01, 0.004, 0.002))


if __name__ == "__main__":
    unittest.main()
