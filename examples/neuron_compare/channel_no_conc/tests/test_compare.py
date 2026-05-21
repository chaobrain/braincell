

import os
import unittest
from unittest import mock

import numpy as np

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, build_case_payload, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_compare_test",
)
compare_module = load_module(
    TEMPLATES_ROOT / "compare.py",
    "channel_no_conc_compare_test",
)


class CompareSingleCaseTest(unittest.TestCase):
    _GOC_LIBNRNMECH = CHANNEL_NO_CONC_ROOT / ".." / "Cerebellum_mod" / "GoC" / "channel" / "x86_64" / "libnrnmech.so"
    _BC_LIBNRNMECH = CHANNEL_NO_CONC_ROOT / ".." / "Cerebellum_mod" / "BC" / "channel" / "x86_64" / "libnrnmech.so"

    def _build_payload(self) -> dict:
        payload = build_case_payload(
            case_id="kv_compare",
            config_name="kv_test",
            template_name="compare",
            stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.01},
        )
        payload["simulation"]["duration_ms"] = 0.1
        return payload

    def test_compare_case_returns_voltage_current_and_gate_metrics(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        result = compare_module.compare_case(case)

        self.assertEqual(result["case_id"], "kv_compare")
        self.assertEqual(result["run_id"], "kv_test__compare")
        self.assertEqual(result["config_name"], "kv_test")
        self.assertEqual(result["template_name"], "compare")
        self.assertIn("voltage", result["metrics"])
        self.assertIn("current", result["metrics"])
        self.assertIn("gates", result["metrics"])
        self.assertIn("ix", result["metrics"]["current"])
        self.assertIn("n", result["metrics"]["gates"])
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["current"]["ix"]["rmse"]))
        self.assertEqual(
            result["alignment"]["gates"],
            [{"canonical_name": "n", "braincell_gate": "n", "neuron_gate": "n"}],
        )
        self.assertAlmostEqual(result["time_ms"][0], case.simulation.dt_ms, places=12)
        self.assertEqual(len(result["time_ms"]), len(result["braincell"]["voltage_mV"]))
        self.assertEqual(len(result["time_ms"]), len(result["neuron"]["current"]["ix"]))

    def test_sine_case_voltage_alignment_stays_within_small_error(self) -> None:
        payload = self._build_payload()
        payload["case_id"] = "kv_compare_sine"
        payload["simulation"]["duration_ms"] = 4.0
        payload["simulation"]["temperature_celsius"] = 10.0
        payload["stimulus"] = {
            "kind": "sine",
            "start_ms": 0.0,
            "duration_ms": 4.0,
            "amplitude_nA": 0.02,
            "frequency_hz": 250.0,
            "phase_rad": 0.0,
            "offset_nA": 0.0,
        }
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = compare_module.compare_case(case)

        self.assertLess(result["metrics"]["voltage"]["mae"], 2e-5)
        self.assertLess(result["metrics"]["voltage"]["max_abs"], 5e-5)
        self.assertLess(result["metrics"]["gates"]["n"]["mae"], 1e-6)

    def test_trims_single_initial_neuron_sample(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array([0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([-0.025, 0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.5, -65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([-0.1, 0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.19, 0.2, 0.21, 0.22], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_module.compare_case(case)

        self.assertTrue(result["alignment"]["time_axis_trimmed_neuron_initial_sample"])
        self.assertEqual(result["time_ms"], [0.0, 0.025, 0.05])
        self.assertEqual(len(result["neuron"]["voltage_mV"]), 3)

    def test_rejects_gate_set_mismatch_against_mapping_config(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array([0.0, 0.025], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0], dtype=float),
            "current": {"ix": np.array([0.0, 0.1], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([0.0, 0.025], dtype=float),
            "voltage_mV": np.array([-65.1, -64.1], dtype=float),
            "current": {"ix": np.array([0.0, 0.1], dtype=float)},
            "gates": {"m": np.array([0.1, 0.11], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            with self.assertRaisesRegex(ValueError, "do not match mapping configuration"):
                compare_module.compare_case(case)

    def test_rejects_other_time_axis_mismatch(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array([0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([0.0, 0.03, 0.06], dtype=float),
            "voltage_mV": np.array([-65.1, -64.1, -63.1], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.19, 0.2, 0.21], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            with self.assertRaisesRegex(ValueError, "time axes do not match"):
                compare_module.compare_case(case)

    def test_uses_neuron_time_axis_when_equal_length_axes_shift_by_one_dt(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array([0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([0.025, 0.05, 0.075], dtype=float),
            "voltage_mV": np.array([-65.1, -64.1, -63.1], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.19, 0.2, 0.21], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_module.compare_case(case)

        self.assertFalse(result["alignment"]["time_axis_trimmed_neuron_initial_sample"])
        self.assertEqual(result["time_ms"], [0.025, 0.05, 0.075])
        self.assertIn("voltage", result["metrics"])

    def test_accepts_equal_length_one_dt_shift_with_float_jitter(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array(
                [0.0, 0.02500000037252903, 0.05000000074505806],
                dtype=float,
            ),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([0.025, 0.05, 0.075], dtype=float),
            "voltage_mV": np.array([-65.1, -64.1, -63.1], dtype=float),
            "current": {"ix": np.array([0.0, 0.1, 0.2], dtype=float)},
            "gates": {"n": np.array([0.19, 0.2, 0.21], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_module.compare_case(case)

        self.assertFalse(result["alignment"]["time_axis_trimmed_neuron_initial_sample"])
        self.assertEqual(result["time_ms"], [0.025, 0.05, 0.075])
        self.assertIn("voltage", result["metrics"])

    def test_current_metrics_apply_neuron_one_step_shift(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        braincell_result = {
            "time_ms": np.array([0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([1.0, 2.0, 3.0], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }
        neuron_result = {
            "time_ms": np.array([0.0, 0.025, 0.05], dtype=float),
            "voltage_mV": np.array([-65.0, -64.0, -63.0], dtype=float),
            "current": {"ix": np.array([0.0, 1.0, 2.0], dtype=float)},
            "gates": {"n": np.array([0.2, 0.21, 0.22], dtype=float)},
        }

        with mock.patch.object(compare_module, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_module,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_module.compare_case(case)

        self.assertEqual(result["alignment"]["current"]["neuron_shift_steps"], 1)
        self.assertEqual(result["alignment"]["current"]["braincell_drop_tail_steps"], 1)
        self.assertEqual(result["aligned"]["current"]["time_ms"], [0.0, 0.025])
        self.assertEqual(result["aligned"]["current"]["braincell_ix"], [1.0, 2.0])
        self.assertEqual(result["aligned"]["current"]["neuron_ix"], [1.0, 2.0])
        self.assertEqual(result["metrics"]["current"]["ix"]["mae"], 0.0)

    def test_repo_nav1p6_bc_vinit_compare_case_runs(self) -> None:
        if not self._BC_LIBNRNMECH.resolve().exists():
            self.skipTest("BC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = TEMPLATES_ROOT.parent / "configs" / "ma25_bc" / "nav1p6_ma25_bc.json"
        template_path = TEMPLATES_ROOT.parent / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = compare_module.compare_case(case)

        self.assertIn("voltage", result["metrics"])
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))
        self.assertEqual(len(result["alignment"]["gates"]), 12)

    def test_repo_kca3p1_goc_vinit_compare_case_runs(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = TEMPLATES_ROOT.parent / "configs" / "ma20_goc" / "kca3p1_ma20_goc.json"
        template_path = TEMPLATES_ROOT.parent / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = compare_module.compare_case(case)

        self.assertEqual(
            result["alignment"]["gates"],
            [{"canonical_name": "Y", "braincell_gate": "p", "neuron_gate": "Y"}],
        )
        self.assertLess(result["metrics"]["voltage"]["mae"], 0.1)
        self.assertLess(result["metrics"]["current"]["ix"]["mae"], 5e-5)
        self.assertLess(result["metrics"]["gates"]["Y"]["mae"], 1e-4)

    def test_repo_kca2p2_goc_vinit_compare_case_runs(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = TEMPLATES_ROOT.parent / "configs" / "ma20_goc" / "kca2p2_ma20_goc.json"
        template_path = TEMPLATES_ROOT.parent / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = compare_module.compare_case(case)

        self.assertEqual(len(result["alignment"]["gates"]), 5)
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["current"]["ix"]["mae"]))
        self.assertTrue(all(np.isfinite(record["mae"]) for record in result["metrics"]["gates"].values()))

    def test_repo_kca1p1_goc_vinit_compare_case_runs(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = TEMPLATES_ROOT.parent / "configs" / "ma20_goc" / "kca1p1_ma20_goc.json"
        template_path = TEMPLATES_ROOT.parent / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = compare_module.compare_case(case)

        self.assertEqual(len(result["alignment"]["gates"]), 9)
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["current"]["ix"]["mae"]))
        self.assertTrue(all(np.isfinite(record["mae"]) for record in result["metrics"]["gates"].values()))

    def test_repo_hcn_ma24_pc_dc_compare_case_runs(self) -> None:
        config_path = TEMPLATES_ROOT.parent / "configs" / "ma24_pc" / "hcn1_ma24_pc.json"
        template_path = TEMPLATES_ROOT.parent / "templates" / "dc.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = compare_module.compare_case(case)

        self.assertEqual(result["case_id"], "dc__000")
        self.assertEqual(result["template_name"], "dc")
        self.assertIn("voltage", result["metrics"])
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))


if __name__ == "__main__":
    unittest.main()
