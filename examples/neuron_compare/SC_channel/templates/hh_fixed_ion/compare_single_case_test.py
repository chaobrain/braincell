from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
case_schema = _load_module(_ROOT / "case_schema.py", "hh_fixed_ion_case_schema_for_compare")
pair_manifest = _load_module(_ROOT / "pair_manifest.py", "hh_fixed_ion_pair_manifest_for_compare")
compare_single_case = _load_module(_ROOT / "compare_single_case.py", "hh_fixed_ion_compare_single_case")


class CompareSingleCaseTest(unittest.TestCase):
    def _build_payload(self) -> dict:
        return {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "kv_compare",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 0.1, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.01},
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
            "channel_overrides": {"g_max_S_cm2": 0.0, "v12_mV": 25.0, "q": 9.0},
        }

    def test_compare_case_returns_voltage_current_and_gate_metrics(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
        result = compare_single_case.compare_case(case)

        self.assertEqual(result["case_id"], "kv_compare")
        self.assertEqual(result["pair_id"], "kv_test")
        self.assertEqual(result["template_variant"], "hh_fixed_ion")
        self.assertIn("voltage", result["metrics"])
        self.assertIn("current", result["metrics"])
        self.assertIn("gates", result["metrics"])
        self.assertIn("ix", result["metrics"]["current"])
        self.assertIn("n", result["metrics"]["gates"])
        self.assertTrue(np.isfinite(result["metrics"]["voltage"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["current"]["ix"]["rmse"]))
        self.assertIn("gates", result["alignment"])
        self.assertEqual(
            result["alignment"]["gates"],
            [{"canonical_name": "n", "braincell_gate": "n", "neuron_gate": "n"}],
        )
        self.assertEqual(len(result["time_ms"]), len(result["braincell"]["voltage_mV"]))
        self.assertEqual(len(result["time_ms"]), len(result["neuron"]["current"]["ix"]))

    def test_trims_single_initial_neuron_sample(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
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

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_single_case.compare_case(case)

        self.assertTrue(result["alignment"]["time_axis_trimmed_neuron_initial_sample"])
        self.assertEqual(result["time_ms"], [0.0, 0.025, 0.05])
        self.assertEqual(len(result["neuron"]["voltage_mV"]), 3)

    def test_rejects_unmapped_gate_set_mismatch(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
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

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            with self.assertRaisesRegex(ValueError, "gate sets do not match"):
                compare_single_case.compare_case(case)

    def test_uses_explicit_gate_name_map(self) -> None:
        payload = self._build_payload()
        payload["compare"] = {"gate_name_map": {"n": "n_neuron"}}
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
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
            "gates": {"n_neuron": np.array([0.19, 0.2], dtype=float)},
        }

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_single_case.compare_case(case)

        self.assertIn("n", result["metrics"]["gates"])
        self.assertEqual(
            result["alignment"]["gates"],
            [{"canonical_name": "n", "braincell_gate": "n", "neuron_gate": "n_neuron"}],
        )

    def test_pair_manifest_gate_name_map_is_used_when_case_omits_one(self) -> None:
        payload = self._build_payload()
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
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
            "gates": {"n_neuron": np.array([0.19, 0.2], dtype=float)},
        }
        mapped_pair = pair_manifest.PairEntry(
            pair_id="kv_test",
            description="mapped",
            neuron_mechanism_name="Kv",
            braincell_channel_name="IK_Kv_test",
            gate_name_map={"n": "n_neuron"},
        )

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ), mock.patch.object(compare_single_case, "get_pair_entry", return_value=mapped_pair):
            result = compare_single_case.compare_case(case)

        self.assertIn("n", result["metrics"]["gates"])
        self.assertEqual(result["alignment"]["gates"][0]["neuron_gate"], "n_neuron")

    def test_rejects_other_time_axis_mismatch(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
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

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            with self.assertRaisesRegex(ValueError, "time axes do not match"):
                compare_single_case.compare_case(case)

    def test_accepts_equal_length_time_axes_shifted_by_one_dt(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
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

        with mock.patch.object(compare_single_case, "run_braincell_case", return_value=braincell_result), mock.patch.object(
            compare_single_case,
            "run_neuron_case",
            return_value=neuron_result,
        ):
            result = compare_single_case.compare_case(case)

        self.assertFalse(result["alignment"]["time_axis_trimmed_neuron_initial_sample"])
        self.assertIn("voltage", result["metrics"])


if __name__ == "__main__":
    unittest.main()
