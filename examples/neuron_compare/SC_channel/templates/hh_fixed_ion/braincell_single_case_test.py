from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
pair_manifest = _load_module(_ROOT / "pair_manifest.py", "hh_fixed_ion_pair_manifest_for_braincell")
discovery = _load_module(_ROOT / "discovery.py", "hh_fixed_ion_discovery_for_braincell")
case_schema = _load_module(_ROOT / "case_schema.py", "hh_fixed_ion_case_schema_for_braincell")
braincell_single_case = _load_module(_ROOT / "braincell_single_case.py", "hh_fixed_ion_braincell_single_case")


class BraincellSingleCaseTest(unittest.TestCase):
    def _build_payload(self, *, stimulus: dict | None = None) -> dict:
        return {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "kv_braincell",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": stimulus or {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.2},
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
            "channel_overrides": {"g_max_S_cm2": 0.0, "v12_mV": 25.0, "q": 9.0},
        }

    def test_run_case_returns_time_voltage_current_and_gates(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
        result = braincell_single_case.run_case(case)

        self.assertEqual(sorted(result.keys()), ["current", "gates", "time_ms", "voltage_mV"])
        self.assertIn("ix", result["current"])
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertEqual(sorted(result["gates"].keys()), ["n"])
        self.assertEqual(result["gates"]["n"].shape, result["time_ms"].shape)

    def test_explicit_ion_type_conflict_raises(self) -> None:
        payload = self._build_payload()
        payload["ion"]["ion_type"] = "na"
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        with self.assertRaises(ValueError):
            braincell_single_case.run_case(case)

    def test_unknown_channel_override_raises(self) -> None:
        payload = self._build_payload()
        payload["channel_overrides"]["bad_param"] = 1.0
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        with self.assertRaises(ValueError):
            braincell_single_case.run_case(case)

    def test_leak_enabled_does_not_break_output(self) -> None:
        payload = self._build_payload()
        payload["leak"] = {"enabled": True, "g_S_cm2": 1e-4, "e_mV": -65.0}
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        result = braincell_single_case.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)

    def test_sine_stimulus_runs(self) -> None:
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(
            self._build_payload(
                stimulus={
                    "kind": "sine",
                    "start_ms": 0.0,
                    "duration_ms": 2.0,
                    "amplitude_nA": 0.2,
                    "frequency_hz": 250.0,
                    "phase_rad": 0.0,
                    "offset_nA": 0.0,
                }
            )
        )
        result = braincell_single_case.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_dc_stimulus_changes_voltage_relative_to_zero_amp_case(self) -> None:
        driven_case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
        quiet_case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(
            self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        )
        driven = braincell_single_case.run_case(driven_case)
        quiet = braincell_single_case.run_case(quiet_case)
        self.assertGreater(np.max(np.abs(driven["voltage_mV"] - quiet["voltage_mV"])), 1e-6)

    def test_reset_state_initializes_gate_and_current_for_nondefault_v_init(self) -> None:
        payload = self._build_payload(
            stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.02}
        )
        payload["simulation"]["v_init_mV"] = 0.0
        payload["simulation"]["temperature_celsius"] = 10.0
        payload["channel_overrides"]["g_max_S_cm2"] = 0.001
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        result = braincell_single_case.run_case(case)
        self.assertGreater(result["gates"]["n"][0], 0.01)
        self.assertGreater(abs(result["current"]["ix"][0]), 1e-3)


if __name__ == "__main__":
    unittest.main()
