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
pair_manifest = _load_module(_ROOT / "pair_manifest.py", "hh_fixed_ion_pair_manifest")
discovery = _load_module(_ROOT / "discovery.py", "hh_fixed_ion_discovery")
case_schema = _load_module(_ROOT / "case_schema.py", "hh_fixed_ion_case_schema")
neuron_single_case = _load_module(_ROOT / "neuron_single_case.py", "hh_fixed_ion_neuron_single_case")
sweep_kv_test = _load_module(_ROOT / "sweep_presets" / "kv_test.py", "hh_fixed_ion_sweep_kv_test")


class CaseSchemaTest(unittest.TestCase):
    def test_builds_minimal_case(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "kv_smoke",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 0.1, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.01},
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
            "channel_overrides": {"g_max_S_cm2": 0.0, "v12_mV": 25.0, "q": 9.0},
        }
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        self.assertEqual(case.template_variant, "hh_fixed_ion")
        self.assertEqual(case.ion.mode, "fixed")
        self.assertIsNone(case.ion.ion_type)
        self.assertAlmostEqual(case.morphology.radius_um, 50.0 / 3.141592653589793, places=12)
        self.assertEqual(case.pair_id, "kv_test")
        self.assertEqual(case.channel_overrides["v12_mV"], 25.0)

    def test_builds_sine_case(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "kv_sine",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {
                "kind": "sine",
                "start_ms": 0.0,
                "duration_ms": 2.0,
                "amplitude_nA": 0.2,
                "frequency_hz": 250.0,
                "phase_rad": 0.0,
                "offset_nA": 0.0,
            },
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
        }
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        self.assertEqual(case.stimulus.kind, "sine")
        self.assertEqual(case.stimulus.frequency_hz, 250.0)

    def test_rejects_wrong_mode(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "bad",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "radius_um": 5.0, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 0.1, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.01},
            "ion": {"mode": "dynamic", "fixed_E_mV": -80.0},
        }
        with self.assertRaises(ValueError):
            case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)

    def test_allows_arbitrary_channel_overrides(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "override_param",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "radius_um": 5.0, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 0.1, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.01},
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
            "channel_overrides": {"bad_param": 1.0},
        }
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        self.assertEqual(case.channel_overrides["bad_param"], 1.0)


class PairManifestTest(unittest.TestCase):
    def test_get_kv_test_pair(self) -> None:
        entry = pair_manifest.get_pair_entry("kv_test")
        self.assertEqual(entry.neuron_mechanism_name, "Kv")
        self.assertEqual(entry.braincell_channel_name, "IK_Kv_test")


class SweepPresetTest(unittest.TestCase):
    def test_kv_test_preset_matches_registry_and_uses_canonical_paths(self) -> None:
        self.assertEqual(sweep_kv_test.channel_id, "kv_test")
        self.assertIn("simulation.v_init_mV", sweep_kv_test.sweep_axes)
        self.assertIn("stimulus.amp_nA", sweep_kv_test.sweep_axes)
        self.assertIn("channel_overrides.g_max_S_cm2", sweep_kv_test.sweep_axes)
        self.assertIn("g_max_S_cm2", sweep_kv_test.base_case_overrides["channel_overrides"])


class DiscoveryTest(unittest.TestCase):
    def test_discover_neuron_channel_metadata_for_kv(self) -> None:
        meta = discovery.discover_neuron_channel_metadata(
            "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "Kv",
        )
        self.assertEqual(meta.ion_type, "k")
        self.assertEqual(meta.current_field, "i")
        self.assertEqual(meta.current_owner, "mechanism")
        self.assertEqual(meta.gate_names, ("n",))

    def test_discover_braincell_channel_metadata_for_ik_kv_test(self) -> None:
        meta = discovery.discover_braincell_channel_metadata("IK_Kv_test")
        self.assertEqual(meta.ion_type, "k")
        self.assertEqual(meta.channel_kind, "hh")
        self.assertEqual(meta.gate_names, ("n",))


class NeuronSingleCaseTest(unittest.TestCase):
    def _build_payload(self, *, stimulus: dict | None = None) -> dict:
        return {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "kv_smoke",
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
        result = neuron_single_case.run_case(case)

        self.assertEqual(sorted(result.keys()), ["current", "gates", "time_ms", "voltage_mV"])
        self.assertIn("ix", result["current"])
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertEqual(sorted(result["gates"].keys()), ["n"])
        self.assertEqual(result["gates"]["n"].shape, result["time_ms"].shape)
        self.assertEqual(len(result["time_ms"]), 80)
        self.assertAlmostEqual(result["time_ms"][0], 0.025, places=12)
        self.assertAlmostEqual(result["time_ms"][-1], 2.0, places=12)

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
        result = neuron_single_case.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_dc_stimulus_changes_voltage_relative_to_zero_amp_case(self) -> None:
        driven = neuron_single_case.run_case(
            case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(self._build_payload())
        )
        quiet = neuron_single_case.run_case(
            case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(
                self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
            )
        )
        self.assertGreater(np.max(np.abs(driven["voltage_mV"] - quiet["voltage_mV"])), 1e-6)

    def test_run_case_respects_nondefault_v_init_under_h_run(self) -> None:
        payload = self._build_payload(
            stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0}
        )
        payload["simulation"]["v_init_mV"] = 0.0
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        result = neuron_single_case.run_case(case)
        self.assertGreater(result["voltage_mV"][0], -1.0)
        self.assertLess(result["voltage_mV"][0], 1.0)

    def test_current_ix_uses_braincell_sign_convention(self) -> None:
        payload = self._build_payload(
            stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.02}
        )
        payload["simulation"]["v_init_mV"] = 0.0
        payload["simulation"]["temperature_celsius"] = 10.0
        payload["channel_overrides"]["g_max_S_cm2"] = 0.001
        case = case_schema.SingleCompartmentChannelHHFixedIonCase.from_dict(payload)
        result = neuron_single_case.run_case(case)
        self.assertLess(result["current"]["ix"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
