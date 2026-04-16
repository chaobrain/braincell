from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
sweep_config = _load_module(_ROOT / "sweep_config.py", "hh_fixed_ion_sweep_config")
sweep_driver = _load_module(_ROOT / "sweep_driver.py", "hh_fixed_ion_sweep_driver")


class SweepConfigTest(unittest.TestCase):
    def _build_config_payload(self) -> dict:
        return {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "config_id": "tiny_smoke",
            "base_case": {
                "pair_id": "kv_test",
                "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
                "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
                "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
                "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
                "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
                "channel_overrides": {"g_max_S_cm2": 0.0, "v12_mV": 25.0, "q": 9.0}
            },
            "sweep_axes": {
                "simulation.v_init_mV": [-65.0, -50.0],
                "simulation.temperature_celsius": [20.0, 25.0]
            },
            "case_groups": [
                {
                    "group_id": "dc_group",
                    "overrides": {},
                    "sweep_axes": {
                        "stimulus.amp_nA": [0.0, 0.02]
                    }
                },
                {
                    "group_id": "sine_group",
                    "overrides": {
                        "stimulus": {
                            "kind": "sine",
                            "start_ms": 0.0,
                            "duration_ms": 2.0,
                            "amplitude_nA": 0.02,
                            "frequency_hz": 250.0,
                            "phase_rad": 0.0,
                            "offset_nA": 0.0
                        }
                    },
                    "sweep_axes": {
                        "stimulus.frequency_hz": [100.0, 250.0]
                    }
                }
            ]
        }

    def test_expand_cases_generates_global_and_local_cartesian_product(self) -> None:
        config = sweep_config.SweepConfig.from_dict(self._build_config_payload())
        expanded = sweep_config.expand_cases(config)
        self.assertEqual(len(expanded), 16)
        self.assertEqual(expanded[0]["case_id"], "dc_group__000")
        self.assertEqual(expanded[7]["case_id"], "dc_group__007")
        self.assertEqual(expanded[8]["case_id"], "sine_group__000")
        self.assertEqual(expanded[-1]["case_id"], "sine_group__007")
        self.assertEqual(expanded[-1]["stimulus"]["kind"], "sine")

    def test_rejects_top_level_sine_axis_for_dc_group(self) -> None:
        payload = self._build_config_payload()
        payload["sweep_axes"]["stimulus.frequency_hz"] = [100.0]
        with self.assertRaisesRegex(ValueError, "Top-level sweep path"):
            sweep_config.SweepConfig.from_dict(payload)


class SweepDriverTest(unittest.TestCase):
    def _build_single_case_config(self, *, stimulus_override: dict) -> dict:
        return {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "config_id": "driver_smoke",
            "base_case": {
                "pair_id": "kv_test",
                "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
                "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
                "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
                "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
                "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
                "channel_overrides": {"g_max_S_cm2": 0.0, "v12_mV": 25.0, "q": 9.0}
            },
            "sweep_axes": {},
            "case_groups": [
                {
                    "group_id": "single_group",
                    "overrides": {
                        "stimulus": stimulus_override
                    },
                    "sweep_axes": {}
                }
            ]
        }

    def test_run_config_expand_only_writes_config_and_expanded_cases(self) -> None:
        config = sweep_config.SweepConfig.from_dict(
            self._build_single_case_config(stimulus_override={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            status = sweep_driver.run_config(config, out_dir=tmpdir, expand_only=True)
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmpdir) / "normalized_config.json").exists())
            self.assertTrue((Path(tmpdir) / "expanded_cases.json").exists())
            expanded = json.loads((Path(tmpdir) / "expanded_cases.json").read_text())
            self.assertEqual(len(expanded), 1)

    def test_run_config_executes_compare_and_writes_artifacts(self) -> None:
        config = sweep_config.SweepConfig.from_dict(
            self._build_single_case_config(
                stimulus_override={
                    "kind": "sine",
                    "start_ms": 0.0,
                    "duration_ms": 2.0,
                    "amplitude_nA": 0.02,
                    "frequency_hz": 100.0,
                    "phase_rad": 0.0,
                    "offset_nA": 0.0
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            status = sweep_driver.run_config(config, out_dir=tmpdir, expand_only=False, plot=False)
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmpdir) / "case_results" / "single_group__000.json").exists())
            self.assertTrue((Path(tmpdir) / "case_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "aggregate.json").exists())
            aggregate = json.loads((Path(tmpdir) / "aggregate.json").read_text())
            self.assertEqual(aggregate["n_total_cases"], 1)
            self.assertEqual(aggregate["n_failed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
