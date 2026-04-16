from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
compare_single_compartment_channel = _load_module(
    _ROOT / "compare_SC_channel.py",
    "compare_SC_channel_dispatch",
)


class CompareSingleCompartmentChannelDispatcherTest(unittest.TestCase):
    def test_single_case_dispatch_writes_json_output(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "case_id": "dispatch_case",
            "pair_id": "kv_test",
            "mod_dir": "/home/swl/braincell/examples/convert_mod/mod_validate/mods",
            "morphology": {"length_um": 10.0, "diam_um": 31.830988618379067, "cm_uF_cm2": 1.0},
            "simulation": {"dt_ms": 0.025, "duration_ms": 0.1, "v_init_mV": -65.0, "temperature_celsius": 25.0},
            "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 0.1, "amp_nA": 0.0},
            "ion": {"mode": "fixed", "fixed_E_mV": -80.0},
        }
        fake_result = {"case_id": "dispatch_case", "metrics": {}, "alignment": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "case.json"
            output_path = Path(tmpdir) / "result.json"
            input_path.write_text(json.dumps(payload))
            with mock.patch.object(compare_single_compartment_channel, "compare_hh_fixed_ion_case", return_value=fake_result):
                with mock.patch.object(
                    compare_single_compartment_channel,
                    "load_hh_fixed_ion_case",
                    return_value=object(),
                ):
                    with mock.patch.object(sys, "argv", ["compare_SC_channel.py", str(input_path), "--output", str(output_path)]):
                        status = compare_single_compartment_channel.main()
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output_path.read_text())["case_id"], "dispatch_case")

    def test_sweep_dispatch_uses_run_config(self) -> None:
        payload = {
            "template_family": "single_compartment_channel",
            "template_variant": "hh_fixed_ion",
            "config_id": "dispatch_sweep",
            "case_groups": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "config.json"
            input_path.write_text(json.dumps(payload))
            with mock.patch.object(compare_single_compartment_channel, "load_hh_fixed_ion_config", return_value="CONFIG"), mock.patch.object(
                compare_single_compartment_channel,
                "run_hh_fixed_ion_sweep",
                return_value=0,
            ) as run_mock:
                with mock.patch.object(sys, "argv", ["compare_SC_channel.py", str(input_path), "--expand-only"]):
                    status = compare_single_compartment_channel.main()
            self.assertEqual(status, 0)
            run_mock.assert_called_once_with("CONFIG", out_dir=None, expand_only=True, plot=None)


if __name__ == "__main__":
    unittest.main()
