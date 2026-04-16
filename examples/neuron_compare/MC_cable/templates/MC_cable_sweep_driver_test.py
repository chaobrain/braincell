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
sweep_config = _load_module(_ROOT / "sweep_config.py", "mc_cable_sweep_config")
sweep_driver = _load_module(_ROOT / "sweep_driver.py", "mc_cable_sweep_driver")
fixtures = _load_module(_ROOT / "fixtures.py", "mc_cable_sweep_fixtures")


class SweepConfigTest(unittest.TestCase):
    def _build_config_payload(self) -> dict:
        return {
            "template_family": "multi_compartment_cable",
            "config_id": "tiny_smoke",
            "case_groups": [
                {
                    "group_id": "dc_group",
                    "base_case": {
                        **fixtures.base_case_payload(
                            swc_path=fixtures.UNBRANCHED_SOMA_SWC,
                            dt_ms=0.1,
                            duration_ms=2.0,
                            cv_per_branch=3,
                            stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=1.0, amp_nA=0.05),
                        ),
                    },
                    "sweep_axes": {
                        "swc.path": [
                            fixtures.UNBRANCHED_SOMA_SWC,
                            fixtures.BRANCHED_DEND_SWC,
                        ],
                        "cv_policy.cv_per_branch": [1, 3],
                    },
                }
            ],
        }

    def test_expand_cases_generates_case_ids_and_multiple_swcs(self) -> None:
        config = sweep_config.SweepConfig.from_dict(self._build_config_payload())
        expanded = sweep_config.expand_cases(config)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(expanded[0]["case_id"], "dc_group__000")
        self.assertEqual(expanded[-1]["case_id"], "dc_group__003")
        swc_paths = {case["swc"]["path"] for case in expanded}
        self.assertEqual(
            swc_paths,
            {
                fixtures.UNBRANCHED_SOMA_SWC,
                fixtures.BRANCHED_DEND_SWC,
            },
        )

    def test_rejects_sine_path_on_dc_group(self) -> None:
        payload = self._build_config_payload()
        payload["case_groups"][0]["sweep_axes"]["stimulus.frequency_hz"] = [100.0]
        with self.assertRaisesRegex(ValueError, "Unsupported sweep path"):
            sweep_config.SweepConfig.from_dict(payload)


class SweepDriverTest(unittest.TestCase):
    def _build_small_run_config(self) -> dict:
        return {
            "template_family": "multi_compartment_cable",
            "config_id": "driver_smoke",
            "case_groups": [
                {
                    "group_id": "two_swcs",
                    "base_case": {
                        **fixtures.base_case_payload(
                            swc_path=fixtures.UNBRANCHED_SOMA_SWC,
                            dt_ms=0.1,
                            duration_ms=2.0,
                            cv_per_branch=3,
                            stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=1.0, amp_nA=0.05),
                        ),
                    },
                    "sweep_axes": {
                        "swc.path": [
                            fixtures.UNBRANCHED_SOMA_SWC,
                            fixtures.BRANCHED_DEND_SWC,
                        ]
                    },
                }
            ],
        }

    def test_run_config_expand_only_writes_config_and_expanded_cases(self) -> None:
        config = sweep_config.SweepConfig.from_dict(self._build_small_run_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            status = sweep_driver.run_config(config, out_dir=tmpdir, expand_only=True)
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmpdir) / "normalized_config.json").exists())
            self.assertTrue((Path(tmpdir) / "expanded_cases.json").exists())
            expanded = json.loads((Path(tmpdir) / "expanded_cases.json").read_text())
            self.assertEqual(len(expanded), 2)

    def test_run_config_executes_compare_for_multiple_swcs(self) -> None:
        config = sweep_config.SweepConfig.from_dict(self._build_small_run_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            status = sweep_driver.run_config(config, out_dir=tmpdir, expand_only=False, plot=True)
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmpdir) / "case_results" / "two_swcs__000.json").exists())
            self.assertTrue((Path(tmpdir) / "case_results" / "two_swcs__001.json").exists())
            self.assertTrue((Path(tmpdir) / "case_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "plots" / "two_swcs__000.png").exists())
            self.assertTrue((Path(tmpdir) / "plots" / "summary_mae_boxplot_by_swc.png").exists())
            self.assertTrue((Path(tmpdir) / "plots" / "summary_max_abs_boxplot_by_swc_cv.png").exists())
            aggregate = json.loads((Path(tmpdir) / "aggregate.json").read_text())
            self.assertEqual(aggregate["n_total_cases"], 2)
            self.assertEqual(aggregate["n_failed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
