from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

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
case_schema = _load_module(_ROOT / "case_schema.py", "multi_compartment_cable_case_schema_runner")
braincell_single_case = _load_module(_ROOT / "braincell_single_case.py", "multi_compartment_cable_braincell_single_case")
neuron_single_case = _load_module(_ROOT / "neuron_single_case.py", "multi_compartment_cable_neuron_single_case")
compare_multi_compartment_cable = _load_module(
    _ROOT / "compare_MC_cable.py",
    "multi_compartment_cable_compare_entry",
)
fixtures = _load_module(_ROOT / "fixtures.py", "multi_compartment_cable_fixtures_runner")


class BraincellRunnerTest(unittest.TestCase):
    def test_braincell_runner_supports_dc_piecewise_and_sine(self) -> None:
        payloads = [
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.0, dur_ms=0.05, amp_nA=0.01),
            ),
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.piecewise_step_stimulus(start_ms=0.0, durations_ms=(0.025, 0.025), amplitudes_nA=(0.0, 0.01)),
            ),
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.sine_stimulus(start_ms=0.0, duration_ms=0.1, amplitude_nA=0.01, frequency_hz=100.0),
            ),
        ]

        for payload in payloads:
            case = case_schema.MultiCompartmentCableCase.from_dict(payload)
            result = braincell_single_case.run_case(case)
            self.assertEqual(result["time_ms"].shape[0], result["voltage_mV"].shape[0])
            self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))
            self.assertTrue(np.isfinite(result["voltage_mV"]).all())


class NeuronRunnerTest(unittest.TestCase):
    def test_neuron_runner_supports_dc_piecewise_and_sine(self) -> None:
        payloads = [
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.0, dur_ms=0.05, amp_nA=0.01),
            ),
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.piecewise_step_stimulus(start_ms=0.0, durations_ms=(0.025, 0.025), amplitudes_nA=(0.0, 0.01)),
            ),
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.sine_stimulus(start_ms=0.0, duration_ms=0.1, amplitude_nA=0.01, frequency_hz=100.0),
            ),
        ]

        for payload in payloads:
            case = case_schema.MultiCompartmentCableCase.from_dict(payload)
            result = neuron_single_case.run_case(case)
            self.assertEqual(result["time_ms"].shape[0], result["voltage_mV"].shape[0])
            self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))
            self.assertTrue(np.isfinite(result["voltage_mV"]).all())


class CompareRunnerTest(unittest.TestCase):
    def test_compare_case_returns_aligned_voltage_metrics(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.0, dur_ms=0.05, amp_nA=0.01),
            )
        )
        result = compare_multi_compartment_cable.compare_case(case)
        self.assertEqual(result["case_id"], "smoke")
        self.assertIn("alignment", result)
        self.assertIn("metrics", result)
        self.assertIn("overall", result["metrics"])
        self.assertIn("per_compartment", result["metrics"])
        self.assertEqual(
            len(result["alignment"]["braincell_labels"]),
            len(result["alignment"]["neuron_labels"]),
        )
        self.assertTrue(np.isfinite(result["metrics"]["overall"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["overall"]["rmse"]))
        self.assertTrue(np.isfinite(result["metrics"]["overall"]["max_abs"]))


if __name__ == "__main__":
    unittest.main()
