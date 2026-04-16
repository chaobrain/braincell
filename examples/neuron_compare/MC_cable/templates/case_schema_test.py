from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
case_schema = _load_module(_ROOT / "case_schema.py", "multi_compartment_cable_case_schema")


class MultiCompartmentCableCaseSchemaTest(unittest.TestCase):
    def test_builds_minimal_dc_step_case(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "dc_smoke",
            "swc": {"path": "/tmp/sample.swc"},
            "simulation": {"dt_ms": 0.025, "duration_ms": 5.0, "v_init_mV": -65.0},
            "cable": {"ra_ohm_cm": 100.0, "cm_uF_cm2": 1.0},
            "cv_policy": {"kind": "CVPerBranch", "cv_per_branch": 3},
            "stimulus": {
                "kind": "dc_step",
                "target": "root_soma_midpoint",
                "delay_ms": 0.5,
                "dur_ms": 1.0,
                "amp_nA": 0.05,
            },
        }

        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        self.assertEqual(case.template_family, "multi_compartment_cable")
        self.assertEqual(case.cv_policy.cv_per_branch, 3)
        self.assertEqual(case.stimulus.kind, "dc_step")

    def test_piecewise_step_requires_equal_lengths(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "piecewise_bad",
            "swc": {"path": "/tmp/sample.swc"},
            "simulation": {"dt_ms": 0.025, "duration_ms": 5.0, "v_init_mV": -65.0},
            "cable": {"ra_ohm_cm": 100.0, "cm_uF_cm2": 1.0},
            "cv_policy": {"kind": "CVPerBranch", "cv_per_branch": 3},
            "stimulus": {
                "kind": "piecewise_step",
                "start_ms": 0.0,
                "durations_ms": [1.0, 2.0],
                "amplitudes_nA": [0.0],
            },
        }

        with self.assertRaises(ValueError):
            case_schema.MultiCompartmentCableCase.from_dict(payload)

    def test_sine_requires_positive_frequency(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "sine_bad",
            "swc": {"path": "/tmp/sample.swc"},
            "simulation": {"dt_ms": 0.025, "duration_ms": 5.0, "v_init_mV": -65.0},
            "cable": {"ra_ohm_cm": 100.0, "cm_uF_cm2": 1.0},
            "cv_policy": {"kind": "CVPerBranch", "cv_per_branch": 3},
            "stimulus": {
                "kind": "sine",
                "start_ms": 0.0,
                "duration_ms": 4.0,
                "amplitude_nA": 0.05,
                "frequency_hz": 0.0,
            },
        }

        with self.assertRaises(ValueError):
            case_schema.MultiCompartmentCableCase.from_dict(payload)

    def test_rejects_even_cv_per_branch(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "even_cv",
            "swc": {"path": "/tmp/sample.swc"},
            "simulation": {"dt_ms": 0.025, "duration_ms": 5.0, "v_init_mV": -65.0},
            "cable": {"ra_ohm_cm": 100.0, "cm_uF_cm2": 1.0},
            "cv_policy": {"kind": "CVPerBranch", "cv_per_branch": 2},
            "stimulus": {
                "kind": "dc_step",
                "delay_ms": 0.5,
                "dur_ms": 1.0,
                "amp_nA": 0.05,
            },
        }

        with self.assertRaises(ValueError):
            case_schema.MultiCompartmentCableCase.from_dict(payload)

    def test_defaults_target_to_root_soma_midpoint(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "default_target",
            "swc": {"path": "/tmp/sample.swc"},
            "simulation": {"dt_ms": 0.025, "duration_ms": 5.0, "v_init_mV": -65.0},
            "cable": {"ra_ohm_cm": 100.0, "cm_uF_cm2": 1.0},
            "cv_policy": {"kind": "CVPerBranch", "cv_per_branch": 1},
            "stimulus": {
                "kind": "sine",
                "start_ms": 0.0,
                "duration_ms": 2.0,
                "amplitude_nA": 0.05,
                "frequency_hz": 25.0,
            },
        }

        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        self.assertEqual(case.stimulus.target, "root_soma_midpoint")


if __name__ == "__main__":
    unittest.main()
