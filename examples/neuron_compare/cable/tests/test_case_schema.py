

import unittest

from ._helpers import TEMPLATES_ROOT, load_module


case_schema = load_module(TEMPLATES_ROOT / "case_schema.py", "multi_compartment_cable_case_schema")


class MultiCompartmentCableCaseSchemaTest(unittest.TestCase):
    def test_builds_minimal_dc_step_case(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "dc_smoke",
            "morphology": {"path": "/tmp/sample.swc"},
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
        self.assertEqual(case.morphology.kind, "swc")
        self.assertEqual(case.cv_policy.cv_per_branch, 3)
        self.assertEqual(case.stimulus.kind, "dc_step")

    def test_accepts_asc_morphology(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "asc_smoke",
            "morphology": {"path": "/tmp/sample.asc"},
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
        self.assertEqual(case.morphology.kind, "asc")

    def test_rejects_unknown_morphology_suffix_when_kind_is_omitted(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "unknown_kind",
            "morphology": {"path": "/tmp/sample.nml"},
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

        with self.assertRaisesRegex(ValueError, "morphology.kind"):
            case_schema.MultiCompartmentCableCase.from_dict(payload)

    def test_accepts_legacy_swc_field_as_compatibility_path(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "legacy_swc",
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
        self.assertEqual(case.morphology.kind, "swc")
        self.assertEqual(case.morphology.path, "/tmp/sample.swc")

    def test_piecewise_step_requires_equal_lengths(self) -> None:
        payload = {
            "template_family": "multi_compartment_cable",
            "case_id": "piecewise_bad",
            "morphology": {"kind": "swc", "path": "/tmp/sample.swc"},
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
            "morphology": {"kind": "swc", "path": "/tmp/sample.swc"},
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
            "morphology": {"kind": "swc", "path": "/tmp/sample.swc"},
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
            "morphology": {"kind": "swc", "path": "/tmp/sample.swc"},
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
