

import os
import unittest

import brainstate
import numpy as np

from ._helpers import TEMPLATES_ROOT, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")
brainstate.environ.set(precision=64)


case_schema = load_module(TEMPLATES_ROOT / "case_schema.py", "multi_compartment_cable_case_schema_runner")
braincell_runner = load_module(TEMPLATES_ROOT / "braincell_runner.py", "multi_compartment_cable_braincell_runner")
neuron_runner = load_module(TEMPLATES_ROOT / "neuron_runner.py", "multi_compartment_cable_neuron_runner")
morphology_io = load_module(TEMPLATES_ROOT / "morphology_io.py", "multi_compartment_cable_morphology_io_runner")
stimulus = load_module(TEMPLATES_ROOT / "stimulus.py", "multi_compartment_cable_stimulus_runner")
compare_module = load_module(TEMPLATES_ROOT / "compare.py", "multi_compartment_cable_compare_entry")
fixtures = load_module(TEMPLATES_ROOT / "fixtures.py", "multi_compartment_cable_fixtures_runner")


def _simple_asc_payload(testcase: unittest.TestCase) -> dict:
    asc_path = fixtures.write_temp_asc(
        testcase,
        """
        ("Cell Body"
          (CellBody)
          (0 0 0 0)
          (4 0 0 0)
          (0 4 0 0)
          (-4 0 0 0)
        )
        ( (Dendrite)
          (0 0 0 1)
          (0 5 0 1)
        )
        """,
    )
    return fixtures.base_case_payload(
        morphology_kind="asc",
        morphology_path=str(asc_path),
        dt_ms=0.025,
        duration_ms=0.1,
        cv_per_branch=1,
        stimulus=fixtures.dc_step_stimulus(delay_ms=0.0, dur_ms=0.05, amp_nA=0.01),
    )


def _synthetic_jump_swc_payload(
    testcase: unittest.TestCase,
    *,
    repeated_point_radius: float,
    case_id: str,
) -> dict:
    swc_path = fixtures.write_temp_swc(
        testcase,
        f"""
        1 1 0 0 0 5 -1
        2 1 5 0 0 5 1
        3 3 5 0 0 1 2
        4 3 15 0 0 1 3
        5 3 15 0 0 {repeated_point_radius} 4
        6 3 25 0 0 2 5
        """,
        filename=f"{case_id}.swc",
    )
    return fixtures.base_case_payload(
        case_id=case_id,
        morphology_kind="swc",
        morphology_path=str(swc_path),
        dt_ms=0.025,
        duration_ms=2.0,
        cv_per_branch=1,
        stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=1.0, amp_nA=0.05),
    )


def _manual_neuron_baseline(case) -> dict[str, np.ndarray | list[dict[str, object]]]:
    from neuron import h

    secs = morphology_io.load_neuron_sections(case)
    h.load_file("stdrun.hoc")

    for sec in secs:
        sec.nseg = int(case.cv_policy.cv_per_branch)
        sec.Ra = float(case.cable.ra_ohm_cm)
        sec.cm = float(case.cable.cm_uF_cm2)

    root_soma = morphology_io.locate_root_neuron_soma(secs)
    stim = h.IClamp(root_soma(0.5))
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0

    time_ms = np.arange(
        0.0,
        float(case.simulation.duration_ms),
        float(case.simulation.dt_ms),
        dtype=float,
    )
    time_ms = np.round(time_ms, decimals=12)
    voltage_mV = np.empty((time_ms.shape[0] + 1, sum(1 for sec in secs for _ in sec)), dtype=float)

    h.cvode_active(0)
    h.dt = float(case.simulation.dt_ms)
    h.steps_per_ms = 1.0 / h.dt
    h.dt = float(case.simulation.dt_ms)
    h.finitialize(float(case.simulation.v_init_mV))
    voltage_mV[0, :] = np.asarray([float(seg.v) for sec in secs for seg in sec], dtype=float)

    try:
        for index, t_ms in enumerate(time_ms):
            stim.amp = float(stimulus.current_at_ms(case.stimulus, float(t_ms)))
            h.fadvance()
            voltage_mV[index + 1, :] = np.asarray([float(seg.v) for sec in secs for seg in sec], dtype=float)
    finally:
        morphology_io.delete_neuron_sections(secs)

    return {
        "time_ms": time_ms,
        "voltage_mV": voltage_mV[1:, :],
    }


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
            result = braincell_runner.run_case(case)
            self.assertEqual(result["time_ms"].shape[0], result["voltage_mV"].shape[0])
            self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))
            self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_braincell_runner_supports_asc_morphology(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(_simple_asc_payload(self))
        result = braincell_runner.run_case(case)
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
            result = neuron_runner.run_case(case)
            self.assertEqual(result["time_ms"].shape[0], result["voltage_mV"].shape[0])
            self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))
            self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_neuron_runner_supports_asc_morphology(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(_simple_asc_payload(self))
        result = neuron_runner.run_case(case)
        self.assertEqual(result["time_ms"].shape[0], result["voltage_mV"].shape[0])
        self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))
        self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_neuron_runner_drops_initial_record_point(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(
                dt_ms=0.025,
                duration_ms=0.1,
                cv_per_branch=1,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.0, dur_ms=0.05, amp_nA=0.01),
            )
        )
        result = neuron_runner.run_case(case)
        self.assertEqual(result["time_ms"].tolist(), [0.0, 0.025, 0.05, 0.075])
        self.assertEqual(result["voltage_mV"].shape[0], 4)

    def test_neuron_runner_supports_inferred_swc_kind(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            {
                **fixtures.base_case_payload(),
                "morphology": {"path": str(fixtures.IO_SWC)},
            }
        )
        result = neuron_runner.run_case(case)
        self.assertEqual(case.morphology.kind, "swc")
        self.assertEqual(result["voltage_mV"].shape[1], len(result["compartment_labels"]))

    def test_neuron_runner_matches_manual_fadvance_baseline_for_dc_step(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(
                dt_ms=0.01,
                duration_ms=1.0,
                cv_per_branch=3,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=0.25, amp_nA=0.05),
            )
        )

        result = neuron_runner.run_case(case)
        baseline = _manual_neuron_baseline(case)
        diff = np.asarray(baseline["voltage_mV"], dtype=float) - np.asarray(result["voltage_mV"], dtype=float)

        self.assertTrue(np.allclose(result["time_ms"], baseline["time_ms"]))
        self.assertAlmostEqual(float(np.mean(np.abs(diff))), 0.0, places=12)
        self.assertAlmostEqual(float(np.max(np.abs(diff))), 0.0, places=12)


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
        result = compare_module.compare_case(case)
        self.assertEqual(result["case_id"], "smoke")
        self.assertIn("alignment", result)
        self.assertIn("metrics", result)
        self.assertIn("observables", result["metrics"])
        self.assertIn("per_cv", result["metrics"])
        self.assertEqual(
            len(result["alignment"]["braincell_labels"]),
            len(result["alignment"]["neuron_labels"]),
        )
        per_cv_mae = [row["mae"] for row in result["metrics"]["per_cv"]]
        self.assertAlmostEqual(
            result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"],
            float(np.mean(per_cv_mae)),
        )
        self.assertTrue(np.isfinite(result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"]))
        self.assertTrue(np.isfinite(result["metrics"]["observables"]["voltage_midpoint_mean"]["rmse"]))
        self.assertTrue(np.isfinite(result["metrics"]["observables"]["voltage_sum"]["max_abs"]))

    def test_compare_case_supports_asc_morphology(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(_simple_asc_payload(self))
        result = compare_module.compare_case(case)
        self.assertEqual(result["case_id"], "smoke")
        self.assertGreater(len(result["alignment"]["branch_pairs"]), 0)
        self.assertTrue(np.isfinite(result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"]))

    def test_compare_case_matches_neuron_for_zero_length_radius_jump_swc(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            _synthetic_jump_swc_payload(
                self,
                repeated_point_radius=2.0,
                case_id="jump_radius_step",
            )
        )
        result = compare_module.compare_case(case)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["max_abs"], 1e-6)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"], 1e-7)
        self.assertLess(result["metrics"]["observables"]["voltage_sum"]["max_abs"], 1e-6)

    def test_compare_case_stays_good_for_zero_length_without_radius_jump(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            _synthetic_jump_swc_payload(
                self,
                repeated_point_radius=1.0,
                case_id="no_jump_same_point",
            )
        )
        result = compare_module.compare_case(case)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["max_abs"], 1e-6)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"], 1e-7)
        self.assertLess(result["metrics"]["observables"]["voltage_sum"]["max_abs"], 1e-6)

    def test_compare_case_supports_real_bc_tree_with_unknown_swc_type(self) -> None:
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(
                case_id="bc_real",
                morphology_path=fixtures.BC_SWC,
                dt_ms=0.025,
                duration_ms=2.0,
                cv_per_branch=1,
                stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=1.0, amp_nA=0.05),
            )
        )
        result = compare_module.compare_case(case)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["max_abs"], 1e-6)
        self.assertLess(result["metrics"]["observables"]["voltage_midpoint_mean"]["mae"], 1e-7)
        self.assertLess(result["metrics"]["observables"]["voltage_sum"]["max_abs"], 1e-6)


if __name__ == "__main__":
    unittest.main()
