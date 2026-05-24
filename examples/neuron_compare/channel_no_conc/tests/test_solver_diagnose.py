import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ._helpers import TEMPLATES_ROOT, build_case_payload, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_solver_diagnose_test",
)
solver_diagnose = load_module(
    TEMPLATES_ROOT / "solver_diagnose.py",
    "channel_no_conc_solver_diagnose_test",
)


class SolverDiagnoseTest(unittest.TestCase):
    def _write_expanded_cases(self, payloads: list[dict]) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "expanded_cases.json"
        path.write_text(json.dumps(payloads))
        return path

    def _build_dc_case_payload(self, *, case_id: str, amp_nA: float) -> dict:
        payload = build_case_payload(
            case_id=case_id,
            config_name="cav2p1_ma24_pc",
            template_name="dc",
            stimulus={"kind": "dc", "delay_ms": 1.0, "dur_ms": 8.0, "amp_nA": amp_nA},
        )
        payload["simulation"]["dt_ms"] = 0.01
        payload["simulation"]["duration_ms"] = 10.0
        payload["simulation"]["temperature_celsius"] = 36.0
        return payload

    def test_diagnose_expanded_cases_reports_solver_improvement(self) -> None:
        expanded_cases_path = self._write_expanded_cases(
            [
                self._build_dc_case_payload(case_id="dc__002", amp_nA=0.1),
                self._build_dc_case_payload(case_id="dc__003", amp_nA=0.05),
            ]
        )

        time_ms = np.arange(0.01, 10.01, 0.01)
        neuron = {
            "time_ms": time_ms,
            "voltage_mV": np.linspace(-65.0, 50.0, time_ms.size),
            "current": {"ix": np.linspace(0.0, 1e-3, time_ms.size)},
            "gates": {"m": np.linspace(0.0, 0.2, time_ms.size)},
        }

        def fake_braincell(case, *, solver=None):
            solver = "staggered" if solver is None else solver
            if solver == "staggered":
                voltage = neuron["voltage_mV"] - 1.0
                current = neuron["current"]["ix"] + 1e-4
                gate = neuron["gates"]["m"] + 1e-2
            elif solver == "exp_euler":
                voltage = neuron["voltage_mV"] - 0.05
                current = neuron["current"]["ix"] + 2e-5
                gate = neuron["gates"]["m"] + 1e-3
            else:
                raise AssertionError(f"unexpected solver: {solver}")
            return {
                "time_ms": time_ms,
                "voltage_mV": voltage,
                "current": {"ix": current},
                "gates": {"m": gate},
            }

        with mock.patch.object(solver_diagnose, "run_neuron_case", return_value=neuron), mock.patch.object(
            solver_diagnose,
            "run_braincell_case",
            side_effect=fake_braincell,
        ):
            report = solver_diagnose.diagnose_expanded_cases(expanded_cases_path=expanded_cases_path)

        self.assertEqual(report["case_ids"], ["dc__002", "dc__003"])
        self.assertEqual(report["solvers"], ["staggered", "exp_euler"])
        self.assertEqual(len(report["cases"]), 2)
        for case_report in report["cases"]:
            self.assertTrue(case_report["improvement"]["one_order_of_magnitude_better"])
            self.assertLessEqual(case_report["improvement"]["voltage_mae_ratio"], 0.1)
            self.assertLessEqual(case_report["improvement"]["voltage_max_abs_ratio"], 0.1)
            self.assertEqual([row["solver"] for row in case_report["solvers"]], ["staggered", "exp_euler"])
            self.assertEqual(len(case_report["solvers"][0]["checkpoints"]), len(solver_diagnose.DEFAULT_CHECKPOINTS_MS))
        self.assertEqual(report["summary"]["n_cases_confirmed_one_order_magnitude"], 2)
        self.assertEqual(len(report["summary"]["rows"]), 4)

    def test_main_writes_json_and_csv_outputs(self) -> None:
        expanded_cases_path = self._write_expanded_cases(
            [self._build_dc_case_payload(case_id="dc__002", amp_nA=0.1)]
        )
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(out_dir, ignore_errors=True))

        fake_report = {
            "expanded_cases_path": str(expanded_cases_path),
            "case_ids": ["dc__002"],
            "solvers": ["staggered", "exp_euler"],
            "checkpoints_ms": [1.0],
            "cases": [],
            "summary": {
                "rows": [
                    {
                        "case_id": "dc__002",
                        "solver": "staggered",
                        "voltage_mae": 0.2,
                        "voltage_max_abs": 0.7,
                        "gate_m_mae": 0.001,
                        "current_ix_shifted_mae": 1e-4,
                    }
                ],
                "n_cases_confirmed_one_order_magnitude": 0,
            },
        }

        with mock.patch.object(solver_diagnose, "diagnose_expanded_cases", return_value=fake_report):
            status = solver_diagnose.main(
                [
                    str(expanded_cases_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )

        self.assertEqual(status, 0)
        json_path = out_dir / "solver_diagnostic_report.json"
        csv_path = out_dir / "solver_diagnostic_summary.csv"
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertIn("dc__002", json_path.read_text())
        self.assertIn("voltage_mae", csv_path.read_text())

