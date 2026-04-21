

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib

from ._helpers import (
    CABLE_ROOT,
    build_model_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


matplotlib.use("Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


workflow_api = load_module(
    CABLE_ROOT / "workflow_api.py",
    "cable_workflow_api_test",
)


class WorkflowInputTest(unittest.TestCase):
    def test_repo_workflow_notebook_defaults_to_io_dc_smoke(self) -> None:
        notebook_path = CABLE_ROOT / "docs" / "workflow.ipynb"
        notebook = json.loads(notebook_path.read_text())
        parameter_cell = "".join(notebook["cells"][3]["source"])

        self.assertIn('config_path = CABLE_ROOT / "configs" / "cable_demo.json"', parameter_cell)
        self.assertIn('template_path = CABLE_ROOT / "scan_templates" / "io_dc_smoke_v1.json"', parameter_cell)

    def test_repo_cable_demo_config_includes_io_dc_smoke_template(self) -> None:
        payload = json.loads((CABLE_ROOT / "configs" / "cable_demo.json").read_text())

        self.assertIn("../scan_templates/io_dc_smoke_v1.json", payload["templates"])

    def test_load_workflow_inputs_reads_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "scan_templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "smoke.json",
                build_scan_template_payload(sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]}),
            )
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["scan_templates/smoke.json"]),
            )

            info = workflow_api.load_workflow_inputs(config_path, template_path)

        self.assertEqual(info["config_name"], "cable_demo")
        self.assertEqual(info["template_name"], "smoke")
        self.assertEqual(info["run_id"], "cable_demo__smoke")
        self.assertEqual(info["group_id"], "smoke")
        self.assertEqual(info["n_expanded_cases"], 2)
        self.assertEqual(info["default_out_dir"].name, "cable_demo__smoke")

    def test_discover_batch_configs_expands_config_x_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "scan_templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            write_json(
                template_dir / "vinit.json",
                build_scan_template_payload(
                    group_id="v_init_scan",
                    sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]},
                ),
            )
            write_json(
                root / "cable_demo.json",
                build_model_config_payload(["scan_templates/smoke.json", "scan_templates/vinit.json"]),
            )

            records = workflow_api.discover_batch_configs(root)

        self.assertEqual([record["run_id"] for record in records], ["cable_demo__smoke", "cable_demo__vinit"])
        self.assertEqual([record["group_id"] for record in records], ["smoke", "v_init_scan"])


class WorkflowRunSmokeTest(unittest.TestCase):
    def _write_config_and_template(self, tmpdir: str) -> tuple[Path, Path]:
        root = Path(tmpdir)
        template_dir = root / "scan_templates"
        template_dir.mkdir()
        template_path = write_json(
            template_dir / "smoke.json",
            build_scan_template_payload(
                group_id="single_group",
                sweep_axes={"stimulus.amp_nA": [0.0, 0.01]},
            ),
        )
        config_path = write_json(
            root / "cable_demo.json",
            build_model_config_payload(["scan_templates/smoke.json"]),
        )
        return config_path, template_path

    def test_run_load_and_plot_workflow_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, template_path = self._write_config_and_template(tmpdir)
            out_dir = Path(tmpdir) / "out"

            run_info = workflow_api.run_notebook_workflow(
                config_path=config_path,
                template_path=template_path,
                out_dir=out_dir,
                plot=False,
            )
            self.assertEqual(run_info["status"], 0)
            self.assertEqual(run_info["run_id"], "cable_demo__smoke")
            self.assertTrue(run_info["normalized_config_path"].exists())
            self.assertTrue(run_info["expanded_cases_path"].exists())
            self.assertTrue(run_info["case_metrics_path"].exists())
            self.assertTrue(run_info["aggregate_path"].exists())

            artifacts = workflow_api.load_run_artifacts(out_dir)
            self.assertEqual(len(artifacts["expanded_cases"]), 2)
            self.assertEqual(artifacts["aggregate"]["n_failed_cases"], 0)
            self.assertEqual(len(artifacts["cases_df"]), 2)

            tables = workflow_api.build_summary_tables(out_dir)
            self.assertIn("merged_df", tables)
            self.assertIn("worst_cases_df", tables)
            self.assertIn("by_morphology_df", tables)
            self.assertEqual(len(tables["ok_df"]), len(tables["metrics_df"]))
            self.assertEqual(len(tables["failed_df"]), 0)

            case_id = artifacts["expanded_cases"][0]["case_id"]
            case_result = workflow_api.load_case_result(out_dir, case_id)
            metric_table = workflow_api.build_case_metric_table(case_result)
            self.assertGreater(len(metric_table), 0)

            sweep_fig, sweep_axes = workflow_api.plot_sweep_summary(tables, metric="max_abs")
            overlay_fig, overlay_axes = workflow_api.plot_case_overlay(case_result)
            error_fig, error_axes = workflow_api.plot_case_error_summary(case_result)
            self.assertEqual(sweep_axes.shape, (2, 2))
            self.assertEqual(len(overlay_axes), 2)
            self.assertEqual(len(error_axes), 2)
            sweep_fig.clf()
            overlay_fig.clf()
            error_fig.clf()


class BatchWorkflowTest(WorkflowRunSmokeTest):
    def test_run_notebook_batch_writes_summary_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "scan_templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            write_json(
                template_dir / "vinit.json",
                build_scan_template_payload(
                    group_id="v_init_scan",
                    sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]},
                ),
            )
            write_json(
                root / "cable_demo.json",
                build_model_config_payload(["scan_templates/smoke.json", "scan_templates/vinit.json"]),
            )

            run_records = workflow_api.discover_batch_configs(root)
            summary_dir = root / "batch_summary"
            batch_result = workflow_api.run_notebook_batch(
                run_records,
                plot=False,
                summary_dir=summary_dir,
            )

            self.assertTrue(batch_result["manifest_path"].exists())
            self.assertTrue(batch_result["config_runs_path"].exists())
            self.assertTrue(batch_result["observable_summary_path"].exists())
            self.assertTrue(batch_result["failures_path"].exists())

            manifest = json.loads(batch_result["manifest_path"].read_text())
            self.assertEqual(manifest["n_runs"], 2)
            self.assertEqual(manifest["status_counts"]["ok"], 2)
            self.assertEqual(manifest["status_counts"]["partial"], 0)
            self.assertEqual(manifest["status_counts"]["failed"], 0)

            tables = workflow_api.build_batch_summary_tables(batch_result)
            self.assertEqual(len(tables["runs_df"]), 2)
            self.assertTrue((tables["runs_df"]["batch_status"] == "ok").all())
            self.assertGreater(len(tables["observables_df"]), 0)
            self.assertEqual(len(tables["failures_df"]), 0)

    def test_run_notebook_batch_continues_after_partial_and_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "cable_demo.json"
            template_path = tmp_path / "scan_templates" / "smoke.json"
            template_path.parent.mkdir()
            config_path.write_text("{}")
            template_path.write_text("{}")

            run_records = [
                {
                    "config_path": config_path,
                    "template_path": template_path,
                    "config_name": "cable_demo",
                    "template_name": "smoke",
                    "run_id": "cable_demo__smoke",
                    "group_id": "smoke",
                    "default_out_dir": tmp_path / "runs" / "cable_demo__smoke",
                    "n_expanded_cases": 4,
                },
                {
                    "config_path": config_path,
                    "template_path": tmp_path / "scan_templates" / "failed.json",
                    "config_name": "cable_demo",
                    "template_name": "failed",
                    "run_id": "cable_demo__failed",
                    "group_id": "failed",
                    "default_out_dir": tmp_path / "runs" / "cable_demo__failed",
                    "n_expanded_cases": 4,
                },
            ]

            def fake_run_notebook_workflow(*, template_path, out_dir=None, **kwargs):
                resolved_template_path = Path(template_path).resolve()
                resolved_out_dir = Path(out_dir).resolve()
                if resolved_template_path.name == "smoke.json":
                    resolved_out_dir.mkdir(parents=True, exist_ok=True)
                    aggregate = {
                        "config_id": "cable_demo__smoke",
                        "n_total_cases": 4,
                        "n_success_cases": 3,
                        "n_failed_cases": 1,
                        "observables": {
                            "voltage": {
                                "n_cases": 3,
                                "mae_mean": 0.1,
                                "rmse_mean": 0.2,
                                "max_abs_max": 0.3,
                                "rel_mae_pct_mean": 0.4,
                            }
                        },
                        "failed_cases": [{"case_id": "smoke__001", "error_message": "bad case"}],
                    }
                    (resolved_out_dir / "aggregate.json").write_text(json.dumps(aggregate))
                    return {
                        "status": 0,
                        "config_path": config_path.resolve(),
                        "template_path": resolved_template_path,
                        "run_id": "cable_demo__smoke",
                        "out_dir": resolved_out_dir,
                    }
                raise RuntimeError("boom")

            with mock.patch.object(workflow_api, "run_notebook_workflow", side_effect=fake_run_notebook_workflow):
                batch_result = workflow_api.run_notebook_batch(run_records, plot=False, summary_dir=tmp_path / "summary")

            self.assertEqual(len(batch_result["runs"]), 2)
            self.assertEqual(batch_result["runs"][0]["batch_status"], "partial")
            self.assertEqual(batch_result["runs"][1]["batch_status"], "failed")
            self.assertEqual(len(batch_result["failures"]), 2)


if __name__ == "__main__":
    unittest.main()
