
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib

from ._helpers import (
    CABLE_ROOT,
    WORKFLOWS_ROOT,
    build_model_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


matplotlib.use("Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


workflow_api = load_module(
    WORKFLOWS_ROOT / "workflow_api.py",
    "cable_workflow_api_test",
)


class WorkflowInputTest(unittest.TestCase):
    def test_repo_workflow_notebook_defaults_to_config_only(self) -> None:
        notebook_path = WORKFLOWS_ROOT / "workflow.ipynb"
        notebook = json.loads(notebook_path.read_text())
        parameter_cell = "".join(notebook["cells"][3]["source"])
        run_cell = "".join(notebook["cells"][7]["source"])
        summary_cell = "".join(notebook["cells"][8]["source"])
        drilldown_cell = "".join(notebook["cells"][9]["source"])

        self.assertIn('config_path = CABLE_ROOT / "configs" /', parameter_cell)
        self.assertIn("plot_cases = True", parameter_cell)
        self.assertNotIn("out_dir =", parameter_cell)
        self.assertNotIn('template_path =', parameter_cell)
        self.assertNotIn("selected_template_name", parameter_cell)
        self.assertNotIn("selected_case_id", parameter_cell)
        self.assertIn("template_run = next(", run_cell)
        self.assertIn("tables = workflow_api.build_summary_tables(template_out_dir)", run_cell)
        self.assertIn('display(tables["by_observable_df"])', summary_cell)
        self.assertIn("plot_observable_metric_boxplots", summary_cell)
        self.assertNotIn("by_morphology_df", summary_cell)
        self.assertIn('worst_case_id = tables["worst_cases_df"].iloc[0]["case_id"]', drilldown_cell)
        self.assertNotIn("selected_case_id", drilldown_cell)

    def test_repo_io_config_includes_expected_templates(self) -> None:
        payload = json.loads((CABLE_ROOT / "configs" / "IO.json").read_text())

        self.assertEqual(
            payload["templates"],
            [
                "../templates/ac.json",
                "../templates/dc.json",
                "../templates/vinit.json",
                "../templates/cv.json",
            ],
        )
        self.assertIn("defaults", payload)
        self.assertIn("morphology", payload["defaults"])
        self.assertIn("path", payload["defaults"]["morphology"])
        self.assertNotIn("simulation", payload["defaults"])

    def test_repo_config_dir_discovers_all_cerebellum_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CABLE_ROOT / "configs")

        self.assertEqual(
            [record["config_name"] for record in records],
            ["BC", "GoC", "GrC", "IO", "PC", "SC"],
        )
        self.assertTrue(all(record["n_templates"] == 4 for record in records))

    def test_load_config_workflow_inputs_reads_declared_templates_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
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
                build_model_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            info = workflow_api.load_config_workflow_inputs(root / "cable_demo.json")

        self.assertEqual(info["config_name"], "cable_demo")
        self.assertEqual(info["n_templates"], 2)
        self.assertEqual(info["template_names"], ("smoke", "vinit"))

    def test_discover_batch_configs_discovers_one_record_per_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
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
                build_model_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            records = workflow_api.discover_batch_configs(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["run_id"], "cable_demo")
        self.assertEqual(records[0]["n_templates"], 2)
        self.assertEqual(records[0]["template_names"], ["smoke", "vinit"])


class WorkflowRunSmokeTest(unittest.TestCase):
    def _write_config(self, tmpdir: str) -> Path:
        root = Path(tmpdir)
        template_dir = root / "templates"
        template_dir.mkdir()
        write_json(
            template_dir / "smoke.json",
            build_scan_template_payload(
                group_id="single_group",
                sweep_axes={"stimulus.amp_nA": [0.0, 0.01]},
            ),
        )
        write_json(
            template_dir / "vinit.json",
            build_scan_template_payload(
                group_id="v_init_scan",
                sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]},
            ),
        )
        return write_json(
            root / "cable_demo.json",
            build_model_config_payload(["templates/smoke.json", "templates/vinit.json"]),
        )

    def test_run_notebook_config_workflow_writes_parent_dir_and_template_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir)
            out_dir = Path(tmpdir) / "out"

            run_info = workflow_api.run_notebook_config_workflow(
                config_path=config_path,
                out_dir=out_dir,
                plot=False,
            )

            self.assertEqual(run_info["status"], 0)
            self.assertEqual(run_info["config_name"], "cable_demo")
            self.assertEqual(run_info["n_templates"], 2)
            self.assertTrue(run_info["manifest_path"].exists())
            self.assertTrue(run_info["config_runs_path"].exists())
            self.assertTrue(run_info["observable_summary_path"].exists())
            self.assertTrue(run_info["observable_summary_json_path"].exists())
            self.assertTrue(run_info["failures_path"].exists())

            smoke_dir = out_dir / "templates" / "smoke"
            vinit_dir = out_dir / "templates" / "vinit"
            self.assertTrue((smoke_dir / "aggregate.json").exists())
            self.assertTrue((vinit_dir / "aggregate.json").exists())

            artifacts = workflow_api.load_run_artifacts(smoke_dir)
            self.assertEqual(len(artifacts["expanded_cases"]), 2)
            self.assertEqual(artifacts["aggregate"]["n_failed_cases"], 0)

            tables = workflow_api.build_summary_tables(smoke_dir)
            self.assertIn("merged_df", tables)
            self.assertIn("worst_cases_df", tables)
            self.assertIn("by_morphology_df", tables)
            self.assertEqual(len(tables["ok_df"]), len(tables["metrics_df"]))

            case_id = artifacts["expanded_cases"][0]["case_id"]
            case_result = workflow_api.load_case_result(smoke_dir, case_id)
            metric_table = workflow_api.build_case_metric_table(case_result)
            self.assertGreater(len(metric_table), 0)

            sweep_fig, sweep_axes = workflow_api.plot_sweep_summary(tables, metric="max_abs")
            family_fig, family_axes = workflow_api.plot_observable_metric_boxplots(tables)
            overlay_fig, overlay_axes = workflow_api.plot_case_overlay(case_result)
            error_fig, error_axes = workflow_api.plot_case_error_summary(case_result)
            self.assertEqual(sweep_axes.shape, (2, 2))
            self.assertEqual(len(family_axes), 4)
            self.assertEqual(len(overlay_axes), 2)
            self.assertEqual(len(error_axes), 2)
            sweep_fig.clf()
            family_fig.clf()
            overlay_fig.clf()
            error_fig.clf()


class BatchWorkflowTest(WorkflowRunSmokeTest):
    def test_run_notebook_batch_writes_summary_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
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
                build_model_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            config_records = workflow_api.discover_batch_configs(root)
            summary_dir = root / "batch_summary"
            batch_result = workflow_api.run_notebook_batch(
                config_records,
                plot=False,
                summary_dir=summary_dir,
                batch_run_id="test_batch",
            )

            self.assertTrue(batch_result["manifest_path"].exists())
            self.assertTrue(batch_result["config_runs_path"].exists())
            self.assertTrue(batch_result["batch_observable_summary_path"].exists())
            self.assertTrue(batch_result["batch_observable_summary_json_path"].exists())
            self.assertTrue(batch_result["batch_failures_path"].exists())

            manifest = json.loads(batch_result["manifest_path"].read_text())
            self.assertEqual(manifest["n_configs"], 1)
            self.assertEqual(manifest["status_counts"]["ok"], 1)
            self.assertEqual(manifest["status_counts"]["partial"], 0)
            self.assertEqual(manifest["status_counts"]["failed"], 0)

            tables = workflow_api.build_batch_summary_tables(batch_result)
            self.assertEqual(len(tables["config_rows_df"]), 1)
            self.assertTrue((tables["config_rows_df"]["batch_status"] == "ok").all())
            self.assertGreater(len(tables["observables_df"]), 0)
            self.assertEqual(len(tables["failures_df"]), 0)

    def test_run_notebook_batch_continues_after_partial_and_failed_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_records = [
                {
                    "config_path": root / "ok.json",
                    "config_name": "ok",
                    "run_id": "ok",
                    "template_paths": [root / "templates" / "smoke.json"],
                    "template_names": ["smoke"],
                    "n_templates": 1,
                    "default_out_dir": root / "results" / "config_runs" / "ok",
                },
                {
                    "config_path": root / "failed.json",
                    "config_name": "failed",
                    "run_id": "failed",
                    "template_paths": [root / "templates" / "boom.json"],
                    "template_names": ["boom"],
                    "n_templates": 1,
                    "default_out_dir": root / "results" / "config_runs" / "failed",
                },
            ]

            def fake_run_notebook_config_workflow(*, config_path, out_dir=None, **kwargs):
                resolved_out_dir = Path(out_dir).resolve()
                resolved_out_dir.mkdir(parents=True, exist_ok=True)
                config_name = Path(config_path).stem
                if config_name == "ok":
                    observable_summary_json_path = resolved_out_dir / "observable_summary.json"
                    observable_summary_json_path.write_text(
                        json.dumps(
                            {
                                "all_templates": {
                                    "observables": {
                                        "voltage_midpoint_mean": {
                                            "n_cases": 3,
                                            "mae_mean": 0.1,
                                            "rmse_mean": 0.2,
                                            "max_abs_max": 0.3,
                                            "rel_mae_pct_mean": 0.4,
                                        }
                                    }
                                }
                            }
                        )
                    )
                    return {
                        "status": 0,
                        "config_path": str(config_path),
                        "config_name": "ok",
                        "out_dir": resolved_out_dir,
                        "observable_summary_json_path": observable_summary_json_path,
                        "failures": [{"template_name": "smoke", "template_path": "smoke.json", "out_dir": "x", "n_failed_cases": 1, "error_message": "partial"}],
                        "status_counts": {"ok": 0, "partial": 1, "failed": 0},
                        "n_templates": 1,
                        "n_total_cases": 3,
                        "n_success_cases": 2,
                        "n_failed_cases": 1,
                    }
                observable_summary_json_path = resolved_out_dir / "observable_summary.json"
                observable_summary_json_path.write_text(json.dumps({"all_templates": {"observables": {}}}))
                return {
                    "status": 1,
                    "config_path": str(config_path),
                    "config_name": "failed",
                    "out_dir": resolved_out_dir,
                    "observable_summary_json_path": observable_summary_json_path,
                    "failures": [{"template_name": "boom", "template_path": "boom.json", "out_dir": "y", "n_failed_cases": "", "error_message": "boom"}],
                    "status_counts": {"ok": 0, "partial": 0, "failed": 1},
                    "n_templates": 1,
                    "n_total_cases": 0,
                    "n_success_cases": 0,
                    "n_failed_cases": 0,
                }

            with mock.patch.object(workflow_api, "run_notebook_config_workflow", side_effect=fake_run_notebook_config_workflow):
                batch_result = workflow_api.run_notebook_batch(
                    config_records,
                    plot=False,
                    summary_dir=root / "summary",
                    batch_run_id="test_batch",
                )

            self.assertEqual(len(batch_result["config_rows"]), 2)
            self.assertEqual(batch_result["config_rows"][0]["batch_status"], "partial")
            self.assertEqual(batch_result["config_rows"][1]["batch_status"], "failed")
            self.assertEqual(len(batch_result["failure_rows"]), 2)


if __name__ == "__main__":
    unittest.main()
