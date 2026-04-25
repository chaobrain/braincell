

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
import numpy as np

from ._helpers import (
    CHANNEL_NO_CONC_ROOT,
    MOD_VALIDATE_MOD_DIR,
    build_main_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


matplotlib.use("Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


workflow_api = load_module(
    CHANNEL_NO_CONC_ROOT / "workflows" / "workflow_api.py",
    "channel_no_conc_workflow_api_test",
)


class WorkflowInputTest(unittest.TestCase):
    def test_load_workflow_inputs_reads_mapping_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "smoke.json",
                build_scan_template_payload(sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]}),
            )
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json"]),
            )

            info = workflow_api.load_workflow_inputs(config_path, template_path)

        self.assertEqual(info["config_name"], "kv_test")
        self.assertEqual(info["template_name"], "smoke")
        self.assertEqual(info["run_id"], "kv_test__smoke")
        self.assertEqual(info["group_id"], "smoke")
        self.assertEqual(info["n_expanded_cases"], 2)
        self.assertEqual(info["mapping"]["current"], "ik")
        self.assertEqual(info["mod_dir"], Path(MOD_VALIDATE_MOD_DIR).resolve())

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
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            info = workflow_api.load_config_workflow_inputs(root / "kv_test.json")

        self.assertEqual(info["config_name"], "kv_test")
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
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            records = workflow_api.discover_batch_configs(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["run_id"], "kv_test")
        self.assertEqual(records[0]["n_templates"], 2)
        self.assertEqual(records[0]["template_names"], ["smoke", "vinit"])

    def test_repo_ma24_pc_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(set(records_by_name), {"hcn1_ma24_pc", "kir2p3_ma24_pc", "kv1p1_ma24_pc", "kv3p4_ma24_pc", "kv4p3_ma24_pc"})
        self.assertIn("hcn1_ma24_pc", records_by_name)
        self.assertEqual(records_by_name["hcn1_ma24_pc"]["n_templates"], 3)

    def test_repo_ma25_bc_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(set(records_by_name), {"hcn1_ma25_bc", "nav1p6_ma25_bc", "kir2p3_ma25_bc", "kv1p1_ma25_bc", "kv3p4_ma25_bc", "kv4p3_ma25_bc"})
        self.assertIn("hcn1_ma25_bc", records_by_name)
        self.assertEqual(records_by_name["hcn1_ma25_bc"]["n_templates"], 3)

    def test_repo_ma20_goc_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(
            set(records_by_name),
            {
                "nav1p6_ma20_goc",
                "hcn1_ma20_goc",
                "hcn2_ma20_goc",
                "km_ma20_goc",
                "kv1p1_ma20_goc",
                "kv3p4_ma20_goc",
                "kv4p3_ma20_goc",
                "cahva_ma20_goc",
                "cav2p3_ma20_goc",
            },
        )
        self.assertEqual(records_by_name["hcn1_ma20_goc"]["n_templates"], 3)

    def test_repo_su15_dcn_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(
            set(records_by_name),
            {"hcn_su15_dcn", "naf_su15_dcn", "nap_su15_dcn", "fkdr_su15_dcn", "skdr_su15_dcn"},
        )
        self.assertEqual(records_by_name["hcn_su15_dcn"]["n_templates"], 3)

    def test_repo_ri21_sc_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(
            set(records_by_name),
            {"hcn1_ri21_sc", "km_ri21_sc", "kir2p3_ri21_sc", "kv1p1_ri21_sc", "kv3p4_ri21_sc", "kv4p3_ri21_sc"},
        )
        self.assertEqual(records_by_name["hcn1_ri21_sc"]["n_templates"], 3)

    def test_repo_su15_dcn_config_dir_discovers_all_supported_examples(self) -> None:
        records = workflow_api.discover_batch_configs(CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn")

        records_by_name = {record["config_name"]: record for record in records}
        self.assertEqual(
            set(records_by_name),
            {"hcn_su15_dcn", "naf_su15_dcn", "nap_su15_dcn", "fkdr_su15_dcn", "skdr_su15_dcn"},
        )
        self.assertEqual(records_by_name["hcn_su15_dcn"]["n_templates"], 3)


class WorkflowRunSmokeTest(unittest.TestCase):
    def _write_config_and_template(self, tmpdir: str) -> tuple[Path, Path]:
        root = Path(tmpdir)
        template_dir = root / "templates"
        template_dir.mkdir()
        template_path = write_json(
            template_dir / "smoke.json",
            build_scan_template_payload(
                group_id="single_group",
                sweep_axes={"stimulus.amp_nA": [0.0, 0.01]},
            ),
        )
        config_path = write_json(
            root / "kv_test.json",
            build_main_config_payload(["templates/smoke.json"]),
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
            self.assertEqual(run_info["run_id"], "kv_test__smoke")
            self.assertTrue(run_info["normalized_config_path"].exists())
            self.assertTrue(run_info["expanded_cases_path"].exists())
            self.assertTrue(run_info["case_metrics_path"].exists())
            self.assertTrue(run_info["aggregate_path"].exists())
            self.assertNotIn("notebook_plots_dir", run_info)

            artifacts = workflow_api.load_run_artifacts(out_dir)
            self.assertEqual(len(artifacts["expanded_cases"]), 2)
            self.assertEqual(artifacts["aggregate"]["n_failed_cases"], 0)
            self.assertGreaterEqual(len(artifacts["metrics_df"]), 2)
            self.assertEqual(len(artifacts["cases_df"]), 2)
            self.assertNotIn("notebook_plots_dir", artifacts)

            tables = workflow_api.build_summary_tables(out_dir)
            self.assertIn("merged_df", tables)
            self.assertIn("worst_cases_df", tables)
            self.assertIn("by_observable_df", tables)
            self.assertEqual(len(tables["ok_df"]), len(tables["metrics_df"]))
            self.assertEqual(len(tables["failed_df"]), 0)

            case_id = artifacts["expanded_cases"][0]["case_id"]
            case_result = workflow_api.load_case_result(out_dir, case_id)
            self.assertEqual(case_result["case_id"], case_id)

            sweep_fig, sweep_axes = workflow_api.plot_sweep_summary(tables, metric="max_abs")
            family_fig, family_axes = workflow_api.plot_observable_metric_boxplots(tables)
            overlay_fig, overlay_axes = workflow_api.plot_case_overlay(case_result)
            error_fig, error_axes = workflow_api.plot_case_error_summary(case_result)
            self.assertEqual(sweep_axes.shape, (2, 2))
            self.assertEqual(len(family_axes), 4)
            self.assertEqual(len(overlay_axes), 3)
            self.assertEqual(len(error_axes), 2)
            for metric in ("max_abs", "mae"):
                summary_fig_metric, summary_axes_metric = workflow_api.plot_sweep_summary(tables, metric=metric)
                summary_text = summary_axes_metric[1, 1].texts[0].get_text()
                self.assertIn("n_total_cases:", summary_text)
                self.assertIn("n_success_cases:", summary_text)
                self.assertIn("n_failed_cases:", summary_text)
                self.assertNotIn("observables:", summary_text)
                self.assertNotIn("max_abs_max=", summary_text)
                summary_fig_metric.clf()
            sweep_fig.clf()
            family_fig.clf()
            overlay_fig.clf()
            error_fig.clf()

    def test_run_notebook_config_workflow_writes_parent_dir_and_template_children(self) -> None:
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
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )
            out_dir = root / "config_run"

            run_info = workflow_api.run_notebook_config_workflow(
                config_path=config_path,
                out_dir=out_dir,
                plot=True,
            )

            self.assertEqual(run_info["status"], 0)
            self.assertTrue(run_info["manifest_path"].exists())
            self.assertTrue(run_info["config_runs_path"].exists())
            self.assertTrue(run_info["observable_summary_path"].exists())
            self.assertTrue(run_info["observable_summary_json_path"].exists())
            self.assertTrue((out_dir / "templates" / "smoke" / "normalized_config.json").exists())
            self.assertTrue((out_dir / "templates" / "vinit" / "expanded_cases.json").exists())
            self.assertEqual(run_info["n_templates"], 2)
            self.assertEqual(run_info["status_counts"]["ok"], 2)
            self.assertTrue((out_dir / "all_templates_observable_summary.png").exists())
            self.assertTrue((out_dir / "boxplot_by_template.png").exists())
            self.assertTrue((out_dir / "boxplot_by_observable_family.png").exists())



class BatchWorkflowTest(WorkflowRunSmokeTest):
    def test_write_batch_summary_artifacts_writes_summary_tables(self) -> None:
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
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json", "templates/vinit.json"]),
            )

            config_records = workflow_api.discover_batch_configs(root)
            batch_run_id = "260419_153045"
            summary_dir = workflow_api.default_batch_run_output_dir(
                batch_run_id=batch_run_id,
                summary_dir=root / "batch_run",
            )
            configs_out_dir = summary_dir / "configs"
            configs_out_dir.mkdir(parents=True, exist_ok=True)
            config_run_infos = []
            for record in config_records:
                config_out_dir = configs_out_dir / str(record["config_name"])
                config_run_infos.append(
                    workflow_api.run_notebook_config_workflow(
                        config_path=record["config_path"],
                        out_dir=config_out_dir,
                        plot=False,
                    )
                )
            batch_result = workflow_api.write_batch_summary_artifacts(
                config_dir=root,
                summary_dir=summary_dir,
                batch_run_id=batch_run_id,
                config_records=config_records,
                config_run_infos=config_run_infos,
                plot_cases=False,
            )

            self.assertTrue(batch_result["manifest_path"].exists())
            self.assertTrue(batch_result["config_runs_path"].exists())
            self.assertTrue(batch_result["batch_observable_summary_path"].exists())
            self.assertTrue(batch_result["batch_observable_summary_json_path"].exists())
            self.assertTrue(batch_result["batch_failures_path"].exists())
            self.assertTrue((configs_out_dir / "kv_test" / "config_manifest.json").exists())

            manifest = json.loads(batch_result["manifest_path"].read_text())
            self.assertEqual(manifest["n_configs"], 1)
            self.assertEqual(manifest["status_counts"]["ok"], 1)
            self.assertEqual(manifest["status_counts"]["partial"], 0)
            self.assertEqual(manifest["status_counts"]["failed"], 0)
            observables = json.loads(batch_result["batch_observable_summary_json_path"].read_text())
            self.assertEqual(observables["batch_run_id"], "260419_153045")
            self.assertEqual(observables["n_configs"], 1)
            self.assertGreater(len(observables["rows"]), 0)

    def test_write_batch_summary_artifacts_handles_partial_and_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_records = [
                {
                    "config_path": Path(tmpdir) / "kv_test.json",
                    "config_name": "kv_test",
                    "run_id": "kv_test",
                    "mod_dir": MOD_VALIDATE_MOD_DIR,
                    "default_out_dir": Path(tmpdir) / "runs" / "kv_test",
                    "n_templates": 2,
                    "template_paths": [Path(tmpdir) / "templates" / "smoke.json"],
                    "template_names": ["smoke", "vinit"],
                },
                {
                    "config_path": Path(tmpdir) / "bad_config.json",
                    "config_name": "bad_config",
                    "run_id": "bad_config",
                    "mod_dir": MOD_VALIDATE_MOD_DIR,
                    "default_out_dir": Path(tmpdir) / "runs" / "bad_config",
                    "n_templates": 1,
                    "template_paths": [Path(tmpdir) / "templates" / "failed.json"],
                    "template_names": ["failed"],
                },
            ]
            kv_summary_path = Path(tmpdir) / "runs" / "kv_test" / "observable_summary.json"
            kv_summary_path.parent.mkdir(parents=True, exist_ok=True)
            kv_summary_path.write_text(
                json.dumps(
                    {
                        "all_templates": {
                            "observables": {
                                "voltage": {
                                    "n_cases": 5,
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
            bad_summary_path = Path(tmpdir) / "runs" / "bad_config" / "observable_summary.json"
            bad_summary_path.parent.mkdir(parents=True, exist_ok=True)
            bad_summary_path.write_text(
                json.dumps(
                    {
                        "all_templates": {
                            "observables": {}
                        }
                    }
                )
            )

            config_run_infos = [
                {
                    "status": 0,
                    "config_path": Path(tmpdir) / "kv_test.json",
                    "config_name": "kv_test",
                    "out_dir": Path(tmpdir) / "runs" / "kv_test",
                    "manifest_path": Path(tmpdir) / "runs" / "kv_test" / "config_manifest.json",
                    "config_runs_path": Path(tmpdir) / "runs" / "kv_test" / "config_runs.csv",
                    "observable_summary_path": Path(tmpdir) / "runs" / "kv_test" / "observable_summary.csv",
                    "observable_summary_json_path": kv_summary_path,
                    "failures_path": Path(tmpdir) / "runs" / "kv_test" / "failures.csv",
                    "n_templates": 2,
                    "n_total_cases": 6,
                    "n_success_cases": 5,
                    "n_failed_cases": 1,
                    "status_counts": {"ok": 1, "partial": 1, "failed": 0},
                    "template_runs": [],
                    "observables": [],
                    "failures": [
                        {
                            "template_name": "smoke",
                            "template_path": str(Path(tmpdir) / "runs" / "kv_test" / "templates" / "smoke.json"),
                            "out_dir": str(Path(tmpdir) / "runs" / "kv_test" / "templates" / "smoke"),
                            "n_failed_cases": 1,
                            "error_message": "1 case(s) failed inside the sweep.",
                        }
                    ],
                },
                {
                    "status": 1,
                    "config_path": Path(tmpdir) / "bad_config.json",
                    "config_name": "bad_config",
                    "out_dir": Path(tmpdir) / "runs" / "bad_config",
                    "manifest_path": Path(tmpdir) / "runs" / "bad_config" / "config_manifest.json",
                    "config_runs_path": Path(tmpdir) / "runs" / "bad_config" / "config_runs.csv",
                    "observable_summary_path": Path(tmpdir) / "runs" / "bad_config" / "observable_summary.csv",
                    "observable_summary_json_path": bad_summary_path,
                    "failures_path": Path(tmpdir) / "runs" / "bad_config" / "failures.csv",
                    "n_templates": 1,
                    "n_total_cases": 0,
                    "n_success_cases": 0,
                    "n_failed_cases": 0,
                    "status_counts": {"ok": 0, "partial": 0, "failed": 1},
                    "template_runs": [],
                    "observables": [],
                    "failures": [
                        {
                            "template_name": "failed",
                            "template_path": str(Path(tmpdir) / "runs" / "bad_config" / "templates" / "failed.json"),
                            "out_dir": str(Path(tmpdir) / "runs" / "bad_config" / "templates" / "failed"),
                            "n_failed_cases": "",
                            "error_message": "Config run exited with non-zero status.",
                        }
                    ],
                },
            ]

            batch_result = workflow_api.write_batch_summary_artifacts(
                config_dir=Path(tmpdir),
                summary_dir=Path(tmpdir) / "summary",
                batch_run_id="260419_153045",
                config_records=config_records,
                config_run_infos=config_run_infos,
                plot_cases=False,
            )

            self.assertEqual(len(batch_result["config_rows"]), 2)
            statuses = {row["config_name"]: row["batch_status"] for row in batch_result["config_rows"]}
            self.assertEqual(statuses["kv_test"], "partial")
            self.assertEqual(statuses["bad_config"], "failed")
            self.assertEqual(len(batch_result["observable_rows"]), 1)
            self.assertEqual(len(batch_result["failure_rows"]), 2)
            batch_payload = json.loads(batch_result["batch_observable_summary_json_path"].read_text())
            self.assertEqual(batch_payload["batch_run_id"], "260419_153045")
            self.assertEqual(len(batch_payload["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
