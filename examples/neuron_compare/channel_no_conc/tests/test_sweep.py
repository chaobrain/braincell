

import json
import os
from pathlib import Path
import tempfile
import unittest

from ._helpers import (
    TEMPLATES_ROOT,
    build_main_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_sweep_test",
)
run_module = load_module(
    TEMPLATES_ROOT / "run.py",
    "channel_no_conc_run_sweep_test",
)


class SweepDriverTest(unittest.TestCase):
    def _write_config_and_template(self, tmpdir: str, *, sweep_axes: dict | None = None) -> tuple[Path, Path]:
        root = Path(tmpdir)
        template_dir = root / "templates"
        template_dir.mkdir()
        template_path = write_json(
            template_dir / "grid.json",
            build_scan_template_payload(
                group_id="grid",
                sweep_axes=sweep_axes or {"simulation.v_init_mV": [-65.0, -50.0]},
            ),
        )
        config_path = write_json(
            root / "kv_test.json",
            build_main_config_payload(["templates/grid.json"]),
        )
        return config_path, template_path

    def test_run_config_expand_only_writes_config_and_expanded_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, template_path = self._write_config_and_template(tmpdir)
            config = experiment_schema.load_sweep_config(config_path, template_path)

            status = run_module.run_sweep_config(
                config,
                out_dir=tmpdir,
                expand_only=True,
            )
            self.assertEqual(status, 0)
            self.assertTrue((run_module.Path(tmpdir) / "normalized_config.json").exists())
            self.assertTrue((run_module.Path(tmpdir) / "expanded_cases.json").exists())
            expanded = json.loads((run_module.Path(tmpdir) / "expanded_cases.json").read_text())
            self.assertEqual(len(expanded), 2)

    def test_run_config_executes_compare_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, template_path = self._write_config_and_template(
                tmpdir,
                sweep_axes={"simulation.temperature_celsius": [25.0]},
            )
            config = experiment_schema.load_sweep_config(config_path, template_path)

            out_dir = Path(tmpdir) / "out"
            status = run_module.run_sweep_config(
                config,
                out_dir=out_dir,
                expand_only=False,
                plot=False,
            )
            self.assertEqual(status, 0)
            self.assertTrue((out_dir / "case_results" / "grid__000.json").exists())
            self.assertTrue((out_dir / "case_metrics.csv").exists())
            self.assertTrue((out_dir / "aggregate.json").exists())
            aggregate = json.loads((out_dir / "aggregate.json").read_text())
            self.assertEqual(aggregate["n_total_cases"], 1)
            self.assertEqual(aggregate["n_failed_cases"], 0)

    def test_run_config_with_plot_writes_case_and_template_summary_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, template_path = self._write_config_and_template(
                tmpdir,
                sweep_axes={"simulation.temperature_celsius": [25.0]},
            )
            config = experiment_schema.load_sweep_config(config_path, template_path)

            out_dir = Path(tmpdir) / "out"
            status = run_module.run_sweep_config(
                config,
                out_dir=out_dir,
                expand_only=False,
                plot=True,
            )
            self.assertEqual(status, 0)
            self.assertTrue((out_dir / "plots" / "grid__000.png").exists())
            self.assertTrue((out_dir / "plots" / "summary_mae.png").exists())
            self.assertTrue((out_dir / "plots" / "summary_rmse.png").exists())
            self.assertTrue((out_dir / "plots" / "summary_max_abs.png").exists())
            self.assertTrue((out_dir / "plots" / "summary_rel_mae_pct.png").exists())
            self.assertTrue((out_dir / "plots" / "observable_metric_boxplots.png").exists())

    def test_run_config_file_runs_all_declared_templates(self) -> None:
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
            result = run_module.run_config_file(
                config_path,
                out_dir=out_dir,
                expand_only=True,
                plot=False,
            )

            self.assertEqual(result["status"], 0)
            self.assertTrue((out_dir / "config_manifest.json").exists())
            self.assertTrue((out_dir / "config_runs.csv").exists())
            self.assertTrue((out_dir / "templates" / "smoke" / "normalized_config.json").exists())
            self.assertTrue((out_dir / "templates" / "vinit" / "expanded_cases.json").exists())
            manifest = json.loads((out_dir / "config_manifest.json").read_text())
            self.assertEqual(manifest["n_templates"], 2)
            self.assertEqual(manifest["status_counts"]["ok"], 2)

    def test_run_config_file_with_plot_writes_config_level_summary_outputs(self) -> None:
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
            legacy_dir = out_dir / "templates" / "smoke" / "notebook_plots"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "legacy.png").write_text("stale", encoding="utf-8")

            result = run_module.run_config_file(
                config_path,
                out_dir=out_dir,
                expand_only=False,
                plot=True,
            )

            self.assertEqual(result["status"], 0)
            self.assertTrue((out_dir / "observable_summary.csv").exists())
            self.assertTrue((out_dir / "observable_summary.json").exists())
            self.assertTrue((out_dir / "all_templates_observable_summary.png").exists())
            self.assertTrue((out_dir / "boxplot_by_template.png").exists())
            self.assertTrue((out_dir / "boxplot_by_observable_family.png").exists())
            self.assertFalse((out_dir / "templates" / "smoke" / "notebook_plots").exists())
            summary_payload = json.loads((out_dir / "observable_summary.json").read_text())
            self.assertIn("all_templates", summary_payload)
            self.assertIn("templates", summary_payload)
            self.assertIn("observables", summary_payload["all_templates"])


if __name__ == "__main__":
    unittest.main()
