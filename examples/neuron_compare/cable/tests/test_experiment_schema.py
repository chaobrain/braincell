

from pathlib import Path
import tempfile
import unittest

from ._helpers import (
    TEMPLATES_ROOT,
    build_case_payload,
    build_model_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "cable_experiment_schema_test",
)


class ExperimentSchemaTest(unittest.TestCase):
    def test_load_model_config_resolves_relative_template_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/smoke.json"]),
            )

            model_config = experiment_schema.load_model_config(config_path)

        self.assertEqual(model_config.template_paths, (template_path.resolve(),))
        self.assertIn("morphology", model_config.defaults)
        self.assertIn("path", model_config.defaults["morphology"])
        self.assertNotIn("kind", model_config.defaults["morphology"])

    def test_load_model_config_resolves_relative_morphology_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            morph_dir = root / "morph"
            morph_dir.mkdir()
            morph_path = write_json(morph_dir / "sample.swc", {}).with_suffix(".swc")
            morph_path.write_text("1 1 0 0 0 1 -1\n")
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(
                    ["templates/smoke.json"],
                    defaults={"morphology": {"path": "morph/sample.swc"}},
                ),
            )

            model_config = experiment_schema.load_model_config(config_path)

        self.assertEqual(model_config.defaults["morphology"]["path"], str(morph_path.resolve()))

    def test_load_scan_template_builds_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = write_json(
                Path(tmpdir) / "dc_scan.json",
                build_scan_template_payload(
                    group_id="dc_scan",
                    sweep_axes={"simulation.v_init_mV": [-70.0, -50.0]},
                ),
            )

            template = experiment_schema.load_scan_template(template_path)

        self.assertEqual(template.template_name, "dc_scan")
        self.assertEqual(template.group_id, "dc_scan")
        self.assertEqual(template.raw_sweep_axes["simulation.v_init_mV"], (-70.0, -50.0))

    def test_load_sweep_config_builds_run_id_from_config_and_template_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/smoke.json"]),
            )

            config = experiment_schema.load_sweep_config(config_path, "templates/smoke.json")

        self.assertEqual(config.config_id, "cable_demo__smoke")
        self.assertEqual(config.group.group_id, "smoke")
        self.assertEqual(config.template_path, template_path.resolve())

    def test_load_sweep_config_rejects_template_not_declared_in_main_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "declared.json", build_scan_template_payload())
            undeclared_path = write_json(template_dir / "other.json", build_scan_template_payload())
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/declared.json"]),
            )

            with self.assertRaisesRegex(ValueError, "not declared in config.templates"):
                experiment_schema.load_sweep_config(config_path, undeclared_path)

    def test_expand_cases_cartesian_product_across_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "grid.json",
                build_scan_template_payload(
                    group_id="grid",
                    sweep_axes={
                        "simulation.v_init_mV": [-70.0, -50.0],
                        "cv_policy.cv_per_branch": [1, 3],
                    },
                ),
            )
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/grid.json"]),
            )

            config = experiment_schema.load_sweep_config(config_path, template_path)
            expanded = experiment_schema.expand_cases(config)

        self.assertEqual(len(expanded), 4)
        self.assertEqual(expanded[0]["case_id"], "grid__000")
        self.assertEqual(expanded[-1]["case_id"], "grid__003")

    def test_load_sweep_config_rejects_unsupported_sweep_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "bad.json",
                build_scan_template_payload(
                    sweep_axes={"stimulus.frequency_hz": [100.0]},
                ),
            )
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/bad.json"]),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported sweep path"):
                experiment_schema.load_sweep_config(config_path, template_path)

    def test_load_model_config_rejects_legacy_sweep_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_json(
                Path(tmpdir) / "legacy.json",
                {
                    "config_id": "legacy",
                    "case_groups": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "Legacy cable sweep config schema"):
                experiment_schema.load_model_config(config_path)

    def test_load_scan_template_rejects_legacy_sweep_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = write_json(
                Path(tmpdir) / "legacy_template.json",
                {
                    "config_id": "legacy",
                    "case_groups": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "Legacy cable scan template schema"):
                experiment_schema.load_scan_template(template_path)

    def test_build_sweep_config_merges_config_defaults_with_template_stimulus_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "compat.json",
                build_scan_template_payload(
                    base={
                        "simulation": build_case_payload()["simulation"],
                        "cable": build_case_payload()["cable"],
                        "cv_policy": build_case_payload()["cv_policy"],
                        "stimulus": {
                            "kind": "dc_step",
                            "target": "root_soma_midpoint",
                            "delay_ms": 0.5,
                            "dur_ms": 1.0,
                            "amp_nA": 0.05,
                        }
                    },
                ),
            )
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["templates/compat.json"]),
            )

            config = experiment_schema.load_sweep_config(config_path, template_path)

        self.assertEqual(config.base_case["morphology"]["path"], build_case_payload()["morphology"]["path"])
        self.assertEqual(config.base_case["morphology"]["kind"], "swc")
        self.assertEqual(config.base_case["stimulus"]["kind"], "dc_step")

    def test_load_model_config_requires_defaults_morphology_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "bad_config.json",
                {
                    "meta": {"label": "bad"},
                    "defaults": {
                        "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0},
                    },
                    "templates": ["templates/smoke.json"],
                },
            )

            with self.assertRaisesRegex((KeyError, ValueError), "morphology.path|morphology"):
                experiment_schema.load_model_config(config_path)

    def test_load_model_config_rejects_non_morphology_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "bad_config.json",
                {
                    "meta": {"label": "bad"},
                    "defaults": {
                        "morphology": {"path": build_case_payload()["morphology"]["path"]},
                        "simulation": {"dt_ms": 0.025},
                    },
                    "templates": ["templates/smoke.json"],
                },
            )

            with self.assertRaisesRegex(ValueError, "only supports 'morphology'"):
                experiment_schema.load_model_config(config_path)

    def test_repo_dc_template_loads_as_amp_scan(self) -> None:
        config_path = Path("/home/swl/braincell/examples/neuron_compare/cable/configs/IO.json")
        template_path = Path("/home/swl/braincell/examples/neuron_compare/cable/templates/dc.json")

        config = experiment_schema.load_sweep_config(config_path, template_path)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.group.group_id, "dc")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(
            expanded[0]["morphology"]["path"],
            "/home/swl/braincell/examples/multi_compartment/morpho_files/Cerebellum_morph/IO.swc",
        )


if __name__ == "__main__":
    unittest.main()
