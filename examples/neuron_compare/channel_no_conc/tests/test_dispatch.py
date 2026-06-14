

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from ._helpers import TEMPLATES_ROOT, build_main_config_payload, build_scan_template_payload, load_module, write_json


os.environ.setdefault("JAX_PLATFORMS", "cpu")


run_module = load_module(
    TEMPLATES_ROOT / "run.py",
    "channel_no_conc_dispatch_test",
)


class CompareSingleCompartmentChannelDispatcherTest(unittest.TestCase):
    def test_sweep_dispatch_uses_config_and_template_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json"]),
            )
            with mock.patch.object(run_module, "load_sweep_config", return_value="CONFIG") as load_mock, mock.patch.object(
                run_module,
                "run_sweep_config",
                return_value=0,
            ) as run_mock:
                with mock.patch.object(
                    sys,
                    "argv",
                    ["run.py", str(config_path), str(template_path), "--expand-only"],
                ):
                    status = run_module.main()
        self.assertEqual(status, 0)
        load_mock.assert_called_once_with(str(config_path), str(template_path))
        run_mock.assert_called_once_with("CONFIG", out_dir=None, expand_only=True, plot=None)

    def test_config_only_dispatch_runs_all_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json"]),
            )
            with mock.patch.object(run_module, "run_config_file", return_value={"status": 0}) as run_mock:
                with mock.patch.object(
                    sys,
                    "argv",
                    ["run.py", str(config_path), "--expand-only"],
                ):
                    status = run_module.main()
        self.assertEqual(status, 0)
        run_mock.assert_called_once_with(str(config_path), out_dir=None, expand_only=True, plot=None)

    def test_parser_accepts_optional_template_path(self) -> None:
        parser = run_module.build_parser()
        parsed = parser.parse_args(["config.json", "template.json", "--no-plot"])
        self.assertEqual(parsed.config_path, "config.json")
        self.assertEqual(parsed.template_path, "template.json")
        self.assertFalse(parsed.plot)
        parsed = parser.parse_args(["config.json"])
        self.assertEqual(parsed.config_path, "config.json")
        self.assertIsNone(parsed.template_path)


if __name__ == "__main__":
    unittest.main()
