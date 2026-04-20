from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ._helpers import (
    TEMPLATES_ROOT,
    build_model_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


run_module = load_module(TEMPLATES_ROOT / "run.py", "cable_run_dispatch_test")


class DispatchTest(unittest.TestCase):
    def test_sweep_dispatch_uses_config_and_template_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "scan_templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "cable_demo.json",
                build_model_config_payload(["scan_templates/smoke.json"]),
            )
            with mock.patch.object(run_module, "load_sweep_config", return_value="CONFIG") as load_mock, mock.patch.object(
                run_module,
                "run_sweep_config",
                return_value=0,
            ) as run_mock:
                with mock.patch("sys.argv", ["run.py", str(config_path), str(template_path), "--expand-only"]):
                    status = run_module.main()

        self.assertEqual(status, 0)
        load_mock.assert_called_once_with(str(config_path), str(template_path))
        run_mock.assert_called_once()

    def test_parser_requires_config_and_template_paths(self) -> None:
        parser = run_module.build_parser()
        parsed = parser.parse_args(["config.json", "template.json", "--no-plot"])
        self.assertEqual(parsed.config_path, "config.json")
        self.assertEqual(parsed.template_path, "template.json")
        self.assertFalse(parsed.plot)


if __name__ == "__main__":
    unittest.main()
