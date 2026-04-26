

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ._helpers import (
    ENGINE_ROOT,
    build_model_config_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


run_module = load_module(ENGINE_ROOT / "run.py", "cable_run_dispatch_test")


class DispatchTest(unittest.TestCase):
    def test_config_only_dispatch_runs_all_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(root / "cable_demo.json", build_model_config_payload(["templates/smoke.json"]))
            with mock.patch.object(run_module, "run_config_file", return_value={"status": 0}) as run_mock:
                with mock.patch("sys.argv", ["run.py", str(config_path), "--expand-only"]):
                    status = run_module.main()

        self.assertEqual(status, 0)
        run_mock.assert_called_once_with(str(config_path), out_dir=None, expand_only=True, plot=None)

    def test_parser_requires_config_path_only(self) -> None:
        parser = run_module.build_parser()
        parsed = parser.parse_args(["config.json", "--no-plot"])
        self.assertEqual(parsed.config_path, "config.json")
        self.assertFalse(parsed.plot)


if __name__ == "__main__":
    unittest.main()
