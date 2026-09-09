# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Worker isolation, failure reporting and offline CLI tests."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .protocol import get_model
from .results import ComparisonResult, Trace, save_result
from .run import run_case


class RunTest(unittest.TestCase):
    def test_worker_gets_protocol_and_unique_destination(self):
        seen = []
        protocol = replace(get_model("io_zh2019").default_protocol, duration_ms=0.2)

        def worker(command, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text())
            self.assertEqual(request["protocol"]["duration_ms"], 0.2)
            self.assertEqual(command[:3], [sys.executable, "-m", "validation.neuron.cell.run"])
            self.assertTrue((Path(kwargs["cwd"]) / "braincell").is_dir())
            seen.append(request_path.parent)
            save_result(
                ComparisonResult(
                    Trace([0, 0.1, 0.2], [-65, -64, -63]),
                    Trace([0.1, 0.2], [-64, -63]),
                    {"format_version": 1, **request},
                ),
                request_path.parent,
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("validation.neuron.cell.run.subprocess.run", side_effect=worker),
        ):
            for _ in range(2):
                result = run_case("io_zh2019", protocol=protocol, output_dir=tmp)
                np.testing.assert_array_equal(result.braincell.time_ms, [0.1, 0.2])
            self.assertNotEqual(seen[0], seen[1])

    def test_worker_failure_identifies_log(self):
        def fail(command, **kwargs):
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            kwargs["stdout"].write("missing test mechanism\n")
            kwargs["stdout"].flush()
            return subprocess.CompletedProcess(command, 1)

        with tempfile.TemporaryDirectory() as tmp, patch("validation.neuron.cell.run.subprocess.run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "missing test mechanism"):
                run_case("io_zh2019", output_dir=tmp)

    def test_offline_cli_never_imports_simulators(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_result(
                ComparisonResult(
                    Trace([0, 1, 2], [-65, -64, -63]), Trace([1, 2], [-64, -63]), {"format_version": 1, "model": "toy"}
                ),
                tmp,
            )
            code = """
import sys
class BlockSimulators:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'neuron', 'braincell', 'brainstate', 'jax'}:
            raise RuntimeError('Offline path imported ' + fullname)
sys.meta_path.insert(0, BlockSimulators())
from validation.neuron.cell.run import main
main(['--load', sys.argv[1]])
"""
            root = Path(__file__).resolve().parents[3]
            result = subprocess.run(
                [sys.executable, "-c", code, tmp],
                cwd=root,
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "comparison.png").is_file())
            self.assertTrue((Path(tmp) / "metrics.json").is_file())
