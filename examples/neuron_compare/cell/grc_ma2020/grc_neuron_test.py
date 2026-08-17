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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import grc_neuron
from .parameters import _LOADED_NRNMECH_PATHS


class GrCNeuronMechanismLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        _LOADED_NRNMECH_PATHS.clear()

    def tearDown(self) -> None:
        _LOADED_NRNMECH_PATHS.clear()

    def test_missing_library_reports_compile_command(self) -> None:
        missing = Path(tempfile.gettempdir()) / "missing" / "x86_64" / ".libs" / "libnrnmech.so"

        with self.assertRaisesRegex(FileNotFoundError, "nrnivmodl channel ion synapse"):
            grc_neuron._load_nrnmech_once(missing)

        self.assertNotIn(str(missing.resolve()), _LOADED_NRNMECH_PATHS)

    def test_successful_library_is_loaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libnrnmech.so"
            library.touch()
            fake_h = mock.Mock()
            fake_h.nrn_load_dll.return_value = 1.0

            with mock.patch.object(grc_neuron, "h", fake_h):
                grc_neuron._load_nrnmech_once(library)
                grc_neuron._load_nrnmech_once(library)

            fake_h.nrn_load_dll.assert_called_once_with(str(library.resolve()))
            self.assertIn(str(library.resolve()), _LOADED_NRNMECH_PATHS)

    def test_failed_load_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libnrnmech.so"
            library.touch()
            fake_h = mock.Mock()
            fake_h.nrn_load_dll.return_value = 0.0

            with mock.patch.object(grc_neuron, "h", fake_h):
                with self.assertRaisesRegex(RuntimeError, "failed to load"):
                    grc_neuron._load_nrnmech_once(library)

            self.assertNotIn(str(library.resolve()), _LOADED_NRNMECH_PATHS)


if __name__ == "__main__":
    unittest.main()
