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

"""``NeuroMlReader`` is a placeholder until NeuroML2 import lands (MORPH-6).

The user documentation calls NeuroML2 import planned and not yet supported.
These tests pin the placeholder so the docs and the code agree; replace them
when the reader is implemented, together with the docs wording.
"""

import unittest

import braincell.io
from braincell.io.neuroml2 import NeuroMlReader


class NeuroMlReaderPlaceholderTest(unittest.TestCase):
    def test_read_says_not_supported_and_names_the_supported_formats(self) -> None:
        # The path is never opened: the reader fails before touching the file.
        with self.assertRaisesRegex(NotImplementedError, r"NeuroML2 import is not supported yet") as cm:
            NeuroMlReader().read("missing.cell.nml")
        message = str(cm.exception)
        self.assertIn("Morphology.from_swc", message)
        self.assertIn("Morphology.from_asc", message)

    def test_docstring_marks_the_reader_as_planned(self) -> None:
        # docs/apis/io.rst renders this docstring on the API page.
        self.assertIs(braincell.io.NeuroMlReader, NeuroMlReader)
        self.assertIn("not yet supported", NeuroMlReader.__doc__)
        self.assertIn("NotImplementedError", NeuroMlReader.read.__doc__)


if __name__ == "__main__":
    unittest.main()
