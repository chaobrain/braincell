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


import unittest

from braincell._testing import has_citation, unreferenced_footnotes

_DOC = """Summary.

Uses the model of Smith [1]_.

References
----------
.. [1] Smith, A. (2000). A model. J. Neurosci., 1, 1-2.
"""


class FootnoteTest(unittest.TestCase):
    def test_a_cited_footnote_passes(self):
        self.assertTrue(has_citation(_DOC))
        self.assertEqual(unreferenced_footnotes(_DOC), [])

    def test_an_uncited_footnote_is_reported(self):
        doc = _DOC.replace("Smith [1]_", "Smith") + ".. [2] Jones, B. (2001). Another.\n"
        self.assertTrue(has_citation(doc))
        self.assertEqual(unreferenced_footnotes(doc), ["1", "2"])

    def test_named_labels(self):
        doc = "Text [JE2011]_.\n\nReferences\n----------\n.. [JE2011] J. E. (2011).\n.. [X1] Other.\n"
        self.assertEqual(unreferenced_footnotes(doc), ["X1"])


if __name__ == "__main__":
    unittest.main()
