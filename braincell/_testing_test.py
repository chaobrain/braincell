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


import tempfile
import textwrap
import unittest
from pathlib import Path

from braincell._testing import UnitSafetyGuardTests, has_citation, unit_safety_hits, unreferenced_footnotes

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


def _patterns(source, **kwargs):
    return [(qualname, pattern) for qualname, pattern, _ in unit_safety_hits(textwrap.dedent(source), **kwargs)]


class UnitSafetyHitsTest(unittest.TestCase):
    """Counter-examples: each construct the guard exists for is reported."""

    def test_bare_number_reads_are_reported(self):
        source = """
        class Pool:
            def rate(self, x):
                return u.get_mantissa(self.kCa) + self.kCa.mantissa + u.get_magnitude(x) + x.to_decimal(u.mV)
        X = jax.tree.map(get_magnitude, Y).magnitude
        """
        expected = [("Pool.rate", p) for p in ("get_mantissa", ".mantissa", "get_magnitude")]
        self.assertEqual(_patterns(source), expected + [("<module>", ".magnitude"), ("<module>", "get_magnitude")])

    def test_type_error_fallbacks_are_reported(self):
        source = """
        def snapshot(f, V):
            try:
                return f(V, include_external=True)
            except TypeError:
                return f(V)
            try:
                g()
            except (TypeError, ValueError):
                pass
            try:
                g()
            except Exception:
                return None
            try:
                g()
            except:
                h()
            try:
                g()
            except TypeError as exc:
                if "excluded_paths" not in str(exc):
                    raise
                g()
        """
        self.assertEqual(_patterns(source), [("snapshot", "except TypeError")] * 5)

    def test_handlers_that_re_raise_or_miss_type_error_are_not_reported(self):
        source = """
        def solve():
            try:
                g()
            except Exception as err:
                raise ValueError("could not be solved") from err
            try:
                g()
            except ValueError:
                return None
        """
        self.assertEqual(_patterns(source), [])

    def test_unit_factors_are_reported_only_when_asked(self):
        source = """
        def igate(self, dn):
            ig = (1e12) * self.gbar / self.gunit * (1e6) * e0 * dn
            ig /= 1e-3
            return 0.001 * ig, 1e-6 * ig, -1e12 * ig
        """
        self.assertEqual(_patterns(source), [])
        labels = [pattern for _, pattern in _patterns(source, check_factors=True)]
        self.assertEqual(labels, ["factor 1e6", "factor 1e12", "factor 1e-3", "factor 1e-3", "factor 1e12"])


class UnitSafetyGuardMixinTest(unittest.TestCase):
    """The mixin fails on a planted construct, a dead entry and a missing reason."""

    def _failures(self, allowlist):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pool.py").write_text("def rate(self):\n    return u.get_mantissa(self.kCa)\n")
            Path(tmp, "pool_test.py").write_text("x = u.get_mantissa(y)\n")  # test modules are skipped
            attrs = {"package_dir": Path(tmp), "allowlist": allowlist}
            case = type("Guard", (UnitSafetyGuardTests, unittest.TestCase), attrs)
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(case).run(result)
        return [test.id().rsplit(".", 1)[-1] for test, _ in result.failures + result.errors]

    def test_planted_construct_dead_entry_and_missing_reason_fail(self):
        entry = ("pool.py", "rate", "get_mantissa", "reason.")
        self.assertEqual(self._failures((entry,)), [])
        self.assertEqual(self._failures(()), ["test_no_unlisted_unit_unsafe_construct"])
        dead = ("pool.py", "missing", "get_magnitude", "gone.")
        self.assertEqual(self._failures((entry, dead)), ["test_unit_safety_allowlist_has_no_dead_entries"])
        self.assertEqual(self._failures((entry[:3] + (" ",),)), ["test_unit_safety_allowlist_entries_have_reasons"])


if __name__ == "__main__":
    unittest.main()
