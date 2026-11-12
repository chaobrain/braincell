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


import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from braincell._testing import (
    UnitSafetyGuardTests,
    UserMaterialImportTests,
    api_reference_entries,
    api_reference_gaps,
    has_citation,
    is_public_import,
    neuron_rate_table_gaps,
    public_paths,
    tabled_mechanisms,
    unit_safety_hits,
    unreferenced_footnotes,
    user_material_imports,
)

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


def _notebook(*cells):
    """Return the JSON text of a notebook whose cells are ``(cell_type, source)`` pairs."""
    return json.dumps({"cells": [{"cell_type": kind, "source": source} for kind, source in cells], "nbformat": 4})


#: A small repository of user material: one file per case the import guard handles.
_USER_MATERIAL = {
    "README.md": "Intro.\n\n```python\nimport braincell as bc\n```\n",
    "examples/public.py": (
        "import braincell\n"
        "import braincell.quad as bq\n"
        "from braincell import Cell\n"
        "from braincell.filter import *\n"
        "from braincell.experimental.optim import RolloutGradientEngine\n"
    ),
    "examples/private.py": "from braincell.morph._spatial import interpolate_branch\n",
    "examples/submodule.py": "from braincell.morph.morphology import Morphology\n",
    "examples/broken.py": "def (:\n",
    "examples/tutorial.ipynb": _notebook(
        ("code", ["%matplotlib inline\n", "import braincell.quad._util\n"]),
        ("markdown", "`import braincell._not_code` is prose, not code"),
        ("code", "%%bash\necho not python\n"),
        ("code", "!pip install braincell\nfrom braincell import Morphology\n"),
    ),
    "docs/page.md": ("```python\nfrom braincell.filter import at\n```\n\n```bash\nimport braincell._not_python\n```\n"),
    "docs/page.rst": (
        "Title\n=====\n\n.. code-block:: python\n   :caption: A cell\n\n   from braincell.mech import _point\n\nText.\n"
    ),
    "docs/developer/guide.md": "```python\nfrom braincell._deprecation import Deprecation\n```\n",
    "examples/data.txt": "import braincell._ignored\n",
}


def _write_user_material(root: Path) -> None:
    for name, text in _USER_MATERIAL.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class PublicPathsTest(unittest.TestCase):
    """``public_paths`` lists the root and the stable packages, not experimental."""

    def test_paths_cover_the_root_and_each_stable_package(self):
        import braincell

        paths = public_paths()
        self.assertIs(paths["braincell.Cell"], braincell.Cell)
        self.assertIs(paths["braincell.quad"], braincell.quad)
        self.assertIs(paths["braincell.filter.at"], braincell.filter.at)
        packages = [
            getattr(braincell, name) for name in braincell.__all__ if isinstance(getattr(braincell, name), ModuleType)
        ]
        self.assertEqual(len(packages), 12)
        self.assertEqual(len(paths), len(braincell.__all__) + sum(len(package.__all__) for package in packages))
        self.assertFalse(any(path.startswith("braincell.experimental") for path in paths))
        self.assertFalse(any(path.startswith("braincell.filter.metric.") for path in paths))

    def test_is_public_import(self):
        paths = public_paths()
        for path in ("braincell", "braincell.quad", "braincell.filter.at", "braincell.experimental.optim.X"):
            with self.subTest(path=path):
                self.assertTrue(is_public_import(path, paths))
        for path in ("braincell.morph.morphology.Morphology", "braincell.quad._util", "braincell.NoSuchName"):
            with self.subTest(path=path):
                self.assertFalse(is_public_import(path, paths))


class UserMaterialScanTest(unittest.TestCase):
    """``user_material_imports`` reads code, not prose, and skips contributor docs."""

    def test_scan_of_a_small_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_user_material(Path(tmp))
            imports, unparsable = user_material_imports(Path(tmp))
        self.assertEqual(
            sorted(imports),
            [
                ("README.md", "braincell"),
                ("docs/page.md", "braincell.filter.at"),
                ("docs/page.rst", "braincell.mech._point"),
                ("examples/private.py", "braincell.morph._spatial.interpolate_branch"),
                ("examples/public.py", "braincell"),
                ("examples/public.py", "braincell.Cell"),
                ("examples/public.py", "braincell.experimental.optim.RolloutGradientEngine"),
                ("examples/public.py", "braincell.filter"),
                ("examples/public.py", "braincell.quad"),
                ("examples/submodule.py", "braincell.morph.morphology.Morphology"),
                ("examples/tutorial.ipynb", "braincell.Morphology"),
                ("examples/tutorial.ipynb", "braincell.quad._util"),
            ],
        )
        self.assertEqual(unparsable, ["examples/broken.py"])


class UserMaterialImportMixinTest(unittest.TestCase):
    """The mixin fails on a private import, a submodule path, an unparsable file and a bad entry."""

    _ALLOW = (
        ("examples/private.py", "braincell.morph._spatial.interpolate_branch", "reason."),
        ("examples/submodule.py", "braincell.morph.morphology.Morphology", "reason."),
        ("examples/tutorial.ipynb", "braincell.quad._util", "reason."),
        ("docs/page.rst", "braincell.mech._point", "reason."),
    )
    _BROKEN = (("examples/broken.py", "reason."),)

    def _failures(self, allowlist, unparsable, *, write=True):
        with tempfile.TemporaryDirectory() as tmp:
            if write:
                _write_user_material(Path(tmp))
            attrs = {"repo": Path(tmp), "allowlist": allowlist, "unparsable": unparsable}
            case = type("Guard", (UserMaterialImportTests, unittest.TestCase), attrs)
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(case).run(result)
        failed = {test.id().rsplit(".", 1)[-1]: message for test, message in result.failures + result.errors}
        return failed, len(result.skipped)

    def test_fully_allowlisted_material_passes(self):
        self.assertEqual(self._failures(self._ALLOW, self._BROKEN), ({}, 0))

    def test_non_public_imports_are_reported_with_file_and_path(self):
        failed, _ = self._failures((), self._BROKEN)
        self.assertEqual(list(failed), ["test_user_material_imports_only_public_paths"])
        message = failed["test_user_material_imports_only_public_paths"]
        for line in (
            "examples/private.py: braincell.morph._spatial.interpolate_branch",
            "examples/submodule.py: braincell.morph.morphology.Morphology",
            "examples/tutorial.ipynb: braincell.quad._util",
            "docs/page.rst: braincell.mech._point",
        ):
            self.assertIn(line, message)
        self.assertNotIn("public.py", message)

    def test_unparsable_file_is_reported_by_name(self):
        failed, _ = self._failures(self._ALLOW, ())
        self.assertEqual(list(failed), ["test_user_material_parses"])
        self.assertIn("examples/broken.py", failed["test_user_material_parses"])

    def test_dead_entries_and_missing_reasons_fail(self):
        dead = (("examples/gone.py", "braincell._gone", "reason."),)
        failed, _ = self._failures(self._ALLOW + dead, self._BROKEN + (("examples/public.py", "reason."),))
        self.assertEqual(list(failed), ["test_user_material_allowlists_have_no_dead_entries"])
        self.assertIn("examples/gone.py", failed["test_user_material_allowlists_have_no_dead_entries"])
        self.assertIn("examples/public.py", failed["test_user_material_allowlists_have_no_dead_entries"])
        failed, _ = self._failures(self._ALLOW[:-1] + (self._ALLOW[-1][:2] + (" ",),), self._BROKEN)
        self.assertEqual(list(failed), ["test_user_material_allowlist_entries_have_reasons"])

    def test_repository_without_examples_is_skipped(self):
        # The reasons check reads only the entries, so it runs; the three scans skip.
        self.assertEqual(self._failures((), (), write=False), ({}, 3))


#: An API reference page using every directive ``api_reference_entries`` reads.
_API_PAGE = """``braincell`` module
====================

.. currentmodule:: braincell
.. automodule:: braincell

Cells
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Cell
    CVPerBranchList

Text after the list ends it.

.. autofunction:: mix_ions
.. autodata:: __version__
.. py:data:: __version_info__

   Written by hand.

.. function:: connect(source, synapse)

.. currentmodule:: braincell.filter

.. autoclass:: braincell.Morphology
.. autofunction:: at
"""


class ApiReferenceEntriesTest(unittest.TestCase):
    """``api_reference_entries`` reads the directives and resolves names against ``currentmodule``."""

    def test_entries_of_one_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "braincell.rst").write_text(_API_PAGE)
            Path(tmp, "notes.txt").write_text(".. autofunction:: ignored\n")
            entries = api_reference_entries(Path(tmp))
        self.assertEqual(
            entries,
            [
                ("braincell", "braincell.rst"),
                ("braincell.Cell", "braincell.rst"),
                ("braincell.CVPerBranchList", "braincell.rst"),
                ("braincell.mix_ions", "braincell.rst"),
                ("braincell.__version__", "braincell.rst"),
                ("braincell.__version_info__", "braincell.rst"),
                ("braincell.connect", "braincell.rst"),
                ("braincell.Morphology", "braincell.rst"),
                ("braincell.filter.at", "braincell.rst"),
            ],
        )


class ApiReferenceGapsTest(unittest.TestCase):
    """``api_reference_gaps`` finds missing, duplicated, non-public and unresolvable entries."""

    def setUp(self):
        import braincell

        self.braincell = braincell
        self.public = {
            "braincell.Cell": braincell.Cell,
            "braincell.CellAlias": braincell.Cell,
            "braincell.CVPerBranchList": braincell.CVPerBranchList,
            "braincell.mix_ions": braincell.mix_ions,
            "braincell.filter": braincell.filter,
        }

    def test_complete_reference_has_no_gaps(self):
        entries = [("braincell.Cell", "a.rst"), ("braincell.CVPerBranchList", "a.rst"), ("braincell.mix_ions", "b.rst")]
        self.assertEqual(api_reference_gaps(entries, self.public), ([], [], [], []))

    def test_deleted_entry_is_missing(self):
        entries = [("braincell.Cell", "a.rst"), ("braincell.mix_ions", "b.rst")]
        self.assertEqual(api_reference_gaps(entries, self.public), (["braincell.CVPerBranchList"], [], [], []))
        missing, *_ = api_reference_gaps(entries, public_paths())
        self.assertIn("braincell.CVPerBranchList", missing)

    def test_alias_with_a_second_entry_is_duplicated(self):
        entries = [
            ("braincell.Cell", "a.rst"),
            ("braincell.CellAlias", "b.rst"),
            ("braincell.CVPerBranchList", "a.rst"),
            ("braincell.mix_ions", "b.rst"),
        ]
        duplicated = [(("braincell.Cell", "a.rst"), ("braincell.CellAlias", "b.rst"))]
        # A hypothetical second path to Cell.
        with mock.patch.object(self.braincell, "CellAlias", self.braincell.Cell, create=True):
            self.assertEqual(api_reference_gaps(entries, self.public), ([], duplicated, [], []))

    def test_entry_for_an_internal_object_is_not_public(self):
        entries = [
            ("braincell.Cell", "a.rst"),
            ("braincell.CVPerBranchList", "a.rst"),
            ("braincell.mix_ions", "b.rst"),
            ("braincell.morph._spatial.MorphologySpatialGeometry", "morphology.rst"),
            ("braincell.morph._spatial", "morphology.rst"),
        ]
        not_public = [("braincell.morph._spatial.MorphologySpatialGeometry", "morphology.rst")]
        self.assertEqual(api_reference_gaps(entries, self.public), ([], [], not_public, []))

    def test_unresolvable_entry_names_its_file(self):
        entries = [
            ("braincell.Cell", "a.rst"),
            ("braincell.CVPerBranchList", "a.rst"),
            ("braincell.mix_ions", "b.rst"),
            ("braincell.NoSuchThing", "b.rst"),
            ("braincell_no_such_package.X", "b.rst"),
        ]
        unresolved = [("braincell.NoSuchThing", "b.rst"), ("braincell_no_such_package.X", "b.rst")]
        self.assertEqual(api_reference_gaps(entries, self.public), ([], [], [], unresolved))


def _notebook(*cells: str) -> str:
    return json.dumps({"cells": [{"cell_type": "code", "source": cell} for cell in cells]})


class TabledMechanismsTest(unittest.TestCase):
    """``tabled_mechanisms`` reads TABLE statements outside comments and names the mechanism."""

    def test_mod_files_with_a_table(self):
        sources = {
            "channel/a.mod": "NEURON { SUFFIX a }\nPROCEDURE rates(v) {\n  TABLE minf FROM -100 TO 100 WITH 200\n}\n",
            "channel/b.mod": "NEURON { SUFFIX b }\n: TABLE minf FROM -100 TO 100 WITH 200\n",
            "c.mod": "COMMENT\nTABLE minf\nENDCOMMENT\nNEURON { SUFFIX c }\n",
            "synapse/d.mod": (
                "NEURON {\n  POINT_PROCESS D\n}\n"
                "PROCEDURE rates(v) { TABLE gate DEPEND celsius FROM -120 TO 30 WITH 1500 }\n"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in sources.items():
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / name).write_text(text)
            self.assertEqual(tabled_mechanisms(Path(tmp)), (frozenset({"hh", "a", "D"}), 4))


class NeuronRateTableScanTest(unittest.TestCase):
    """``neuron_rate_table_gaps`` finds inserted tabled mechanisms and checks that their table is switched off."""

    files = {
        # The NetStim comparison before it set usetable_hh (0.165 mV of the gap came from the table).
        "pre_fix.ipynb": _notebook(
            "from neuron import h\n",
            'section = h.Section(name="soma")\nsection.insert("hh")\nh.finitialize(-65.0)\n',
        ),
        "fixed.ipynb": _notebook(
            "%matplotlib inline\nsection.insert('hh')\n", "previous = h.usetable_hh\nh.usetable_hh = 0\n"
        ),
        "helper.py": (
            "def add(sec, mechanism):\n    sec.insert(mechanism)\n\nadd(soma, 'hh')\nadd(soma, mechanism='pas')\n"
        ),
        "method.py": (
            "class Cell:\n"
            "    def add(self, sec, name):\n"
            "        sec.insert(name)\n\n"
            "    def build(self, sec):\n"
            "        self.add(sec, 'hh')\n"
            "        setattr(h, 'usetable_hh', 0)\n"
        ),
        "local.py": (
            "def build(sec):\n"
            "    mechanism = 'hh'\n"
            "    sec.insert(mechanism)\n"
            "    setattr(h, f'usetable_{mechanism}', 0)\n"
        ),
        "point.py": "synapse = h.SynWithTable(section(0.5))\n",
        "dynamic.py": "for name in config.mechanisms:\n    sec.insert(name)\n",
        "dynamic_switched.py": "sec.insert(config.name)\ndisable_rate_table(h, config.name)\n",
        "helper_dynamic.py": "def add(sec, mechanism):\n    sec.insert(mechanism)\n\nadd(soma, config.name)\n",
        "plain.py": (
            "import sys\nsys.path.insert(0, 'x')\nmetrics.insert(1, 'hh')\nsec.insert('pas')\nh.IClamp(sec(0.5))\n"
        ),
        "broken.py": "def (:\n",
        "artifacts/generated.py": "sec.insert('hh')\n",
        "notes.md": "sec.insert('hh')\n",
    }

    def _scan(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "validation" / "neuron"
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            return neuron_rate_table_gaps(root, frozenset({"hh", "SynWithTable"}))

    def test_scan_of_a_small_tree(self):
        scanned, matched, gaps, unparsable = self._scan(self.files)
        prefix = "validation/neuron/"
        self.assertEqual(len(scanned), 11)  # artifacts/ and Markdown are not read
        self.assertNotIn(prefix + "artifacts/generated.py", scanned)
        self.assertEqual(
            matched,
            {
                prefix + "dynamic.py": ["sec.insert(name)"],
                prefix + "dynamic_switched.py": ["sec.insert(config.name)"],
                prefix + "fixed.ipynb": ["hh"],
                prefix + "helper.py": ["hh"],
                prefix + "helper_dynamic.py": ["sec.insert(mechanism) in add()"],
                prefix + "local.py": ["hh"],
                prefix + "method.py": ["hh"],
                prefix + "point.py": ["SynWithTable"],
                prefix + "pre_fix.ipynb": ["hh"],
            },
        )
        self.assertEqual(
            gaps,
            {
                prefix + "dynamic.py": ["sec.insert(name)"],
                prefix + "helper.py": ["hh"],
                prefix + "helper_dynamic.py": ["sec.insert(mechanism) in add()"],
                prefix + "point.py": ["SynWithTable"],
                prefix + "pre_fix.ipynb": ["hh"],
            },
        )
        self.assertEqual(unparsable, [prefix + "broken.py"])

    def test_switching_another_mechanism_does_not_count(self):
        _, _, gaps, _ = self._scan({"a.py": "sec.insert('hh')\nh.usetable_kv = 0\n"})
        self.assertEqual(gaps, {"validation/neuron/a.py": ["hh"]})


if __name__ == "__main__":
    unittest.main()
