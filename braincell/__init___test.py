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

"""Guards on ``braincell``'s own public surface.

``braincell/__init__.py`` exports the stable top-level API and domain packages.
Experimental namespaces are imported explicitly and have separate stability
contracts. Both kinds of non-underscored package are registered here so a new
directory cannot silently bypass the guard.
A name that drops out of ``__all__`` is a silent
break for ``import *`` users, and a name that stays in it after its module
is deleted is an ``AttributeError`` at import time.
"""

import importlib
import inspect
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


import braincell
from braincell._testing import (
    ReExportTests,
    UserMaterialImportTests,
    api_reference_entries,
    api_reference_gaps,
    neuron_rate_table_gaps,
    own_docstring,
    public_paths,
    tabled_mechanisms,
)

#: Stable domain packages must be reachable from a bare ``import braincell``
#: and advertised in its ``__all__``.
_DOMAIN_PACKAGES = (
    "channel",
    "filter",
    "io",
    "ion",
    "mech",
    "morph",
    "network",
    "plasticity",
    "quad",
    "reduction",
    "synapse",
    "trainable",
)

#: Explicitly imported namespaces outside the stable top-level export contract.
_EXPERIMENTAL_PACKAGES = ("experimental",)


class PublicSurfaceTest(ReExportTests, unittest.TestCase):
    """``braincell.__all__`` is the package's contract."""

    package = braincell
    require_sorted_all = True


class DomainPackageTest(unittest.TestCase):
    """Register every non-underscored package and check stable domain exports."""

    def test_every_domain_package_resolves_after_a_bare_import(self) -> None:
        # ``braincell.io`` used to fail here: nothing in the import graph
        # reached it, so the attribute simply did not exist, while
        # ``braincell.filter`` and ``braincell.morph`` happened to work as a
        # side effect of an unrelated module importing them.
        unreachable = [name for name in _DOMAIN_PACKAGES if not hasattr(braincell, name)]
        self.assertEqual(unreachable, [], f"public packages not reachable from ``import braincell``: {unreachable}")

    def test_every_domain_package_is_advertised(self) -> None:
        unlisted = [name for name in _DOMAIN_PACKAGES if name not in braincell.__all__]
        self.assertEqual(unlisted, [], f"public packages missing from __all__: {unlisted}")

    def test_the_list_matches_the_directory(self) -> None:
        package_dir = pathlib.Path(braincell.__file__).parent
        on_disk = sorted(
            path.name
            for path in package_dir.iterdir()
            if path.is_dir() and not path.name.startswith(("_", ".")) and (path / "__init__.py").exists()
        )
        self.assertEqual(on_disk, sorted(_DOMAIN_PACKAGES + _EXPERIMENTAL_PACKAGES))


class ExperimentalPackageTest(unittest.TestCase):
    """Experimental namespaces support explicit imports without stable exports."""

    def test_every_experimental_package_can_be_imported_explicitly(self) -> None:
        for name in _EXPERIMENTAL_PACKAGES:
            with self.subTest(package=name):
                module = importlib.import_module(f"braincell.{name}")
                self.assertEqual(module.__name__, f"braincell.{name}")
                self.assertIs(getattr(braincell, name), module)

    def test_experimental_packages_are_not_top_level_exports(self) -> None:
        self.assertTrue(set(_EXPERIMENTAL_PACKAGES).isdisjoint(braincell.__all__))


class PlottingLivesInBraintoolsTest(unittest.TestCase):
    """Plotting moved to ``braintools.visualize.morphology``; ``braincell.vis`` is gone."""

    def test_braincell_vis_no_longer_exists(self) -> None:
        self.assertFalse(hasattr(braincell, "vis"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("braincell.vis")

    def test_import_braincell_loads_no_plotting_code(self) -> None:
        code = (
            "import sys, braincell; "
            "print(sorted(m for m in sys.modules if m.startswith('braintools.visualize.morphology')))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")


class RemovedNamesTest(unittest.TestCase):
    """CELL-28: braincell 0.3 removed ``SingleCompartment`` and the ``MultiCompartment`` alias of ``Cell``."""

    #: Each removed name and the replacements its message must name.
    _REPLACEMENTS = {
        "SingleCompartment": ("braincell.Cell.point(", "paint()"),
        "MultiCompartment": ("braincell.Cell",),
    }

    def test_access_names_the_release_and_the_replacement(self) -> None:
        for name, replacements in self._REPLACEMENTS.items():
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as caught:
                    getattr(braincell, name)
                message = str(caught.exception)
                self.assertTrue(message.startswith(f"braincell.{name} was removed in braincell 0.3."), message)
                for replacement in replacements:
                    self.assertIn(replacement, message)
                self.assertFalse(hasattr(braincell, name))
                self.assertNotIn(name, braincell.__all__)

    def test_from_import_raises_import_error_with_the_message(self) -> None:
        with self.assertRaisesRegex(ImportError, r"^braincell\.SingleCompartment was removed in braincell 0\.3\."):
            from braincell import SingleCompartment  # noqa: F401
        with self.assertRaisesRegex(ImportError, r"^braincell\.MultiCompartment was removed in braincell 0\.3\."):
            from braincell import MultiCompartment  # noqa: F401

    def test_star_import_and_other_names_are_unaffected(self) -> None:
        namespace = {}
        exec("from braincell import *", namespace)
        self.assertIs(namespace["Cell"], braincell.Cell)
        self.assertTrue(set(self._REPLACEMENTS).isdisjoint(namespace))
        with self.assertRaisesRegex(AttributeError, "module 'braincell' has no attribute 'NoSuchName'"):
            braincell.NoSuchName


class UserMaterialImportsTest(UserMaterialImportTests, unittest.TestCase):
    """Examples, the user docs and the README import braincell by public paths only (API-2b).

    A reader copies these imports. A public path is ``braincell``, a name in
    ``braincell.__all__``, a stable package or a name in its ``__all__``, or
    anything under ``braincell.experimental``. A submodule path such as
    ``braincell.morph.morphology`` is internal even without an underscore.
    """

    repo = pathlib.Path(braincell.__file__).resolve().parents[1]
    allowlist = (
        (
            "examples/quad/quad.ipynb",
            "braincell.quad._util.apply_standard_solver_step",
            "the custom-integrator tutorial calls the solver-step helper; kept until API-3 classifies quad._util",
        ),
        (
            "examples/quad/quad.ipynb",
            "braincell.quad._util.split_diffeq_states",
            "the same tutorial splits the states by hand; kept until API-3 classifies quad._util",
        ),
    )
    unparsable = (
        ("examples/convert_mod/nmodl/templates/density_channel.py", "Jinja template; it is Python only once rendered"),
        ("examples/convert_mod/nmodl/templates/one_ion_hh_ohmic.py", "Jinja template; it is Python only once rendered"),
    )


class NeuronRateTableTest(unittest.TestCase):
    """NEURON comparisons switch off rate tables (``usetable_<mech> = 0``).

    NEURON interpolates the rates of its built-in ``hh`` and of every MOD
    mechanism with a ``TABLE``; BrainCell computes them exactly, so a
    comparison that keeps the table reports a gap that is not BrainCell's
    (0.165 mV in the NetStim comparison, 5e-3 in a channel I-V curve). Every
    ``.py`` file and notebook under ``validation/neuron`` that inserts such a
    mechanism, or a mechanism whose name is not a literal, sets
    ``usetable_<mech>`` or calls ``disable_rate_table``. Skipped without
    ``validation/neuron`` beside the package, as for an installed package.
    """

    repo = pathlib.Path(braincell.__file__).resolve().parents[1]
    allowlist = (
        (
            "validation/neuron/_clamp_test.py",
            "hh only hosts checks of NEURON's own clamp and Nernst potential; nothing is compared with BrainCell",
        ),
    )

    @classmethod
    def setUpClass(cls) -> None:
        root = cls.repo / "validation" / "neuron"
        if not root.is_dir():
            raise unittest.SkipTest("no validation/neuron beside the package; nothing to scan")
        cls.tabled, cls.mod_files = tabled_mechanisms(cls.repo / "data")
        cls.scanned, cls.matched, cls.gaps, cls.unparsable = neuron_rate_table_gaps(root, cls.tabled)

    def test_comparisons_switch_off_rate_tables(self) -> None:
        counts = (
            f"{len(self.scanned)} files scanned, {len(self.matched)} insert a tabled or non-literal mechanism; "
            f"tables from {self.mod_files} MOD files: {sorted(self.tabled)}"
        )
        # Floors, so that no gap cannot mean nothing checked (2026-11-12: 217 files, 5 matched, 122 MOD files).
        self.assertGreaterEqual(len(self.scanned), 150, counts)
        self.assertGreaterEqual(len(self.matched), 3, counts)
        self.assertGreaterEqual(self.mod_files, 100, counts)
        exempt = {file for file, _ in self.allowlist}
        gaps = {file: items for file, items in self.gaps.items() if file not in exempt}
        self.assertEqual(gaps, {}, f"set usetable_<mech> = 0 (and restore it) where these insert it; {counts}")

    def test_every_scanned_file_parses(self) -> None:
        self.assertEqual(self.unparsable, [], "these files hold code that does not parse, so they cannot be checked")

    def test_allowlist_entries_are_live_and_have_reasons(self) -> None:
        self.assertEqual([file for file, _ in self.allowlist if file not in self.gaps], [], "dead allowlist entries")
        self.assertTrue(all(reason.strip() for _, reason in self.allowlist))


class ModuleAttributeTest(unittest.TestCase):
    """A class's ``__module__`` must name a path the class is reachable at."""

    def test_internal_mixins_report_their_real_module(self) -> None:
        # Both used to claim ``braincell``, where neither is exported, so
        # ``pickle.dumps`` raised "attribute lookup Container on braincell
        # failed" and ``help()`` pointed at a path that does not resolve.
        from braincell._misc import Container, TreeNode

        for cls in (Container, TreeNode):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "braincell._misc")
                self.assertIs(getattr(braincell._misc, cls.__name__), cls)

    def test_every_exported_class_is_picklable_by_reference(self) -> None:
        unpicklable = []
        for name in braincell.__all__:
            obj = getattr(braincell, name)
            if not isinstance(obj, type):
                continue
            try:
                pickle.dumps(obj)
            except Exception as exc:  # noqa: BLE001 - the message is the report
                unpicklable.append(f"{name}: {type(exc).__name__}: {exc}")
        self.assertEqual(unpicklable, [], f"__module__ does not resolve for: {unpicklable}")


if __name__ == "__main__":
    unittest.main()


#: The API reference sources, beside the package in a checkout.
_API_REFERENCE = pathlib.Path(braincell.__file__).resolve().parents[1] / "docs" / "apis"


def _without_own_docstring(public: dict) -> list[str]:
    """Return the first public path of each class or callable that has no docstring of its own."""
    seen, undocumented = set(), []
    for path, obj in public.items():
        if id(obj) in seen or isinstance(obj, types.ModuleType) or not (inspect.isclass(obj) or callable(obj)):
            continue
        seen.add(id(obj))
        if own_docstring(obj) is None:
            undocumented.append(path)
    return undocumented


class ApiReferenceCoverageTest(unittest.TestCase):
    """Every public object has exactly one API reference entry, and a docstring of its own (API-2a).

    The API reference (``docs/apis``) is where a public object's contract is
    written. An object exported under two paths still gets one entry, and the
    page text names the other path. Two entries for one class under its own
    name fail ``sphinx -W`` (both register the same canonical name); an alias
    such as ``filter.BranchPoints`` would get a second page silently.
    """

    def setUp(self) -> None:
        if not _API_REFERENCE.is_dir():
            self.skipTest("no docs/apis beside the package")

    def test_every_public_object_has_exactly_one_entry(self) -> None:
        missing, duplicated, not_public, unresolved = api_reference_gaps(
            api_reference_entries(_API_REFERENCE), public_paths()
        )
        self.assertEqual(missing, [], "add an entry for each to a page in docs/apis")
        self.assertEqual(duplicated, [], "one object, one entry: keep one and mention the other path in the text")
        self.assertEqual(not_public, [], "these entries document objects outside every public __all__")
        self.assertEqual(unresolved, [], "these entries name nothing")

    def test_every_public_class_and_callable_has_its_own_docstring(self) -> None:
        self.assertEqual(_without_own_docstring(public_paths()), [])

    def test_a_class_without_a_docstring_is_reported(self) -> None:
        with mock.patch.object(braincell.filter.LocsetExpr, "__doc__", None):
            self.assertEqual(_without_own_docstring(public_paths()), ["braincell.filter.LocsetExpr"])


class CollectionKeepsTheSessionPrecisionTest(unittest.TestCase):
    """Importing a test module does not change the precision other tests run at.

    ``brainstate.environ.set(precision=64)`` is process-wide. The NEURON
    comparison suites under ``validation/neuron`` call it when imported, so
    collecting one of them together with braincell's tests ran every test of
    the session in float64. The root ``conftest.py`` restores the session's
    precision once collection ends.
    """

    def test_a_module_that_sets_float64_on_import_leaves_the_others_at_float32(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "sets_float64_on_import_test.py").write_text(
                "import brainstate\nbrainstate.environ.set(precision=64)\n\ndef test_imported():\n    pass\n"
            )
            pathlib.Path(tmp, "session_precision_test.py").write_text(
                "import jax.numpy as jnp\n\ndef test_runs_in_float32():\n    assert jnp.ones(1).dtype == jnp.float32\n"
            )
            env = {key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"}
            env.update(
                JAX_PLATFORMS="cpu", PYTHONPATH=os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
            )
            # ``-p conftest`` loads the root conftest.py, which the temporary
            # directory is outside of.
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "conftest"]
                + ["sets_float64_on_import_test.py", "session_precision_test.py"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("2 passed", result.stdout)
