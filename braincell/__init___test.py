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

import gc
import importlib
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell._testing import ReExportTests, UserMaterialImportTests

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


#: Long-lived host constants, one per direction, like brainunit's
#: ``gas_constant.mantissa`` that ``_nernst`` multiplies by a temperature.
_CONSTANT_FOR_X64_THEN_X32 = np.array(8.31446262)
_CONSTANT_FOR_X32_THEN_X64 = np.array(96485.33212)


def _leave_a_dead_jaxpr_holding(constant) -> None:
    # The jaxpr's literal keeps the constant's canonical form, at the current
    # precision, alive; the cycle leaves it for the cyclic collector. Moving it
    # to the oldest generation first means only a full collection frees it, as
    # for a model a long test built.
    cycle = [jax.make_jaxpr(lambda x: constant * x)(jnp.ones(2))]
    cycle.append(cycle)
    gc.collect()


class PrecisionSwitchIsolationTest(unittest.TestCase):
    """A test runs at its own precision whatever ran before it (OPT-12).

    jaxlib reuses a numpy array's canonical form while it is alive, even
    after ``jax_enable_x64`` changed; a dead jaxpr from an earlier test at the
    other precision then fed a wrong-width buffer to a fresh program
    (``RuntimeProgramInputMismatch``), so failures depended on test order. The
    root ``conftest.py`` collects garbage around precision switches. Each pair
    runs in name order, the first test leaving the garbage and the second
    hitting it; the pairs fail without that ``conftest.py`` hook.
    """

    def test_1_x64_test_leaves_a_float64_literal(self) -> None:
        with brainstate.environ.context(precision=64):
            _leave_a_dead_jaxpr_holding(_CONSTANT_FOR_X64_THEN_X32)

    def test_2_a_following_float32_test_gets_float32(self) -> None:
        # The shape is unique to this test, so the multiply compiles afresh.
        result = _CONSTANT_FOR_X64_THEN_X32 * jnp.ones((1, 7, 3))
        self.assertEqual(result.dtype, jnp.float32)

    def test_3_float32_test_leaves_a_float32_literal(self) -> None:
        _leave_a_dead_jaxpr_holding(_CONSTANT_FOR_X32_THEN_X64)

    def test_4_a_following_x64_test_gets_float64(self) -> None:
        with brainstate.environ.context(precision=64):
            result = _CONSTANT_FOR_X32_THEN_X64 * jnp.ones((1, 7, 4))
            self.assertEqual(result.dtype, jnp.float64)


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
