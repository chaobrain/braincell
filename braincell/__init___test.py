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

``braincell/__init__.py`` is the whole supported API: every other module
path carries a leading underscore or is reached through one of the domain
packages listed here. A name that drops out of ``__all__`` is a silent
break for ``import *`` users, and a name that stays in it after its module
is deleted is an ``AttributeError`` at import time.
"""

import pathlib
import pickle
import unittest

import braincell
from braincell._testing import ReExportTests

#: Every public domain package. AGENTS.md names these as the ones without a
#: leading underscore, so each must be reachable from a bare ``import
#: braincell`` and each must be advertised in ``__all__``.
_DOMAIN_PACKAGES = (
    "channel",
    "filter",
    "io",
    "ion",
    "mech",
    "morph",
    "network",
    "quad",
    "synapse",
    "vis",
)


class PublicSurfaceTest(ReExportTests, unittest.TestCase):
    """``braincell.__all__`` is the package's contract."""

    package = braincell
    require_sorted_all = True


class DomainPackageTest(unittest.TestCase):
    """Every non-underscored package is public, and uniformly so."""

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
        self.assertEqual(on_disk, sorted(_DOMAIN_PACKAGES))


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
