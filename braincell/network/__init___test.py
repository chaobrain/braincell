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

"""Package-scope guards for the :mod:`braincell.network` import graph.

:mod:`braincell.network.event` and :mod:`braincell.network.recording` are
leaves that lower layers import directly --
:mod:`braincell._multi_compartment.cell` and
:mod:`braincell._multi_compartment.run` at module scope, and
:mod:`braincell._compute.layouts` for ``NetStim``. Python executes a
package's ``__init__`` before any submodule, so if
``braincell/network/__init__.py`` eagerly imported ``.connection``,
``.engine``, or ``.lowering``, that would re-enter
``braincell._multi_compartment`` in the middle of its own initialization and
``import braincell`` would die with an ``ImportError``.

``test_importing_braincell_succeeds`` is the empirical guard -- it fails
outright if any of those three cycles returns. The AST checks that follow
turn the same invariant into a targeted, readable failure so a future editor
learns *why* rather than staring at a 30-frame traceback.
"""

import ast
import pathlib
import unittest

import braincell
import braincell.network

_NETWORK_DIR = pathlib.Path(braincell.network.__file__).parent
_MECH_DIR = pathlib.Path(braincell.mech.__file__).parent

#: Submodules whose import pulls in ``braincell._multi_compartment``.
_HEAVY_SUBMODULES = {".connection", ".engine", ".lowering"}


def _module_level_statements(path: pathlib.Path):
    """Yield top-level statements, descending into non-TYPE_CHECKING blocks."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        yield node


class LazyInitTest(unittest.TestCase):
    def test_importing_braincell_succeeds(self) -> None:
        """The empirical guard: all three known cycles are import-time fatal."""
        self.assertTrue(hasattr(braincell, "Network"))

    def test_init_does_not_eagerly_import_the_heavy_submodules(self) -> None:
        offenders = []
        for node in _module_level_statements(_NETWORK_DIR / "__init__.py"):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                name = f".{node.module}" if node.module else "."
                if name in _HEAVY_SUBMODULES:
                    offenders.append(f"line {node.lineno}: from {name} import ...")
        self.assertEqual(
            offenders,
            [],
            "braincell/network/__init__.py must not import .connection/.engine/.lowering "
            "at module scope -- doing so re-enters braincell._multi_compartment mid-import. "
            "Add the name to _LAZY_ATTRS instead.",
        )

    def test_every_heavy_name_resolves_through_getattr(self) -> None:
        self.assertIs(braincell.network.Network, braincell.network.engine.Network)
        self.assertIs(
            braincell.network.NetworkConnections,
            braincell.network.connection.NetworkConnections,
        )
        self.assertIs(
            braincell.network.ConnectionBlock,
            braincell.network.lowering.ConnectionBlock,
        )

    def test_every_exported_name_is_reachable(self) -> None:
        for name in braincell.network.__all__:
            self.assertTrue(hasattr(braincell.network, name), f"{name} is exported but unreachable")

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with self.assertRaisesRegex(AttributeError, "has no attribute 'NotAThing'"):
            braincell.network.NotAThing

    def test_dir_lists_the_lazy_names(self) -> None:
        listed = dir(braincell.network)
        for name in ("ConnectionBlock", "Network", "NetworkConnections"):
            self.assertIn(name, listed)


class MechIsALeafTest(unittest.TestCase):
    """``braincell.mech`` owns the declaration contracts, so it may not
    import any other ``braincell`` package -- that is what lets
    ``_base_channel``, ``_compute.state`` and ``network.connection`` all
    depend on it without a cycle."""

    def test_no_mech_module_imports_another_braincell_package(self) -> None:
        offenders = []
        for path in sorted(_MECH_DIR.rglob("*.py")):
            if path.name.endswith("_test.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "braincell" or alias.name.startswith("braincell."):
                            if not alias.name.startswith("braincell.mech"):
                                offenders.append(f"{path.name}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module and node.module.startswith("braincell"):
                        if not node.module.startswith("braincell.mech"):
                            offenders.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
                    elif node.level >= 2:
                        offenders.append(f"{path.name}:{node.lineno}: relative import escapes braincell.mech")
        self.assertEqual(
            offenders,
            [],
            "braincell.mech must stay a leaf; move the shared declaration into mech instead.",
        )


if __name__ == "__main__":
    unittest.main()
