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

This package is imported from the middle of
:mod:`braincell._multi_compartment`'s initialization --
:mod:`braincell._multi_compartment.cell` imports
:mod:`braincell.network.event` and :mod:`braincell.network.recording` at
module scope, and :mod:`braincell._compute.layouts` imports ``NetStim``.
Python runs ``braincell/network/__init__.py`` before any of those
submodules, so it executes while ``braincell._multi_compartment`` is in
:data:`sys.modules` but only partially built.

``test_importing_braincell_succeeds`` is the empirical guard. The two AST
checks that follow turn the same invariant into a targeted failure so a
future editor learns *why* rather than staring at a 30-frame traceback:

- ``PartialParentTest`` checks the property the eager ``__init__`` actually
  depends on -- that no module here imports a *name* from the
  ``braincell._multi_compartment`` package root, only from its submodules.
  A submodule resolves against a partial parent; a name does not.
- ``ImportGraphTest`` pins the intra-package edges so any new sibling
  dependency surfaces in review, in the same idiom as
  ``braincell/_compute/__init___test.py``.
"""

import ast
import pathlib
import unittest

import braincell
import braincell.network

_NETWORK_DIR = pathlib.Path(braincell.network.__file__).parent
_MECH_DIR = pathlib.Path(braincell.mech.__file__).parent

#: Sibling modules each ``braincell.network`` module imports at load time.
#:
#: Layer order is ``{core, event, recording}`` -> ``{lowering, pairing}``
#: -> ``{connection, delivery}`` -> ``engine`` -> ``__init__``. Every edge
#: below respects it, which is what keeps the graph acyclic.
_EXPECTED_GRAPH = {
    "__init__": {"connection", "core", "engine"},
    "_testing": set(),
    # ``connection`` reads ``core`` to recognise a ``Population`` handed to
    # ``connect()``; ``core`` is a leaf, so the edge keeps the graph a DAG.
    "connection": {"core", "event", "pairing", "recording"},
    "core": set(),
    "delivery": {"lowering"},
    "engine": {"connection", "core", "delivery", "lowering", "recording"},
    "event": set(),
    "lowering": {"core", "event"},
    "pairing": {"event"},
    "recording": set(),
}


def _module_files():
    """Yield ``(module_name, path)`` for every non-test module here."""
    for path in sorted(_NETWORK_DIR.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        yield path.stem, path


def _load_time_nodes(path: pathlib.Path):
    """Yield import nodes that run at module load.

    Skips ``if TYPE_CHECKING:`` bodies and function bodies -- a deferred
    import inside a function cannot participate in an import cycle.
    """

    def walk(body):
        for node in body:
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node
            elif isinstance(node, ast.ClassDef):
                yield from walk(node.body)
            elif isinstance(node, ast.If):
                yield from walk(node.body)
                yield from walk(node.orelse)
            elif isinstance(node, ast.Try):
                yield from walk(node.body)
                yield from walk(node.orelse)
                yield from walk(node.finalbody)
                for handler in node.handlers:
                    yield from walk(handler.body)

    yield from walk(ast.parse(path.read_text(encoding="utf-8")).body)


class ImportSucceedsTest(unittest.TestCase):
    def test_importing_braincell_succeeds(self) -> None:
        """The empirical guard: an import cycle here is import-time fatal."""
        self.assertTrue(hasattr(braincell, "Network"))

    def test_every_exported_name_is_reachable(self) -> None:
        for name in braincell.network.__all__:
            self.assertTrue(hasattr(braincell.network, name), f"{name} is exported but unreachable")


class PartialParentTest(unittest.TestCase):
    """No module here may import a name from a partially built parent.

    ``braincell/network/__init__.py`` runs while
    ``braincell._multi_compartment`` is mid-initialization.
    ``from braincell._multi_compartment.synapses import SynapseView`` is
    fine -- Python imports the submodule against the partial parent. But
    ``from braincell._multi_compartment import Cell`` would read an
    attribute the parent has not bound yet and raise ``ImportError``.
    """

    def test_no_name_is_imported_from_the_multi_compartment_package_root(self) -> None:
        offenders = []
        for name, path in _module_files():
            for node in _load_time_nodes(path):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                if node.module != "braincell._multi_compartment":
                    continue
                for alias in node.names:
                    # ``from pkg import submodule`` is a submodule import and
                    # is safe; a capitalised name is a class attribute and is
                    # not. Resolve it rather than guessing.
                    if not (_NETWORK_DIR.parent / "_multi_compartment" / f"{alias.name}.py").is_file():
                        offenders.append(
                            f"{name}.py:{node.lineno}: from braincell._multi_compartment import {alias.name}"
                        )
        self.assertEqual(
            offenders,
            [],
            "braincell/network runs while braincell._multi_compartment is only partially "
            "initialized, so it may import that package's submodules but not names bound "
            "by its __init__. Import from the submodule directly.",
        )


class ImportGraphTest(unittest.TestCase):
    """Pin the intra-package edges so a new one surfaces in review."""

    @staticmethod
    def _actual_graph() -> dict:
        graph = {}
        for name, path in _module_files():
            siblings = set()
            for node in _load_time_nodes(path):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    siblings.add(node.module)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.startswith("braincell.network."):
                        siblings.add(node.module.split(".")[2])
            graph[name] = siblings
        return graph

    def test_graph_matches_the_declared_layering(self) -> None:
        self.assertEqual(self._actual_graph(), _EXPECTED_GRAPH)

    def test_graph_is_acyclic(self) -> None:
        graph = self._actual_graph()
        visiting, done = set(), set()

        def visit(node, trail):
            if node in done:
                return
            self.assertNotIn(node, visiting, f"import cycle: {' -> '.join((*trail, node))}")
            visiting.add(node)
            for child in sorted(graph.get(node, ())):
                visit(child, (*trail, node))
            visiting.discard(node)
            done.add(node)

        for module in sorted(graph):
            visit(module, ())


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
