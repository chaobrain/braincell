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

"""Package-scope guards for :mod:`braincell.morph`.

Two invariants live here.

``__all__`` is this package's contract. A name that drops out of it is a
silent break for ``import *`` users; a name that stays in it after its
definition moves is an ``AttributeError`` the next time someone runs
``from braincell.morph import *``.

The import order is the other. ``morph`` owns ``Branch`` and
``Morphology``, so ``filter``, ``io``, ``vis``, ``_discretization``, and
``_multi_compartment`` all import it at *their* module scope -- and yet
twelve ``morph`` methods import back up into ``io``, ``vis``, ``filter``,
and the ``braincell`` root. Every one of those is a function-body import,
and hoisting any of them to module scope makes ``import braincell`` raise
``ImportError``. The failure often surfaces in a package the editor never
touched: hoisting ``from braincell.vis.plot3d import plot3d`` reports
``cannot import name 'RegionMask' from partially initialized module
'braincell.filter'``, because ``vis`` imports ``filter`` on the way up.

``docs/design/morph-layering-invariants.md`` records why each edge exists
and the measured failure for each one. The two checks below turn the
invariant into a targeted failure so a future editor learns *why* rather
than bisecting a traceback that does not mention their edit.
"""

import ast
import pathlib
import unittest

import braincell.morph
from braincell._testing import ReExportTests

_MORPH_DIR = pathlib.Path(braincell.morph.__file__).parent

#: Packages ``morph`` sits below. A module-scope import of any of these
#: closes an import cycle and is fatal at ``import braincell``.
_UPWARD = ("braincell.io", "braincell.vis", "braincell.filter")

#: Every ``braincell`` module a ``morph`` module may import at load time,
#: as ``{module_stem: {target, ...}}``. ``braincell._misc`` is a leaf;
#: the rest are intra-package.
_EXPECTED_LOAD_TIME_IMPORTS = {
    "__init__": {"braincell.morph.branch", "braincell.morph.morphology"},
    "_spatial": {"braincell.morph.morphology"},
    "_testing": {"braincell.morph.branch", "braincell.morph.morphology"},
    "branch": {"braincell._misc"},
    "morphology": {"braincell.morph.branch"},
}


def _module_files():
    """Yield ``(module_name, path)`` for every non-test module here."""
    for path in sorted(_MORPH_DIR.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        yield path.stem, path


def _load_time_nodes(tree: ast.Module):
    """Yield import nodes that run at module load.

    Skips ``if TYPE_CHECKING:`` bodies and function bodies -- a deferred
    import inside a function cannot participate in an import cycle, which
    is exactly the mechanism this package relies on.
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

    yield from walk(tree.body)


def _braincell_targets(node):
    """Yield the ``braincell`` modules one import node names."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "braincell" or alias.name.startswith("braincell."):
                yield alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            # ``from . import x`` / ``from .branch import Branch``.
            yield f"braincell.morph.{node.module}" if node.module else "braincell.morph"
        elif node.module == "braincell" or (node.module or "").startswith("braincell."):
            yield node.module


def _load_time_graph() -> dict:
    graph = {}
    for name, path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = set()
        for node in _load_time_nodes(tree):
            targets.update(_braincell_targets(node))
        graph[name] = targets
    return graph


class MorphReExportTest(ReExportTests, unittest.TestCase):
    """``braincell.morph.__all__`` resolves, has no duplicates, and stays sorted."""

    package = braincell.morph
    require_sorted_all = True


class UpwardImportsAreDeferredTest(unittest.TestCase):
    """No ``morph`` module may import ``io``, ``vis``, ``filter``, or the root at load time."""

    def test_no_module_scope_import_reaches_upward(self) -> None:
        offenders = []
        for name, targets in sorted(_load_time_graph().items()):
            for target in sorted(targets):
                if target == "braincell" or any(target == pkg or target.startswith(f"{pkg}.") for pkg in _UPWARD):
                    offenders.append(f"{name}.py: {target}")
        self.assertEqual(
            offenders,
            [],
            "braincell.morph is imported before braincell.io / .vis / .filter exist, so a "
            "module-scope import of one of them makes `import braincell` raise ImportError -- "
            "often from a package you did not touch. Move it into the method body. See "
            "docs/design/morph-layering-invariants.md.",
        )


class LoadTimeImportsTest(unittest.TestCase):
    """Pin the load-time ``braincell`` imports so a new one surfaces in review.

    :class:`UpwardImportsAreDeferredTest` catches the fatal case. This
    catches the case that is merely load bearing: a new module-scope edge
    into a package that happens to be imported earlier today, and would
    silently become fatal if the order in ``braincell/__init__.py``
    changed.
    """

    def test_load_time_imports_match_the_declared_set(self) -> None:
        self.assertEqual(_load_time_graph(), _EXPECTED_LOAD_TIME_IMPORTS)


if __name__ == "__main__":
    unittest.main()
