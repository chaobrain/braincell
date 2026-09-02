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

"""Architectural guard for :mod:`braincell.vis` and its public surface.

Two properties are pinned here, neither of which any other test covers.

**One-way dependency.** ``braincell.vis`` sits at the top of the package:
it imports ``morph``, ``filter``, ``_discretization`` and
``_multi_compartment``, and none of them may import it back at load time.
``braincell/vis/cell_topology.py`` imports ``Cell`` at module level, so a
load-time edge in the other direction closes a cycle and breaks
``import braincell`` outright. The two model classes that *do* offer
visualization methods — ``Morphology.vis2d`` / ``vis3d`` and
``Branch.vis2d`` / ``vis3d`` — import ``braincell.vis`` inside the method
body precisely to stay off this graph, and that is what the AST scan
below enforces.

The graph is recovered with :mod:`ast` rather than by importing, because
a cycle of this kind is importable under *some* orders and not others: a
runtime probe that happened to import ``braincell`` first would pass
against the very code this guard rejects. Only load-time statements count
— a function body's imports are deferred and create no edge — and
``if TYPE_CHECKING:`` bodies are skipped because they never execute.

**Exported surface.** ``__all__`` is the module's contract, and unlike
:mod:`braincell.channel` and :mod:`braincell.ion` this package had no
conformance check, so a name could be added to ``__all__`` without being
importable, or exported without being listed.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Iterator, Sequence

import braincell.vis
from braincell._testing import ReExportTests

_PACKAGE_DIR = Path(braincell.vis.__file__).resolve().parent
_BRAINCELL_DIR = _PACKAGE_DIR.parent

#: No production module outside ``braincell/vis/`` may reach this at load time.
_FORBIDDEN = "braincell.vis"


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Return whether an ``if`` test is a ``TYPE_CHECKING`` guard."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _iter_load_time_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield the statements in ``body`` that execute when the module loads.

    Descends into ``if`` / ``try`` / ``class`` / ``with`` / ``for`` /
    ``while`` bodies, all of which run at import. Never descends into a
    function or method body, whose imports are deferred until the call.
    The body of a ``TYPE_CHECKING`` guard is skipped; its ``else`` branch
    is kept, because that one does run.
    """
    for statement in body:
        yield statement
        if isinstance(statement, ast.If):
            if not _is_type_checking_guard(statement.test):
                yield from _iter_load_time_statements(statement.body)
            yield from _iter_load_time_statements(statement.orelse)
        elif isinstance(statement, ast.Try):
            yield from _iter_load_time_statements(statement.body)
            yield from _iter_load_time_statements(statement.orelse)
            yield from _iter_load_time_statements(statement.finalbody)
            for handler in statement.handlers:
                yield from _iter_load_time_statements(handler.body)
        elif isinstance(statement, (ast.ClassDef, ast.With, ast.For, ast.While)):
            yield from _iter_load_time_statements(statement.body)
            yield from _iter_load_time_statements(getattr(statement, "orelse", []))


def _imported_modules(statement: ast.stmt, *, module_package: str) -> Iterator[str]:
    """Yield the absolute module names one import statement names."""
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            yield alias.name
    elif isinstance(statement, ast.ImportFrom):
        if statement.level:
            # Relative: resolve against the importing module's package.
            parts = module_package.split(".")
            base = parts[: len(parts) - statement.level + 1]
            prefix = ".".join(base + ([statement.module] if statement.module else []))
        else:
            prefix = statement.module or ""
        yield prefix
        for alias in statement.names:
            yield f"{prefix}.{alias.name}" if prefix else alias.name


def _production_modules() -> Iterator[Path]:
    """Yield every production ``.py`` file under ``braincell/`` outside ``vis/``.

    Test modules and their helpers are excluded: they are not part of the
    package's import graph, and several legitimately import
    ``braincell.vis`` at module level.
    """
    for path in sorted(_BRAINCELL_DIR.rglob("*.py")):
        if _PACKAGE_DIR in path.parents or path.parent == _PACKAGE_DIR:
            continue
        if path.name.endswith("_test.py") or path.name in {"_testing.py", "conftest.py"}:
            continue
        yield path


class VisIsNotImportedAtLoadTimeTest(unittest.TestCase):
    """Nothing below ``braincell.vis`` may import it while loading."""

    def test_no_production_module_outside_vis_imports_it_at_load_time(self) -> None:
        offenders = []
        for path in _production_modules():
            relative = path.relative_to(_BRAINCELL_DIR.parent)
            dotted = ".".join(relative.with_suffix("").parts)
            package = dotted.rsplit(".", 1)[0] if path.name != "__init__.py" else dotted
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for statement in _iter_load_time_statements(tree.body):
                if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                    continue
                for name in _imported_modules(statement, module_package=package):
                    if name == _FORBIDDEN or name.startswith(f"{_FORBIDDEN}."):
                        offenders.append(f"{relative}:{statement.lineno}: {ast.unparse(statement)}")

        self.assertEqual(
            offenders,
            [],
            "braincell.vis must only be imported lazily from outside the package — "
            "a load-time edge closes a cycle with braincell/vis/cell_topology.py, "
            "which imports Cell at module level. Move these into the function body:\n" + "\n".join(offenders),
        )

    def test_the_package_imports_cleanly_as_the_very_first_import(self) -> None:
        # The cycle this guards against is order-dependent: importing
        # `braincell` first binds `Cell` before `vis` loads and hides it.
        # Importing `braincell.vis` cold is the order that would break.
        result = subprocess.run(
            [sys.executable, "-c", "import braincell.vis; print(braincell.vis.plot_cell_topology.__name__)"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("plot_cell_topology", result.stdout)


class VisPublicSurfaceTest(ReExportTests, unittest.TestCase):
    """``braincell.vis.__all__`` is the module's contract."""

    package = braincell.vis
    require_sorted_all = True

    def test_no_public_callable_is_exported_without_being_listed(self) -> None:
        exported = set(braincell.vis.__all__)
        submodules = {path.stem for path in _PACKAGE_DIR.glob("*.py")} | {
            path.name for path in _PACKAGE_DIR.iterdir() if path.is_dir()
        }
        unlisted = [
            name
            for name in dir(braincell.vis)
            if not name.startswith("_") and name not in exported and name not in submodules
        ]
        self.assertEqual(unlisted, [], f"public but absent from __all__: {unlisted}")

    def test_plot_cell_topology_is_the_only_cell_entry_point(self) -> None:
        # The Cell.vis_* methods were removed in favour of this function;
        # nothing should quietly reintroduce a second spelling.
        cell_entry_points = [name for name in braincell.vis.__all__ if "cell" in name]
        self.assertEqual(cell_entry_points, ["plot_cell_topology"])


if __name__ == "__main__":
    unittest.main()
