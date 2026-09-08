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

"""Architectural guard for the layering of :mod:`braincell._compute`.

This module pins three properties of the package that were established by the
split of the old ``runtime.py`` monolith, and that nothing else enforces:

1. **No upward imports.** No production module in ``braincell._compute`` has a
   load-time import from ``braincell._multi_compartment``. ``_compute`` is the
   lower layer: ``Cell`` compiles itself down into a
   :class:`~braincell._compute.state.CellRuntimeState`, never the reverse. A
   package-level cycle between the two packages existed before the split and is
   what this assertion exists to stop from coming back.
2. **Acyclic.** The intra-package load-time import graph has no cycles.
3. **Layer order respected.** Within the runtime-construction chain the only
   legal direction is ``layouts -> ions -> bindings -> state``: a module may
   import its predecessors and never its successors.

Why static parsing
------------------

The graph is recovered with :mod:`ast`, not by importing anything. Three
reasons, in order of importance:

- A runtime check cannot see the property being asserted. The original
  ``_compute`` ↔ ``_multi_compartment`` cycle was importable — it only broke
  under particular import orders — so an :mod:`importlib`-based probe would
  have passed against the very code this guard is meant to reject.
- Parsing has no import side effects, so the guard cannot be perturbed by, or
  perturb, module state, JAX initialisation, or registry population.
- It runs in milliseconds over the whole package.

What counts as an edge
----------------------

Only *load-time* imports create an edge:

- Bodies guarded by ``if TYPE_CHECKING:`` (or ``if typing.TYPE_CHECKING:``) are
  skipped. Four modules deliberately import under that guard to break an
  annotation cycle — ``layouts``, ``ions`` and ``bindings`` pull
  ``CellRuntimeState`` from ``.state``, and ``state`` pulls ``Cell`` from
  ``braincell._multi_compartment.cell``. All four are correct by design and
  must not be flagged.
- Imports nested inside a function or method body are deferred until the
  function is called and create no load-time edge, so the walker never
  descends into them.
- Imports inside a ``class`` body, or inside a module-level ``with``, ``for``
  or ``while`` body, **do** create a load-time edge: a class body and the
  bodies of these compound statements all execute when the enclosing module
  is imported, exactly like an ``if`` or ``try`` body does. The walker
  descends into all of them.
- Test modules (``*_test.py``, ``_testing.py``) are excluded entirely. A test
  importing ``Cell`` is not an architecture violation.

Known limitations
-----------------

The walker is a pragmatic AST match, not a full static-import resolver. The
following constructs are invisible to it. None appear in the codebase today;
if one is introduced, this guard will not catch it:

- ``match``/``case`` bodies (:class:`ast.Match`) are not descended into.
- ``except*`` bodies (:class:`ast.TryStar`) are not descended into.
- Dynamic imports — ``importlib.import_module(...)`` and ``__import__(...)``
  — are invisible; only literal :class:`ast.Import` / :class:`ast.ImportFrom`
  statements are matched.
- :func:`_production_modules` globs ``*.py`` non-recursively over
  ``_PACKAGE_DIR``, so a future ``_compute/<subpackage>/`` would be unscanned.
- :func:`_is_type_checking_guard` matches any attribute named
  ``TYPE_CHECKING`` regardless of its owner, so an unrelated
  ``if _Cfg.TYPE_CHECKING:`` would be (mis)treated as the real guard and its
  body skipped.
"""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

# The package under guard, and the directory its modules live in.
_PACKAGE = "braincell._compute"
_PACKAGE_DIR = Path(__file__).resolve().parent

# The package `_compute` must never import at load time. `_multi_compartment`
# sits above `_compute` and depends on it; an edge in this direction closes a
# package-level cycle.
_FORBIDDEN_UPWARD = "braincell._multi_compartment"

# The runtime-construction chain, in the order a `CellRuntimeState` is built.
# A module may import anything earlier in this tuple and nothing later.
_LAYER_ORDER = ("layouts", "ions", "bindings", "state")


@dataclass(frozen=True)
class ImportRecord:
    """One load-time import statement, resolved to the modules it names.

    Attributes
    ----------
    lineno : int
        1-based line number of the statement in its source file.
    statement : str
        The statement rendered back to source, used in failure messages so a
        contributor who trips the guard sees the exact line to change.
    paths : tuple of str
        Absolute dotted module paths the statement could refer to. A
        ``from X import a, b`` yields ``X`` alongside ``X.a`` and ``X.b``,
        because whether ``a`` is a submodule or an attribute is not knowable
        from the syntax alone. Over-generating here is safe: consumers match
        the candidates against known module names.
    """

    lineno: int
    statement: str
    paths: Tuple[str, ...]


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Return whether an ``if`` test is a ``TYPE_CHECKING`` guard.

    Parameters
    ----------
    test : ast.expr
        The test expression of an :class:`ast.If` node.

    Returns
    -------
    bool
        ``True`` for both the bare ``if TYPE_CHECKING:`` form and the qualified
        ``if typing.TYPE_CHECKING:`` form.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _iter_load_time_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield the statements in ``body`` that execute when the module loads.

    Descends into ``if``, ``try``, ``class``, ``with``, ``for`` and ``while``
    bodies, all of which run when the enclosing module is imported — a class
    body executes immediately to build the class namespace, and the other
    three are ordinary compound statements with no deferral of their own. The
    walker never descends into function or method bodies (or comprehension
    scopes), whose imports are deferred until the function is called and so
    create no load-time edge. The body of a ``TYPE_CHECKING`` guard is skipped
    while its ``else`` branch — which does run — is kept.

    Parameters
    ----------
    body : sequence of ast.stmt
        A statement list, typically ``ast.Module.body`` or the ``body`` of a
        ``class``, ``if``, ``try``, ``with``, ``for`` or ``while`` statement.

    Yields
    ------
    ast.stmt
        Each statement reached at module load time.
    """
    for node in body:
        yield node
        if isinstance(node, ast.If):
            if not _is_type_checking_guard(node.test):
                yield from _iter_load_time_statements(node.body)
            yield from _iter_load_time_statements(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _iter_load_time_statements(node.body)
            for handler in node.handlers:
                yield from _iter_load_time_statements(handler.body)
            yield from _iter_load_time_statements(node.orelse)
            yield from _iter_load_time_statements(node.finalbody)
        elif isinstance(node, ast.ClassDef):
            yield from _iter_load_time_statements(node.body)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _iter_load_time_statements(node.body)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            yield from _iter_load_time_statements(node.body)
            yield from _iter_load_time_statements(node.orelse)


def _resolve_relative(level: int, module: Optional[str], package: str) -> str:
    """Resolve a relative ``from`` import to an absolute dotted path.

    Parameters
    ----------
    level : int
        Number of leading dots on the import (``0`` for an absolute import).
    module : str or None
        The module part of the statement, ``None`` for ``from . import x``.
    package : str
        Dotted name of the package containing the importing module.

    Returns
    -------
    str
        The absolute dotted path the statement's names are looked up in.
    """
    if level == 0:
        return module or ""
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (level - 1)]) if level > 1 else package
    return f"{base}.{module}" if module else base


def _import_paths(node: ast.stmt, package: str) -> Tuple[str, ...]:
    """Return the absolute module paths an import statement could name.

    Parameters
    ----------
    node : ast.stmt
        An :class:`ast.Import` or :class:`ast.ImportFrom` node. Any other node
        yields an empty result.
    package : str
        Dotted name of the package containing the importing module, used to
        resolve relative imports.

    Returns
    -------
    tuple of str
        Candidate absolute dotted paths, deduplicated and order-preserving.
    """
    paths = []
    if isinstance(node, ast.Import):
        paths.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        base = _resolve_relative(node.level, node.module, package)
        if base:
            paths.append(base)
            paths.extend(f"{base}.{alias.name}" for alias in node.names)
    return tuple(dict.fromkeys(paths))


def load_time_imports(source: str, package: str = _PACKAGE) -> Tuple[ImportRecord, ...]:
    """Extract the load-time imports of a module from its source text.

    Parameters
    ----------
    source : str
        Full text of a Python module.
    package : str, optional
        Dotted name of the package the module belongs to, used to resolve
        relative imports. Defaults to ``braincell._compute``.

    Returns
    -------
    tuple of ImportRecord
        One record per import statement that executes at module load time, in
        source order.
    """
    tree = ast.parse(source)
    records = []
    for node in _iter_load_time_statements(tree.body):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        paths = _import_paths(node, package)
        if paths:
            records.append(ImportRecord(lineno=node.lineno, statement=ast.unparse(node), paths=paths))
    return tuple(records)


def is_upward(path: str) -> bool:
    """Return whether a dotted path lies in the forbidden upward package.

    Parameters
    ----------
    path : str
        An absolute dotted module path.

    Returns
    -------
    bool
        ``True`` for ``braincell._multi_compartment`` and anything below it.
    """
    return path == _FORBIDDEN_UPWARD or path.startswith(f"{_FORBIDDEN_UPWARD}.")


def sibling_of(path: str, known: Sequence[str], package: str = _PACKAGE) -> Optional[str]:
    """Map a dotted path onto the sibling module of ``package`` it refers to.

    Parameters
    ----------
    path : str
        An absolute dotted module path.
    known : sequence of str
        Names of the modules that exist in the package.
    package : str, optional
        Dotted name of the package. Defaults to ``braincell._compute``.

    Returns
    -------
    str or None
        The sibling module name, or ``None`` if the path names something
        outside the package. Both ``from .layouts import X`` and the absolute
        ``from braincell._compute.layouts import X`` resolve to ``"layouts"``;
        the trailing ``X`` is discarded because it is not a known module.
    """
    prefix = f"{package}."
    if not path.startswith(prefix):
        return None
    head = path[len(prefix) :].split(".", 1)[0]
    return head if head in known else None


def is_test_module(filename: str) -> bool:
    """Return whether a filename belongs to the test surface, not production.

    Parameters
    ----------
    filename : str
        Base name of a Python file, e.g. ``"layouts_test.py"``.

    Returns
    -------
    bool
        ``True`` for ``*_test.py`` and ``_testing.py``, which are exempt from
        every assertion in this module.
    """
    return filename.endswith("_test.py") or filename == "_testing.py"


def _production_modules() -> Dict[str, Path]:
    """Return the production modules of the package, keyed by module name.

    Returns
    -------
    dict of str to pathlib.Path
        Module name (``"__init__"`` for the package initialiser) to file path,
        in sorted name order, with test modules excluded.
    """
    return {path.stem: path for path in sorted(_PACKAGE_DIR.glob("*.py")) if not is_test_module(path.name)}


def _scan_package() -> Dict[str, Tuple[ImportRecord, ...]]:
    """Parse every production module and return its load-time imports.

    Returns
    -------
    dict of str to tuple of ImportRecord
        Module name to the imports that execute when it loads.
    """
    return {name: load_time_imports(path.read_text(encoding="utf-8")) for name, path in _production_modules().items()}


def _sibling_graph(scan: Dict[str, Tuple[ImportRecord, ...]]) -> Dict[str, Dict[str, ImportRecord]]:
    """Reduce a package scan to its intra-package import graph.

    Parameters
    ----------
    scan : dict of str to tuple of ImportRecord
        Output of :func:`_scan_package`.

    Returns
    -------
    dict of str to dict of str to ImportRecord
        For each module, the siblings it imports mapped to the first statement
        that imports them. Self-edges are dropped.
    """
    known = tuple(scan)
    graph: Dict[str, Dict[str, ImportRecord]] = {name: {} for name in known}
    for name, records in scan.items():
        for record in records:
            for path in record.paths:
                sibling = sibling_of(path, known)
                if sibling is not None and sibling != name and sibling not in graph[name]:
                    graph[name][sibling] = record
    return graph


def _find_cycle(graph: Dict[str, Dict[str, ImportRecord]]) -> Optional[Tuple[str, ...]]:
    """Return one cycle in a directed graph, or ``None`` if it is acyclic.

    Parameters
    ----------
    graph : dict of str to dict of str to ImportRecord
        Adjacency mapping; only the successor keys are used.

    Returns
    -------
    tuple of str or None
        The nodes of a cycle, closed by repeating its first node (e.g.
        ``("a", "b", "a")``). ``None`` when no cycle exists.
    """
    visiting: list[str] = []
    on_path = set()
    done = set()

    def walk(node: str) -> Optional[Tuple[str, ...]]:
        visiting.append(node)
        on_path.add(node)
        for successor in sorted(graph.get(node, ())):
            if successor in on_path:
                start = visiting.index(successor)
                return tuple(visiting[start:]) + (successor,)
            if successor not in done:
                found = walk(successor)
                if found is not None:
                    return found
        visiting.pop()
        on_path.discard(node)
        done.add(node)
        return None

    for name in sorted(graph):
        if name not in done:
            cycle = walk(name)
            if cycle is not None:
                return cycle
    return None


class NoUpwardImportTest(unittest.TestCase):
    """Guard: ``_compute`` never imports ``_multi_compartment`` at load time."""

    def test_no_production_module_imports_multi_compartment(self) -> None:
        offences = []
        for name, records in _scan_package().items():
            for record in records:
                if any(is_upward(path) for path in record.paths):
                    offences.append(f"  {name}.py:{record.lineno}: {record.statement}")
        self.assertEqual(
            [],
            offences,
            "braincell._compute must not import braincell._multi_compartment at load time; "
            "_compute is the lower layer and such an import re-closes the package cycle the "
            "runtime split removed. Move the dependency down, or defer it behind "
            "`if TYPE_CHECKING:` if it is only needed for annotations. Offending imports:\n" + "\n".join(offences),
        )

    def test_the_type_checking_cell_import_in_state_is_not_flagged(self) -> None:
        # `state.py` imports `Cell` under `if TYPE_CHECKING:` on purpose. This
        # pins the exemption so a future tightening of the walker cannot make
        # the guard above pass for the wrong reason.
        source = (_PACKAGE_DIR / "state.py").read_text(encoding="utf-8")
        self.assertIn("from braincell._multi_compartment.cell import Cell", source)
        self.assertEqual(
            [], [record for record in load_time_imports(source) if any(is_upward(path) for path in record.paths)]
        )


class AcyclicTest(unittest.TestCase):
    """Guard: the intra-package import graph is a DAG."""

    def test_intra_package_import_graph_is_acyclic(self) -> None:
        cycle = _find_cycle(_sibling_graph(_scan_package()))
        self.assertIsNone(
            cycle,
            "braincell._compute modules must form an acyclic import graph, but found: "
            f"{' -> '.join(cycle) if cycle else ''}. Break the cycle by moving the shared "
            "code into the lower module, or defer one edge behind `if TYPE_CHECKING:`.",
        )

    def test_graph_matches_the_declared_layering(self) -> None:
        # The expected graph is spelled out so that *any* new intra-package
        # edge — not merely an illegal one — surfaces in review.
        expected = {
            "__init__": set(),
            "bindings": {"ions", "layouts", "parameters"},
            "bridge": set(),
            "ions": {"layouts"},
            "layouts": {"parameters"},
            "parameters": set(),
            "scheduling": set(),
            "state": {"bindings", "bridge", "layouts", "parameters"},
            # ``table`` builds its rows by matching mechanisms against
            # layout signatures, so it reads ``layouts`` directly rather
            # than through ``state``. ``layouts`` is a leaf, so the edge
            # keeps the graph a DAG.
            "table": {"layouts", "state"},
        }
        actual = {name: set(edges) for name, edges in _sibling_graph(_scan_package()).items()}
        self.assertEqual(expected, actual)


class LayerOrderTest(unittest.TestCase):
    """Guard: ``layouts -> ions -> bindings -> state`` is one-directional."""

    def test_no_module_imports_a_later_layer(self) -> None:
        rank = {name: index for index, name in enumerate(_LAYER_ORDER)}
        graph = _sibling_graph(_scan_package())
        offences = []
        for name, edges in graph.items():
            if name not in rank:
                continue
            for sibling, record in sorted(edges.items()):
                if rank.get(sibling, -1) > rank[name]:
                    offences.append(f"  {name}.py:{record.lineno} imports {sibling}: {record.statement}")
        self.assertEqual(
            [],
            offences,
            "braincell._compute layers must be imported in the order "
            f"{' -> '.join(_LAYER_ORDER)}; a module may import its predecessors and never its "
            "successors. Offending imports:\n" + "\n".join(offences),
        )

    def test_every_layer_module_exists(self) -> None:
        # Guards against the layer assertion silently becoming a no-op if a
        # module is renamed.
        self.assertLessEqual(set(_LAYER_ORDER), set(_production_modules()))


class WalkerTest(unittest.TestCase):
    """The walker itself, exercised on synthetic sources.

    Without these, the three guards above would pass just as happily with a
    walker that finds nothing at all.
    """

    def test_type_checking_body_is_skipped_in_both_spellings(self) -> None:
        source = (
            "from typing import TYPE_CHECKING\n"
            "import typing\n"
            "if TYPE_CHECKING:\n"
            "    from .state import CellRuntimeState\n"
            "if typing.TYPE_CHECKING:\n"
            "    from braincell._multi_compartment.cell import Cell\n"
        )
        statements = [record.statement for record in load_time_imports(source)]
        self.assertEqual(["from typing import TYPE_CHECKING", "import typing"], statements)

    def test_type_checking_else_branch_is_kept(self) -> None:
        source = "if TYPE_CHECKING:\n    from .state import CellRuntimeState\nelse:\n    from .layouts import Layout\n"
        self.assertEqual(["from .layouts import Layout"], [r.statement for r in load_time_imports(source)])

    def test_deferred_imports_are_not_load_time_edges(self) -> None:
        source = (
            "def build():\n"
            "    from braincell._multi_compartment.cell import Cell\n"
            "    return Cell\n"
            "\n"
            "class Runner:\n"
            "    def go(self):\n"
            "        from .state import CellRuntimeState\n"
            "        return CellRuntimeState\n"
        )
        self.assertEqual((), load_time_imports(source))

    def test_plain_top_level_if_body_is_walked(self) -> None:
        source = "import sys\nif sys.version_info >= (3, 12):\n    from .state import CellRuntimeState\n"
        self.assertEqual(
            ("braincell._compute.state", "braincell._compute.state.CellRuntimeState"),
            load_time_imports(source)[1].paths,
        )

    def test_relative_import_of_a_bare_submodule_resolves(self) -> None:
        (record,) = load_time_imports("from . import bridge\n")
        self.assertEqual("bridge", sibling_of(record.paths[-1], ("bridge", "state")))

    def test_upward_import_is_detected_in_every_spelling(self) -> None:
        sources = (
            "from braincell._multi_compartment.cell import Cell\n",
            "from braincell._multi_compartment import cell\n",
            "from braincell import _multi_compartment\n",
            "import braincell._multi_compartment.cell\n",
        )
        for source in sources:
            with self.subTest(source=source.strip()):
                (record,) = load_time_imports(source)
                self.assertTrue(any(is_upward(path) for path in record.paths))

    def test_unrelated_imports_are_not_upward(self) -> None:
        for source in ("import braincell\n", "from braincell.mech import Density\n", "import numpy as np\n"):
            with self.subTest(source=source.strip()):
                (record,) = load_time_imports(source)
                self.assertFalse(any(is_upward(path) for path in record.paths))

    def test_attribute_import_is_not_mistaken_for_a_submodule(self) -> None:
        (record,) = load_time_imports("from .layouts import MechanismLayout\n")
        known = ("layouts", "state")
        self.assertEqual({"layouts"}, {sibling_of(path, known) for path in record.paths} - {None})

    def test_find_cycle_detects_a_cycle(self) -> None:
        record = ImportRecord(lineno=1, statement="from .b import x", paths=())
        graph = {"a": {"b": record}, "b": {"a": record}}
        self.assertEqual(("a", "b", "a"), _find_cycle(graph))

    def test_find_cycle_accepts_a_diamond(self) -> None:
        record = ImportRecord(lineno=1, statement="from .x import y", paths=())
        graph = {"a": {"b": record, "c": record}, "b": {"d": record}, "c": {"d": record}, "d": {}}
        self.assertIsNone(_find_cycle(graph))

    def test_test_modules_are_excluded_from_the_scan(self) -> None:
        scanned = set(_production_modules())
        self.assertNotIn("state_test", scanned)
        self.assertNotIn("_testing", scanned)
        self.assertIn("state", scanned)


if __name__ == "__main__":
    unittest.main()
