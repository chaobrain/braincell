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

"""Test-only helpers shared by the package-scope conformance guards.

Not a test module: the leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Iterator

_SECTION = re.compile(r"^(?P<title>[A-Z][A-Za-z ]{2,})\n(?P<rule>-{3,})[ \t]*$", re.MULTILINE)
_CITATION = re.compile(r"^\s*\.\.\s+\[\d+\]\s+\S", re.MULTILINE)


def public_symbols(module: ModuleType) -> Iterator[tuple[str, object]]:
    """Yield ``(name, obj)`` for every entry in ``module.__all__``."""
    for name in getattr(module, "__all__", ()):
        yield name, getattr(module, name)


def own_docstring(obj) -> str | None:
    """Return the docstring defined on ``obj`` itself, never an inherited one."""
    doc = obj.__dict__.get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        return inspect.cleandoc(doc)
    return None


def sections(doc: str) -> dict[str, str]:
    """Split a NumPy-doc docstring into ``{section title: section body}``."""
    found = {}
    matches = list(_SECTION.finditer(doc))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        found[match.group("title").rstrip()] = doc[match.end() : end]
    return found


def has_citation(doc: str) -> bool:
    """True when ``doc`` has a References section holding a ``.. [n]`` entry."""
    body = sections(doc).get("References")
    return bool(body) and bool(_CITATION.search(body))


_FOOTNOTE = re.compile(r"^\s*\.\.\s+\[(?P<label>[^\]\s]+)\]\s", re.MULTILINE)


def unreferenced_footnotes(doc: str) -> list[str]:
    """Labels of ``.. [label]`` entries that the text never cites as ``[label]_``.

    Sphinx reports such a footnote ("Footnote [1] is not referenced"), and the
    docs build runs with ``-W``, so it fails there; ``has_citation`` only checks
    that an entry exists.
    """
    return [m.group("label") for m in _FOOTNOTE.finditer(doc) if f"[{m.group('label')}]_" not in doc]


class ReExportTests:
    """Assertions shared by the per-package ``__all__`` guards.

    Mix into a :class:`unittest.TestCase` subclass that sets ``package``.
    Like :class:`DocstringConformanceTests` this is deliberately not a
    ``TestCase`` itself, so it is never collected on its own.

    A stale ``__all__`` is invisible until someone writes ``import *``, and
    then it fails far from the edit that caused it. Four packages had grown
    their own copy of these checks; this is the one copy.

    Attributes
    ----------
    package : ModuleType
        The package whose ``__all__`` is under test.
    reexport_sources : tuple of ModuleType
        Submodules whose own ``__all__`` must be fully re-exported by
        ``package``. Empty means the check is skipped.
    require_sorted_all : bool
        Whether ``package.__all__`` must be ASCII-sorted. Off by default:
        most packages predate the convention.
    """

    package: ModuleType | None = None
    reexport_sources: tuple[ModuleType, ...] = ()
    require_sorted_all: bool = False

    def test_every_exported_name_resolves(self):
        names = self.package.__all__
        missing = [name for name in names if not hasattr(self.package, name)]
        self.assertEqual(missing, [], f"listed in __all__ but not importable: {missing}")

    def test_all_has_no_duplicates(self):
        names = list(self.package.__all__)
        duplicated = sorted({name for name in names if names.count(name) > 1})
        self.assertEqual(duplicated, [], f"duplicated entries in __all__: {duplicated}")

    def test_all_is_sorted(self):
        if not self.require_sorted_all:
            self.skipTest("this package does not require a sorted __all__")
        names = list(self.package.__all__)
        self.assertEqual(names, sorted(names), "__all__ must stay ASCII-sorted")

    def test_every_source_module_export_is_re_exported(self):
        if not self.reexport_sources:
            self.skipTest("no source modules declared")
        expected = set()
        for module in self.reexport_sources:
            expected.update(module.__all__)
        dropped = sorted(expected - set(self.package.__all__))
        self.assertEqual(dropped, [], f"public in a submodule but not re-exported: {dropped}")


class DocstringConformanceTests:
    """Assertions shared by the per-package docstring guards.

    Mix into a :class:`unittest.TestCase` subclass that sets
    ``covered_modules`` and ``no_primary_source``. This class is deliberately
    not a ``TestCase`` itself, so it is never collected on its own.
    """

    covered_modules: tuple[ModuleType, ...] = ()
    no_primary_source: frozenset[str] = frozenset()

    def _symbols(self):
        for module in self.covered_modules:
            for name, obj in public_symbols(module):
                yield module.__name__, name, obj

    def test_every_public_symbol_defines_its_own_docstring(self):
        missing = [f"{mod}.{name}" for mod, name, obj in self._symbols() if own_docstring(obj) is None]
        self.assertEqual(missing, [], f"undocumented public symbols: {missing}")

    def test_summary_is_a_single_sentence_line(self):
        bad = []
        for mod, name, obj in self._symbols():
            doc = own_docstring(obj)
            if doc is None:
                continue
            summary = doc.splitlines()[0].strip()
            if not summary.endswith("."):
                bad.append(f"{mod}.{name}: {summary!r}")
        self.assertEqual(bad, [], f"summary must be one sentence ending in '.': {bad}")

    def test_every_public_symbol_cites_a_reference(self):
        uncited = []
        for mod, name, obj in self._symbols():
            if name in self.no_primary_source:
                continue
            doc = own_docstring(obj)
            if doc is None or not has_citation(doc):
                uncited.append(f"{mod}.{name}")
        self.assertEqual(uncited, [], f"missing References with '.. [n]': {uncited}")

    def test_every_footnote_is_referenced(self):
        bad = []
        for mod, name, obj in self._symbols():
            doc = own_docstring(obj)
            labels = unreferenced_footnotes(doc) if doc else []
            if labels:
                bad.append(f"{mod}.{name}: {labels}")
        self.assertEqual(bad, [], f"footnotes never cited as '[n]_' (the docs build fails): {bad}")

    def test_no_primary_source_allowlist_has_no_dead_entries(self):
        if not self.covered_modules:
            self.skipTest("no modules covered yet")
        live = {name for _, name, _ in self._symbols()}
        dead = sorted(n for n in self.no_primary_source if n not in live)
        self.assertEqual(dead, [], f"allowlist names no longer public: {dead}")


_BARE_NUMBER_CALLS = frozenset({"get_mantissa", "get_magnitude"})
_CATCHES_TYPE_ERROR = frozenset({"TypeError", "Exception", "BaseException"})
# NMODL unit conversions (``(1e12)`` pS to S, ``(1e6)`` C/ms to mA, ``(1e-3)``):
# applied to quantities that carry units they convert twice.
_UNIT_FACTORS = {1e6: "1e6", 1e12: "1e12", 1e-3: "1e-3"}


def _caught_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Tuple):
        return [name for elt in node.elts for name in _caught_names(elt)]
    return [node.id] if isinstance(node, ast.Name) else [node.attr] if isinstance(node, ast.Attribute) else []


def _unit_factor(node: ast.expr) -> str | None:
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _UNIT_FACTORS.get(node.value)
    return None


class _UnitSafetyVisitor(ast.NodeVisitor):
    def __init__(self, check_factors: bool):
        self.check_factors = check_factors
        self.scope: list[str] = []
        self.hits: list[tuple[str, str, int]] = []

    def _hit(self, pattern: str, node: ast.AST) -> None:
        self.hits.append((".".join(self.scope) or "<module>", pattern, node.lineno))

    def _scoped(self, node) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BARE_NUMBER_CALLS:
            self._hit(node.id, node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BARE_NUMBER_CALLS:
            self._hit(node.attr, node)
        elif node.attr in ("mantissa", "magnitude"):
            self._hit(f".{node.attr}", node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        caught = node.type is None or _CATCHES_TYPE_ERROR.intersection(_caught_names(node.type))
        if caught and not isinstance(node.body[-1], ast.Raise):
            self._hit("except TypeError", node)
        self.generic_visit(node)

    def _factor(self, node, op, operands) -> None:
        if self.check_factors and isinstance(op, (ast.Mult, ast.Div)):
            for label in filter(None, map(_unit_factor, operands)):
                self._hit(f"factor {label}", node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._factor(node, node.op, (node.left, node.right))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._factor(node, node.op, (node.value,))
        self.generic_visit(node)


def unit_safety_hits(source: str, *, check_factors: bool = False) -> list[tuple[str, str, int]]:
    """Return ``(qualname, pattern, line)`` for each construct that can hide a unit error.

    Reported: ``get_mantissa``/``get_magnitude`` under any prefix and the
    ``.mantissa``/``.magnitude`` attributes, which assume a unit instead of
    converting; an ``except`` catching ``TypeError`` (alone, in a tuple, via
    ``Exception``/``BaseException`` or bare) whose body does not end in
    ``raise``, which turns an error into a fallback; and, with
    ``check_factors``, a literal ``1e6``, ``1e12`` or ``1e-3`` multiplying or
    dividing anything. ``qualname`` joins the enclosing classes and functions
    with dots, or is ``"<module>"``.
    """
    visitor = _UnitSafetyVisitor(check_factors)
    visitor.visit(ast.parse(source))
    return visitor.hits


class UnitSafetyGuardTests:
    """Assertions shared by the per-package unit-safety guards.

    Mix into a :class:`unittest.TestCase` subclass that sets ``package_dir``.
    Each non-test ``.py`` file in that directory is parsed, not imported;
    every :func:`unit_safety_hits` hit must match an ``allowlist`` entry
    ``(file name, qualname, pattern, reason)``, and every entry a hit.
    ``check_factors`` turns on the unit-factor check.
    """

    package_dir: Path | None = None
    allowlist: tuple[tuple[str, str, str, str], ...] = ()
    check_factors: bool = False

    def _unit_safety_hits(self) -> list[tuple[str, str, str, int]]:
        if "_hits" not in type(self).__dict__:
            paths = sorted(p for p in Path(self.package_dir).glob("*.py") if not p.name.endswith("_test.py"))
            type(self)._hits = [
                (path.name, *hit)
                for path in paths
                for hit in unit_safety_hits(path.read_text(), check_factors=self.check_factors)
            ]
        return type(self)._hits

    def test_no_unlisted_unit_unsafe_construct(self):
        listed = {entry[:3] for entry in self.allowlist}
        unlisted = [f"{f}:{line} {q}: {p}" for f, q, p, line in self._unit_safety_hits() if (f, q, p) not in listed]
        self.assertEqual(unlisted, [], "convert with .to_decimal(unit) or re-raise; else allowlist it with a reason")

    def test_unit_safety_allowlist_has_no_dead_entries(self):
        live = {hit[:3] for hit in self._unit_safety_hits()}
        dead = [entry[:3] for entry in self.allowlist if entry[:3] not in live]
        self.assertEqual(dead, [], f"allowlist entries that match nothing: {dead}")

    def test_unit_safety_allowlist_entries_have_reasons(self):
        bad = [entry[:3] for entry in self.allowlist if len(entry) != 4 or not entry[3].strip()]
        self.assertEqual(bad, [], f"allowlist entries without a reason: {bad}")
