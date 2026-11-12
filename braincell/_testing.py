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
import importlib
import inspect
import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from typing import Iterator, Mapping

_SECTION = re.compile(r"^(?P<title>[A-Z][A-Za-z ]{2,})\n(?P<rule>-{3,})[ \t]*$", re.MULTILINE)
_CITATION = re.compile(r"^\s*\.\.\s+\[\d+\]\s+\S", re.MULTILINE)


def public_symbols(module: ModuleType) -> Iterator[tuple[str, object]]:
    """Yield ``(name, obj)`` for every entry in ``module.__all__``."""
    for name in getattr(module, "__all__", ()):
        yield name, getattr(module, name)


def public_paths() -> dict[str, object]:
    """Return ``{path: object}`` for every public path of braincell.

    The public paths are ``braincell.<name>`` for each name in
    ``braincell.__all__``, and ``braincell.<package>.<name>`` for each name in
    the ``__all__`` of a stable package (a module listed in
    ``braincell.__all__``). ``braincell.experimental`` is imported explicitly
    and is not among them. Every other path is internal, including a submodule
    path without an underscore such as ``braincell.morph.morphology``.
    """
    import braincell

    paths = {}
    for name, obj in public_symbols(braincell):
        paths[f"braincell.{name}"] = obj
        if isinstance(obj, ModuleType):
            for sub_name, sub_obj in public_symbols(obj):
                paths[f"braincell.{name}.{sub_name}"] = sub_obj
    return paths


def is_public_import(path: str, public: Mapping[str, object]) -> bool:
    """Return whether importing the dotted ``path`` uses a public path.

    ``public`` is :func:`public_paths`. ``braincell`` itself and anything
    under ``braincell.experimental`` also count.
    """
    return path in public or path == "braincell" or path.split(".")[:2] == ["braincell", "experimental"]


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


_RST_MODULE = re.compile(r"^\s*\.\. (?:currentmodule|module|automodule)::\s+(?P<name>\S+)\s*$")
_RST_AUTO = re.compile(r"^\s*\.\. (?:py:)?(?:auto)?(?:class|function|data|exception|module)::\s+(?P<name>[^\s(]+)")
_RST_AUTOSUMMARY = re.compile(r"^(?P<indent>[ \t]*)\.\. autosummary::\s*$")


def _qualify(name: str, module: str | None) -> str:
    """Resolve an rst target the way Sphinx does: relative to ``module`` unless absolute."""
    if module is None or name == "braincell" or name.startswith("braincell."):
        return name
    return f"{module}.{name}"


def api_reference_entries(apis_dir: Path) -> list[tuple[str, str]]:
    """Return ``(dotted path, rst file name)`` for each entry in the API reference.

    Reads every ``*.rst`` file in ``apis_dir``. An entry is a name in an
    ``.. autosummary::`` block (option lines skipped) or the target of an
    object directive: ``autoclass``, ``autofunction``, ``autodata``,
    ``autoexception`` or ``automodule``, or the same without ``auto`` (with or
    without the ``py:`` prefix; a signature after the name is dropped). Names
    are resolved against the latest ``.. currentmodule::`` (or ``module::`` /
    ``automodule::``).
    """
    entries = []
    for page in sorted(Path(apis_dir).glob("*.rst")):
        module = None
        lines = page.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1
            auto = _RST_AUTO.match(line)
            if auto is not None:
                entries.append((_qualify(auto["name"], module), page.name))
            current = _RST_MODULE.match(line)
            if current is not None:
                module = current["name"]
                continue
            summary = _RST_AUTOSUMMARY.match(line)
            if summary is None:
                continue
            indent = len(summary["indent"])
            while index < len(lines) and (
                not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent
            ):
                name = lines[index].strip()
                index += 1
                if name and not name.startswith((":", "..")):
                    entries.append((_qualify(name, module), page.name))
    return entries


_UNRESOLVED = object()


def _resolve(path: str) -> object:
    """Return the object a dotted path names, or ``_UNRESOLVED``."""
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for part in parts[split:]:
            obj = getattr(obj, part, _UNRESOLVED)
            if obj is _UNRESOLVED:
                break
        return obj
    return _UNRESOLVED


def api_reference_gaps(
    entries: list[tuple[str, str]], public: Mapping[str, object]
) -> tuple[list[str], list[tuple[tuple[str, str], ...]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Compare API reference entries with the public objects.

    Objects are compared by identity, so an object exported under two paths
    is one object; modules are left out on both sides.

    Parameters
    ----------
    entries : list of (str, str)
        ``(dotted path, file)`` pairs, as from :func:`api_reference_entries`.
    public : mapping of str to object
        ``{path: object}``, as from :func:`public_paths`.

    Returns
    -------
    missing : list of str
        One path (the first in ``public``) per public object without an entry.
    duplicated : list of tuple
        For each object with two or more entries, its entries.
    not_public : list of (str, str)
        Entries naming an object that is not public.
    unresolved : list of (str, str)
        Entries naming nothing.
    """
    first_path = {}
    for path, obj in public.items():
        if not isinstance(obj, ModuleType):
            first_path.setdefault(id(obj), path)
    by_object: dict[int, list[tuple[str, str]]] = {}
    not_public, unresolved = [], []
    for path, file in entries:
        obj = _resolve(path)
        if obj is _UNRESOLVED:
            unresolved.append((path, file))
        elif isinstance(obj, ModuleType):
            continue
        elif id(obj) not in first_path:
            not_public.append((path, file))
        else:
            by_object.setdefault(id(obj), []).append((path, file))
    missing = sorted(path for key, path in first_path.items() if key not in by_object)
    duplicated = [tuple(group) for group in by_object.values() if len(group) > 1]
    return missing, duplicated, not_public, unresolved


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


#: Where users copy code from. ``docs/developer/`` is written for
#: contributors and may show internal paths.
_USER_MATERIAL = ("examples", "docs", "README.md")
_CONTRIBUTOR_DOCS = "docs/developer/"
_USER_MATERIAL_SUFFIXES = (".py", ".ipynb", ".md", ".rst")
_MD_PYTHON_FENCE = re.compile(r"^\s*```\s*(?:python|py|ipython3?)\s*$")
_MD_FENCE_END = re.compile(r"^\s*```\s*$")
_RST_PYTHON_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)\.\. (?:code-block|code|sourcecode)::\s*(?:python|py|ipython3?)\s*$"
)
_RST_OPTION = re.compile(r"^\s*:[\w-]+:")


def _user_material_files(repo: Path) -> list[str]:
    """Return the user-material files under ``repo``, relative and sorted.

    In a git checkout only tracked files count, so build output and
    scratch files never enter the scan.
    """
    if (repo / ".git").exists():
        listed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--", *_USER_MATERIAL], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    else:
        listed = [
            path.relative_to(repo).as_posix()
            for root in _USER_MATERIAL
            for path in ([repo / root] if (repo / root).is_file() else sorted((repo / root).rglob("*")))
            if path.is_file()
        ]
    return sorted(
        name
        for name in listed
        if name.endswith(_USER_MATERIAL_SUFFIXES) and not name.startswith(_CONTRIBUTOR_DOCS) and (repo / name).is_file()
    )


def _markdown_python(text: str) -> Iterator[str]:
    """Yield the body of every Python code fence in a Markdown text."""
    block = None
    for line in text.splitlines():
        if block is None:
            if _MD_PYTHON_FENCE.match(line):
                block = []
        elif _MD_FENCE_END.match(line):
            yield textwrap.dedent("\n".join(block))
            block = None
        else:
            block.append(line)


def _rst_python(text: str) -> Iterator[str]:
    """Yield the body of every Python ``code-block`` in a reStructuredText text."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = _RST_PYTHON_BLOCK.match(lines[index])
        index += 1
        if match is None:
            continue
        indent = len(match["indent"])
        block = []
        while index < len(lines) and (
            not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent
        ):
            if not _RST_OPTION.match(lines[index]):
                block.append(lines[index])
            index += 1
        yield textwrap.dedent("\n".join(block))


def _notebook_python(text: str) -> Iterator[str]:
    """Yield each code cell of a notebook, without IPython magics and shell lines."""
    for cell in json.loads(text).get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        lines = ("".join(source) if isinstance(source, list) else source).splitlines()
        if lines and lines[0].lstrip().startswith("%%"):
            continue  # a cell magic: the cell body is not Python
        yield "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))


def _braincell_imports(tree: ast.AST) -> Iterator[str]:
    """Yield the dotted path of every absolute ``braincell`` import in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module if alias.name == "*" else f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        yield from (name for name in names if name == "braincell" or name.startswith("braincell."))


def user_material_imports(repo: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Scan the code users read for its ``braincell`` imports.

    The material is ``examples/``, ``docs/`` without ``docs/developer/``, and
    ``README.md``. The code is every ``.py`` file, every notebook code cell
    (IPython magics and ``!`` lines dropped), and every Python code fence in
    ``.md`` and ``.rst`` files.

    Returns
    -------
    imports : list of (str, str)
        ``(file, dotted path)`` for each import, in file order;
        ``from braincell.x import *`` gives ``braincell.x``.
    unparsable : list of str
        Files with a piece of code that does not parse; their remaining
        pieces are still scanned.
    """
    repo = Path(repo)
    imports, unparsable = [], []
    for name in _user_material_files(repo):
        text = (repo / name).read_text(encoding="utf-8")
        if name.endswith(".py"):
            sources = [text]
        elif name.endswith(".ipynb"):
            sources = _notebook_python(text)
        elif name.endswith(".md"):
            sources = _markdown_python(text)
        else:
            sources = _rst_python(text)
        for source in sources:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                if name not in unparsable:
                    unparsable.append(name)
                continue
            imports.extend((name, path) for path in _braincell_imports(tree))
    return imports, unparsable


class UserMaterialImportTests:
    """Assertions for the guard on the ``braincell`` imports in user material.

    Mix into a :class:`unittest.TestCase` subclass that sets ``repo``, the
    repository root, which :func:`user_material_imports` scans. Every import
    must be public (:func:`is_public_import`) or match an ``allowlist`` entry
    ``(file, dotted path, reason)``. Every file must parse or match an
    ``unparsable`` entry ``(file, reason)``. Every entry must match something
    and give a reason. The tests skip when ``repo`` has no ``examples/``, as
    for an installed package.
    """

    repo: Path | None = None
    allowlist: tuple[tuple[str, str, str], ...] = ()
    unparsable: tuple[tuple[str, str], ...] = ()

    def _scan(self) -> tuple[list[tuple[str, str]], list[str]]:
        if self.repo is None or not (Path(self.repo) / "examples").is_dir():
            raise unittest.SkipTest("no examples/ beside the package; nothing to scan")
        if "_scanned" not in type(self).__dict__:
            type(self)._scanned = user_material_imports(Path(self.repo))
        return type(self)._scanned

    def test_user_material_imports_only_public_paths(self):
        imports, _ = self._scan()
        public = public_paths()
        listed = {entry[:2] for entry in self.allowlist}
        bad = sorted(
            {
                f"{file}: {path}"
                for file, path in imports
                if (file, path) not in listed and not is_public_import(path, public)
            }
        )
        self.assertEqual(
            bad,
            [],
            "user material imports braincell only by public paths: braincell, a name in braincell.__all__, a stable "
            "package or a name in its __all__, or braincell.experimental. A submodule path such as "
            "braincell.morph.morphology is internal. Import the public name, or allowlist the import with a reason.",
        )

    def test_user_material_parses(self):
        _, unparsable = self._scan()
        exempt = {entry[0] for entry in self.unparsable}
        bad = [file for file in unparsable if file not in exempt]
        self.assertEqual(bad, [], "these files hold code that does not parse, so their imports cannot be checked")

    def test_user_material_allowlists_have_no_dead_entries(self):
        imports, unparsable = self._scan()
        live = set(imports)
        dead = [entry[:2] for entry in self.allowlist if entry[:2] not in live]
        dead += [entry[:1] for entry in self.unparsable if entry[0] not in unparsable]
        self.assertEqual(dead, [], f"allowlist entries that match nothing: {dead}")

    def test_user_material_allowlist_entries_have_reasons(self):
        bad = [entry[:2] for entry in self.allowlist if len(entry) != 3 or not entry[2].strip()]
        bad += [entry[:1] for entry in self.unparsable if len(entry) != 2 or not entry[1].strip()]
        self.assertEqual(bad, [], f"allowlist entries without a reason: {bad}")
