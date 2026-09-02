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

import inspect
import re
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

    def test_no_primary_source_allowlist_has_no_dead_entries(self):
        if not self.covered_modules:
            self.skipTest("no modules covered yet")
        live = {name for _, name, _ in self._symbols()}
        dead = sorted(n for n in self.no_primary_source if n not in live)
        self.assertEqual(dead, [], f"allowlist names no longer public: {dead}")
