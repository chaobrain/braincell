"""Test-only helpers for the docstring conformance guards.

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
        found[match.group("title").rstrip()] = doc[match.end():end]
    return found


def has_citation(doc: str) -> bool:
    """True when ``doc`` has a References section holding a ``.. [n]`` entry."""
    body = sections(doc).get("References")
    return bool(body) and bool(_CITATION.search(body))


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
        missing = [
            f"{mod}.{name}"
            for mod, name, obj in self._symbols()
            if own_docstring(obj) is None
        ]
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
