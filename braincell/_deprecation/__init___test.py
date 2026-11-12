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

"""Tests for the deprecation helper, and the library-wide deprecation checks (API-7)."""

import ast
import contextlib
import dataclasses
import io
import os
import re
import runpy
import subprocess
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import brainstate

import braincell
from braincell import _deprecation as dep
from braincell._deprecation import _PACKAGE_DIR, Deprecation, deprecated_attributes, removed_attributes, removed_member

#: The deprecations that must be registered once every module is imported.
_KNOWN = (
    "Constructing an invalid mechanism declaration",
    "Directly constructing a Channel or Ion with wrongly united parameters",
)
_CHANGELOG = Path(_PACKAGE_DIR).parent / "changelog.md"


def _line() -> int:
    """The caller's current line: ``line, result = _line(), call()`` pairs a call with its line."""
    return sys._getframe(1).f_lineno


def _old(name: str = "braincell.Old", **fields) -> Deprecation:
    """A test deprecation; build it inside a test, whose registry is restored afterwards."""
    return Deprecation(name, **{"since": "0.3", "removed_in": "0.4", "replacement": "braincell.New", **fields})


def _as_braincell_code(source: str, namespace: dict) -> dict:
    """Run ``source`` as if it were a non-test module of the braincell package."""
    exec(compile(source, os.path.join(_PACKAGE_DIR, "fake.py"), "exec"), namespace)
    return namespace


@contextlib.contextmanager
def _recorded():
    """Record every warning, each one shown (``always``)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _places(caught, category=DeprecationWarning) -> list:
    """``(filename, lineno)`` of each recorded warning of exactly ``category``."""
    return [(w.filename, w.lineno) for w in caught if w.category is category]


@contextlib.contextmanager
def _temporary_package(name: str, getattr_function):
    """A package ``name`` (``__path__ = []``) in ``sys.modules`` whose ``__getattr__`` is given."""
    package = types.ModuleType(name)
    package.__path__ = []
    package.__getattr__ = getattr_function
    sys.modules[name] = package
    try:
        yield package
    finally:
        del sys.modules[name]


class _RegistryTest(unittest.TestCase):
    """Restores the registry after each test, so test deprecations do not leak."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(dep._REGISTRY)
        patcher.start()
        self.addCleanup(patcher.stop)


class CategoryAndMessageTest(_RegistryTest):
    def test_default_category_is_deprecation_warning(self) -> None:
        old = _old()
        with self.assertWarns(DeprecationWarning):
            old.warn()

    def test_future_warning_category(self) -> None:
        old = _old("braincell.f(x=...)", category=FutureWarning)
        with _recorded() as caught:
            old.warn()
        self.assertEqual([w.category for w in caught], [FutureWarning])

    def test_message_format(self) -> None:
        old = Deprecation("braincell.X", since="0.3", removed_in="0.4", replacement="Y")
        self.assertEqual(
            old.message,
            "braincell.X is deprecated since braincell 0.3 and will be removed in braincell 0.4. Use Y instead.",
        )
        noted = _old("braincell.Z", note="It never worked.")
        self.assertTrue(noted.message.endswith("Use braincell.New instead. It never worked."))
        bare = _old("braincell.W", replacement=None, note="Delete the call.")
        self.assertTrue(bare.message.endswith("will be removed in braincell 0.4. Delete the call."))

    def test_custom_message_is_issued_verbatim(self) -> None:
        check = _old("A check", category=FutureWarning, replacement=None, note="Fix it.")
        with self.assertWarns(FutureWarning) as cm:
            check.warn("p got 1 mV. This will raise ValueError at declaration from braincell 0.4.")
        self.assertEqual(str(cm.warning), "p got 1 mV. This will raise ValueError at declaration from braincell 0.4.")


class AttributionTest(_RegistryTest):
    """Each shape warns once, at the line in this file that used the deprecated API."""

    def setUp(self) -> None:
        super().setUp()
        self.old = _old()

    def assertWarnedOnceAt(self, caught, line: int) -> None:
        self.assertEqual(_places(caught), [(__file__, line)])

    def test_function_call(self) -> None:
        old = self.old

        def deprecated():
            old.warn()
            return 1

        with _recorded() as caught:
            line, result = _line(), deprecated()
        self.assertWarnedOnceAt(caught, line)
        self.assertEqual(result, 1)

    def test_node_construction_through_the_metaclass(self) -> None:
        old = self.old

        class Deprecated(brainstate.graph.Node):
            def __init__(self):
                old.warn()
                super().__init__()

        with _recorded() as caught:
            line, _ = _line(), Deprecated()
        self.assertWarnedOnceAt(caught, line)

    def test_user_subclass_warns_at_its_super_init_line(self) -> None:
        old = self.old

        class Deprecated(brainstate.graph.Node):
            def __init__(self):
                old.warn()
                super().__init__()

        class UserNode(Deprecated):
            def __init__(self):
                super().__init__()

        with _recorded() as caught:
            UserNode()
        self.assertWarnedOnceAt(caught, UserNode.__init__.__code__.co_firstlineno + 1)

    def test_frozen_dataclass_post_init(self) -> None:
        old = self.old

        @dataclasses.dataclass(frozen=True)
        class Deprecated:
            x: int = 0

            def __post_init__(self):
                old.warn()

        with _recorded() as caught:
            line, _ = _line(), Deprecated(x=1)
        self.assertWarnedOnceAt(caught, line)

    def test_module_attribute_access(self) -> None:
        module = types.ModuleType("_api7_attribute_module")
        module.__getattr__ = deprecated_attributes(module.__name__, {"Old": (self.old, int)})
        with _recorded() as caught:
            line, value = _line(), module.Old
        self.assertWarnedOnceAt(caught, line)
        self.assertIs(value, int)
        with self.assertRaisesRegex(AttributeError, "has no attribute 'missing'"):
            module.missing

    def test_from_import(self) -> None:
        getattr_function = deprecated_attributes("_api7_package", {"Old": (self.old, int)})
        with _temporary_package("_api7_package", getattr_function), _recorded() as caught:
            line = _line() + 1
            from _api7_package import Old
        self.assertWarnedOnceAt(caught, line)
        self.assertIs(Old, int)

    def test_deep_check_at_user_goes_to_the_nearest_user_frame(self) -> None:
        check = _old("A check", category=FutureWarning, replacement=None, note="Fix it.")
        namespace = _as_braincell_code(
            "def declare():\n    check.warn('Bad. This will raise ValueError from braincell 0.4.', at='user')\n",
            {"check": check},
        )
        with _recorded() as caught:
            line, _ = _line(), namespace["declare"]()
        self.assertEqual(_places(caught, FutureWarning), [(__file__, line)])

    def test_without_frames_the_warning_goes_to_the_deprecated_callables_caller(self) -> None:
        old = self.old

        def deprecated():
            old.warn()

        for getframe in (None, lambda depth: None):
            line = _line() + 3  # ``sys._getframe`` is patched below
            with self.subTest(getframe=getframe), mock.patch.object(sys, "_getframe", getframe):
                with _recorded() as caught:
                    deprecated()
                self.assertWarnedOnceAt(caught, line)

    def test_without_frames_a_module_attribute_still_warns(self) -> None:
        module = types.ModuleType("_api7_no_frames")
        module.__getattr__ = deprecated_attributes(module.__name__, {"Old": (self.old, int)})
        with mock.patch.object(sys, "_getframe", None), _recorded() as caught:
            self.assertIs(module.Old, int)
        self.assertEqual(len(_places(caught)), 1)

    def test_when_every_frame_is_braincells_the_warning_goes_to_the_caller(self) -> None:
        old = self.old

        def deprecated():
            old.warn(at="user")

        with mock.patch.object(dep, "_is_internal", return_value=True), _recorded() as caught:
            line, _ = _line(), deprecated()
        self.assertWarnedOnceAt(caught, line)

    def test_under_an_error_filter_the_use_raises(self) -> None:
        old = self.old

        def deprecated():
            old.warn()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with self.assertRaisesRegex(DeprecationWarning, "braincell.Old is deprecated since braincell 0.3"):
                deprecated()


class InternalUseTest(_RegistryTest):
    """braincell's own uses do not warn; the same calls from a user's file do."""

    def setUp(self) -> None:
        super().setUp()
        old = _old()

        def deprecated_function():
            old.warn()

        class DeprecatedNode(brainstate.graph.Node):
            def __init__(self):
                old.warn()
                super().__init__()

        @dataclasses.dataclass(frozen=True)
        class DeprecatedDeclaration:
            name: str | None = None

            def __post_init__(self):
                old.warn()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            declaration = DeprecatedDeclaration()
        self.calls = {
            "function": deprecated_function,
            "class": DeprecatedNode,
            "replace": lambda: dataclasses.replace(declaration, name="auto"),
        }
        self.internal = {
            "function": "deprecated_function()",
            "class": "DeprecatedNode()",
            "replace": "dataclasses.replace(declaration, name='auto')",
        }
        self.namespace = {
            "dataclasses": dataclasses,
            "deprecated_function": deprecated_function,
            "DeprecatedNode": DeprecatedNode,
            "declaration": declaration,
        }

    def test_braincell_code_gets_no_warning(self) -> None:
        for shape, source in self.internal.items():
            with self.subTest(shape=shape), _recorded() as caught:
                _as_braincell_code(source, dict(self.namespace))
            self.assertEqual(_places(caught), [], shape)

    def test_user_code_gets_one_warning(self) -> None:
        for shape, call in self.calls.items():
            with self.subTest(shape=shape), _recorded() as caught:
                call()
            self.assertEqual(len(_places(caught)), 1, shape)


class CountTest(_RegistryTest):
    """Every use warns; the warning filters decide what is shown."""

    def setUp(self) -> None:
        super().setUp()
        old = self.old = _old()

        def deprecated():
            old.warn()

        self.deprecated = deprecated

    def test_two_calls_on_one_line_give_two_warnings(self) -> None:
        with _recorded() as caught:
            line, _ = _line(), [self.deprecated() for _ in range(2)]
        self.assertEqual(_places(caught), [(__file__, line)] * 2)

    def test_from_import_gives_one_warning(self) -> None:
        getattr_function = deprecated_attributes("_api7_package", {"Old": (self.old, int)})
        with _temporary_package("_api7_package", getattr_function), _recorded() as caught:
            from _api7_package import Old  # noqa: F401
        self.assertEqual(len(_places(caught)), 1)

    def test_a_hand_written_getattr_warns_twice_on_from_import(self) -> None:
        # Control: importlib's ``_handle_fromlist`` and ``IMPORT_FROM`` both look the name up.
        def __getattr__(name):
            warnings.warn("Old is deprecated", DeprecationWarning, stacklevel=2)
            return int

        with _temporary_package("_api7_package", __getattr__), _recorded() as caught:
            from _api7_package import Old  # noqa: F401
        self.assertEqual(len(_places(caught)), 2)

    def test_default_filter_shows_one_warning_per_line(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                self.deprecated()
        self.assertEqual(len(_places(caught)), 1)


class InvalidDeprecationTest(_RegistryTest):
    def test_construction_rejects_bad_fields(self) -> None:
        cases = {
            "removed_in == since": (ValueError, dict(removed_in="0.3")),
            "removed_in < since": (ValueError, dict(since="0.4", removed_in="0.3")),
            "three-part since": (ValueError, dict(since="0.3.0")),
            "non-string removed_in": (ValueError, dict(removed_in=0.4)),
            "replacement=None without a note": (ValueError, dict(replacement=None)),
            "UserWarning": (TypeError, dict(category=UserWarning)),
            "subclass of DeprecationWarning": (TypeError, dict(category=PendingDeprecationWarning)),
        }
        for label, (error, fields) in cases.items():
            with self.subTest(label), self.assertRaises(error):
                _old(f"braincell.{label}", **fields)
        with self.assertRaises(TypeError):
            Deprecation("braincell.no_replacement", since="0.3", removed_in="0.4")
        with self.assertRaises(TypeError):
            Deprecation("", since="0.3", removed_in="0.4", replacement="x")

    def test_a_name_registers_once(self) -> None:
        first = _old("braincell.Twice")
        self.assertIs(dep._REGISTRY["braincell.Twice"], first)
        _old("braincell.Twice")  # the same fields again, as on a module reload
        self.assertIs(dep._REGISTRY["braincell.Twice"], first)
        with self.assertRaisesRegex(ValueError, "already registered with different fields"):
            _old("braincell.Twice", removed_in="0.5")

    def test_warn_rejects_a_bad_place_or_a_message_without_the_removal(self) -> None:
        old = _old()
        with self.assertRaisesRegex(ValueError, "braincell 0.4"):
            old.warn("Old is going away.")
        with self.assertRaisesRegex(ValueError, "at must be 'caller' or 'user'"):
            old.warn(at="x")

    def test_deprecated_attributes_rejects_a_value_without_its_deprecation(self) -> None:
        with self.assertRaisesRegex(TypeError, "expected a \\(Deprecation, value\\) pair"):
            deprecated_attributes("m", {"Old": int})


class RemovedAttributesTest(unittest.TestCase):
    """A removed name raises with its message: ``AttributeError``, or ``ImportError`` on a ``from`` import."""

    _MESSAGE = "_removed_package.Old was removed in braincell 0.3. Use _removed_package.New."

    def _package(self):
        return _temporary_package("_removed_package", removed_attributes("_removed_package", {"Old": self._MESSAGE}))

    def test_attribute_access_raises_attribute_error_with_the_message(self) -> None:
        with self._package() as package:
            with self.assertRaises(AttributeError) as caught:
                package.Old
            self.assertEqual(str(caught.exception), self._MESSAGE)
            self.assertFalse(hasattr(package, "Old"))
            self.assertIsNone(getattr(package, "Old", None))

    def test_from_import_raises_import_error_with_the_message(self) -> None:
        # Python replaces an AttributeError from a module ``__getattr__`` on
        # ``from M import X`` by its own "cannot import name"; the ImportError
        # raised there keeps the message.
        with self._package():
            with self.assertRaises(ImportError) as caught:
                from _removed_package import Old  # noqa: F401
        self.assertEqual(str(caught.exception), self._MESSAGE)
        self.assertEqual(caught.exception.name, "_removed_package")

    def test_other_names_are_missing_as_usual(self) -> None:
        with self._package() as package:
            with self.assertRaisesRegex(AttributeError, "module '_removed_package' has no attribute 'Other'"):
                package.Other
            with self.assertRaisesRegex(ImportError, "cannot import name 'Other'"):
                from _removed_package import Other  # noqa: F401

    def test_without_frames_a_from_import_still_fails(self) -> None:
        # The message is lost to Python's generic ImportError, the name is not.
        for getframe in (None, lambda depth: None):
            with self.subTest(getframe=getframe), self._package():
                with mock.patch.object(sys, "_getframe", getframe):
                    with self.assertRaisesRegex(ImportError, "cannot import name 'Old'"):
                        from _removed_package import Old  # noqa: F401

    def test_a_message_is_required(self) -> None:
        for message in ("", None, 3):
            with self.subTest(message=message), self.assertRaisesRegex(TypeError, "expected a non-empty message"):
                removed_attributes("_removed_package", {"Old": message})


class RemovedMemberTest(unittest.TestCase):
    """A removed method raises ``AttributeError`` with its message, on the class and on instances."""

    _MESSAGE = "Owner.old() was removed in braincell 0.3. Use Owner.new()."

    def _owner(self):
        class Owner:
            old = removed_member(self._MESSAGE)

            def new(self):
                return "new"

        return Owner

    def test_access_raises_attribute_error_with_the_message(self) -> None:
        owner = self._owner()
        for target in (owner, owner()):
            with self.subTest(target=target):
                with self.assertRaises(AttributeError) as caught:
                    target.old
                self.assertEqual(str(caught.exception), self._MESSAGE)
                self.assertFalse(hasattr(target, "old"))
                self.assertIsNone(getattr(target, "old", None))

    def test_assignment_and_deletion_on_an_instance_raise_too(self) -> None:
        instance = self._owner()()
        with self.assertRaisesRegex(AttributeError, "^Owner\\.old\\(\\) was removed"):
            instance.old = 1
        with self.assertRaisesRegex(AttributeError, "^Owner\\.old\\(\\) was removed"):
            del instance.old

    def test_other_members_are_unaffected(self) -> None:
        self.assertEqual(self._owner()().new(), "new")

    def test_a_message_is_required(self) -> None:
        for message in ("", None, 3):
            with self.subTest(message=message), self.assertRaisesRegex(TypeError, "expected a non-empty message"):
                removed_member(message)


class ExpiryTest(unittest.TestCase):
    def test_an_item_is_due_from_its_removal_release(self) -> None:
        before = dep.registered()
        with mock.patch.dict(dep._REGISTRY):
            due = Deprecation("braincell.Due", since="0.2", removed_in="0.3", replacement="braincell.New")
            self.assertIn(due, dep.expired((0, 3)))
            self.assertNotIn(due, dep.expired((0, 2)))
            with mock.patch("braincell._version.__version__", "0.3.0"):
                self.assertIn(due, dep.expired())
        self.assertEqual(dep.registered(), before)


def _changelog_deprecations(changelog: str, since: tuple[int, int], next_minor: tuple[int, int]) -> str:
    """Text of the ``### Deprecations`` subsections of the changelog section(s) for ``since``."""
    if since == next_minor:
        heading = re.compile(r"Unreleased\s*$")
    else:
        heading = re.compile(rf"Version {since[0]}\.{since[1]}\.\d+\b")
    text = []
    for section in re.split(r"(?m)^## ", changelog)[1:]:
        title, _, body = section.partition("\n")
        if heading.match(title.strip()):
            match = re.search(r"(?ms)^### Deprecations\s*$(.*?)(?=^### |\Z)", body)
            if match:
                text.append(match[1])
    return "\n".join(text)


class LibraryDeprecationsTest(unittest.TestCase):
    """Every braincell deprecation is announced in the changelog, and none outlives its removal release."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.modules = dep.import_all()
        cls.items = dep.registered()
        cls.current = dep._release_minor()
        cls.next_minor = (cls.current[0], cls.current[1] + 1)

    def test_every_module_imports_and_the_known_deprecations_register(self) -> None:
        self.assertGreaterEqual(len(self.modules), 150, f"only {len(self.modules)} braincell modules found")
        names = [item.name for item in self.items]
        self.assertGreaterEqual(len(names), len(_KNOWN), f"{len(names)} deprecations registered: {names}")
        self.assertEqual(sorted(set(_KNOWN) - set(names)), [], f"checked {len(names)} registered deprecations")

    def test_no_deprecation_is_due_by_the_current_release(self) -> None:
        version = f"{self.current[0]}.{self.current[1]}"
        due = [item for item in self.items if dep._minor(item.removed_in) <= self.current]
        self.assertEqual(
            [f"{item.name} (removed_in={item.removed_in})" for item in due],
            [],
            f"checked {len(self.items)} deprecations against braincell {version}: remove each due one "
            "(a Breaking Changes entry in changelog.md), or postpone it (a later removed_in and a "
            "Deprecations entry).",
        )

    def test_no_deprecation_starts_after_the_next_release(self) -> None:
        early = [f"{item.name} (since={item.since})" for item in self.items if dep._minor(item.since) > self.next_minor]
        self.assertEqual(
            early, [], f"checked {len(self.items)} deprecations; since cannot be later than the next minor"
        )

    def test_the_changelog_announces_each_deprecation(self) -> None:
        if not _CHANGELOG.is_file():
            self.skipTest("changelog.md is not beside the package (an installed braincell)")
        changelog = _CHANGELOG.read_text(encoding="utf-8")
        missing = [
            f"{item.name} (since {item.since})"
            for item in self.items
            if item.name not in _changelog_deprecations(changelog, dep._minor(item.since), self.next_minor)
        ]
        self.assertEqual(
            missing,
            [],
            f"checked {len(self.items)} deprecations: each name must appear verbatim under '### Deprecations' "
            "of the changelog section of its since release ('## Unreleased' for the next one)",
        )


class ChangelogSectionTest(unittest.TestCase):
    def test_sections_are_matched_by_release(self) -> None:
        changelog = (
            "# Notes\n\n## Unreleased\n\n### Deprecations\n\n- **`a`** new\n\n### Bug Fixes\n\n- `b` fixed\n\n"
            "## Version 0.2.1\n\n### Deprecations\n\n- **`c`** patch\n\n"
            "## Version 0.2.0\n\n### Breaking Changes\n\n- `d`\n\n### Deprecations\n\n- **`e`** old\n\n"
            "## Version 0.12.0\n\n### Deprecations\n\n- **`f`** other\n"
        )
        unreleased = _changelog_deprecations(changelog, (0, 3), (0, 3))
        self.assertIn("`a`", unreleased)
        self.assertNotIn("`b`", unreleased)
        released = _changelog_deprecations(changelog, (0, 2), (0, 3))
        self.assertEqual([name for name in "abcdef" if f"`{name}`" in released], ["c", "e"])
        self.assertEqual(_changelog_deprecations(changelog, (0, 1), (0, 3)), "")


class CommandLineTest(unittest.TestCase):
    """``python -m braincell._deprecation MAJOR.MINOR``, run before a release."""

    def _run(self, version: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(Path(_PACKAGE_DIR).parent), JAX_PLATFORMS="cpu")
        return subprocess.run(
            [sys.executable, "-m", "braincell._deprecation", version],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )

    def test_lists_the_items_due_and_fails(self) -> None:
        result = self._run("0.4")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for name in _KNOWN:
            self.assertIn(f"  {name} (since 0.3, removed in 0.4)", result.stdout)
        self.assertNotIn("RuntimeWarning", result.stderr)

    def test_passes_when_nothing_is_due(self) -> None:
        result = self._run("0.3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No deprecation is due by braincell 0.3", result.stdout)

    def test_main_in_process(self) -> None:
        for argv, code in ((["0.3"], 0), (["0.4"], 1)):
            with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dep.main(argv), code)
        with mock.patch.object(sys, "argv", ["braincell._deprecation", "0.3"]):
            with self.assertRaises(SystemExit) as exit_, contextlib.redirect_stdout(io.StringIO()):
                runpy.run_module("braincell._deprecation", run_name="__main__")
        self.assertEqual(exit_.exception.code, 0)

    def test_a_version_must_be_major_minor(self) -> None:
        with self.assertRaises(SystemExit) as exit_, contextlib.redirect_stderr(io.StringIO()) as stderr:
            dep.main(["0.3.0"])
        self.assertEqual(exit_.exception.code, 2)
        self.assertIn("'MAJOR.MINOR'", stderr.getvalue())


class ModuleScopeTest(unittest.TestCase):
    def test_module_scope_imports_only_the_standard_library(self) -> None:
        # Every braincell package imports this module (``braincell.mech`` included, a leaf
        # otherwise), so it must not import braincell back at module scope.
        imported = []
        for path in sorted(Path(dep.__file__).parent.glob("*.py")):
            if path.name.endswith("_test.py"):
                continue
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Import):
                    imported += [(path.name, alias.name) for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported.append((path.name, node.module))
        allowed = sys.stdlib_module_names | {"__future__"}
        outside = [(name, module) for name, module in imported if module.split(".")[0] not in allowed]
        # ``__main__.py`` runs the command line; nothing imports it.
        self.assertEqual(outside, [("__main__.py", "braincell._deprecation")], f"checked {len(imported)} imports")


class UserFrameTest(unittest.TestCase):
    def test_trampolines_are_skipped(self) -> None:
        frames = []

        @dataclasses.dataclass
        class Built:
            def __post_init__(self):
                frames.append(dep.user_frame(sys._getframe(1)))

        Built()
        frame, hops = frames[0]
        self.assertEqual(frame.f_code.co_filename, __file__)
        self.assertEqual(hops, 1)  # the dataclass-generated ``__init__``

    def test_the_package_directory_is_braincells(self) -> None:
        self.assertEqual(_PACKAGE_DIR, os.path.dirname(os.path.abspath(braincell.__file__)) + os.sep)


if __name__ == "__main__":
    unittest.main()
