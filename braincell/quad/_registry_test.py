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

"""Tests for :mod:`braincell.quad._registry`.

These tests exercise the registry in isolation by constructing a fresh
:class:`IntegratorRegistry` for most cases. The integration with the global
singleton (used by ``braincell.quad.get_integrator``) is also covered.
"""

import unittest
import warnings

import braincell.quad as quad
from braincell.quad._registry import (
    IntegratorEntry,
    IntegratorRegistry,
)


def _noop(target, *args, **kwargs):  # pragma: no cover - body irrelevant
    return None


class IntegratorRegistryTest(unittest.TestCase):

    def setUp(self):
        self.registry = IntegratorRegistry()

    # ------------------------------------------------------------------ #
    # basic registration / lookup
    # ------------------------------------------------------------------ #
    def test_register_and_lookup(self):
        entry = self.registry.register("foo", _noop, category="explicit", order=1)
        self.assertIsInstance(entry, IntegratorEntry)
        self.assertEqual(entry.name, "foo")
        self.assertIs(self.registry["foo"], _noop)
        self.assertIn("foo", self.registry)
        self.assertEqual(self.registry.names(), ["foo"])
        self.assertEqual(self.registry.entry("foo").order, 1)

    def test_aliases_resolve_to_canonical(self):
        self.registry.register("foo", _noop, aliases=("bar", "baz"))
        self.assertIs(self.registry["bar"], _noop)
        self.assertIs(self.registry["baz"], _noop)
        self.assertEqual(self.registry.resolve("bar"), "foo")
        self.assertEqual(self.registry.resolve("foo"), "foo")
        self.assertIn("bar", self.registry)

    def test_get_returns_default_on_miss(self):
        sentinel = object()
        self.assertIs(self.registry.get("missing", sentinel), sentinel)
        self.assertIsNone(self.registry.get("missing"))

    def test_module_field_populated(self):
        self.registry.register("foo", _noop)
        self.assertEqual(self.registry.entry("foo").module, _noop.__module__)

    # ------------------------------------------------------------------ #
    # collisions
    # ------------------------------------------------------------------ #
    def test_duplicate_canonical_name_rejected(self):
        self.registry.register("foo", _noop)
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register("foo", _noop)

    def test_alias_collides_with_canonical_name(self):
        self.registry.register("foo", _noop)
        with self.assertRaisesRegex(ValueError, "canonical integrator"):
            self.registry.register("bar", _noop, aliases=("foo",))

    def test_alias_collides_with_other_alias(self):
        self.registry.register("foo", _noop, aliases=("shared",))
        with self.assertRaisesRegex(ValueError, "already.*alias"):
            self.registry.register("baz", _noop, aliases=("shared",))

    def test_canonical_collides_with_existing_alias(self):
        self.registry.register("foo", _noop, aliases=("shared",))
        with self.assertRaisesRegex(ValueError, "already an alias"):
            self.registry.register("shared", _noop)

    # ------------------------------------------------------------------ #
    # override + unregister
    # ------------------------------------------------------------------ #
    def test_override_emits_runtime_warning(self):
        self.registry.register("foo", _noop, aliases=("bar",))

        def replacement(target, *a, **k):
            return "replacement"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.registry.register("foo", replacement, override=True)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertIs(self.registry["foo"], replacement)
        # Old aliases should be cleared.
        self.assertNotIn("bar", self.registry)

    def test_unregister_removes_aliases(self):
        self.registry.register("foo", _noop, aliases=("bar",))
        self.registry.unregister("foo")
        self.assertNotIn("foo", self.registry)
        self.assertNotIn("bar", self.registry)

    def test_unregister_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.registry.unregister("missing")

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def test_register_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            self.registry.register("foo", "not callable")  # type: ignore[arg-type]

    def test_register_rejects_empty_name(self):
        with self.assertRaises(TypeError):
            self.registry.register("", _noop)

    def test_register_rejects_empty_alias(self):
        with self.assertRaises(TypeError):
            self.registry.register("foo", _noop, aliases=("",))

    # ------------------------------------------------------------------ #
    # deprecation + suggestions
    # ------------------------------------------------------------------ #
    def test_deprecation_warning_emitted_once(self):
        self.registry.register("foo", _noop, deprecated=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = self.registry["foo"]
            _ = self.registry["foo"]
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecation_warnings), 1)

    def test_suggest_close_match(self):
        self.registry.register("euler", _noop)
        self.registry.register("midpoint", _noop)
        self.assertEqual(self.registry.suggest("eler", n=1), ["euler"])
        self.assertEqual(self.registry.suggest("nothinglikeit"), [])

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def test_by_category(self):
        self.registry.register("a", _noop, category="explicit")
        self.registry.register("b", _noop, category="implicit")
        self.registry.register("c", _noop, category="explicit")
        explicit_names = [e.name for e in self.registry.by_category("explicit")]
        self.assertEqual(explicit_names, ["a", "c"])

    def test_as_dict_includes_aliases_by_default(self):
        self.registry.register("foo", _noop, aliases=("bar",))
        flat = self.registry.as_dict()
        self.assertEqual(set(flat), {"foo", "bar"})
        self.assertIs(flat["bar"], _noop)
        only_canonical = self.registry.as_dict(include_aliases=False)
        self.assertEqual(set(only_canonical), {"foo"})

    def test_names_with_and_without_aliases(self):
        self.registry.register("foo", _noop, aliases=("bar",))
        self.assertEqual(self.registry.names(), ["foo"])
        self.assertEqual(
            self.registry.names(include_aliases=True), ["bar", "foo"]
        )


class GlobalRegistryIntegrationTest(unittest.TestCase):
    """Tests against the singleton populated by importing braincell.quad."""

    def test_get_integrator_returns_callable(self):
        func = quad.get_integrator("euler")
        self.assertTrue(callable(func))
        self.assertIs(func, quad.euler_step)

    def test_explicit_alias_resolves_to_euler(self):
        self.assertIs(
            quad.get_integrator("explicit"),
            quad.get_integrator("euler"),
        )

    def test_stagger_alias_resolves_to_staggered(self):
        self.assertIs(
            quad.get_integrator("stagger"),
            quad.get_integrator("staggered"),
        )

    def test_get_integrator_passes_callable_through(self):
        def custom(target, *args):
            return None

        self.assertIs(quad.get_integrator(custom), custom)

    def test_unknown_integrator_raises_with_suggestion(self):
        with self.assertRaises(ValueError) as ctx:
            quad.get_integrator("eler")
        # Either the suggestion or the available list should be present.
        self.assertIn("euler", str(ctx.exception))

    def test_unknown_integrator_no_suggestion(self):
        with self.assertRaises(ValueError) as ctx:
            quad.get_integrator("totally_unknown_xyz")
        self.assertIn("Available", str(ctx.exception))

    def test_get_integrator_rejects_non_string_non_callable(self):
        with self.assertRaises(TypeError):
            quad.get_integrator(42)  # type: ignore[arg-type]

    def test_all_integrators_view_in_sync(self):
        # Legacy mapping must reflect what the registry holds.
        self.assertIn("euler", quad.all_integrators)
        self.assertIn("rk4", quad.all_integrators)
        self.assertIn("staggered", quad.all_integrators)
        self.assertIs(quad.all_integrators["euler"], quad.euler_step)
        # Aliases are exposed too.
        self.assertIn("explicit", quad.all_integrators)
        self.assertIn("stagger", quad.all_integrators)

    def test_registered_canonical_names_include_known_methods(self):
        names = set(quad.get_registry().names())
        expected = {
            "euler",
            "midpoint",
            "rk2",
            "rk3",
            "rk4",
            "heun2",
            "heun3",
            "ssprk3",
            "ralston2",
            "ralston3",
            "ralston4",
            "exp_euler",
            "ind_exp_euler",
            "backward_euler",
            "implicit_euler",
            "splitting",
            "implicit_rk4",
            "implicit_exp_euler",
            "cn_rk4",
            "cn_exp_euler",
            "exp_exp_euler",
            "staggered",
            "dhs_voltage",
        }
        missing = expected - names
        self.assertFalse(missing, f"Missing canonical names: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main()
