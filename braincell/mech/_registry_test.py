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

import unittest

from braincell.mech import (
    MechanismEntry,
    MechanismRegistry,
    register_channel,
    register_ion,
    register_synapse,
)
from braincell.mech._registry import _REGISTRY


class _DummyChannel:
    """Stand-in for a concrete channel class in these isolated tests."""


class _DummyIon:
    pass


class _DummySynapse:
    pass


class MechanismRegistryIsolationMixin:
    """Each test gets a fresh :class:`MechanismRegistry`.

    Tests that mutate the global singleton MUST snapshot and restore
    it in addCleanup so they don't leak state into later tests that
    exercise the real builtin registrations.
    """

    def make_registry(self) -> MechanismRegistry:
        return MechanismRegistry()


class MechanismRegistryBasicsTest(
    MechanismRegistryIsolationMixin, unittest.TestCase
):
    def test_register_and_get(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="Foo", cls=_DummyChannel)
        )
        self.assertIs(reg.get("channel", "Foo"), _DummyChannel)

    def test_register_and_entry(self) -> None:
        reg = self.make_registry()
        entry = MechanismEntry(
            category="channel",
            name="Foo",
            cls=_DummyChannel,
            aliases=("foo_alias",),
        )
        reg.register(entry)
        self.assertEqual(reg.entry("channel", "Foo"), entry)
        self.assertEqual(reg.entry("channel", "foo_alias"), entry)

    def test_contains(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="ion", name="Bar", cls=_DummyIon)
        )
        self.assertTrue(reg.contains("ion", "Bar"))
        self.assertFalse(reg.contains("ion", "Baz"))
        self.assertFalse(reg.contains("channel", "Bar"))

    def test_len_and_repr(self) -> None:
        reg = self.make_registry()
        self.assertEqual(len(reg), 0)
        reg.register(
            MechanismEntry(category="channel", name="A", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="ion", name="B", cls=_DummyIon)
        )
        self.assertEqual(len(reg), 2)
        self.assertIn("total=2", repr(reg))

    def test_duplicate_canonical_raises(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="Foo", cls=_DummyChannel)
        )
        with self.assertRaises(ValueError):
            reg.register(
                MechanismEntry(
                    category="channel", name="Foo", cls=_DummyChannel
                )
            )

    def test_alias_collision_with_canonical(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="Foo", cls=_DummyChannel)
        )
        with self.assertRaises(ValueError):
            reg.register(
                MechanismEntry(
                    category="channel",
                    name="Bar",
                    cls=_DummyChannel,
                    aliases=("Foo",),
                )
            )
        # First entry must still be intact.
        self.assertTrue(reg.contains("channel", "Foo"))
        self.assertFalse(reg.contains("channel", "Bar"))

    def test_alias_collision_with_alias(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(
                category="channel",
                name="Foo",
                cls=_DummyChannel,
                aliases=("leaky",),
            )
        )
        with self.assertRaises(ValueError):
            reg.register(
                MechanismEntry(
                    category="channel",
                    name="Bar",
                    cls=_DummyChannel,
                    aliases=("leaky",),
                )
            )

    def test_categories_are_independent(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="X", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="ion", name="X", cls=_DummyIon)
        )
        self.assertIs(reg.get("channel", "X"), _DummyChannel)
        self.assertIs(reg.get("ion", "X"), _DummyIon)

    def test_invalid_category_raises(self) -> None:
        reg = self.make_registry()
        with self.assertRaises(ValueError):
            reg.register(
                MechanismEntry(
                    category="whatever", name="X", cls=_DummyChannel
                )
            )

    def test_unknown_get_raises_with_suggestion(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(
                category="channel", name="INa_HH1952", cls=_DummyChannel
            )
        )
        with self.assertRaises(KeyError) as ctx:
            reg.get("channel", "INa_HH1951")
        self.assertIn("INa_HH1951", str(ctx.exception))
        self.assertIn("Did you mean", str(ctx.exception))


class MechanismRegistryAliasTest(
    MechanismRegistryIsolationMixin, unittest.TestCase
):
    def test_alias_resolves_to_same_class(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(
                category="channel",
                name="IL",
                cls=_DummyChannel,
                aliases=("leaky",),
            )
        )
        self.assertIs(reg.get("channel", "IL"), _DummyChannel)
        self.assertIs(reg.get("channel", "leaky"), _DummyChannel)

    def test_add_alias_to_existing(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="IL", cls=_DummyChannel)
        )
        reg.add_alias(category="channel", alias="passive", name="IL")
        self.assertIs(reg.get("channel", "passive"), _DummyChannel)

    def test_add_alias_to_missing_entry_raises(self) -> None:
        reg = self.make_registry()
        with self.assertRaises(KeyError):
            reg.add_alias(category="channel", alias="a", name="missing")

    def test_add_alias_collision_raises(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="IL", cls=_DummyChannel)
        )
        reg.add_alias(category="channel", alias="leaky", name="IL")
        with self.assertRaises(ValueError):
            reg.add_alias(category="channel", alias="leaky", name="IL")


class MechanismRegistryRemovalTest(
    MechanismRegistryIsolationMixin, unittest.TestCase
):
    def test_unregister_removes_canonical_and_aliases(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(
                category="channel",
                name="Foo",
                cls=_DummyChannel,
                aliases=("foo_alias",),
            )
        )
        reg.unregister("channel", "Foo")
        self.assertFalse(reg.contains("channel", "Foo"))
        self.assertFalse(reg.contains("channel", "foo_alias"))

    def test_unregister_missing_raises(self) -> None:
        reg = self.make_registry()
        with self.assertRaises(KeyError):
            reg.unregister("channel", "Foo")

    def test_clear_empties_registry(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="A", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="ion", name="B", cls=_DummyIon)
        )
        reg.clear()
        self.assertEqual(len(reg), 0)
        self.assertEqual(reg.names(), ())


class MechanismRegistryListingTest(
    MechanismRegistryIsolationMixin, unittest.TestCase
):
    def test_names_all(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="B", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="channel", name="A", cls=_DummyChannel)
        )
        self.assertEqual(reg.names("channel"), ("A", "B"))

    def test_names_include_aliases(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(
                category="channel",
                name="IL",
                cls=_DummyChannel,
                aliases=("leaky",),
            )
        )
        names = reg.names("channel", include_aliases=True)
        self.assertIn("IL", names)
        self.assertIn("leaky", names)

    def test_items_returns_pairs(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="A", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="ion", name="B", cls=_DummyIon)
        )
        pairs = reg.items("channel")
        self.assertEqual(pairs, (("A", _DummyChannel),))

    def test_items_across_categories(self) -> None:
        reg = self.make_registry()
        reg.register(
            MechanismEntry(category="channel", name="A", cls=_DummyChannel)
        )
        reg.register(
            MechanismEntry(category="ion", name="B", cls=_DummyIon)
        )
        pairs = reg.items()
        self.assertEqual(len(pairs), 2)


class DecoratorRegistrationTest(unittest.TestCase):
    """Decorators register into the global singleton.

    These tests snapshot and restore the singleton so the global
    registry is not polluted.
    """

    def setUp(self) -> None:
        self._snapshot_entries = dict(_REGISTRY._entries)
        self._snapshot_aliases = dict(_REGISTRY._aliases)

    def tearDown(self) -> None:
        _REGISTRY._entries = self._snapshot_entries
        _REGISTRY._aliases = self._snapshot_aliases

    def test_register_channel_decorator(self) -> None:
        @register_channel("__test_channel__", aliases=("__tc_alias__",))
        class _TC:
            pass

        self.assertIs(_REGISTRY.get("channel", "__test_channel__"), _TC)
        self.assertIs(_REGISTRY.get("channel", "__tc_alias__"), _TC)

    def test_register_ion_decorator(self) -> None:
        @register_ion("__test_ion__")
        class _TI:
            pass

        self.assertIs(_REGISTRY.get("ion", "__test_ion__"), _TI)

    def test_register_synapse_decorator(self) -> None:
        @register_synapse("__test_synapse__")
        class _TS:
            pass

        self.assertIs(_REGISTRY.get("synapse", "__test_synapse__"), _TS)

    def test_decorator_returns_class_unchanged(self) -> None:
        @register_channel("__test_channel2__")
        class _TC2:
            x = 42

        self.assertEqual(_TC2.x, 42)


class BuiltinRegistrationTest(unittest.TestCase):
    """Smoke-test that concrete classes from braincell.channel / .ion /
    .synapse are present in the global registry after normal import.
    """

    def test_channel_IL_registered(self) -> None:
        import braincell.channel as channel  # noqa: F401

        self.assertIs(_REGISTRY.get("channel", "IL"), channel.IL)

    def test_channel_IL_alias_leaky(self) -> None:
        import braincell.channel as channel

        self.assertIs(_REGISTRY.get("channel", "leaky"), channel.IL)

    def test_channel_INa_HH1952_registered(self) -> None:
        import braincell.channel as channel

        self.assertIs(
            _REGISTRY.get("channel", "INa_HH1952"), channel.INa_HH1952
        )

    def test_channel_IK_HH1952_registered(self) -> None:
        import braincell.channel as channel

        self.assertIs(
            _REGISTRY.get("channel", "IK_HH1952"), channel.IK_HH1952
        )

    def test_ion_SodiumFixed_registered(self) -> None:
        import braincell.ion as ion

        self.assertIs(_REGISTRY.get("ion", "SodiumFixed"), ion.SodiumFixed)

    def test_ion_PotassiumFixed_registered(self) -> None:
        import braincell.ion as ion

        self.assertIs(
            _REGISTRY.get("ion", "PotassiumFixed"), ion.PotassiumFixed
        )

    def test_ion_CalciumFixed_registered(self) -> None:
        import braincell.ion as ion

        self.assertIs(_REGISTRY.get("ion", "CalciumFixed"), ion.CalciumFixed)

    def test_synapse_AMPA_registered(self) -> None:
        import braincell.synapse as synapse

        self.assertIs(_REGISTRY.get("synapse", "AMPA"), synapse.AMPA)


if __name__ == "__main__":
    unittest.main()
