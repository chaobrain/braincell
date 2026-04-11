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

import brainunit as u

from braincell.mech import Channel, DensityMechanism, Ion, Params


class DensityMechanismConstructionTest(unittest.TestCase):
    def test_channel_factory_sets_category(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        self.assertEqual(spec.category, "channel")
        self.assertEqual(spec.class_name, "IL")

    def test_ion_factory_sets_category(self) -> None:
        spec = Ion("SodiumFixed", E=50 * u.mV)
        self.assertEqual(spec.category, "ion")
        self.assertEqual(spec.class_name, "SodiumFixed")

    def test_factory_params_are_Params(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertIsInstance(spec.params, Params)
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm ** 2))

    def test_direct_construction_with_dict_params(self) -> None:
        spec = DensityMechanism(
            category="channel",
            class_name="IL",
            params={"g_max": 0.1 * (u.mS / u.cm ** 2)},
        )
        self.assertIsInstance(spec.params, Params)
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm ** 2))

    def test_direct_construction_with_tuple_params(self) -> None:
        spec = DensityMechanism(
            category="channel",
            class_name="IL",
            params=(("g_max", 0.1 * (u.mS / u.cm ** 2)),),
        )
        self.assertIsInstance(spec.params, Params)
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm ** 2))

    def test_default_params_is_empty(self) -> None:
        spec = DensityMechanism(category="channel", class_name="IL")
        self.assertEqual(len(spec.params), 0)

    def test_default_coverage_is_one(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertEqual(spec.coverage_area_fraction, 1.0)


class DensityMechanismValidationTest(unittest.TestCase):
    def test_invalid_category_raises(self) -> None:
        with self.assertRaises(ValueError):
            DensityMechanism(category="bogus", class_name="X")

    def test_synapse_category_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DensityMechanism(category="synapse", class_name="AMPA")

    def test_empty_class_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DensityMechanism(category="channel", class_name="")

    def test_non_string_name_rejected(self) -> None:
        with self.assertRaises(TypeError):
            DensityMechanism(
                category="channel", class_name="IL", name=42  # type: ignore[arg-type]
            )

    def test_out_of_range_coverage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DensityMechanism(
                category="channel",
                class_name="IL",
                coverage_area_fraction=2.0,
            )
        with self.assertRaises(ValueError):
            DensityMechanism(
                category="channel",
                class_name="IL",
                coverage_area_fraction=-0.1,
            )


class DensityMechanismIdentityTest(unittest.TestCase):
    def test_default_instance_name_is_class_name(self) -> None:
        spec = Channel("IL")
        self.assertEqual(spec.instance_name, "IL")
        self.assertEqual(spec.identity, ("IL", "IL"))

    def test_override_instance_name(self) -> None:
        spec = Channel("INa_HH1952", name="na_main")
        self.assertEqual(spec.instance_name, "na_main")
        self.assertEqual(spec.identity, ("na_main", "INa_HH1952"))


class DensityMechanismEqualityTest(unittest.TestCase):
    def test_keyword_order_insensitive_equality(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        b = Channel("IL", E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_different_coverage_are_unequal(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        b = a.with_coverage(0.5)
        self.assertNotEqual(a, b)

    def test_different_names_are_unequal(self) -> None:
        a = Channel("INa_HH1952", name="na_main")
        b = Channel("INa_HH1952", name="na_alt")
        self.assertNotEqual(a, b)

    def test_can_use_as_dict_key(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        b = Channel("IL", E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))
        bucket = {a: "one"}
        self.assertEqual(bucket[b], "one")


class DensityMechanismUpdatesTest(unittest.TestCase):
    def test_with_params_non_mutating(self) -> None:
        original = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        updated = original.with_params(
            g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV
        )
        self.assertEqual(
            original.params["g_max"], 0.1 * (u.mS / u.cm ** 2)
        )
        self.assertEqual(
            updated.params["g_max"], 0.2 * (u.mS / u.cm ** 2)
        )
        self.assertEqual(updated.params["E"], -70 * u.mV)

    def test_with_coverage_non_mutating(self) -> None:
        original = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        updated = original.with_coverage(0.5)
        self.assertEqual(original.coverage_area_fraction, 1.0)
        self.assertEqual(updated.coverage_area_fraction, 0.5)

    def test_with_name_non_mutating(self) -> None:
        original = Channel("IL")
        updated = original.with_name("leak_soma")
        self.assertIsNone(original.name)
        self.assertEqual(updated.name, "leak_soma")


if __name__ == "__main__":
    unittest.main()
