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

import braincell
from braincell.mech import Channel, Density, Ion, Params


class DensityConstructionTest(unittest.TestCase):
    def test_channel_sets_category(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        self.assertEqual(spec.category, "channel")
        self.assertEqual(spec.class_name, "IL")

    def test_ion_sets_category(self) -> None:
        spec = Ion("SodiumFixed", E=50 * u.mV)
        self.assertEqual(spec.category, "ion")
        self.assertEqual(spec.class_name, "SodiumFixed")

    def test_channel_is_density_subclass(self) -> None:
        spec = Channel("IL")
        self.assertIsInstance(spec, Density)

    def test_ion_is_density_subclass(self) -> None:
        spec = Ion("SodiumFixed")
        self.assertIsInstance(spec, Density)

    def test_params_are_Params(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertIsInstance(spec.params, Params)
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm ** 2))

    def test_default_params_is_empty(self) -> None:
        spec = Channel("IL")
        self.assertEqual(len(spec.params), 0)

    def test_default_coverage_is_one(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertEqual(spec.coverage_area_fraction, 1.0)


class DensityClassArgumentTest(unittest.TestCase):
    def test_channel_accepts_type_argument(self) -> None:
        spec = Channel(
            braincell.channel.IL, g_max=0.1 * (u.mS / u.cm ** 2)
        )
        self.assertEqual(spec.class_name, "IL")
        self.assertEqual(spec.category, "channel")

    def test_ion_accepts_type_argument(self) -> None:
        spec = Ion(braincell.ion.PotassiumFixed)
        self.assertEqual(spec.class_name, "PotassiumFixed")
        self.assertEqual(spec.category, "ion")

    def test_type_and_string_produce_equal_specs(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        b = Channel(braincell.channel.IL, g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class DensityValidationTest(unittest.TestCase):
    def test_abstract_density_rejects_instantiation(self) -> None:
        with self.assertRaises(TypeError):
            Density("IL")

    def test_empty_class_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Channel("")

    def test_invalid_class_name_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel(42)  # type: ignore[arg-type]

    def test_non_string_name_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel("IL", name=42)  # type: ignore[arg-type]

    def test_out_of_range_coverage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Channel("IL", coverage_area_fraction=2.0)
        with self.assertRaises(ValueError):
            Channel("IL", coverage_area_fraction=-0.1)


class DensityIdentityTest(unittest.TestCase):
    def test_default_instance_name_is_class_name(self) -> None:
        spec = Channel("IL")
        self.assertEqual(spec.instance_name, "IL")
        self.assertEqual(spec.identity, ("IL", "IL"))

    def test_override_instance_name(self) -> None:
        spec = Channel("INa_HH1952", name="na_main")
        self.assertEqual(spec.instance_name, "na_main")
        self.assertEqual(spec.identity, ("na_main", "INa_HH1952"))


class DensityEqualityTest(unittest.TestCase):
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

    def test_channel_and_ion_are_unequal_even_with_same_fields(self) -> None:
        c = Channel("SodiumFixed")
        i = Ion("SodiumFixed")
        self.assertNotEqual(c, i)

    def test_can_use_as_dict_key(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        b = Channel("IL", E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))
        bucket = {a: "one"}
        self.assertEqual(bucket[b], "one")


class DensityUpdatesTest(unittest.TestCase):
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
        self.assertIsInstance(updated, Channel)

    def test_with_coverage_non_mutating(self) -> None:
        original = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2))
        updated = original.with_coverage(0.5)
        self.assertEqual(original.coverage_area_fraction, 1.0)
        self.assertEqual(updated.coverage_area_fraction, 0.5)
        self.assertIsInstance(updated, Channel)

    def test_with_name_non_mutating(self) -> None:
        original = Channel("IL")
        updated = original.with_name("leak_soma")
        self.assertIsNone(original.name)
        self.assertEqual(updated.name, "leak_soma")
        self.assertIsInstance(updated, Channel)

    def test_updates_preserve_ion_subclass(self) -> None:
        original = Ion("SodiumFixed", c0=12.0)
        updated = original.with_params(c0=15.0)
        self.assertIsInstance(updated, Ion)
        self.assertEqual(updated.params["c0"], 15.0)


class DensityImmutabilityTest(unittest.TestCase):
    def test_cannot_set_attributes(self) -> None:
        spec = Channel("IL")
        with self.assertRaises(AttributeError):
            spec.class_name = "IK_HH1952"  # type: ignore[misc]

    def test_cannot_delete_attributes(self) -> None:
        spec = Channel("IL")
        with self.assertRaises(AttributeError):
            del spec.class_name  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
