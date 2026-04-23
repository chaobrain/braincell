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
import numpy as np

from braincell.mech import Params


class ParamsBasicsTest(unittest.TestCase):
    def test_empty_params(self) -> None:
        p = Params()
        self.assertEqual(len(p), 0)
        self.assertEqual(list(p), [])
        self.assertEqual(dict(p), {})
        self.assertEqual(repr(p), "Params()")

    def test_kwargs_construction(self) -> None:
        p = Params(g_max=0.1, E=-70)
        self.assertEqual(p["g_max"], 0.1)
        self.assertEqual(p["E"], -70)
        self.assertEqual(len(p), 2)
        self.assertIn("g_max", p)
        self.assertNotIn("other", p)

    def test_mapping_construction(self) -> None:
        p = Params({"g_max": 0.1, "E": -70})
        self.assertEqual(p["g_max"], 0.1)
        self.assertEqual(p["E"], -70)

    def test_tuple_pairs_construction(self) -> None:
        p = Params((("g_max", 0.1), ("E", -70)))
        self.assertEqual(p["g_max"], 0.1)
        self.assertEqual(p["E"], -70)

    def test_mapping_and_kwargs_merge(self) -> None:
        base = {"g_max": 0.1, "E": -70}
        p = Params(base, E=-65)  # kwargs override base
        self.assertEqual(p["E"], -65)
        self.assertEqual(p["g_max"], 0.1)

    def test_iteration_order_is_declared(self) -> None:
        p = Params(c=3, a=1, b=2)
        self.assertEqual(list(p), ["c", "a", "b"])

    def test_unpacking_works(self) -> None:
        # Params must be usable with **kwargs
        def f(**kwargs):
            return kwargs

        p = Params(g_max=0.1, E=-70)
        out = f(**p)
        self.assertEqual(out, {"g_max": 0.1, "E": -70})


class ParamsEqualityHashTest(unittest.TestCase):
    def test_equality_is_order_insensitive(self) -> None:
        a = Params(g_max=0.1, E=-70)
        b = Params(E=-70, g_max=0.1)
        self.assertEqual(a, b)

    def test_hash_is_order_insensitive(self) -> None:
        a = Params(g_max=0.1, E=-70)
        b = Params(E=-70, g_max=0.1)
        self.assertEqual(hash(a), hash(b))

    def test_usable_as_dict_key(self) -> None:
        a = Params(g_max=0.1, E=-70)
        b = Params(E=-70, g_max=0.1)
        bucket = {a: "hello"}
        self.assertEqual(bucket[b], "hello")

    def test_equality_with_plain_mapping(self) -> None:
        a = Params(g_max=0.1, E=-70)
        self.assertEqual(a, {"g_max": 0.1, "E": -70})
        self.assertEqual(a, {"E": -70, "g_max": 0.1})

    def test_inequality_on_different_values(self) -> None:
        a = Params(g_max=0.1)
        b = Params(g_max=0.2)
        self.assertNotEqual(a, b)

    def test_inequality_on_different_keys(self) -> None:
        a = Params(g_max=0.1)
        b = Params(gbar=0.1)
        self.assertNotEqual(a, b)

    def test_equality_with_brainunit_quantities(self) -> None:
        a = Params(g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        b = Params(E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class ParamsImmutabilityTest(unittest.TestCase):
    def test_cannot_set_attribute(self) -> None:
        p = Params(g_max=0.1)
        with self.assertRaises(AttributeError):
            p.g_max = 0.2  # type: ignore[attr-defined]

    def test_cannot_delete_attribute(self) -> None:
        p = Params(g_max=0.1)
        with self.assertRaises(AttributeError):
            del p._items

    def test_no_setitem_method(self) -> None:
        p = Params(g_max=0.1)
        self.assertFalse(hasattr(p, "__setitem__"))


class ParamsUpdatesTest(unittest.TestCase):
    def test_with_updates_returns_new_instance(self) -> None:
        original = Params(g_max=0.1, E=-70)
        updated = original.with_updates(g_max=0.2)
        self.assertEqual(original["g_max"], 0.1)
        self.assertEqual(updated["g_max"], 0.2)
        self.assertEqual(updated["E"], -70)

    def test_with_updates_preserves_order_for_existing_keys(self) -> None:
        original = Params(a=1, b=2, c=3)
        updated = original.with_updates(b=20)
        self.assertEqual(list(updated), ["a", "b", "c"])

    def test_with_updates_appends_new_keys(self) -> None:
        original = Params(a=1)
        updated = original.with_updates(b=2)
        self.assertEqual(list(updated), ["a", "b"])

    def test_without_drops_keys(self) -> None:
        original = Params(a=1, b=2, c=3)
        pruned = original.without("b")
        self.assertEqual(dict(pruned), {"a": 1, "c": 3})

    def test_without_ignores_unknown_keys(self) -> None:
        original = Params(a=1)
        pruned = original.without("missing")
        self.assertEqual(pruned, original)


class ParamsCoerceTest(unittest.TestCase):
    def test_coerce_passes_through_params(self) -> None:
        p = Params(g_max=0.1)
        self.assertIs(Params.coerce(p), p)

    def test_coerce_from_dict(self) -> None:
        p = Params.coerce({"g_max": 0.1})
        self.assertIsInstance(p, Params)
        self.assertEqual(p["g_max"], 0.1)

    def test_coerce_from_tuple_pairs(self) -> None:
        p = Params.coerce((("g_max", 0.1), ("E", -70)))
        self.assertEqual(p["g_max"], 0.1)
        self.assertEqual(p["E"], -70)

    def test_coerce_from_none(self) -> None:
        p = Params.coerce(None)
        self.assertEqual(len(p), 0)


class ParamsErrorTest(unittest.TestCase):
    def test_bad_data_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            Params(42)  # type: ignore[arg-type]

    def test_bad_tuple_entry_raises(self) -> None:
        with self.assertRaises(TypeError):
            Params((("g_max",),))  # type: ignore[arg-type]


class ParamsRejectsUnhashableValuesTest(unittest.TestCase):
    """MED-01: array-valued params must fail at construction, not at eq/hash."""

    def test_constructor_rejects_array_value(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            Params(g=np.array([1.0, 2.0, 3.0]))
        self.assertIn("hashable", str(ctx.exception).lower())

    def test_constructor_accepts_scalar_quantity(self) -> None:
        p = Params(g=0.1 * u.mS / u.cm ** 2)
        self.assertEqual(p["g"], 0.1 * u.mS / u.cm ** 2)


if __name__ == "__main__":
    unittest.main()
