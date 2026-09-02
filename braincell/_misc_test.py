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

"""Unit tests for :mod:`braincell._misc`."""

import re
import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import Container, normalize_param, scalar_decimal, validate_time_quantity


class IsTracedValueTest(unittest.TestCase):
    """ARCH-06: ``is_traced_value`` lives on the shared misc module."""

    def test_import_site(self) -> None:
        from braincell._misc import is_traced_value

        self.assertTrue(callable(is_traced_value))

    def test_concrete_number_is_not_traced(self) -> None:
        from braincell._misc import is_traced_value

        self.assertFalse(is_traced_value(1.0))
        self.assertFalse(is_traced_value(jnp.asarray(1.0)))
        self.assertFalse(is_traced_value(jnp.asarray([1.0, 2.0]) * u.mV))

    def test_jax_tracer_is_traced(self) -> None:
        from braincell._misc import is_traced_value

        results = []

        def probe(x):
            results.append(is_traced_value(x))
            return x

        jax.jit(probe)(jnp.asarray(1.0))
        self.assertTrue(results[-1])


class SetModuleAsTest(unittest.TestCase):
    """``set_module_as`` re-homes ``__module__`` and leaves ``__name__``."""

    def test_it_sets_module_and_preserves_name(self) -> None:
        from braincell._misc import set_module_as

        @set_module_as("braincell.quad")
        def some_step():
            pass

        # The bug this guards: an earlier version assigned the path to
        # ``__name__``, so every decorated function claimed to be called
        # "braincell.quad" while ``__module__`` still pointed at the
        # private module users are not meant to import from.
        self.assertEqual(some_step.__module__, "braincell.quad")
        self.assertEqual(some_step.__name__, "some_step")

    def test_it_returns_the_same_object(self) -> None:
        from braincell._misc import set_module_as

        def some_step():
            return 7

        self.assertIs(set_module_as("braincell")(some_step), some_step)
        self.assertEqual(some_step(), 7)

    def test_every_decorated_public_function_is_re_homed(self) -> None:
        # The decorator exists so the public API does not advertise private
        # module paths; assert that for the exported functions it is applied
        # to, rather than trusting the 24 individual call sites.
        import braincell
        import braincell.quad as quad

        for name in ("exp_euler_step", "rk4_step", "staggered_step"):
            with self.subTest(function=name):
                fun = getattr(quad, name)
                self.assertEqual(fun.__module__, "braincell.quad")
                self.assertEqual(fun.__name__, name)

        for name in ("state_grouping", "state", "hidden_state"):
            with self.subTest(function=name):
                fun = getattr(braincell, name)
                self.assertEqual(fun.__module__, "braincell")
                self.assertEqual(fun.__name__, name)


class ValidateTimeQuantityTest(unittest.TestCase):
    """``dt`` and ``delay`` are validated under different, explicit rules.

    Both used to run through a single local helper that decided what to
    check by comparing the parameter *name* against ``"dt"``, so anything
    not called ``dt`` silently skipped every real check. The rules are now
    named arguments; these tests pin each half of that split. They live
    here rather than in ``network/lowering_test.py`` because they drive the
    shared validator directly, not the network lowering that calls it.
    """

    def test_a_bare_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "Network dt must be a time quantity"):
            validate_time_quantity(0.1, name="dt", prefix="Network")

    def test_prefix_names_the_calling_layer(self) -> None:
        # The network layer used to report single-cell wording because it
        # borrowed Cell.run's validator wholesale.
        with self.assertRaisesRegex(TypeError, r"Network\.run\(\.\.\.\) duration"):
            validate_time_quantity(1.0, name="duration", prefix="Network.run(...)")
        with self.assertRaisesRegex(TypeError, r"Cell\.run\(\.\.\.\) duration"):
            validate_time_quantity(1.0, name="duration", prefix="Cell.run(...)")

    def test_delay_may_be_a_zero_or_vector_quantity(self) -> None:
        # A delay is legitimately per-contact and zero means immediate
        # delivery, so neither rule applies to it.
        validate_time_quantity(
            [0.0, 1.0] * u.ms,
            name="delay",
            prefix="Network",
            require_scalar=False,
            require_positive=False,
        )

    def test_a_length_one_quantity_counts_as_scalar(self) -> None:
        validate_time_quantity([0.1] * u.ms, name="dt", prefix="Network")

    def test_a_longer_vector_is_rejected_when_scalar_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must be scalar, got shape \(2,\)"):
            validate_time_quantity([0.1, 0.2] * u.ms, name="dt", prefix="Network")

    def test_positivity_applies_elementwise_when_scalar_is_not_required(self) -> None:
        # The positivity branch used to re-check the shape purely so it
        # could call ``reshape(())``, which made this combination raise a
        # misleading "must be scalar" instead of checking the values.
        validate_time_quantity(
            [0.5, 1.0] * u.ms,
            name="delay",
            prefix="Network",
            require_scalar=False,
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            validate_time_quantity(
                [0.5, -1.0] * u.ms,
                name="delay",
                prefix="Network",
                require_scalar=False,
            )

    def test_zero_is_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            validate_time_quantity(0.0 * u.ms, name="dt", prefix="Network")


class ScalarDecimalTest(unittest.TestCase):
    """``scalar_decimal`` is the one spelling of the static-assembly cast."""

    def test_it_converts_and_rescales(self) -> None:
        self.assertEqual(scalar_decimal(0.1 * u.ms, u.ms), 0.1)
        self.assertEqual(scalar_decimal(1.0 * u.second, u.ms), 1000.0)
        self.assertIsInstance(scalar_decimal(0.1 * u.ms, u.ms), float)

    def test_it_accepts_a_length_one_quantity(self) -> None:
        # The spelling without ``reshape(())`` raises TypeError here under
        # NumPy 2, so the two inlined forms disagreed on exactly the shape
        # ``validate_time_quantity`` documents as scalar.
        self.assertEqual(scalar_decimal([0.25] * u.ms, u.ms), 0.25)
        self.assertEqual(scalar_decimal(np.full((1, 1), 0.25) * u.ms, u.ms), 0.25)

    def test_it_rejects_a_genuinely_non_scalar_quantity(self) -> None:
        with self.assertRaises(ValueError):
            scalar_decimal([0.1, 0.2] * u.ms, u.ms)


class NormalizeParamBoundsTest(unittest.TestCase):
    """The bound table drives both the key check and the comparison."""

    def test_each_supported_bound_is_enforced(self) -> None:
        cases = (
            ("ge", 2.0 * u.ms, ">="),
            ("gt", 2.0 * u.ms, ">"),
            ("le", 0.5 * u.ms, "<="),
            ("lt", 1.0 * u.ms, "<"),
        )
        for key, bound, symbol in cases:
            with self.subTest(bound=key):
                with self.assertRaisesRegex(ValueError, f"must satisfy {re.escape(symbol)}"):
                    normalize_param(1.0 * u.ms, name="dt", unit=u.ms, bounds={key: bound})

    def test_a_satisfied_bound_passes(self) -> None:
        result = normalize_param(1.0 * u.ms, name="dt", unit=u.ms, bounds={"gt": 0.0 * u.ms})
        self.assertEqual(float(result.to_decimal(u.ms)), 1.0)

    def test_an_unknown_bound_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"unsupported bound keys \('eq',\)"):
            normalize_param(1.0 * u.ms, name="dt", unit=u.ms, bounds={"eq": 1.0 * u.ms})


class ContainerTest(unittest.TestCase):
    """``Container`` reaches children by dot and by index."""

    class _Bag(Container):
        _container_name = "parts"

        def __init__(self, **parts):
            self.parts = dict(parts)

        def add(self, **elements):
            self.parts.update(elements)

    def test_dot_and_index_access_find_the_same_child(self) -> None:
        child = object()
        bag = self._Bag(na=child)
        self.assertIs(bag.na, child)
        self.assertIs(bag["na"], child)
        self.assertIs(bag.parts["na"], child)

    def test_container_name_still_resolves_as_a_normal_attribute(self) -> None:
        # ``__getattr__`` carried an ``item == '_container_name'`` branch
        # that could never run: the class attribute resolves through normal
        # lookup, so ``__getattr__`` is never entered for that name.
        bag = self._Bag(na=object())
        self.assertEqual(bag._container_name, "parts")

    def test_an_unknown_child_reports_what_is_there(self) -> None:
        bag = self._Bag(na=object())
        with self.assertRaisesRegex(ValueError, r"Unknown item ca, we only found \['na'\]"):
            bag["ca"]

    def test_format_elements_type_checks(self) -> None:
        with self.assertRaisesRegex(TypeError, "Should be instance of int"):
            Container._format_elements(int, ok=1, bad="x")


if __name__ == "__main__":
    unittest.main()
