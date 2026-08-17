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

import unittest

import brainunit as u
import jax
import jax.numpy as jnp


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


if __name__ == "__main__":
    unittest.main()
