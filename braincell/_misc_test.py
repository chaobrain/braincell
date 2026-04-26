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


if __name__ == "__main__":
    unittest.main()
