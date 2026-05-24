"""Unit tests for :mod:`braincell._multi_compartment.probes`.

Full probe-sampling flows require an initialized ``Cell`` and are
exercised in ``cell_test.py``. These tests lock down the small pure
helpers so regressions stay local.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._multi_compartment.probes import (
    _probe_attr_value,
    _pack_probe_samples,
    _select_last_axis,
    sample_probe,
    sample_probes,
)


class TestSelectLastAxis(unittest.TestCase):
    def test_scalar_passthrough(self):
        self.assertEqual(_select_last_axis(5, 0), 5)

    def test_vector_indexes_last_axis(self):
        arr = jnp.asarray([10.0, 20.0, 30.0])
        np.testing.assert_allclose(float(_select_last_axis(arr, 1)), 20.0)

    def test_brainstate_state_unwraps(self):
        state = brainstate.ShortTermState(jnp.asarray([1.0, 2.0]))
        np.testing.assert_allclose(float(_select_last_axis(state, 0)), 1.0)


class TestPackProbeSamples(unittest.TestCase):
    def test_single_sample_passthrough(self):
        self.assertEqual(_pack_probe_samples([42.0]), 42.0)

    def test_unitless_samples_stacked(self):
        out = _pack_probe_samples([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(out), [1.0, 2.0, 3.0])

    def test_unit_bearing_samples_stacked_with_unit(self):
        samples = [1.0 * u.mV, 2.0 * u.mV, 3.0 * u.mV]
        out = _pack_probe_samples(samples)
        self.assertTrue(hasattr(out, "unit"))
        np.testing.assert_allclose(out.to_decimal(u.mV), [1.0, 2.0, 3.0])


class TestProbeAttrValue(unittest.TestCase):
    def test_state_field_returns_value(self):
        class _Owner:
            x = brainstate.ShortTermState(jnp.asarray([1.0, 2.0]))

        out = _probe_attr_value(_Owner(), "x", probe_name="p")
        np.testing.assert_allclose(np.asarray(out), [1.0, 2.0])

    def test_plain_field_returns_raw_value(self):
        class _Owner:
            y = jnp.asarray([3.0, 4.0])

        out = _probe_attr_value(_Owner(), "y", probe_name="p")
        np.testing.assert_allclose(np.asarray(out), [3.0, 4.0])


class TestPublicHelpersExist(unittest.TestCase):
    def test_callables(self):
        self.assertTrue(callable(sample_probe))
        self.assertTrue(callable(sample_probes))


if __name__ == "__main__":
    unittest.main()
