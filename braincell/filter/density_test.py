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

"""Tests for :mod:`braincell.filter.density`.

The factory's *sampling* behaviour -- that a density actually bends the
draw distribution -- is exercised end-to-end in ``_sampling_test.py``,
which owns the inverse-CDF machinery. What lives here is the module's own
surface: argument validation, the log-density evaluation of the frozen
density, and the tombstones of the two field helpers braincell 0.3 removed.
"""

import unittest
from types import SimpleNamespace

import brainunit as u
import numpy as np

from braincell.filter import density


class RemovedHelpersTest(unittest.TestCase):
    """API-15: ``exponential`` and ``gaussian`` were removed in braincell 0.3 in favour of ``metric``."""

    def test_access_names_the_release_and_the_replacement(self) -> None:
        for name in ("exponential", "gaussian"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as caught:
                    getattr(density, name)
                message = str(caught.exception)
                self.assertTrue(
                    message.startswith(f"braincell.filter.density.{name}() was removed in braincell 0.3."), message
                )
                self.assertIn("braincell.filter.metric", message)
                self.assertIn("metric.branch_x(context)", message)
                self.assertFalse(hasattr(density, name))
                self.assertNotIn(name, density.__all__)

    def test_from_import_raises_import_error_with_the_message(self) -> None:
        with self.assertRaisesRegex(
            ImportError, r"^braincell\.filter\.density\.exponential\(\) was removed in braincell 0\.3\."
        ):
            from braincell.filter.density import exponential  # noqa: F401
        with self.assertRaisesRegex(
            ImportError, r"^braincell\.filter\.density\.gaussian\(\) was removed in braincell 0\.3\."
        ):
            from braincell.filter.density import gaussian  # noqa: F401

    def test_other_missing_names_keep_the_generic_message(self) -> None:
        with self.assertRaisesRegex(AttributeError, "module 'braincell.filter.density' has no attribute 'cauchy'"):
            density.cauchy

    def test_the_suggested_replacements_are_densities(self) -> None:
        # The two forms the messages suggest run as written on a sampling context.
        from braincell.filter import metric

        context = SimpleNamespace(branch_x=np.asarray([0.25, 0.5]))
        rising = np.exp(metric.branch_x(context) / 0.2)
        bell = np.exp(-0.5 * ((metric.branch_x(context) - 0.5) / 0.1) ** 2)
        np.testing.assert_allclose(rising, np.exp([1.25, 2.5]), rtol=1e-12)
        np.testing.assert_allclose(bell, [np.exp(-3.125), 1.0], rtol=1e-12)


class SpatialGaussianValidationTest(unittest.TestCase):
    def test_center_must_be_a_three_dimensional_quantity(self) -> None:
        with self.assertRaisesRegex(TypeError, "center must be a three-dimensional quantity"):
            density.spatial_gaussian(center=[0.0, 0.0, 0.0], sigma=10.0 * u.um)
        with self.assertRaisesRegex(ValueError, "finite three-dimensional coordinate"):
            density.spatial_gaussian(center=[0.0, 0.0] * u.um, sigma=10.0 * u.um)

    def test_sigma_must_be_a_length_quantity(self) -> None:
        with self.assertRaisesRegex(TypeError, "sigma must be a length quantity"):
            density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0)

    def test_center_and_sigma_units_must_be_compatible(self) -> None:
        with self.assertRaises(Exception):
            density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.ms)

    def test_density_peaks_at_the_center_position(self) -> None:
        bell = density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.um)
        at_center = float(bell(SimpleNamespace(position=[0.0, 0.0, 0.0] * u.um)))
        off_center = float(bell(SimpleNamespace(position=[30.0, 0.0, 0.0] * u.um)))

        self.assertGreater(at_center, off_center)
        self.assertTrue(np.isfinite(at_center))


class DensityValueTest(unittest.TestCase):
    """The factory returns a non-negative, finite, dimensionless callable."""

    def test_returns_non_negative_finite_values(self) -> None:
        fn = density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.um)
        value = np.asarray(fn(SimpleNamespace(position=[1.0, 2.0, 3.0] * u.um)), dtype=float)
        self.assertTrue(np.all(np.isfinite(value)))
        self.assertTrue(np.all(value >= 0.0))

    def test_densities_are_frozen(self) -> None:
        bell = density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.um)
        with self.assertRaises(Exception):
            bell.sigma = 1.0 * u.um


if __name__ == "__main__":
    unittest.main()
