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

The three factories' *sampling* behaviour — that a density actually bends
the draw distribution — is exercised end-to-end in ``_sampling_test.py``,
which owns the inverse-CDF machinery. What lives here is the module's own
surface: argument validation, the deprecation contract on the two legacy
field helpers, and the log-density evaluation each frozen density performs.
"""

import unittest
import warnings
from types import SimpleNamespace

import brainunit as u
import numpy as np

from braincell.filter import density


def _quiet(factory, *args, **kwargs):
    """Call a deprecated factory without the warning failing the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return factory(*args, **kwargs)


class DeprecationContractTest(unittest.TestCase):
    """``exponential`` and ``gaussian`` are deprecated in favour of ``metric``."""

    def test_legacy_field_helpers_emit_deprecation_warning(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, "filter.metric"):
            density.exponential("branch_x", 0.2)
        with self.assertWarnsRegex(DeprecationWarning, "filter.metric"):
            density.gaussian("branch_x", 0.5, 0.2)

    def test_spatial_gaussian_is_not_deprecated(self) -> None:
        # It has no field-name indirection to replace, so it is the one
        # factory metric.py does not supersede.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.um)


class ExponentialValidationTest(unittest.TestCase):
    def test_field_must_be_a_non_empty_string(self) -> None:
        for bad in ("", 0, None):
            with self.subTest(field=bad):
                with self.assertRaisesRegex(TypeError, "field must be a non-empty string"):
                    _quiet(density.exponential, bad, 0.2)

    def test_scale_must_be_positive(self) -> None:
        for bad in (0.0, -1.0):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _quiet(density.exponential, "branch_x", bad)

    def test_direction_must_be_a_known_word(self) -> None:
        with self.assertRaisesRegex(ValueError, "increasing.*decreasing"):
            _quiet(density.exponential, "branch_x", 0.2, "sideways")

    def test_direction_sets_the_sign_of_the_exponent(self) -> None:
        rising = _quiet(density.exponential, "branch_x", 0.5, "increasing")
        falling = _quiet(density.exponential, "branch_x", 0.5, "decreasing")
        near = SimpleNamespace(branch_x=0.1)
        far = SimpleNamespace(branch_x=0.9)

        self.assertGreater(float(rising(far)), float(rising(near)))
        self.assertLess(float(falling(far)), float(falling(near)))


class GaussianValidationTest(unittest.TestCase):
    def test_field_must_be_a_non_empty_string(self) -> None:
        with self.assertRaisesRegex(TypeError, "field must be a non-empty string"):
            _quiet(density.gaussian, "", 0.5, 0.2)

    def test_sigma_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            _quiet(density.gaussian, "branch_x", 0.5, 0.0)

    def test_density_peaks_at_the_center(self) -> None:
        bell = _quiet(density.gaussian, "branch_x", 0.5, 0.1)
        at_center = float(bell(SimpleNamespace(branch_x=0.5)))
        off_center = float(bell(SimpleNamespace(branch_x=0.9)))

        self.assertGreater(at_center, off_center)
        self.assertTrue(np.isfinite(at_center))

    def test_a_missing_field_names_the_field_it_wanted(self) -> None:
        bell = _quiet(density.gaussian, "branch_x", 0.5, 0.1)
        with self.assertRaisesRegex(ValueError, "no density field 'branch_x'"):
            bell(SimpleNamespace())


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
    """Every factory returns a non-negative, finite, dimensionless callable."""

    def test_all_three_return_non_negative_finite_values(self) -> None:
        cases = (
            (_quiet(density.exponential, "branch_x", 0.5), SimpleNamespace(branch_x=0.3)),
            (_quiet(density.gaussian, "branch_x", 0.5, 0.2), SimpleNamespace(branch_x=0.3)),
            (
                density.spatial_gaussian(center=[0.0, 0.0, 0.0] * u.um, sigma=10.0 * u.um),
                SimpleNamespace(position=[1.0, 2.0, 3.0] * u.um),
            ),
        )
        for fn, context in cases:
            with self.subTest(density=type(fn).__name__):
                value = np.asarray(fn(context), dtype=float)
                self.assertTrue(np.all(np.isfinite(value)))
                self.assertTrue(np.all(value >= 0.0))

    def test_densities_are_frozen(self) -> None:
        bell = _quiet(density.gaussian, "branch_x", 0.5, 0.2)
        with self.assertRaises(Exception):
            bell.field = "radius"


if __name__ == "__main__":
    unittest.main()
