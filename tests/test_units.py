from __future__ import annotations

import unittest

from ._support import u, np

from braincell._units import mantissa, normalize_param, segment_lengths_from_points


class UnitHelpersTest(unittest.TestCase):
    def test_normalize_param_rejects_unitless_numeric_inputs(self) -> None:
        with self.assertRaises(TypeError):
            normalize_param(
                [0.01, 0.02],
                name="lengths",
                unit=u.mm,
                shape=(None,),
                bounds={"gt": 0},
            )

        with self.assertRaises(TypeError):
            normalize_param(
                0.5,
                name="radius",
                unit=u.um,
            )

        with self.assertRaises(TypeError):
            normalize_param(
                np.array([1.0, 2.0]),
                name="lengths",
                unit=u.um,
            )

    def test_normalize_param_converts_to_base_unit(self) -> None:
        converted = normalize_param(
            np.array([10.0, 20.0]) * u.mm,
            name="lengths",
            unit=u.um,
            shape=(None,),
            bounds={"gt": 0},
        )

        self.assertTrue(u.math.allclose(converted, np.array([10_000.0, 20_000.0]) * u.um))

    def test_segment_lengths_from_points_keeps_units(self) -> None:
        points = normalize_param(
            np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            name="points",
            unit=u.um,
            shape=(None, 3),
        )

        lengths = segment_lengths_from_points(points)

        self.assertTrue(u.math.array_equal(lengths, np.array([5.0, 12.0]) * u.um))

    def test_normalize_param_rejects_invalid_shape_bounds_and_dimension(self) -> None:
        with self.assertRaises(ValueError):
            normalize_param(
                np.array([[0.0, 1.0], [2.0, 3.0]]) * u.um,
                name="points",
                unit=u.um,
                shape=(None, 3),
            )

        with self.assertRaises(ValueError):
            normalize_param(
                np.array([-1.0, 2.0]) * u.um,
                name="radii",
                unit=u.um,
                shape=(None,),
                bounds={"ge": 0},
            )

        with self.assertRaises(Exception):
            normalize_param(
                np.array([1.0, 2.0]) * u.mV,
                name="lengths",
                unit=u.um,
                shape=(None,),
            )
