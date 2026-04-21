"""Unit tests for :mod:`braincell._multi_compartment.clamp_table`.

These exercise :func:`build_clamp_active_table` directly with stubbed
layouts and CV records. A full integration test (clamp placed on a
real ``Cell`` → table appears on ``rcell.runtime``) runs once
``Cell.build()`` lands.
"""

import unittest
from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell._multi_compartment.clamp_table import (
    CLAMP_KINDS,
    ClampActiveTable,
    build_clamp_active_table,
)


@dataclass
class _StubLayout:
    target: str
    kind: str
    point_index: np.ndarray | None


@dataclass
class _StubCV:
    id: int
    area: object  # brainunit Quantity in cm^2


@dataclass
class _StubPointTree:
    cv_midpoint_point_id: np.ndarray


def _point_tree(n_cv: int) -> _StubPointTree:
    return _StubPointTree(
        cv_midpoint_point_id=np.arange(n_cv, dtype=np.int32),
    )


def _cv(cv_id: int, area_cm2: float) -> _StubCV:
    return _StubCV(id=cv_id, area=area_cm2 * u.cm ** 2)


class TestBuildClampActiveTable(unittest.TestCase):
    def test_no_clamp_layouts_returns_none(self):
        layouts = (
            _StubLayout(target="density", kind="IL", point_index=np.asarray([0], dtype=np.int32)),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_cv(0, 1e-6)],
            point_tree=_point_tree(1),
            n_point=1,
        )
        self.assertIsNone(table)

    def test_current_clamp_builds_table(self):
        layouts = (
            _StubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([1], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_cv(0, 1e-6), _cv(1, 2e-6)],
            point_tree=_point_tree(2),
            n_point=2,
        )
        self.assertIsInstance(table, ClampActiveTable)
        np.testing.assert_array_equal(table.ids, np.asarray([1], dtype=np.int32))
        np.testing.assert_allclose(table.area, np.asarray([2e-6]))

    def test_each_clamp_kind_is_recognized(self):
        self.assertEqual(CLAMP_KINDS, frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"}))

    def test_ids_are_sorted_and_unique(self):
        layouts = (
            _StubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([3, 1], dtype=np.int32),
            ),
            _StubLayout(
                target="point",
                kind="SineClamp",
                point_index=np.asarray([1, 2], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_cv(i, 1e-6 * (i + 1)) for i in range(4)],
            point_tree=_point_tree(4),
            n_point=4,
        )
        np.testing.assert_array_equal(table.ids, np.asarray([1, 2, 3], dtype=np.int32))

    def test_zero_area_raises(self):
        layouts = (
            _StubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        with self.assertRaises(ValueError):
            build_clamp_active_table(
                layouts=layouts,
                cvs=[_cv(0, 0.0)],
                point_tree=_point_tree(1),
                n_point=1,
            )

    def test_non_clamp_point_layout_ignored(self):
        layouts = (
            _StubLayout(
                target="point",
                kind="Synapse",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        self.assertIsNone(
            build_clamp_active_table(
                layouts=layouts,
                cvs=[_cv(0, 1e-6)],
                point_tree=_point_tree(1),
                n_point=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
