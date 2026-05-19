"""Unit tests for :mod:`braincell._multi_compartment.bridge`.

These use a minimal stub for :class:`CellRuntimeState` so the scatter
/ gather behaviour is testable without a full :class:`Cell` build.
End-to-end roundtrip on a real ``Cell`` is covered by
``runnable_test.py`` once ``Cell.build()`` lands.
"""

import unittest
from dataclasses import dataclass

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._multi_compartment import bridge


@dataclass
class _StubNodeTree:
    cv_to_mid_node_id: np.ndarray


@dataclass
class _StubRuntime:
    node_tree: _StubNodeTree
    n_point: int
    n_cv: int


@dataclass
class _StubCV:
    length: object
    area: object
    diam_mid: object
    radius_prox: object
    radius_dist: object


def _runtime(point_ids: list[int], n_point: int) -> _StubRuntime:
    return _StubRuntime(
        node_tree=_StubNodeTree(
            cv_to_mid_node_id=np.asarray(point_ids, dtype=np.int32),
        ),
        n_point=n_point,
        n_cv=len(point_ids),
    )


class TestBridge(unittest.TestCase):
    def test_cv_to_point_scatters_to_midpoints(self):
        runtime = _runtime(point_ids=[1, 3], n_point=5)
        cv_values = jnp.asarray([10.0, 20.0]) * u.mV

        out = bridge.cv_to_point(cv_values, runtime)

        self.assertEqual(out.shape, (5,))
        np.testing.assert_allclose(
            out.to_decimal(u.mV),
            np.asarray([0.0, 10.0, 0.0, 20.0, 0.0]),
        )

    def test_point_to_cv_gathers_at_midpoints(self):
        runtime = _runtime(point_ids=[1, 3], n_point=5)
        point_values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]) * u.mV

        out = bridge.point_to_cv(point_values, runtime)

        self.assertEqual(out.shape, (2,))
        np.testing.assert_allclose(out.to_decimal(u.mV), np.asarray([2.0, 4.0]))

    def test_roundtrip_restores_cv_values(self):
        runtime = _runtime(point_ids=[0, 2, 5], n_point=8)
        cv_values = jnp.asarray([1.5, -2.0, 7.25]) * u.mV

        back = bridge.point_to_cv(bridge.cv_to_point(cv_values, runtime), runtime)

        np.testing.assert_allclose(back.to_decimal(u.mV), cv_values.to_decimal(u.mV))

    def test_attach_runtime_ion_geometry_injects_cv_geometry_fields(self):
        cvs = (
            _StubCV(
                length=10.0 * u.um,
                area=100.0 * u.um ** 2,
                diam_mid=6.0 * u.um,
                radius_prox=3.5 * u.um,
                radius_dist=2.5 * u.um,
            ),
            _StubCV(
                length=20.0 * u.um,
                area=200.0 * u.um ** 2,
                diam_mid=8.0 * u.um,
                radius_prox=4.5 * u.um,
                radius_dist=3.5 * u.um,
            ),
        )

        class _Ion:
            pass

        ion = _Ion()
        bridge.attach_runtime_ion_geometry(
            ions={"ca": ion},
            cvs=cvs,
            point_ids=np.asarray([1, 3], dtype=np.int32),
            n_point=5,
        )

        np.testing.assert_allclose(
            ion.length.to_decimal(u.um),
            np.asarray([0.0, 10.0, 0.0, 20.0, 0.0]),
        )
        np.testing.assert_allclose(
            ion.area.to_decimal(u.um ** 2),
            np.asarray([0.0, 100.0, 0.0, 200.0, 0.0]),
        )
        np.testing.assert_allclose(
            ion.diam_mid.to_decimal(u.um),
            np.asarray([0.0, 6.0, 0.0, 8.0, 0.0]),
        )
        np.testing.assert_allclose(
            (u.math.pi * ion.diam_mid).to_decimal(u.um),
            np.asarray([0.0, np.pi * 6.0, 0.0, np.pi * 8.0, 0.0]),
        )


class IsPythonZeroRejectsBoolTest(unittest.TestCase):
    """LOW-08: ``is_python_zero(False)`` must not short-circuit to zero."""

    def test_false_is_not_python_zero(self) -> None:
        self.assertFalse(bridge.is_python_zero(False))

    def test_true_is_not_python_zero(self) -> None:
        self.assertFalse(bridge.is_python_zero(True))

    def test_zero_float_still_matches(self) -> None:
        self.assertTrue(bridge.is_python_zero(0))
        self.assertTrue(bridge.is_python_zero(0.0))


if __name__ == "__main__":
    unittest.main()
