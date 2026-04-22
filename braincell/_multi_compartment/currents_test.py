"""Unit tests for :mod:`braincell._multi_compartment.currents`.

These exercise the shape/unit normalization table directly with a
stubbed :class:`CellRuntimeState`. A full channel-current integration
runs once ``Cell.build()`` lands.
"""

import unittest
from dataclasses import dataclass

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._multi_compartment.currents import _normalize_ext_to_point_density


@dataclass
class _StubPointTree:
    cv_midpoint_point_id: np.ndarray


@dataclass
class _StubRuntime:
    point_tree: _StubPointTree
    n_point: int
    n_cv: int
    cv_area: object
    clamp_active_table: object = None


def _runtime(*, point_ids: list[int], n_point: int, cv_area_cm2: list[float]) -> _StubRuntime:
    return _StubRuntime(
        point_tree=_StubPointTree(cv_midpoint_point_id=np.asarray(point_ids, dtype=np.int32)),
        n_point=n_point,
        n_cv=len(point_ids),
        cv_area=u.Quantity(jnp.asarray(cv_area_cm2, dtype=float), u.cm ** 2),
    )


class TestNormalizeExtToPointDensity(unittest.TestCase):
    def test_python_zero_scalar(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        out = _normalize_ext_to_point_density(0.0, rt)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), 0.0)

    def test_python_int_zero(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        out = _normalize_ext_to_point_density(0, rt)
        self.assertEqual(out.shape, (3,))

    def test_scalar_density_broadcasts_full_vector(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        out = _normalize_ext_to_point_density(1.5 * u.nA / u.cm ** 2, rt)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), 1.5)

    def test_scalar_total_current_divides_by_cv_area_and_scatters(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 2e-6])
        out = _normalize_ext_to_point_density(0.2 * u.nA, rt)
        self.assertEqual(out.shape, (3,))
        expected = np.zeros(3)
        expected[0] = 0.2 / 1e-6  # nA/cm^2 at midpoint 0
        expected[2] = 0.2 / 2e-6
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), expected)

    def test_n_point_total_current_rejected(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        bad = jnp.ones((3,)) * u.nA
        with self.assertRaises(ValueError):
            _normalize_ext_to_point_density(bad, rt)

    def test_cv_shape_density_scatters(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        arr = jnp.asarray([1.0, 2.0]) * u.nA / u.cm ** 2
        out = _normalize_ext_to_point_density(arr, rt)
        self.assertEqual(out.shape, (3,))
        expected = np.zeros(3)
        expected[0] = 1.0
        expected[2] = 2.0
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), expected)

    def test_cv_shape_total_current_divides_then_scatters(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 2e-6])
        arr = jnp.asarray([0.1, 0.4]) * u.nA
        out = _normalize_ext_to_point_density(arr, rt)
        self.assertEqual(out.shape, (3,))
        expected = np.zeros(3)
        expected[0] = 0.1 / 1e-6
        expected[2] = 0.4 / 2e-6
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), expected)

    def test_n_point_shape_density_passes_through(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        arr = jnp.asarray([1.0, 2.0, 3.0]) * u.nA / u.cm ** 2
        out = _normalize_ext_to_point_density(arr, rt)
        np.testing.assert_allclose(out.to_decimal(u.nA / u.cm ** 2), [1.0, 2.0, 3.0])

    def test_incompatible_unit_raises(self):
        rt = _runtime(point_ids=[0, 2], n_point=3, cv_area_cm2=[1e-6, 1e-6])
        with self.assertRaises(ValueError):
            _normalize_ext_to_point_density(1.0 * u.mV, rt)


class IsPythonZeroRejectsBoolTest(unittest.TestCase):
    """LOW-08: ``_is_python_zero(False)`` must not short-circuit to zero."""

    def test_false_is_not_python_zero(self) -> None:
        from braincell._multi_compartment.currents import _is_python_zero
        self.assertFalse(_is_python_zero(False))

    def test_true_is_not_python_zero(self) -> None:
        from braincell._multi_compartment.currents import _is_python_zero
        self.assertFalse(_is_python_zero(True))

    def test_zero_float_still_matches(self) -> None:
        from braincell._multi_compartment.currents import _is_python_zero
        self.assertTrue(_is_python_zero(0))
        self.assertTrue(_is_python_zero(0.0))


class TotalMembraneCurrentNarrowExceptTest(unittest.TestCase):
    """MED-04: channel errors outside the numeric set must not be swallowed."""

    def test_non_numeric_exception_passes_through(self) -> None:
        import brainstate
        from braincell import Branch, CVPerBranch, Cell, Morphology
        from braincell._base import IonChannel
        from braincell._multi_compartment.currents import total_membrane_current

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.init_state()

        channels = cell.nodes(IonChannel, allowed_hierarchy=(1, 1))
        self.assertGreater(len(channels), 0)

        first_channel = next(iter(channels.values()))

        def boom(V):
            raise AttributeError("lookup failed")

        first_channel.current = boom  # type: ignore[assignment]

        with brainstate.environ.context(t=0.0 * u.ms):
            with self.assertRaises(AttributeError):
                total_membrane_current(
                    cell,
                    V_cv=cell.V.value,
                    I_ext=0.0,
                    t=0.0 * u.ms,
                )


if __name__ == "__main__":
    unittest.main()
