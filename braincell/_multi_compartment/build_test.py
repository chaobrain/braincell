"""Tests for :func:`braincell._multi_compartment.build.build`."""

from __future__ import annotations

import unittest

import brainunit as u

from braincell import Branch, CVPerBranch, Cell, Morphology, RunnableCell


class TestBuildPipeline(unittest.TestCase):
    def _simple_cell(self) -> Cell:
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        return Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())

    def test_build_returns_runnable_cell(self):
        cell = self._simple_cell()
        rcell = cell.build()
        self.assertIsInstance(rcell, RunnableCell)
        self.assertGreater(rcell.n_cv, 0)
        self.assertGreater(rcell.n_point, 0)

    def test_runnable_has_runtime_state(self):
        cell = self._simple_cell()
        rcell = cell.build()
        self.assertIsNotNone(rcell.runtime)
        self.assertIsNotNone(rcell.point_tree())
        self.assertIsNotNone(rcell._axial_jax)

    def test_build_twice_produces_independent_runnables(self):
        cell = self._simple_cell()
        a = cell.build()
        b = cell.build()
        self.assertIsNot(a, b)
        self.assertIsNot(a.V, b.V)

    def test_clamp_active_table_set_on_runtime(self):
        cell = self._simple_cell()
        rcell = cell.build()
        # No clamps placed -> table stays None.
        self.assertIsNone(rcell.runtime.clamp_active_table)

    def test_cv_area_set_on_runtime(self):
        cell = self._simple_cell()
        rcell = cell.build()
        self.assertIsNotNone(rcell.runtime.cv_area)
        self.assertEqual(rcell.runtime.cv_area.shape, (rcell.n_cv,))


if __name__ == "__main__":
    unittest.main()
