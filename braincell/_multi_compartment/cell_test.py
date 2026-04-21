"""Unit tests for :class:`braincell.Cell`."""

from __future__ import annotations

import unittest

import brainunit as u

from braincell import Branch, CVPerBranch, Cell, CurrentClamp, Morphology
from braincell.filter import RootLocation


def _simple_cell() -> Cell:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())


class TestCellDeclaration(unittest.TestCase):
    def test_constructs_with_defaults(self):
        cell = _simple_cell()
        self.assertGreater(len(cell.paint_rules), 0)
        self.assertEqual(len(cell.place_rules), 0)

    def test_rejects_non_morphology(self):
        with self.assertRaises(TypeError):
            Cell("not-a-morpho")  # type: ignore[arg-type]

    def test_cv_policy_mutation_invalidates_cache(self):
        cell = _simple_cell()
        _ = cell.cvs
        cell.cv_policy = CVPerBranch()
        _ = cell.cvs

    def test_paint_returns_self_for_chaining(self):
        cell = _simple_cell()
        from braincell import CableProperty
        result = cell.paint(
            cell.paint_rules[0].region,
            CableProperty(
                resting_potential=-70 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm ** 2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
        )
        self.assertIs(result, cell)

    def test_place_dedups_identical_rules(self):
        cell = _simple_cell()
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms, delay=1 * u.ms)
        cell.place(RootLocation(0.0), clamp)
        cell.place(RootLocation(0.0), clamp)
        self.assertEqual(len(cell.place_rules), 1)

    def test_build_twice_produces_independent_runnables(self):
        cell = _simple_cell()
        a = cell.build()
        b = cell.build()
        self.assertIsNot(a, b)
        self.assertEqual(a.n_cv, b.n_cv)


if __name__ == "__main__":
    unittest.main()
