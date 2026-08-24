import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.filter import BranchSlice, LocsetBatch, LocsetMask, at


def _cell(population_size=3):
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = braincell.Branch.from_lengths(
        lengths=[60.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="dendrite",
    )
    morpho = braincell.Morphology.from_root(soma, name="soma")
    morpho.soma.dend_a = dend
    return braincell.Cell(
        morpho,
        cv_policy=braincell.CVPerBranchList((1, 3)),
        pop_size=(population_size,),
    )


class CellSpatialSelectionTest(unittest.TestCase):
    def test_population_branch_and_cv_selection_compose(self) -> None:
        cell = _cell()
        view = cell[[2, 0]].branch["dend_a"].cv[1:]

        np.testing.assert_array_equal(view.spatial_pairs, [[2, 2], [2, 3], [0, 2], [0, 3]])
        np.testing.assert_array_equal(view.cv.ids, [2, 3])
        np.testing.assert_array_equal(view.cv.by_id([3, 0, 99]).cv.ids, [3])
        self.assertEqual(cell.soma.cv.ids.tolist(), [0])
        self.assertEqual(cell.dendrite.cv.ids.tolist(), [1, 2, 3])

    def test_region_preserves_positive_coverage(self) -> None:
        cell = _cell()
        view = cell.on(BranchSlice(branch_index=1, prox=0.0, dist=0.5))

        np.testing.assert_array_equal(view.cv.ids, [1, 2])
        self.assertTrue(np.all(view.cv.coverage_fraction > 0.0))
        self.assertLess(view.cv.coverage_fraction[-1], 1.0)

    def test_locset_preserves_duplicate_rows_and_deduplicates_pairs(self) -> None:
        cell = _cell(population_size=2)
        locations = LocsetMask.from_columns([1, 1, 1], [0.1, 0.1, 0.8])
        view = cell[1].loc(locations)

        self.assertEqual(len(view.locations), 3)
        np.testing.assert_array_equal(view.spatial_pairs, [[1, 1], [1, 3]])

    def test_locset_batch_maps_rows_to_selected_population(self) -> None:
        cell = _cell(population_size=3)
        locations = LocsetBatch.from_columns(
            [[0, 1], [1, 1]],
            [[0.5, 0.1], [0.5, 0.9]],
        )
        view = cell[[2, 0]].loc(locations)

        np.testing.assert_array_equal(view.spatial_pairs, [[2, 0], [2, 1], [0, 2], [0, 3]])

    def test_at_x_requires_one_exact_branch(self) -> None:
        cell = _cell()
        np.testing.assert_array_equal(cell[0].branch[1].at_x(0.5).spatial_pairs, [[0, 2]])
        with self.assertRaisesRegex(ValueError, "exactly one branch"):
            cell.dendrite.on(BranchSlice(1, 0.0, 1.0)).at_x(0.5)

    def test_spatial_view_cannot_place(self) -> None:
        cell = _cell()
        with self.assertRaisesRegex(RuntimeError, "Spatial CellView.place"):
            cell.soma.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn"))


if __name__ == "__main__":
    unittest.main()
