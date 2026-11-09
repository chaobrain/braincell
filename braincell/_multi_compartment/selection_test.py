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

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.filter import AllRegion, BranchSlice, LocsetBatch, LocsetMask, RootLocation, at, branch_in


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

    def test_region_view_cannot_place(self) -> None:
        cell = _cell()
        with self.assertRaisesRegex(ValueError, "needs a location view"):
            cell.soma.place(braincell.mech.Synapse("ExpSyn"))
        self.assertEqual(cell.place_rules, ())


def _leak():
    return braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV)


def _clamp():
    return braincell.mech.CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)


class ViewDeclarationVerbTest(unittest.TestCase):
    """CELL-22: the same paint/place verbs on the Cell and its spatial views."""

    def test_paint_without_region_paints_the_whole_cell(self) -> None:
        implicit, explicit = _cell(), _cell()
        self.assertIs(implicit.paint(_leak()), implicit)
        explicit.paint(AllRegion(), _leak())
        self.assertEqual(implicit.paint_rules, explicit.paint_rules)

    def test_region_views_paint_their_region(self) -> None:
        cases = (
            ("soma", lambda cell: cell.soma, branch_in("type", ["soma"])),
            ("on", lambda cell: cell.on(branch_in("type", ["dendrite"])), branch_in("type", ["dendrite"])),
            ("branch by name", lambda cell: cell.branch["dend_a"], branch_in("name", ["dend_a"])),
            ("branch by index", lambda cell: cell.branch[1], branch_in("index", [1])),
            (
                "nested",
                lambda cell: cell.on(branch_in("type", ["dendrite"])).branch["dend_a"],
                branch_in("type", ["dendrite"]) & branch_in("name", ["dend_a"]),
            ),
            ("root view", lambda cell: cell[:], AllRegion()),
        )
        for label, select, region in cases:
            with self.subTest(view=label):
                through_view, direct = _cell(), _cell()
                self.assertIs(select(through_view).paint(_leak()), through_view)
                direct.paint(region, _leak())
                self.assertEqual(through_view.paint_rules, direct.paint_rules)

    def test_location_view_places_any_point_mechanism(self) -> None:
        for mechanism in (_clamp(), braincell.mech.Synapse("ExpSyn", name="syn")):
            with self.subTest(mechanism=type(mechanism).__name__):
                through_view, direct = _cell(), _cell()
                self.assertIs(through_view.loc(RootLocation(0.5)).place(mechanism), through_view)
                direct.place(RootLocation(0.5), mechanism)
                self.assertEqual(through_view.place_rules, direct.place_rules)

    def test_population_location_view_places_on_selected_members(self) -> None:
        through_view, direct = _cell(), _cell()
        synapse = braincell.mech.Synapse("ExpSyn", name="syn")
        self.assertIs(through_view[1:].loc(at("soma", 0.5)).place(synapse), through_view)
        direct[1:].place(at("soma", 0.5), synapse)
        self.assertEqual(through_view.place_rules, direct.place_rules)

    def test_lumped_cell_place_without_locset_is_unchanged(self) -> None:
        implicit = braincell.Cell.point(2)
        implicit.place(_clamp())
        explicit = braincell.Cell.point(2)
        explicit.place(explicit.cv_midpoints, _clamp())
        self.assertEqual(implicit.place_rules, explicit.place_rules)
        implicit.paint(_leak())
        self.assertEqual(implicit.paint_rules[-1].region, AllRegion())

    def test_declaring_through_views_after_init_raises(self) -> None:
        cell = _cell()
        cell.paint(_leak())
        cell.init_state()
        uses = {
            "paint": lambda: cell.paint(_leak()),
            "soma.paint": lambda: cell.soma.paint(_leak()),
            "loc.place": lambda: cell.loc(RootLocation(0.5)).place(_clamp()),
        }
        for label, use in uses.items():
            with self.subTest(use=label):
                with self.assertRaisesRegex(RuntimeError, "after init_state"):
                    use()

    def test_misuse_raises_at_the_call(self) -> None:
        cases = (
            ("population view paint", ValueError, "every population member alike", lambda c: c[0].paint(_leak())),
            ("population spatial paint", ValueError, "every population member", lambda c: c[0].soma.paint(_leak())),
            ("no mechanism", TypeError, "at least one mechanism", lambda c: c.paint()),
            ("region and view region", TypeError, "its own region", lambda c: c.soma.paint(AllRegion(), _leak())),
            ("cv view paint", ValueError, "control-volume selection", lambda c: c.cv[0].paint(_leak())),
            (
                "location view paint",
                ValueError,
                "location selection",
                lambda c: c.loc(RootLocation(0.5)).paint(_leak()),
            ),
            (
                "locset and location view",
                TypeError,
                "its own locations",
                lambda c: c.loc(RootLocation(0.5)).place(RootLocation(0.5), _clamp()),
            ),
            (
                "nested location view",
                ValueError,
                "inside another spatial selection",
                lambda c: c.soma.loc(RootLocation(0.5)).place(_clamp()),
            ),
            ("population view without locset", TypeError, "needs a locset", lambda c: c[0].place()),
            ("region inside cv view", ValueError, "control-volume selection", lambda c: c.cv[0].soma.paint(_leak())),
            (
                "region inside location view",
                ValueError,
                "inside a location",
                lambda c: c.loc(RootLocation(0.5)).soma.paint(_leak()),
            ),
            (
                "location inside cv view",
                ValueError,
                "control-volume selection",
                lambda c: c.cv[0].loc(RootLocation(0.5)).place(_clamp()),
            ),
        )
        for label, error, message, use in cases:
            with self.subTest(case=label):
                cell = _cell()
                with self.assertRaisesRegex(error, message):
                    use(cell)
                self.assertEqual(len(cell.paint_rules), 1)
                self.assertEqual(cell.place_rules, ())


if __name__ == "__main__":
    unittest.main()
