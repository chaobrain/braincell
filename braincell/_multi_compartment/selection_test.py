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

import inspect
import re
import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.filter import AllRegion, BranchSlice, LocsetBatch, LocsetMask, RootLocation, at, branch_in
from braincell.reduction.toy import EventAccumulatorReduction


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

    def test_a_missing_branch_type_names_itself(self) -> None:
        # On main every case raised "TypeError: branch ids must be a
        # one-dimensional integer selection."
        soma_only = braincell.Cell(braincell.Morphology.from_soma())
        lumped = braincell.Cell(braincell.Morphology.from_soma(), cv_policy=braincell.Lumped())
        cases = (
            ("axon", lambda: soma_only.axon, "['soma']"),
            ("apical_dendrite", lambda: lumped.apical_dendrite, "['soma']"),
            ("dendrite", lambda: soma_only[0].dendrite, "['soma']"),
            ("axn", lambda: _cell().branch.by_type("axn"), "['dendrite', 'soma']"),
        )
        for branch_type, select, types in cases:
            with self.subTest(branch_type=branch_type):
                message = f"no branch of type '{branch_type}'; its branch types are {types}"
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    select()
        self.assertEqual(soma_only.soma.cv.ids.tolist(), [0])

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


def _grid(kind="lumped", pop_size=(2, 3)):
    """A Cell whose population has several axes: lumped, or a 3-CV soma cable."""
    if kind == "lumped":
        cell = braincell.Cell.point(pop_size, V_init=-65.0 * u.mV)
    else:
        soma = braincell.Branch.from_lengths(lengths=[60.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = braincell.Cell(
            braincell.Morphology.from_root(soma), cv_policy=braincell.CVPerBranch(3), pop_size=pop_size
        )
    cell.paint(
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
    )
    cell.place(RootLocation(0.5), _clamp())
    return cell


#: Member selections of a (2, 3) population: the flat C-order members and the view shape (CELL-47).
_GRID_SELECTIONS = {
    "cell[1, 2]": (lambda cell: cell[1, 2], (5,), (1,)),
    "cell[0]": (lambda cell: cell[0], (0, 1, 2), (3,)),
    "cell[:, 1]": (lambda cell: cell[:, 1], (1, 4), (2,)),
    "cell[1, 0:2]": (lambda cell: cell[1, 0:2], (3, 4), (2,)),
    "cell[:, ::-2]": (lambda cell: cell[:, ::-2], (2, 0, 5, 3), (2, 2)),
    "cell[-1, -1]": (lambda cell: cell[-1, -1], (5,), (1,)),
    "cell[:, 1:][1, 0]": (lambda cell: cell[:, 1:][1, 0], (4,), (1,)),
    "cell[1][[2, 0]]": (lambda cell: cell[1][[2, 0]], (5, 3), (2,)),
    # A one-axis selection keeps the one-dimensional rules: a tuple lists members.
    "cell[1][2, 0]": (lambda cell: cell[1][2, 0], (5, 3), (2,)),
    "cell.loc(RootLocation(0.5))[1]": (lambda cell: cell.loc(RootLocation(0.5))[1], (3, 4, 5), (3,)),
    "cell[np.int64(1), np.int32(2)]": (lambda cell: cell[np.int64(1), np.int32(2)], (5,), (1,)),
    "cell[2:]": (lambda cell: cell[2:], (), (0, 3)),
}

#: CellView properties that need a one-dimensional pop_size (CELL-47): they read or write state.
_ONE_AXIS_PROPERTIES = frozenset(
    {
        "V",
        "spike",
        "V_init",
        "V_th",
        "channels",
        "ions",
        "geometry",
        "length",
        "Ra",
        "cm",
        "radius_mid",
        "radius_scale",
        "diam_arc_mean",
        "outputs",
        "reductions",
    }
)

#: CellView properties that behave on a multi-dimensional population as on a one-dimensional one.
_ANY_AXES_PROPERTIES = frozenset(
    {
        "apical_dendrite",
        "axon",
        "basal_dendrite",
        "branch",
        "cell",
        "clamps",
        "connections",
        "current_time",
        "cv",
        "cv_contexts",
        "cv_midpoints",
        "cv_policy",
        "cv_tree",
        "cvs",
        "dendrite",
        "event_outputs",
        "indices",
        "layouts",
        "locations",
        "membrane_linearizer",
        "morpho",
        "n_compartment",
        "n_cv",
        "n_point",
        "name",
        "node_tree",
        "paint_rules",
        "place_rules",
        "point_placements",
        "pop_size",
        "population_indices",
        "root",
        "shape",
        "size",
        "soma",
        "solver",
        "solver_name",
        "spatial_pairs",
        "spk_fun",
        "subsolver",
        "subsolver_name",
        "substeps",
        "synapses",
        "varshape",
        "voltage_shape",
    }
)


class GridSelectionTest(unittest.TestCase):
    """CELL-47: selecting members of a population with several axes."""

    def test_integers_and_slices_select_members_in_c_order(self) -> None:
        for kind in ("lumped", "cable"):
            cell = _grid(kind)
            for label, (select, members, shape) in _GRID_SELECTIONS.items():
                with self.subTest(kind=kind, selection=label):
                    view = select(cell)
                    self.assertEqual(view.population_indices, members)
                    self.assertEqual(view.shape, shape)
                    self.assertEqual(len(view), len(members))
                    self.assertEqual(view.pop_size, shape)
            self.assertEqual(cell.soma.shape, (2, 3))

    def test_a_third_axis_follows_the_same_rules(self) -> None:
        cell = _grid("lumped", (2, 2, 2))
        self.assertEqual(cell[1, :, 0].population_indices, (4, 6))
        self.assertEqual(cell[1, :, 0].shape, (2,))
        self.assertEqual(cell[1].shape, (2, 2))
        self.assertEqual(cell[1][1, 0].population_indices, (6,))

    def test_spatial_selections_keep_the_member_shape(self) -> None:
        cell = _grid("cable")
        self.assertEqual(cell.soma[1, 2].spatial_pairs.tolist(), cell[1, 2].soma.spatial_pairs.tolist())
        self.assertEqual(cell.soma[1, 2].spatial_pairs.tolist(), [[5, 0], [5, 1], [5, 2]])
        self.assertEqual(cell.cv[1].shape, (2, 3))
        self.assertEqual(cell.cv[1][:, 2].spatial_pairs.tolist(), [[2, 1], [5, 1]])

    def test_positional_members_are_flat(self) -> None:
        view = braincell.CellView(_grid("lumped"), (3, 4))
        self.assertEqual(view.population_indices, (3, 4))
        self.assertEqual(view.shape, (2,))

    def test_invalid_selections_raise_where_they_are_made(self) -> None:
        cases = {
            IndexError: (
                lambda cell: cell[2, 0],
                lambda cell: cell[0, 3],
                lambda cell: cell[-3, 0],
                lambda cell: cell[0, 1, 0],
                lambda cell: cell[0][3],
                lambda cell: cell.soma[2, 0],
            ),
            TypeError: (
                lambda cell: cell[[0, 1]],
                lambda cell: cell[np.array([0, 1])],
                lambda cell: cell[np.ones((2, 3), dtype=bool)],
                lambda cell: cell[True],
                lambda cell: cell[0, np.True_],
                lambda cell: cell[0.0],
                lambda cell: cell[..., 1],
                lambda cell: cell[None],
            ),
        }
        cell = _grid("lumped")
        for error, selections in cases.items():
            for index, select in enumerate(selections):
                with self.subTest(error=error.__name__, case=index):
                    with self.assertRaises(error) as caught:
                        select(cell).record("v", "V")
                    if error is TypeError:
                        self.assertIn("one integer or slice per axis", str(caught.exception))
                        self.assertIn("cell[0][[0, 2]]", str(caught.exception))
                    self.assertEqual(dict(cell.recordings), {})

    def test_views_of_a_grid_select_and_record_only(self) -> None:
        # Every public CellView property is classified; a new one fails here
        # until it is put in one of the two sets.
        properties = {
            name
            for name, member in inspect.getmembers(braincell.CellView, lambda item: isinstance(item, property))
            if not name.startswith("_")
        }
        self.assertEqual(properties, _ONE_AXIS_PROPERTIES | _ANY_AXES_PROPERTIES)
        self.assertFalse(_ONE_AXIS_PROPERTIES & _ANY_AXES_PROPERTIES)
        for kind in ("lumped", "cable"):
            grid, line = _grid(kind), _grid(kind, (6,))
            grid.init_state()
            line.init_state()
            for label, on_grid, on_line in (
                ("cell[1, 2]", grid[1, 2], line[5]),
                ("cell[:, 1:]", grid[:, 1:], line[[1, 2, 4, 5]]),
            ):
                for name in sorted(properties):
                    with self.subTest(kind=kind, view=label, property=name):
                        if name in _ONE_AXIS_PROPERTIES:
                            with self.assertRaisesRegex(
                                ValueError, "requires a one-dimensional pop_size; got \\(2, 3\\)"
                            ):
                                getattr(on_grid, name)
                            continue
                        try:
                            getattr(on_line, name)
                        except Exception as error:
                            with self.assertRaises(type(error)):
                                getattr(on_grid, name)
                        else:
                            getattr(on_grid, name)

    def test_grid_views_reject_writes_and_state_reads(self) -> None:
        cell = _grid("cable")
        cell.init_state()
        uses = {
            "set": lambda: cell[1, 2].set(V_init=-60.0 * u.mV),
            "paint": lambda: cell[0].paint(_leak()),
            "place": lambda: cell[0].loc(RootLocation(0.5)).place(_clamp()),
            "get_ion": lambda: cell[1, 2].get_ion("na"),
            "get_runtime_node": lambda: cell[1, 2].get_runtime_node(0),
            "root channels": lambda: cell.channels,
            "root ions": lambda: cell.ions,
            "root geometry": lambda: cell.geometry,
            "root outputs": lambda: cell.outputs,
        }
        for label, use in uses.items():
            with self.subTest(use=label):
                with self.assertRaisesRegex(ValueError, "one-dimensional pop_size"):
                    use()

    def test_model_outputs_need_one_axis(self) -> None:
        cell = _grid("lumped")
        for target in (cell, cell[0]):
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    ValueError, r"observe.output\(\.\.\.\) requires a one-dimensional pop_size"
                ):
                    target.record("out", braincell.observe.output("spike"))
                self.assertEqual(dict(cell.recordings), {})
        with self.assertRaisesRegex(ValueError, "one-dimensional pop_size"):
            cell.add_reduction("toy", EventAccumulatorReduction())


if __name__ == "__main__":
    unittest.main()
