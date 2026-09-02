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

"""Tests for :mod:`braincell.vis.cell_topology`."""

import dataclasses
import unittest
from unittest import mock

import brainunit as u
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braincell import Cell, mech
from braincell._discretization import CVPerBranch
from braincell._multi_compartment import field_resolution
from braincell.filter import BranchSlice, RootLocation
from braincell.vis import plot_cell_topology
from braincell.vis._testing import make_soma_dend_tree

_DISCRETE_RENDERER = "braincell.vis.cell_topology._plot_discrete_topology_graph"
_POINT_RENDERER = "braincell.vis.cell_topology.plot_point_topology"


def _cell(**kwargs) -> Cell:
    return Cell(make_soma_dend_tree(), cv_policy=CVPerBranch(), **kwargs)


class PlotCellTopologyGuardTest(unittest.TestCase):
    """Argument validation that happens before any level is selected."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_a_non_cell_first_argument_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"expects Cell, got Morphology"):
            plot_cell_topology(make_soma_dend_tree())

    def test_an_unknown_level_names_the_valid_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, r"level must be one of \{.*'node'.*'cv'.*'branch'.*\}"):
            plot_cell_topology(_cell(), level="layout")

    def test_node_level_requires_init_state(self) -> None:
        # The only level that reads runtime state, and the only one that
        # therefore refuses a declaration-only cell.
        with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
            plot_cell_topology(_cell(), level="node")


class PlotCellTopologyCvLevelTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_smoke_returns_axes_before_init(self) -> None:
        self.assertIsInstance(plot_cell_topology(_cell(), level="cv"), matplotlib.axes.Axes)

    def test_fraction_passes_partial_coverage(self) -> None:
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(_cell(), level="cv", region=BranchSlice(branch_index=1, prox=0.0, dist=0.5))

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        # CV shading is the membrane-area fraction, the same one that scales
        # a painted conductance. The dendrite tapers, so its proximal half
        # is 0.5833... of the area and only 0.5 of the length. Branch-level
        # shading below is a drawing extent and stays length-based.
        self.assertAlmostEqual(coverage[1], 0.5833333333333334)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_any_mode_is_forwarded(self) -> None:
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(
                _cell(),
                level="cv",
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="any",
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "any")

    def test_locset_is_supported(self) -> None:
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(_cell(), level="cv", locset=RootLocation(0.5))

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][0], 1.0)

    def test_value_V_uses_cv_values(self) -> None:
        cell = _cell()
        cell.init_state()
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, level="cv", value="V")

        self.assertEqual(mocked.call_args.kwargs["values"].shape, (cell.n_cv,))

    def test_raw_point_values_are_gathered_to_cv(self) -> None:
        cell = _cell()
        cell.init_state()
        point_values = np.asarray([np.nan, 1.0, np.nan, 2.0, np.nan], dtype=float)
        expected = cell._point_to_cv(point_values)

        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, level="cv", value=point_values)

        np.testing.assert_allclose(mocked.call_args.kwargs["values"], expected)

    def test_layout_id_lookup_works(self) -> None:
        cell = _cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, level="cv", value=("layout_id", layout.id, "g_max"))

        self.assertEqual(mocked.call_args.kwargs["values"].shape, (cell.n_cv,))

    def test_value_with_region_is_rejected(self) -> None:
        cell = _cell()
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            plot_cell_topology(
                cell,
                level="cv",
                region=BranchSlice(branch_index=1, prox=0.0, dist=1.0),
                value="V",
            )

    def test_more_than_one_root_cv_is_rejected(self) -> None:
        # A forest has no single node to lay the graph out from. Branch
        # level tolerates this; CV level does not, and the asymmetry is
        # deliberate. A Morphology always has one root, so the second one
        # has to be injected to reach the guard at all.
        cell = _cell()
        cvs = tuple(cell.cvs)
        forest = cvs + (dataclasses.replace(cvs[-1], id=len(cvs), parent_cv=None),)
        with mock.patch.object(Cell, "cvs", new_callable=mock.PropertyMock) as mocked_cvs:
            mocked_cvs.return_value = forest
            with self.assertRaisesRegex(ValueError, "expects exactly one root CV"):
                plot_cell_topology(cell, level="cv")


class PlotCellTopologyBranchLevelTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_smoke_returns_axes_before_init(self) -> None:
        self.assertIsInstance(plot_cell_topology(_cell(), level="branch"), matplotlib.axes.Axes)

    def test_fraction_passes_partial_branch_coverage(self) -> None:
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(_cell(), level="branch", region=BranchSlice(branch_index=1, prox=0.25, dist=0.75))

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertAlmostEqual(coverage[1], 0.5)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_all_mode_is_forwarded(self) -> None:
        with mock.patch(_DISCRETE_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(
                _cell(),
                level="branch",
                region=BranchSlice(branch_index=1, prox=0.0, dist=1.0),
                coverage_mode="all",
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "all")

    def test_locset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support: locset"):
            plot_cell_topology(_cell(), level="branch", locset=RootLocation(0.5))

    def test_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support: value"):
            plot_cell_topology(_cell(), level="branch", value="V")

    def test_every_value_colormap_parameter_is_rejected(self) -> None:
        for name, argument in (
            ("cmap", "viridis"),
            ("vmin", 0.0),
            ("vmax", 1.0),
            ("norm", object()),
            ("value_label", "V"),
            ("show_colorbar", False),
        ):
            with self.subTest(parameter=name):
                with self.assertRaisesRegex(ValueError, f"does not support: {name}"):
                    plot_cell_topology(_cell(), level="branch", **{name: argument})

    def test_several_unsupported_parameters_are_reported_together(self) -> None:
        # One error listing everything beats making the caller re-run and
        # discover the rejections one at a time.
        with self.assertRaisesRegex(ValueError, r"does not support: locset, value, cmap\."):
            plot_cell_topology(
                _cell(),
                level="branch",
                locset=RootLocation(0.5),
                value="V",
                cmap="viridis",
            )


class PlotCellTopologyNodeLevelTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_node_is_the_default_level(self) -> None:
        cell = _cell()
        cell.init_state()
        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, value="V")
        mocked.assert_called_once()

    def test_region_maps_to_midpoint_point_ids(self) -> None:
        cell = _cell()
        cell.init_state()
        target_point_id = int(cell.node_tree.cv_to_mid_node_id[1])

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, region=BranchSlice(branch_index=1, prox=0.0, dist=1.0))

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_locset_maps_to_midpoint_point_ids(self) -> None:
        cell = _cell()
        cell.init_state()
        target_point_id = int(cell.node_tree.cv_to_mid_node_id[0])

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, locset=RootLocation(0.5))

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_value_with_region_is_rejected(self) -> None:
        cell = _cell()
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            plot_cell_topology(cell, region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), value="V")

    def test_raw_cv_values_are_converted_to_point_values(self) -> None:
        cell = _cell()
        cell.init_state()
        cv_values = np.asarray([1.0, 2.0], dtype=float)
        expected = field_resolution.cv_to_node_values(cell, cv_values)

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, value=cv_values)

        np.testing.assert_allclose(
            np.nan_to_num(mocked.call_args.kwargs["values"], nan=-1.0),
            np.nan_to_num(expected, nan=-1.0),
        )

    def test_value_V_uses_point_voltage(self) -> None:
        cell = _cell()
        cell.init_state()
        expected = field_resolution.cv_to_node_values(cell, cell.V.value)

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, value="V")

        np.testing.assert_allclose(
            mocked.call_args.kwargs["values"].to_decimal(u.mV),
            expected.to_decimal(u.mV),
        )
        self.assertEqual(mocked.call_args.kwargs["value_label"], "V")

    def test_channel_lookup_ambiguous_raises(self) -> None:
        cell = _cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.paint(full, mech.Channel("IL", name="leak_b", g_max=5.0 * (u.mS / u.cm**2), E=-67.0 * u.mV))
        cell.init_state()

        with self.assertRaisesRegex(ValueError, "multiple runtime layouts"):
            plot_cell_topology(cell, value=("channel", "IL", "g_max"))

    def test_layout_id_lookup_works(self) -> None:
        cell = _cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, value=("layout_id", layout.id, "g_max"))

        self.assertEqual(mocked.call_args.kwargs["values"].shape, (cell.n_point,))

    def test_ion_lookup_works(self) -> None:
        cell = _cell()
        cell.init_state()

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(cell, value=("ion", "na", "E"))

        self.assertEqual(mocked.call_args.kwargs["values"].shape, (cell.n_point,))
        self.assertEqual(mocked.call_args.kwargs["value_label"], "na.E")

    def test_region_coverage_is_forwarded(self) -> None:
        cell = _cell()
        cell.init_state()

        with mock.patch(_POINT_RENDERER, autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            plot_cell_topology(
                cell,
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="fraction",
            )

        self.assertIn("highlight_fractions", mocked.call_args.kwargs)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_smoke_returns_axes(self) -> None:
        cell = _cell()
        cell.init_state()
        self.assertIsInstance(plot_cell_topology(cell, value="V"), matplotlib.axes.Axes)


class PlotCellTopologyPopulationAxisTest(unittest.TestCase):
    """Visualization draws one morphology, so it needs one population member."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_default_single_member_population_still_draws(self) -> None:
        cell = _cell()
        cell.init_state()
        self.assertEqual(cell.pop_size, (1,))
        for level in ("cv", "node"):
            with self.subTest(level=level):
                ax = plot_cell_topology(cell, level=level, value="V")
                self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_multi_member_population_is_refused_with_a_useful_message(self) -> None:
        cell = _cell(pop_size=4)
        cell.init_state()
        for level in ("cv", "node"):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, r"population shape \(4,\).*pop_size=\(4,\)"):
                    plot_cell_topology(cell, level=level, value="V")

    def test_the_refusal_names_the_level_that_was_asked_for(self) -> None:
        cell = _cell(pop_size=4)
        cell.init_state()
        with self.assertRaisesRegex(ValueError, r"plot_cell_topology\(level='cv', \.\.\.\)"):
            plot_cell_topology(cell, level="cv", value="V")


if __name__ == "__main__":
    unittest.main()
