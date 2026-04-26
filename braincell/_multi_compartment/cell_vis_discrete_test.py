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
from unittest import mock

import brainunit as u
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import braincell.mech as mech
from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell.filter import BranchSlice, RootLocation


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class CellVisDiscreteTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_vis_cv_smoke_returns_axes_before_init(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        ax = cell.vis_cv(show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_vis_branch_smoke_returns_axes_before_init(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        ax = cell.vis_branch(show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_vis_cv_fraction_passes_partial_coverage(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(region=BranchSlice(branch_index=1, prox=0.0, dist=0.5), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertAlmostEqual(coverage[1], 0.5)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_cv_any_mode_is_forwarded(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="any",
                show=False,
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "any")

    def test_vis_cv_locset_is_supported(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(locset=RootLocation(0.5), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertEqual(coverage[0], 1.0)

    def test_vis_cv_value_V_uses_cv_values(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value="V", show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_cv,))

    def test_vis_cv_raw_point_values_are_gathered_to_cv(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        point_values = np.asarray([np.nan, 1.0, np.nan, 2.0, np.nan], dtype=float)
        expected = cell._point_to_cv(point_values)

        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value=point_values, show=False)

        np.testing.assert_allclose(mocked.call_args.kwargs["values"], expected)

    def test_vis_cv_layout_id_lookup_works(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value=("layout_id", layout.id, "g_max"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_cv,))

    def test_vis_branch_fraction_passes_partial_branch_coverage(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_branch(region=BranchSlice(branch_index=1, prox=0.25, dist=0.75), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertAlmostEqual(coverage[1], 0.5)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_branch_all_mode_is_forwarded(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_branch(
                region=BranchSlice(branch_index=1, prox=0.0, dist=1.0),
                coverage_mode="all",
                show=False,
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "all")

    def test_vis_topology_cv_dispatches(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch.object(cell, "vis_cv", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="cv", show=False)
        mocked.assert_called_once()

    def test_vis_topology_branch_dispatches(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with mock.patch.object(cell, "vis_branch", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="branch", show=False)
        mocked.assert_called_once()

    def test_vis_topology_branch_rejects_locset(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with self.assertRaisesRegex(ValueError, "does not support locset"):
            cell.vis_topology(level="branch", locset=RootLocation(0.5), show=False)

    def test_vis_topology_branch_rejects_value(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        with self.assertRaisesRegex(ValueError, "does not support value"):
            cell.vis_topology(level="branch", value="V", show=False)


if __name__ == "__main__":
    unittest.main()
