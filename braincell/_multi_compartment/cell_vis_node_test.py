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


class CellVisNodeTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_vis_node_region_maps_to_midpoint_point_ids(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        target_point_id = int(cell.point_tree().cv_midpoint_point_id[1])

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), show=False)

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_vis_node_locset_maps_to_midpoint_point_ids(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        target_point_id = int(cell.point_tree().cv_midpoint_point_id[0])

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(locset=RootLocation(0.5), show=False)

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_vis_node_rejects_value_with_region(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            cell.vis_node(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), value="V", show=False)

    def test_vis_node_converts_raw_cv_values_to_point_values(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        cv_values = np.asarray([1.0, 2.0], dtype=float)
        expected = cell._cv_to_node_values(cv_values)

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=cv_values, show=False)

        np.testing.assert_allclose(
            np.nan_to_num(mocked.call_args.kwargs["values"], nan=-1.0),
            np.nan_to_num(expected, nan=-1.0),
        )

    def test_vis_node_value_V_uses_point_voltage(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        expected = cell._cv_to_node_values(cell.V.value)

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value="V", show=False)

        np.testing.assert_allclose(
            mocked.call_args.kwargs["values"].to_decimal(u.mV),
            expected.to_decimal(u.mV),
        )
        self.assertEqual(mocked.call_args.kwargs["value_label"], "V")

    def test_vis_node_channel_lookup_ambiguous_raises(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV))
        cell.paint(full, mech.Channel("IL", name="leak_b", g_max=5.0 * (u.mS / u.cm ** 2), E=-67.0 * u.mV))
        cell.init_state()

        with self.assertRaisesRegex(ValueError, "multiple runtime layouts"):
            cell.vis_node(value=("channel", "IL", "g_max"), show=False)

    def test_vis_node_layout_id_lookup_works(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=("layout_id", layout.id, "g_max"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_point,))

    def test_vis_node_ion_lookup_works(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=("ion", "na", "E"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_point,))
        self.assertEqual(mocked.call_args.kwargs["value_label"], "na.E")

    def test_vis_node_region_coverage_is_forwarded(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="fraction",
                show=False,
            )

        self.assertIn("highlight_fractions", mocked.call_args.kwargs)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_topology_node_dispatches(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with mock.patch.object(cell, "vis_node", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="node", show=False)
        mocked.assert_called_once()

    def test_vis_node_smoke_returns_axes(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        ax = cell.vis_node(value="V", show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main()
