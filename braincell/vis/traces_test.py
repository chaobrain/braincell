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
import matplotlib.pyplot as plt
import numpy as np

from braincell.filter import Terminals
from braincell.vis._testing import make_length_only_tree
from braincell.vis.traces import TracesResult, plot_traces


class PlotTracesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = make_length_only_tree()
        self.t = np.linspace(0.0, 10.0, 20) * u.ms

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_traces_returns_axes_matching_trace_count(self) -> None:
        # make_length_only_tree() has two branches; Terminals() returns
        # the terminal (dendrite tip) only, so we synthesize two points
        # by asking for head+tail of the soma.
        locset = Terminals().evaluate(self.tree)
        n_locs = len(locset.points)
        values = np.random.default_rng(0).standard_normal((self.t.shape[0], n_locs))

        result = plot_traces(self.tree, self.t, values, locset=locset)

        self.assertIsInstance(result, TracesResult)
        self.assertEqual(len(result.trace_axes), n_locs)
        self.assertIsNotNone(result.morpho_axes)

    def test_plot_traces_without_morphology_still_returns_trace_axes(self) -> None:
        values = np.zeros((self.t.shape[0], 2))
        result = plot_traces(self.tree, self.t, values, show_morphology=False)
        self.assertEqual(len(result.trace_axes), 2)
        self.assertIsNone(result.morpho_axes)

    def test_plot_traces_detects_time_mismatch(self) -> None:
        values = np.zeros((self.t.shape[0] + 3, 2))
        with self.assertRaisesRegex(ValueError, "time length"):
            plot_traces(self.tree, self.t, values)

    def test_plot_traces_validates_locset_count(self) -> None:
        locset = Terminals().evaluate(self.tree)
        values = np.zeros((self.t.shape[0], len(locset.points) + 1))
        with self.assertRaisesRegex(ValueError, "locset has"):
            plot_traces(self.tree, self.t, values, locset=locset)

    def test_plot_traces_labels_override_default(self) -> None:
        values = np.zeros((self.t.shape[0], 2))
        result = plot_traces(
            self.tree,
            self.t,
            values,
            labels=["soma", "tip"],
            show_morphology=False,
        )
        ylabels = [ax.get_ylabel() for ax in result.trace_axes]
        self.assertEqual(ylabels, ["soma", "tip"])

    def test_plot_traces_label_count_mismatch_raises(self) -> None:
        values = np.zeros((self.t.shape[0], 2))
        with self.assertRaisesRegex(ValueError, "1 labels"):
            plot_traces(
                self.tree,
                self.t,
                values,
                labels=["only-one"],
                show_morphology=False,
            )


if __name__ == "__main__":
    unittest.main()
