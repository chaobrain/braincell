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

import matplotlib.pyplot as plt
import numpy as np

from braincell.vis._testing import make_length_only_tree
from braincell.vis.movie import MovieResult, plot_movie


class PlotMovie2dTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = make_length_only_tree()

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_movie_returns_animation_and_frame_count(self) -> None:
        values = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.3],
                [1.0, 0.9],
            ]
        )

        self.result = plot_movie(self.tree, values, dimensionality="2d", shape="line")

        self.assertIsInstance(self.result, MovieResult)
        self.assertEqual(self.result.frames, 3)
        self.assertIsNone(self.result.output_path)
        self.assertIsNotNone(self.result.animation)

    def test_plot_movie_rejects_non_2d_values_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            plot_movie(self.tree, np.array([1.0, 2.0]), dimensionality="2d")

    def test_plot_movie_rejects_empty_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            plot_movie(self.tree, np.zeros((0, 2)), dimensionality="2d")

    def test_plot_movie_accepts_existing_ax(self) -> None:
        fig, ax = plt.subplots()
        values = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = plot_movie(self.tree, values, dimensionality="2d", shape="line", ax=ax)

        self.assertEqual(result.frames, 2)
        # At least one value-bearing collection should have been added.
        from matplotlib.collections import LineCollection, PolyCollection

        has_value_collection = any(
            isinstance(coll, (LineCollection, PolyCollection)) and coll.get_array() is not None
            for coll in ax.collections
        )
        self.assertTrue(has_value_collection)
        plt.close(fig)

    def test_plot_movie_rejects_unknown_dimensionality(self) -> None:
        values = np.array([[0.1, 0.2]])
        with self.assertRaisesRegex(ValueError, "must be '2d' or '3d'"):
            plot_movie(self.tree, values, dimensionality="4d")

    def test_plot_movie_auto_bounds_when_vmin_vmax_equal(self) -> None:
        # All-zero values previously produced a degenerate Normalize;
        # the movie builder pads to ±0.5 so matplotlib doesn't warn.
        values = np.zeros((2, 2))
        result = plot_movie(self.tree, values, dimensionality="2d", shape="line")
        self.assertEqual(result.frames, 2)


if __name__ == "__main__":
    unittest.main()
