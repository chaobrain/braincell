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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from braincell.vis import plot2d
from braincell.vis._testing import make_length_only_tree
from braincell.vis.backend_matplotlib import (
    _BC_PICK_META,
    _find_hover_meta,
    _pick_info_from_meta,
    _resolve_pick_meta,
    connect_hooks,
)
from braincell.vis.hooks import PickInfo, VisHooks


class VisHooksDataclassTest(unittest.TestCase):
    def test_is_active_returns_false_without_callbacks(self) -> None:
        self.assertFalse(VisHooks().is_active())

    def test_is_active_true_when_any_callback_present(self) -> None:
        hooks = VisHooks(on_pick=lambda info: None)
        self.assertTrue(hooks.is_active())


class MatplotlibPickMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = make_length_only_tree()

    def tearDown(self) -> None:
        plt.close("all")

    def test_base_polylines_carry_pick_metadata(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)

        pick_lines = [line for line in ax.lines if hasattr(line, _BC_PICK_META)]
        self.assertGreater(len(pick_lines), 0)
        meta = getattr(pick_lines[0], _BC_PICK_META)
        self.assertIn("branch_index", meta)
        self.assertIn("branch_name", meta)
        self.assertIn("branch_type", meta)
        plt.close(fig)

    def test_base_frustum_polygons_carry_pick_metadata(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="frustum", ax=ax)

        pick_patches = [p for p in ax.patches if hasattr(p, _BC_PICK_META)]
        self.assertGreater(len(pick_patches), 0)
        meta = getattr(pick_patches[0], _BC_PICK_META)
        self.assertIn("branch_index", meta)
        plt.close(fig)

    def test_value_line_collection_carries_per_segment_metadata(self) -> None:
        fig, ax = plt.subplots()
        values = np.linspace(0.0, 1.0, len(self.tree.branches))
        plot2d(self.tree, layout="stem", shape="line", values=values, ax=ax, show_colorbar=False)

        from matplotlib.collections import LineCollection

        line_collections = [
            coll for coll in ax.collections
            if isinstance(coll, LineCollection) and hasattr(coll, _BC_PICK_META)
        ]
        self.assertGreater(len(line_collections), 0)
        meta = getattr(line_collections[0], _BC_PICK_META)
        self.assertIsInstance(meta, list)
        self.assertGreater(len(meta), 0)
        self.assertIn("segment_index", meta[0])
        self.assertIn("value", meta[0])
        plt.close(fig)


class ResolvePickMetaTest(unittest.TestCase):
    def test_single_artist_returns_meta_directly(self) -> None:
        artist = SimpleNamespace()
        setattr(artist, _BC_PICK_META, {"branch_index": 7, "branch_name": "x", "branch_type": "soma"})
        event = SimpleNamespace(ind=None)
        meta = _resolve_pick_meta(artist, event)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["branch_index"], 7)

    def test_batched_artist_uses_event_ind(self) -> None:
        artist = SimpleNamespace()
        setattr(artist, _BC_PICK_META, [
            {"branch_index": 0, "branch_name": "a", "branch_type": "soma", "segment_index": 0},
            {"branch_index": 0, "branch_name": "a", "branch_type": "soma", "segment_index": 1},
        ])
        event = SimpleNamespace(ind=[1])
        meta = _resolve_pick_meta(artist, event)
        self.assertEqual(meta["segment_index"], 1)

    def test_missing_meta_returns_none(self) -> None:
        artist = SimpleNamespace()
        event = SimpleNamespace(ind=None)
        self.assertIsNone(_resolve_pick_meta(artist, event))


class PickInfoConversionTest(unittest.TestCase):
    def test_segment_meta_yields_midpoint_x_and_value(self) -> None:
        meta = {
            "branch_index": 3,
            "branch_name": "dend",
            "branch_type": "apical_dendrite",
            "segment_index": 1,
            "x_start": 0.25,
            "x_end": 0.5,
            "value": 0.42,
        }
        info = _pick_info_from_meta(meta, artist=None, xdata=1.0, ydata=2.0)
        self.assertIsInstance(info, PickInfo)
        self.assertEqual(info.branch_index, 3)
        self.assertAlmostEqual(info.x or 0.0, 0.375)
        self.assertAlmostEqual(info.value or 0.0, 0.42)
        self.assertIsNotNone(info.position_um)


class ConnectHooksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = make_length_only_tree()

    def tearDown(self) -> None:
        plt.close("all")

    def test_connect_hooks_returns_empty_for_inactive_hooks(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)
        ids = connect_hooks(ax, VisHooks())
        self.assertEqual(ids, {})
        plt.close(fig)

    def test_connect_hooks_registers_pick_handler(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)
        ids = connect_hooks(ax, VisHooks(on_pick=lambda info: None))
        self.assertIn("pick", ids)
        plt.close(fig)

    def test_plot2d_with_hooks_enables_pickers(self) -> None:
        fig, ax = plt.subplots()
        captured: list[PickInfo] = []
        plot2d(
            self.tree,
            layout="stem",
            shape="line",
            ax=ax,
            hooks=VisHooks(on_pick=lambda info: captured.append(info)),
        )
        pick_lines = [line for line in ax.lines if hasattr(line, _BC_PICK_META)]
        for line in pick_lines:
            self.assertTrue(line.get_picker())
        plt.close(fig)


class HoverMetaLookupTest(unittest.TestCase):
    """``_find_hover_meta`` should return ``None`` outside the axes."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_event_outside_axes_returns_none(self) -> None:
        fig, ax = plt.subplots()
        plot2d(make_length_only_tree(), layout="stem", shape="line", ax=ax)
        event = SimpleNamespace(inaxes=None, xdata=None, ydata=None)
        self.assertIsNone(_find_hover_meta(ax, event))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
