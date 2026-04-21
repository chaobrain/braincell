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


import sys
import types
import unittest
from unittest import mock

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.vis.backend_plotly import PlotlyBackend
from braincell.vis.scene import OverlaySpec, RenderRequest, ValueSpec
from braincell.vis.scene3d import build_render_scene_3d


class _FakeScatter3d:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeFigure:
    def __init__(self) -> None:
        self.traces = []
        self.layout = None

    def add_trace(self, trace) -> None:
        self.traces.append(trace)

    def update_layout(self, **kwargs) -> None:
        self.layout = kwargs


def _fake_plotly():
    go_module = types.SimpleNamespace(
        Figure=_FakeFigure,
        Scatter3d=_FakeScatter3d,
    )
    plotly_module = types.ModuleType("plotly")
    graph_objects = types.ModuleType("plotly.graph_objects")
    graph_objects.Figure = _FakeFigure
    graph_objects.Scatter3d = _FakeScatter3d
    plotly_module.graph_objects = graph_objects
    return plotly_module, graph_objects


def _make_tree() -> Morphology:
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[10.0, 0.0, 0.0], [10.0, 20.0, 0.0], [10.0, 40.0, 0.0]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class PlotlyBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _make_tree()
        self.backend = PlotlyBackend()
        self.plotly, self.go = _fake_plotly()

    def test_supports_3d_only(self) -> None:
        self.assertEqual(self.backend.supported_scene_kinds, frozenset({"3d"}))

    def test_render_rejects_non_3d_scene(self) -> None:
        request = RenderRequest(morpho=self.tree, dimensionality="2d", scene=None)
        with self.assertRaisesRegex(ValueError, "RenderScene3D"):
            self.backend.render(request)

    def test_render_produces_one_trace_per_branch_type(self) -> None:
        scene = build_render_scene_3d(self.tree)
        request = RenderRequest(morpho=self.tree, dimensionality="3d", scene=scene)

        with mock.patch.dict(
            sys.modules,
            {"plotly": self.plotly, "plotly.graph_objects": self.go},
        ):
            fig = self.backend.render(request)

        self.assertIsInstance(fig, _FakeFigure)
        self.assertEqual(len(fig.traces), len(scene.batches))
        for trace in fig.traces:
            self.assertEqual(trace.kwargs.get("mode"), "lines")

    def test_render_with_values_emits_shared_colorscale(self) -> None:
        values = np.linspace(0.0, 1.0, len(self.tree.branches))
        overlay = OverlaySpec(values=ValueSpec(values=values, cmap="viridis", label="V_m"))
        scene = build_render_scene_3d(self.tree, overlay=overlay)
        request = RenderRequest(morpho=self.tree, dimensionality="3d", scene=scene, overlay=overlay)

        with mock.patch.dict(
            sys.modules,
            {"plotly": self.plotly, "plotly.graph_objects": self.go},
        ):
            fig = self.backend.render(request)

        first = fig.traces[0]
        self.assertIn("line", first.kwargs)
        line = first.kwargs["line"]
        # Shared colorscale inlined in first trace, suppressed in later ones.
        self.assertEqual(line.get("colorscale"), "viridis")
        self.assertTrue(line.get("showscale", False))
        if len(fig.traces) > 1:
            self.assertFalse(fig.traces[1].kwargs["line"].get("showscale", False))

    def test_render_raises_when_plotly_missing(self) -> None:
        backend = PlotlyBackend()
        scene = build_render_scene_3d(self.tree)
        request = RenderRequest(morpho=self.tree, dimensionality="3d", scene=scene)
        with mock.patch(
            "braincell.vis.backend_plotly.importlib.util.find_spec",
            return_value=None,
        ):
            with mock.patch.dict(sys.modules, {"plotly": None}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "Plotly backend is not available"):
                    backend.render(request)


if __name__ == "__main__":
    unittest.main()
