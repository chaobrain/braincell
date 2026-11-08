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

"""Pixel baselines of representative figures, drawn through braincell's adapters.

Plotting lives in ``braintools.visualize.morphology``; braincell only converts
its objects (``Morphology.to_vis_data``, ``Cell.topology_data``,
``NodeTree.to_topology_data``). These tests draw the same nine figures as
before the move, from braincell objects, and compare them with the PNGs in
``braincell/vis_baselines/`` (the same files, and the same ``tol``, as
braintools' own baselines).

* Rendering is pinned by :func:`baseline_style`: Agg, default ``rcParams``,
  the bundled DejaVu Sans font, no hinting and a fixed dpi.
* Pixels are compared with :func:`matplotlib.testing.compare.compare_images`
  (RMS difference over 0-255, ``tol=0`` means identical), on Linux only:
  font rasterisation differs between operating systems, so every other
  platform skips.
* ``BRAINCELL_UPDATE_BASELINES=1`` rewrites the baselines instead of
  comparing.
"""

import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import brainunit as u
import numpy as np

from braincell import Branch, Cell, Morphology
from braincell._discretization import CVPerBranch
from braincell.filter import BranchSlice
from braincell.io._testing import ALLOWED_TYPES, FIXTURE_DIR, VALID_SWC_FIXTURES

BASELINE_DIR = Path(__file__).resolve().parent / "vis_baselines"
"""Committed baseline PNGs, one per figure; not packaged in the sdist or wheel."""

UPDATE_BASELINES_ENV = "BRAINCELL_UPDATE_BASELINES"
"""Set to ``1`` to rewrite the baselines instead of comparing against them."""

BASELINE_FIGSIZE = (4.0, 3.0)
"""Figure size in inches; at ``BASELINE_DPI`` every baseline is 400 x 300 px."""

BASELINE_TOL = 0.0
"""RMS tolerance (0-255) for every baseline comparison."""

#: Resolution every baseline is rendered at; with a fixed ``figsize`` it fixes
#: the image size in pixels.
BASELINE_DPI = 100


@contextlib.contextmanager
def baseline_style():
    """Pin Matplotlib's rendering for the duration of the ``with`` block.

    Switches to the Agg backend's default ``rcParams`` plus the font and
    reproducibility settings Matplotlib itself uses for its image tests, and
    restores the caller's ``rcParams`` on exit.
    """
    import matplotlib as mpl
    from matplotlib.testing import set_font_settings_for_testing, set_reproducibility_for_testing

    defaults = {k: v for k, v in mpl.rcParamsDefault.items() if k not in ("backend", "backend_fallback")}
    with mpl.rc_context(defaults):
        set_font_settings_for_testing()
        set_reproducibility_for_testing()
        yield


def _updating(update_env: str) -> bool:
    return os.environ.get(update_env, "").strip().lower() in ("1", "true", "yes")


def assert_figure_matches(
    fig,
    baseline,
    *,
    tol: float = 0.0,
    update_env: str,
    dpi: int = BASELINE_DPI,
) -> None:
    """Assert that ``fig`` renders to the same pixels as the PNG at ``baseline``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render. Give it a fixed ``figsize`` and draw it inside
        :func:`baseline_style`.
    baseline : str or os.PathLike
        Baseline PNG. Missing baselines fail with a hint to generate them.
    tol : float
        RMS tolerance passed to ``compare_images``; ``0`` demands identical pixels.
    update_env : str
        Name of the environment variable that, set to ``1``, rewrites the
        baseline instead of comparing.
    dpi : int
        Render resolution.

    Raises
    ------
    unittest.SkipTest
        On any platform other than Linux.
    AssertionError
        When the baseline is missing or the pixels differ by more than ``tol``.
    """
    from matplotlib.testing.compare import compare_images

    if not sys.platform.startswith("linux"):
        raise unittest.SkipTest("pixel baselines are compared on Linux only (font rendering differs elsewhere)")

    baseline = Path(baseline)
    if _updating(update_env):
        baseline.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(baseline, dpi=dpi, format="png")
        return
    if not baseline.is_file():
        raise AssertionError(f"missing baseline {baseline}; generate it with {update_env}=1")

    result_dir = Path(tempfile.mkdtemp(prefix="baseline-"))
    actual = result_dir / baseline.name
    fig.savefig(actual, dpi=dpi, format="png")
    # compare_images writes ``<actual stem>-failed-diff.png`` beside ``actual``.
    message = compare_images(str(baseline), str(actual), tol=tol)
    if message is None:
        shutil.rmtree(result_dir, ignore_errors=True)
        return
    raise AssertionError(f"{baseline.name} differs from its baseline (images kept in {result_dir}):\n{message}")


def make_baseline_tree(*, apical_scale: float = 1.0) -> Morphology:
    """The branching 3D tree every pixel baseline draws.

    A soma, an apical dendrite that forks once, a basal dendrite and an axon,
    all with explicit 3D points so the projected layout and the Sholl profile
    are defined. ``apical_scale`` stretches the trunk (and so moves its
    children) away from the soma: ``1.01`` is the one-percent length change
    the baselines must detect.
    """
    s = float(apical_scale)

    def pts(rows, origin=(0.0, 0.0, 0.0)):
        o = np.asarray(origin, dtype=float)
        return (o + s * np.asarray(rows, dtype=float)) * u.um

    soma = Branch.from_points(points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    trunk_rows = [[0.0, 0.0, 0.0], [10.0, 30.0, 0.0], [5.0, 70.0, 5.0]]
    trunk_end = np.asarray([20.0, 0.0, 0.0]) + s * np.asarray(trunk_rows[-1])
    trunk = Branch.from_points(
        points=pts(trunk_rows, (20.0, 0.0, 0.0)), radii=[3.0, 2.5, 2.0] * u.um, type="apical_dendrite"
    )
    fork_a = Branch.from_points(
        points=(trunk_end + np.asarray([[0.0, 0.0, 0.0], [-25.0, 30.0, 0.0], [-30.0, 60.0, 0.0]])) * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    fork_b = Branch.from_points(
        points=(trunk_end + np.asarray([[0.0, 0.0, 0.0], [25.0, 35.0, 0.0], [40.0, 55.0, 0.0]])) * u.um,
        radii=[2.0, 1.4, 1.0] * u.um,
        type="apical_dendrite",
    )
    basal = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [-30.0, -20.0, 0.0], [-60.0, -25.0, 0.0]] * u.um,
        radii=[2.5, 2.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    axon = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [50.0, -20.0, 0.0], [90.0, -30.0, 0.0]] * u.um,
        radii=[1.2, 1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=fork_a, child_name="fork_a", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=fork_b, child_name="fork_b", parent_x=1.0)
    tree.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=0.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


class _BaselineTestCase(unittest.TestCase):
    """Render in the pinned baseline style and close every figure afterwards."""

    def setUp(self) -> None:  # noqa: N802 - unittest naming
        import braintools.visualize.morphology as bvm
        import matplotlib.pyplot as plt

        bvm.reset_defaults()
        self.addCleanup(bvm.reset_defaults)
        self.enterContext(baseline_style())
        self.addCleanup(plt.close, "all")

    @staticmethod
    def new_axes():
        import matplotlib.pyplot as plt

        return plt.subplots(figsize=BASELINE_FIGSIZE, dpi=BASELINE_DPI)

    @staticmethod
    def assert_baseline(fig, name: str, *, tol: float = BASELINE_TOL) -> None:
        assert_figure_matches(fig, BASELINE_DIR / f"{name}.png", tol=tol, update_env=UPDATE_BASELINES_ENV)

    def assert_baseline_differs(self, fig, name: str) -> None:
        """Assert that ``fig`` does *not* match baseline ``name``, never rewriting it."""
        if os.environ.get(UPDATE_BASELINES_ENV):
            self.skipTest("baselines are being regenerated")
        with self.assertRaisesRegex(AssertionError, rf"{name}\.png differs from its baseline"):
            self.assert_baseline(fig, name)


class Vis2dBaselineTest(_BaselineTestCase):
    """``Morphology.vis2d``; regenerate with ``BRAINCELL_UPDATE_BASELINES=1``."""

    def _draw(self, tree=None, **kwargs):
        fig, ax = self.new_axes()
        tree = tree if tree is not None else make_baseline_tree()
        tree.vis2d(backend="matplotlib", ax=ax, show=False, **kwargs)
        return fig

    def test_projected_layout(self) -> None:
        self.assert_baseline(self._draw(layout="projected", shape="line"), "plot2d_projected")

    def test_stem_layout(self) -> None:
        self.assert_baseline(self._draw(layout="stem", shape="frustum"), "plot2d_stem")

    def test_balloon_layout(self) -> None:
        self.assert_baseline(self._draw(layout="balloon", shape="frustum"), "plot2d_balloon")

    def test_radial_layout(self) -> None:
        self.assert_baseline(self._draw(layout="radial_360", shape="frustum"), "plot2d_radial")

    def test_values_with_colorbar(self) -> None:
        import braintools.visualize.morphology as bvm

        values = np.linspace(-70.0, -40.0, 11)
        fig, ax = self.new_axes()
        bvm.plot2d(make_baseline_tree(), layout="stem", shape="frustum", values=values, value_label="V (mV)", ax=ax)
        self.assert_baseline(fig, "plot2d_values_colorbar")

    def test_baseline_detects_one_percent_branch_length_change(self) -> None:
        # Guard on the comparison itself: a trunk 1 % longer must not pass for the stem baseline.
        fig = self._draw(make_baseline_tree(apical_scale=1.01), layout="stem", shape="frustum")
        self.assert_baseline_differs(fig, "plot2d_stem")

    def test_baseline_detects_a_different_colormap(self) -> None:
        import braintools.visualize.morphology as bvm

        values = np.linspace(-70.0, -40.0, 11)
        fig, ax = self.new_axes()
        bvm.plot2d(
            make_baseline_tree(),
            layout="stem",
            shape="frustum",
            values=values,
            value_label="V (mV)",
            cmap="plasma",
            ax=ax,
        )
        self.assert_baseline_differs(fig, "plot2d_values_colorbar")


class MorphometryBaselineTest(_BaselineTestCase):
    """Morphometry plots of a ``Morphology``; regenerate with ``BRAINCELL_UPDATE_BASELINES=1``."""

    def test_dendrogram(self) -> None:
        import braintools.visualize.morphology as bvm

        fig, ax = self.new_axes()
        bvm.plot_dendrogram(make_baseline_tree(), ax=ax)
        self.assert_baseline(fig, "dendrogram")

    def test_sholl(self) -> None:
        import braintools.visualize.morphology as bvm

        fig, ax = self.new_axes()
        bvm.plot_sholl(make_baseline_tree(), ax=ax, step_um=10.0)
        self.assert_baseline(fig, "sholl")


class TopologyBaselineTest(_BaselineTestCase):
    """Topology graphs of a ``Cell``; regenerate with ``BRAINCELL_UPDATE_BASELINES=1``."""

    def test_cv_level(self) -> None:
        import braintools.visualize.morphology as bvm

        cell = Cell(make_baseline_tree(), cv_policy=CVPerBranch())
        fig, ax = self.new_axes()
        graph = cell.topology_data("cv", region=BranchSlice(branch_index=1, prox=0.0, dist=0.5))
        bvm.plot_topology_graph(graph, layout="kamada_kawai", ax=ax)
        self.assert_baseline(fig, "cell_topology_cv")

    def test_node_tree(self) -> None:
        import braintools.visualize.morphology as bvm

        cell = Cell(make_baseline_tree(), cv_policy=CVPerBranch())
        fig, ax = self.new_axes()
        bvm.plot_topology_graph(cell.node_tree, layout="kamada_kawai", ax=ax)
        self.assert_baseline(fig, "point_topology")


class SwcFixture3dTest(unittest.TestCase):
    """Real SWC reconstructions render in 3D (SWC reading stays in braincell)."""

    def test_swc_fixtures_build_a_3d_scene(self) -> None:
        from braintools.visualize.morphology._tree import as_tree
        from braintools.visualize.morphology.scene3d import build_render_scene_3d

        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morphology.from_swc(FIXTURE_DIR / fixture_name)
                scene = build_render_scene_3d(as_tree(tree))

                self.assertEqual(len(scene.branches), len(tree.branches))
                self.assertGreaterEqual(len(scene.batches), 1)
                for branch in scene.branches:
                    self.assertTrue(branch.branch_name)
                    self.assertIn(branch.branch_type, ALLOWED_TYPES)
                    self.assertEqual(branch.points_um.ndim, 2)
                    self.assertEqual(branch.points_um.shape[1], 3)
                    self.assertGreaterEqual(len(branch.points_um), 2)
                    self.assertEqual(len(branch.radii_um), len(branch.points_um))

    def test_swc_fixtures_render_with_pyvista(self) -> None:
        try:
            import pyvista as pv
        except ImportError:
            self.skipTest("pyvista is not installed.")
        from braintools.visualize.morphology.backend import BackendChooser
        from braintools.visualize.morphology.backend_pyvista import PyVistaBackend

        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morphology.from_swc(FIXTURE_DIR / fixture_name)
                backend = PyVistaBackend(plotter_kwargs={"off_screen": True}, show_axes=False)
                plotter = tree.vis3d(backend="pyvista", chooser=BackendChooser(backends=(backend,)), show=False)

                self.assertIsInstance(plotter, pv.Plotter)
                self.assertGreater(len(plotter.renderer.actors), 0)
                plotter.close()


if __name__ == "__main__":
    unittest.main()
