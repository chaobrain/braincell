from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import braincell._test_support  # noqa: F401

from braincell import Branch, Morpho
from braincell._test_support import u
from braincell.vis import PyVistaBackend, RenderRequest, build_render_scene_3d


class _FakePolyData:
    def __init__(self) -> None:
        self.points = None
        self.lines = None
        self.point_data = {}

    def tube(self, *, scalars, absolute, n_sides):
        return {
            "scalars": scalars,
            "absolute": absolute,
            "n_sides": n_sides,
        }


class _FakeTrameConfig:
    def __init__(self) -> None:
        self.jupyter_extension_available = False
        self.jupyter_extension_enabled = False
        self.server_proxy_enabled = False
        self.server_proxy_prefix = "/proxy/"


class _FakePlotter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.background = None
        self.axes_shown = False
        self.meshes = []
        self.show_calls = []
        self.closed = False
        self._first_time = True
        self._rendered = False

    def set_background(self, color) -> None:
        self.background = color

    def show_axes(self) -> None:
        self.axes_shown = True

    def add_mesh(self, mesh, color=None) -> None:
        self.meshes.append((mesh, color))

    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        return {"viewer": kwargs}

    def _on_first_render_request(self) -> None:
        self._first_time = False

    def render(self) -> None:
        if not self._first_time:
            self._rendered = True

    def close(self) -> None:
        self.closed = True




class _NeedsRenderedPlotter(_FakePlotter):
    def show(self, **kwargs):
        if not self._rendered:
            raise AttributeError(
                "This plotter has not yet been set up and rendered with ``show()``."
            )
        return super().show(**kwargs)


class _ClientFailsPlotter(_FakePlotter):
    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        if kwargs["jupyter_backend"] == "client":
            raise RuntimeError("client boom")
        return {"viewer": kwargs}


class _TrameFailsPlotter(_FakePlotter):
    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        if kwargs["jupyter_backend"] == "trame":
            raise RuntimeError("trame boom")
        return {"viewer": kwargs}


class _NoneViewerPlotter(_FakePlotter):
    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        return None


class _AlwaysFailPlotter(_FakePlotter):
    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        raise RuntimeError(f"{kwargs['jupyter_backend']} boom")


def _request(*, notebook: bool | None = None, jupyter_backend: str | None = None, return_plotter: bool = False):
    branch = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
        radii=[2.0, 2.0] * u.um,
        type="soma",
    )
    tree = Morpho.from_root(branch, name="soma")
    return RenderRequest(
        morpho=tree,
        dimensionality="3d",
        scene=build_render_scene_3d(tree),
        notebook=notebook,
        jupyter_backend=jupyter_backend,
        return_plotter=return_plotter,
    )


def _fake_pyvista(plotter_cls, *, extension_available=False):
    return types.SimpleNamespace(
        Plotter=plotter_cls,
        PolyData=_FakePolyData,
        global_theme=types.SimpleNamespace(
            trame=_FakeTrameConfig(),
        ),
    )


class PyVistaBackendTest(unittest.TestCase):
    def test_render_rejects_non_3d_scene(self) -> None:
        branch = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 2.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(branch, name="soma")
        backend = PyVistaBackend()

        with self.assertRaisesRegex(ValueError, "RenderScene3D"):
            backend.render(RenderRequest(morpho=tree, dimensionality="2d", scene=None))

    def test_render_returns_plotter_outside_notebook(self) -> None:
        fake_pv = _fake_pyvista(_FakePlotter)
        backend = PyVistaBackend(plotter_kwargs={"off_screen": True}, show_axes=False)

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            plotter = backend.render(_request(notebook=False))

        self.assertIsInstance(plotter, _FakePlotter)
        self.assertEqual(plotter.kwargs, {"off_screen": True})
        self.assertEqual(plotter.background, "white")
        self.assertEqual(len(plotter.meshes), 1)
        self.assertEqual(plotter.show_calls, [])

    def test_render_uses_client_first_for_headless_notebook_auto_mode(self) -> None:
        fake_pv = _fake_pyvista(_FakePlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {}, clear=True):
                viewer = backend.render(_request(notebook=True))

        self.assertEqual(viewer["viewer"]["jupyter_backend"], "client")
        self.assertTrue(fake_pv.global_theme.trame.server_proxy_enabled)
        self.assertEqual(fake_pv.global_theme.trame.server_proxy_prefix, "/proxy/")


    def test_render_prerenders_plotter_before_client_show(self) -> None:
        fake_pv = _fake_pyvista(_NeedsRenderedPlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {}, clear=True):
                viewer = backend.render(_request(notebook=True, jupyter_backend="client"))

        self.assertEqual(viewer["viewer"]["jupyter_backend"], "client")

    def test_render_falls_back_from_client_to_html(self) -> None:
        fake_pv = _fake_pyvista(_ClientFailsPlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {}, clear=True):
                viewer = backend.render(_request(notebook=True))

        self.assertEqual(viewer["viewer"]["jupyter_backend"], "html")

    def test_render_uses_trame_first_when_display_exists(self) -> None:
        fake_pv = _fake_pyvista(_FakePlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {"DISPLAY": ":1"}, clear=True):
                viewer = backend.render(_request(notebook=True))

        self.assertEqual(viewer["viewer"]["jupyter_backend"], "trame")

    def test_render_can_return_plotter_after_notebook_show(self) -> None:
        fake_pv = _fake_pyvista(_FakePlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {}, clear=True):
                plotter = backend.render(_request(notebook=True, return_plotter=True))

        self.assertIsInstance(plotter, _FakePlotter)
        self.assertEqual(len(plotter.show_calls), 1)
        self.assertEqual(plotter.show_calls[0]["jupyter_backend"], "client")

    def test_render_preserves_explicit_client_request(self) -> None:
        fake_pv = _fake_pyvista(_FakePlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            viewer = backend.render(_request(notebook=True, jupyter_backend="client"))

        self.assertEqual(viewer["viewer"]["jupyter_backend"], "client")

    def test_render_reports_attempted_backends_on_failure(self) -> None:
        fake_pv = _fake_pyvista(_AlwaysFailPlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Attempted backends: client, html") as ctx:
                    backend.render(_request(notebook=True))

        self.assertIn("client: RuntimeError: client boom", str(ctx.exception))
        self.assertIn("html: RuntimeError: html boom", str(ctx.exception))
        self.assertIn("jupyter_server_proxy=enabled", str(ctx.exception))

    def test_render_treats_none_viewer_as_failure(self) -> None:
        fake_pv = _fake_pyvista(_NoneViewerPlotter)
        backend = PyVistaBackend()

        with mock.patch.dict("sys.modules", {"pyvista": fake_pv}):
            with self.assertRaisesRegex(RuntimeError, "PyVista returned no notebook viewer"):
                backend.render(_request(notebook=True, jupyter_backend="html"))
