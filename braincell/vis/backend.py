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


import importlib.util
import sys
from dataclasses import dataclass
from typing import Protocol

from .scene import RenderScene2D, RenderScene3D


def module_available(module_name: str) -> bool:
    """Return whether *module_name* can be imported, without importing it.

    This is the probe every backend's ``available()`` uses, so an
    optional dependency is detected the same way everywhere and none of
    the backends import their heavy third-party module at import time.

    Parameters
    ----------
    module_name : str
        Top-level module name, e.g. ``"plotly"``.

    Returns
    -------
    bool
        ``True`` when the module is installed or already imported.

    Notes
    -----
    :func:`importlib.util.find_spec` raises :class:`ValueError` for a
    module that is present in :data:`sys.modules` but has no ``__spec__``
    (which is what a test double injected into ``sys.modules`` looks
    like); that case falls back to a membership test.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


class RenderBackend(Protocol):
    """Capability-based backend contract.

    ``supported_scene_kinds`` declares the set of scene kinds a backend can
    render. ``"2d"`` and ``"3d"`` are currently defined; a future backend
    (e.g. Plotly) that can serve both should advertise
    ``frozenset({"2d", "3d"})``.
    """

    name: str
    supported_scene_kinds: frozenset[str]

    def available(self) -> bool: ...

    def render(self, request: object) -> object: ...


def _backend_supports(backend: RenderBackend, scene_kind: str) -> bool:
    kinds = getattr(backend, "supported_scene_kinds", None)
    if kinds is None:
        return True  # permissive fallback for test doubles
    return scene_kind in kinds


@dataclass(frozen=True)
class BackendChooser:
    backends: tuple[RenderBackend, ...]

    @classmethod
    def default(cls) -> "BackendChooser":
        from .backend_matplotlib import MatplotlibBackend
        from .backend_plotly import PlotlyBackend
        from .backend_pyvista import PyVistaBackend

        # Order matters: PyVista wins over Plotly for 3D when both are
        # installed because it's the higher-fidelity backend; matplotlib
        # sits first overall because it's always installed in dev envs.
        return cls(backends=(MatplotlibBackend(), PyVistaBackend(), PlotlyBackend()))

    def pick(self, *, requested: str | None = None, scene_kind: str | None = None) -> RenderBackend:
        if requested is not None:
            for backend in self.backends:
                if backend.name != requested:
                    continue
                if backend.available():
                    return backend
                raise RuntimeError(f"Visualization backend {requested!r} is not available.")
            raise ValueError(f"Unknown visualization backend {requested!r}.")

        if scene_kind is not None:
            for backend in self.backends:
                if not _backend_supports(backend, scene_kind):
                    continue
                if backend.available():
                    return backend

        for backend in self.backends:
            if backend.available():
                return backend
        raise RuntimeError("No visualization backend is available.")


def validate_backend_for_scene(
    backend: RenderBackend,
    scene: RenderScene2D | RenderScene3D | None,
) -> None:
    kinds = getattr(backend, "supported_scene_kinds", None)
    if kinds is None or scene is None:
        return
    if isinstance(scene, RenderScene2D) and "2d" not in kinds:
        raise ValueError(f"Visualization backend {backend.name!r} only supports 3D scenes.")
    if isinstance(scene, RenderScene3D) and "3d" not in kinds:
        raise ValueError(f"Visualization backend {backend.name!r} only supports 2D scenes.")
