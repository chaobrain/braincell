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


from dataclasses import dataclass
from typing import Protocol

from .scene import RenderScene2D, RenderScene3D


class RenderBackend(Protocol):
    """Capability-based backend contract.

    ``supported_scene_kinds`` declares the set of scene kinds a backend can
    render. ``"2d"`` and ``"3d"`` are currently defined; a future backend
    (e.g. Plotly) that can serve both should advertise
    ``frozenset({"2d", "3d"})``.
    """

    name: str
    supported_scene_kinds: frozenset[str]

    def available(self) -> bool:
        ...

    def render(self, request: object) -> object:
        ...


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
        from .backend_pyvista import PyVistaBackend

        return cls(backends=(MatplotlibBackend(), PyVistaBackend()))

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
