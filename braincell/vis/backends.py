from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .pyvista_backend import PyVistaBackend


class RenderBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def render(self, request: object) -> object:
        ...


@dataclass(frozen=True)
class BackendChooser:
    backends: tuple[RenderBackend, ...]

    @classmethod
    def default(cls) -> "BackendChooser":
        return cls(backends=(PyVistaBackend(),))

    def pick(self, *, requested: str | None = None) -> RenderBackend:
        if requested is not None:
            for backend in self.backends:
                if backend.name != requested:
                    continue
                if backend.available():
                    return backend
                raise RuntimeError(f"Visualization backend {requested!r} is not available.")
            raise ValueError(f"Unknown visualization backend {requested!r}.")
        for backend in self.backends:
            if backend.available():
                return backend
        raise RuntimeError("No visualization backend is available.")
