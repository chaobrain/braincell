from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import brainunit as u
import numpy as np

try:
    import jax.numpy as jnp
except ModuleNotFoundError:
    jnp = None


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.last_request = None

    def available(self) -> bool:
        return True

    def render(self, request):
        self.last_request = request
        return request
