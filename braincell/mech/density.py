from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DensityMechanism:
    ion_type: str | None = None
    channel_type: str | None = None
    params: tuple[tuple[str, Any], ...] = ()
