from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from ...morpho import Morpho


@dataclass(frozen=True)
class NeuroMlReader:
    def read(self, path: str | PathLike[str]) -> Morpho:
        raise NotImplementedError("Parse NeuroML2 into an editable Morpho tree.")
