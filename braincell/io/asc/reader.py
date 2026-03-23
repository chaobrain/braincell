from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from ...morpho import Morpho


@dataclass(frozen=True)
class AscReader:
    def read(self, path: str | PathLike[str]) -> Morpho:
        raise NotImplementedError("Parse Neurolucida ASC into an editable Morpho tree.")
