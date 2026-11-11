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

from braincell._typing import FilePath
from braincell.morph.morphology import Morphology


@dataclass(frozen=True)
class NeuroMlReader:
    """Placeholder for a NeuroML2 morphology reader; not yet supported.

    NeuroML2 import is planned. Until it lands, :meth:`read` raises
    ``NotImplementedError`` for every path. Load SWC or Neurolucida ASC
    reconstructions with ``Morphology.from_swc`` or ``Morphology.from_asc``.

    Examples
    --------
    >>> from braincell.io import NeuroMlReader
    >>> NeuroMlReader().read("cell.nml")
    Traceback (most recent call last):
        ...
    NotImplementedError: NeuroML2 import is not supported yet. ...
    """

    def read(self, path: FilePath) -> Morphology:
        """Read a NeuroML2 cell morphology (not supported yet).

        Parameters
        ----------
        path : str or os.PathLike
            Path to a NeuroML2 document. It is not opened.

        Raises
        ------
        NotImplementedError
            Always; NeuroML2 import is planned but not implemented.
        """
        raise NotImplementedError(
            "NeuroML2 import is not supported yet. Load SWC or Neurolucida ASC "
            "reconstructions with Morphology.from_swc or Morphology.from_asc."
        )
