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

from .calcium import *
from .calcium import CalciumFixed
from .calcium import __all__ as _calcium_all
from .potassium import *
from .potassium import PotassiumFixed
from .potassium import __all__ as _potassium_all
from .sodium import *
from .sodium import SodiumFixed
from .sodium import __all__ as _sodium_all

__all__ = _calcium_all + _potassium_all + _sodium_all + ["build_placeholder_ions"]


def build_placeholder_ions(size=(1,)) -> dict[str, object]:
    """Return default Na/K/Ca fixed-ion containers for scaffolding.

    Used by test doubles and by :class:`HHTypedNeuron` construction
    before the real runtime ion containers are instantiated.

    Parameters
    ----------
    size : tuple of int, optional
        Varshape of the ion containers. Defaults to ``(1,)``.
    """
    return {
        "na": SodiumFixed(size=size),
        "k": PotassiumFixed(size=size),
        "ca": CalciumFixed(size=size),
    }
