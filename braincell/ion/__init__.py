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

from .calcium import (
    Calcium,
    CalciumFixed,
    CalciumInitNernst,
    CalciumDetailed,
    CalciumFirstOrder,
    ToyCaBindingKinetic_SU2015_DCN,
    ToyCaBindingSourceKinetic_SU2015_DCN,
    ToyCaBindingIcaSourceKinetic_SU2015_DCN,
    ToyDiamFactorKinetic_SU2015_DCN,
    ToyCaPumpFactorKinetic_SU2015_DCN,
    CdpStC_CAMOnly_MA2020_GoC,
    CdpStC_NoCAM_MA2020_GoC,
    CdpStC_MA2025_BC,
    CdpStC_MA2020_GoC,
    CdpCAM_MA2024_PC,
    CdpCR_MA2020_GrC,
    CdpStC_RI2021_SC,
    CdpHVA_SU2015_DCN,
    CdpLVA_SU2015_DCN,
)
from .calcium import __all__ as _calcium_all
from .nonspecific import (
    NonSpecific,
    NonSpecificFixed,
)
from .nonspecific import __all__ as _nonspecific_all
from .potassium import (
    Potassium,
    PotassiumFixed,
    PotassiumInitNernst,
)
from .potassium import __all__ as _potassium_all
from .sodium import (
    Sodium,
    SodiumFixed,
    SodiumInitNernst,
)
from .sodium import __all__ as _sodium_all

__all__ = _calcium_all + _nonspecific_all + _potassium_all + _sodium_all + ["build_placeholder_ions"]


def build_placeholder_ions(size=(1,)) -> dict[str, object]:
    """Return one fixed-ion container per ion family, keyed by family symbol.

    Scaffolding for the compute layer: ``braincell._compute.ions`` calls
    this to obtain a default container for a family a model refers to but
    never declares. The containers carry only the class-level defaults,
    so they stand in for a real ion rather than modelling one.

    Not called during :class:`~braincell.HHTypedNeuron` construction --
    ``braincell/_multi_compartment/cell_test.py`` asserts the opposite,
    since building four ion containers per cell at ``__init__`` would be
    wasted work for the common case where every family is declared.

    Parameters
    ----------
    size : tuple of int, optional
        Varshape of the ion containers. Defaults to ``(1,)``.

    Returns
    -------
    dict
        Maps each family symbol -- ``"na"``, ``"k"``, ``"ca"``, ``"no"``
        -- to a freshly constructed fixed-ion container.
    """
    return {
        "na": SodiumFixed(size=size),
        "k": PotassiumFixed(size=size),
        "ca": CalciumFixed(size=size),
        "no": NonSpecificFixed(size=size),
    }
