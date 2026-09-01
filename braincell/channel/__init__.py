# -*- coding: utf-8 -*-
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


import warnings

from ._base import (
    Gate,
    Transition,
    HH,
    OhmicHH,
    Markov,
    ghk_flux,
)
from ._base import __all__ as _base_all
from .calcium import (
    CaN_IS2008,
    CaT_HM1992,
    CaT_HP1992,
    CaHT_HM1992,
    CaHT_Re1993,
    CaL_IS2008,
    CaHVA_SU2015_DCN,
    CaL_SU2015_DCN,
    CaLVA_SU2015_DCN,
    Cav1p2_MA2020_GoC,
    Cav1p2_MA2025_BC,
    Cav1p3_MA2020_GoC,
    Cav1p3_MA2025_BC,
    Cav3p1_MA2020_GoC,
    Cav3p1_MA2020_GoC_Frozen,
    Cav3p1_MA2024_PC,
    Cav3p1_MA2024_PC_Frozen,
    Cav3p1Test_PC24,
    Cav2p1_MA2025_BC,
    Cav2p1_MA2025_BC_Frozen,
    Cav2p1_MA2024_PC,
    Cav2p1_MA2024_PC_Frozen,
    Cav2p1_RI2021_SC,
    Cav2p1_RI2021_SC_Frozen,
    Cav3p2_MA2025_BC,
    Cav3p2_MA2024_PC,
    Cav3p2_RI2021_SC,
    Cav3p3_MA2024_PC,
    Cav3p3_MA2024_PC_Frozen,
    Cav3p3_RI2021_SC,
    CaHVA_MA2020_GoC,
    CaHVA_MA2020_GrC,
    Cav2p3_MA2020_GoC,
    Ca_ZH2019_IO,
    Ca_ZH2019_IO_Frozen,
)
from .calcium import __all__ as _calcium_all
from .hyperpolarization_activated import (
    HCN_HM1992,
    HCN1_MA2025_BC,
    HCN1_MA2024_PC,
    HCN1_RI2021_SC,
    HCN1_MA2020_GoC,
    HCN2_MA2020_GoC,
    HCN_SU2015_DCN,
    HCN_ZH2019_IO,
)
from .hyperpolarization_activated import __all__ as _hyperpolarization_activated_all
from .leaky import (
    LeakageChannel,
    IL,
)
from .leaky import __all__ as _leaky_all
from .potassium import (
    KDR_Ba2002,
    K_TM1991,
    K_HH1952,
    KA1_HM1992,
    KA2_HM1992,
    KK2A_HM1992,
    KK2B_HM1992,
    KNI_Ya1989,
    K_Leak,
    K_Kv_test,
    fKdr_SU2015_DCN,
    sKdr_SU2015_DCN,
    KM_RI2021_SC,
    Kir2p3_MA2025_BC,
    Kir2p3_MA2024_PC,
    Kir2p3_RI2021_SC,
    Kv1p1_MA2025_BC,
    Kv1p1_MA2024_PC,
    Kv1p1_RI2021_SC,
    Kv1p5_MA2024_PC,
    Kv3p3_MA2024_PC,
    Kv3p4_MA2025_BC,
    Kv3p4_MA2024_PC,
    Kv3p4_RI2021_SC,
    Kv4p3_MA2025_BC,
    Kv4p3_MA2024_PC,
    Kv4p3_RI2021_SC,
    KM_MA2020_GoC,
    Kv1p1_MA2020_GoC,
    Kv3p4_MA2020_GoC,
    Kv4p3_MA2020_GoC,
    KM_MA2020_GrC,
    Kir2p3_MA2020_GrC,
    Kv1p1_MA2020_GrC,
    Kv2p2_0010_MA2020_GrC,
    Kv3p4_MA2020_GrC,
    Kv4p3_MA2020_GrC,
    Kdr_ZH2019_IO,
)
from .potassium import __all__ as _potassium_all
from .potassium_calcium import (
    AHP_De1994,
    SK_SU2015_DCN,
    Kca3p1_MA2020_GoC,
    Kca3p1_MA2025_BC,
    Kca3p1_MA2024_PC,
    Kca2p2_MA2020_GoC,
    Kca2p2_MA2025_BC,
    Kca2p2_MA2020_GrC,
    Kca2p2_MA2024_PC,
    Kca2p2_RI2021_SC,
    Kca1p1_MA2020_GoC,
    Kca1p1_MA2025_BC,
    Kca1p1_MA2020_GrC,
    Kca1p1_MA2024_PC,
    Kca1p1_RI2021_SC,
)
from .potassium_calcium import __all__ as _potassium_calcium_all
from .potassium_sodium import (
    Kv1p5_MA2020_GrC,
)
from .potassium_sodium import __all__ as _potassium_sodium_all
from .sodium import (
    Na_Ba2002,
    Na_TM1991,
    Na_HH1952,
    NaF_SU2015_DCN,
    NaP_SU2015_DCN,
    Na_ZH2019_IO,
    Nav1p6_MA2020_GoC,
    Nav1p6_MA2024_PC,
    Nav1p6_MA2025_BC,
    Nav1p6_RI2021_SC,
    Nav1p1_MA2025_BC,
    Nav1p1_RI2021_SC,
    Nav_MA2020_GrC,
    NaFHF_MA2020_GrC,
)
from .sodium import __all__ as _sodium_all

__all__ = (
    _base_all
    + _calcium_all
    + _hyperpolarization_activated_all
    + _leaky_all
    + _potassium_all
    + _potassium_calcium_all
    + _potassium_sodium_all
    + _sodium_all
)


# Backward-compatibility aliases for channel classes renamed in the
# v0.x normalization (PRs #80/#93). Only the unambiguous 1:1 renames
# (old name == new name with the leading ``I`` dropped) are aliased.
# Ambiguous renames (one old name now split into region variants) and
# removed classes are intentionally absent and raise ``AttributeError``.
_DEPRECATED_ALIASES = {
    "INa_HH1952": "Na_HH1952",
    "INa_Ba2002": "Na_Ba2002",
    "INa_TM1991": "Na_TM1991",
    "IK_HH1952": "K_HH1952",
    "IK_TM1991": "K_TM1991",
    "IK_Leak": "K_Leak",
    "IKDR_Ba2002": "KDR_Ba2002",
    "IKNI_Ya1989": "KNI_Ya1989",
    "IKA1_HM1992": "KA1_HM1992",
    "IKA2_HM1992": "KA2_HM1992",
    "IKK2A_HM1992": "KK2A_HM1992",
    "IKK2B_HM1992": "KK2B_HM1992",
    "ICaN_IS2008": "CaN_IS2008",
    "ICaL_IS2008": "CaL_IS2008",
    "ICaT_HM1992": "CaT_HM1992",
    "ICaT_HP1992": "CaT_HP1992",
    "ICaHT_HM1992": "CaHT_HM1992",
    "ICaHT_Re1993": "CaHT_Re1993",
    "IAHP_De1994": "AHP_De1994",
}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"braincell.channel.{name} is deprecated and will be removed; use braincell.channel.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
