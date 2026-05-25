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

from .calcium import *
from .calcium import __all__ as calcium_all
from .hyperpolarization_activated import *
from .hyperpolarization_activated import __all__ as hyperpolarization_activated_all
from .leaky import *
from .leaky import __all__ as leaky_all
from .potassium import *
from .potassium import __all__ as potassium_all
from .potassium_calcium import *
from .potassium_calcium import __all__ as potassium_calcium_all
from .sodium import *
from .sodium import __all__ as sodium_all

__all__ = (
    calcium_all +
    hyperpolarization_activated_all +
    leaky_all +
    potassium_all +
    potassium_calcium_all +
    sodium_all
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
            f"braincell.channel.{name} is deprecated and will be removed; "
            f"use braincell.channel.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

