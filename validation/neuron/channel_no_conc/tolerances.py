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

"""Tolerance table and coverage list of the channel_no_conc comparisons against NEURON.

Every configuration and protocol not listed here is T0 (``_tolerance.T0_LIMITS``).
Each entry is T1: a limit about 1.5-2x the measured value, rounded up to 1, 2
or 5 times a power of ten, with the deviation class and the heading of
``validation/neuron/known-differences.md`` that explains it. Adding an entry
needs the root cause and a patch that brings it to T0; raising a limit or
moving a T0 configuration here loosens a tolerance and needs the lead's
approval.

``NON_FINITE_REFERENCE`` names the voltage-clamp commands whose NEURON run is
not finite and that the comparison therefore leaves out.

``VARIANTS`` names the default-off options that are compared separately;
their entries use the variant name as the configuration.

``UNCOVERED`` lists the registry channels that have no configuration under
``configs/``, each with its reason. ``tests/test_coverage.py`` fails when a
registered channel has neither, and when a listed name is not registered or
already has a configuration.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from validation.neuron._tolerance import ToleranceEntry, ToleranceTable

KNOWN_DIFFERENCES = Path(__file__).resolve().parents[1] / "known-differences.md"

_GHK = "Unfrozen GHK calcium channels"
_CALVA = "CaLVA forward-difference conductance"
_NA_ZH19 = "Na_ZH2019_IO limit branch"
_KCA2P2 = "Kca2.2 steady-state initialization"
_KCA1P1 = "Kca1.1 steady-state initialization"
_GHK_ZERO = "GHK drive near 0 mV in mod files without a limit branch"
_CA_ZH19_M_INF = "Ca_ZH2019_IO with freeze_m_inf=False"
_KDR_ZH19 = "Kdr_ZH2019_IO limit branch"
_GATE_CURRENT = "Kv1.1 and Kv3.3 gating current"
_NAV1P1_INIT = "Nav1.1 steady-state initialization"
_NAV1P1_GATE_CURRENT = "Nav1.1 gating current"
_KV1P1 = tuple(f"{model}/kv1p1_{model}" for model in ("ma20_goc", "ma20_grc", "ma24_pc", "ma25_bc", "ri21_sc"))
_NAV1P1 = ("ma25_bc/nav1p1_ma25_bc", "ri21_sc/nav1p1_ri21_sc")
_CAV3P3 = ("ma24_pc/cav3p3_ma24_pc", "ma24_pc/cav3p3_ma24_pc_frozen", "ri21_sc/cav3p3_ri21_sc")


class Variant(NamedTuple):
    """A default-off option of a configuration, compared on its own.

    ``channel_params`` overrides the configuration's channel parameters on
    both sides (a BrainCell-only option is a BrainCell-only parameter of the
    mapping). ``extra_neuron_currents`` are the mechanism's currents that
    NEURON keeps apart and BrainCell folds into the channel current; the
    voltage clamp adds them to NEURON's ``ix``.
    """

    config: str
    channel_params: Mapping[str, object]
    extra_neuron_currents: tuple[str, ...] = ()


VARIANTS = {
    "zh19_io/ca_zh19_io:freeze_m_inf=False": Variant("zh19_io/ca_zh19_io", {"freeze_m_inf": False}),
    **{
        f"{config}:gateCurrent=1": Variant(config, {"gate_current": 1.0}, ("i",))
        for config in (*_KV1P1, "ma24_pc/kv3p3_ma24_pc")
    },
    "ma20_grc/kv1p5_ma20_grc:gnonspec": Variant("ma20_grc/kv1p5_ma20_grc", {"gnonspec_S_cm2": 1.3195e-4}, ("ino",)),
    **{f"{config}:gateCurrent=1": Variant(config, {"gate_current": 1.0}, ("i",)) for config in _NAV1P1},
}
"""Name (the table key of the variant) -> variant."""


def _current_clamp(config: str, vinit: float | None, dc: float | None, ac: float | None, cls: str, section: str):
    limits = {"vinit_celsius": vinit, "dc": dc, "ac": ac}
    return [
        ToleranceEntry(config, protocol, {"e_V": limit}, cls, section)
        for protocol, limit in limits.items()
        if limit is not None
    ]


ENTRIES = [
    # Current clamp, e_V in mV (MECH-16).
    *(
        entry
        for config in ("ma24_pc/cav2p1_ma24_pc", "ma25_bc/cav2p1_ma25_bc", "ri21_sc/cav2p1_ri21_sc")
        for entry in _current_clamp(config, 5e-4, 1.0, 1e-4, "d", _GHK)
    ),
    *(
        entry
        for config in ("ma20_goc/cav3p1_ma20_goc", "ma24_pc/cav3p1_ma24_pc", "ma24_pc/cav3p1_test2_pc24")
        for entry in _current_clamp(config, 0.06, 0.25, 0.25, "d", _GHK)
    ),
    *_current_clamp("su15_dcn/calva_su15_dcn", 5e-4, 2e-4, 2e-4, "c", _CALVA),
    *_current_clamp("zh19_io/na_zh19_io", 5e-5, None, None, "c", _NA_ZH19),
    *(
        entry
        for config in ("ma20_goc/kca2p2_ma20_goc", "ma20_grc/kca2p2_ma20_grc", "ma25_bc/kca2p2_ma25_bc", "ri21_sc/kca2p2_ri21_sc")
        for entry in _current_clamp(config, 1e-5, 1e-5, 1e-5, "c", _KCA2P2)
    ),
    *_current_clamp("zh19_io/ca_zh19_io:freeze_m_inf=False", 1e-2, 0.2, 0.5, "d", _CA_ZH19_M_INF),
    # Current clamp, e_V in mV (MECH-19b).
    *(
        entry
        for config in _KV1P1
        for entry in _current_clamp(f"{config}:gateCurrent=1", 1e-3, 1e-2, 5e-3, "d", _GATE_CURRENT)
    ),
    *_current_clamp("ma24_pc/kv3p3_ma24_pc:gateCurrent=1", 5e-6, 5e-3, 1e-3, "d", _GATE_CURRENT),
    # Current clamp, e_V in mV (MECH-19c).
    *(
        entry
        for config in _NAV1P1
        for entry in _current_clamp(f"{config}:gateCurrent=1", 0.2, 10.0, 20.0, "d", _NAV1P1_GATE_CURRENT)
    ),
    # Voltage clamp, e_I and e_s (MECH-19a).
    *(
        ToleranceEntry(config, protocol, {"e_I": e_I, "e_s": 5e-9}, "c", _KCA2P2)
        for config in ("ma20_goc/kca2p2_ma20_goc", "ma20_grc/kca2p2_ma20_grc", "ma25_bc/kca2p2_ma25_bc", "ri21_sc/kca2p2_ri21_sc")
        for protocol, e_I in (("vc_steps", 2e-6), ("vc_ramp", 1e-6))
    ),
    *(
        ToleranceEntry(config, "vc_ramp", {"e_I": 1e-9}, "c", _KCA1P1)
        for config in ("ma20_goc/kca1p1_ma20_goc", "ma20_grc/kca1p1_ma20_grc", "ma25_bc/kca1p1_ma25_bc", "ri21_sc/kca1p1_ri21_sc")
    ),
    *(ToleranceEntry(config, "vc_ramp", {"e_I": 5e-5}, "c", _GHK_ZERO) for config in _CAV3P3),
    ToleranceEntry("ma24_pc/cav3p1_test2_pc24", "vc_steps", {"e_I": 1e-5}, "c", _GHK_ZERO),
    ToleranceEntry("ma24_pc/cav3p1_test2_pc24", "vc_ramp", {"e_I": 5e-6}, "c", _GHK_ZERO),
    ToleranceEntry("su15_dcn/cahva_su15_dcn", "vc_steps", {"e_I": 2e-5}, "c", _GHK_ZERO),
    ToleranceEntry("su15_dcn/cahva_su15_dcn", "vc_ramp", {"e_I": 2e-5}, "c", _GHK_ZERO),
    ToleranceEntry("su15_dcn/calva_su15_dcn", "vc_steps", {"e_I": 1e-9}, "c", _GHK_ZERO),
    # Voltage clamp (MECH-19b).
    ToleranceEntry("zh19_io/kdr_zh19_io", "vc_ramp", {"e_I": 2e-9, "e_s": 1e-8}, "c", _KDR_ZH19),
    # Voltage clamp (MECH-19c).
    ToleranceEntry("zh19_io/na_zh19_io", "vc_ramp", {"e_I": 1e-7, "e_s": 2e-8}, "c", _NA_ZH19),
    *(
        ToleranceEntry(name, protocol, limits, "c", _NAV1P1_INIT)
        for config in _NAV1P1
        for name, protocol, limits in (
            (config, "vc_steps", {"e_s": 2e-9}),
            (config, "vc_ramp", {"e_s": 1e-9}),
            (f"{config}:gateCurrent=1", "vc_steps", {"e_s": 2e-9}),
            # The gating current weighs the states with rates up to 150 /ms.
            (f"{config}:gateCurrent=1", "vc_ramp", {"e_I": 2e-10, "e_s": 1e-9}),
        )
    ),
]

TABLE = ToleranceTable(ENTRIES)

NON_FINITE_REFERENCE = {(config, "vc_steps"): (0.0, _GHK_ZERO) for config in _CAV3P3}
"""(configuration, protocol) -> (test potential in mV, known-differences heading).

NEURON's run of that command is not finite, so the command is left out of the
comparison; the catalogue test checks that it is still not finite.
"""

_NO_MOD = "no .mod reference in the repository"
# The candidate references below are ModelDB files without a licence statement;
# whether such files may enter the repository is a pending lead decision.
_MODELDB = "no .mod reference in the repository; candidate reference"
_HM1992 = "no candidate .mod file identified for Huguenard & McCormick (1992)"

UNCOVERED = {
    "AHP_De1994": f"{_MODELDB} ModelDB 3670 IAHP.mod, which also scales tau_m by 3^((celsius - 22)/10) (MECH-6a)",
    "CaHT_HM1992": f"{_NO_MOD}; no published model: CaT_HM1992 with the gates shifted by +25 mV",
    "CaHT_Re1993": f"{_MODELDB} ModelDB 2488 ca.mod, which also scales gbar by tadj (MECH-6a)",
    "CaL_IS2008": f"{_NO_MOD}; port of BrainPy ICaL_IS2008, original model unresolved (MECH-6a)",
    "CaN_IS2008": f"{_NO_MOD}; port of BrainPy ICaN_IS2008, original model unresolved (MECH-6a)",
    "CaT_HM1992": f"{_MODELDB} ModelDB 279 ITGHK.mod, which uses a GHK drive (MECH-6a)",
    "CaT_HP1992": f"{_MODELDB} ModelDB 3670 IT2.mod",
    "HCN_HM1992": f"{_NO_MOD}; {_HM1992}; its +43 mV reversal default is unchecked (MECH-6c)",
    "KA1_HM1992": f"{_NO_MOD}; {_HM1992}",
    "KA2_HM1992": f"{_NO_MOD}; {_HM1992}",
    "KDR_Ba2002": f"{_MODELDB} ModelDB 3670 HH2.mod: the rates of K_TM1991 with V_sh = -50 mV and q10 = 3",
    "KK2A_HM1992": f"{_NO_MOD}; {_HM1992}",
    "KK2B_HM1992": f"{_NO_MOD}; {_HM1992}",
    "KNI_Ya1989": f"{_MODELDB} ModelDB 3817 IM.mod, which ships taumax = 1000 ms and gkbar = 1e-6 S/cm2",
    "K_Leak": f"{_NO_MOD}; no source model: an ohmic leak at E_K with a placeholder default",
    "K_TM1991": f"{_MODELDB} ModelDB 3670 HH2.mod, whose rate factor tadj = 3^((celsius - 36)/10) is 1 at 36 degC",
    "Na_Ba2002": f"{_MODELDB} ModelDB 3670 HH2.mod: Traub-Miles rates with V_sh = -50 mV and q10 = 3 (MECH-6c)",
    "Na_Leak": f"{_NO_MOD}; no source model: an ohmic leak at E_Na with a placeholder default; "
    "compared with a dynamic sodium ion under validation/neuron/ion (MECH-35)",
    "Na_TM1991": f"{_MODELDB} ModelDB 3670 HH2.mod, which scales the rates by 3^((celsius - 36)/10); the class defaults to q10 = 1 (MECH-6c)",
}
"""Registry name -> why the channel has no NEURON comparison configuration."""
