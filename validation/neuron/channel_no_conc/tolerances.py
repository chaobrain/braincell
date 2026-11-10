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

"""Tolerance table of the channel_no_conc comparisons against NEURON.

Every configuration and protocol not listed here is T0 (``_tolerance.T0_LIMITS``).
Each entry is T1: a limit about 1.5-2x the measured value, rounded up to 1, 2
or 5 times a power of ten, with the deviation class and the heading of
``validation/neuron/known-differences.md`` that explains it. Adding an entry
needs the root cause and a patch that brings it to T0; raising a limit or
moving a T0 configuration here loosens a tolerance and needs the lead's
approval.
"""

from pathlib import Path

from validation.neuron._tolerance import ToleranceEntry, ToleranceTable

KNOWN_DIFFERENCES = Path(__file__).resolve().parents[1] / "known-differences.md"

_GHK = "Unfrozen GHK calcium channels"
_CALVA = "CaLVA forward-difference conductance"
_NA_ZH19 = "Na_ZH2019_IO limit branch"
_KCA2P2 = "Kca2.2 steady-state initialization"


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
        for entry in _current_clamp(config, 5e-4, 1.0, 2e-4, "d", _GHK)
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
]

TABLE = ToleranceTable(ENTRIES)
