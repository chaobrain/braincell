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

"""Tolerance table of the ion-pool comparisons against NEURON.

Every configuration and protocol not listed here is T0 (``_tolerance.T0_LIMITS``).
Each entry is T1: a limit of the measured value times 1.5, rounded up to 1, 2
or 5 times a power of ten (``_convergence.rounded_limit``), with the deviation
class and the heading of ``validation/neuron/known-differences.md`` that
explains it. Two causes of one protocol are two entries over different metrics.
With every cause of a configuration patched (``tolerances_test.py``), all of
its protocols are T0. Raising a limit or adding an entry for a T0
configuration loosens a tolerance and needs the lead's approval.
"""

from pathlib import Path

from validation.neuron._tolerance import ToleranceEntry, ToleranceTable

KNOWN_DIFFERENCES = Path(__file__).resolve().parents[1] / "known-differences.md"

NEWTON = "Kinetic pools: one linearized implicit Euler step"
EXPLICIT_EULER = "CdpHVA_SU15_DCN explicit Euler update"
FORWARD_DIFFERENCE = "CaHVA_SU15_DCN forward-difference conductance"
GHK = "Unfrozen GHK calcium channels"

_ZERO, _VC, _CC = "pool_zero_current", "pool_vc", "pool_cc"


def _newton(config: str, protocol: str, **limits: float) -> ToleranceEntry:
    return ToleranceEntry(config, protocol, limits, "d", NEWTON)


ENTRIES = [
    ToleranceEntry("dcn_su2015/cdphva_su15_dcn", _ZERO, {"e_Ci": 5e-5}, "c", EXPLICIT_EULER),
    ToleranceEntry("dcn_su2015/cdphva_su15_dcn", _VC, {"e_I": 2e-8, "e_Ci": 5e-4}, "c", EXPLICIT_EULER),
    ToleranceEntry("dcn_su2015/cdphva_su15_dcn", _CC, {"e_Ci": 5e-4}, "c", EXPLICIT_EULER),
    ToleranceEntry("dcn_su2015/cdphva_su15_dcn", _CC, {"e_V": 1e-5}, "c", FORWARD_DIFFERENCE),
    _newton("goc_ma2020/cdpstc_ma20_goc", _ZERO, e_Ci=1e-5, e_pool=2e-2),
    _newton("goc_ma2020/cdpstc_ma20_goc", _VC, e_I=5e-7, e_Ci=1e-5, e_pool=2e-3),
    _newton("goc_ma2020/cdpstc_ma20_goc", _CC, e_Ci=1e-5, e_pool=2e-2),
    _newton("goc_ma2020/cdpstc_ma20_goc_cav2p1_ri21_sc", _VC, e_I=1e-9, e_Ci=5e-4, e_pool=2e-2),
    ToleranceEntry("goc_ma2020/cdpstc_ma20_goc_cav2p1_ri21_sc", _CC, {"e_V": 2.0, "e_Ci": 5e-2, "e_pool": 0.2}, "d", GHK),
    _newton("goc_ma2020/cdpstc_ma20_goc_cav3p1", _VC, e_I=1e-9, e_Ci=2e-4, e_pool=2e-2),
    ToleranceEntry("goc_ma2020/cdpstc_ma20_goc_cav3p1", _CC, {"e_V": 0.5, "e_Ci": 1e-2, "e_pool": 2e-2}, "d", GHK),
    _newton("pc_ma2024/cdp5_cam_pc24_reference", _ZERO, e_Ci=2e-6, e_pool=5e-4),
    _newton("pc_ma2024/cdpcam_ma24_pc", _ZERO, e_Ci=1e-6, e_pool=5e-4),
    _newton("pc_ma2024/cdpcam_ma24_pc", _VC, e_Ci=2e-5, e_pool=5e-4),
    ToleranceEntry("pc_ma2024/cdpcam_ma24_pc", _CC, {"e_V": 0.5, "e_Ci": 5e-3, "e_pool": 2e-2}, "d", GHK),
    _newton("pc_ma2024/cdpcam_ma24_pc_cav3p2", _VC, e_I=2e-8, e_Ci=5e-7, e_pool=2e-4),
    _newton("pc_ma2024/cdpcam_ma24_pc_cav3p2", _CC, e_Ci=5e-7, e_pool=1e-4),
    _newton("testing/toy_ca_binding_kinetic", _ZERO, e_Ci=2e-5, e_pool=1e-5),
    _newton("testing/toy_ca_binding_source_kinetic", _ZERO, e_Ci=2e-5, e_pool=1e-5),
    _newton("testing/toy_ca_pump_factor_kinetic", _ZERO, e_Ci=2e-5, e_pool=1e-5),
    _newton("testing/toy_ca_pump_factor_kinetic", _VC, e_I=5e-9, e_Ci=1e-4, e_pool=1e-4),
    _newton("testing/toy_ca_pump_factor_kinetic", _CC, e_Ci=1e-4, e_pool=1e-4),
    ToleranceEntry("testing/toy_ca_pump_factor_kinetic", _CC, {"e_V": 5e-6}, "c", FORWARD_DIFFERENCE),
    _newton("testing/toy_diam_factor_kinetic", _ZERO, e_Ci=2e-4, e_pool=2e-4),
]

TABLE = ToleranceTable(ENTRIES)
