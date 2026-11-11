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

"""``dt`` convergence and stiffness table of the channel_no_conc catalogue.

Measured by ``validation/neuron/_convergence.py``, without NEURON; the NEURON
limits are in ``tolerances.py`` and the two tables do not read each other.

- :data:`CONFIGS`: per configuration, the classification of the Cell's
  default path under ``vc_steps`` (or the relaxation protocol) and, for a
  convergent path, the limits of ``e_I`` and ``e_s`` at ``dt = 0.025 ms``:
  about 1.5-2x the measured value, rounded up to 1, 2 or 5 times a power of
  ten. An exact path is asserted at 1e-10 for every ``dt``. ``ramp_dts_ms``
  replaces ``RAMP_DTS_MS`` for a channel whose ramp is still pre-asymptotic
  there.
- :data:`CLASSES`: per channel class, the decade of ``rho_max`` and whether
  the class is stiff at ``dt = 0.025 ms``.
- :data:`KNOWN_SINGULARITIES`: the classes whose gate values, derivatives or
  current are not finite somewhere on the 0.001 mV grid, and where. A fix
  removes its entry; any other non-finite value fails.

:data:`COVERAGE_SCOPE` lists the names the tables must cover; ``None``
covers every registered channel class or every file under ``configs/``, as
both now do. The ``*_FAMILY_*`` tuples name each family's share, which its
slow module checks against the channel modules.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from validation.neuron import _convergence as cv

CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class ConfigEntry:
    """Classification of ``config`` (path under ``configs/`` without ``.json``) and its limits."""

    config: str
    classification: str
    protocol: str = cv.STEPS
    limits: Mapping[str, float] = field(default_factory=dict)
    ramp_dts_ms: tuple[float, ...] = cv.RAMP_DTS_MS

    def __post_init__(self) -> None:
        if self.protocol not in (cv.STEPS, cv.RELAXATION):
            raise ValueError(f"{self.config}: protocol must be {cv.STEPS!r} or {cv.RELAXATION!r}.")
        if self.ramp_dts_ms != cv.RAMP_DTS_MS and not max(self.ramp_dts_ms) < min(cv.RAMP_DTS_MS):
            raise ValueError(f"{self.config}: ramp_dts_ms {self.ramp_dts_ms!r} must be finer than {cv.RAMP_DTS_MS!r}.")
        if self.classification == cv.EXACT:
            if self.limits:
                raise ValueError(f"{self.config}: an exact path is asserted at 1e-10 and takes no limits.")
            return
        if self.classification not in (cv.CONVERGENT, cv.ABOVE_GUIDE_LINE):
            raise ValueError(
                f"{self.config}: classification must be one of {cv.EXACT!r}, {cv.CONVERGENT!r} or "
                f"{cv.ABOVE_GUIDE_LINE!r}; a path that does not converge is fixed, not listed."
            )
        if not self.limits:
            raise ValueError(f"{self.config}: a convergent path needs its e_I/e_s limits.")
        for metric, limit in self.limits.items():
            if metric not in ("e_I", "e_s"):
                raise ValueError(f"{self.config}: unknown metric {metric!r}.")
            if not limit > cv.EXACT_LIMIT:
                raise ValueError(f"{self.config}: the {metric} limit {limit!r} is not above 1e-10.")


@dataclass(frozen=True)
class ClassEntry:
    """``floor(log10(rho_max))`` of channel class ``name`` and whether it is stiff at 0.025 ms."""

    name: str
    decade: int | None
    stiff: bool


CONFIGS = (
    ConfigEntry("ma20_goc/kv3p4_ma20_goc", cv.EXACT),
    # backward_euler x 1, measured 7.8e-2 / 3.2e-2 at 0.025 ms (order 0.90 / 0.95).
    ConfigEntry("ma20_goc/nav1p6_ma20_goc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    # calcium.py and potassium_calcium.py: HH channels on ind_exp_euler are exact (<= 2.9e-14 at every dt).
    ConfigEntry("ma20_goc/cahva_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/cav1p2_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/cav1p3_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/cav2p3_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/cav3p1_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/cav3p1_ma20_goc_frozen", cv.EXACT),
    # backward_euler x 1, measured 2.208e-02 / 1.175e-04 at 0.025 ms (order 0.97 / 0.97).
    ConfigEntry("ma20_goc/kca1p1_ma20_goc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-2, "e_s": 2e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ma20_goc/kca2p2_ma20_goc", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    ConfigEntry("ma20_goc/kca3p1_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_grc/cahva_ma20_grc", cv.EXACT),
    # backward_euler x 1, measured 2.208e-02 / 1.175e-04 at 0.025 ms (order 0.97 / 0.97).
    ConfigEntry("ma20_grc/kca1p1_ma20_grc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-2, "e_s": 2e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ma20_grc/kca2p2_ma20_grc", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    ConfigEntry("ma24_pc/cav2p1_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/cav2p1_ma24_pc_frozen", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p1_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p1_ma24_pc_frozen", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p1_test2_pc24", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p1_test_pc24", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p2_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p3_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/cav3p3_ma24_pc_frozen", cv.EXACT),
    # backward_euler x 1, measured 2.208e-02 / 1.175e-04 at 0.025 ms (order 0.97 / 0.97).
    ConfigEntry("ma24_pc/kca1p1_ma24_pc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-2, "e_s": 2e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ma24_pc/kca2p2_ma24_pc", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ma24_pc/kca2p2_ma24_pc_dep_o1", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    ConfigEntry("ma24_pc/kca3p1_ma24_pc", cv.EXACT),
    ConfigEntry("ma25_bc/cav1p2_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/cav1p3_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/cav2p1_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/cav2p1_ma25_bc_frozen", cv.EXACT),
    ConfigEntry("ma25_bc/cav3p2_ma25_bc", cv.EXACT),
    # backward_euler x 1, measured 2.208e-02 / 1.175e-04 at 0.025 ms (order 0.97 / 0.97).
    ConfigEntry("ma25_bc/kca1p1_ma25_bc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-2, "e_s": 2e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ma25_bc/kca2p2_ma25_bc", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    ConfigEntry("ma25_bc/kca3p1_ma25_bc", cv.EXACT),
    ConfigEntry("ri21_sc/cav2p1_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/cav2p1_ri21_sc_frozen", cv.EXACT),
    ConfigEntry("ri21_sc/cav3p2_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/cav3p3_ri21_sc", cv.EXACT),
    # backward_euler x 1, measured 2.208e-02 / 1.175e-04 at 0.025 ms (order 0.97 / 0.97).
    ConfigEntry("ri21_sc/kca1p1_ri21_sc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-2, "e_s": 2e-4}),
    # Relaxation, backward_euler x 1, measured e_s 6.258e-05 at 0.025 ms (order 0.98).
    ConfigEntry("ri21_sc/kca2p2_ri21_sc", cv.CONVERGENT, protocol=cv.RELAXATION, limits={"e_s": 1e-4}),
    ConfigEntry("su15_dcn/cahva_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/cal_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/calva_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/sk_su15_dcn", cv.EXACT, protocol=cv.RELAXATION),
    ConfigEntry("zh19_io/ca_zh19_io", cv.EXACT),
    ConfigEntry("zh19_io/ca_zh19_io_frozen", cv.EXACT),
    # potassium.py and potassium_sodium.py: HH channels on ind_exp_euler are exact (<= 1.8e-14 at every dt).
    ConfigEntry("builtin/k_hh1952_hh", cv.EXACT),
    ConfigEntry("ma20_goc/km_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/kv1p1_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/kv4p3_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_grc/kir2p3_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/km_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/kv1p1_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/kv1p5_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/kv2p2_0010_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/kv3p4_ma20_grc", cv.EXACT),
    ConfigEntry("ma20_grc/kv4p3_ma20_grc", cv.EXACT),
    ConfigEntry("ma24_pc/kir2p3_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/kv1p1_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/kv1p5_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/kv3p3_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/kv3p4_ma24_pc", cv.EXACT),
    ConfigEntry("ma24_pc/kv4p3_ma24_pc", cv.EXACT),
    ConfigEntry("ma25_bc/kir2p3_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/kv1p1_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/kv3p4_ma25_bc", cv.EXACT),
    ConfigEntry("ma25_bc/kv4p3_ma25_bc", cv.EXACT),
    ConfigEntry("ri21_sc/kir2p3_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/km_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/kv1p1_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/kv3p4_ri21_sc", cv.EXACT),
    ConfigEntry("ri21_sc/kv4p3_ri21_sc", cv.EXACT),
    ConfigEntry("su15_dcn/fkdr_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/skdr_su15_dcn", cv.EXACT),
    ConfigEntry("testing/kv_testing", cv.EXACT),
    ConfigEntry("zh19_io/kdr_zh19_io", cv.EXACT),
    # sodium.py, hyperpolarization_activated.py and leaky.py: HH channels on ind_exp_euler are
    # exact (<= 1.4e-14 at every dt), IL has no state (e_I 0 at every dt).
    # Na_HH1952 at 36 degC with temp_ref 6.3 degC (rates x 26, up to 28 /ms): on RAMP_DTS_MS the
    # e_I Richardson order is 1.37, then 0.43, 0.68, 0.86, 0.93 halving down to 0.00078125 ms.
    ConfigEntry("builtin/na_hh1952_hh", cv.EXACT, ramp_dts_ms=(0.00625, 0.003125, 0.0015625)),
    ConfigEntry("builtin/il_pas", cv.EXACT),
    ConfigEntry("ma20_goc/hcn1_ma20_goc", cv.EXACT),
    ConfigEntry("ma20_goc/hcn2_ma20_goc", cv.EXACT),
    # backward_euler x 1, measured 1.957e-01 / 4.213e-02 at 0.025 ms (order 0.94 / 0.95).
    ConfigEntry("ma20_grc/nafhf_ma20_grc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-1, "e_s": 1e-1}),
    # backward_euler x 1, measured 1.383e-01 / 3.930e-02 at 0.025 ms (order 0.94 / 0.95).
    ConfigEntry("ma20_grc/nav_ma20_grc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 5e-1, "e_s": 1e-1}),
    ConfigEntry("ma24_pc/hcn1_ma24_pc", cv.EXACT),
    # backward_euler x 1, measured 7.841e-02 / 3.241e-02 at 0.025 ms (order 0.90 / 0.95).
    ConfigEntry("ma24_pc/nav1p6_ma24_pc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    ConfigEntry("ma25_bc/hcn1_ma25_bc", cv.EXACT),
    # backward_euler x 1, measured 6.998e-02 / 2.993e-02 at 0.025 ms (order 0.89 / 0.95).
    ConfigEntry("ma25_bc/nav1p1_ma25_bc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    # backward_euler x 1, measured 7.841e-02 / 3.241e-02 at 0.025 ms (order 0.90 / 0.95).
    ConfigEntry("ma25_bc/nav1p6_ma25_bc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    ConfigEntry("ri21_sc/hcn1_ri21_sc", cv.EXACT),
    # backward_euler x 1, measured 6.998e-02 / 2.993e-02 at 0.025 ms (order 0.89 / 0.95).
    ConfigEntry("ri21_sc/nav1p1_ri21_sc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    # backward_euler x 1, measured 7.841e-02 / 3.241e-02 at 0.025 ms (order 0.90 / 0.95).
    ConfigEntry("ri21_sc/nav1p6_ri21_sc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
    ConfigEntry("su15_dcn/hcn_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/naf_su15_dcn", cv.EXACT),
    ConfigEntry("su15_dcn/nap_su15_dcn", cv.EXACT),
    ConfigEntry("zh19_io/hcn_zh19_io", cv.EXACT),
    ConfigEntry("zh19_io/na_zh19_io", cv.EXACT),
    # chloride.py: Cl_Leak has no state (e_I 0 at every dt).
    ConfigEntry("testing/cl_leak_testing", cv.EXACT),
)

CLASSES = (
    ClassEntry("Kv3p4_MA2020_GoC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Nav1p6_MA2020_GoC", 5, True),  # 1.96e5 /ms at +60 mV
    ClassEntry("Cl_Leak", None, False),  # stateless
    # calcium.py and potassium_calcium.py (Ci = the placeholder ion's, 36 degC).
    ClassEntry("AHP_De1994", -2, False),  # 0.09 /ms, p gate at -100 mV
    ClassEntry("CaHT_HM1992", 0, False),  # 6.83 /ms, p gate at +60 mV
    ClassEntry("CaHT_Re1993", 1, False),  # 14.1 /ms, p gate at +60 mV
    ClassEntry("CaHVA_MA2020_GoC", 1, False),  # 78.4 /ms, s gate at +60 mV
    ClassEntry("CaHVA_MA2020_GrC", 1, False),  # 78.4 /ms, s gate at +60 mV
    ClassEntry("CaHVA_SU2015_DCN", 1, False),  # 31.2 /ms, m gate at +60 mV
    ClassEntry("CaLVA_SU2015_DCN", 0, False),  # 4.78 /ms, m gate at +60 mV
    ClassEntry("CaL_IS2008", 1, False),  # 11.4 /ms, p gate at -100 mV
    ClassEntry("CaL_SU2015_DCN", 0, False),  # 4.78 /ms, m gate at +60 mV
    ClassEntry("CaN_IS2008", -1, False),  # 0.625 /ms, p gate at +60 mV
    ClassEntry("CaT_HM1992", 0, False),  # 7.32 /ms, p gate at +60 mV
    ClassEntry("CaT_HP1992", 0, False),  # 2.3 /ms, p gate at +60 mV
    ClassEntry("Ca_ZH2019_IO", -2, False),  # 0.025 /ms, h gate at +60 mV
    ClassEntry("Ca_ZH2019_IO_Frozen", -2, False),  # 0.025 /ms, h gate at +60 mV
    ClassEntry("Cav1p2_MA2020_GoC", 6, True),  # 3.66e6 /ms, m gate at -100 mV
    ClassEntry("Cav1p2_MA2025_BC", 6, True),  # 3.66e6 /ms, m gate at -100 mV
    ClassEntry("Cav1p3_MA2020_GoC", 6, True),  # 1.47e6 /ms, m gate at -100 mV
    ClassEntry("Cav1p3_MA2025_BC", 6, True),  # 1.47e6 /ms, m gate at -100 mV
    ClassEntry("Cav2p1_MA2024_PC", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p1_MA2024_PC_Frozen", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p1_MA2025_BC", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p1_MA2025_BC_Frozen", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p1_RI2021_SC", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p1_RI2021_SC_Frozen", 1, False),  # 15.4 /ms, m gate at +52.5 mV
    ClassEntry("Cav2p3_MA2020_GoC", -1, False),  # 0.2 /ms, h gate at -100 mV
    ClassEntry("Cav3p1Test_PC24", 0, False),  # 1 /ms, p gate at -100 mV
    ClassEntry("Cav3p1_MA2020_GoC", 0, False),  # 1 /ms, p gate at -100 mV
    ClassEntry("Cav3p1_MA2020_GoC_Frozen", 0, False),  # 1 /ms, p gate at -100 mV
    ClassEntry("Cav3p1_MA2024_PC", 0, False),  # 1 /ms, p gate at -100 mV
    ClassEntry("Cav3p1_MA2024_PC_Frozen", 0, False),  # 1 /ms, p gate at -100 mV
    ClassEntry("Cav3p2_MA2024_PC", 0, False),  # 3.63 /ms, m gate at +60 mV
    ClassEntry("Cav3p2_MA2025_BC", 0, False),  # 3.63 /ms, m gate at +60 mV
    ClassEntry("Cav3p2_RI2021_SC", 0, False),  # 3.63 /ms, m gate at +60 mV
    ClassEntry("Cav3p3_MA2024_PC", 0, False),  # 1.37 /ms, n gate at -100 mV
    ClassEntry("Cav3p3_MA2024_PC_Frozen", 0, False),  # 1.37 /ms, n gate at -100 mV
    ClassEntry("Cav3p3_RI2021_SC", 0, False),  # 1.37 /ms, n gate at -100 mV
    ClassEntry("Kca1p1_MA2020_GoC", 2, True),  # 204 /ms, at -100 mV
    ClassEntry("Kca1p1_MA2020_GrC", 2, True),  # 204 /ms, at -100 mV
    ClassEntry("Kca1p1_MA2024_PC", 2, True),  # 204 /ms, at -100 mV
    ClassEntry("Kca1p1_MA2025_BC", 2, True),  # 204 /ms, at -100 mV
    ClassEntry("Kca1p1_RI2021_SC", 2, True),  # 204 /ms, at -100 mV
    ClassEntry("Kca2p2_MA2020_GoC", 0, False),  # 6.2 /ms, at -100 mV
    ClassEntry("Kca2p2_MA2020_GrC", 0, False),  # 6.2 /ms, at -100 mV
    ClassEntry("Kca2p2_MA2024_PC", 0, False),  # 6.2 /ms, at -100 mV
    ClassEntry("Kca2p2_MA2025_BC", 0, False),  # 6.2 /ms, at -100 mV
    ClassEntry("Kca2p2_RI2021_SC", 0, False),  # 6.2 /ms, at -100 mV
    ClassEntry("Kca3p1_MA2020_GoC", -2, False),  # 0.0593 /ms, p gate at +60 mV
    ClassEntry("Kca3p1_MA2024_PC", -2, False),  # 0.0593 /ms, p gate at +60 mV
    ClassEntry("Kca3p1_MA2025_BC", -2, False),  # 0.0593 /ms, p gate at +60 mV
    ClassEntry("SK_SU2015_DCN", 0, False),  # 1.01 /ms, z gate at -100 mV
    # potassium.py and potassium_sodium.py (36 degC).
    ClassEntry("KA1_HM1992", 0, False),  # 2.65 /ms, p gate at +60 mV
    ClassEntry("KA2_HM1992", 0, False),  # 2.65 /ms, p gate at +60 mV
    ClassEntry("KDR_Ba2002", 0, False),  # 3.08 /ms, p gate at +60 mV
    ClassEntry("KK2A_HM1992", -2, False),  # 0.0822 /ms, p gate at +60 mV
    ClassEntry("KK2B_HM1992", -1, False),  # 0.112 /ms, q gate at -55.5 mV
    ClassEntry("KM_MA2020_GoC", -1, False),  # 0.511 /ms, n gate at -100 mV
    ClassEntry("KM_MA2020_GrC", -1, False),  # 0.511 /ms, n gate at -100 mV
    ClassEntry("KM_RI2021_SC", -1, False),  # 0.511 /ms, n gate at -100 mV
    ClassEntry("KNI_Ya1989", -2, False),  # 0.0954 /ms, p gate at +60 mV
    ClassEntry("K_HH1952", 0, False),  # 1.18 /ms, p gate at +60 mV
    ClassEntry("K_Kv_test", -1, False),  # 0.75 /ms, n gate at -100 mV
    ClassEntry("K_Leak", None, False),  # stateless
    ClassEntry("K_TM1991", 0, False),  # 3.39 /ms, p gate at +60 mV
    ClassEntry("Kdr_ZH2019_IO", 1, False),  # 10.4 /ms, n gate at +60 mV
    ClassEntry("Kir2p3_MA2020_GrC", 1, False),  # 55.5 /ms, d gate at +60 mV
    ClassEntry("Kir2p3_MA2024_PC", 1, False),  # 55.5 /ms, d gate at +60 mV
    ClassEntry("Kir2p3_MA2025_BC", 1, False),  # 55.5 /ms, d gate at +60 mV
    ClassEntry("Kir2p3_RI2021_SC", 1, False),  # 55.5 /ms, d gate at +60 mV
    ClassEntry("Kv1p1_MA2020_GoC", 1, False),  # 43.5 /ms, n gate at -100 mV
    ClassEntry("Kv1p1_MA2020_GrC", 1, False),  # 43.5 /ms, n gate at -100 mV
    ClassEntry("Kv1p1_MA2024_PC", 1, False),  # 43.5 /ms, n gate at -100 mV
    ClassEntry("Kv1p1_MA2025_BC", 1, False),  # 43.5 /ms, n gate at -100 mV
    ClassEntry("Kv1p1_RI2021_SC", 1, False),  # 43.5 /ms, n gate at -100 mV
    ClassEntry("Kv1p5_MA2020_GrC", 0, False),  # 3 /ms, m gate at +60 mV
    ClassEntry("Kv1p5_MA2024_PC", 0, False),  # 3 /ms, m gate at +60 mV
    ClassEntry("Kv2p2_0010_MA2020_GrC", -2, False),  # 0.0335 /ms, m gate at -100 mV
    ClassEntry("Kv3p3_MA2024_PC", 1, False),  # 21.1 /ms, n gate at -100 mV
    ClassEntry("Kv3p4_MA2020_GrC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Kv3p4_MA2024_PC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Kv3p4_MA2025_BC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Kv3p4_RI2021_SC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Kv4p3_MA2020_GoC", 1, False),  # 34.9 /ms, a gate at -100 mV
    ClassEntry("Kv4p3_MA2020_GrC", 1, False),  # 34.9 /ms, a gate at -100 mV
    ClassEntry("Kv4p3_MA2024_PC", 1, False),  # 34.9 /ms, a gate at -100 mV
    ClassEntry("Kv4p3_MA2025_BC", 1, False),  # 34.9 /ms, a gate at -100 mV
    ClassEntry("Kv4p3_RI2021_SC", 1, False),  # 34.9 /ms, a gate at -100 mV
    ClassEntry("fKdr_SU2015_DCN", 0, False),  # 9.68 /ms, m gate at +60 mV
    ClassEntry("sKdr_SU2015_DCN", 0, False),  # 6.9 /ms, m gate at +60 mV
    # sodium.py, hyperpolarization_activated.py and leaky.py (36 degC).
    ClassEntry("HCN1_MA2020_GoC", -2, False),  # 0.042 /ms, o_fast gate at -100 mV
    ClassEntry("HCN1_MA2024_PC", 5, True),  # 3.62e5 /ms, h gate at +60 mV
    ClassEntry("HCN1_MA2025_BC", 5, True),  # 3.62e5 /ms, h gate at +60 mV
    ClassEntry("HCN1_RI2021_SC", 5, True),  # 3.62e5 /ms, h gate at +60 mV
    ClassEntry("HCN2_MA2020_GoC", -3, False),  # 0.005 /ms, o_fast gate at -100 mV
    ClassEntry("HCN_HM1992", 1, False),  # 10.3 /ms, p gate at +60 mV
    ClassEntry("HCN_SU2015_DCN", -3, False),  # 0.0025 /ms, m gate at -79.25 mV
    ClassEntry("HCN_ZH2019_IO", 1, False),  # 10.3 /ms, q gate at +60 mV
    ClassEntry("IL", None, False),  # stateless
    ClassEntry("NaFHF_MA2020_GrC", 6, True),  # 5.54e6 /ms at +60 mV
    ClassEntry("NaF_SU2015_DCN", 1, False),  # 39.9 /ms, m gate at -100 mV
    ClassEntry("NaP_SU2015_DCN", -2, False),  # 0.02 /ms, m gate at -100 mV
    ClassEntry("Na_Ba2002", 1, False),  # 31 /ms, p gate at +60 mV
    ClassEntry("Na_HH1952", 1, False),  # 28 /ms, p gate at -100 mV
    ClassEntry("Na_Leak", None, False),  # stateless
    ClassEntry("Na_TM1991", 1, False),  # 35.2 /ms, p gate at +60 mV
    ClassEntry("Na_ZH2019_IO", 3, True),  # 1e3 /ms, m gate at -99.5 mV
    ClassEntry("Nav1p1_MA2025_BC", 5, True),  # 2.24e5 /ms at +60 mV
    ClassEntry("Nav1p1_RI2021_SC", 5, True),  # 2.24e5 /ms at +60 mV
    ClassEntry("Nav1p6_MA2024_PC", 5, True),  # 1.96e5 /ms at +60 mV
    ClassEntry("Nav1p6_MA2025_BC", 5, True),  # 1.96e5 /ms at +60 mV
    ClassEntry("Nav1p6_RI2021_SC", 5, True),  # 1.96e5 /ms at +60 mV
    ClassEntry("Nav_MA2020_GrC", 6, True),  # 2.84e6 /ms at +60 mV
)

KNOWN_SINGULARITIES: dict[str, dict[str, tuple[float, ...]]] = {}
"""Removable 0/0 singularities, each to be replaced by its analytic limit; none is left."""

CALCIUM_FAMILY_CONFIGS = (
    "ma20_goc/cahva_ma20_goc", "ma20_goc/cav1p2_ma20_goc", "ma20_goc/cav1p3_ma20_goc",
    "ma20_goc/cav2p3_ma20_goc", "ma20_goc/cav3p1_ma20_goc", "ma20_goc/cav3p1_ma20_goc_frozen",
    "ma20_goc/kca1p1_ma20_goc", "ma20_goc/kca2p2_ma20_goc", "ma20_goc/kca3p1_ma20_goc",
    "ma20_grc/cahva_ma20_grc", "ma20_grc/kca1p1_ma20_grc", "ma20_grc/kca2p2_ma20_grc", "ma24_pc/cav2p1_ma24_pc",
    "ma24_pc/cav2p1_ma24_pc_frozen", "ma24_pc/cav3p1_ma24_pc", "ma24_pc/cav3p1_ma24_pc_frozen",
    "ma24_pc/cav3p1_test2_pc24", "ma24_pc/cav3p1_test_pc24", "ma24_pc/cav3p2_ma24_pc", "ma24_pc/cav3p3_ma24_pc",
    "ma24_pc/cav3p3_ma24_pc_frozen", "ma24_pc/kca1p1_ma24_pc", "ma24_pc/kca2p2_ma24_pc",
    "ma24_pc/kca2p2_ma24_pc_dep_o1", "ma24_pc/kca3p1_ma24_pc", "ma25_bc/cav1p2_ma25_bc",
    "ma25_bc/cav1p3_ma25_bc", "ma25_bc/cav2p1_ma25_bc", "ma25_bc/cav2p1_ma25_bc_frozen",
    "ma25_bc/cav3p2_ma25_bc", "ma25_bc/kca1p1_ma25_bc", "ma25_bc/kca2p2_ma25_bc", "ma25_bc/kca3p1_ma25_bc",
    "ri21_sc/cav2p1_ri21_sc", "ri21_sc/cav2p1_ri21_sc_frozen", "ri21_sc/cav3p2_ri21_sc",
    "ri21_sc/cav3p3_ri21_sc", "ri21_sc/kca1p1_ri21_sc", "ri21_sc/kca2p2_ri21_sc", "su15_dcn/cahva_su15_dcn",
    "su15_dcn/cal_su15_dcn", "su15_dcn/calva_su15_dcn", "su15_dcn/sk_su15_dcn", "zh19_io/ca_zh19_io",
    "zh19_io/ca_zh19_io_frozen",
)
"""The configurations of the classes in ``calcium.py`` and ``potassium_calcium.py``."""
CALCIUM_FAMILY_CLASSES = (
    "AHP_De1994", "CaHT_HM1992", "CaHT_Re1993", "CaHVA_MA2020_GoC", "CaHVA_MA2020_GrC", "CaHVA_SU2015_DCN",
    "CaLVA_SU2015_DCN", "CaL_IS2008", "CaL_SU2015_DCN", "CaN_IS2008", "CaT_HM1992", "CaT_HP1992",
    "Ca_ZH2019_IO", "Ca_ZH2019_IO_Frozen", "Cav1p2_MA2020_GoC", "Cav1p2_MA2025_BC", "Cav1p3_MA2020_GoC",
    "Cav1p3_MA2025_BC", "Cav2p1_MA2024_PC", "Cav2p1_MA2024_PC_Frozen", "Cav2p1_MA2025_BC",
    "Cav2p1_MA2025_BC_Frozen", "Cav2p1_RI2021_SC", "Cav2p1_RI2021_SC_Frozen", "Cav2p3_MA2020_GoC",
    "Cav3p1Test_PC24", "Cav3p1_MA2020_GoC", "Cav3p1_MA2020_GoC_Frozen", "Cav3p1_MA2024_PC",
    "Cav3p1_MA2024_PC_Frozen", "Cav3p2_MA2024_PC", "Cav3p2_MA2025_BC", "Cav3p2_RI2021_SC", "Cav3p3_MA2024_PC",
    "Cav3p3_MA2024_PC_Frozen", "Cav3p3_RI2021_SC", "Kca1p1_MA2020_GoC", "Kca1p1_MA2020_GrC", "Kca1p1_MA2024_PC",
    "Kca1p1_MA2025_BC", "Kca1p1_RI2021_SC", "Kca2p2_MA2020_GoC", "Kca2p2_MA2020_GrC", "Kca2p2_MA2024_PC",
    "Kca2p2_MA2025_BC", "Kca2p2_RI2021_SC", "Kca3p1_MA2020_GoC", "Kca3p1_MA2024_PC", "Kca3p1_MA2025_BC",
    "SK_SU2015_DCN",
)
"""The registered channel classes defined in ``calcium.py`` and ``potassium_calcium.py``."""

POTASSIUM_FAMILY_CONFIGS = (
    "builtin/k_hh1952_hh", "ma20_goc/km_ma20_goc", "ma20_goc/kv1p1_ma20_goc", "ma20_goc/kv3p4_ma20_goc",
    "ma20_goc/kv4p3_ma20_goc", "ma20_grc/kir2p3_ma20_grc", "ma20_grc/km_ma20_grc", "ma20_grc/kv1p1_ma20_grc",
    "ma20_grc/kv1p5_ma20_grc", "ma20_grc/kv2p2_0010_ma20_grc", "ma20_grc/kv3p4_ma20_grc", "ma20_grc/kv4p3_ma20_grc",
    "ma24_pc/kir2p3_ma24_pc", "ma24_pc/kv1p1_ma24_pc", "ma24_pc/kv1p5_ma24_pc", "ma24_pc/kv3p3_ma24_pc",
    "ma24_pc/kv3p4_ma24_pc", "ma24_pc/kv4p3_ma24_pc", "ma25_bc/kir2p3_ma25_bc", "ma25_bc/kv1p1_ma25_bc",
    "ma25_bc/kv3p4_ma25_bc", "ma25_bc/kv4p3_ma25_bc", "ri21_sc/kir2p3_ri21_sc", "ri21_sc/km_ri21_sc",
    "ri21_sc/kv1p1_ri21_sc", "ri21_sc/kv3p4_ri21_sc", "ri21_sc/kv4p3_ri21_sc", "su15_dcn/fkdr_su15_dcn",
    "su15_dcn/skdr_su15_dcn", "testing/kv_testing", "zh19_io/kdr_zh19_io",
)
"""The configurations of the classes in ``potassium.py`` and ``potassium_sodium.py``."""
POTASSIUM_FAMILY_CLASSES = (
    "KA1_HM1992", "KA2_HM1992", "KDR_Ba2002", "KK2A_HM1992", "KK2B_HM1992", "KM_MA2020_GoC", "KM_MA2020_GrC",
    "KM_RI2021_SC", "KNI_Ya1989", "K_HH1952", "K_Kv_test", "K_Leak", "K_TM1991", "Kdr_ZH2019_IO",
    "Kir2p3_MA2020_GrC", "Kir2p3_MA2024_PC", "Kir2p3_MA2025_BC", "Kir2p3_RI2021_SC", "Kv1p1_MA2020_GoC",
    "Kv1p1_MA2020_GrC", "Kv1p1_MA2024_PC", "Kv1p1_MA2025_BC", "Kv1p1_RI2021_SC", "Kv1p5_MA2020_GrC",
    "Kv1p5_MA2024_PC", "Kv2p2_0010_MA2020_GrC", "Kv3p3_MA2024_PC", "Kv3p4_MA2020_GoC", "Kv3p4_MA2020_GrC",
    "Kv3p4_MA2024_PC", "Kv3p4_MA2025_BC", "Kv3p4_RI2021_SC", "Kv4p3_MA2020_GoC", "Kv4p3_MA2020_GrC",
    "Kv4p3_MA2024_PC", "Kv4p3_MA2025_BC", "Kv4p3_RI2021_SC", "fKdr_SU2015_DCN", "sKdr_SU2015_DCN",
)
"""The registered channel classes defined in ``potassium.py`` and ``potassium_sodium.py``."""

SODIUM_HCN_LEAK_FAMILY_CONFIGS = (
    "builtin/il_pas", "builtin/na_hh1952_hh", "ma20_goc/hcn1_ma20_goc", "ma20_goc/hcn2_ma20_goc",
    "ma20_goc/nav1p6_ma20_goc", "ma20_grc/nafhf_ma20_grc", "ma20_grc/nav_ma20_grc", "ma24_pc/hcn1_ma24_pc",
    "ma24_pc/nav1p6_ma24_pc", "ma25_bc/hcn1_ma25_bc", "ma25_bc/nav1p1_ma25_bc", "ma25_bc/nav1p6_ma25_bc",
    "ri21_sc/hcn1_ri21_sc", "ri21_sc/nav1p1_ri21_sc", "ri21_sc/nav1p6_ri21_sc", "su15_dcn/hcn_su15_dcn",
    "su15_dcn/naf_su15_dcn", "su15_dcn/nap_su15_dcn", "zh19_io/hcn_zh19_io", "zh19_io/na_zh19_io",
)
"""The configurations of the classes in ``sodium.py``, ``hyperpolarization_activated.py`` and ``leaky.py``."""
SODIUM_HCN_LEAK_FAMILY_CLASSES = (
    "HCN1_MA2020_GoC", "HCN1_MA2024_PC", "HCN1_MA2025_BC", "HCN1_RI2021_SC", "HCN2_MA2020_GoC", "HCN_HM1992",
    "HCN_SU2015_DCN", "HCN_ZH2019_IO", "IL", "NaFHF_MA2020_GrC", "NaF_SU2015_DCN", "NaP_SU2015_DCN", "Na_Ba2002",
    "Na_HH1952", "Na_Leak", "Na_TM1991", "Na_ZH2019_IO", "Nav1p1_MA2025_BC", "Nav1p1_RI2021_SC", "Nav1p6_MA2020_GoC",
    "Nav1p6_MA2024_PC", "Nav1p6_MA2025_BC", "Nav1p6_RI2021_SC", "Nav_MA2020_GrC",
)
"""The registered channel classes defined in ``sodium.py``, ``hyperpolarization_activated.py`` and ``leaky.py``."""

COVERAGE_SCOPE: dict[str, tuple[str, ...] | None] = {"configs": None, "classes": None}
"""Every configuration and every registered channel class must have an entry."""


def assert_matches(report: cv.ConvergenceReport, entry: ConfigEntry) -> None:
    """Raise ``AssertionError`` unless ``report`` has the protocol, classification and limits of ``entry``.

    An exact entry needs every error at every ``dt`` within 1e-10; a
    convergent one needs its fitted orders in the step window and its errors
    at 0.025 ms within the entry's limits.
    """
    if report.protocol != entry.protocol:
        raise AssertionError(f"{entry.config}: protocol {report.protocol!r}, table {entry.protocol!r}")
    if report.classification != entry.classification:
        raise AssertionError(f"{entry.config}: {report.classification!r}, table {entry.classification!r}; errors {report.errors}")
    if entry.classification == cv.EXACT:
        for dt, values in report.errors.items():
            for metric, value in values.items():
                if value is not None and not value <= cv.EXACT_LIMIT:
                    raise AssertionError(f"{entry.config} dt = {dt} {metric}: {value:.3e} > {cv.EXACT_LIMIT}")
        return
    low, high = cv.STEP_ORDER_WINDOW
    for metric, order in report.orders.items():
        if not low <= order <= high:
            raise AssertionError(f"{entry.config} {metric}: order {order:.3f} outside [{low}, {high}]")
    for metric, limit in entry.limits.items():
        if not report.error(metric) <= limit:
            raise AssertionError(f"{entry.config} {metric}: {report.error(metric):.3e} > {limit}")


def config_names() -> tuple[str, ...]:
    """Every configuration under ``configs/``, as a path without ``.json``."""
    return tuple(sorted(path.relative_to(CONFIGS_ROOT).with_suffix("").as_posix() for path in CONFIGS_ROOT.rglob("*.json")))


def class_names() -> tuple[str, ...]:
    """Every registered channel class."""
    from braincell.mech import get_registry

    return tuple(sorted(get_registry().names("channel")))


def coverage_problems(
    configs: Iterable[ConfigEntry] = CONFIGS,
    classes: Iterable[ClassEntry] = CLASSES,
    scope: Mapping[str, Iterable[str] | None] = COVERAGE_SCOPE,
) -> list[str]:
    """Return what the tables miss in ``scope`` and the names they list that do not exist."""
    problems = []
    for kind, entries, existing in (
        ("configs", [entry.config for entry in configs], config_names()),
        ("classes", [entry.name for entry in classes], class_names()),
    ):
        duplicates = sorted({name for name in entries if entries.count(name) > 1})
        problems += [f"{kind}: {name} is listed twice" for name in duplicates]
        problems += [f"{kind}: {name} does not exist" for name in entries if name not in existing]
        required = existing if scope[kind] is None else scope[kind]
        problems += [f"{kind}: {name} is not in the table" for name in required if name not in entries]
    return problems


def assert_covered(**kwargs) -> None:
    """Raise ``AssertionError`` listing every :func:`coverage_problems` item."""
    problems = coverage_problems(**kwargs)
    if problems:
        raise AssertionError("convergence table coverage:\n" + "\n".join(problems))
