from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CELL_DIR.parents[3]

SOURCE_DCN_DIR = Path("/home/swl/Cerebellum_circuit/DCN/DCN")
SOURCE_TEMPLATE_PATH = SOURCE_DCN_DIR / "DCN_template_1.hoc"
SOURCE_MORPH_PATH = SOURCE_DCN_DIR / "DCN_mor.hoc"

DEFAULT_NRNMECH_PATH = (
    REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "DCN" / "x86_64" / ".libs" / "libnrnmech.so"
)
DEFAULT_NATIVE_DIR = REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "DCN" / "morphology"

DCN_REGION_NAMES = (
    "soma",
    "axHillock",
    "axIniSeg",
    "axNode",
    "proxDend",
    "distDend",
)

EXPECTED_REGION_COUNTS = {
    "soma": 1,
    "axHillock": 1,
    "axIniSeg": 10,
    "axNode": 20,
    "proxDend": 83,
    "distDend": 402,
}
EXPECTED_SOMA_COUNT = EXPECTED_REGION_COUNTS["soma"]
EXPECTED_AXON_COUNT = (
    EXPECTED_REGION_COUNTS["axHillock"] + EXPECTED_REGION_COUNTS["axIniSeg"] + EXPECTED_REGION_COUNTS["axNode"]
)
EXPECTED_DEND_COUNT = EXPECTED_REGION_COUNTS["proxDend"] + EXPECTED_REGION_COUNTS["distDend"]
EXPECTED_TOTAL_COUNT = sum(EXPECTED_REGION_COUNTS.values())


@dataclass(frozen=True)
class DcnTemplateScales:
    celsius: float = 32.0
    temp_orig_dcn: float = 32.0
    q10_channel_gating: float = 3.0
    q10_conductances: float = 1.4
    q10_ca_conc: float = 2.0
    kdr_block: float = 1.0

    @property
    def qdt_channel_gating(self) -> float:
        return self.q10_channel_gating ** ((self.celsius - self.temp_orig_dcn) / 10.0)

    @property
    def qdt_conductances(self) -> float:
        return self.q10_conductances ** ((self.celsius - self.temp_orig_dcn) / 10.0)

    @property
    def qdt_ca_conc(self) -> float:
        return self.q10_ca_conc ** ((self.celsius - self.temp_orig_dcn) / 10.0)


@dataclass(frozen=True)
class DcnTemplateParameters:
    scales: DcnTemplateScales = DcnTemplateScales()

    ra: float = 235.3
    cm: float = 1.57
    passcond: float = 2.81e-5
    passcondmyel_factor: float = 1.0 / 2.81
    shell_thick: float = 0.2

    sodium_rev_pot: float = 71.0
    potassium_rev_pot: float = -90.0
    h_rev_pot: float = -45.0
    tnc_rev_pot: float = -35.0
    calcium_co: float = 2.0
    calcium_ci: float = 50e-6

    g_na_f_soma: float = 2.5e-2
    g_na_p_soma: float = 2e-4
    g_fkdr_soma: float = 1.5e-2
    g_skdr_soma: float = 1.25e-2
    g_sk_soma: float = 2.2e-4
    perm_ca_lva_soma: float = 2.33 * 1.77e-5
    perm_ca_hva_soma: float = 7.5e-6
    tau_ca_conc_soma: float = 70.0
    k_ca_ca_conc_soma: float = 3.45e-7
    k_ca_ca_conc_dend: float = 1.04e-6
    g_h_soma: float = 0.5e-4
    g_tnc_soma: float = 3e-5
    g_tnc_ax: float = 3.5e-5

    def qconductance(self, value: float) -> float:
        return float(value) * self.scales.qdt_conductances

    @property
    def pass_g(self) -> float:
        return self.qconductance(self.passcond)

    @property
    def pass_g_myel(self) -> float:
        return self.qconductance(self.passcond) * self.passcondmyel_factor

    @property
    def qdeltat(self) -> float:
        return self.scales.qdt_channel_gating

    @property
    def tau_ca_conc(self) -> float:
        return self.tau_ca_conc_soma / self.scales.qdt_ca_conc

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["scales"] = self.scales.__dict__.copy()
        return data


def load_dcn15_params(*, temperature_celsius: float = 32.0) -> DcnTemplateParameters:
    scales = DcnTemplateScales(celsius=float(temperature_celsius))
    return DcnTemplateParameters(scales=scales)


def branch_type_for_region(region: str) -> str:
    if region == "soma":
        return "soma"
    if region in {"axHillock", "axIniSeg", "axNode"}:
        return "axon"
    if region in {"proxDend", "distDend"}:
        return "dendrite"
    raise ValueError(f"Unknown DCN region {region!r}.")
