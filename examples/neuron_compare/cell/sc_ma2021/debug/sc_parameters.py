from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CELL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CELL_DIR.parents[4]

SOURCE_SC_DIR = Path("/home/swl/Cerebellum_circuit/SC2021")
SOURCE_MORPH_PATH = SOURCE_SC_DIR / "morphology" / "stellate.ASC"
SOURCE_PARAMETERS_PATH = SOURCE_SC_DIR / "SC_param.py"

DEFAULT_MORPH_PATH = REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "SC" / "morphology" / "SC.asc"
DEFAULT_NRNMECH_PATH = (
    REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "SC" / "x86_64" / ".libs" / "libnrnmech.so"
)

RA_OHM_CM = 110.0
LEAK_E_MV = -52.0
NA_E_MV = 60.0
K_E_MV = -84.0
K_E_AXON_MV = -88.0
H_E_MV = -34.0
CA_E_MV = 137.5
SOMA_CM_UF_CM2 = 1.0
DEND_CM_UF_CM2 = 1.5
AXON_CM_UF_CM2 = 1.0
CV_MAX_LEN_UM = 40.0
CDP_PUMP_SOMA = 1e-8
CDP_PUMP_DEND = 1e-9
CDP_PUMP_AXON = 1e-9
CAV3P3_G_SCALE = 1.0e-5

EXPECTED_SOMA_COUNT = 1
EXPECTED_DEND_COUNT = 104
EXPECTED_AXON_COUNT = 15

DENDPROX_INDICES = (2, 3, 15, 16, 20, 31, 34, 35, 36, 50, 66, 67, 81, 103)

ALL_REGION_LOGICAL_MECHANISMS = {
    "soma": (
        "leak",
        "nav1p1",
        "cav3p2",
        "cav3p3",
        "kir2p3",
        "kv1p1",
        "kv3p4",
        "kv4p3",
        "kca1p1",
        "kca2p2",
        "cav2p1",
        "hcn1",
        "cdp",
    ),
    "dendprox": (
        "leak",
        "cav2p1",
        "cav3p2",
        "cav3p3",
        "kca1p1",
        "kca2p2",
        "kv4p3",
        "kv1p1",
        "cdp",
    ),
    "denddist": (
        "leak",
        "cav2p1",
        "kca1p1",
        "kca2p2",
        "kv1p1",
        "cdp",
    ),
    "axon_ais": (
        "leak",
        "nav1p6",
        "kv3p4",
        "kv1p1",
        "hcn1",
        "km",
        "cdp",
    ),
    "axon_regular": (
        "leak",
        "nav1p6",
        "kv3p4",
        "kv1p1",
        "hcn1",
        "cdp",
    ),
}


@dataclass(frozen=True)
class SCToggles:
    leak: bool = True
    nav1p1: bool = True
    nav1p6: bool = True
    cav2p1: bool = True
    cav3p2: bool = True
    cav3p3: bool = True
    kir2p3: bool = True
    kv1p1: bool = True
    kv3p4: bool = True
    kv4p3: bool = True
    km: bool = True
    kca1p1: bool = True
    kca2p2: bool = True
    hcn1: bool = True
    cdp: bool = True


@dataclass(frozen=True)
class SCConfig:
    toggles: SCToggles = SCToggles()
    temperature_celsius: float = 32.0
    v_init_mV: float = -65.0


@dataclass(frozen=True)
class RegionParameters:
    leak: float = 0.0
    nav1p1: float = 0.0
    nav1p6: float = 0.0
    cav2p1: float = 0.0
    cav3p2: float = 0.0
    cav3p3: float = 0.0
    cav3p3_g_scale: float = CAV3P3_G_SCALE
    kir2p3: float = 0.0
    kv1p1: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    km: float = 0.0
    kca1p1: float = 0.0
    kca2p2: float = 0.0
    hcn1: float = 0.0
    cdp_pump: float = CDP_PUMP_DEND


class SCParameters:
    def __init__(self):
        self.soma = RegionParameters(
            leak=0.00023,
            nav1p1=0.2,
            cav3p2=0.00163912063769,
            cav3p3=0.00001615552993,
            kir2p3=0.00001093425575,
            kv1p1=0.00107430134923,
            kv3p4=0.015,
            kv4p3=0.00404228168138,
            kca1p1=0.00518036298671,
            kca2p2=0.00054166094878,
            cav2p1=0.0005,
            hcn1=0.00058451678362,
            cdp_pump=CDP_PUMP_SOMA,
        )
        self.dendprox = RegionParameters(
            leak=0.000008,
            cav2p1=0.0008,
            cav3p2=0.00070661092763,
            cav3p3=0.00001526216781,
            kca1p1=0.00499205404769,
            kca2p2=0.00000326194117,
            kv1p1=0.00906810561650,
            kv4p3=0.00264204713540,
            cdp_pump=CDP_PUMP_DEND,
        )
        self.denddist = RegionParameters(
            leak=0.000008,
            cav2p1=0.00025,
            kca1p1=0.00226329455766,
            kca2p2=0.00001079984416,
            kv1p1=0.00237825442906,
            cdp_pump=CDP_PUMP_DEND,
        )
        self.axon_ais = RegionParameters(
            leak=0.000008,
            nav1p6=0.3,
            kv3p4=0.03351450571128,
            kv1p1=0.00492841685426,
            hcn1=0.00099184971498,
            km=0.00007960307413,
            cdp_pump=CDP_PUMP_AXON,
        )
        self.axon_regular = RegionParameters(
            leak=0.000008,
            nav1p6=0.00835931586458,
            kv3p4=0.01153520393521,
            kv1p1=0.00271359229578,
            hcn1=0.00070017344082,
            cdp_pump=CDP_PUMP_AXON,
        )

    def region(self, region_name: str) -> RegionParameters:
        return getattr(self, region_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "soma": self.soma.__dict__.copy(),
            "dendprox": self.dendprox.__dict__.copy(),
            "denddist": self.denddist.__dict__.copy(),
            "axon_ais": self.axon_ais.__dict__.copy(),
            "axon_regular": self.axon_regular.__dict__.copy(),
        }


def load_sc21_params() -> SCParameters:
    return SCParameters()


def sc21_nseg_rule(length_um: float) -> int:
    return 1 + 2 * int(float(length_um) / CV_MAX_LEN_UM)


def dend_region_name(dend_index: int) -> str:
    return "dendprox" if int(dend_index) in DENDPROX_INDICES else "denddist"


def axon_region_name(axon_index: int) -> str:
    return "axon_ais" if int(axon_index) == 0 else "axon_regular"


def toggle_names() -> list[str]:
    return [field.name for field in fields(SCToggles)]


def toggles_to_dict(toggles: SCToggles) -> dict[str, bool]:
    return {name: bool(getattr(toggles, name)) for name in toggle_names()}
