from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CELL_DIR.parents[3]

DEFAULT_MORPH_PATH = REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "BC" / "morphology" / "BC.asc"
DEFAULT_NRNMECH_PATH = (
    REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "BC" / "x86_64" / ".libs" / "libnrnmech.so"
)

RA_OHM_CM = 122.0
LEAK_E_MV = -55.0
NA_E_MV = 60.0
K_E_MV = -80.0
H_E_MV = -34.0
CA_E_MV = 137.5
CM_UF_CM2 = 1.0
CV_MAX_LEN_UM = 40.0
CDP_PUMP = 2e-9

EXPECTED_SOMA_COUNT = 1
EXPECTED_DEND_COUNT = 42
EXPECTED_AXON_COUNT = 71


@dataclass(frozen=True)
class RegionParameters:
    leak: float = 0.0
    nav1p1: float = 0.0
    nav1p6: float = 0.0
    cav1p2: float = 0.0
    cav1p3: float = 0.0
    cav2p1: float = 0.0
    cav3p2: float = 0.0
    kir2p3: float = 0.0
    kv1p1: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    kca1p1: float = 0.0
    kca2p2: float = 0.0
    kca3p1: float = 0.0
    hcn1: float = 0.0
    cdp_pump: float = CDP_PUMP


class BCParameters:
    def __init__(self):
        self.soma = RegionParameters(
            leak=0.00004,
            nav1p1=0.2,
            cav3p2=0.0001,
            cav1p2=0.0007,
            cav1p3=0.000005,
            kir2p3=0.0001,
            kv3p4=0.097,
            kv4p3=0.01,
            kca3p1=0.001,
            hcn1=0.001,
        )
        self.dend = RegionParameters(
            leak=0.00001,
            cav3p2=0.00005,
            cav1p2=0.0002,
            cav1p3=0.000005,
            kv4p3=0.00987201764943,
            kca2p2=0.0065,
        )
        self.axon_ais = RegionParameters(
            leak=0.00001,
            nav1p6=0.3,
            kv3p4=0.002,
            hcn1=0.001,
            kca1p1=0.01,
            cav2p1=2.2e-4,
        )
        self.axon_regular = RegionParameters(
            leak=0.000001,
            nav1p6=0.001,
            kv3p4=0.001,
            kv1p1=0.0005,
            hcn1=0.0001,
            kca1p1=0.001,
            cav2p1=0.00008,
        )

    def region(self, region_name: str) -> RegionParameters:
        return getattr(self, region_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "soma": self.soma.__dict__.copy(),
            "dend": self.dend.__dict__.copy(),
            "axon_ais": self.axon_ais.__dict__.copy(),
            "axon_regular": self.axon_regular.__dict__.copy(),
        }


def load_bc25_params() -> BCParameters:
    return BCParameters()


def bc25_nseg_rule(length_um: float) -> int:
    return 1 + 2 * int(float(length_um) / CV_MAX_LEN_UM)


def axon_region_name(axon_index: int) -> str:
    return "axon_ais" if int(axon_index) == 0 else "axon_regular"
