from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

CELL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CELL_DIR.parents[4]

SOURCE_GOC_DIR = Path("/home/swl/Cerebellum_circuit/GoC/GoC")
SOURCE_MORPH_PATH = SOURCE_GOC_DIR / "pair-140514-C2-1_split_1.asc"
SOURCE_OPTIMIZATION_PATH = SOURCE_GOC_DIR / "Optimization_result.txt"
SOURCE_NRNMECH_PATH = (
    REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "GoC" / "x86_64" / ".libs" / "libnrnmech.so"
)

DEFAULT_MORPH_PATH = REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "GoC" / "morphology" / "GoC.asc"

RA_OHM_CM = 122.0
LEAK_E_MV = -55.0
NA_E_MV = 60.0
K_E_MV = -80.0
CA_E_MV = 137.0
LEAK_G_DEFAULT_S_CM2 = 0.00003
LEAK_G_REGULAR_AXON_S_CM2 = 0.000001
SOMA_CM_UF_CM2 = 1.0
DEND_CM_UF_CM2 = 2.5
AXON_CM_UF_CM2 = 1.0
CV_MAX_LEN_UM = 40.0
CDP_PUMP_SOMA = 1e-7
CDP_PUMP_DEND_APICAL = 5e-9
CDP_PUMP_DEND_BASAL = 2e-9
CDP_PUMP_AXON = 1e-8
REGULAR_AXON_NAV = 0.0115
REGULAR_AXON_KV3P4 = 0.0091

DEND_BASAL_RANGES = ((0, 3), (16, 17), (33, 41), (84, 84), (105, 150))
DEND_APICAL_RANGES = ((4, 15), (18, 32), (42, 83), (85, 104))

ALL_REGION_LOGICAL_MECHANISMS = {
    "soma": (
        "leak",
        "kv1p1",
        "kv3p4",
        "kv4p3",
        "nav",
        "kca1p1",
        "kca3p1",
        "cahva",
        "cav3p1",
        "cdp",
    ),
    "dend_apical": (
        "leak",
        "nav",
        "kca1p1",
        "kca2p2",
        "cav2p3",
        "cav3p1",
        "cdp",
    ),
    "dend_basal": (
        "leak",
        "nav",
        "kca1p1",
        "kca2p2",
        "cahva",
        "cdp",
    ),
    "axon_ais": (
        "leak",
        "hcn1",
        "hcn2",
        "nav",
        "km",
        "kca1p1",
        "cahva",
        "cdp",
    ),
    "axon_regular": (
        "leak",
        "kv3p4",
        "nav",
        "cdp",
    ),
}


@dataclass(frozen=True)
class GoCToggles:
    leak: bool = True
    nav: bool = True
    kv1p1: bool = True
    kv3p4: bool = True
    kv4p3: bool = True
    km: bool = True
    kca1p1: bool = True
    kca2p2: bool = True
    kca3p1: bool = True
    cahva: bool = True
    cav2p3: bool = True
    cav3p1: bool = True
    hcn1: bool = True
    hcn2: bool = True
    cdp: bool = True


@dataclass(frozen=True)
class GoCConfig:
    toggles: GoCToggles = GoCToggles()
    temperature_celsius: float = 34.0
    v_init_mV: float = -65.0


class GoCParameters:
    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) < 23:
            raise ValueError(f"Expected at least 23 GoC conductance values, got {len(values)}.")
        self.values = values.copy()

        self.nav_dend_apical = float(values[0])
        self.kca1p1_dend_apical = float(values[1])
        self.kca2p2_dend_apical = float(values[2])
        self.cav2p3_dend_apical = float(values[3])
        self.cav3p1_dend_apical = float(values[4])

        self.nav_dend_basal = float(values[5])
        self.kca1p1_dend_basal = float(values[6])
        self.kca2p2_dend_basal = float(values[7])
        self.cahva_dend_basal = float(values[8])

        self.nav_soma = float(values[9])
        self.kv1p1_soma = float(values[10])
        self.kv3p4_soma = float(values[11])
        self.kv4p3_soma = float(values[12])
        self.kca1p1_soma = float(values[13])
        self.kca3p1_soma = float(values[14])
        self.cahva_soma = float(values[15])
        self.cav3p1_soma = float(values[16])

        self.hcn1_ais = float(values[17])
        self.hcn2_ais = float(values[18])
        self.nav_ais = float(values[19])
        self.km_ais = float(values[20])
        self.kca1p1_ais = float(values[21])
        self.cahva_ais = float(values[22])

        self.nav_axon_regular = float(values[23]) if len(values) > 23 else REGULAR_AXON_NAV
        self.kv3p4_axon_regular = float(values[24]) if len(values) > 24 else REGULAR_AXON_KV3P4

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "values"}


def load_goc20_params(path: Path | str = SOURCE_OPTIMIZATION_PATH) -> GoCParameters:
    return GoCParameters(np.genfromtxt(Path(path)))


def goc20_nseg_rule(length_um: float) -> int:
    return 1 + 2 * int(float(length_um) / CV_MAX_LEN_UM)


def _in_ranges(index: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= int(index) <= stop for start, stop in ranges)


def dend_region_name(dend_index: int) -> str:
    if _in_ranges(dend_index, DEND_BASAL_RANGES):
        return "dend_basal"
    if _in_ranges(dend_index, DEND_APICAL_RANGES):
        return "dend_apical"
    raise ValueError(f"GoC dend index {dend_index} is not covered by the source region map.")


def axon_region_name(axon_index: int) -> str:
    return "axon_ais" if int(axon_index) == 0 else "axon_regular"


def toggle_names() -> list[str]:
    return [field.name for field in fields(GoCToggles)]


def toggles_to_dict(toggles: GoCToggles) -> dict[str, bool]:
    return {name: bool(getattr(toggles, name)) for name in toggle_names()}
