from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

CELL_DIR = Path(__file__).resolve().parent
DEFAULT_POPULATION_PATH = CELL_DIR / "R_01_final_pop.txt"
DEFAULT_MORPH_PATH = CELL_DIR.parent.parent.parent / "Cerebellum_mod" / "PC" / "morphology" / "PC.asc"
DEFAULT_NRNMECH_PATH = CELL_DIR.parent.parent.parent / "Cerebellum_mod" / "PC" / "x86_64" / ".libs" / "libnrnmech.so"

DEFAULT_INDIV = 138

RA_OHM_CM = 122.0
LEAK_E_MV = -61.0
NA_E_MV = 60.0
K_E_MV = -88.0
H_E_MV = -34.4
CA_E_MV = 137.52625
SOMA_CM_UF_CM2 = 2.0
LEAK_G_SOMA_MS_CM2 = 1.0
LEAK_G_DEND_MS_CM2 = 0.3
CDP_PUMP_SOMA = 5.0e-8
CDP_PUMP_DEND = 6.0e-8
THICK_DEND_DIAM_UM = 1.6
NAV_DEND_DIAM_UM = 3.3
CV_MAX_LEN_UM = 40.0


@dataclass(frozen=True)
class PCToggles:
    leak: bool = True
    nav: bool = True
    kv1p1: bool = True
    kv1p5: bool = True
    kv3p3: bool = True
    kv3p4: bool = True
    kv4p3: bool = True
    kir2p3: bool = True
    kca1p1: bool = True
    kca2p2: bool = True
    kca3p1: bool = True
    cav21: bool = True
    cav31: bool = True
    cav32: bool = True
    cav33: bool = True
    hcn1: bool = True
    cdp: bool = True


@dataclass(frozen=True)
class PCConfig:
    toggles: PCToggles = PCToggles()
    temperature_celsius: float = 32.0
    v_init_mV: float = -65.0


class PCParameters:
    def __init__(self, row: np.ndarray, *, indiv: int):
        self.indiv = int(indiv)

        # Column indices follow the original R_01_final_pop.txt / PC24 script layout.
        self.nav_dend = float(row[0])
        self.kv1p1_dend = float(row[1])
        self.kv1p5_dend = float(row[2])
        self.kv3p3_dend = float(row[3])
        self.kv4p3_dend = float(row[4])
        self.kir2p3_dend = float(row[5])
        self.cav21_dend = float(row[6]) * 6.0  # Original dendritic Cav2.1 scale.
        self.cav31_dend = float(row[7])
        self.cav32_dend = float(row[8])
        self.cav33_dend_perm = float(row[9])
        self.kca1p1_dend = float(row[10])
        self.kca2p2_dend = float(row[11])
        self.kca3p1_dend = float(row[12])
        self.hcn1_dend = float(row[13])

        self.nav_soma = float(row[14])
        self.kv1p1_soma = float(row[15])
        self.kv3p4_soma = float(row[16])
        self.kir2p3_soma = float(row[17])
        self.cav21_soma = float(row[18])
        self.cav31_soma = float(row[19])
        self.cav32_soma = float(row[20])
        self.cav33_soma_perm = float(row[21])
        self.kca1p1_soma = float(row[22])
        self.kca2p2_soma = float(row[23])
        self.kca3p1_soma = float(row[24])
        self.hcn1_soma = float(row[25])
        self.kv1p5_soma = float(row[35])  # Soma Kv1.5 is stored out of the contiguous soma block.
        self.cav33_g_scale = 1.0e-5

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_pc24_params(
    population_path: Path | str = DEFAULT_POPULATION_PATH,
    *,
    indiv: int = DEFAULT_INDIV,
) -> PCParameters:
    data = np.genfromtxt(Path(population_path))
    return PCParameters(data[int(indiv)], indiv=int(indiv))


def pc24_dend_cm(diam_arc_mean_um: float) -> float:
    diam = float(diam_arc_mean_um)
    if diam >= THICK_DEND_DIAM_UM:
        return SOMA_CM_UF_CM2
    return 11.510294 * math.exp(-1.376463 * diam) + 2.120503


def pc24_nseg_rule(length_um: float) -> int:
    return 1 + 2 * int(float(length_um) / CV_MAX_LEN_UM)


def toggle_names() -> list[str]:
    return [field.name for field in fields(PCToggles)]


def toggles_to_dict(toggles: PCToggles) -> dict[str, bool]:
    return {name: bool(getattr(toggles, name)) for name in toggle_names()}


ALL_REGION_LOGICAL_MECHANISMS = {
    "soma": (
        "leak",
        "nav",
        "kv1p1",
        "kv1p5",
        "kv3p4",
        "kir2p3",
        "cav21",
        "cav31",
        "cav32",
        "cav33",
        "kca1p1",
        "kca2p2",
        "kca3p1",
        "hcn1",
        "cdp",
    ),
    "dend_all": (
        "leak",
        "kv3p3",
        "kv4p3",
        "cav21",
        "cav33",
        "kca1p1",
        "kca2p2",
        "hcn1",
        "cdp",
    ),
    "dend_thick": (
        "kv1p1",
        "kv1p5",
        "kir2p3",
        "cav31",
        "cav32",
        "kca3p1",
    ),
    "dend_nav": ("nav",),
}
