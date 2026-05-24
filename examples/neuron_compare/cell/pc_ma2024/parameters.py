from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CELL_DIR = Path(__file__).resolve().parent
DEFAULT_POPULATION_PATH = CELL_DIR / "R_01_final_pop.txt"
DEFAULT_MORPH_PATH = CELL_DIR.parent.parent / "Cerebellum_mod" / "PC" / "morphology" / "PC.asc"
DEFAULT_NRNMECH_PATH = CELL_DIR.parent.parent / "Cerebellum_mod" / "PC" / "x86_64" / ".libs" / "libnrnmech.so"

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
        self.cav21_dend_perm = float(row[6]) * 6.0  # Original dendritic Cav2.1 scale.
        self.cav31_dend_perm = float(row[7])
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
        self.cav21_soma_perm = float(row[18])
        self.cav31_soma_perm = float(row[19])
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
