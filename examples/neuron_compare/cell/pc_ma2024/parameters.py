from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PCChannelRegionParameters:
    nav1p6: float = 0.0
    kv1p1: float = 0.0
    kv1p5: float = 0.0
    kv3p3: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    kir2p3: float = 0.0
    cav2p1_perm: float = 0.0
    cav3p1_perm: float = 0.0
    cav3p2: float = 0.0
    cav3p3_perm: float = 0.0
    kca1p1: float = 0.0
    kca2p2: float = 0.0
    kca3p1: float = 0.0
    hcn1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PCChannelParameters:
    indiv: int
    soma: PCChannelRegionParameters
    dend: PCChannelRegionParameters
    cav3p3_g_scale: float = 1.0e-5

    @classmethod
    def from_population_row(cls, row: np.ndarray, *, indiv: int) -> PCChannelParameters:
        # Column indices follow the original R_01_final_pop.txt / PC24 script layout.
        dend = PCChannelRegionParameters(
            nav1p6=float(row[0]),
            kv1p1=float(row[1]),
            kv1p5=float(row[2]),
            kv3p3=float(row[3]),
            kv4p3=float(row[4]),
            kir2p3=float(row[5]),
            cav2p1_perm=float(row[6]) * 6.0,  # Original dendritic Cav2.1 scale.
            cav3p1_perm=float(row[7]),
            cav3p2=float(row[8]),
            cav3p3_perm=float(row[9]),
            kca1p1=float(row[10]),
            kca2p2=float(row[11]),
            kca3p1=float(row[12]),
            hcn1=float(row[13]),
        )
        soma = PCChannelRegionParameters(
            nav1p6=float(row[14]),
            kv1p1=float(row[15]),
            kv3p4=float(row[16]),
            kir2p3=float(row[17]),
            cav2p1_perm=float(row[18]),
            cav3p1_perm=float(row[19]),
            cav3p2=float(row[20]),
            cav3p3_perm=float(row[21]),
            kca1p1=float(row[22]),
            kca2p2=float(row[23]),
            kca3p1=float(row[24]),
            hcn1=float(row[25]),
            kv1p5=float(row[35]),  # Soma Kv1.5 is stored out of the contiguous soma block.
        )
        return cls(indiv=int(indiv), soma=soma, dend=dend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indiv": self.indiv,
            "soma": self.soma.to_dict(),
            "dend": self.dend.to_dict(),
            "cav3p3_g_scale": self.cav3p3_g_scale,
        }


@dataclass(frozen=True)
class PCCableParameters:
    ra_ohm_cm: float = RA_OHM_CM
    leak_e_mV: float = LEAK_E_MV
    soma_cm_uF_cm2: float = SOMA_CM_UF_CM2
    leak_g_soma_mS_cm2: float = LEAK_G_SOMA_MS_CM2
    leak_g_dend_mS_cm2: float = LEAK_G_DEND_MS_CM2
    thick_dend_diam_um: float = THICK_DEND_DIAM_UM
    nav_dend_diam_um: float = NAV_DEND_DIAM_UM
    cv_max_len_um: float = CV_MAX_LEN_UM

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PCIonParameters:
    na_e_mV: float = NA_E_MV
    k_e_mV: float = K_E_MV
    h_e_mV: float = H_E_MV
    ca_e_mV: float = CA_E_MV
    cdp_pump_soma: float = CDP_PUMP_SOMA
    cdp_pump_dend: float = CDP_PUMP_DEND

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PCParameters:
    channel: PCChannelParameters
    cable: PCCableParameters = PCCableParameters()
    ion: PCIonParameters = PCIonParameters()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.to_dict(),
            "cable": self.cable.to_dict(),
            "ion": self.ion.to_dict(),
        }


def load_pc24_params(
    population_path: Path | str = DEFAULT_POPULATION_PATH,
    *,
    indiv: int = DEFAULT_INDIV,
) -> PCParameters:
    data = np.genfromtxt(Path(population_path))
    return PCParameters(channel=PCChannelParameters.from_population_row(data[int(indiv)], indiv=int(indiv)))


def pc24_nseg_rule(length_um: float, *, max_len_um: float = CV_MAX_LEN_UM) -> int:
    return 1 + 2 * int(float(length_um) / float(max_len_um))
