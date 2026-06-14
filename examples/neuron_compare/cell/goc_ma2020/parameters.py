from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CELL_DIR = Path(__file__).resolve().parent

DEFAULT_OPTIMIZATION_PATH = CELL_DIR / "Optimization_result.txt"
DEFAULT_MORPH_PATH = CELL_DIR.parent.parent / "Cerebellum_mod" / "GoC" / "morphology" / "GoC.asc"
DEFAULT_NRNMECH_PATH = CELL_DIR.parent.parent / "Cerebellum_mod" / "GoC" / "x86_64" / ".libs" / "libnrnmech.so"

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


@dataclass(frozen=True)
class GoCChannelRegionParameters:
    nav: float = 0.0
    kv1p1: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    km: float = 0.0
    kca1p1: float = 0.0
    kca2p2: float = 0.0
    kca3p1: float = 0.0
    cahva: float = 0.0
    cav2p3: float = 0.0
    cav3p1: float = 0.0
    hcn1: float = 0.0
    hcn2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GoCChannelParameters:
    soma: GoCChannelRegionParameters
    dend_apical: GoCChannelRegionParameters
    dend_basal: GoCChannelRegionParameters
    axon_ais: GoCChannelRegionParameters
    axon_regular: GoCChannelRegionParameters

    @classmethod
    def from_optimization_values(cls, values: np.ndarray) -> GoCChannelParameters:
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) < 23:
            raise ValueError(f"Expected at least 23 GoC conductance values, got {len(values)}.")
        return cls(
            dend_apical=GoCChannelRegionParameters(
                nav=float(values[0]),
                kca1p1=float(values[1]),
                kca2p2=float(values[2]),
                cav2p3=float(values[3]),
                cav3p1=float(values[4]),
            ),
            dend_basal=GoCChannelRegionParameters(
                nav=float(values[5]),
                kca1p1=float(values[6]),
                kca2p2=float(values[7]),
                cahva=float(values[8]),
            ),
            soma=GoCChannelRegionParameters(
                nav=float(values[9]),
                kv1p1=float(values[10]),
                kv3p4=float(values[11]),
                kv4p3=float(values[12]),
                kca1p1=float(values[13]),
                kca3p1=float(values[14]),
                cahva=float(values[15]),
                cav3p1=float(values[16]),
            ),
            axon_ais=GoCChannelRegionParameters(
                hcn1=float(values[17]),
                hcn2=float(values[18]),
                nav=float(values[19]),
                km=float(values[20]),
                kca1p1=float(values[21]),
                cahva=float(values[22]),
            ),
            axon_regular=GoCChannelRegionParameters(
                nav=float(values[23]) if len(values) > 23 else REGULAR_AXON_NAV,
                kv3p4=float(values[24]) if len(values) > 24 else REGULAR_AXON_KV3P4,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "soma": self.soma.to_dict(),
            "dend_apical": self.dend_apical.to_dict(),
            "dend_basal": self.dend_basal.to_dict(),
            "axon_ais": self.axon_ais.to_dict(),
            "axon_regular": self.axon_regular.to_dict(),
        }


@dataclass(frozen=True)
class GoCCableParameters:
    ra_ohm_cm: float = RA_OHM_CM
    leak_e_mV: float = LEAK_E_MV
    leak_g_default_S_cm2: float = LEAK_G_DEFAULT_S_CM2
    leak_g_regular_axon_S_cm2: float = LEAK_G_REGULAR_AXON_S_CM2
    soma_cm_uF_cm2: float = SOMA_CM_UF_CM2
    dend_cm_uF_cm2: float = DEND_CM_UF_CM2
    axon_cm_uF_cm2: float = AXON_CM_UF_CM2
    cv_max_len_um: float = CV_MAX_LEN_UM

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GoCIonParameters:
    na_e_mV: float = NA_E_MV
    k_e_mV: float = K_E_MV
    ca_e_mV: float = CA_E_MV
    cdp_pump_soma: float = CDP_PUMP_SOMA
    cdp_pump_dend_apical: float = CDP_PUMP_DEND_APICAL
    cdp_pump_dend_basal: float = CDP_PUMP_DEND_BASAL
    cdp_pump_axon: float = CDP_PUMP_AXON

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GoCParameters:
    channel: GoCChannelParameters
    cable: GoCCableParameters = GoCCableParameters()
    ion: GoCIonParameters = GoCIonParameters()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.to_dict(),
            "cable": self.cable.to_dict(),
            "ion": self.ion.to_dict(),
        }


def load_goc20_params(path: Path | str = DEFAULT_OPTIMIZATION_PATH) -> GoCParameters:
    return GoCParameters(channel=GoCChannelParameters.from_optimization_values(np.genfromtxt(Path(path))))


def goc20_nseg_rule(length_um: float, *, max_len_um: float = CV_MAX_LEN_UM) -> int:
    return 1 + 2 * int(float(length_um) / float(max_len_um))


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
