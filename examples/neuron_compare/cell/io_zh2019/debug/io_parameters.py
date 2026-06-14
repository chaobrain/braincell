from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CELL_DIR = Path(__file__).resolve().parent
IO_SOURCE_DIR = CELL_DIR.parent.parent.parent / "Cerebellum_mod" / "IO"
SOURCE_README_PATH = IO_SOURCE_DIR / "README.md"
SOURCE_CHANNEL_DIR = IO_SOURCE_DIR / "channel"
SOURCE_OTHER_DIR = IO_SOURCE_DIR / "other"
SOURCE_SWC_PATH = IO_SOURCE_DIR / "morphology" / "IO.swc"
DEFAULT_NRNMECH_PATH = IO_SOURCE_DIR / "x86_64" / ".libs" / "libnrnmech.so"

SOMA_LENGTH_UM = 20.0
SOMA_DIAM_UM = 20.0
SOMA_NSEG = 1
RA_OHM_CM = 100.0
SOMA_CM_UF_CM2 = 1.0

V_INIT_MV = -65.0
TEMPERATURE_CELSIUS = 36.0
LEAK_E_MV = -65.0
LEAK_G_S_CM2 = 1.0e-5
NA_E_MV = 55.0
K_E_MV = -75.0

NA_GBAR_MS_CM2 = 70.0
KDR_GBAR_MS_CM2 = 18.0
CA_GBAR_MS_CM2 = 0.4
CA_E_MV = 120.0
CA_M_MID_MV = -61.0
HCN_GBAR_MS_CM2 = 0.15
HCN_E_MV = -43.0


@dataclass(frozen=True)
class IOToggles:
    leak: bool = True
    na: bool = True
    kdr: bool = True
    ca: bool = True
    hcn: bool = True


@dataclass(frozen=True)
class IOConfig:
    toggles: IOToggles = IOToggles()
    temperature_celsius: float = TEMPERATURE_CELSIUS
    v_init_mV: float = V_INIT_MV


@dataclass(frozen=True)
class IOSomaGeometry:
    length_um: float = SOMA_LENGTH_UM
    diam_um: float = SOMA_DIAM_UM
    nseg: int = SOMA_NSEG
    ra_ohm_cm: float = RA_OHM_CM
    cm_uF_cm2: float = SOMA_CM_UF_CM2


@dataclass(frozen=True)
class IOParameters:
    soma: IOSomaGeometry = IOSomaGeometry()
    leak_g_S_cm2: float = LEAK_G_S_CM2
    leak_e_mV: float = LEAK_E_MV
    na_gbar_mS_cm2: float = NA_GBAR_MS_CM2
    na_e_mV: float = NA_E_MV
    kdr_gbar_mS_cm2: float = KDR_GBAR_MS_CM2
    k_e_mV: float = K_E_MV
    ca_gbar_mS_cm2: float = CA_GBAR_MS_CM2
    ca_e_mV: float = CA_E_MV
    ca_m_mid_mV: float = CA_M_MID_MV
    hcn_gbar_mS_cm2: float = HCN_GBAR_MS_CM2
    hcn_e_mV: float = HCN_E_MV

    def to_dict(self) -> dict[str, Any]:
        return {
            "soma": self.soma.__dict__.copy(),
            **{key: value for key, value in self.__dict__.items() if key != "soma"},
        }


ALL_REGION_LOGICAL_MECHANISMS = {
    "soma": ("leak", "na", "kdr", "ca", "hcn"),
}


def load_io19_params() -> IOParameters:
    return IOParameters()


def toggle_names() -> list[str]:
    return [field.name for field in fields(IOToggles)]


def toggles_to_dict(toggles: IOToggles) -> dict[str, bool]:
    return {name: bool(getattr(toggles, name)) for name in toggle_names()}


def enabled_region_list(config: IOConfig, region: str = "soma") -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]
