from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CELL_DIR.parents[3]

SOURCE_GRC_DIR = Path("/home/swl/Cerebellum_circuit/GrC/GrC")
SOURCE_MORPH_PATH = SOURCE_GRC_DIR / "GrC2020.asc"

DEFAULT_MORPH_PATH = REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "GrC" / "morphology" / "GrC.asc"
DEFAULT_NRNMECH_PATH = (
    REPO_ROOT / "examples" / "neuron_compare" / "Cerebellum_mod" / "GrC" / "x86_64" / ".libs" / "libnrnmech.so"
)

RA_OHM_CM = 100.0
LEAK_E_MV = -60.0
NA_E_MV = 87.39
K_E_MV = -88.0
CA_E_MV = 137.5
CV_MAX_LEN_UM = 40.0
CA_CO_M_M = 2.0
CA_CI_INITIALIZER_M_M = 45e-6

EXPECTED_SOMA_COUNT = 1
EXPECTED_DEND_COUNT = 4
EXPECTED_AXON_COUNT = 0

REGION_NAMES = ("soma", "dend")


@dataclass(frozen=True)
class RegionParameters:
    cm_uF_cm2: float
    leak: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    kir2p3: float = 0.0
    cahva: float = 0.0
    kv1p1: float = 0.0
    kv1p5: float = 0.0
    kv2p2: float = 0.0
    kca1p1: float = 0.0


class GrCParameters:
    def __init__(self):
        self.soma = RegionParameters(
            cm_uF_cm2=2.0,
            leak=0.00029038073716,
            kv3p4=0.00076192450951999995,
            kv4p3=0.0028149683906099998,
            kir2p3=0.00074725514701999996,
            cahva=0.00060938071783999998,
            kv1p1=0.0056973826455499997,
            kv1p5=0.00083407556713999999,
            kv2p2=1.203410852e-05,
        )
        self.dend = RegionParameters(
            cm_uF_cm2=2.5,
            leak=0.00025029700736999997,
            cahva=0.0050012800845900002,
            kca1p1=0.010018074546510001,
            kv1p1=0.00381819207934,
        )

    def region(self, region_name: str) -> RegionParameters:
        return getattr(self, region_name)

    def to_dict(self) -> dict[str, Any]:
        return {name: self.region(name).__dict__.copy() for name in REGION_NAMES}


def load_grc20_params() -> GrCParameters:
    return GrCParameters()


def grc20_nseg_rule(length_um: float, *, max_len_um: float = CV_MAX_LEN_UM) -> int:
    return 1 + 2 * int(float(length_um) / float(max_len_um))


_LOADED_NRNMECH_PATHS: set[str] = set()


def mark_nrnmech_loaded(path: Path) -> bool:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return True
    _LOADED_NRNMECH_PATHS.add(resolved)
    return False
