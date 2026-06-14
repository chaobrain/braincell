from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameters import (
    CA_CI_INITIALIZER_M_M,
    CA_CO_M_M,
    CA_E_MV,
    CV_MAX_LEN_UM,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    SOURCE_GRC_DIR,
    SOURCE_MORPH_PATH,
    grc20_nseg_rule,
    mark_nrnmech_loaded,
)

EXPECTED_FULL_SOMA_COUNT = 1
EXPECTED_FULL_DEND_COUNT = 4
EXPECTED_FULL_HILOCK_COUNT = 1
EXPECTED_FULL_AIS_COUNT = 1
EXPECTED_FULL_AA_COUNT = 4
EXPECTED_FULL_PF1_COUNT = 142
EXPECTED_FULL_PF2_COUNT = 142
EXPECTED_FULL_AXON_COUNT = (
    EXPECTED_FULL_HILOCK_COUNT
    + EXPECTED_FULL_AIS_COUNT
    + EXPECTED_FULL_AA_COUNT
    + EXPECTED_FULL_PF1_COUNT
    + EXPECTED_FULL_PF2_COUNT
)

AA_SECTION_LEN_UM = 7.0
PF_SECTION_LEN_UM = 7.0
PF_SECTION_COUNT = EXPECTED_FULL_PF1_COUNT

FULL_REGION_NAMES = ("soma", "dend", "hilock", "ais", "aa", "pf")


@dataclass(frozen=True)
class FullRegionParameters:
    cm_uF_cm2: float
    leak: float = 0.0
    nav: float = 0.0
    nafhhf: float = 0.0
    kv3p4: float = 0.0
    kv4p3: float = 0.0
    kir2p3: float = 0.0
    cahva: float = 0.0
    kv1p1: float = 0.0
    kv1p5: float = 0.0
    kv2p2: float = 0.0
    kca1p1: float = 0.0
    km: float = 0.0


class GrCFullParameters:
    def __init__(self):
        self.soma = FullRegionParameters(
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
        self.dend = FullRegionParameters(
            cm_uF_cm2=2.5,
            leak=0.00025029700736999997,
            cahva=0.0050012800845900002,
            kca1p1=0.010018074546510001,
            kv1p1=0.00381819207934,
        )
        self.hilock = FullRegionParameters(
            cm_uF_cm2=2.0,
            leak=0.00036958189720000001,
            nafhhf=0.0092880585146199995,
            kv3p4=0.020373463109149999,
            cahva=0.00057726155447,
        )
        self.ais = FullRegionParameters(
            cm_uF_cm2=1.0,
            leak=0.00029276697557000002,
            nafhhf=1.28725006737226,
            kv3p4=0.0064959534065400001,
            cahva=0.00031198539471999999,
            km=0.00056671971737000002,
        )
        self.aa = FullRegionParameters(
            cm_uF_cm2=1.0,
            leak=9.3640921249999996e-05,
            nav=0.026301636815019999,
            kv3p4=0.00237386061632,
            cahva=0.00068197420273000001,
        )
        self.pf = FullRegionParameters(
            cm_uF_cm2=1.0,
            leak=3.5301616000000001e-07,
            nav=0.017718484492610001,
            kv3p4=0.0081756804703699993,
            cahva=0.00020856833529999999,
        )

    def region(self, region_name: str) -> FullRegionParameters:
        return getattr(self, region_name)

    def to_dict(self) -> dict[str, Any]:
        return {name: self.region(name).__dict__.copy() for name in FULL_REGION_NAMES}


def load_grc20_full_params() -> GrCFullParameters:
    return GrCFullParameters()
