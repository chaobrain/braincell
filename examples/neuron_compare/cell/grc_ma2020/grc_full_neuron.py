from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .grc_full_parameters import (
    AA_SECTION_LEN_UM,
    CA_E_MV,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_FULL_AA_COUNT,
    EXPECTED_FULL_AIS_COUNT,
    EXPECTED_FULL_DEND_COUNT,
    EXPECTED_FULL_HILOCK_COUNT,
    EXPECTED_FULL_PF1_COUNT,
    EXPECTED_FULL_PF2_COUNT,
    EXPECTED_FULL_SOMA_COUNT,
    GrCFullParameters,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    PF_SECTION_COUNT,
    PF_SECTION_LEN_UM,
    RA_OHM_CM,
    grc20_nseg_rule,
    mark_nrnmech_loaded,
)


class GrCFull:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCFullParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
        temperature_celsius: float = 25.0,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.temperature_celsius = float(temperature_celsius)
        self.sections: tuple[Any, ...] = ()
        self.soma_sections: tuple[Any, ...] = ()
        self.dend_sections: tuple[Any, ...] = ()
        self.hilock_section: Any | None = None
        self.ais_section: Any | None = None
        self.aa_sections: tuple[Any, ...] = ()
        self.pf1_sections: tuple[Any, ...] = ()
        self.pf2_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None
        self._section_region: dict[Any, str] = {}

    def build(self) -> GrCFull:
        self._load_support()
        self._load_asc_sections()
        self._create_manual_sections()
        self._configure_sections()
        return self

    def cleanup(self) -> None:
        for sec in self.sections:
            try:
                h.delete_section(sec=sec)
            except Exception:
                pass
        self.sections = ()
        self.soma_sections = ()
        self.dend_sections = ()
        self.hilock_section = None
        self.ais_section = None
        self.aa_sections = ()
        self.pf1_sections = ()
        self.pf2_sections = ()
        self.root_soma = None
        self._section_region.clear()

    def _load_support(self) -> None:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
        h.celsius = self.temperature_celsius
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

    def _load_asc_sections(self) -> None:
        existing_count = sum(1 for _ in h.allsec())
        reader = h.Import3d_Neurolucida3()
        reader.input(str(self.morph_path))
        h.Import3d_GUI(reader, 0).instantiate(None)
        imported = tuple(h.allsec())[existing_count:]
        if not imported:
            raise RuntimeError(f"NEURON import3d instantiated no sections from {self.morph_path!s}.")

        by_prefix: dict[str, list[Any]] = defaultdict(list)
        for sec in imported:
            by_prefix[_infer_section_prefix(sec.name())].append(sec)
        self.soma_sections = tuple(by_prefix["soma"])
        self.dend_sections = tuple(by_prefix["dend"])
        if len(self.soma_sections) != EXPECTED_FULL_SOMA_COUNT or len(self.dend_sections) != EXPECTED_FULL_DEND_COUNT:
            raise RuntimeError(
                "Unexpected GrC ASC base counts: "
                f"soma={len(self.soma_sections)}, dend={len(self.dend_sections)}."
            )
        self.root_soma = self.soma_sections[0]
        self.sections = imported
        for sec in self.soma_sections:
            self._section_region[sec] = "soma"
        for sec in self.dend_sections:
            self._section_region[sec] = "dend"

    def _create_manual_sections(self) -> None:
        if self.root_soma is None:
            raise RuntimeError("ASC soma must load before manual sections.")
        self.hilock_section = h.Section(name="hilock")
        _set_pt3d_line(self.hilock_section, (0.0, -5.62232, 0.0, 1.5), (0.0, -6.62232, 0.0, 1.5))
        self.hilock_section.L = 1.0
        self.hilock_section.diam = 1.5
        self.hilock_section.connect(self.root_soma, 0, 0)

        self.ais_section = h.Section(name="ais")
        _set_pt3d_line(self.ais_section, (0.0, -6.62232, 0.0, 0.7), (0.0, -16.62232, 0.0, 0.7))
        self.ais_section.L = 10.0
        self.ais_section.diam = 0.7
        self.ais_section.connect(self.hilock_section, 1, 0)

        aa_sections: list[Any] = []
        len_initial_ais = -16.62232
        for index in range(EXPECTED_FULL_AA_COUNT):
            sec = h.Section(name=f"aa_{index}")
            _set_pt3d_line(sec, (0.0, len_initial_ais, 0.0, 0.3), (0.0, len_initial_ais - AA_SECTION_LEN_UM, 0.0, 0.3))
            sec.L = AA_SECTION_LEN_UM
            sec.diam = 0.3
            aa_sections.append(sec)
            len_initial_ais -= AA_SECTION_LEN_UM
        aa_sections[0].connect(self.ais_section, 1, 0)
        for index in range(EXPECTED_FULL_AA_COUNT - 1):
            aa_sections[index + 1].connect(aa_sections[index], 1, 0)

        pf1_sections: list[Any] = []
        len_initial_aa = -142.62232
        for index in range(PF_SECTION_COUNT):
            sec = h.Section(name=f"pf1_{index}")
            _set_pt3d_line(
                sec,
                (len_initial_aa, len_initial_aa, 0.0, 0.15),
                (len_initial_aa + PF_SECTION_LEN_UM, len_initial_aa, 0.0, 0.15),
            )
            sec.L = PF_SECTION_LEN_UM
            sec.diam = 0.15
            pf1_sections.append(sec)
            len_initial_aa += PF_SECTION_LEN_UM

        pf2_sections: list[Any] = []
        for index in range(PF_SECTION_COUNT):
            sec = h.Section(name=f"pf2_{index}")
            _set_pt3d_line(
                sec,
                (len_initial_aa, len_initial_aa, 0.0, 0.15),
                (len_initial_aa - PF_SECTION_LEN_UM, len_initial_aa, 0.0, 0.15),
            )
            sec.L = PF_SECTION_LEN_UM
            sec.diam = 0.15
            pf2_sections.append(sec)
            len_initial_aa -= PF_SECTION_LEN_UM

        pf1_sections[0].connect(aa_sections[-1], 1, 0)
        pf2_sections[0].connect(aa_sections[-1], 1, 0)
        for index in range(PF_SECTION_COUNT - 1):
            pf1_sections[index + 1].connect(pf1_sections[index], 1, 0)
            pf2_sections[index + 1].connect(pf2_sections[index], 1, 0)

        self.aa_sections = tuple(aa_sections)
        self.pf1_sections = tuple(pf1_sections)
        self.pf2_sections = tuple(pf2_sections)
        manual_sections = (self.hilock_section, self.ais_section, *self.aa_sections, *self.pf1_sections, *self.pf2_sections)
        self.sections = (*self.sections, *manual_sections)
        for sec, region in ((self.hilock_section, "hilock"), (self.ais_section, "ais")):
            self._section_region[sec] = region
        for sec in self.aa_sections:
            self._section_region[sec] = "aa"
        for sec in self.pf1_sections:
            self._section_region[sec] = "pf"
        for sec in self.pf2_sections:
            self._section_region[sec] = "pf"

    def _configure_sections(self) -> None:
        for sec in self.sections:
            region = self._section_region[sec]
            if region == "soma":
                self._configure_soma(sec)
            elif region == "dend":
                self._configure_dend(sec)
            elif region == "hilock":
                self._configure_hilock(sec)
            elif region == "ais":
                self._configure_ais(sec)
            elif region == "aa":
                self._configure_aa(sec)
            elif region == "pf":
                self._configure_pf(sec)
            else:
                raise ValueError(f"Unsupported full GrC region {region!r}.")

    def _configure_soma(self, sec: Any) -> None:
        p = self.params.soma
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_kv3p4(sec, p.kv3p4)
        sec.insert("Kv4p3_MA20_GrC")
        sec.gkbar_Kv4p3_MA20_GrC = p.kv4p3
        sec.insert("Kir2p3_MA20_GrC")
        sec.gkbar_Kir2p3_MA20_GrC = p.kir2p3
        _insert_cahva(sec, p.cahva)
        sec.insert("Kv1p1_MA20_GrC")
        sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        sec.insert("Kv1p5_MA20_GrC")
        sec.gKur_Kv1p5_MA20_GrC = p.kv1p5
        sec.insert("Kv2p2_0010_MA20_GrC")
        sec.gKv2_2bar_Kv2p2_0010_MA20_GrC = p.kv2p2
        _insert_cdp(sec)
        _set_k_ca(sec)

    def _configure_dend(self, sec: Any) -> None:
        p = self.params.dend
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_cahva(sec, p.cahva)
        sec.insert("Kca1p1_MA20_GrC")
        sec.gbar_Kca1p1_MA20_GrC = p.kca1p1
        sec.insert("Kv1p1_MA20_GrC")
        sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        _insert_cdp(sec)
        _set_k_ca(sec)

    def _configure_hilock(self, sec: Any) -> None:
        p = self.params.hilock
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_nafhhf(sec, p.nafhhf)
        _insert_kv3p4(sec, p.kv3p4)
        _insert_cahva(sec, p.cahva)
        _insert_cdp(sec)
        _set_na_k_ca(sec)

    def _configure_ais(self, sec: Any) -> None:
        p = self.params.ais
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_nafhhf(sec, p.nafhhf)
        _insert_kv3p4(sec, p.kv3p4)
        _insert_cahva(sec, p.cahva)
        sec.insert("KM_MA20_GrC")
        sec.gkbar_KM_MA20_GrC = p.km
        _insert_cdp(sec)
        _set_na_k_ca(sec)

    def _configure_aa(self, sec: Any) -> None:
        p = self.params.aa
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_nav(sec, p.nav)
        _insert_kv3p4(sec, p.kv3p4)
        _insert_cahva(sec, p.cahva)
        _insert_cdp(sec)
        _set_na_k_ca(sec)

    def _configure_pf(self, sec: Any) -> None:
        p = self.params.pf
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        _insert_nav(sec, p.nav)
        _insert_kv3p4(sec, p.kv3p4)
        _insert_cahva(sec, p.cahva)
        _insert_cdp(sec)
        _set_na_k_ca(sec)


def _set_cable(sec: Any, cm_uF_cm2: float) -> None:
    sec.nseg = grc20_nseg_rule(float(sec.L))
    sec.Ra = RA_OHM_CM
    sec.cm = cm_uF_cm2


def _set_pt3d_line(sec: Any, prox: tuple[float, float, float, float], dist: tuple[float, float, float, float]) -> None:
    sec.push()
    h.pt3dclear()
    h.pt3dadd(*prox)
    h.pt3dadd(*dist)
    h.pop_section()


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = g_max
    sec.e_pas = LEAK_E_MV


def _insert_kv3p4(sec: Any, g_max: float) -> None:
    sec.insert("Kv3p4_MA20_GrC")
    sec.gkbar_Kv3p4_MA20_GrC = g_max


def _insert_cahva(sec: Any, g_max: float) -> None:
    sec.insert("CaHVA_MA20_GrC")
    sec.gcabar_CaHVA_MA20_GrC = g_max


def _insert_nafhhf(sec: Any, g_max: float) -> None:
    sec.insert("NaFHF_MA20_GrC")
    sec.gnabar_NaFHF_MA20_GrC = g_max


def _insert_nav(sec: Any, g_max: float) -> None:
    sec.insert("Nav_MA20_GrC")
    sec.gnabar_Nav_MA20_GrC = g_max


def _insert_cdp(sec: Any) -> None:
    sec.insert("CdpCR_MA20_GrC")


def _set_k_ca(sec: Any) -> None:
    sec.ek = K_E_MV
    sec.push()
    sec.eca = CA_E_MV
    h.pop_section()


def _set_na_k_ca(sec: Any) -> None:
    sec.ena = NA_E_MV
    _set_k_ca(sec)


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _load_nrnmech_once(path: Path) -> None:
    if mark_nrnmech_loaded(path):
        return
    h.nrn_load_dll(str(path.resolve()))
