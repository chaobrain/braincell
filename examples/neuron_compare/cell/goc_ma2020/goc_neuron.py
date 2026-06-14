from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    GoCParameters,
    axon_region_name,
    dend_region_name,
    goc20_nseg_rule,
)


class GoC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GoCParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.soma_sections: tuple[Any, ...] = ()
        self.dend_sections: tuple[Any, ...] = ()
        self.axon_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None

    def build(self) -> GoC:
        self._load_support()
        self._load_sections()
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
        self.axon_sections = ()
        self.root_soma = None

    def _load_support(self) -> None:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

    def _load_sections(self) -> None:
        existing_count = sum(1 for _ in h.allsec())
        reader = h.Import3d_Neurolucida3()
        reader.input(str(self.morph_path))
        h.Import3d_GUI(reader, 0).instantiate(None)
        self.sections = tuple(h.allsec())[existing_count:]
        if len(self.sections) == 0:
            raise RuntimeError(f"NEURON import3d instantiated no sections from {self.morph_path!s}.")

        by_prefix: dict[str, list[Any]] = defaultdict(list)
        for sec in self.sections:
            by_prefix[_infer_section_prefix(sec.name())].append(sec)
        self.soma_sections = tuple(by_prefix["soma"])
        self.dend_sections = tuple(by_prefix["dend"])
        self.axon_sections = tuple(by_prefix["axon"])
        if len(self.soma_sections) != 1 or len(self.dend_sections) != 151 or len(self.axon_sections) != 75:
            raise RuntimeError(
                "Unexpected GoC section counts: "
                f"soma={len(self.soma_sections)}, dend={len(self.dend_sections)}, axon={len(self.axon_sections)}."
            )
        self.root_soma = self.soma_sections[0]

    def _configure_sections(self) -> None:
        self._configure_soma(self.soma_sections[0])
        for dend_index, sec in enumerate(self.dend_sections):
            self._configure_dend(sec, dend_index)
        for axon_index, sec in enumerate(self.axon_sections):
            self._configure_axon(sec, axon_index)

    def _configure_soma(self, sec: Any) -> None:
        cable = self.params.cable
        ion = self.params.ion
        ch = self.params.channel.soma
        sec.nseg = goc20_nseg_rule(float(sec.L), max_len_um=cable.cv_max_len_um)
        sec.Ra = cable.ra_ohm_cm
        sec.cm = cable.soma_cm_uF_cm2
        _insert_leak(sec, cable.leak_g_default_S_cm2, cable.leak_e_mV)

        sec.insert("Kv1p1_MA20_GoC")
        sec.gbar_Kv1p1_MA20_GoC = ch.kv1p1
        sec.insert("Kv3p4_MA20_GoC")
        sec.gkbar_Kv3p4_MA20_GoC = ch.kv3p4
        sec.insert("Kv4p3_MA20_GoC")
        sec.gkbar_Kv4p3_MA20_GoC = ch.kv4p3
        sec.insert("Nav1p6_MA20_GoC")
        sec.gbar_Nav1p6_MA20_GoC = ch.nav
        sec.ena = ion.na_e_mV
        sec.insert("Kca1p1_MA20_GoC")
        sec.gbar_Kca1p1_MA20_GoC = ch.kca1p1
        sec.insert("Kca3p1_MA20_GoC")
        sec.gkbar_Kca3p1_MA20_GoC = ch.kca3p1
        sec.insert("CaHVA_MA20_GoC")
        sec.gcabar_CaHVA_MA20_GoC = ch.cahva
        sec.insert("Cav3p1_MA20_GoC")
        sec.pcabar_Cav3p1_MA20_GoC = ch.cav3p1
        _insert_cdp(sec, self.params.ion.cdp_pump_soma)
        sec.ek = ion.k_e_mV
        sec.eca = ion.ca_e_mV

    def _configure_dend(self, sec: Any, dend_index: int) -> None:
        cable = self.params.cable
        ion = self.params.ion
        region = dend_region_name(dend_index)
        sec.nseg = goc20_nseg_rule(float(sec.L), max_len_um=cable.cv_max_len_um)
        sec.Ra = cable.ra_ohm_cm
        sec.cm = cable.dend_cm_uF_cm2
        _insert_leak(sec, cable.leak_g_default_S_cm2, cable.leak_e_mV)

        if region == "dend_apical":
            ch = self.params.channel.dend_apical
            sec.insert("Nav1p6_MA20_GoC")
            sec.gbar_Nav1p6_MA20_GoC = ch.nav
            sec.ena = ion.na_e_mV
            sec.insert("Kca1p1_MA20_GoC")
            sec.gbar_Kca1p1_MA20_GoC = ch.kca1p1
            sec.insert("Kca2p2_MA20_GoC")
            sec.gkbar_Kca2p2_MA20_GoC = ch.kca2p2
            sec.insert("Cav2p3_MA20_GoC")
            sec.gcabar_Cav2p3_MA20_GoC = ch.cav2p3
            sec.insert("Cav3p1_MA20_GoC")
            sec.pcabar_Cav3p1_MA20_GoC = ch.cav3p1
            _insert_cdp(sec, self.params.ion.cdp_pump_dend_apical)
        else:
            ch = self.params.channel.dend_basal
            sec.insert("Nav1p6_MA20_GoC")
            sec.gbar_Nav1p6_MA20_GoC = ch.nav
            sec.ena = ion.na_e_mV
            sec.insert("Kca1p1_MA20_GoC")
            sec.gbar_Kca1p1_MA20_GoC = ch.kca1p1
            sec.insert("Kca2p2_MA20_GoC")
            sec.gkbar_Kca2p2_MA20_GoC = ch.kca2p2
            sec.insert("CaHVA_MA20_GoC")
            sec.gcabar_CaHVA_MA20_GoC = ch.cahva
            _insert_cdp(sec, self.params.ion.cdp_pump_dend_basal)
        sec.ek = ion.k_e_mV
        sec.eca = ion.ca_e_mV

    def _configure_axon(self, sec: Any, axon_index: int) -> None:
        cable = self.params.cable
        ion = self.params.ion
        region = axon_region_name(axon_index)
        sec.nseg = goc20_nseg_rule(float(sec.L), max_len_um=cable.cv_max_len_um)
        sec.Ra = cable.ra_ohm_cm
        sec.cm = cable.axon_cm_uF_cm2

        if region == "axon_ais":
            ch = self.params.channel.axon_ais
            _insert_leak(sec, cable.leak_g_default_S_cm2, cable.leak_e_mV)
            sec.insert("HCN1_MA20_GoC")
            sec.gbar_HCN1_MA20_GoC = ch.hcn1
            sec.insert("HCN2_MA20_GoC")
            sec.gbar_HCN2_MA20_GoC = ch.hcn2
            sec.insert("Nav1p6_MA20_GoC")
            sec.gbar_Nav1p6_MA20_GoC = ch.nav
            sec.ena = ion.na_e_mV
            sec.insert("KM_MA20_GoC")
            sec.gkbar_KM_MA20_GoC = ch.km
            sec.insert("Kca1p1_MA20_GoC")
            sec.gbar_Kca1p1_MA20_GoC = ch.kca1p1
            sec.insert("CaHVA_MA20_GoC")
            sec.gcabar_CaHVA_MA20_GoC = ch.cahva
        else:
            ch = self.params.channel.axon_regular
            _insert_leak(sec, cable.leak_g_regular_axon_S_cm2, cable.leak_e_mV)
            sec.insert("Kv3p4_MA20_GoC")
            sec.gkbar_Kv3p4_MA20_GoC = ch.kv3p4
            sec.insert("Nav1p6_MA20_GoC")
            sec.gbar_Nav1p6_MA20_GoC = ch.nav
            sec.ena = ion.na_e_mV
        _insert_cdp(sec, self.params.ion.cdp_pump_axon)
        sec.ek = ion.k_e_mV
        sec.eca = ion.ca_e_mV


def _insert_leak(sec: Any, g_max: float, e_pas_mV: float) -> None:
    sec.insert("pas")
    sec.g_pas = g_max
    sec.e_pas = e_pas_mV


def _insert_cdp(sec: Any, total_pump: float) -> None:
    sec.insert("CdpStC_MA20_GoC")
    sec.TotalPump_CdpStC_MA20_GoC = total_pump


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


_LOADED_NRNMECH_PATHS: set[str] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return
    h.nrn_load_dll(resolved)
    _LOADED_NRNMECH_PATHS.add(resolved)
