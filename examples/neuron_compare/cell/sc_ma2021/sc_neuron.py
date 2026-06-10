from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    CA_E_MV,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    DEND_CM_UF_CM2,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    H_E_MV,
    K_E_AXON_MV,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    SCParameters,
    axon_region_name,
    dend_region_name,
    sc21_nseg_rule,
)


class SC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: SCParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
        temperature_celsius: float = 32.0,
        v_init_mV: float = -65.0,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.sections: tuple[Any, ...] = ()
        self.soma_sections: tuple[Any, ...] = ()
        self.dend_sections: tuple[Any, ...] = ()
        self.axon_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None

    def build(self) -> SC:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
        h.celsius = self.temperature_celsius
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

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
        if (
            len(self.soma_sections) != EXPECTED_SOMA_COUNT
            or len(self.dend_sections) != EXPECTED_DEND_COUNT
            or len(self.axon_sections) != EXPECTED_AXON_COUNT
        ):
            raise RuntimeError(
                "Unexpected SC section counts: "
                f"soma={len(self.soma_sections)}, dend={len(self.dend_sections)}, axon={len(self.axon_sections)}."
            )

        self._configure_soma(self.soma_sections[0])
        for dend_index, sec in enumerate(self.dend_sections):
            self._configure_dend(sec, dend_index)
        for axon_index, sec in enumerate(self.axon_sections):
            self._configure_axon(sec, axon_index)
        self.root_soma = self.soma_sections[0]
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

    def _configure_soma(self, sec: Any) -> None:
        p = self.params.soma
        _set_cable(sec, "soma")

        _insert_leak(sec, p.leak)

        sec.insert("Nav1p1_RI21_SC")
        sec.gbar_Nav1p1_RI21_SC = p.nav1p1
        sec.ena = NA_E_MV

        sec.insert("Cav3p2_RI21_SC")
        sec.gcabar_Cav3p2_RI21_SC = p.cav3p2

        sec.insert("Cav3p3_RI21_SC")
        sec.pcabar_Cav3p3_RI21_SC = p.cav3p3
        sec.gCav3_3bar_Cav3p3_RI21_SC = p.cav3p3_g_scale

        sec.insert("Kir2p3_RI21_SC")
        sec.gkbar_Kir2p3_RI21_SC = p.kir2p3

        sec.insert("Kv1p1_RI21_SC")
        sec.gbar_Kv1p1_RI21_SC = p.kv1p1

        sec.insert("Kv3p4_RI21_SC")
        sec.gkbar_Kv3p4_RI21_SC = p.kv3p4

        sec.insert("Kv4p3_RI21_SC")
        sec.gkbar_Kv4p3_RI21_SC = p.kv4p3

        sec.insert("Kca1p1_RI21_SC")
        sec.gbar_Kca1p1_RI21_SC = p.kca1p1

        sec.insert("Kca2p2_RI21_SC")
        sec.gkbar_Kca2p2_RI21_SC = p.kca2p2

        sec.insert("Cav2p1_RI21_SC")
        sec.pcabar_Cav2p1_RI21_SC = p.cav2p1

        sec.insert("HCN1_RI21_SC")
        sec.gbar_HCN1_RI21_SC = p.hcn1
        sec.eh = H_E_MV

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()

    def _configure_dend(self, sec: Any, dend_index: int) -> None:
        region = dend_region_name(dend_index)
        p = self.params.region(region)
        _set_cable(sec, region)

        _insert_leak(sec, p.leak)

        sec.insert("Cav2p1_RI21_SC")
        sec.pcabar_Cav2p1_RI21_SC = p.cav2p1

        if region == "dendprox":
            sec.insert("Cav3p2_RI21_SC")
            sec.gcabar_Cav3p2_RI21_SC = p.cav3p2

            sec.insert("Cav3p3_RI21_SC")
            sec.pcabar_Cav3p3_RI21_SC = p.cav3p3
            sec.gCav3_3bar_Cav3p3_RI21_SC = p.cav3p3_g_scale

            sec.insert("Kv4p3_RI21_SC")
            sec.gkbar_Kv4p3_RI21_SC = p.kv4p3

        sec.insert("Kca1p1_RI21_SC")
        sec.gbar_Kca1p1_RI21_SC = p.kca1p1

        sec.insert("Kca2p2_RI21_SC")
        sec.gkbar_Kca2p2_RI21_SC = p.kca2p2

        sec.insert("Kv1p1_RI21_SC")
        sec.gbar_Kv1p1_RI21_SC = p.kv1p1

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()

    def _configure_axon(self, sec: Any, axon_index: int) -> None:
        region = axon_region_name(axon_index)
        p = self.params.region(region)
        _set_cable(sec, region)

        _insert_leak(sec, p.leak)

        sec.insert("Nav1p6_RI21_SC")
        sec.gbar_Nav1p6_RI21_SC = p.nav1p6
        sec.ena = NA_E_MV

        sec.insert("Kv3p4_RI21_SC")
        sec.gkbar_Kv3p4_RI21_SC = p.kv3p4

        sec.insert("Kv1p1_RI21_SC")
        sec.gbar_Kv1p1_RI21_SC = p.kv1p1

        sec.insert("HCN1_RI21_SC")
        sec.gbar_HCN1_RI21_SC = p.hcn1
        sec.eh = H_E_MV

        if region == "axon_ais":
            sec.insert("KM_RI21_SC")
            sec.gkbar_KM_RI21_SC = p.km

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_AXON_MV


def _set_cable(sec: Any, region: str) -> None:
    sec.nseg = sc21_nseg_rule(float(sec.L))
    sec.Ra = RA_OHM_CM
    sec.cm = DEND_CM_UF_CM2 if region.startswith("dend") else 1.0


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = float(g_max)
    sec.e_pas = LEAK_E_MV


def _insert_cdp(sec: Any, total_pump: float) -> None:
    sec.insert("CdpStC_RI21_SC")
    sec.TotalPump_CdpStC_RI21_SC = float(total_pump)


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
