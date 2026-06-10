from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    CA_E_MV,
    CM_UF_CM2,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    H_E_MV,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    BCParameters,
    axon_region_name,
    bc25_nseg_rule,
)


class BC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: BCParameters | None = None,
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

    def build(self) -> BC:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
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
                "Unexpected BC section counts: "
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
        _set_cable(sec)

        _insert_leak(sec, p.leak)

        sec.insert("Nav1p1_MA25_BC")
        sec.gbar_Nav1p1_MA25_BC = p.nav1p1
        sec.ena = NA_E_MV

        sec.insert("Cav3p2_MA25_BC")
        sec.gcabar_Cav3p2_MA25_BC = p.cav3p2

        sec.insert("Cav1p2_MA25_BC")
        sec.gbar_Cav1p2_MA25_BC = p.cav1p2

        sec.insert("Cav1p3_MA25_BC")
        sec.gbar_Cav1p3_MA25_BC = p.cav1p3

        sec.insert("Kir2p3_MA25_BC")
        sec.gkbar_Kir2p3_MA25_BC = p.kir2p3

        sec.insert("Kv3p4_MA25_BC")
        sec.gkbar_Kv3p4_MA25_BC = p.kv3p4

        sec.insert("Kv4p3_MA25_BC")
        sec.gkbar_Kv4p3_MA25_BC = p.kv4p3

        sec.insert("Kca3p1_MA25_BC")
        sec.gkbar_Kca3p1_MA25_BC = p.kca3p1

        sec.insert("HCN1_MA25_BC")
        sec.gbar_HCN1_MA25_BC = p.hcn1
        sec.eh = H_E_MV

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()

    def _configure_dend(self, sec: Any, dend_index: int) -> None:
        del dend_index
        p = self.params.dend
        _set_cable(sec)

        _insert_leak(sec, p.leak)

        sec.insert("Cav3p2_MA25_BC")
        sec.gcabar_Cav3p2_MA25_BC = p.cav3p2

        sec.insert("Cav1p2_MA25_BC")
        sec.gbar_Cav1p2_MA25_BC = p.cav1p2

        sec.insert("Cav1p3_MA25_BC")
        sec.gbar_Cav1p3_MA25_BC = p.cav1p3

        sec.insert("Kv4p3_MA25_BC")
        sec.gkbar_Kv4p3_MA25_BC = p.kv4p3

        sec.insert("Kca2p2_MA25_BC")
        sec.gkbar_Kca2p2_MA25_BC = p.kca2p2

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()

    def _configure_axon(self, sec: Any, axon_index: int) -> None:
        region = axon_region_name(axon_index)
        p = self.params.region(region)
        _set_cable(sec)

        _insert_leak(sec, p.leak)

        sec.insert("Nav1p6_MA25_BC")
        sec.gbar_Nav1p6_MA25_BC = p.nav1p6
        sec.ena = NA_E_MV

        sec.insert("Kv3p4_MA25_BC")
        sec.gkbar_Kv3p4_MA25_BC = p.kv3p4

        if region == "axon_regular":
            sec.insert("Kv1p1_MA25_BC")
            sec.gbar_Kv1p1_MA25_BC = p.kv1p1

        sec.insert("HCN1_MA25_BC")
        sec.gbar_HCN1_MA25_BC = p.hcn1
        sec.eh = H_E_MV

        sec.insert("Kca1p1_MA25_BC")
        sec.gbar_Kca1p1_MA25_BC = p.kca1p1

        sec.insert("Cav2p1_MA25_BC")
        sec.pcabar_Cav2p1_MA25_BC = p.cav2p1

        _insert_cdp(sec, p.cdp_pump)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()


def _set_cable(sec: Any) -> None:
    sec.nseg = bc25_nseg_rule(float(sec.L))
    sec.Ra = RA_OHM_CM
    sec.cm = CM_UF_CM2


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = float(g_max)
    sec.e_pas = LEAK_E_MV


def _insert_cdp(sec: Any, total_pump: float) -> None:
    sec.insert("CdpStC_MA25_BC")
    sec.TotalPump_CdpStC_MA25_BC = float(total_pump)


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
