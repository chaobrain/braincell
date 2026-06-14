from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    CA_E_MV,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    GrCParameters,
    K_E_MV,
    LEAK_E_MV,
    RA_OHM_CM,
    grc20_nseg_rule,
    mark_nrnmech_loaded,
)


class GrC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCParameters | None = None,
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
        self.axon_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None

    def build(self) -> GrC:
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
        h.celsius = self.temperature_celsius
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
        if (
            len(self.soma_sections) != EXPECTED_SOMA_COUNT
            or len(self.dend_sections) != EXPECTED_DEND_COUNT
            or len(self.axon_sections) != EXPECTED_AXON_COUNT
        ):
            raise RuntimeError(
                "Unexpected ASC-only GrC section counts: "
                f"soma={len(self.soma_sections)}, dend={len(self.dend_sections)}, axon={len(self.axon_sections)}."
            )
        self.root_soma = self.soma_sections[0]

    def _configure_sections(self) -> None:
        self._configure_soma(self.soma_sections[0])
        for sec in self.dend_sections:
            self._configure_dend(sec)

    def _configure_soma(self, sec: Any) -> None:
        p = self.params.soma
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        sec.insert("Kv3p4_MA20_GrC")
        sec.gkbar_Kv3p4_MA20_GrC = p.kv3p4
        sec.insert("Kv4p3_MA20_GrC")
        sec.gkbar_Kv4p3_MA20_GrC = p.kv4p3
        sec.insert("Kir2p3_MA20_GrC")
        sec.gkbar_Kir2p3_MA20_GrC = p.kir2p3
        sec.insert("CaHVA_MA20_GrC")
        sec.gcabar_CaHVA_MA20_GrC = p.cahva
        sec.insert("Kv1p1_MA20_GrC")
        sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        sec.insert("Kv1p5_MA20_GrC")
        sec.gKur_Kv1p5_MA20_GrC = p.kv1p5
        sec.insert("Kv2p2_0010_MA20_GrC")
        sec.gKv2_2bar_Kv2p2_0010_MA20_GrC = p.kv2p2
        _insert_cdp(sec)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()

    def _configure_dend(self, sec: Any) -> None:
        p = self.params.dend
        _set_cable(sec, p.cm_uF_cm2)
        _insert_leak(sec, p.leak)
        sec.insert("CaHVA_MA20_GrC")
        sec.gcabar_CaHVA_MA20_GrC = p.cahva
        sec.insert("Kca1p1_MA20_GrC")
        sec.gbar_Kca1p1_MA20_GrC = p.kca1p1
        sec.insert("Kv1p1_MA20_GrC")
        sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        _insert_cdp(sec)
        sec.ek = K_E_MV
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()


def _set_cable(sec: Any, cm_uF_cm2: float) -> None:
    sec.nseg = grc20_nseg_rule(float(sec.L))
    sec.Ra = RA_OHM_CM
    sec.cm = cm_uF_cm2


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = g_max
    sec.e_pas = LEAK_E_MV


def _insert_cdp(sec: Any) -> None:
    sec.insert("CdpCR_MA20_GrC")


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _load_nrnmech_once(path: Path) -> None:
    if mark_nrnmech_loaded(path):
        return
    h.nrn_load_dll(str(path.resolve()))
