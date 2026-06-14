from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuron import h

from .goc_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    AXON_CM_UF_CM2,
    CA_E_MV,
    CDP_PUMP_AXON,
    CDP_PUMP_DEND_APICAL,
    CDP_PUMP_DEND_BASAL,
    CDP_PUMP_SOMA,
    DEFAULT_MORPH_PATH,
    DEND_CM_UF_CM2,
    K_E_MV,
    LEAK_E_MV,
    LEAK_G_DEFAULT_S_CM2,
    LEAK_G_REGULAR_AXON_S_CM2,
    NA_E_MV,
    RA_OHM_CM,
    SOMA_CM_UF_CM2,
    SOURCE_NRNMECH_PATH,
    GoCConfig,
    GoCParameters,
    axon_region_name,
    dend_region_name,
    goc20_nseg_rule,
    toggles_to_dict,
)


@dataclass
class _NeuronVoltageProbeBundle:
    soma_vector: Any | None
    compartment_vectors: list[Any]


class GoC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GoCParameters | None = None,
        config: GoCConfig | None = None,
        *,
        nrnmech_path: Path | str | None = SOURCE_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else GoCConfig()
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.soma_sections: tuple[Any, ...] = ()
        self.dend_sections: tuple[Any, ...] = ()
        self.axon_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> GoC:
        self._load_support()
        self._load_sections()
        self._configure_sections()
        self._summary = self._build_summary()
        return self

    def summary(self) -> dict[str, Any]:
        return self._summary

    def branch_table(self) -> pd.DataFrame:
        return self._branch_table.copy()

    def compartment_table(self) -> pd.DataFrame:
        return self._compartment_table.copy()

    def attach_voltage_probes(self, *, all_compartments: bool = True, soma: bool = True) -> dict[str, Any]:
        soma_vector = None
        if soma:
            if self.root_soma is None:
                raise RuntimeError("build() must run before attaching probes.")
            soma_vector = h.Vector().record(self.root_soma(0.5)._ref_v)
        compartment_vectors: list[Any] = []
        if all_compartments:
            for sec in self.sections:
                for seg in sec:
                    compartment_vectors.append(h.Vector().record(seg._ref_v))
        return {
            "bundle": _NeuronVoltageProbeBundle(
                soma_vector=soma_vector,
                compartment_vectors=compartment_vectors,
            ),
            "compartment_table": self.compartment_table() if all_compartments else pd.DataFrame(),
        }

    def collect_voltage_results(self, probes: dict[str, Any]) -> dict[str, Any]:
        bundle = probes["bundle"]
        soma_voltage = None
        if bundle.soma_vector is not None:
            soma_voltage = np.asarray(bundle.soma_vector, dtype=float).reshape(-1)
        compartment_voltage = None
        if bundle.compartment_vectors:
            compartment_voltage = np.column_stack(
                [np.asarray(vec, dtype=float).reshape(-1) for vec in bundle.compartment_vectors]
            )
        return {
            "soma_voltage_mV": soma_voltage,
            "compartment_voltage_mV": compartment_voltage,
            "compartment_table": probes["compartment_table"].copy(),
        }

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

    def _configure_sections(self) -> None:
        branch_rows: list[dict[str, Any]] = []
        compartment_rows: list[dict[str, Any]] = []

        self._configure_soma(self.soma_sections[0])
        for dend_index, sec in enumerate(self.dend_sections):
            self._configure_dend(sec, dend_index)
        for axon_index, sec in enumerate(self.axon_sections):
            self._configure_axon(sec, axon_index)

        branch_index_by_sec = {sec: index for index, sec in enumerate(self.sections)}
        for sec in self.sections:
            prefix = _infer_section_prefix(sec.name())
            local_index = _extract_section_index(sec.name())
            region = _region_for_section(prefix, local_index)
            enabled = _enabled_region_mechanisms(self.config, region)
            branch_index = branch_index_by_sec[sec]
            branch_rows.append(
                {
                    "branch_index": int(branch_index),
                    "branch_name": sec.name(),
                    "branch_type": _branch_type_for_prefix(prefix),
                    "source_region": region,
                    "source_local_index": int(local_index),
                    "diam_um": float(sec.diam),
                    "diam_arc_mean_um": float(sec.diam),
                    "cm_uF_cm2": float(sec.cm),
                    "nseg": int(sec.nseg),
                    **_mechanism_flag_row(enabled),
                    "enabled_mechanisms": sorted(enabled),
                }
            )
            for seg_index, seg in enumerate(sec):
                compartment_rows.append(
                    {
                        "compartment_index": int(len(compartment_rows)),
                        "branch_index": int(branch_index),
                        "branch_name": sec.name(),
                        "branch_type": _branch_type_for_prefix(prefix),
                        "source_region": region,
                        "source_local_index": int(local_index),
                        "local_index": int(seg_index),
                        "seg_x": float(seg.x),
                        "prox": float(seg.x),
                        "dist": float(seg.x),
                    }
                )
        self.root_soma = self.soma_sections[0]
        self._branch_table = pd.DataFrame(branch_rows).reset_index(drop=True)
        self._compartment_table = pd.DataFrame(compartment_rows).reset_index(drop=True)

    def _configure_soma(self, sec: Any) -> None:
        p = self.params
        t = self.config.toggles
        sec.nseg = goc20_nseg_rule(float(sec.L))
        sec.Ra = RA_OHM_CM
        sec.cm = SOMA_CM_UF_CM2
        if t.leak:
            _insert_leak(sec, LEAK_G_DEFAULT_S_CM2)
        if t.kv1p1:
            sec.insert("Kv1p1_MA20_GoC")
            sec.gbar_Kv1p1_MA20_GoC = p.kv1p1_soma
        if t.kv3p4:
            sec.insert("Kv3p4_MA20_GoC")
            sec.gkbar_Kv3p4_MA20_GoC = p.kv3p4_soma
        if t.kv4p3:
            sec.insert("Kv4p3_MA20_GoC")
            sec.gkbar_Kv4p3_MA20_GoC = p.kv4p3_soma
        if t.nav:
            sec.insert("Nav1p6_MA20_GoC")
            sec.gbar_Nav1p6_MA20_GoC = p.nav_soma
            sec.ena = NA_E_MV
        if t.kca1p1:
            sec.insert("Kca1p1_MA20_GoC")
            sec.gbar_Kca1p1_MA20_GoC = p.kca1p1_soma
        if t.kca3p1:
            sec.insert("Kca3p1_MA20_GoC")
            sec.gkbar_Kca3p1_MA20_GoC = p.kca3p1_soma
        if t.cahva:
            sec.insert("CaHVA_MA20_GoC")
            sec.gcabar_CaHVA_MA20_GoC = p.cahva_soma
        if t.cav3p1:
            sec.insert("Cav3p1_MA20_GoC")
            sec.pcabar_Cav3p1_MA20_GoC = p.cav3p1_soma
        if t.cdp:
            _insert_cdp(sec, CDP_PUMP_SOMA)
        if _uses_k(self.config, "soma"):
            sec.ek = K_E_MV
        if _uses_ca(self.config, "soma"):
            sec.eca = CA_E_MV

    def _configure_dend(self, sec: Any, dend_index: int) -> None:
        p = self.params
        t = self.config.toggles
        region = dend_region_name(dend_index)
        sec.nseg = goc20_nseg_rule(float(sec.L))
        sec.Ra = RA_OHM_CM
        sec.cm = DEND_CM_UF_CM2
        if t.leak:
            _insert_leak(sec, LEAK_G_DEFAULT_S_CM2)
        if region == "dend_apical":
            if t.nav:
                sec.insert("Nav1p6_MA20_GoC")
                sec.gbar_Nav1p6_MA20_GoC = p.nav_dend_apical
                sec.ena = NA_E_MV
            if t.kca1p1:
                sec.insert("Kca1p1_MA20_GoC")
                sec.gbar_Kca1p1_MA20_GoC = p.kca1p1_dend_apical
            if t.kca2p2:
                sec.insert("Kca2p2_MA20_GoC")
                sec.gkbar_Kca2p2_MA20_GoC = p.kca2p2_dend_apical
            if t.cav2p3:
                sec.insert("Cav2p3_MA20_GoC")
                sec.gcabar_Cav2p3_MA20_GoC = p.cav2p3_dend_apical
            if t.cav3p1:
                sec.insert("Cav3p1_MA20_GoC")
                sec.pcabar_Cav3p1_MA20_GoC = p.cav3p1_dend_apical
            if t.cdp:
                _insert_cdp(sec, CDP_PUMP_DEND_APICAL)
        else:
            if t.nav:
                sec.insert("Nav1p6_MA20_GoC")
                sec.gbar_Nav1p6_MA20_GoC = p.nav_dend_basal
                sec.ena = NA_E_MV
            if t.kca1p1:
                sec.insert("Kca1p1_MA20_GoC")
                sec.gbar_Kca1p1_MA20_GoC = p.kca1p1_dend_basal
            if t.kca2p2:
                sec.insert("Kca2p2_MA20_GoC")
                sec.gkbar_Kca2p2_MA20_GoC = p.kca2p2_dend_basal
            if t.cahva:
                sec.insert("CaHVA_MA20_GoC")
                sec.gcabar_CaHVA_MA20_GoC = p.cahva_dend_basal
            if t.cdp:
                _insert_cdp(sec, CDP_PUMP_DEND_BASAL)
        if _uses_k(self.config, region):
            sec.ek = K_E_MV
        if _uses_ca(self.config, region):
            sec.eca = CA_E_MV

    def _configure_axon(self, sec: Any, axon_index: int) -> None:
        p = self.params
        t = self.config.toggles
        region = axon_region_name(axon_index)
        sec.nseg = goc20_nseg_rule(float(sec.L))
        sec.Ra = RA_OHM_CM
        sec.cm = AXON_CM_UF_CM2
        if region == "axon_ais":
            if t.leak:
                _insert_leak(sec, LEAK_G_DEFAULT_S_CM2)
            if t.hcn1:
                sec.insert("HCN1_MA20_GoC")
                sec.gbar_HCN1_MA20_GoC = p.hcn1_ais
            if t.hcn2:
                sec.insert("HCN2_MA20_GoC")
                sec.gbar_HCN2_MA20_GoC = p.hcn2_ais
            if t.nav:
                sec.insert("Nav1p6_MA20_GoC")
                sec.gbar_Nav1p6_MA20_GoC = p.nav_ais
                sec.ena = NA_E_MV
            if t.km:
                sec.insert("KM_MA20_GoC")
                sec.gkbar_KM_MA20_GoC = p.km_ais
            if t.kca1p1:
                sec.insert("Kca1p1_MA20_GoC")
                sec.gbar_Kca1p1_MA20_GoC = p.kca1p1_ais
            if t.cahva:
                sec.insert("CaHVA_MA20_GoC")
                sec.gcabar_CaHVA_MA20_GoC = p.cahva_ais
            if t.cdp:
                _insert_cdp(sec, CDP_PUMP_AXON)
        else:
            if t.leak:
                _insert_leak(sec, LEAK_G_REGULAR_AXON_S_CM2)
            if t.kv3p4:
                sec.insert("Kv3p4_MA20_GoC")
                sec.gkbar_Kv3p4_MA20_GoC = p.kv3p4_axon_regular
            if t.nav:
                sec.insert("Nav1p6_MA20_GoC")
                sec.gbar_Nav1p6_MA20_GoC = p.nav_axon_regular
                sec.ena = NA_E_MV
            if t.cdp:
                _insert_cdp(sec, CDP_PUMP_AXON)
        if _uses_k(self.config, region):
            sec.ek = K_E_MV
        if _uses_ca(self.config, region):
            sec.eca = CA_E_MV

    def _build_summary(self) -> dict[str, Any]:
        bt = self._branch_table
        ct = self._compartment_table
        return {
            "backend": "neuron",
            "morph_path": str(self.morph_path),
            "toggles": toggles_to_dict(self.config.toggles),
            "branch_counts": {
                "n_soma": int((bt["branch_type"] == "soma").sum()),
                "n_dend": int((bt["branch_type"] == "dendrite").sum()),
                "n_axon": int((bt["branch_type"] == "axon").sum()),
                "n_total": int(len(bt)),
            },
            "region_counts": bt["source_region"].value_counts().sort_index().to_dict(),
            "compartment_counts": {"n_total_nseg": int(len(ct))},
            "enabled_mechanisms": {
                region: _enabled_region_list(self.config, region)
                for region in ALL_REGION_LOGICAL_MECHANISMS
            },
        }


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = float(g_max)
    sec.e_pas = LEAK_E_MV


def _insert_cdp(sec: Any, total_pump: float) -> None:
    sec.insert("CdpStC_MA20_GoC")
    sec.TotalPump_CdpStC_MA20_GoC = float(total_pump)


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _extract_section_index(section_name: str) -> int:
    return int(section_name.rsplit("[", 1)[1].split("]", 1)[0])


def _branch_type_for_prefix(prefix: str) -> str:
    if prefix == "soma":
        return "soma"
    if prefix == "dend":
        return "dendrite"
    if prefix == "axon":
        return "axon"
    return prefix


def _region_for_section(prefix: str, local_index: int) -> str:
    if prefix == "soma":
        return "soma"
    if prefix == "dend":
        return dend_region_name(local_index)
    if prefix == "axon":
        return axon_region_name(local_index)
    raise ValueError(f"Unsupported GoC section prefix {prefix!r}.")


def _enabled_region_mechanisms(config: GoCConfig, region: str) -> set[str]:
    return set(_enabled_region_list(config, region))


def _enabled_region_list(config: GoCConfig, region: str) -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in ALL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _uses_k(config: GoCConfig, region: str) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for name in ("kv1p1", "kv3p4", "kv4p3", "km", "kca1p1", "kca2p2", "kca3p1")
    )


def _uses_ca(config: GoCConfig, region: str) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for name in ("kca1p1", "kca2p2", "kca3p1", "cahva", "cav2p3", "cav3p1", "cdp")
    )


_LOADED_NRNMECH_PATHS: set[str] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return
    h.nrn_load_dll(resolved)
    _LOADED_NRNMECH_PATHS.add(resolved)
