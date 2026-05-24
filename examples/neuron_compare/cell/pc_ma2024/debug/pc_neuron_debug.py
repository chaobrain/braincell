from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuron import h

from .pc_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    CA_E_MV,
    CDP_PUMP_DEND,
    CDP_PUMP_SOMA,
    CV_MAX_LEN_UM,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    H_E_MV,
    K_E_MV,
    LEAK_E_MV,
    LEAK_G_DEND_MS_CM2,
    LEAK_G_SOMA_MS_CM2,
    NA_E_MV,
    PCConfig,
    PCParameters,
    RA_OHM_CM,
    THICK_DEND_DIAM_UM,
    NAV_DEND_DIAM_UM,
    pc24_dend_cm,
    pc24_nseg_rule,
    toggles_to_dict,
)


@dataclass
class _NeuronVoltageProbeBundle:
    soma_vector: Any | None
    compartment_vectors: list[Any]


class PC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: PCParameters | None = None,
        config: PCConfig | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else PCConfig()
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> PC:
        self._load_support()
        self._load_sections()
        self._configure_sections()
        self._finalize_tables()
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
                branch_type = _infer_branch_type(sec.name())
                if branch_type not in {"soma", "dend"}:
                    continue
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

    def _configure_sections(self) -> None:
        branch_rows: list[dict[str, Any]] = []
        compartment_rows: list[dict[str, Any]] = []
        root_soma = None
        for branch_index, sec in enumerate(self.sections):
            branch_type = _infer_branch_type(sec.name())
            if branch_type not in {"soma", "dend"}:
                continue
            diam_um = float(sec.diam)
            length_um = float(sec.L)
            is_thick = branch_type == "dend" and diam_um >= THICK_DEND_DIAM_UM
            is_nav = branch_type == "dend" and diam_um >= NAV_DEND_DIAM_UM
            sec.nseg = pc24_nseg_rule(length_um)
            sec.Ra = RA_OHM_CM
            sec.cm = 2.0 if branch_type == "soma" else pc24_dend_cm(diam_um)
            enabled = set()
            uses_na = False
            uses_k = False
            uses_ca = False
            uses_h = False
            t = self.config.toggles
            if t.leak:
                sec.insert("pas")
                sec.e_pas = LEAK_E_MV
                sec.g_pas = 1e-3 if branch_type == "soma" else 3e-4
                enabled.add("leak")
            if branch_type == "soma":
                if t.nav:
                    sec.insert("Nav1p6_MA24_PC")
                    sec.gbar_Nav1p6_MA24_PC = self.params.nav_soma
                    enabled.add("nav")
                    uses_na = True
                if t.kv1p1:
                    sec.insert("Kv1p1_MA24_PC")
                    sec.gbar_Kv1p1_MA24_PC = self.params.kv1p1_soma
                    enabled.add("kv1p1")
                    uses_k = True
                if t.kv1p5:
                    sec.insert("Kv1p5_MA24_PC")
                    sec.gKur_Kv1p5_MA24_PC = self.params.kv1p5_soma
                    enabled.add("kv1p5")
                    uses_k = True
                if t.kv3p4:
                    sec.insert("Kv3p4_MA24_PC")
                    sec.gkbar_Kv3p4_MA24_PC = self.params.kv3p4_soma
                    enabled.add("kv3p4")
                    uses_k = True
                if t.kir2p3:
                    sec.insert("Kir2p3_MA24_PC")
                    sec.gkbar_Kir2p3_MA24_PC = self.params.kir2p3_soma
                    enabled.add("kir2p3")
                    uses_k = True
                if t.cav21:
                    sec.insert("Cav2p1_MA24_PC")
                    sec.pcabar_Cav2p1_MA24_PC = self.params.cav21_soma
                    enabled.add("cav21")
                    uses_ca = True
                if t.cav31:
                    sec.insert("Cav3p1_MA24_PC")
                    sec.pcabar_Cav3p1_MA24_PC = self.params.cav31_soma
                    enabled.add("cav31")
                    uses_ca = True
                if t.cav32:
                    sec.insert("Cav3p2_MA24_PC")
                    sec.gcabar_Cav3p2_MA24_PC = self.params.cav32_soma
                    enabled.add("cav32")
                    uses_ca = True
                if t.cav33:
                    sec.insert("Cav3p3_MA24_PC")
                    sec.pcabar_Cav3p3_MA24_PC = self.params.cav33_soma_perm
                    enabled.add("cav33")
                    uses_ca = True
                if t.kca1p1:
                    sec.insert("Kca1p1_MA24_PC")
                    sec.gbar_Kca1p1_MA24_PC = self.params.kca1p1_soma
                    enabled.add("kca1p1")
                    uses_k = True
                    uses_ca = True
                if t.kca2p2:
                    sec.insert("Kca2p2_MA24_PC")
                    sec.gkbar_Kca2p2_MA24_PC = self.params.kca2p2_soma
                    enabled.add("kca2p2")
                    uses_k = True
                    uses_ca = True
                if t.kca3p1:
                    sec.insert("Kca3p1_MA24_PC")
                    sec.gkbar_Kca3p1_MA24_PC = self.params.kca3p1_soma
                    enabled.add("kca3p1")
                    uses_k = True
                    uses_ca = True
                if t.hcn1:
                    sec.insert("HCN1_MA24_PC")
                    sec.gbar_HCN1_MA24_PC = self.params.hcn1_soma
                    enabled.add("hcn1")
                    uses_h = True
                if t.cdp:
                    sec.insert("CdpCAM_MA24_PC")
                    sec.TotalPump_CdpCAM_MA24_PC = CDP_PUMP_SOMA
                    enabled.add("cdp")
                    uses_ca = True
                root_soma = sec
            else:
                if t.kv3p3:
                    sec.insert("Kv3p3_MA24_PC")
                    sec.gbar_Kv3p3_MA24_PC = self.params.kv3p3_dend
                    enabled.add("kv3p3")
                    uses_k = True
                if t.kv4p3:
                    sec.insert("Kv4p3_MA24_PC")
                    sec.gkbar_Kv4p3_MA24_PC = self.params.kv4p3_dend
                    enabled.add("kv4p3")
                    uses_k = True
                if t.cav21:
                    sec.insert("Cav2p1_MA24_PC")
                    sec.pcabar_Cav2p1_MA24_PC = self.params.cav21_dend
                    enabled.add("cav21")
                    uses_ca = True
                if t.cav33:
                    sec.insert("Cav3p3_MA24_PC")
                    sec.pcabar_Cav3p3_MA24_PC = self.params.cav33_dend_perm
                    enabled.add("cav33")
                    uses_ca = True
                if t.kca1p1:
                    sec.insert("Kca1p1_MA24_PC")
                    sec.gbar_Kca1p1_MA24_PC = self.params.kca1p1_dend
                    enabled.add("kca1p1")
                    uses_k = True
                    uses_ca = True
                if t.hcn1:
                    sec.insert("HCN1_MA24_PC")
                    sec.gbar_HCN1_MA24_PC = self.params.hcn1_dend
                    enabled.add("hcn1")
                    uses_h = True
                if t.kca2p2:
                    sec.insert("Kca2p2_MA24_PC")
                    sec.gkbar_Kca2p2_MA24_PC = self.params.kca2p2_dend
                    enabled.add("kca2p2")
                    uses_k = True
                    uses_ca = True
                if t.cdp:
                    sec.insert("CdpCAM_MA24_PC")
                    sec.TotalPump_CdpCAM_MA24_PC = CDP_PUMP_DEND
                    enabled.add("cdp")
                    uses_ca = True
                if is_thick:
                    sec.cm = 2.0
                    if t.kv1p1:
                        sec.insert("Kv1p1_MA24_PC")
                        sec.gbar_Kv1p1_MA24_PC = self.params.kv1p1_dend
                        enabled.add("kv1p1")
                        uses_k = True
                    if t.kv1p5:
                        sec.insert("Kv1p5_MA24_PC")
                        sec.gKur_Kv1p5_MA24_PC = self.params.kv1p5_dend
                        enabled.add("kv1p5")
                        uses_k = True
                    if t.kir2p3:
                        sec.insert("Kir2p3_MA24_PC")
                        sec.gkbar_Kir2p3_MA24_PC = self.params.kir2p3_dend
                        enabled.add("kir2p3")
                        uses_k = True
                    if t.cav31:
                        sec.insert("Cav3p1_MA24_PC")
                        sec.pcabar_Cav3p1_MA24_PC = self.params.cav31_dend
                        enabled.add("cav31")
                        uses_ca = True
                    if t.cav32:
                        sec.insert("Cav3p2_MA24_PC")
                        sec.gcabar_Cav3p2_MA24_PC = self.params.cav32_dend
                        enabled.add("cav32")
                        uses_ca = True
                    if t.kca3p1:
                        sec.insert("Kca3p1_MA24_PC")
                        sec.gkbar_Kca3p1_MA24_PC = self.params.kca3p1_dend
                        enabled.add("kca3p1")
                        uses_k = True
                        uses_ca = True
                    if is_nav and t.nav:
                        sec.insert("Nav1p6_MA24_PC")
                        sec.gbar_Nav1p6_MA24_PC = self.params.nav_dend
                        enabled.add("nav")
                        uses_na = True
            if uses_na:
                sec.ena = NA_E_MV
            if uses_k:
                sec.ek = K_E_MV
            if uses_ca:
                sec.eca = CA_E_MV
            if uses_h:
                sec.eh = H_E_MV
            branch_rows.append(
                {
                    "branch_index": branch_index,
                    "branch_name": sec.name(),
                    "branch_type": "soma" if branch_type == "soma" else "dendrite",
                    "diam_um": diam_um,
                    "diam_arc_mean_um": diam_um,
                    "cm_uF_cm2": float(sec.cm),
                    "nseg": int(sec.nseg),
                    "is_thick_dend": bool(is_thick),
                    "is_nav_dend": bool(is_nav),
                    **_mechanism_flag_row(enabled),
                    "enabled_mechanisms": sorted(enabled),
                }
            )
            for local_index, seg in enumerate(sec):
                compartment_rows.append(
                    {
                        "compartment_index": int(len(compartment_rows)),
                        "branch_index": branch_index,
                        "branch_name": sec.name(),
                        "branch_type": "soma" if branch_type == "soma" else "dendrite",
                        "local_index": int(local_index),
                        "seg_x": float(seg.x),
                        "prox": float(seg.x),
                        "dist": float(seg.x),
                    }
                )
        if root_soma is None:
            raise ValueError("Expected exactly one root soma section.")
        self.root_soma = root_soma
        self._branch_table = pd.DataFrame(branch_rows)
        self._compartment_table = pd.DataFrame(compartment_rows)

    def _finalize_tables(self) -> None:
        if len(self._branch_table) == 0:
            return
        self._branch_table = self._branch_table.reset_index(drop=True)
        self._compartment_table = self._compartment_table.reset_index(drop=True)

    def _build_summary(self) -> dict[str, Any]:
        bt = self._branch_table
        ct = self._compartment_table
        soma_enabled = _enabled_region_list(self.config, "soma")
        dend_all_enabled = _enabled_region_list(self.config, "dend_all")
        dend_thick_enabled = _enabled_region_list(self.config, "dend_thick")
        dend_nav_enabled = _enabled_region_list(self.config, "dend_nav")
        uses_any = lambda names: any(getattr(self.config.toggles, name) for name in names)
        return {
            "backend": "neuron",
            "morph_path": str(self.morph_path),
            "indiv": int(self.params.indiv),
            "toggles": toggles_to_dict(self.config.toggles),
            "branch_counts": {
                "n_soma": int((bt["branch_type"] == "soma").sum()),
                "n_dend": int((bt["branch_type"] == "dendrite").sum()),
                "n_total": int(len(bt)),
            },
            "compartment_counts": {"n_total_nseg": int(len(ct))},
            "threshold_hits": {
                "n_thick_dend": int(bt["is_thick_dend"].sum()),
                "n_nav_dend": int(bt["is_nav_dend"].sum()),
            },
            "enabled_mechanisms": {
                "soma": soma_enabled,
                "dend_all": dend_all_enabled,
                "dend_thick": dend_thick_enabled,
                "dend_nav": dend_nav_enabled,
            },
            "ion_status": {
                "ena_enabled": bool(getattr(self.config.toggles, "nav")),
                "ek_enabled": bool(
                    uses_any(("kv1p1", "kv1p5", "kv3p3", "kv3p4", "kv4p3", "kir2p3", "kca1p1", "kca2p2", "kca3p1"))
                ),
                "eca_enabled": bool(
                    uses_any(("cav21", "cav31", "cav32", "cav33", "kca1p1", "kca2p2", "kca3p1", "cdp"))
                ),
                "eh_enabled": bool(getattr(self.config.toggles, "hcn1")),
            },
        }


def _infer_branch_type(section_name: str) -> str:
    prefix = section_name.split("[", 1)[0]
    if prefix == "soma":
        return "soma"
    if prefix == "dend":
        return "dend"
    return prefix


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in ALL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _enabled_region_list(config: PCConfig, region: str) -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


_LOADED_NRNMECH_PATHS: set[str] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return
    h.nrn_load_dll(resolved)
    _LOADED_NRNMECH_PATHS.add(resolved)
