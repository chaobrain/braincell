from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    FULL_REGION_LOGICAL_MECHANISMS,
    GrCFullConfig,
    GrCFullParameters,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    PF_SECTION_COUNT,
    PF_SECTION_LEN_UM,
    RA_OHM_CM,
    full_toggles_to_dict,
    grc20_nseg_rule,
)


@dataclass
class _NeuronVoltageProbeBundle:
    soma_vector: Any | None
    compartment_vectors: list[Any]


class GrCFull:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCFullParameters | None = None,
        config: GrCFullConfig | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else GrCFullConfig()
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
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
        self._section_local_index: dict[Any, int] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> GrCFull:
        self._load_support()
        self._load_asc_sections()
        self._create_manual_sections()
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
        self.hilock_section = None
        self.ais_section = None
        self.aa_sections = ()
        self.pf1_sections = ()
        self.pf2_sections = ()
        self.root_soma = None
        self._section_region.clear()
        self._section_local_index.clear()

    def _load_support(self) -> None:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
        h.celsius = self.config.temperature_celsius
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
        for sec in self.soma_sections:
            self._section_region[sec] = "soma"
            self._section_local_index[sec] = _extract_section_index(sec.name())
        for sec in self.dend_sections:
            self._section_region[sec] = "dend"
            self._section_local_index[sec] = _extract_section_index(sec.name())
        self.sections = imported

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
            _set_pt3d_line(
                sec,
                (0.0, len_initial_ais, 0.0, 0.3),
                (0.0, len_initial_ais - AA_SECTION_LEN_UM, 0.0, 0.3),
            )
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
        for sec, region, local_index in (
            (self.hilock_section, "hilock", 0),
            (self.ais_section, "ais", 0),
        ):
            self._section_region[sec] = region
            self._section_local_index[sec] = local_index
        for index, sec in enumerate(self.aa_sections):
            self._section_region[sec] = "aa"
            self._section_local_index[sec] = index
        for index, sec in enumerate(self.pf1_sections):
            self._section_region[sec] = "pf"
            self._section_local_index[sec] = index
        for index, sec in enumerate(self.pf2_sections):
            self._section_region[sec] = "pf"
            self._section_local_index[sec] = index

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
        self._collect_tables()

    def _configure_soma(self, sec: Any) -> None:
        p = self.params.soma
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2, use_source_nseg=True)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.kv3p4:
            _insert_kv3p4(sec, p.kv3p4)
        if t.kv4p3:
            sec.insert("Kv4p3_MA20_GrC")
            sec.gkbar_Kv4p3_MA20_GrC = p.kv4p3
        if t.kir2p3:
            sec.insert("Kir2p3_MA20_GrC")
            sec.gkbar_Kir2p3_MA20_GrC = p.kir2p3
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        if t.kv1p1:
            sec.insert("Kv1p1_MA20_GrC")
            sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        if t.kv1p5:
            sec.insert("Kv1p5_MA20_GrC")
            sec.gKur_Kv1p5_MA20_GrC = p.kv1p5
        if t.kv2p2:
            sec.insert("Kv2p2_0010_MA20_GrC")
            sec.gKv2_2bar_Kv2p2_0010_MA20_GrC = p.kv2p2
        _finalize_ions(sec, self.config, "soma")

    def _configure_dend(self, sec: Any) -> None:
        p = self.params.dend
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2, use_source_nseg=True)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        if t.kca1p1:
            sec.insert("Kca1p1_MA20_GrC")
            sec.gbar_Kca1p1_MA20_GrC = p.kca1p1
        if t.kv1p1:
            sec.insert("Kv1p1_MA20_GrC")
            sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        _finalize_ions(sec, self.config, "dend")

    def _configure_hilock(self, sec: Any) -> None:
        p = self.params.hilock
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.nafhhf:
            _insert_nafhhf(sec, p.nafhhf)
        if t.kv3p4:
            _insert_kv3p4(sec, p.kv3p4)
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        _finalize_ions(sec, self.config, "hilock")

    def _configure_ais(self, sec: Any) -> None:
        p = self.params.ais
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.nafhhf:
            _insert_nafhhf(sec, p.nafhhf)
        if t.kv3p4:
            _insert_kv3p4(sec, p.kv3p4)
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        if t.km:
            sec.insert("KM_MA20_GrC")
            sec.gkbar_KM_MA20_GrC = p.km
        _finalize_ions(sec, self.config, "ais")

    def _configure_aa(self, sec: Any) -> None:
        p = self.params.aa
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.nav:
            _insert_nav(sec, p.nav)
        if t.kv3p4:
            _insert_kv3p4(sec, p.kv3p4)
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        _finalize_ions(sec, self.config, "aa")

    def _configure_pf(self, sec: Any) -> None:
        p = self.params.pf
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.nav:
            _insert_nav(sec, p.nav)
        if t.kv3p4:
            _insert_kv3p4(sec, p.kv3p4)
        if t.cahva:
            _insert_cahva(sec, p.cahva)
        _finalize_ions(sec, self.config, "pf")

    def _collect_tables(self) -> None:
        branch_index_by_sec = {sec: index for index, sec in enumerate(self.sections)}
        branch_rows: list[dict[str, Any]] = []
        compartment_rows: list[dict[str, Any]] = []
        for sec in self.sections:
            region = self._section_region[sec]
            local_index = self._section_local_index[sec]
            enabled = _enabled_region_mechanisms(self.config, region)
            branch_index = branch_index_by_sec[sec]
            branch_rows.append(
                {
                    "branch_index": int(branch_index),
                    "branch_name": sec.name(),
                    "branch_type": "soma" if region == "soma" else "dendrite" if region == "dend" else "axon",
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
                        "branch_type": "soma" if region == "soma" else "dendrite" if region == "dend" else "axon",
                        "source_region": region,
                        "source_local_index": int(local_index),
                        "local_index": int(seg_index),
                        "seg_x": float(seg.x),
                        "prox": float(seg.x),
                        "dist": float(seg.x),
                    }
                )
        self._branch_table = pd.DataFrame(branch_rows).reset_index(drop=True)
        self._compartment_table = pd.DataFrame(compartment_rows).reset_index(drop=True)

    def _build_summary(self) -> dict[str, Any]:
        bt = self._branch_table
        ct = self._compartment_table
        return {
            "backend": "neuron",
            "morph_path": str(self.morph_path),
            "toggles": full_toggles_to_dict(self.config.toggles),
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
                for region in FULL_REGION_LOGICAL_MECHANISMS
            },
            "asc_only": False,
            "manual_pf": True,
        }


def _set_pt3d_line(sec: Any, first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> None:
    sec.push()
    h.pt3dclear()
    h.pt3dadd(*first)
    h.pt3dadd(*second)
    h.pop_section()


def _set_cable(sec: Any, cm_uF_cm2: float, *, use_source_nseg: bool = False) -> None:
    sec.nseg = grc20_nseg_rule(float(sec.L)) if use_source_nseg else 1
    sec.Ra = RA_OHM_CM
    sec.cm = float(cm_uF_cm2)


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = float(g_max)
    sec.e_pas = LEAK_E_MV


def _insert_nav(sec: Any, g_max: float) -> None:
    sec.insert("Nav_MA20_GrC")
    sec.gnabar_Nav_MA20_GrC = float(g_max)
    sec.ena = NA_E_MV


def _insert_nafhhf(sec: Any, g_max: float) -> None:
    sec.insert("NaFHF_MA20_GrC")
    sec.gnabar_NaFHF_MA20_GrC = float(g_max)
    sec.ena = NA_E_MV


def _insert_kv3p4(sec: Any, g_max: float) -> None:
    sec.insert("Kv3p4_MA20_GrC")
    sec.gkbar_Kv3p4_MA20_GrC = float(g_max)


def _insert_cahva(sec: Any, g_max: float) -> None:
    sec.insert("CaHVA_MA20_GrC")
    sec.gcabar_CaHVA_MA20_GrC = float(g_max)


def _insert_cdp(sec: Any) -> None:
    sec.insert("CdpCR_MA20_GrC")


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _extract_section_index(section_name: str) -> int:
    if "[" not in section_name:
        return 0
    return int(section_name.rsplit("[", 1)[1].split("]", 1)[0])


def _enabled_region_mechanisms(config: GrCFullConfig, region: str) -> set[str]:
    return set(_enabled_region_list(config, region))


def _enabled_region_list(config: GrCFullConfig, region: str) -> list[str]:
    return [name for name in FULL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in FULL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _finalize_ions(sec: Any, config: GrCFullConfig, region: str) -> None:
    if config.toggles.cdp and "cdp" in FULL_REGION_LOGICAL_MECHANISMS[region]:
        _insert_cdp(sec)
    if _uses_k(config, region):
        sec.ek = K_E_MV
    if _uses_ca(config, region):
        sec.push()
        sec.eca = CA_E_MV
        h.pop_section()


def _uses_k(config: GrCFullConfig, region: str) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for name in ("kv3p4", "kv4p3", "kir2p3", "kv1p1", "kv1p5", "kv2p2", "kca1p1", "km")
    )


def _uses_ca(config: GrCFullConfig, region: str) -> bool:
    return any(name in _enabled_region_list(config, region) for name in ("cahva", "kca1p1", "cdp"))


_LOADED_NRNMECH_PATHS: set[str] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return
    if _density_mechanism_registered("CaHVA_MA20_GrC"):
        _LOADED_NRNMECH_PATHS.add(resolved)
        return
    h.nrn_load_dll(resolved)
    _LOADED_NRNMECH_PATHS.add(resolved)


def _density_mechanism_registered(name: str) -> bool:
    mechanisms = h.MechanismType(0)
    selected = h.ref("")
    for index in range(int(mechanisms.count())):
        mechanisms.select(index)
        mechanisms.selected(selected)
        if selected[0] == name:
            return True
    return False
