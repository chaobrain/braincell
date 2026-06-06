from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuron import h

from .grc_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    CA_E_MV,
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    GrCConfig,
    GrCParameters,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    grc20_nseg_rule,
    toggles_to_dict,
)


@dataclass
class _NeuronVoltageProbeBundle:
    soma_vector: Any | None
    compartment_vectors: list[Any]


class GrC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCParameters | None = None,
        config: GrCConfig | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else GrCConfig()
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.soma_sections: tuple[Any, ...] = ()
        self.dend_sections: tuple[Any, ...] = ()
        self.axon_sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> GrC:
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
        h.celsius = self.config.temperature_celsius
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

    def _configure_sections(self) -> None:
        branch_rows: list[dict[str, Any]] = []
        compartment_rows: list[dict[str, Any]] = []

        self._configure_soma(self.soma_sections[0])
        for dend_index, sec in enumerate(self.dend_sections):
            self._configure_dend(sec, dend_index)

        branch_index_by_sec = {sec: index for index, sec in enumerate(self.sections)}
        for sec in self.sections:
            prefix = _infer_section_prefix(sec.name())
            local_index = _extract_section_index(sec.name())
            region = _region_for_section(prefix)
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
        p = self.params.soma
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.kv3p4:
            sec.insert("Kv3p4_MA20_GrC")
            sec.gkbar_Kv3p4_MA20_GrC = p.kv3p4
        if t.kv4p3:
            sec.insert("Kv4p3_MA20_GrC")
            sec.gkbar_Kv4p3_MA20_GrC = p.kv4p3
        if t.kir2p3:
            sec.insert("Kir2p3_MA20_GrC")
            sec.gkbar_Kir2p3_MA20_GrC = p.kir2p3
        if t.cahva:
            sec.insert("CaHVA_MA20_GrC")
            sec.gcabar_CaHVA_MA20_GrC = p.cahva
        if t.kv1p1:
            sec.insert("Kv1p1_MA20_GrC")
            sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        if t.kv1p5:
            sec.insert("Kv1p5_MA20_GrC")
            sec.gKur_Kv1p5_MA20_GrC = p.kv1p5
        if t.kv2p2:
            sec.insert("Kv2p2_0010_MA20_GrC")
            sec.gKv2_2bar_Kv2p2_0010_MA20_GrC = p.kv2p2
        if t.cdp:
            _insert_cdp(sec)
        if _uses_k(self.config, "soma"):
            sec.ek = K_E_MV
        if _uses_ca(self.config, "soma"):
            sec.push()
            sec.eca = CA_E_MV
            h.pop_section()

    def _configure_dend(self, sec: Any, dend_index: int) -> None:
        del dend_index
        p = self.params.dend
        t = self.config.toggles
        _set_cable(sec, p.cm_uF_cm2)
        if t.leak:
            _insert_leak(sec, p.leak)
        if t.cahva:
            sec.insert("CaHVA_MA20_GrC")
            sec.gcabar_CaHVA_MA20_GrC = p.cahva
        if t.kca1p1:
            sec.insert("Kca1p1_MA20_GrC")
            sec.gbar_Kca1p1_MA20_GrC = p.kca1p1
        if t.kv1p1:
            sec.insert("Kv1p1_MA20_GrC")
            sec.gbar_Kv1p1_MA20_GrC = p.kv1p1
        if t.cdp:
            _insert_cdp(sec)
        if _uses_k(self.config, "dend"):
            sec.ek = K_E_MV
        if _uses_ca(self.config, "dend"):
            sec.push()
            sec.eca = CA_E_MV
            h.pop_section()

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
            "asc_only": True,
        }


def _set_cable(sec: Any, cm_uF_cm2: float) -> None:
    sec.nseg = grc20_nseg_rule(float(sec.L))
    sec.Ra = RA_OHM_CM
    sec.cm = float(cm_uF_cm2)


def _insert_leak(sec: Any, g_max: float) -> None:
    sec.insert("pas")
    sec.g_pas = float(g_max)
    sec.e_pas = LEAK_E_MV


def _insert_cdp(sec: Any) -> None:
    sec.insert("CdpCR_MA20_GrC")


def _infer_section_prefix(section_name: str) -> str:
    name = section_name.rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _extract_section_index(section_name: str) -> int:
    if "[" not in section_name:
        return 0
    return int(section_name.rsplit("[", 1)[1].split("]", 1)[0])


def _branch_type_for_prefix(prefix: str) -> str:
    if prefix == "soma":
        return "soma"
    if prefix == "dend":
        return "dendrite"
    if prefix == "axon":
        return "axon"
    return prefix


def _region_for_section(prefix: str) -> str:
    if prefix == "soma":
        return "soma"
    if prefix == "dend":
        return "dend"
    raise ValueError(f"Unsupported ASC-only GrC section prefix {prefix!r}.")


def _enabled_region_mechanisms(config: GrCConfig, region: str) -> set[str]:
    return set(_enabled_region_list(config, region))


def _enabled_region_list(config: GrCConfig, region: str) -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in ALL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _uses_k(config: GrCConfig, region: str) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for name in ("kv3p4", "kv4p3", "kir2p3", "kv1p1", "kv1p5", "kv2p2", "kca1p1")
    )


def _uses_ca(config: GrCConfig, region: str) -> bool:
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
