from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuron import h

from ..io_neuron import _load_nrnmech_once
from .io_parameters import (
    DEFAULT_NRNMECH_PATH,
    IOConfig,
    IOParameters,
    enabled_region_list,
    load_io19_params,
    toggles_to_dict,
)


@dataclass
class _NeuronVoltageProbeBundle:
    soma_vector: Any | None
    compartment_vectors: list[Any]


class IO:
    def __init__(
        self,
        params: IOParameters | None = None,
        config: IOConfig | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        self.params = params if params is not None else load_io19_params()
        self.config = config if config is not None else IOConfig()
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> IO:
        self._load_support()
        self._create_soma()
        self._configure_soma()
        self._collect_tables()
        self._summary = self._build_summary()
        return self

    def summary(self) -> dict[str, Any]:
        return self._summary

    def branch_table(self) -> pd.DataFrame:
        return self._branch_table.copy()

    def compartment_table(self) -> pd.DataFrame:
        return self._compartment_table.copy()

    def attach_voltage_probes(self, *, all_compartments: bool = True, soma: bool = True) -> dict[str, Any]:
        if self.root_soma is None:
            raise RuntimeError("build() must run before attaching probes.")
        soma_vector = h.Vector().record(self.root_soma(0.5)._ref_v) if soma else None
        compartment_vectors: list[Any] = []
        if all_compartments:
            for seg in self.root_soma:
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
        h.load_file("stdrun.hoc")
        h.celsius = self.config.temperature_celsius
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

    def _create_soma(self) -> None:
        soma = h.Section(name="soma")
        self.sections = (soma,)
        self.root_soma = soma

    def _configure_soma(self) -> None:
        if self.root_soma is None:
            raise RuntimeError("Soma section must be created before configuration.")
        p = self.params
        t = self.config.toggles
        soma = self.root_soma
        soma.L = p.soma.length_um
        soma.diam = p.soma.diam_um
        soma.nseg = int(p.soma.nseg)
        soma.Ra = p.soma.ra_ohm_cm
        soma.cm = p.soma.cm_uF_cm2

        if t.leak:
            soma.insert("pas")
            soma.g_pas = p.leak_g_S_cm2
            soma.e_pas = p.leak_e_mV
        if t.na:
            soma.insert("Na_ZH19_IO")
            soma.gbar_Na_ZH19_IO = p.na_gbar_mS_cm2
            soma.ena = p.na_e_mV
        if t.kdr:
            soma.insert("Kdr_ZH19_IO")
            soma.gbar_Kdr_ZH19_IO = p.kdr_gbar_mS_cm2
            soma.ek = p.k_e_mV
        if t.ca:
            soma.insert("Ca_ZH19_IO")
            soma.gbar_Ca_ZH19_IO = p.ca_gbar_mS_cm2
            soma.ecas_Ca_ZH19_IO = p.ca_e_mV
            soma.mMidV_Ca_ZH19_IO = p.ca_m_mid_mV
        if t.hcn:
            soma.insert("HCN_ZH19_IO")
            soma.gbar_HCN_ZH19_IO = p.hcn_gbar_mS_cm2
            soma.eh_HCN_ZH19_IO = p.hcn_e_mV

    def _collect_tables(self) -> None:
        if self.root_soma is None:
            raise RuntimeError("build() must create soma before collecting tables.")
        soma = self.root_soma
        self._branch_table = pd.DataFrame(
            [
                {
                    "branch_index": 0,
                    "branch_name": "soma[0]",
                    "branch_type": "soma",
                    "length_um": float(soma.L),
                    "diam_um": float(soma.diam),
                    "nseg": int(soma.nseg),
                }
            ]
        )
        self._compartment_table = pd.DataFrame(
            [
                {
                    "compartment_index": int(index),
                    "branch_index": 0,
                    "branch_name": "soma[0]",
                    "branch_type": "soma",
                    "local_index": int(index),
                    "prox": float((index / soma.nseg)),
                    "dist": float(((index + 1) / soma.nseg)),
                }
                for index, _seg in enumerate(soma)
            ]
        )

    def _build_summary(self) -> dict[str, Any]:
        return {
            "backend": "neuron",
            "manual_soma": True,
            "nrnmech_path": None if self.nrnmech_path is None else str(self.nrnmech_path),
            "toggles": toggles_to_dict(self.config.toggles),
            "branch_counts": {"n_soma": 1, "n_total": 1},
            "compartment_counts": {"n_total_nseg": int(len(self._compartment_table))},
            "enabled_mechanisms": {"soma": enabled_region_list(self.config, "soma")},
            "ion_status": {
                "ena_enabled": bool(self.config.toggles.na),
                "ek_enabled": bool(self.config.toggles.kdr),
                "ca_channel_enabled": bool(self.config.toggles.ca),
                "hcn_enabled": bool(self.config.toggles.hcn),
            },
        }
