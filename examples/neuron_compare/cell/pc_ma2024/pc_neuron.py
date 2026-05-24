from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
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
    PCParameters,
    RA_OHM_CM,
    SOMA_CM_UF_CM2,
    THICK_DEND_DIAM_UM,
    NAV_DEND_DIAM_UM,
)


class PC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: PCParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None

    def build(self) -> PC:
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.load_file("stdrun.hoc")
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

        existing_count = sum(1 for _ in h.allsec())
        reader = h.Import3d_Neurolucida3()
        reader.input(str(self.morph_path))
        h.Import3d_GUI(reader, 0).instantiate(self)
        self.sections = tuple(h.allsec())[existing_count:]
        if len(self.sections) == 0:
            raise RuntimeError(f"NEURON import3d instantiated no sections from {self.morph_path!s}.")

        soma = self.soma[0]
        soma.nseg = 1 + 2 * int(soma.L / CV_MAX_LEN_UM)
        soma.cm = SOMA_CM_UF_CM2
        soma.Ra = RA_OHM_CM

        soma.insert("pas")
        soma.e_pas = LEAK_E_MV
        soma.g_pas = LEAK_G_SOMA_MS_CM2 * 1e-3

        soma.insert("Nav1p6_MA24_PC")
        soma.gbar_Nav1p6_MA24_PC = self.params.nav_soma
        soma.ena = NA_E_MV

        soma.insert("Kv1p1_MA24_PC")
        soma.gbar_Kv1p1_MA24_PC = self.params.kv1p1_soma

        soma.insert("Kv1p5_MA24_PC")
        soma.gKur_Kv1p5_MA24_PC = self.params.kv1p5_soma

        soma.insert("Kv3p4_MA24_PC")
        soma.gkbar_Kv3p4_MA24_PC = self.params.kv3p4_soma

        soma.insert("Kir2p3_MA24_PC")
        soma.gkbar_Kir2p3_MA24_PC = self.params.kir2p3_soma

        soma.insert("Cav2p1_MA24_PC")
        soma.pcabar_Cav2p1_MA24_PC = self.params.cav21_soma_perm

        soma.insert("Cav3p1_MA24_PC")
        soma.pcabar_Cav3p1_MA24_PC = self.params.cav31_soma_perm

        soma.insert("Cav3p2_MA24_PC")
        soma.gcabar_Cav3p2_MA24_PC = self.params.cav32_soma

        soma.insert("Cav3p3_MA24_PC")
        soma.pcabar_Cav3p3_MA24_PC = self.params.cav33_soma_perm

        soma.insert("Kca1p1_MA24_PC")
        soma.gbar_Kca1p1_MA24_PC = self.params.kca1p1_soma

        soma.insert("Kca2p2_MA24_PC")
        soma.gkbar_Kca2p2_MA24_PC = self.params.kca2p2_soma

        soma.insert("Kca3p1_MA24_PC")
        soma.gkbar_Kca3p1_MA24_PC = self.params.kca3p1_soma
        soma.ek = K_E_MV

        soma.insert("HCN1_MA24_PC")
        soma.gbar_HCN1_MA24_PC = self.params.hcn1_soma
        soma.eh = H_E_MV

        soma.insert("CdpCAM_MA24_PC")
        soma.TotalPump_CdpCAM_MA24_PC = CDP_PUMP_SOMA
        soma.push()
        soma.eca = CA_E_MV
        h.define_shape()
        h.pop_section()

        self.root_soma = soma

        for dend in self.dend:
            dend.Ra = RA_OHM_CM
            dend.cm = 11.510294 * math.exp(-1.376463 * dend.diam) + 2.120503
            dend.nseg = 1 + 2 * int(dend.L / CV_MAX_LEN_UM)

            dend.insert("pas")
            dend.e_pas = LEAK_E_MV
            dend.g_pas = LEAK_G_DEND_MS_CM2 * 1e-3

            dend.insert("Kv3p3_MA24_PC")
            dend.gbar_Kv3p3_MA24_PC = self.params.kv3p3_dend

            dend.insert("Kv4p3_MA24_PC")
            dend.gkbar_Kv4p3_MA24_PC = self.params.kv4p3_dend

            dend.insert("Cav2p1_MA24_PC")
            dend.pcabar_Cav2p1_MA24_PC = self.params.cav21_dend_perm

            dend.insert("Cav3p3_MA24_PC")
            dend.pcabar_Cav3p3_MA24_PC = self.params.cav33_dend_perm

            dend.insert("Kca1p1_MA24_PC")
            dend.gbar_Kca1p1_MA24_PC = self.params.kca1p1_dend

            dend.insert("HCN1_MA24_PC")
            dend.gbar_HCN1_MA24_PC = self.params.hcn1_dend
            dend.eh = H_E_MV

            dend.insert("Kca2p2_MA24_PC")
            dend.gkbar_Kca2p2_MA24_PC = self.params.kca2p2_dend

            dend.insert("CdpCAM_MA24_PC")
            dend.TotalPump_CdpCAM_MA24_PC = CDP_PUMP_DEND

            if dend.diam >= THICK_DEND_DIAM_UM:
                dend.cm = SOMA_CM_UF_CM2

                dend.insert("Kv1p1_MA24_PC")
                dend.gbar_Kv1p1_MA24_PC = self.params.kv1p1_dend

                dend.insert("Kv1p5_MA24_PC")
                dend.gKur_Kv1p5_MA24_PC = self.params.kv1p5_dend

                dend.insert("Kir2p3_MA24_PC")
                dend.gkbar_Kir2p3_MA24_PC = self.params.kir2p3_dend

                dend.insert("Cav3p1_MA24_PC")
                dend.pcabar_Cav3p1_MA24_PC = self.params.cav31_dend_perm

                dend.insert("Cav3p2_MA24_PC")
                dend.gcabar_Cav3p2_MA24_PC = self.params.cav32_dend

                dend.insert("Kca3p1_MA24_PC")
                dend.gkbar_Kca3p1_MA24_PC = self.params.kca3p1_dend

                if dend.diam >= NAV_DEND_DIAM_UM:
                    dend.insert("Nav1p6_MA24_PC")
                    dend.gbar_Nav1p6_MA24_PC = self.params.nav_dend
                    dend.ena = NA_E_MV

            dend.ek = K_E_MV
            dend.push()
            dend.eca = CA_E_MV
            h.pop_section()

        return self

    def cleanup(self) -> None:
        for sec in self.sections:
            try:
                h.delete_section(sec=sec)
            except Exception:
                pass
        self.sections = ()
        self.root_soma = None


_LOADED_NRNMECH_PATHS: set[str] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _LOADED_NRNMECH_PATHS:
        return
    h.nrn_load_dll(resolved)
    _LOADED_NRNMECH_PATHS.add(resolved)
