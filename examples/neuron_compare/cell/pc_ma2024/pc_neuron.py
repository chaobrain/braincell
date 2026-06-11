from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    DEFAULT_MORPH_PATH,
    DEFAULT_NRNMECH_PATH,
    PCParameters,
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
        ch = self.params.channel
        cable = self.params.cable
        ion = self.params.ion

        existing_count = sum(1 for _ in h.allsec())
        reader = h.Import3d_Neurolucida3()
        reader.input(str(self.morph_path))
        h.Import3d_GUI(reader, 0).instantiate(self)
        self.sections = tuple(h.allsec())[existing_count:]
        if len(self.sections) == 0:
            raise RuntimeError(f"NEURON import3d instantiated no sections from {self.morph_path!s}.")

        soma = self.soma[0]
        soma.nseg = 1 + 2 * int(soma.L / cable.cv_max_len_um)
        soma.cm = cable.soma_cm_uF_cm2
        soma.Ra = cable.ra_ohm_cm

        soma.insert("pas")
        soma.e_pas = cable.leak_e_mV
        soma.g_pas = cable.leak_g_soma_mS_cm2 * 1e-3

        soma.insert("Nav1p6_MA24_PC")
        soma.gbar_Nav1p6_MA24_PC = ch.soma.nav1p6
        soma.ena = ion.na_e_mV

        soma.insert("Kv1p1_MA24_PC")
        soma.gbar_Kv1p1_MA24_PC = ch.soma.kv1p1

        soma.insert("Kv1p5_MA24_PC")
        soma.gKur_Kv1p5_MA24_PC = ch.soma.kv1p5

        soma.insert("Kv3p4_MA24_PC")
        soma.gkbar_Kv3p4_MA24_PC = ch.soma.kv3p4

        soma.insert("Kir2p3_MA24_PC")
        soma.gkbar_Kir2p3_MA24_PC = ch.soma.kir2p3

        soma.insert("Cav2p1_MA24_PC")
        soma.pcabar_Cav2p1_MA24_PC = ch.soma.cav2p1_perm

        soma.insert("Cav3p1_MA24_PC")
        soma.pcabar_Cav3p1_MA24_PC = ch.soma.cav3p1_perm

        soma.insert("Cav3p2_MA24_PC")
        soma.gcabar_Cav3p2_MA24_PC = ch.soma.cav3p2

        soma.insert("Cav3p3_MA24_PC")
        soma.pcabar_Cav3p3_MA24_PC = ch.soma.cav3p3_perm

        soma.insert("Kca1p1_MA24_PC")
        soma.gbar_Kca1p1_MA24_PC = ch.soma.kca1p1

        soma.insert("Kca2p2_MA24_PC")
        soma.gkbar_Kca2p2_MA24_PC = ch.soma.kca2p2

        soma.insert("Kca3p1_MA24_PC")
        soma.gkbar_Kca3p1_MA24_PC = ch.soma.kca3p1
        soma.ek = ion.k_e_mV

        soma.insert("HCN1_MA24_PC")
        soma.gbar_HCN1_MA24_PC = ch.soma.hcn1
        soma.eh = ion.h_e_mV

        soma.insert("CdpCAM_MA24_PC")
        soma.TotalPump_CdpCAM_MA24_PC = ion.cdp_pump_soma
        soma.push()
        soma.eca = ion.ca_e_mV
        h.define_shape()
        h.pop_section()

        self.root_soma = soma

        for dend in self.dend:
            dend.Ra = cable.ra_ohm_cm
            dend.cm = 11.510294 * math.exp(-1.376463 * dend.diam) + 2.120503
            dend.nseg = 1 + 2 * int(dend.L / cable.cv_max_len_um)

            dend.insert("pas")
            dend.e_pas = cable.leak_e_mV
            dend.g_pas = cable.leak_g_dend_mS_cm2 * 1e-3

            dend.insert("Kv3p3_MA24_PC")
            dend.gbar_Kv3p3_MA24_PC = ch.dend.kv3p3

            dend.insert("Kv4p3_MA24_PC")
            dend.gkbar_Kv4p3_MA24_PC = ch.dend.kv4p3

            dend.insert("Cav2p1_MA24_PC")
            dend.pcabar_Cav2p1_MA24_PC = ch.dend.cav2p1_perm

            dend.insert("Cav3p3_MA24_PC")
            dend.pcabar_Cav3p3_MA24_PC = ch.dend.cav3p3_perm

            dend.insert("Kca1p1_MA24_PC")
            dend.gbar_Kca1p1_MA24_PC = ch.dend.kca1p1

            dend.insert("HCN1_MA24_PC")
            dend.gbar_HCN1_MA24_PC = ch.dend.hcn1
            dend.eh = ion.h_e_mV

            dend.insert("Kca2p2_MA24_PC")
            dend.gkbar_Kca2p2_MA24_PC = ch.dend.kca2p2

            dend.insert("CdpCAM_MA24_PC")
            dend.TotalPump_CdpCAM_MA24_PC = ion.cdp_pump_dend

            if dend.diam >= cable.thick_dend_diam_um:
                dend.cm = cable.soma_cm_uF_cm2

                dend.insert("Kv1p1_MA24_PC")
                dend.gbar_Kv1p1_MA24_PC = ch.dend.kv1p1

                dend.insert("Kv1p5_MA24_PC")
                dend.gKur_Kv1p5_MA24_PC = ch.dend.kv1p5

                dend.insert("Kir2p3_MA24_PC")
                dend.gkbar_Kir2p3_MA24_PC = ch.dend.kir2p3

                dend.insert("Cav3p1_MA24_PC")
                dend.pcabar_Cav3p1_MA24_PC = ch.dend.cav3p1_perm

                dend.insert("Cav3p2_MA24_PC")
                dend.gcabar_Cav3p2_MA24_PC = ch.dend.cav3p2

                dend.insert("Kca3p1_MA24_PC")
                dend.gkbar_Kca3p1_MA24_PC = ch.dend.kca3p1

                if dend.diam >= cable.nav_dend_diam_um:
                    dend.insert("Nav1p6_MA24_PC")
                    dend.gbar_Nav1p6_MA24_PC = ch.dend.nav1p6
                    dend.ena = ion.na_e_mV

            dend.ek = ion.k_e_mV
            dend.push()
            dend.eca = ion.ca_e_mV
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
