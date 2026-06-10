from __future__ import annotations

from pathlib import Path
from typing import Any

from neuron import h

from .parameters import DEFAULT_NRNMECH_PATH, IOParameters


class IO:
    def __init__(
        self,
        params: IOParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.params = params
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.root_soma: Any | None = None

    def build(self) -> IO:
        h.load_file("stdlib.hoc")
        h.load_file("stdrun.hoc")
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)

        p = self.params
        ch = p.channel
        cable = p.cable
        ion = p.ion

        soma = h.Section(name="soma")
        self.sections = (soma,)
        self.root_soma = soma

        soma.L = p.soma.length_um
        soma.diam = p.soma.diam_um
        soma.nseg = int(p.soma.nseg)
        soma.Ra = cable.ra_ohm_cm
        soma.cm = cable.soma_cm_uF_cm2

        soma.insert("pas")
        soma.g_pas = cable.leak_g_S_cm2
        soma.e_pas = cable.leak_e_mV

        soma.insert("Na_ZH19_IO")
        soma.gbar_Na_ZH19_IO = ch.na_gbar_mS_cm2
        soma.ena = ion.na_e_mV

        soma.insert("Kdr_ZH19_IO")
        soma.gbar_Kdr_ZH19_IO = ch.kdr_gbar_mS_cm2
        soma.ek = ion.k_e_mV

        soma.insert("Ca_ZH19_IO")
        soma.gbar_Ca_ZH19_IO = ch.ca_gbar_mS_cm2
        soma.ecas_Ca_ZH19_IO = ch.ca_e_mV
        soma.mMidV_Ca_ZH19_IO = ch.ca_m_mid_mV

        soma.insert("HCN_ZH19_IO")
        soma.gbar_HCN_ZH19_IO = ch.hcn_gbar_mS_cm2
        soma.eh_HCN_ZH19_IO = ch.hcn_e_mV

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
    try:
        h.nrn_load_dll(resolved)
    except RuntimeError as exc:
        if "user defined name already exists" not in str(exc):
            raise
    _LOADED_NRNMECH_PATHS.add(resolved)
