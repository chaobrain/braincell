from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neuron import h

from .parameters import (
    DCN_REGION_NAMES,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_REGION_COUNTS,
    EXPECTED_TOTAL_COUNT,
    DcnTemplateParameters,
    SOURCE_MORPH_PATH,
    branch_type_for_region,
    load_dcn15_params,
)


class DCN:
    def __init__(
        self,
        morph_path: Path | str = SOURCE_MORPH_PATH,
        params: DcnTemplateParameters | None = None,
        *,
        nrnmech_path: Path | str | None = DEFAULT_NRNMECH_PATH,
        temperature_celsius: float = 32.0,
        v_init_mV: float = -65.0,
    ):
        self.morph_path = Path(morph_path)
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.params = params if params is not None else load_dcn15_params(temperature_celsius=self.temperature_celsius)
        self.nrnmech_path = None if nrnmech_path is None else Path(nrnmech_path)
        self.sections: tuple[Any, ...] = ()
        self.sections_by_region: dict[str, tuple[Any, ...]] = {}
        self.root_soma: Any | None = None
        self._ohmic_terms_by_sec: dict[Any, list[tuple[float, float]]] = defaultdict(list)

    def build(self) -> DCN:
        self._load_support()
        self._instantiate_cell()
        self._collect_sections()
        self._configure_sections()
        return self

    def cleanup(self) -> None:
        for sec in self.sections:
            try:
                h.delete_section(sec=sec)
            except Exception:
                pass
        self.sections = ()
        self.sections_by_region = {}
        self.root_soma = None
        self._ohmic_terms_by_sec.clear()

    def _load_support(self) -> None:
        h.load_file("stdlib.hoc")
        h.load_file("stdrun.hoc")
        h.celsius = self.temperature_celsius
        if self.nrnmech_path is not None and self.nrnmech_path.exists():
            _load_nrnmech_once(self.nrnmech_path)
        if not self.morph_path.exists():
            raise FileNotFoundError(f"Missing DCN morphology HOC: {self.morph_path}")

    def _instantiate_cell(self) -> None:
        existing = set(h.allsec())
        h.xopen(str(self.morph_path))
        new_sections = [sec for sec in h.allsec() if sec not in existing]
        if len(new_sections) != EXPECTED_TOTAL_COUNT:
            raise RuntimeError(f"Unexpected DCN section count after morphology load: {len(new_sections)}.")
        self.sections = tuple(new_sections)

    def _collect_sections(self) -> None:
        sections_by_region: dict[str, tuple[Any, ...]] = {}
        sections_by_region["soma"] = tuple(sec for sec in self.sections if _source_section_name(sec) == "soma")
        for region in DCN_REGION_NAMES[1:]:
            section_list = getattr(h, region)
            sections_by_region[region] = tuple(section_list)
        counts = {region: len(sections_by_region[region]) for region in DCN_REGION_NAMES}
        if counts != EXPECTED_REGION_COUNTS:
            raise RuntimeError(f"Unexpected DCN SectionList counts: {counts}.")
        seen = {sec for values in sections_by_region.values() for sec in values}
        if seen != set(self.sections):
            raise RuntimeError("DCN SectionLists do not cover exactly the instantiated sections.")
        self.sections_by_region = sections_by_region
        self.root_soma = sections_by_region["soma"][0]

    def _configure_sections(self) -> None:
        p = self.params
        h.celsius = self.temperature_celsius
        h.calo0_cal_ion = p.calcium_co
        for sec in self.sections:
            sec.Ra = p.ra
            sec.cm = p.cm
            self._add_ohmic(sec, p.pass_g, self.v_init_mV)
        for sec in self.sections_by_region["axNode"]:
            sec.cm = p.cm / 100.0
            self._add_ohmic(sec, p.pass_g_myel - p.pass_g, self.v_init_mV)

        self._configure_soma()
        self._configure_ax_hillock()
        self._configure_ax_ini_seg()
        self._configure_prox_dend()
        self._configure_dist_dend()
        for sec, terms in self._ohmic_terms_by_sec.items():
            _insert_merged_pas(sec, terms)

    def _add_ohmic(self, sec: Any, g_s_cm2: float, e_mV: float) -> None:
        if abs(float(g_s_cm2)) > 0.0:
            self._ohmic_terms_by_sec[sec].append((float(g_s_cm2), float(e_mV)))

    def _configure_soma(self) -> None:
        sec = self.sections_by_region["soma"][0]
        p = self.params
        q = p.qdeltat
        g_na_soma = p.qconductance(p.g_na_f_soma)
        g_fkdr_soma = p.qconductance(p.g_fkdr_soma) * p.scales.kdr_block
        g_skdr_soma = p.qconductance(p.g_skdr_soma) * p.scales.kdr_block
        g_sk_soma = p.qconductance(p.g_sk_soma)
        g_h_soma = p.qconductance(p.g_h_soma)
        g_tnc_soma = p.qconductance(p.g_tnc_soma)
        perm_lva_soma = p.qconductance(p.perm_ca_lva_soma)
        perm_hva_soma = p.qconductance(p.perm_ca_hva_soma)
        _insert_gbar(sec, "NaF_SU15_DCN", g_na_soma, q)
        _insert_gbar(sec, "NaP_SU15_DCN", p.qconductance(p.g_na_p_soma), q)
        sec.ena = p.sodium_rev_pot
        _insert_gbar(sec, "fKdr_SU15_DCN", g_fkdr_soma, q)
        _insert_gbar(sec, "sKdr_SU15_DCN", g_skdr_soma, q)
        _insert_gbar(sec, "SK_SU15_DCN", g_sk_soma, q)
        sec.ek = p.potassium_rev_pot
        _insert_hcn(sec, g_h_soma, p.h_rev_pot, q)
        self._add_ohmic(sec, g_tnc_soma, p.tnc_rev_pot)
        _insert_perm(sec, "CaLVA_SU15_DCN", perm_lva_soma, q)
        sec.calo = p.calcium_co
        _insert_perm(sec, "CaHVA_SU15_DCN", perm_hva_soma, q)
        sec.cao = p.calcium_co
        _insert_cdp_hva(sec, p.k_ca_ca_conc_soma, _soma_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)
        _insert_cdp_lva(sec, p.k_ca_ca_conc_soma, _soma_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)

    def _configure_ax_hillock(self) -> None:
        p = self.params
        g_na_soma, g_fkdr_soma, g_skdr_soma = _axon_base_g(p)
        for sec in self.sections_by_region["axHillock"]:
            self._configure_active_axon(sec, 2.0 * g_na_soma, 2.0 * g_fkdr_soma, 2.0 * g_skdr_soma, p.qconductance(p.g_tnc_ax))

    def _configure_ax_ini_seg(self) -> None:
        p = self.params
        g_na_soma, g_fkdr_soma, g_skdr_soma = _axon_base_g(p)
        for sec in self.sections_by_region["axIniSeg"]:
            self._configure_active_axon(sec, 2.0 * g_na_soma, 2.0 * g_fkdr_soma, 2.0 * g_skdr_soma, p.qconductance(p.g_tnc_ax))

    def _configure_active_axon(self, sec: Any, naf: float, fkdr: float, skdr: float, tnc: float) -> None:
        p = self.params
        _insert_gbar(sec, "NaF_SU15_DCN", naf, p.qdeltat)
        sec.ena = p.sodium_rev_pot
        _insert_gbar(sec, "fKdr_SU15_DCN", fkdr, p.qdeltat)
        _insert_gbar(sec, "sKdr_SU15_DCN", skdr, p.qdeltat)
        sec.ek = p.potassium_rev_pot
        self._add_ohmic(sec, tnc, p.tnc_rev_pot)

    def _configure_prox_dend(self) -> None:
        p = self.params
        q = p.qdeltat
        g_na_soma = p.qconductance(p.g_na_f_soma)
        g_fkdr_soma = p.qconductance(p.g_fkdr_soma) * p.scales.kdr_block
        g_skdr_soma = p.qconductance(p.g_skdr_soma) * p.scales.kdr_block
        g_sk_soma = p.qconductance(p.g_sk_soma)
        g_h_soma = p.qconductance(p.g_h_soma)
        g_tnc_soma = p.qconductance(p.g_tnc_soma)
        perm_lva_soma = p.qconductance(p.perm_ca_lva_soma)
        perm_hva_soma = p.qconductance(p.perm_ca_hva_soma)
        for sec in self.sections_by_region["proxDend"]:
            _insert_gbar(sec, "NaF_SU15_DCN", 0.4 * g_na_soma, q)
            sec.ena = p.sodium_rev_pot
            _insert_gbar(sec, "fKdr_SU15_DCN", 0.6 * g_fkdr_soma, q)
            _insert_gbar(sec, "sKdr_SU15_DCN", 0.6 * g_skdr_soma, q)
            _insert_gbar(sec, "SK_SU15_DCN", 0.3 * g_sk_soma, q)
            sec.ek = p.potassium_rev_pot
            _insert_hcn(sec, 2.0 * g_h_soma, p.h_rev_pot, q)
            self._add_ohmic(sec, 0.2 * g_tnc_soma, p.tnc_rev_pot)
            _insert_perm(sec, "CaLVA_SU15_DCN", 2.0 * perm_lva_soma, q)
            sec.calo = p.calcium_co
            _insert_perm(sec, "CaHVA_SU15_DCN", perm_hva_soma / 1.5, q)
            sec.cao = p.calcium_co
            _insert_cdp_hva(sec, p.k_ca_ca_conc_dend, _dend_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)
            _insert_cdp_lva(sec, p.k_ca_ca_conc_dend, _dend_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)

    def _configure_dist_dend(self) -> None:
        p = self.params
        q = p.qdeltat
        g_sk_soma = p.qconductance(p.g_sk_soma)
        g_h_soma = p.qconductance(p.g_h_soma)
        perm_lva_soma = p.qconductance(p.perm_ca_lva_soma)
        perm_hva_soma = p.qconductance(p.perm_ca_hva_soma)
        for sec in self.sections_by_region["distDend"]:
            _insert_gbar(sec, "SK_SU15_DCN", 0.3 * g_sk_soma, q)
            sec.ek = p.potassium_rev_pot
            _insert_hcn(sec, 3.0 * g_h_soma, p.h_rev_pot, q)
            _insert_perm(sec, "CaLVA_SU15_DCN", 2.0 * perm_lva_soma, q)
            sec.calo = p.calcium_co
            _insert_perm(sec, "CaHVA_SU15_DCN", perm_hva_soma / 1.5, q)
            sec.cao = p.calcium_co
            _insert_cdp_hva(sec, p.k_ca_ca_conc_dend, _dend_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)
            _insert_cdp_lva(sec, p.k_ca_ca_conc_dend, _dend_shell_depth(sec, p.shell_thick), p.tau_ca_conc, p.calcium_ci)


_LOADED_NRNMECH: set[Path] = set()


def _load_nrnmech_once(path: Path) -> None:
    resolved = path.resolve()
    if resolved in _LOADED_NRNMECH:
        return
    h.nrn_load_dll(str(resolved))
    _LOADED_NRNMECH.add(resolved)


def _source_section_name(sec: Any) -> str:
    return sec.name().rsplit(".", 1)[-1]


def _insert_merged_pas(sec: Any, terms: list[tuple[float, float]]) -> None:
    if not terms:
        return
    g_total = sum(g for g, _ in terms)
    if g_total == 0.0:
        return
    e_total = sum(g * e for g, e in terms) / g_total
    sec.insert("pas")
    sec.g_pas = float(g_total)
    sec.e_pas = float(e_total)


def _insert_gbar(sec: Any, mechanism: str, g_s_cm2: float, qdeltat: float) -> None:
    sec.insert(mechanism)
    setattr(sec, f"gbar_{mechanism}", float(g_s_cm2))
    setattr(h, f"qdeltat_{mechanism}", float(qdeltat))


def _insert_hcn(sec: Any, g_s_cm2: float, eh_mV: float, qdeltat: float) -> None:
    mechanism = "HCN_SU15_DCN"
    sec.insert(mechanism)
    sec.gbar_HCN_SU15_DCN = float(g_s_cm2)
    sec.eh_HCN_SU15_DCN = float(eh_mV)
    h.qdeltat_HCN_SU15_DCN = float(qdeltat)


def _insert_perm(sec: Any, mechanism: str, perm_cm_s: float, qdeltat: float) -> None:
    sec.insert(mechanism)
    setattr(sec, f"perm_{mechanism}", float(perm_cm_s))
    setattr(h, f"qdeltat_{mechanism}", float(qdeltat))


def _insert_cdp_hva(sec: Any, k_ca: float, depth: float, tau: float, cai_base: float) -> None:
    mechanism = "CdpHVA_SU15_DCN"
    sec.insert(mechanism)
    sec.kCa_CdpHVA_SU15_DCN = float(k_ca)
    sec.depth_CdpHVA_SU15_DCN = float(depth)
    h.tauCa_CdpHVA_SU15_DCN = float(tau)
    h.caiBase_CdpHVA_SU15_DCN = float(cai_base)


def _insert_cdp_lva(sec: Any, k_cal: float, depth: float, tau: float, cali_base: float) -> None:
    mechanism = "CdpLVA_SU15_DCN"
    sec.insert(mechanism)
    sec.kCal_CdpLVA_SU15_DCN = float(k_cal)
    sec.depth_CdpLVA_SU15_DCN = float(depth)
    h.tauCal_CdpLVA_SU15_DCN = float(tau)
    h.caliBase_CdpLVA_SU15_DCN = float(cali_base)


def _soma_shell_depth(sec: Any, shell_thick: float) -> float:
    diam = float(sec.diam)
    return shell_thick - 2.0 * shell_thick**2 / diam + 4.0 * shell_thick**3 / (3.0 * diam**2)


def _dend_shell_depth(sec: Any, shell_thick: float) -> float:
    return shell_thick - (shell_thick * shell_thick / float(sec.diam))


def _axon_base_g(p: DcnTemplateParameters) -> tuple[float, float, float]:
    return (
        p.qconductance(p.g_na_f_soma),
        p.qconductance(p.g_fkdr_soma) * p.scales.kdr_block,
        p.qconductance(p.g_skdr_soma) * p.scales.kdr_block,
    )
