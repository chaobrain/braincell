from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

import brainunit as u
import numpy as np
from braincell import Cell, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, BranchSlice

from .parameters import (
    DCN_REGION_NAMES,
    DEFAULT_NATIVE_DIR,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_REGION_COUNTS,
    EXPECTED_SOMA_COUNT,
    DcnTemplateParameters,
    load_dcn15_params,
)

if str(DEFAULT_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_NATIVE_DIR))

from dcn_native import DcnMorphology, dcn_region, load_dcn_morphology


class DCN:
    def __init__(
        self,
        native: DcnMorphology | None = None,
        params: DcnTemplateParameters | None = None,
        *,
        temperature_celsius: float = 32.0,
        v_init_mV: float = -65.0,
    ):
        self.native = native
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.params = params if params is not None else load_dcn15_params(temperature_celsius=self.temperature_celsius)
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> DCN:
        self.native = load_dcn_morphology() if self.native is None else self.native
        self.morpho = self.native.morpho
        self.cell = Cell(
            self.morpho,
            cv_policy=CVPerBranchList(tuple(1 for _ in self.morpho.branches)),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order="family",
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    def _define_regions(self) -> None:
        if self.morpho is None or self.native is None:
            raise RuntimeError("Morphology must be loaded before defining regions.")
        type_counts = Counter(branch.type for branch in self.morpho.branches)
        if (
            type_counts.get("soma", 0) != EXPECTED_SOMA_COUNT
            or type_counts.get("dendrite", 0) != EXPECTED_DEND_COUNT
            or type_counts.get("axon", 0) != EXPECTED_AXON_COUNT
        ):
            raise RuntimeError(f"Unexpected DCN branch counts from BrainCell native import: {dict(type_counts)}.")
        self.regions = {region: dcn_region(self.morpho, region) for region in DCN_REGION_NAMES}
        local_index_by_region: dict[str, int] = defaultdict(int)
        for spec in self.native.specs:
            local_index_by_region[spec.region] += 1
        counts = {region: local_index_by_region[region] for region in DCN_REGION_NAMES}
        if counts != EXPECTED_REGION_COUNTS:
            raise RuntimeError(f"Unexpected native DCN region counts: {counts}.")

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        temp = u.celsius2kelvin(self.temperature_celsius)
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=self.v_init_mV * u.mV,
                membrane_capacitance=self.params.cm * (u.uF / u.cm**2),
                axial_resistivity=self.params.ra * (u.ohm * u.cm),
                temperature=temp,
            ),
        )
        self.cell.paint(
            self.regions["axNode"],
            mech.CableProperty(
                resting_potential=self.v_init_mV * u.mV,
                membrane_capacitance=(self.params.cm / 100.0) * (u.uF / u.cm**2),
                axial_resistivity=self.params.ra * (u.ohm * u.cm),
                temperature=temp,
            ),
        )
        self.cell.paint(
            AllRegion() - self.regions["axNode"],
            mech.Channel("IL", name="IL", g_max=self.params.pass_g * (u.siemens / u.cm**2), E=self.v_init_mV * u.mV),
        )
        self.cell.paint(
            self.regions["axNode"],
            mech.Channel("IL", name="IL", g_max=self.params.pass_g_myel * (u.siemens / u.cm**2), E=self.v_init_mV * u.mV),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=self.params.sodium_rev_pot * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=self.params.potassium_rev_pot * u.mV))
        self._paint_cdp_hva_ions()
        self._paint_cdp_lva_ions()

    def _paint_cdp_hva_ions(self) -> None:
        self._paint_cdp_ions(
            class_name="CdpHVA_SU2015_DCN",
            name="ca_hva",
            k_name="kCa",
            k_soma=self.params.k_ca_ca_conc_soma,
            k_dend=self.params.k_ca_ca_conc_dend,
            tau_name="tauCa",
            base_name="caiBase",
        )

    def _paint_cdp_lva_ions(self) -> None:
        self._paint_cdp_ions(
            class_name="CdpLVA_SU2015_DCN",
            name="ca_lva",
            k_name="kCal",
            k_soma=self.params.k_ca_ca_conc_soma,
            k_dend=self.params.k_ca_ca_conc_dend,
            tau_name="tauCal",
            base_name="caliBase",
        )

    def _paint_cdp_ions(
        self,
        *,
        class_name: str,
        name: str,
        k_name: str,
        k_soma: float,
        k_dend: float,
        tau_name: str,
        base_name: str,
    ) -> None:
        if self.cell is None or self.morpho is None or self.native is None:
            raise RuntimeError("Cell and morphology must be ready before painting calcium pools.")
        active_regions = {"soma", "proxDend", "distDend"}
        for spec in self.native.specs:
            if spec.region not in active_regions:
                continue
            branch = self.morpho.branch(name=spec.branch_name)
            diam_um = _section_diam_um(spec)
            depth_um = (
                _soma_shell_depth_um(diam_um, self.params.shell_thick)
                if spec.region == "soma"
                else _dend_shell_depth_um(diam_um, self.params.shell_thick)
            )
            k_value = k_soma if spec.region == "soma" else k_dend
            self.cell.paint(
                BranchSlice(branch_index=int(branch.index), prox=0.0, dist=1.0),
                mech.Ion(
                    class_name,
                    name=name,
                    Co=self.params.calcium_co * u.mM,
                    Ci_initializer=self.params.calcium_ci * u.mM,
                    depth=depth_um * u.um,
                    **{
                        k_name: k_value / u.coulomb,
                        tau_name: self.params.tau_ca_conc * u.ms,
                        base_name: self.params.calcium_ci * u.mM,
                    },
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        q = self.params.qdeltat
        temp = u.celsius2kelvin(self.temperature_celsius)

        g_na_soma = self.params.qconductance(self.params.g_na_f_soma)
        g_fkdr_soma = self.params.qconductance(self.params.g_fkdr_soma) * self.params.scales.kdr_block
        g_skdr_soma = self.params.qconductance(self.params.g_skdr_soma) * self.params.scales.kdr_block
        g_sk_soma = self.params.qconductance(self.params.g_sk_soma)
        perm_lva_soma = self.params.qconductance(self.params.perm_ca_lva_soma)
        perm_hva_soma = self.params.qconductance(self.params.perm_ca_hva_soma)
        g_h_soma = self.params.qconductance(self.params.g_h_soma)
        g_tnc_soma = self.params.qconductance(self.params.g_tnc_soma)

        self._paint_active_set(
            "soma",
            naf=g_na_soma,
            nap=self.params.qconductance(self.params.g_na_p_soma),
            fkdr=g_fkdr_soma,
            skdr=g_skdr_soma,
            sk=g_sk_soma,
            hcn=g_h_soma,
            tnc=g_tnc_soma,
            calva=perm_lva_soma,
            cahva=perm_hva_soma,
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "axHillock",
            naf=2.0 * g_na_soma,
            fkdr=2.0 * g_fkdr_soma,
            skdr=2.0 * g_skdr_soma,
            tnc=self.params.qconductance(self.params.g_tnc_ax),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "axIniSeg",
            naf=2.0 * g_na_soma,
            fkdr=2.0 * g_fkdr_soma,
            skdr=2.0 * g_skdr_soma,
            tnc=self.params.qconductance(self.params.g_tnc_ax),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "proxDend",
            naf=0.4 * g_na_soma,
            fkdr=0.6 * g_fkdr_soma,
            skdr=0.6 * g_skdr_soma,
            sk=0.3 * g_sk_soma,
            hcn=2.0 * g_h_soma,
            tnc=0.2 * g_tnc_soma,
            calva=2.0 * perm_lva_soma,
            cahva=perm_hva_soma / 1.5,
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "distDend",
            sk=0.3 * g_sk_soma,
            hcn=3.0 * g_h_soma,
            calva=2.0 * perm_lva_soma,
            cahva=perm_hva_soma / 1.5,
            temp=temp,
            qdeltat=q,
        )

    def _paint_active_set(
        self,
        region_name: str,
        *,
        temp: Any,
        qdeltat: float,
        naf: float | None = None,
        nap: float | None = None,
        fkdr: float | None = None,
        skdr: float | None = None,
        sk: float | None = None,
        hcn: float | None = None,
        tnc: float | None = None,
        calva: float | None = None,
        cahva: float | None = None,
    ) -> None:
        region = self.regions[region_name]
        suffix = region_name
        if naf is not None:
            _paint_channel(self.cell, region, "NaF_SU2015_DCN", f"NaF_{suffix}", naf, "na")
        if nap is not None:
            _paint_channel(self.cell, region, "NaP_SU2015_DCN", f"NaP_{suffix}", nap, "na")
        if fkdr is not None:
            _paint_channel(self.cell, region, "fKdr_SU2015_DCN", f"fKdr_{suffix}", fkdr, "k")
        if skdr is not None:
            _paint_channel(self.cell, region, "sKdr_SU2015_DCN", f"sKdr_{suffix}", skdr, "k")
        if sk is not None:
            _paint_channel(self.cell, region, "SK_SU2015_DCN", f"SK_{suffix}", sk, {"k": "k", "ca": "ca_hva"}, qdeltat)
        if hcn is not None:
            self.cell.paint(
                region,
                mech.Channel("HCN_SU2015_DCN", name=f"HCN_{suffix}", g_max=hcn * (u.siemens / u.cm**2), E=self.params.h_rev_pot * u.mV),
            )
        if tnc is not None:
            self.cell.paint(
                region,
                mech.Channel("IL", name=f"TNC_{suffix}", g_max=tnc * (u.siemens / u.cm**2), E=self.params.tnc_rev_pot * u.mV),
            )
        if calva is not None:
            _paint_ca(self.cell, region, f"CaLVA_{suffix}", "CaLVA_SU2015_DCN", calva, "ca_lva", temp, qdeltat)
        if cahva is not None:
            _paint_ca(self.cell, region, f"CaHVA_{suffix}", "CaHVA_SU2015_DCN", cahva, "ca_hva", temp, qdeltat)


def _paint_channel(cell: Cell, region, class_name: str, name: str, g_s_cm2: float, ion_name, qdeltat: float | None = None) -> None:
    kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
    if qdeltat is not None:
        kwargs["qdeltat"] = qdeltat
    cell.paint(region, mech.Channel(class_name, name=name, g_max=g_s_cm2 * (u.siemens / u.cm**2), **kwargs))


def _paint_ca(cell: Cell, region, name: str, class_name: str, perm_cm_s: float, ion_name: str, temp, qdeltat: float) -> None:
    cell.paint(
        region,
        mech.Channel(
            class_name,
            name=name,
            perm=perm_cm_s * (u.cm / u.second),
            temp=temp,
            qdeltat=qdeltat,
            ion_name=ion_name,
        ),
    )


def _section_diam_um(spec) -> float:
    return float(np.mean([spec.prox.diam, spec.dist.diam]))


def _soma_shell_depth_um(diam_um: float, shell_thick_um: float) -> float:
    return shell_thick_um - 2.0 * shell_thick_um**2 / diam_um + 4.0 * shell_thick_um**3 / (3.0 * diam_um**2)


def _dend_shell_depth_um(diam_um: float, shell_thick_um: float) -> float:
    return shell_thick_um - shell_thick_um**2 / diam_um
