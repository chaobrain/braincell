from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import brainunit as u
import numpy as np
import pandas as pd
from braincell import Cell, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, BranchSlice, at

from .dcn_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    DCN_REGION_NAMES,
    DEFAULT_NATIVE_DIR,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_REGION_COUNTS,
    EXPECTED_SOMA_COUNT,
    DcnConfig,
    DcnTemplateParameters,
    branch_type_for_region,
    enabled_region_list,
    enabled_region_mechanisms,
    load_dcn15_params,
    mechanism_flag_row,
    toggles_to_dict,
)

if str(DEFAULT_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_NATIVE_DIR))

from dcn_native import DcnMorphology, dcn_region, load_dcn_morphology
from dcn_native import DEFAULT_SOURCE_HOC as DEFAULT_SOURCE_HOC_MORPH


@dataclass
class _BrainCellVoltageProbeBundle:
    soma_probe_name: str | None
    compartment_probe_names: list[str]


class DCN:
    def __init__(
        self,
        native: DcnMorphology | None = None,
        params: DcnTemplateParameters | None = None,
        config: DcnConfig | None = None,
    ):
        self.config = config if config is not None else DcnConfig()
        self.params = params if params is not None else load_dcn15_params(self.config)
        self.native = native
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}
        self._source_region_by_branch_id: dict[int, str] = {}
        self._source_local_index_by_branch_id: dict[int, int] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> DCN:
        self.native = load_dcn_morphology() if self.native is None else self.native
        self.morpho = self.native.morpho
        self.cell = Cell(
            self.morpho,
            cv_policy=CVPerBranchList(tuple(1 for _ in self.morpho.branches)),
            V_init=self.config.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order="family",
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
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
        if self.cell is None or self.morpho is None:
            raise RuntimeError("build() must run before attaching probes.")
        soma_probe_name = None
        if soma:
            soma_probe_name = "v_soma"
            self.cell.place(at("soma", 0.5), mech.StateProbe(name=soma_probe_name))
        compartment_probe_names: list[str] = []
        if all_compartments:
            local_index_by_branch: dict[int, int] = defaultdict(int)
            rows: list[dict[str, Any]] = []
            for cv in self.cell.cvs:
                branch_id = int(cv.branch_id)
                branch = self.morpho.branch(index=branch_id)
                if branch.type not in {"soma", "dendrite", "axon"}:
                    continue
                local_index = local_index_by_branch[branch_id]
                local_index_by_branch[branch_id] += 1
                probe_name = f"cv_{int(cv.id)}_v"
                midpoint_x = 0.5 * (float(cv.prox) + float(cv.dist))
                self.cell.place(at(branch_id, midpoint_x), mech.StateProbe(name=probe_name))
                compartment_probe_names.append(probe_name)
                rows.append(
                    {
                        "compartment_index": int(len(rows)),
                        "branch_index": branch_id,
                        "branch_name": branch.name,
                        "source_section": self.native.source_name_by_branch[branch.name],
                        "branch_type": branch.type,
                        "source_region": self._source_region_by_branch_id[branch_id],
                        "source_local_index": self._source_local_index_by_branch_id[branch_id],
                        "local_index": int(local_index),
                        "cv_id": int(cv.id),
                        "prox": float(cv.prox),
                        "dist": float(cv.dist),
                    }
                )
            compartment_table = pd.DataFrame(rows)
        else:
            compartment_table = pd.DataFrame()
        return {
            "bundle": _BrainCellVoltageProbeBundle(
                soma_probe_name=soma_probe_name,
                compartment_probe_names=compartment_probe_names,
            ),
            "compartment_table": compartment_table,
        }

    def collect_voltage_results(self, probes: dict[str, Any], run_result: Any) -> dict[str, Any]:
        bundle = probes["bundle"]
        soma_voltage = None
        if bundle.soma_probe_name is not None:
            soma_voltage = np.asarray(run_result.traces[bundle.soma_probe_name].to_decimal(u.mV), dtype=float).reshape(-1)
        compartment_voltage = None
        if bundle.compartment_probe_names:
            compartment_voltage = np.column_stack(
                [
                    np.asarray(run_result.traces[name].to_decimal(u.mV), dtype=float).reshape(-1)
                    for name in bundle.compartment_probe_names
                ]
            )
        return {
            "soma_voltage_mV": soma_voltage,
            "compartment_voltage_mV": compartment_voltage,
            "compartment_table": probes["compartment_table"].copy(),
        }

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
            branch = self.morpho.branch(name=spec.branch_name)
            local_index = local_index_by_region[spec.region]
            local_index_by_region[spec.region] += 1
            self._source_region_by_branch_id[int(branch.index)] = spec.region
            self._source_local_index_by_branch_id[int(branch.index)] = int(local_index)
        counts = {region: local_index_by_region[region] for region in DCN_REGION_NAMES}
        if counts != EXPECTED_REGION_COUNTS:
            raise RuntimeError(f"Unexpected native DCN region counts: {counts}.")

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=self.config.v_init_mV * u.mV,
                membrane_capacitance=self.params.cm * (u.uF / u.cm**2),
                axial_resistivity=self.params.ra * (u.ohm * u.cm),
                temperature=temp,
            ),
        )
        self.cell.paint(
            self.regions["axNode"],
            mech.CableProperty(
                resting_potential=self.config.v_init_mV * u.mV,
                membrane_capacitance=(self.params.cm / 100.0) * (u.uF / u.cm**2),
                axial_resistivity=self.params.ra * (u.ohm * u.cm),
                temperature=temp,
            ),
        )
        if self.config.toggles.leak:
            non_axnode = AllRegion() - self.regions["axNode"]
            # TODO: The non-overlapping paint keeps the DCN debug model correct,
            # but currently lowers to two IL runtime layouts because g_max differs.
            # Optimize density lowering later so same class/name channels can share
            # one vectorized instance with per-point parameters.
            self.cell.paint(
                non_axnode,
                mech.Channel("IL", name="IL", g_max=self.params.pass_g * (u.siemens / u.cm**2), E=self.config.v_init_mV * u.mV),
            )
            self.cell.paint(
                self.regions["axNode"],
                mech.Channel(
                    "IL",
                    name="IL",
                    g_max=self.params.pass_g_myel * (u.siemens / u.cm**2),
                    E=self.config.v_init_mV * u.mV,
                ),
            )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=self.params.sodium_rev_pot * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=self.params.potassium_rev_pot * u.mV))
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        if self.config.toggles.ca_conc:
            self._paint_cdp_hva_ions()
        elif self.config.toggles.cahva or self.config.toggles.sk:
            self.cell.paint(
                self.regions["soma"] | self.regions["proxDend"] | self.regions["distDend"],
                mech.Ion(
                    "CalciumInitNernst",
                    name="ca_hva",
                    Co=self.params.calcium_co * u.mM,
                    Ci=self.params.calcium_ci * u.mM,
                    temp=temp,
                ),
            )
        if self.config.toggles.cal_conc:
            self._paint_cdp_lva_ions()
        elif self.config.toggles.calva:
            self.cell.paint(
                self.regions["soma"] | self.regions["proxDend"] | self.regions["distDend"],
                mech.Ion(
                    "CalciumInitNernst",
                    name="ca_lva",
                    Co=self.params.calcium_co * u.mM,
                    Ci=self.params.calcium_ci * u.mM,
                    temp=temp,
                ),
            )

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
        # TODO: Correctness currently uses one paint rule per source section
        # because depth is section-specific. Later, lower same class/name ion
        # paints into one vectorized runtime instance with per-point params.
        active_regions = {"soma", "proxDend", "distDend"}
        for spec in self.native.specs:
            if spec.region not in active_regions:
                continue
            branch = self.morpho.branch(name=spec.branch_name)
            diam_um = _section_diam_um(spec)
            depth_um = _soma_shell_depth_um(diam_um, self.params.shell_thick) if spec.region == "soma" else _dend_shell_depth_um(diam_um, self.params.shell_thick)
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
        t = self.config.toggles
        q = self.params.qdeltat
        temp = u.celsius2kelvin(self.config.temperature_celsius)

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
            naf=(g_na_soma if t.naf else None),
            nap=(self.params.qconductance(self.params.g_na_p_soma) if t.nap else None),
            fkdr=(g_fkdr_soma if t.fkdr else None),
            skdr=(g_skdr_soma if t.skdr else None),
            sk=(g_sk_soma if t.sk else None),
            hcn=(g_h_soma if t.hcn else None),
            tnc=(g_tnc_soma if t.tnc else None),
            calva=(perm_lva_soma if t.calva else None),
            cahva=(perm_hva_soma if t.cahva else None),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "axHillock",
            naf=(2.0 * g_na_soma if t.naf else None),
            fkdr=(2.0 * g_fkdr_soma if t.fkdr else None),
            skdr=(2.0 * g_skdr_soma if t.skdr else None),
            tnc=(self.params.qconductance(self.params.g_tnc_ax) if t.tnc else None),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "axIniSeg",
            naf=(2.0 * g_na_soma if t.naf else None),
            fkdr=(2.0 * g_fkdr_soma if t.fkdr else None),
            skdr=(2.0 * g_skdr_soma if t.skdr else None),
            tnc=(self.params.qconductance(self.params.g_tnc_ax) if t.tnc else None),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "proxDend",
            naf=(0.4 * g_na_soma if t.naf else None),
            fkdr=(0.6 * g_fkdr_soma if t.fkdr else None),
            skdr=(0.6 * g_skdr_soma if t.skdr else None),
            sk=(0.3 * g_sk_soma if t.sk else None),
            hcn=(2.0 * g_h_soma if t.hcn else None),
            tnc=(0.2 * g_tnc_soma if t.tnc else None),
            calva=(2.0 * perm_lva_soma if t.calva else None),
            cahva=(perm_hva_soma / 1.5 if t.cahva else None),
            temp=temp,
            qdeltat=q,
        )
        self._paint_active_set(
            "distDend",
            sk=(0.3 * g_sk_soma if t.sk else None),
            hcn=(3.0 * g_h_soma if t.hcn else None),
            calva=(2.0 * perm_lva_soma if t.calva else None),
            cahva=(perm_hva_soma / 1.5 if t.cahva else None),
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
                mech.Channel(
                    "HCN_SU2015_DCN",
                    name=f"HCN_{suffix}",
                    g_max=hcn * (u.siemens / u.cm**2),
                    E=self.params.h_rev_pot * u.mV,
                ),
            )
        if tnc is not None:
            self.cell.paint(
                region,
                mech.Channel(
                    "IL",
                    name=f"TNC_{suffix}",
                    g_max=tnc * (u.siemens / u.cm**2),
                    E=self.params.tnc_rev_pot * u.mV,
                ),
            )
        if calva is not None:
            _paint_ca(self.cell, region, f"CaLVA_{suffix}", "CaLVA_SU2015_DCN", calva, "ca_lva", temp, qdeltat)
        if cahva is not None:
            _paint_ca(self.cell, region, f"CaHVA_{suffix}", "CaHVA_SU2015_DCN", cahva, "ca_hva", temp, qdeltat)

    def _collect_tables(self) -> None:
        if self.morpho is None or self.cell is None or self.native is None:
            raise RuntimeError("build() must run before collecting tables.")
        cv_counts = Counter(int(cv.branch_id) for cv in self.cell.cvs)
        branch_rows: list[dict[str, Any]] = []
        for branch in self.morpho.branches:
            branch_id = int(branch.index)
            region = self._source_region_by_branch_id[branch_id]
            enabled = enabled_region_mechanisms(self.config, region)
            diam_um = float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float))
            source_section = self.native.source_name_by_branch[branch.name]
            branch_rows.append(
                {
                    "branch_index": branch_id,
                    "branch_name": branch.name,
                    "source_section": source_section,
                    "branch_type": branch.type,
                    "source_region": region,
                    "source_local_index": self._source_local_index_by_branch_id[branch_id],
                    "diam_arc_mean_um": diam_um,
                    "cm_uF_cm2": self.params.cm / 100.0 if region == "axNode" else self.params.cm,
                    "ra_ohm_cm": self.params.ra,
                    "n_cv": int(cv_counts.get(branch_id, 0)),
                    **mechanism_flag_row(enabled),
                    "enabled_mechanisms": sorted(enabled),
                }
            )
        local_index_by_branch: dict[int, int] = defaultdict(int)
        compartment_rows: list[dict[str, Any]] = []
        for cv in self.cell.cvs:
            branch_id = int(cv.branch_id)
            branch = self.morpho.branch(index=branch_id)
            local_index = local_index_by_branch[branch_id]
            local_index_by_branch[branch_id] += 1
            compartment_rows.append(
                {
                    "compartment_index": int(len(compartment_rows)),
                    "branch_index": branch_id,
                    "branch_name": branch.name,
                    "source_section": self.native.source_name_by_branch[branch.name],
                    "branch_type": branch.type,
                    "source_region": self._source_region_by_branch_id[branch_id],
                    "source_local_index": self._source_local_index_by_branch_id[branch_id],
                    "local_index": int(local_index),
                    "cv_id": int(cv.id),
                    "prox": float(cv.prox),
                    "dist": float(cv.dist),
                }
            )
        self._branch_table = pd.DataFrame(branch_rows).reset_index(drop=True)
        self._compartment_table = pd.DataFrame(compartment_rows).reset_index(drop=True)

    def _build_summary(self) -> dict[str, Any]:
        bt = self._branch_table
        ct = self._compartment_table
        return {
            "backend": "braincell",
            "morph_path": str(DEFAULT_SOURCE_HOC_MORPH),
            "toggles": toggles_to_dict(self.config.toggles),
            "branch_counts": {
                "n_soma": int((bt["branch_type"] == "soma").sum()),
                "n_dend": int((bt["branch_type"] == "dendrite").sum()),
                "n_axon": int((bt["branch_type"] == "axon").sum()),
                "n_total": int(len(bt)),
            },
            "region_counts": bt["source_region"].value_counts().sort_index().to_dict(),
            "compartment_counts": {"n_total_cv": int(len(ct))},
            "enabled_mechanisms": {
                region: enabled_region_list(self.config, region)
                for region in ALL_REGION_LOGICAL_MECHANISMS
            },
            "native_hoc": True,
        }


def _paint_channel(
    cell: Cell,
    region,
    class_name: str,
    name: str,
    g_s_cm2: float,
    ion_name,
    qdeltat: float | None = None,
) -> None:
    kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
    if qdeltat is not None:
        kwargs["qdeltat"] = qdeltat
    cell.paint(
        region,
        mech.Channel(
            class_name,
            name=name,
            g_max=g_s_cm2 * (u.siemens / u.cm**2),
            **kwargs,
        ),
    )


def _paint_ca(
    cell: Cell,
    region,
    name: str,
    class_name: str,
    perm_cm_s: float,
    ion_name: str,
    temp,
    qdeltat: float,
) -> None:
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


def _section_diam_um(spec: Any) -> float:
    return 0.5 * (float(spec.prox.diam) + float(spec.dist.diam))


def _soma_shell_depth_um(diam_um: float, shell_thick_um: float) -> float:
    diam = float(diam_um)
    shell = float(shell_thick_um)
    return shell - 2.0 * shell**2 / diam + 4.0 * shell**3 / (3.0 * diam**2)


def _dend_shell_depth_um(diam_um: float, shell_thick_um: float) -> float:
    diam = float(diam_um)
    shell = float(shell_thick_um)
    return shell - shell**2 / diam
