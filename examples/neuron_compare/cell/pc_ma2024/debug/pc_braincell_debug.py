from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np
import pandas as pd
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import MaxCVLen
from braincell.filter import AllRegion, at, branch_in, branch_range

from .pc_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    CA_E_MV,
    CDP_PUMP_DEND,
    CDP_PUMP_SOMA,
    CV_MAX_LEN_UM,
    DEFAULT_MORPH_PATH,
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
    toggles_to_dict,
)


@dataclass
class _BrainCellVoltageProbeBundle:
    soma_probe_name: str | None
    compartment_probe_names: list[str]


class PC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: PCParameters | None = None,
        config: PCConfig | None = None,
        *,
        frozen: bool = True,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else PCConfig()
        self.frozen = bool(frozen)
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> PC:
        self.morpho = Morphology.from_asc(self.morph_path)
        self.cell = Cell(
            self.morpho,
            cv_policy=MaxCVLen(CV_MAX_LEN_UM * u.um, keep_odd=True),
            V_init=self.config.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order="family", #"family" or "integration"
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
                if branch.type not in {"soma", "dendrite"}:
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
                        "branch_type": branch.type,
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
        dend_region = branch_in("type", "dendrite")
        self.regions = {
            "soma": branch_in("name", "soma"),
            "dend": dend_region,
            "thick_dend": dend_region
            & branch_range("diam_arc_mean", (THICK_DEND_DIAM_UM * u.um, None), closed="left"),
            "nav_dend": dend_region
            & branch_range("diam_arc_mean", (NAV_DEND_DIAM_UM * u.um, None), closed="left"),
        }

    def _paint_cable(self) -> None:
        if self.cell is None or self.morpho is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=LEAK_E_MV * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
            ),
        )
        for branch in self.morpho.branches:
            if branch.type not in {"soma", "dendrite"}:
                continue
            diam_um = float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float))
            cm_value = 2.0 if branch.type == "soma" else pc24_dend_cm(diam_um)
            self.cell.paint(
                branch_in("name", branch.name),
                mech.CableProperty(
                    resting_potential=LEAK_E_MV * u.mV,
                    membrane_capacitance=cm_value * (u.uF / u.cm**2),
                    axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
                ),
            )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        toggles = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        if toggles.nav:
            self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=NA_E_MV * u.mV))
        if any(
            getattr(toggles, name)
            for name in ("kv1p1", "kv1p5", "kv3p3", "kv3p4", "kv4p3", "kir2p3", "kca1p1", "kca2p2", "kca3p1")
        ):
            self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=K_E_MV * u.mV))
        needs_ca = any(
            getattr(toggles, name)
            for name in ("cav21", "cav31", "cav32", "cav33", "kca1p1", "kca2p2", "kca3p1")
        )
        if toggles.cdp:
            self.cell.paint(
                self.regions["soma"],
                mech.Ion(
                    "CdpCAM_MA2024_PC",
                    name="ca",
                    temp=temp,
                    Co=2.0 * u.mM,
                    Ci_initializer=45e-6 * u.mM,
                    TotalPump=CDP_PUMP_SOMA * (u.mol / u.cm**2),
                ),
            )
            self.cell.paint(
                self.regions["dend"],
                mech.Ion(
                    "CdpCAM_MA2024_PC",
                    name="ca",
                    temp=temp,
                    Co=2.0 * u.mM,
                    Ci_initializer=45e-6 * u.mM,
                    TotalPump=CDP_PUMP_DEND * (u.mol / u.cm**2),
                ),
            )
        elif needs_ca:
            self.cell.paint(
                AllRegion(),
                mech.Ion(
                    "CalciumFixed",
                    name="ca",
                    E=CA_E_MV * u.mV,
                    Ci=5.0e-05 * u.mM,
                    Co=2.0 * u.mM,
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        t = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        soma = self.regions["soma"]
        dend = self.regions["dend"]
        thick = self.regions["thick_dend"]
        nav_dend = self.regions["nav_dend"]
        frozen_suffix = "_Frozen" if self.frozen else ""
        if t.leak:
            self.cell.paint(soma, mech.Channel("IL", g_max=LEAK_G_SOMA_MS_CM2 * (u.mS / u.cm**2), E=LEAK_E_MV * u.mV))
            self.cell.paint(
                dend,
                mech.Channel("IL", name="IL_dend", g_max=LEAK_G_DEND_MS_CM2 * (u.mS / u.cm**2), E=LEAK_E_MV * u.mV),
            )
        if t.nav:
            self.cell.paint(
                soma,
                mech.Channel(
                    "Nav1p6_MA2024_PC",
                    g_max=self.params.nav_soma * (u.siemens / u.cm**2),
                    ion_name="na",
                    temp=temp,
                ),
            )
            self.cell.paint(
                nav_dend,
                mech.Channel(
                    "Nav1p6_MA2024_PC",
                    name="Nav_dend",
                    g_max=self.params.nav_dend * (u.siemens / u.cm**2),
                    ion_name="na",
                    temp=temp,
                ),
            )
        if t.kv1p1:
            self.cell.paint(
                soma,
                mech.Channel("Kv1p1_MA2024_PC", g_max=self.params.kv1p1_soma * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    "Kv1p1_MA2024_PC",
                    name="Kv1p1_dend_thick",
                    g_max=self.params.kv1p1_dend * (u.siemens / u.cm**2),
                    ion_name="k",
                    temp=temp,
                ),
            )
        if t.kv1p5:
            self.cell.paint(
                soma,
                mech.Channel("Kv1p5_MA2024_PC", g_max=self.params.kv1p5_soma * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    "Kv1p5_MA2024_PC",
                    name="Kv1p5_dend_thick",
                    g_max=self.params.kv1p5_dend * (u.siemens / u.cm**2),
                    ion_name="k",
                    temp=temp,
                ),
            )
        if t.kv3p3:
            self.cell.paint(
                dend,
                mech.Channel("Kv3p3_MA2024_PC", name="Kv3p3_dend", g_max=self.params.kv3p3_dend * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
        if t.kv3p4:
            self.cell.paint(
                soma,
                mech.Channel("Kv3p4_MA2024_PC", g_max=self.params.kv3p4_soma * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
        if t.kv4p3:
            self.cell.paint(
                dend,
                mech.Channel("Kv4p3_MA2024_PC", name="Kv4p3_dend", g_max=self.params.kv4p3_dend * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
        if t.kir2p3:
            self.cell.paint(
                soma,
                mech.Channel("Kir2p3_MA2024_PC", g_max=self.params.kir2p3_soma * (u.siemens / u.cm**2), ion_name="k", temp=temp),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    "Kir2p3_MA2024_PC",
                    name="Kir2p3_dend_thick",
                    g_max=self.params.kir2p3_dend * (u.siemens / u.cm**2),
                    ion_name="k",
                    temp=temp,
                ),
            )
        if t.kca1p1:
            self.cell.paint(
                soma,
                mech.Channel("Kca1p1_MA2024_PC", g_max=self.params.kca1p1_soma * (u.siemens / u.cm**2), ion_names={"k": "k", "ca": "ca"}, temp=temp),
            )
            self.cell.paint(
                dend,
                mech.Channel(
                    "Kca1p1_MA2024_PC",
                    name="Kca1p1_dend",
                    g_max=self.params.kca1p1_dend * (u.siemens / u.cm**2),
                    ion_names={"k": "k", "ca": "ca"},
                    temp=temp,
                ),
            )
        if t.kca2p2:
            self.cell.paint(
                soma,
                mech.Channel("Kca2p2_MA2024_PC", g_max=self.params.kca2p2_soma * (u.siemens / u.cm**2), ion_names={"k": "k", "ca": "ca"}, temp=temp),
            )
            self.cell.paint(
                dend,
                mech.Channel(
                    "Kca2p2_MA2024_PC",
                    name="Kca2p2_dend",
                    g_max=self.params.kca2p2_dend * (u.siemens / u.cm**2),
                    ion_names={"k": "k", "ca": "ca"},
                    temp=temp,
                ),
            )
        if t.kca3p1:
            self.cell.paint(
                soma,
                mech.Channel("Kca3p1_MA2024_PC", g_max=self.params.kca3p1_soma * (u.siemens / u.cm**2), ion_names={"k": "k", "ca": "ca"}, temp=temp),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    "Kca3p1_MA2024_PC",
                    name="Kca3p1_dend_thick",
                    g_max=self.params.kca3p1_dend * (u.siemens / u.cm**2),
                    ion_names={"k": "k", "ca": "ca"},
                    temp=temp,
                ),
            )
        if t.cav21:
            self.cell.paint(
                soma,
                mech.Channel(
                    f"Cav2p1_MA2024_PC{frozen_suffix}",
                    g_max=self.params.cav21_soma * (u.cm / u.second),
                    ion_name="ca",
                    temp=temp,
                ),
            )
            self.cell.paint(
                dend,
                mech.Channel(
                    f"Cav2p1_MA2024_PC{frozen_suffix}",
                    name="Cav2p1_dend_Frozen",
                    g_max=self.params.cav21_dend * (u.cm / u.second),
                    ion_name="ca",
                    temp=temp,
                ),
            )
        if t.cav31:
            self.cell.paint(
                soma,
                mech.Channel(
                    f"Cav3p1_MA2024_PC{frozen_suffix}",
                    g_max=self.params.cav31_soma * (u.cm / u.second),
                    ion_name="ca",
                    temp=temp,
                ),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    f"Cav3p1_MA2024_PC{frozen_suffix}",
                    name="Cav3p1_dend_Frozen",
                    g_max=self.params.cav31_dend * (u.cm / u.second),
                    ion_name="ca",
                    temp=temp,
                ),
            )
        if t.cav32:
            self.cell.paint(
                soma,
                mech.Channel(
                    "Cav3p2_MA2024_PC",
                    g_max=self.params.cav32_soma * (u.siemens / u.cm**2),
                    ion_name="ca",
                    temp=temp,
                ),
            )
            self.cell.paint(
                thick,
                mech.Channel(
                    "Cav3p2_MA2024_PC",
                    name="Cav3p2_dend_thick",
                    g_max=self.params.cav32_dend * (u.siemens / u.cm**2),
                    ion_name="ca",
                    temp=temp,
                ),
            )
        if t.cav33:
            self.cell.paint(
                soma,
                mech.Channel(
                    f"Cav3p3_MA2024_PC",
                    perm=self.params.cav33_soma_perm * (u.cm / u.second),
                    g_scale=self.params.cav33_g_scale,
                    ion_name="ca",
                    temp=temp,
                ),
            )
            self.cell.paint(
                dend,
                mech.Channel(
                    f"Cav3p3_MA2024_PC",
                    name="Cav3p3_dend_Frozen",
                    perm=self.params.cav33_dend_perm * (u.cm / u.second),
                    g_scale=self.params.cav33_g_scale,
                    ion_name="ca",
                    temp=temp,
                ),
            )
        if t.hcn1:
            self.cell.paint(
                soma,
                mech.Channel("HCN1_MA2024_PC", g_max=self.params.hcn1_soma * (u.siemens / u.cm**2), E=H_E_MV * u.mV, temp=temp),
            )
            self.cell.paint(
                dend,
                mech.Channel("HCN1_MA2024_PC", name="HCN1_dend", g_max=self.params.hcn1_dend * (u.siemens / u.cm**2), E=H_E_MV * u.mV, temp=temp),
            )

    def _collect_tables(self) -> None:
        if self.morpho is None or self.cell is None:
            raise RuntimeError("build() must run before collecting tables.")
        cv_counts = Counter(int(cv.branch_id) for cv in self.cell.cvs)
        branch_rows: list[dict[str, Any]] = []
        for branch in self.morpho.branches:
            if branch.type not in {"soma", "dendrite"}:
                continue
            diam_um = float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float))
            is_thick = branch.type == "dendrite" and diam_um >= THICK_DEND_DIAM_UM
            is_nav = branch.type == "dendrite" and diam_um >= NAV_DEND_DIAM_UM
            enabled = _enabled_branch_mechanisms(self.config, branch.type, is_thick=is_thick, is_nav=is_nav)
            branch_rows.append(
                {
                    "branch_index": int(branch.index),
                    "branch_name": branch.name,
                    "branch_type": branch.type,
                    "diam_arc_mean_um": diam_um,
                    "cm_uF_cm2": 2.0 if branch.type == "soma" else pc24_dend_cm(diam_um),
                    "n_cv": int(cv_counts.get(int(branch.index), 0)),
                    "is_thick_dend": bool(is_thick),
                    "is_nav_dend": bool(is_nav),
                    **_mechanism_flag_row(enabled),
                    "enabled_mechanisms": sorted(enabled),
                }
            )
        local_index_by_branch: dict[int, int] = defaultdict(int)
        compartment_rows: list[dict[str, Any]] = []
        for cv in self.cell.cvs:
            branch_id = int(cv.branch_id)
            branch = self.morpho.branch(index=branch_id)
            if branch.type not in {"soma", "dendrite"}:
                continue
            local_index = local_index_by_branch[branch_id]
            local_index_by_branch[branch_id] += 1
            compartment_rows.append(
                {
                    "compartment_index": int(len(compartment_rows)),
                    "branch_index": branch_id,
                    "branch_name": branch.name,
                    "branch_type": branch.type,
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
        toggles = self.config.toggles
        uses_k = any(
            getattr(toggles, name)
            for name in ("kv1p1", "kv1p5", "kv3p3", "kv3p4", "kv4p3", "kir2p3", "kca1p1", "kca2p2", "kca3p1")
        )
        needs_ca = any(
            getattr(toggles, name)
            for name in ("cav21", "cav31", "cav32", "cav33", "kca1p1", "kca2p2", "kca3p1")
        )
        if toggles.cdp:
            ca_backend = "CdpCAM_MA2024_PC"
        elif needs_ca:
            ca_backend = "CalciumFixed"
        else:
            ca_backend = None
        return {
            "backend": "braincell",
            "morph_path": str(self.morph_path),
            "indiv": int(self.params.indiv),
            "toggles": toggles_to_dict(toggles),
            "branch_counts": {
                "n_soma": int((bt["branch_type"] == "soma").sum()),
                "n_dend": int((bt["branch_type"] == "dendrite").sum()),
                "n_total": int(len(bt)),
            },
            "compartment_counts": {"n_total_cv": int(len(ct))},
            "threshold_hits": {
                "n_thick_dend": int(bt["is_thick_dend"].sum()),
                "n_nav_dend": int(bt["is_nav_dend"].sum()),
            },
            "enabled_mechanisms": {
                "soma": _enabled_region_list(self.config, "soma"),
                "dend_all": _enabled_region_list(self.config, "dend_all"),
                "dend_thick": _enabled_region_list(self.config, "dend_thick"),
                "dend_nav": _enabled_region_list(self.config, "dend_nav"),
            },
            "ion_status": {
                "na": "SodiumFixed" if toggles.nav else None,
                "k": "PotassiumFixed" if uses_k else None,
                "ca": ca_backend,
            },
        }


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in ALL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _enabled_region_list(config: PCConfig, region: str) -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _enabled_branch_mechanisms(config: PCConfig, branch_type: str, *, is_thick: bool, is_nav: bool) -> set[str]:
    enabled: set[str] = set()
    if branch_type == "soma":
        region_names = ("soma",)
    else:
        region_names = ("dend_all",)
        if is_thick:
            region_names += ("dend_thick",)
        if is_nav:
            region_names += ("dend_nav",)
    for region_name in region_names:
        for name in ALL_REGION_LOGICAL_MECHANISMS[region_name]:
            if getattr(config.toggles, name):
                enabled.add(name)
    return enabled
