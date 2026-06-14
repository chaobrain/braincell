from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np
import pandas as pd
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, at, branch_in

from .bc_parameters import (
    ALL_REGION_LOGICAL_MECHANISMS,
    CA_E_MV,
    CDP_PUMP,
    CM_UF_CM2,
    DEFAULT_MORPH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    H_E_MV,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    BCConfig,
    BCParameters,
    axon_region_name,
    bc25_nseg_rule,
    toggles_to_dict,
)


@dataclass
class _BrainCellVoltageProbeBundle:
    soma_probe_name: str | None
    compartment_probe_names: list[str]


class BC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: BCParameters | None = None,
        config: BCConfig | None = None,
        *,
        frozen: bool = True,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else BCConfig()
        self.frozen = bool(frozen)
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}
        self._source_region_by_branch_id: dict[int, str] = {}
        self._source_local_index_by_branch_id: dict[int, int] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> BC:
        self.morpho = Morphology.from_asc(self.morph_path)
        cv_counts = tuple(_bc25_cv_count(branch) for branch in self.morpho.branches)
        self.cell = Cell(
            self.morpho,
            cv_policy=CVPerBranchList(cv_counts),
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
        if self.morpho is None:
            raise RuntimeError("Morphology must be loaded before defining regions.")
        type_counts = Counter(branch.type for branch in self.morpho.branches)
        if (
            type_counts.get("soma", 0) != EXPECTED_SOMA_COUNT
            or type_counts.get("dendrite", 0) != EXPECTED_DEND_COUNT
            or type_counts.get("axon", 0) != EXPECTED_AXON_COUNT
        ):
            raise RuntimeError(f"Unexpected BC branch counts from BrainCell ASC import: {dict(type_counts)}.")

        soma_branches = [branch for branch in self.morpho.branches if branch.type == "soma"]
        dend_branches = [branch for branch in self.morpho.branches if branch.type == "dendrite"]
        axon_branches = [branch for branch in self.morpho.branches if branch.type == "axon"]
        self.regions = {
            "soma": branch_in("name", soma_branches[0].name),
            "dend": _branch_names_region(branch.name for branch in dend_branches),
            "axon_ais": _branch_names_region(
                branch.name for index, branch in enumerate(axon_branches) if axon_region_name(index) == "axon_ais"
            ),
            "axon_regular": _branch_names_region(
                branch.name for index, branch in enumerate(axon_branches) if axon_region_name(index) == "axon_regular"
            ),
        }

        for branch in self.morpho.branches:
            if branch.type == "soma":
                region = "soma"
                local_index = 0
            elif branch.type == "dendrite":
                local_index = dend_branches.index(branch)
                region = "dend"
            elif branch.type == "axon":
                local_index = axon_branches.index(branch)
                region = axon_region_name(local_index)
            else:
                continue
            self._source_region_by_branch_id[int(branch.index)] = region
            self._source_local_index_by_branch_id[int(branch.index)] = int(local_index)

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=LEAK_E_MV * u.mV,
                membrane_capacitance=CM_UF_CM2 * (u.uF / u.cm**2),
                axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
            ),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        toggles = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        if toggles.nav1p1 or toggles.nav1p6:
            self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=NA_E_MV * u.mV))
        if any(
            getattr(toggles, name)
            for name in ("kir2p3", "kv1p1", "kv3p4", "kv4p3", "kca1p1", "kca2p2", "kca3p1")
        ):
            self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=K_E_MV * u.mV))
        needs_ca = any(
            getattr(toggles, name)
            for name in ("cav1p2", "cav1p3", "cav2p1", "cav3p2", "kca1p1", "kca2p2", "kca3p1")
        )
        if toggles.cdp:
            for region_name in ALL_REGION_LOGICAL_MECHANISMS:
                self.cell.paint(
                    self.regions[region_name],
                    mech.Ion(
                        "CdpStC_MA2025_BC",
                        name=self._ca_name(region_name),
                        temp=temp,
                        Co=2.0 * u.mM,
                        Ci_initializer=45e-6 * u.mM,
                        TotalPump=CDP_PUMP * (u.mol / u.cm**2),
                    ),
                )
        elif needs_ca:
            self.cell.paint(
                AllRegion(),
                mech.Ion(
                    "CalciumFixed",
                    name="ca",
                    E=CA_E_MV * u.mV,
                    Ci=45e-6 * u.mM,
                    Co=2.0 * u.mM,
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        t = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        cav2p1_class = "Cav2p1_MA2025_BC_Frozen" if self.frozen else "Cav2p1_MA2025_BC"

        for region_name in ALL_REGION_LOGICAL_MECHANISMS:
            p = self.params.region(region_name)
            if t.leak:
                self.cell.paint(
                    self.regions[region_name],
                    mech.Channel(
                        "IL",
                        name=f"IL_{region_name}",
                        g_max=p.leak * (u.siemens / u.cm**2),
                        E=LEAK_E_MV * u.mV,
                    ),
                )

        if t.nav1p1:
            self._paint_region_channel("soma", "Nav1p1_MA2025_BC", "Nav1p1_soma", self.params.soma.nav1p1, "na", temp)
        if t.nav1p6:
            self._paint_region_channel("axon_ais", "Nav1p6_MA2025_BC", "Nav1p6_ais", self.params.axon_ais.nav1p6, "na", temp)
            self._paint_region_channel(
                "axon_regular",
                "Nav1p6_MA2025_BC",
                "Nav1p6_axon_regular",
                self.params.axon_regular.nav1p6,
                "na",
                temp,
            )
        if t.cav3p2:
            self._paint_region_channel("soma", "Cav3p2_MA2025_BC", "Cav3p2_soma", self.params.soma.cav3p2, self._ca_name("soma"), temp)
            self._paint_region_channel("dend", "Cav3p2_MA2025_BC", "Cav3p2_dend", self.params.dend.cav3p2, self._ca_name("dend"), temp)
        if t.cav1p2:
            self._paint_region_channel("soma", "Cav1p2_MA2025_BC", "Cav1p2_soma", self.params.soma.cav1p2, self._ca_name("soma"), temp)
            self._paint_region_channel("dend", "Cav1p2_MA2025_BC", "Cav1p2_dend", self.params.dend.cav1p2, self._ca_name("dend"), temp)
        if t.cav1p3:
            self._paint_region_channel("soma", "Cav1p3_MA2025_BC", "Cav1p3_soma", self.params.soma.cav1p3, self._ca_name("soma"), temp)
            self._paint_region_channel("dend", "Cav1p3_MA2025_BC", "Cav1p3_dend", self.params.dend.cav1p3, self._ca_name("dend"), temp)
        if t.cav2p1:
            self._paint_region_channel(
                "axon_ais",
                cav2p1_class,
                "Cav2p1_ais",
                self.params.axon_ais.cav2p1,
                self._ca_name("axon_ais"),
                temp,
                permeability=True,
            )
            self._paint_region_channel(
                "axon_regular",
                cav2p1_class,
                "Cav2p1_axon_regular",
                self.params.axon_regular.cav2p1,
                self._ca_name("axon_regular"),
                temp,
                permeability=True,
            )
        if t.kir2p3:
            self._paint_region_channel("soma", "Kir2p3_MA2025_BC", "Kir2p3_soma", self.params.soma.kir2p3, "k", temp)
        if t.kv3p4:
            self._paint_region_channel("soma", "Kv3p4_MA2025_BC", "Kv3p4_soma", self.params.soma.kv3p4, "k", temp)
            self._paint_region_channel("axon_ais", "Kv3p4_MA2025_BC", "Kv3p4_ais", self.params.axon_ais.kv3p4, "k", temp)
            self._paint_region_channel(
                "axon_regular",
                "Kv3p4_MA2025_BC",
                "Kv3p4_axon_regular",
                self.params.axon_regular.kv3p4,
                "k",
                temp,
            )
        if t.kv4p3:
            self._paint_region_channel("soma", "Kv4p3_MA2025_BC", "Kv4p3_soma", self.params.soma.kv4p3, "k", temp)
            self._paint_region_channel("dend", "Kv4p3_MA2025_BC", "Kv4p3_dend", self.params.dend.kv4p3, "k", temp)
        if t.kv1p1:
            self._paint_region_channel(
                "axon_regular",
                "Kv1p1_MA2025_BC",
                "Kv1p1_axon_regular",
                self.params.axon_regular.kv1p1,
                "k",
                temp,
            )
        if t.kca3p1:
            self._paint_region_channel(
                "soma",
                "Kca3p1_MA2025_BC",
                "Kca3p1_soma",
                self.params.soma.kca3p1,
                {"k": "k", "ca": self._ca_name("soma")},
                temp,
            )
        if t.kca2p2:
            self._paint_region_channel(
                "dend",
                "Kca2p2_MA2025_BC",
                "Kca2p2_dend",
                self.params.dend.kca2p2,
                {"k": "k", "ca": self._ca_name("dend")},
                temp,
            )
        if t.kca1p1:
            self._paint_region_channel(
                "axon_ais",
                "Kca1p1_MA2025_BC",
                "Kca1p1_ais",
                self.params.axon_ais.kca1p1,
                {"k": "k", "ca": self._ca_name("axon_ais")},
                temp,
            )
            self._paint_region_channel(
                "axon_regular",
                "Kca1p1_MA2025_BC",
                "Kca1p1_axon_regular",
                self.params.axon_regular.kca1p1,
                {"k": "k", "ca": self._ca_name("axon_regular")},
                temp,
            )
        if t.hcn1:
            self._paint_hcn("soma", "HCN1_soma", self.params.soma.hcn1, temp)
            self._paint_hcn("axon_ais", "HCN1_ais", self.params.axon_ais.hcn1, temp)
            self._paint_hcn("axon_regular", "HCN1_axon_regular", self.params.axon_regular.hcn1, temp)

    def _ca_name(self, region_name: str) -> str:
        return f"ca_{region_name}" if self.config.toggles.cdp else "ca"

    def _paint_region_channel(
        self,
        region_name: str,
        channel_name: str,
        instance_name: str,
        value: float,
        ion_name: str | dict[str, str],
        temp: Any,
        *,
        permeability: bool = False,
    ) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        unit = u.cm / u.second if permeability else u.siemens / u.cm**2
        kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(
                channel_name,
                name=instance_name,
                g_max=float(value) * unit,
                temp=temp,
                **kwargs,
            ),
        )

    def _paint_hcn(self, region_name: str, instance_name: str, value: float, temp: Any) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(
                "HCN1_MA2025_BC",
                name=instance_name,
                g_max=float(value) * (u.siemens / u.cm**2),
                E=H_E_MV * u.mV,
                temp=temp,
            ),
        )

    def _collect_tables(self) -> None:
        if self.morpho is None or self.cell is None:
            raise RuntimeError("build() must run before collecting tables.")
        cv_counts = Counter(int(cv.branch_id) for cv in self.cell.cvs)
        branch_rows: list[dict[str, Any]] = []
        for branch in self.morpho.branches:
            if branch.type not in {"soma", "dendrite", "axon"}:
                continue
            region = self._source_region_by_branch_id[int(branch.index)]
            local_index = self._source_local_index_by_branch_id[int(branch.index)]
            enabled = _enabled_region_mechanisms(self.config, region)
            diam_um = float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float))
            branch_rows.append(
                {
                    "branch_index": int(branch.index),
                    "branch_name": branch.name,
                    "branch_type": branch.type,
                    "source_region": region,
                    "source_local_index": int(local_index),
                    "diam_arc_mean_um": diam_um,
                    "cm_uF_cm2": CM_UF_CM2,
                    "n_cv": int(cv_counts.get(int(branch.index), 0)),
                    **_mechanism_flag_row(enabled),
                    "enabled_mechanisms": sorted(enabled),
                }
            )
        local_index_by_branch: dict[int, int] = defaultdict(int)
        compartment_rows: list[dict[str, Any]] = []
        for cv in self.cell.cvs:
            branch_id = int(cv.branch_id)
            branch = self.morpho.branch(index=branch_id)
            if branch.type not in {"soma", "dendrite", "axon"}:
                continue
            local_index = local_index_by_branch[branch_id]
            local_index_by_branch[branch_id] += 1
            compartment_rows.append(
                {
                    "compartment_index": int(len(compartment_rows)),
                    "branch_index": branch_id,
                    "branch_name": branch.name,
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
            "morph_path": str(self.morph_path),
            "toggles": toggles_to_dict(self.config.toggles),
            "frozen": bool(self.frozen),
            "branch_counts": {
                "n_soma": int((bt["branch_type"] == "soma").sum()),
                "n_dend": int((bt["branch_type"] == "dendrite").sum()),
                "n_axon": int((bt["branch_type"] == "axon").sum()),
                "n_total": int(len(bt)),
            },
            "region_counts": bt["source_region"].value_counts().sort_index().to_dict(),
            "compartment_counts": {"n_total_cv": int(len(ct))},
            "enabled_mechanisms": {
                region: _enabled_region_list(self.config, region)
                for region in ALL_REGION_LOGICAL_MECHANISMS
            },
        }


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("BC region cannot be empty.")
    region = branch_in("name", names[0])
    for name in names[1:]:
        region = region | branch_in("name", name)
    return region


def _bc25_cv_count(branch: Any) -> int:
    length_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    return bc25_nseg_rule(length_um)


def _enabled_region_mechanisms(config: BCConfig, region: str) -> set[str]:
    return set(_enabled_region_list(config, region))


def _enabled_region_list(config: BCConfig, region: str) -> list[str]:
    return [name for name in ALL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in ALL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}
