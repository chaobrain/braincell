from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np
import pandas as pd
from braincell import Axon, Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, at, branch_in

from .grc_full_parameters import (
    AA_SECTION_LEN_UM,
    CA_CI_INITIALIZER_M_M,
    CA_CO_M_M,
    CA_E_MV,
    DEFAULT_MORPH_PATH,
    EXPECTED_FULL_AA_COUNT,
    EXPECTED_FULL_AIS_COUNT,
    EXPECTED_FULL_DEND_COUNT,
    EXPECTED_FULL_HILOCK_COUNT,
    EXPECTED_FULL_PF1_COUNT,
    EXPECTED_FULL_PF2_COUNT,
    EXPECTED_FULL_SOMA_COUNT,
    FULL_REGION_LOGICAL_MECHANISMS,
    GrCFullConfig,
    GrCFullParameters,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    PF_SECTION_COUNT,
    PF_SECTION_LEN_UM,
    RA_OHM_CM,
    full_toggles_to_dict,
    grc20_nseg_rule,
)


@dataclass
class _BrainCellVoltageProbeBundle:
    soma_probe_name: str | None
    compartment_probe_names: list[str]


class GrCFull:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCFullParameters | None = None,
        config: GrCFullConfig | None = None,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.config = config if config is not None else GrCFullConfig()
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}
        self._source_region_by_branch_id: dict[int, str] = {}
        self._source_local_index_by_branch_id: dict[int, int] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> GrCFull:
        self.morpho = Morphology.from_asc(self.morph_path)
        self._attach_manual_morphology()
        cv_counts = tuple(_grc20_cv_count(branch) for branch in self.morpho.branches)
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

    def _attach_manual_morphology(self) -> None:
        if self.morpho is None:
            raise RuntimeError("Morphology must be loaded before attaching manual branches.")
        hilock = _axon_line("hilock", (0.0, -5.62232, 0.0), (0.0, -6.62232, 0.0), 1.5)
        self.morpho.attach(parent="soma", child_branch=hilock, child_name="hilock", parent_x=0.0)
        ais = _axon_line("ais", (0.0, -6.62232, 0.0), (0.0, -16.62232, 0.0), 0.7)
        self.morpho.attach(parent="hilock", child_branch=ais, child_name="ais", parent_x=1.0)

        len_initial_ais = -16.62232
        parent = "ais"
        for index in range(EXPECTED_FULL_AA_COUNT):
            name = f"aa_{index}"
            branch = _axon_line(
                name,
                (0.0, len_initial_ais, 0.0),
                (0.0, len_initial_ais - AA_SECTION_LEN_UM, 0.0),
                0.3,
            )
            self.morpho.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
            len_initial_ais -= AA_SECTION_LEN_UM

        len_initial_aa = -142.62232
        parent = f"aa_{EXPECTED_FULL_AA_COUNT - 1}"
        for index in range(PF_SECTION_COUNT):
            name = f"pf1_{index}"
            branch = _axon_line(
                name,
                (len_initial_aa, len_initial_aa, 0.0),
                (len_initial_aa + PF_SECTION_LEN_UM, len_initial_aa, 0.0),
                0.15,
            )
            self.morpho.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
            len_initial_aa += PF_SECTION_LEN_UM

        parent = f"aa_{EXPECTED_FULL_AA_COUNT - 1}"
        for index in range(PF_SECTION_COUNT):
            name = f"pf2_{index}"
            branch = _axon_line(
                name,
                (len_initial_aa, len_initial_aa, 0.0),
                (len_initial_aa - PF_SECTION_LEN_UM, len_initial_aa, 0.0),
                0.15,
            )
            self.morpho.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
            len_initial_aa -= PF_SECTION_LEN_UM

    def _define_regions(self) -> None:
        if self.morpho is None:
            raise RuntimeError("Morphology must be loaded before defining regions.")
        type_counts = Counter(branch.type for branch in self.morpho.branches)
        expected_axon = EXPECTED_FULL_HILOCK_COUNT + EXPECTED_FULL_AIS_COUNT + EXPECTED_FULL_AA_COUNT + EXPECTED_FULL_PF1_COUNT + EXPECTED_FULL_PF2_COUNT
        if (
            type_counts.get("soma", 0) != EXPECTED_FULL_SOMA_COUNT
            or type_counts.get("dendrite", 0) != EXPECTED_FULL_DEND_COUNT
            or type_counts.get("axon", 0) != expected_axon
        ):
            raise RuntimeError(f"Unexpected full GrC branch counts from BrainCell morphology: {dict(type_counts)}.")

        soma_branches = [branch for branch in self.morpho.branches if branch.type == "soma"]
        dend_branches = [branch for branch in self.morpho.branches if branch.type == "dendrite"]
        aa_names = [f"aa_{index}" for index in range(EXPECTED_FULL_AA_COUNT)]
        pf_names = [f"pf1_{index}" for index in range(PF_SECTION_COUNT)] + [f"pf2_{index}" for index in range(PF_SECTION_COUNT)]
        self.regions = {
            "soma": branch_in("name", soma_branches[0].name),
            "dend": _branch_names_region(branch.name for branch in dend_branches),
            "hilock": branch_in("name", "hilock"),
            "ais": branch_in("name", "ais"),
            "aa": _branch_names_region(aa_names),
            "pf": _branch_names_region(pf_names),
        }

        for branch in self.morpho.branches:
            name = branch.name
            if branch.type == "soma":
                region = "soma"
                local_index = 0
            elif branch.type == "dendrite":
                region = "dend"
                local_index = dend_branches.index(branch)
            elif name == "hilock":
                region = "hilock"
                local_index = 0
            elif name == "ais":
                region = "ais"
                local_index = 0
            elif name.startswith("aa_"):
                region = "aa"
                local_index = int(name.split("_", 1)[1])
            elif name.startswith("pf1_") or name.startswith("pf2_"):
                region = "pf"
                local_index = int(name.split("_", 1)[1])
            else:
                raise RuntimeError(f"Unexpected full GrC branch name {name!r}.")
            self._source_region_by_branch_id[int(branch.index)] = region
            self._source_local_index_by_branch_id[int(branch.index)] = int(local_index)

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        for region_name in FULL_REGION_LOGICAL_MECHANISMS:
            self.cell.paint(
                self.regions[region_name],
                mech.CableProperty(
                    resting_potential=LEAK_E_MV * u.mV,
                    membrane_capacitance=self.params.region(region_name).cm_uF_cm2 * (u.uF / u.cm**2),
                    axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
                ),
            )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        toggles = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        if _needs_na(self.config):
            self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=NA_E_MV * u.mV))
        if _needs_k(self.config):
            self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=K_E_MV * u.mV))
        if toggles.cdp:
            self.cell.paint(
                AllRegion(),
                mech.Ion(
                    "CdpCR_MA2020_GrC",
                    name="ca",
                    temp=temp,
                    Co=CA_CO_M_M * u.mM,
                    Ci_initializer=CA_CI_INITIALIZER_M_M * u.mM,
                ),
            )
        elif _needs_ca(self.config):
            self.cell.paint(
                AllRegion(),
                mech.Ion(
                    "CalciumFixed",
                    name="ca",
                    E=CA_E_MV * u.mV,
                    Ci=CA_CI_INITIALIZER_M_M * u.mM,
                    Co=CA_CO_M_M * u.mM,
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        t = self.config.toggles
        temp = u.celsius2kelvin(self.config.temperature_celsius)
        for region_name in FULL_REGION_LOGICAL_MECHANISMS:
            p = self.params.region(region_name)
            if t.leak and p.leak:
                self.cell.paint(
                    self.regions[region_name],
                    mech.Channel(
                        "IL",
                        name=f"IL_{region_name}",
                        g_max=p.leak * (u.siemens / u.cm**2),
                        E=LEAK_E_MV * u.mV,
                    ),
                )

        if t.nav:
            self._paint_region_channel("aa", "Nav_MA2020_GrC", "Nav_aa", self.params.aa.nav, "na", temp)
            self._paint_region_channel("pf", "Nav_MA2020_GrC", "Nav_pf", self.params.pf.nav, "na", temp)
        if t.nafhhf:
            self._paint_region_channel("hilock", "NaFHF_MA2020_GrC", "NaFHF_hilock", self.params.hilock.nafhhf, "na", temp)
            self._paint_region_channel("ais", "NaFHF_MA2020_GrC", "NaFHF_ais", self.params.ais.nafhhf, "na", temp)
        if t.kv3p4:
            self._paint_region_channel("soma", "Kv3p4_MA2020_GrC", "Kv3p4_soma", self.params.soma.kv3p4, "k", temp)
            self._paint_region_channel("hilock", "Kv3p4_MA2020_GrC", "Kv3p4_hilock", self.params.hilock.kv3p4, "k", temp)
            self._paint_region_channel("ais", "Kv3p4_MA2020_GrC", "Kv3p4_ais", self.params.ais.kv3p4, "k", temp)
            self._paint_region_channel("aa", "Kv3p4_MA2020_GrC", "Kv3p4_aa", self.params.aa.kv3p4, "k", temp)
            self._paint_region_channel("pf", "Kv3p4_MA2020_GrC", "Kv3p4_pf", self.params.pf.kv3p4, "k", temp)
        if t.kv4p3:
            self._paint_region_channel("soma", "Kv4p3_MA2020_GrC", "Kv4p3_soma", self.params.soma.kv4p3, "k", temp)
        if t.kir2p3:
            self._paint_region_channel("soma", "Kir2p3_MA2020_GrC", "Kir2p3_soma", self.params.soma.kir2p3, "k", temp)
        if t.cahva:
            self._paint_region_channel("soma", "CaHVA_MA2020_GrC", "CaHVA_soma", self.params.soma.cahva, "ca", temp)
            self._paint_region_channel("dend", "CaHVA_MA2020_GrC", "CaHVA_dend", self.params.dend.cahva, "ca", temp)
            self._paint_region_channel("hilock", "CaHVA_MA2020_GrC", "CaHVA_hilock", self.params.hilock.cahva, "ca", temp)
            self._paint_region_channel("ais", "CaHVA_MA2020_GrC", "CaHVA_ais", self.params.ais.cahva, "ca", temp)
            self._paint_region_channel("aa", "CaHVA_MA2020_GrC", "CaHVA_aa", self.params.aa.cahva, "ca", temp)
            self._paint_region_channel("pf", "CaHVA_MA2020_GrC", "CaHVA_pf", self.params.pf.cahva, "ca", temp)
        if t.kv1p1:
            self._paint_region_channel("soma", "Kv1p1_MA2020_GrC", "Kv1p1_soma", self.params.soma.kv1p1, "k", temp)
            self._paint_region_channel("dend", "Kv1p1_MA2020_GrC", "Kv1p1_dend", self.params.dend.kv1p1, "k", temp)
        if t.kv1p5:
            self._paint_region_channel("soma", "Kv1p5_MA2020_GrC", "Kv1p5_soma", self.params.soma.kv1p5, "k", temp)
        if t.kv2p2:
            self._paint_region_channel(
                "soma",
                "Kv2p2_0010_MA2020_GrC",
                "Kv2p2_soma",
                self.params.soma.kv2p2,
                "k",
                temp,
                include_temp=False,
            )
        if t.kca1p1:
            self._paint_region_channel(
                "dend",
                "Kca1p1_MA2020_GrC",
                "Kca1p1_dend",
                self.params.dend.kca1p1,
                {"k": "k", "ca": "ca"},
                temp,
            )
        if t.km:
            self._paint_region_channel("ais", "KM_MA2020_GrC", "KM_ais", self.params.ais.km, "k", temp)

    def _paint_region_channel(
        self,
        region_name: str,
        channel_name: str,
        instance_name: str,
        value: float,
        ion_name: str | dict[str, str],
        temp: Any,
        *,
        include_temp: bool = True,
    ) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
        if include_temp:
            kwargs["temp"] = temp
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(
                channel_name,
                name=instance_name,
                g_max=float(value) * (u.siemens / u.cm**2),
                **kwargs,
            ),
        )

    def _collect_tables(self) -> None:
        if self.morpho is None or self.cell is None:
            raise RuntimeError("build() must run before collecting tables.")
        cv_counts = Counter(int(cv.branch_id) for cv in self.cell.cvs)
        branch_rows: list[dict[str, Any]] = []
        for branch in self.morpho.branches:
            region = self._source_region_by_branch_id[int(branch.index)]
            enabled = _enabled_region_mechanisms(self.config, region)
            diam_um = float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float))
            branch_rows.append(
                {
                    "branch_index": int(branch.index),
                    "branch_name": branch.name,
                    "branch_type": branch.type,
                    "source_region": region,
                    "source_local_index": self._source_local_index_by_branch_id[int(branch.index)],
                    "diam_arc_mean_um": diam_um,
                    "cm_uF_cm2": self.params.region(region).cm_uF_cm2,
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
            "toggles": full_toggles_to_dict(self.config.toggles),
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
                for region in FULL_REGION_LOGICAL_MECHANISMS
            },
            "asc_only": False,
            "manual_pf": True,
        }


def _axon_line(name: str, start: tuple[float, float, float], end: tuple[float, float, float], diam_um: float) -> Any:
    del name
    return Axon.from_points(
        points=[start, end] * u.um,
        radii=[diam_um / 2.0, diam_um / 2.0] * u.um,
    )


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("GrC full region cannot be empty.")
    region = branch_in("name", names[0])
    for name in names[1:]:
        region = region | branch_in("name", name)
    return region


def _grc20_cv_count(branch: Any) -> int:
    length_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    return grc20_nseg_rule(length_um)


def _enabled_region_mechanisms(config: GrCFullConfig, region: str) -> set[str]:
    return set(_enabled_region_list(config, region))


def _enabled_region_list(config: GrCFullConfig, region: str) -> list[str]:
    return [name for name in FULL_REGION_LOGICAL_MECHANISMS[region] if getattr(config.toggles, name)]


def _mechanism_flag_row(enabled: set[str]) -> dict[str, bool]:
    names = sorted({name for values in FULL_REGION_LOGICAL_MECHANISMS.values() for name in values})
    return {f"has_{name}": bool(name in enabled) for name in names}


def _needs_na(config: GrCFullConfig) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for region in FULL_REGION_LOGICAL_MECHANISMS
        for name in ("nav", "nafhhf")
    )


def _needs_k(config: GrCFullConfig) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for region in FULL_REGION_LOGICAL_MECHANISMS
        for name in ("kv3p4", "kv4p3", "kir2p3", "kv1p1", "kv1p5", "kv2p2", "kca1p1", "km")
    )


def _needs_ca(config: GrCFullConfig) -> bool:
    return any(
        name in _enabled_region_list(config, region)
        for region in FULL_REGION_LOGICAL_MECHANISMS
        for name in ("cahva", "kca1p1", "cdp")
    )
