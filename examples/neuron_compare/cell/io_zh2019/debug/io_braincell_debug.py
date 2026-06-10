from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import brainunit as u
import numpy as np
import pandas as pd
from braincell import Branch, Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, at, branch_in

from .io_parameters import (
    IOConfig,
    IOParameters,
    enabled_region_list,
    load_io19_params,
    toggles_to_dict,
)


@dataclass
class _BrainCellVoltageProbeBundle:
    soma_probe_name: str | None
    compartment_probe_names: list[str]


class IO:
    def __init__(
        self,
        params: IOParameters | None = None,
        config: IOConfig | None = None,
        *,
        ion_channel_update_order: str = "family",
    ):
        self.params = params if params is not None else load_io19_params()
        self.config = config if config is not None else IOConfig()
        self.ion_channel_update_order = ion_channel_update_order
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}
        self._branch_table = pd.DataFrame()
        self._compartment_table = pd.DataFrame()
        self._summary: dict[str, Any] = {}

    def build(self) -> IO:
        self.morpho = self._build_manual_morphology()
        self.cell = Cell(
            self.morpho,
            cv_policy=CVPerBranchList((int(self.params.soma.nseg),)),
            V_init=self.config.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order=self.ion_channel_update_order,
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
        rows: list[dict[str, Any]] = []
        if all_compartments:
            for index, cv in enumerate(self.cell.cvs):
                probe_name = f"cv_{int(cv.id)}_v"
                midpoint_x = 0.5 * (float(cv.prox) + float(cv.dist))
                self.cell.place(at("soma", midpoint_x), mech.StateProbe(name=probe_name))
                compartment_probe_names.append(probe_name)
                rows.append(
                    {
                        "compartment_index": int(index),
                        "branch_index": int(cv.branch_id),
                        "branch_name": "soma",
                        "branch_type": "soma",
                        "local_index": int(index),
                        "cv_id": int(cv.id),
                        "prox": float(cv.prox),
                        "dist": float(cv.dist),
                    }
                )
        return {
            "bundle": _BrainCellVoltageProbeBundle(
                soma_probe_name=soma_probe_name,
                compartment_probe_names=compartment_probe_names,
            ),
            "compartment_table": pd.DataFrame(rows),
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

    def _build_manual_morphology(self) -> Morphology:
        soma_radius_um = 0.5 * self.params.soma.diam_um
        soma = Branch.from_lengths(
            lengths=[self.params.soma.length_um] * u.um,
            radii=[soma_radius_um, soma_radius_um] * u.um,
            type="soma",
        )
        return Morphology.from_root(soma, name="soma")

    def _define_regions(self) -> None:
        if self.morpho is None:
            raise RuntimeError("Morphology must be created before defining regions.")
        self.regions = {"soma": branch_in("name", "soma")}

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=self.params.leak_e_mV * u.mV,
                membrane_capacitance=self.params.soma.cm_uF_cm2 * (u.uF / u.cm**2),
                axial_resistivity=self.params.soma.ra_ohm_cm * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(self.config.temperature_celsius),
            ),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        t = self.config.toggles
        if t.na:
            self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=self.params.na_e_mV * u.mV))
        if t.kdr:
            self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=self.params.k_e_mV * u.mV))

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        t = self.config.toggles
        soma = self.regions["soma"]
        if t.leak:
            self.cell.paint(
                soma,
                mech.Channel(
                    "IL",
                    name="IL_soma",
                    g_max=self.params.leak_g_S_cm2 * (u.siemens / u.cm**2),
                    E=self.params.leak_e_mV * u.mV,
                ),
            )
        if t.na:
            self.cell.paint(
                soma,
                mech.Channel(
                    "Na_ZH2019_IO",
                    name="Na_soma",
                    g_max=self.params.na_gbar_mS_cm2 * (u.mS / u.cm**2),
                ),
            )
        if t.kdr:
            self.cell.paint(
                soma,
                mech.Channel(
                    "Kdr_ZH2019_IO",
                    name="Kdr_soma",
                    g_max=self.params.kdr_gbar_mS_cm2 * (u.mS / u.cm**2),
                ),
            )
        if t.ca:
            self.cell.paint(
                soma,
                mech.Channel(
                    "Ca_ZH2019_IO",
                    name="Ca_soma",
                    g_max=self.params.ca_gbar_mS_cm2 * (u.mS / u.cm**2),
                    E=self.params.ca_e_mV * u.mV,
                    mMidV=self.params.ca_m_mid_mV * u.mV,
                ),
            )
        if t.hcn:
            self.cell.paint(
                soma,
                mech.Channel(
                    "HCN_ZH2019_IO",
                    name="HCN_soma",
                    g_max=self.params.hcn_gbar_mS_cm2 * (u.mS / u.cm**2),
                    E=self.params.hcn_e_mV * u.mV,
                ),
            )

    def _collect_tables(self) -> None:
        if self.cell is None or self.morpho is None:
            raise RuntimeError("build() must create cell before collecting tables.")
        branch = self.morpho.branch(name="soma")
        self._branch_table = pd.DataFrame(
            [
                {
                    "branch_index": int(branch.index),
                    "branch_name": branch.name,
                    "branch_type": branch.type,
                    "length_um": float(np.asarray(branch.length.to_decimal(u.um), dtype=float)),
                    "diam_um": float(np.asarray(branch.diam_arc_mean.to_decimal(u.um), dtype=float)),
                    "nseg": int(self.params.soma.nseg),
                }
            ]
        )
        self._compartment_table = pd.DataFrame(
            [
                {
                    "compartment_index": int(index),
                    "branch_index": int(cv.branch_id),
                    "branch_name": branch.name,
                    "branch_type": branch.type,
                    "local_index": int(index),
                    "cv_id": int(cv.id),
                    "prox": float(cv.prox),
                    "dist": float(cv.dist),
                }
                for index, cv in enumerate(self.cell.cvs)
            ]
        )

    def _build_summary(self) -> dict[str, Any]:
        return {
            "backend": "braincell",
            "manual_soma": True,
            "ion_channel_update_order": self.ion_channel_update_order,
            "toggles": toggles_to_dict(self.config.toggles),
            "branch_counts": {"n_soma": 1, "n_total": 1},
            "compartment_counts": {"n_total_nseg": int(len(self._compartment_table))},
            "enabled_mechanisms": {"soma": enabled_region_list(self.config, "soma")},
            "ion_status": {
                "na_enabled": bool(self.config.toggles.na),
                "k_enabled": bool(self.config.toggles.kdr),
                "ca_channel_enabled": bool(self.config.toggles.ca),
                "hcn_enabled": bool(self.config.toggles.hcn),
            },
        }
