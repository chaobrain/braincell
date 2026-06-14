from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, branch_in

from .parameters import (
    DEFAULT_MORPH_PATH,
    GoCCableParameters,
    GoCParameters,
    axon_region_name,
    dend_region_name,
    goc20_nseg_rule,
)


class GoC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GoCParameters | None = None,
        *,
        temperature_celsius: float = 34.0,
        v_init_mV: float = -65.0,
        pop_size=(),
        name: str | None = None,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.temperature_celsius = temperature_celsius
        self.v_init_mV = v_init_mV
        self.pop_size = pop_size
        self.name = name
        self.morph = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> GoC:
        self.morph = Morphology.from_asc(self.morph_path)
        cable = self.params.cable
        cv_counts = tuple(_goc20_cv_count(branch, cable) for branch in self.morph.branches)
        self.cell = Cell(
            self.morph,
            pop_size=self.pop_size,
            cv_policy=CVPerBranchList(cv_counts),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
            cache_ion_total_current=True,
            ion_channel_update_order="family",
            name=self.name,
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    def _define_regions(self) -> None:
        if self.morph is None:
            raise RuntimeError("Morphology must be loaded before defining regions.")
        type_counts = Counter(branch.type for branch in self.morph.branches)
        if type_counts.get("soma", 0) != 1 or type_counts.get("dendrite", 0) != 151 or type_counts.get("axon", 0) != 75:
            raise RuntimeError(f"Unexpected GoC branch counts from BrainCell ASC import: {dict(type_counts)}.")

        dend_branches = [branch for branch in self.morph.branches if branch.type == "dendrite"]
        axon_branches = [branch for branch in self.morph.branches if branch.type == "axon"]
        self.regions = {
            "soma": branch_in("name", "soma"),
            "dend_apical": _branch_names_region(
                branch.name for index, branch in enumerate(dend_branches) if dend_region_name(index) == "dend_apical"
            ),
            "dend_basal": _branch_names_region(
                branch.name for index, branch in enumerate(dend_branches) if dend_region_name(index) == "dend_basal"
            ),
            "axon_ais": _branch_names_region(
                branch.name for index, branch in enumerate(axon_branches) if axon_region_name(index) == "axon_ais"
            ),
            "axon_regular": _branch_names_region(
                branch.name for index, branch in enumerate(axon_branches) if axon_region_name(index) == "axon_regular"
            ),
        }

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        cable = self.params.cable
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=cable.leak_e_mV * u.mV,
                membrane_capacitance=cable.soma_cm_uF_cm2 * (u.uF / u.cm**2),
                axial_resistivity=cable.ra_ohm_cm * (u.ohm * u.cm),
            ),
        )
        self.cell.paint(
            self.regions["dend_apical"] | self.regions["dend_basal"],
            mech.CableProperty(
                resting_potential=cable.leak_e_mV * u.mV,
                membrane_capacitance=cable.dend_cm_uF_cm2 * (u.uF / u.cm**2),
                axial_resistivity=cable.ra_ohm_cm * (u.ohm * u.cm),
            ),
        )
        self.cell.paint(
            self.regions["axon_ais"] | self.regions["axon_regular"],
            mech.CableProperty(
                resting_potential=cable.leak_e_mV * u.mV,
                membrane_capacitance=cable.axon_cm_uF_cm2 * (u.uF / u.cm**2),
                axial_resistivity=cable.ra_ohm_cm * (u.ohm * u.cm),
            ),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        ion = self.params.ion
        temp = u.celsius2kelvin(self.temperature_celsius)
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=ion.na_e_mV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=ion.k_e_mV * u.mV))
        for region_name, pump in (
            ("soma", ion.cdp_pump_soma),
            ("dend_apical", ion.cdp_pump_dend_apical),
            ("dend_basal", ion.cdp_pump_dend_basal),
            ("axon_ais", ion.cdp_pump_axon),
            ("axon_regular", ion.cdp_pump_axon),
        ):
            self.cell.paint(
                self.regions[region_name],
                mech.Ion(
                    "CdpStC_MA2020_GoC",
                    name=_ca_name(region_name),
                    temp=temp,
                    Co=2.0 * u.mM,
                    Ci_initializer=45e-6 * u.mM,
                    TotalPump=pump * (u.mol / u.cm**2),
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        cable = self.params.cable
        ch = self.params.channel
        temp = u.celsius2kelvin(self.temperature_celsius)

        for region_name in ("soma", "dend_apical", "dend_basal", "axon_ais", "axon_regular"):
            leak_g = cable.leak_g_regular_axon_S_cm2 if region_name == "axon_regular" else cable.leak_g_default_S_cm2
            self.cell.paint(
                self.regions[region_name],
                mech.Channel("IL", name=f"IL_{region_name}", g_max=leak_g * (u.siemens / u.cm**2), E=cable.leak_e_mV * u.mV),
            )

        self._paint_region_channel("soma", "Nav1p6_MA2020_GoC", "Nav_soma", ch.soma.nav * (u.siemens / u.cm**2), "na", temp)
        self._paint_region_channel(
            "dend_apical", "Nav1p6_MA2020_GoC", "Nav_dend_apical", ch.dend_apical.nav * (u.siemens / u.cm**2), "na", temp
        )
        self._paint_region_channel(
            "dend_basal", "Nav1p6_MA2020_GoC", "Nav_dend_basal", ch.dend_basal.nav * (u.siemens / u.cm**2), "na", temp
        )
        self._paint_region_channel("axon_ais", "Nav1p6_MA2020_GoC", "Nav_ais", ch.axon_ais.nav * (u.siemens / u.cm**2), "na", temp)
        self._paint_region_channel(
            "axon_regular", "Nav1p6_MA2020_GoC", "Nav_axon_regular", ch.axon_regular.nav * (u.siemens / u.cm**2), "na", temp
        )

        self._paint_region_channel("soma", "Kv1p1_MA2020_GoC", "Kv1p1_soma", ch.soma.kv1p1 * (u.siemens / u.cm**2), "k", temp)
        self._paint_region_channel("soma", "Kv3p4_MA2020_GoC", "Kv3p4_soma", ch.soma.kv3p4 * (u.siemens / u.cm**2), "k", temp)
        self._paint_region_channel(
            "axon_regular", "Kv3p4_MA2020_GoC", "Kv3p4_axon_regular", ch.axon_regular.kv3p4 * (u.siemens / u.cm**2), "k", temp
        )
        self._paint_region_channel("soma", "Kv4p3_MA2020_GoC", "Kv4p3_soma", ch.soma.kv4p3 * (u.siemens / u.cm**2), "k", temp)
        self._paint_region_channel("axon_ais", "KM_MA2020_GoC", "KM_ais", ch.axon_ais.km * (u.siemens / u.cm**2), "k", temp)

        self._paint_region_channel(
            "soma", "Kca1p1_MA2020_GoC", "Kca1p1_soma", ch.soma.kca1p1 * (u.siemens / u.cm**2), {"k": "k", "ca": _ca_name("soma")}, temp
        )
        self._paint_region_channel(
            "dend_apical",
            "Kca1p1_MA2020_GoC",
            "Kca1p1_dend_apical",
            ch.dend_apical.kca1p1 * (u.siemens / u.cm**2),
            {"k": "k", "ca": _ca_name("dend_apical")},
            temp,
        )
        self._paint_region_channel(
            "dend_basal",
            "Kca1p1_MA2020_GoC",
            "Kca1p1_dend_basal",
            ch.dend_basal.kca1p1 * (u.siemens / u.cm**2),
            {"k": "k", "ca": _ca_name("dend_basal")},
            temp,
        )
        self._paint_region_channel(
            "axon_ais",
            "Kca1p1_MA2020_GoC",
            "Kca1p1_ais",
            ch.axon_ais.kca1p1 * (u.siemens / u.cm**2),
            {"k": "k", "ca": _ca_name("axon_ais")},
            temp,
        )
        self._paint_region_channel(
            "dend_apical",
            "Kca2p2_MA2020_GoC",
            "Kca2p2_dend_apical",
            ch.dend_apical.kca2p2 * (u.siemens / u.cm**2),
            {"k": "k", "ca": _ca_name("dend_apical")},
            temp,
        )
        self._paint_region_channel(
            "dend_basal",
            "Kca2p2_MA2020_GoC",
            "Kca2p2_dend_basal",
            ch.dend_basal.kca2p2 * (u.siemens / u.cm**2),
            {"k": "k", "ca": _ca_name("dend_basal")},
            temp,
        )
        self._paint_region_channel(
            "soma", "Kca3p1_MA2020_GoC", "Kca3p1_soma", ch.soma.kca3p1 * (u.siemens / u.cm**2), {"k": "k", "ca": _ca_name("soma")}, temp
        )

        self._paint_region_channel("soma", "CaHVA_MA2020_GoC", "CaHVA_soma", ch.soma.cahva * (u.siemens / u.cm**2), _ca_name("soma"), temp)
        self._paint_region_channel(
            "dend_basal", "CaHVA_MA2020_GoC", "CaHVA_dend_basal", ch.dend_basal.cahva * (u.siemens / u.cm**2), _ca_name("dend_basal"), temp
        )
        self._paint_region_channel(
            "axon_ais", "CaHVA_MA2020_GoC", "CaHVA_ais", ch.axon_ais.cahva * (u.siemens / u.cm**2), _ca_name("axon_ais"), temp
        )
        self._paint_region_channel(
            "dend_apical", "Cav2p3_MA2020_GoC", "Cav2p3_dend_apical", ch.dend_apical.cav2p3 * (u.siemens / u.cm**2), _ca_name("dend_apical"), temp
        )
        self._paint_region_channel(
            "soma",
            "Cav3p1_MA2020_GoC_Frozen",
            "Cav3p1_soma",
            ch.soma.cav3p1 * (u.cm / u.second),
            _ca_name("soma"),
            temp,
        )
        self._paint_region_channel(
            "dend_apical",
            "Cav3p1_MA2020_GoC_Frozen",
            "Cav3p1_dend_apical",
            ch.dend_apical.cav3p1 * (u.cm / u.second),
            _ca_name("dend_apical"),
            temp,
        )

        self.cell.paint(
            self.regions["axon_ais"],
            mech.Channel("HCN1_MA2020_GoC", name="HCN1_ais", g_max=ch.axon_ais.hcn1 * (u.siemens / u.cm**2), E=-20.0 * u.mV, temp=temp),
            mech.Channel("HCN2_MA2020_GoC", name="HCN2_ais", g_max=ch.axon_ais.hcn2 * (u.siemens / u.cm**2), E=-20.0 * u.mV, temp=temp),
        )

    def _paint_region_channel(
        self,
        region_name: str,
        channel_name: str,
        instance_name: str,
        g_max: Any,
        ion_name: str | dict[str, str],
        temp: Any,
    ) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        kwargs = {"ion_names": ion_name} if isinstance(ion_name, dict) else {"ion_name": ion_name}
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(channel_name, name=instance_name, g_max=g_max, temp=temp, **kwargs),
        )


def _ca_name(region_name: str) -> str:
    return f"ca_{region_name}"


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("GoC region cannot be empty.")
    region = branch_in("name", names[0])
    for name in names[1:]:
        region = region | branch_in("name", name)
    return region


def _goc20_cv_count(branch: Any, cable: GoCCableParameters) -> int:
    length_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    return goc20_nseg_rule(length_um, max_len_um=cable.cv_max_len_um)
