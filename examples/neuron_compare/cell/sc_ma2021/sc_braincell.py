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
    DEND_CM_UF_CM2,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    H_E_MV,
    K_E_AXON_MV,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    RA_OHM_CM,
    SCParameters,
    axon_region_name,
    dend_region_name,
    sc21_nseg_rule,
)

REGION_NAMES = ("soma", "dendprox", "denddist", "axon_ais", "axon_regular")


class SC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: SCParameters | None = None,
        *,
        temperature_celsius: float = 32.0,
        v_init_mV: float = -65.0,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.morpho = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> SC:
        self.morpho = Morphology.from_asc(self.morph_path)
        cv_counts = tuple(_sc21_cv_count(branch) for branch in self.morpho.branches)
        self.cell = Cell(
            self.morpho,
            cv_policy=CVPerBranchList(cv_counts),
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
        if self.morpho is None:
            raise RuntimeError("Morphology must be loaded before defining regions.")
        type_counts = Counter(branch.type for branch in self.morpho.branches)
        if (
            type_counts.get("soma", 0) != EXPECTED_SOMA_COUNT
            or type_counts.get("dendrite", 0) != EXPECTED_DEND_COUNT
            or type_counts.get("axon", 0) != EXPECTED_AXON_COUNT
        ):
            raise RuntimeError(f"Unexpected SC branch counts from BrainCell ASC import: {dict(type_counts)}.")

        soma_branches = [branch for branch in self.morpho.branches if branch.type == "soma"]
        dend_branches = [branch for branch in self.morpho.branches if branch.type == "dendrite"]
        axon_branches = [branch for branch in self.morpho.branches if branch.type == "axon"]
        self.regions = {
            "soma": branch_in("name", soma_branches[0].name),
            "dendprox": _branch_names_region(
                branch.name for index, branch in enumerate(dend_branches) if dend_region_name(index) == "dendprox"
            ),
            "denddist": _branch_names_region(
                branch.name for index, branch in enumerate(dend_branches) if dend_region_name(index) == "denddist"
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
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=LEAK_E_MV * u.mV,
                membrane_capacitance=1.0 * (u.uF / u.cm**2),
                axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
            ),
        )
        self.cell.paint(
            self.regions["dendprox"] | self.regions["denddist"],
            mech.CableProperty(
                resting_potential=LEAK_E_MV * u.mV,
                membrane_capacitance=DEND_CM_UF_CM2 * (u.uF / u.cm**2),
                axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
            ),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        temp = u.celsius2kelvin(self.temperature_celsius)

        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=NA_E_MV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=K_E_MV * u.mV))
        self.cell.paint(
            self.regions["axon_ais"] | self.regions["axon_regular"],
            mech.Ion("PotassiumFixed", name="k_axon", E=K_E_AXON_MV * u.mV),
        )

        for region_name in REGION_NAMES:
            self.cell.paint(
                self.regions[region_name],
                mech.Ion(
                    "CdpStC_RI2021_SC",
                    name=_ca_name(region_name),
                    temp=temp,
                    Co=2.0 * u.mM,
                    Ci_initializer=45e-6 * u.mM,
                    TotalPump=self.params.region(region_name).cdp_pump * (u.mol / u.cm**2),
                ),
            )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        temp = u.celsius2kelvin(self.temperature_celsius)
        cav2p1_class = "Cav2p1_RI2021_SC_Frozen"

        for region_name in REGION_NAMES:
            p = self.params.region(region_name)
            self.cell.paint(
                self.regions[region_name],
                mech.Channel(
                    "IL",
                    name=f"IL_{region_name}",
                    g_max=p.leak * (u.siemens / u.cm**2),
                    E=LEAK_E_MV * u.mV,
                ),
            )

        self._paint_region_channel("soma", "Nav1p1_RI2021_SC", "Nav1p1_soma", self.params.soma.nav1p1, "na", temp)

        self._paint_region_channel("axon_ais", "Nav1p6_RI2021_SC", "Nav1p6_ais", self.params.axon_ais.nav1p6, "na", temp)
        self._paint_region_channel(
            "axon_regular",
            "Nav1p6_RI2021_SC",
            "Nav1p6_axon_regular",
            self.params.axon_regular.nav1p6,
            "na",
            temp,
        )

        self._paint_region_channel("soma", cav2p1_class, "Cav2p1_soma", self.params.soma.cav2p1, _ca_name("soma"), temp, permeability=True)
        self._paint_region_channel(
            "dendprox",
            cav2p1_class,
            "Cav2p1_dendprox",
            self.params.dendprox.cav2p1,
            _ca_name("dendprox"),
            temp,
            permeability=True,
        )
        self._paint_region_channel(
            "denddist",
            cav2p1_class,
            "Cav2p1_denddist",
            self.params.denddist.cav2p1,
            _ca_name("denddist"),
            temp,
            permeability=True,
        )

        self._paint_region_channel("soma", "Cav3p2_RI2021_SC", "Cav3p2_soma", self.params.soma.cav3p2, _ca_name("soma"), temp)
        self._paint_region_channel(
            "dendprox",
            "Cav3p2_RI2021_SC",
            "Cav3p2_dendprox",
            self.params.dendprox.cav3p2,
            _ca_name("dendprox"),
            temp,
        )

        self._paint_cav3p3("soma", "Cav3p3_soma", self.params.soma, temp)
        self._paint_cav3p3("dendprox", "Cav3p3_dendprox", self.params.dendprox, temp)

        self._paint_region_channel("soma", "Kir2p3_RI2021_SC", "Kir2p3_soma", self.params.soma.kir2p3, "k", temp)

        self._paint_region_channel("soma", "Kv1p1_RI2021_SC", "Kv1p1_soma", self.params.soma.kv1p1, "k", temp)
        self._paint_region_channel("dendprox", "Kv1p1_RI2021_SC", "Kv1p1_dendprox", self.params.dendprox.kv1p1, "k", temp)
        self._paint_region_channel("denddist", "Kv1p1_RI2021_SC", "Kv1p1_denddist", self.params.denddist.kv1p1, "k", temp)
        self._paint_region_channel("axon_ais", "Kv1p1_RI2021_SC", "Kv1p1_ais", self.params.axon_ais.kv1p1, "k_axon", temp)
        self._paint_region_channel(
            "axon_regular",
            "Kv1p1_RI2021_SC",
            "Kv1p1_axon_regular",
            self.params.axon_regular.kv1p1,
            "k_axon",
            temp,
        )

        self._paint_region_channel("soma", "Kv3p4_RI2021_SC", "Kv3p4_soma", self.params.soma.kv3p4, "k", temp)
        self._paint_region_channel("axon_ais", "Kv3p4_RI2021_SC", "Kv3p4_ais", self.params.axon_ais.kv3p4, "k_axon", temp)
        self._paint_region_channel(
            "axon_regular",
            "Kv3p4_RI2021_SC",
            "Kv3p4_axon_regular",
            self.params.axon_regular.kv3p4,
            "k_axon",
            temp,
        )

        self._paint_region_channel("soma", "Kv4p3_RI2021_SC", "Kv4p3_soma", self.params.soma.kv4p3, "k", temp)
        self._paint_region_channel("dendprox", "Kv4p3_RI2021_SC", "Kv4p3_dendprox", self.params.dendprox.kv4p3, "k", temp)

        self._paint_region_channel("axon_ais", "KM_RI2021_SC", "KM_ais", self.params.axon_ais.km, "k_axon", temp)

        self._paint_kca("soma", "Kca1p1_RI2021_SC", "Kca1p1_soma", self.params.soma.kca1p1, "k", temp)
        self._paint_kca("dendprox", "Kca1p1_RI2021_SC", "Kca1p1_dendprox", self.params.dendprox.kca1p1, "k", temp)
        self._paint_kca("denddist", "Kca1p1_RI2021_SC", "Kca1p1_denddist", self.params.denddist.kca1p1, "k", temp)

        self._paint_kca("soma", "Kca2p2_RI2021_SC", "Kca2p2_soma", self.params.soma.kca2p2, "k", temp)
        self._paint_kca("dendprox", "Kca2p2_RI2021_SC", "Kca2p2_dendprox", self.params.dendprox.kca2p2, "k", temp)
        self._paint_kca("denddist", "Kca2p2_RI2021_SC", "Kca2p2_denddist", self.params.denddist.kca2p2, "k", temp)

        self._paint_hcn("soma", "HCN1_soma", self.params.soma.hcn1, temp)
        self._paint_hcn("axon_ais", "HCN1_ais", self.params.axon_ais.hcn1, temp)
        self._paint_hcn("axon_regular", "HCN1_axon_regular", self.params.axon_regular.hcn1, temp)

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

    def _paint_cav3p3(self, region_name: str, instance_name: str, params: Any, temp: Any) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(
                "Cav3p3_RI2021_SC",
                name=instance_name,
                perm=float(params.cav3p3) * (u.cm / u.second),
                g_scale=float(params.cav3p3_g_scale),
                temp=temp,
                ion_name=_ca_name(region_name),
            ),
        )

    def _paint_kca(
        self,
        region_name: str,
        channel_name: str,
        instance_name: str,
        value: float,
        k_name: str,
        temp: Any,
    ) -> None:
        self._paint_region_channel(
            region_name,
            channel_name,
            instance_name,
            value,
            {"k": k_name, "ca": _ca_name(region_name)},
            temp,
        )

    def _paint_hcn(self, region_name: str, instance_name: str, value: float, temp: Any) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        self.cell.paint(
            self.regions[region_name],
            mech.Channel(
                "HCN1_RI2021_SC",
                name=instance_name,
                g_max=float(value) * (u.siemens / u.cm**2),
                E=H_E_MV * u.mV,
                temp=temp,
            ),
        )


def _ca_name(region_name: str) -> str:
    return f"ca_{region_name}"


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("SC region cannot be empty.")
    region = branch_in("name", names[0])
    for name in names[1:]:
        region = region | branch_in("name", name)
    return region


def _sc21_cv_count(branch: Any) -> int:
    length_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    return sc21_nseg_rule(length_um)
