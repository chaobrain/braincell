from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import brainunit as u
from braincell import Axon, Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, branch_in

from .grc_full_parameters import (
    AA_SECTION_LEN_UM,
    CA_CI_INITIALIZER_M_M,
    CA_CO_M_M,
    DEFAULT_MORPH_PATH,
    EXPECTED_FULL_AA_COUNT,
    EXPECTED_FULL_AIS_COUNT,
    EXPECTED_FULL_DEND_COUNT,
    EXPECTED_FULL_HILOCK_COUNT,
    EXPECTED_FULL_PF1_COUNT,
    EXPECTED_FULL_PF2_COUNT,
    EXPECTED_FULL_SOMA_COUNT,
    FULL_REGION_NAMES,
    GrCFullParameters,
    K_E_MV,
    LEAK_E_MV,
    NA_E_MV,
    PF_SECTION_COUNT,
    PF_SECTION_LEN_UM,
    RA_OHM_CM,
    grc20_nseg_rule,
)


class GrCFull:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: GrCFullParameters | None = None,
        *,
        temperature_celsius: float = 25.0,
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

    def build(self) -> GrCFull:
        self.morpho = Morphology.from_asc(self.morph_path)
        self._attach_manual_morphology()
        cv_counts = tuple(_grc20_cv_count(branch) for branch in self.morpho.branches)
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
            branch = _axon_line(name, (0.0, len_initial_ais, 0.0), (0.0, len_initial_ais - AA_SECTION_LEN_UM, 0.0), 0.3)
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
        expected_axon = (
            EXPECTED_FULL_HILOCK_COUNT
            + EXPECTED_FULL_AIS_COUNT
            + EXPECTED_FULL_AA_COUNT
            + EXPECTED_FULL_PF1_COUNT
            + EXPECTED_FULL_PF2_COUNT
        )
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

    def _paint_cable(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        for region_name in FULL_REGION_NAMES:
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
        temp = u.celsius2kelvin(self.temperature_celsius)
        self.cell.paint(self.regions["soma"], mech.Ion("SodiumInitNernst", name="na", temp=temp))
        for region_name in ("hilock", "ais", "aa", "pf"):
            self.cell.paint(self.regions[region_name], mech.Ion("SodiumFixed", name="na_fixed", E=NA_E_MV * u.mV))
        self.cell.paint(self.regions["soma"], mech.Ion("PotassiumInitNernst", name="k", temp=temp))
        for region_name in ("dend", "hilock", "ais", "aa", "pf"):
            self.cell.paint(self.regions[region_name], mech.Ion("PotassiumFixed", name="k_fixed", E=K_E_MV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("NonSpecificFixed", name="no"))
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

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        temp = u.celsius2kelvin(self.temperature_celsius)
        for region_name in FULL_REGION_NAMES:
            p = self.params.region(region_name)
            self.cell.paint(
                self.regions[region_name],
                mech.Channel("IL", name=f"IL_{region_name}", g_max=p.leak * (u.siemens / u.cm**2), E=LEAK_E_MV * u.mV),
            )

        self._paint_region_channel("aa", "Nav_MA2020_GrC", "Nav_aa", self.params.aa.nav, "na_fixed", temp)
        self._paint_region_channel("pf", "Nav_MA2020_GrC", "Nav_pf", self.params.pf.nav, "na_fixed", temp)
        self._paint_region_channel("hilock", "NaFHF_MA2020_GrC", "NaFHF_hilock", self.params.hilock.nafhhf, "na_fixed", temp)
        self._paint_region_channel("ais", "NaFHF_MA2020_GrC", "NaFHF_ais", self.params.ais.nafhhf, "na_fixed", temp)

        self._paint_region_channel("soma", "Kv3p4_MA2020_GrC", "Kv3p4_soma", self.params.soma.kv3p4, "k", temp)
        self._paint_region_channel("hilock", "Kv3p4_MA2020_GrC", "Kv3p4_hilock", self.params.hilock.kv3p4, "k_fixed", temp)
        self._paint_region_channel("ais", "Kv3p4_MA2020_GrC", "Kv3p4_ais", self.params.ais.kv3p4, "k_fixed", temp)
        self._paint_region_channel("aa", "Kv3p4_MA2020_GrC", "Kv3p4_aa", self.params.aa.kv3p4, "k_fixed", temp)
        self._paint_region_channel("pf", "Kv3p4_MA2020_GrC", "Kv3p4_pf", self.params.pf.kv3p4, "k_fixed", temp)
        self._paint_region_channel("soma", "Kv4p3_MA2020_GrC", "Kv4p3_soma", self.params.soma.kv4p3, "k", temp)
        self._paint_region_channel("soma", "Kir2p3_MA2020_GrC", "Kir2p3_soma", self.params.soma.kir2p3, "k", temp)

        self._paint_region_channel("soma", "CaHVA_MA2020_GrC", "CaHVA_soma", self.params.soma.cahva, "ca", temp)
        self._paint_region_channel("dend", "CaHVA_MA2020_GrC", "CaHVA_dend", self.params.dend.cahva, "ca", temp)
        self._paint_region_channel("hilock", "CaHVA_MA2020_GrC", "CaHVA_hilock", self.params.hilock.cahva, "ca", temp)
        self._paint_region_channel("ais", "CaHVA_MA2020_GrC", "CaHVA_ais", self.params.ais.cahva, "ca", temp)
        self._paint_region_channel("aa", "CaHVA_MA2020_GrC", "CaHVA_aa", self.params.aa.cahva, "ca", temp)
        self._paint_region_channel("pf", "CaHVA_MA2020_GrC", "CaHVA_pf", self.params.pf.cahva, "ca", temp)

        self._paint_region_channel("soma", "Kv1p1_MA2020_GrC", "Kv1p1_soma", self.params.soma.kv1p1, "k", temp)
        self._paint_region_channel("dend", "Kv1p1_MA2020_GrC", "Kv1p1_dend", self.params.dend.kv1p1, "k_fixed", temp)
        self._paint_region_channel("soma", "Kv1p5_MA2020_GrC", "Kv1p5_soma", self.params.soma.kv1p5, {"k": "k", "na": "na", "no": "no"}, temp)
        self._paint_region_channel(
            "soma",
            "Kv2p2_0010_MA2020_GrC",
            "Kv2p2_soma",
            self.params.soma.kv2p2,
            "k",
            temp,
            include_temp=False,
        )
        self._paint_region_channel("dend", "Kca1p1_MA2020_GrC", "Kca1p1_dend", self.params.dend.kca1p1, {"k": "k_fixed", "ca": "ca"}, temp)
        self._paint_region_channel("ais", "KM_MA2020_GrC", "KM_ais", self.params.ais.km, "k_fixed", temp)

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
            mech.Channel(channel_name, name=instance_name, g_max=float(value) * (u.siemens / u.cm**2), **kwargs),
        )


def _axon_line(name: str, prox_xyz: tuple[float, float, float], dist_xyz: tuple[float, float, float], diam_um: float) -> Axon:
    del name
    return Axon.from_points(
        points=[prox_xyz, dist_xyz] * u.um,
        radii=[diam_um / 2.0, diam_um / 2.0] * u.um,
    )


def _branch_names_region(names: Any) -> Any:
    names = tuple(names)
    if not names:
        raise ValueError("GrC region cannot be empty.")
    expr = branch_in("name", names[0])
    for name in names[1:]:
        expr = expr | branch_in("name", name)
    return expr


def _grc20_cv_count(branch: Any) -> int:
    length_um = float(branch.length.to_decimal(u.um))
    return grc20_nseg_rule(length_um)
