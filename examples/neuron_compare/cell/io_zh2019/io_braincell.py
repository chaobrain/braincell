from __future__ import annotations

"""BrainCell version of the IO ZH2019 single-soma cell assembly.

This file mirrors ``io_neuron.py``.  IO currently has no formal morphology
import: both backends manually build one soma and install the same four
ZH2019 density mechanisms plus a passive leak baseline.
"""

from typing import Any

import brainunit as u
from braincell import Branch, Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, branch_in

from .parameters import IOParameters


class IO:
    def __init__(
        self,
        params: IOParameters | None = None,
        *,
        temperature_celsius: float = 36.0,
        v_init_mV: float = -65.0,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.params = params
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.morph = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> IO:
        self.morph = self._build_manual_morphology()
        self.cell = Cell(
            self.morph,
            cv_policy=CVPerBranchList((int(self.params.soma.nseg),)),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    def _build_manual_morphology(self) -> Morphology:
        soma_radius_um = 0.5 * self.params.soma.diam_um
        soma = Branch.from_lengths(
            lengths=[self.params.soma.length_um] * u.um,
            radii=[soma_radius_um, soma_radius_um] * u.um,
            type="soma",
        )
        return Morphology.from_root(soma, name="soma")

    def _define_regions(self) -> None:
        self.regions = {"soma": branch_in("name", "soma")}

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
                temperature=u.celsius2kelvin(self.temperature_celsius),
            ),
        )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        ion = self.params.ion
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=ion.na_e_mV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=ion.k_e_mV * u.mV))

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        ch = self.params.channel
        cable = self.params.cable
        soma = self.regions["soma"]
        self.cell.paint(
            soma,
            mech.Channel(
                "IL",
                name="IL_soma",
                g_max=cable.leak_g_S_cm2 * (u.siemens / u.cm**2),
                E=cable.leak_e_mV * u.mV,
            ),
            mech.Channel(
                "Na_ZH2019_IO",
                name="Na_soma",
                g_max=ch.na_gbar_mS_cm2 * (u.mS / u.cm**2),
            ),
            mech.Channel(
                "Kdr_ZH2019_IO",
                name="Kdr_soma",
                g_max=ch.kdr_gbar_mS_cm2 * (u.mS / u.cm**2),
            ),
            mech.Channel(
                "Ca_ZH2019_IO",
                name="Ca_soma",
                g_max=ch.ca_gbar_mS_cm2 * (u.mS / u.cm**2),
                E=ch.ca_e_mV * u.mV,
                mMidV=ch.ca_m_mid_mV * u.mV,
            ),
            mech.Channel(
                "HCN_ZH2019_IO",
                name="HCN_soma",
                g_max=ch.hcn_gbar_mS_cm2 * (u.mS / u.cm**2),
                E=ch.hcn_e_mV * u.mV,
            ),
        )
