from __future__ import annotations

"""BrainCell version of the PC24 Purkinje cell assembly.

This file is intentionally structured to mirror ``pc_neuron.py``.  A useful way
to read the translation is:

- NEURON ``Import3d_*`` morphology loading -> ``Morphology.from_asc``.
- NEURON ``sec.nseg = ...`` discretization -> ``CVPerBranchList(...)``.
- NEURON ``sec.Ra`` / ``sec.cm`` -> ``mech.CableProperty`` painted on regions.
- NEURON ``insert pas`` -> ``mech.Channel("IL", ...)``.
- NEURON ``insert X`` plus ``sec.gbar_X = value`` -> ``mech.Channel("X", ...)``.
- NEURON ion reversal fields and calcium pump mechanism -> ``mech.Ion(...)``.

One important ordering difference: in NEURON, section ion fields such as
``ena``/``ek``/``eca`` only exist after a mechanism using that ion has been
inserted, so the script often does ``insert(...)`` before setting the reversal
field.  BrainCell declares ions first with ``paint(Ion(...))``; channels painted
later bind to those ions automatically.

The goal is not to hide the correspondence behind helper abstractions.  The
paint blocks below stay close to the original soma/dendrite/thick-dendrite
sections so that new readers can compare this file with ``pc_neuron.py`` line by
line.
"""

import math
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np
from braincell import Cell, Morphology, mech
from braincell._discretization.policy import CVPerBranchList
from braincell.filter import AllRegion, branch_in, branch_range

from .parameters import (
    DEFAULT_MORPH_PATH,
    PCCableParameters,
    PCParameters,
    pc24_nseg_rule,
)


def pc24_dend_cm(diam_arc_mean_um: float, cable: PCCableParameters) -> float:
    """Match the dendritic ``sec.cm`` rule from ``pc_neuron.py``.

    In NEURON the dendrite loop does:

    ``dend.cm = 11.510294 * exp(-1.376463 * dend.diam) + 2.120503``

    and then overwrites thick dendrites with the soma capacitance.  BrainCell
    uses the branch arc-mean diameter for the same role.
    """
    diam = float(diam_arc_mean_um)
    if diam >= cable.thick_dend_diam_um:
        return cable.soma_cm_uF_cm2
    return 11.510294 * math.exp(-1.376463 * diam) + 2.120503


class PC:
    def __init__(
        self,
        morph_path: Path | str = DEFAULT_MORPH_PATH,
        params: PCParameters | None = None,
        *,
        temperature_celsius: float = 36.0,
        v_init_mV: float = -65.0,
        pop_size=(),
        name: str | None = None,
    ):
        if params is None:
            raise ValueError("params is required.")
        self.morph_path = Path(morph_path)
        self.params = params
        self.temperature_celsius = float(temperature_celsius)
        self.v_init_mV = float(v_init_mV)
        self.pop_size = pop_size
        self.name = name
        self.morph = None
        self.cell = None
        self.regions: dict[str, Any] = {}

    def build(self) -> PC:
        # NEURON path:
        #   reader = h.Import3d_Neurolucida3()
        #   reader.input(...)
        #   h.Import3d_GUI(reader, 0).instantiate(self)
        # BrainCell path: read the same ASC morphology into a Morphology tree.
        self.morph = Morphology.from_asc(self.morph_path)
        cable = self.params.cable
        cv_counts = tuple(_pc24_cv_count(branch, cable) for branch in self.morph.branches)

        # ``CVPerBranchList`` is the BrainCell counterpart of the repeated
        # NEURON rule:
        #
        #   sec.nseg = 1 + 2 * int(sec.L / CV_MAX_LEN_UM)
        self.cell = Cell(
            self.morph,
            pop_size=self.pop_size,
            cv_policy=CVPerBranchList(cv_counts),
            V_init=self.v_init_mV * u.mV,
            solver="staggered",
            name=self.name,
        )
        self._define_regions()
        self._paint_cable()
        self._paint_ions()
        self._paint_channels()
        return self

    def _define_regions(self) -> None:
        # Regions replace NEURON's explicit section lists and ``if dend.diam``
        # branches.  The names are used only inside this file to keep the paint
        # blocks aligned with the original soma/dendrite code.
        cable = self.params.cable
        dend_region = branch_in("type", "dendrite")
        self.regions = {
            "soma": branch_in("type", "soma"),
            "dend": dend_region,
            # NEURON:
            #   if dend.diam >= THICK_DEND_DIAM_UM:
            #       insert Kv1.1/Kv1.5/Kir/Cav3.1/Cav3.2/Kca3.1
            "thick_dend": dend_region
            & branch_range("diam_arc_mean", (cable.thick_dend_diam_um * u.um, None), closed="left"),
            # NEURON nested inside the thick-dendrite branch:
            #   if dend.diam >= NAV_DEND_DIAM_UM:
            #       insert Nav1.6
            "nav_dend": dend_region & branch_range("diam_arc_mean", (cable.nav_dend_diam_um * u.um, None), closed="left"),
        }

    def _paint_cable(self) -> None:
        if self.cell is None or self.morph is None:
            raise RuntimeError("Cell must be created before painting cable properties.")
        cable = self.params.cable

        # First paint a harmless global default.  The branch-specific cable
        # rules below overwrite it on every PC soma/dendrite branch, mirroring
        # the explicit ``soma.Ra/cm`` and ``dend.Ra/cm`` assignments in
        # ``pc_neuron.py``.
        self.cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=cable.leak_e_mV * u.mV,
                membrane_capacitance=cable.soma_cm_uF_cm2 * (u.uF / u.cm**2),
                axial_resistivity=cable.ra_ohm_cm * (u.ohm * u.cm),
            ),
        )

        # NEURON assigns capacitance per section:
        #
        #   soma.cm = SOMA_CM_UF_CM2
        #   dend.cm = pc24_dend_cm(dend.diam)
        #   if dend.diam >= THICK_DEND_DIAM_UM:
        #       dend.cm = SOMA_CM_UF_CM2
        #
        # BrainCell currently paints scalar CableProperty values, so this stays
        # as one branch-level paint per morphology branch.  A future callable
        # CableProperty API could replace this loop with one expression over a
        # spatial context.
        for branch in self.morph.branches:
            diam_um = branch.diam_arc_mean.to_decimal(u.um)
            cm_value = cable.soma_cm_uF_cm2 if branch.type == "soma" else pc24_dend_cm(diam_um, cable)
            self.cell.paint(
                branch_in("name", branch.name),
                mech.CableProperty(
                    resting_potential=cable.leak_e_mV * u.mV,
                    membrane_capacitance=cm_value * (u.uF / u.cm**2),
                    axial_resistivity=cable.ra_ohm_cm * (u.ohm * u.cm),
                ),
            )

    def _paint_ions(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting ions.")
        ion = self.params.ion
        temp = u.celsius2kelvin(self.temperature_celsius)

        # NEURON stores Na/K reversal potentials directly on sections:
        #
        #   soma.ena = NA_E_MV
        #   dend.ena = NA_E_MV  # only where Nav is inserted
        #   soma.ek = dend.ek = K_E_MV
        #
        # In NEURON these fields are normally written after the corresponding
        # mechanisms have been inserted, because the insert creates the ion
        # variables on the section.  BrainCell reverses that assembly order:
        # paint the ion first, then paint channels that bind to it.
        #
        # BrainCell declares one fixed Na ion and one fixed K ion over the
        # whole cell.  Channels can omit ``ion_name`` because each family has a
        # single candidate ion here.
        self.cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=ion.na_e_mV * u.mV))
        self.cell.paint(AllRegion(), mech.Ion("PotassiumFixed", name="k", E=ion.k_e_mV * u.mV))

        # NEURON inserts CdpCAM_MA24_PC on soma and dendrites, then sets eca and
        # a region-specific pump density:
        #
        #   sec.insert("CdpCAM_MA24_PC")
        #   sec.TotalPump_CdpCAM_MA24_PC = CDP_PUMP_*
        #   sec.eca = CA_E_MV
        #
        # In BrainCell the calcium dynamics and reversal state live in the Ca
        # ion declaration.  Painting the same ion name ("ca") on soma and dend
        # produces one runtime Ca ion with per-region parameters.
        #
        # ``sec.eca = CA_E_MV`` is not translated as a fixed constant here:
        # once the dynamic calcium pump is inserted, eca follows cai/cao through
        # the calcium ion dynamics, so that fixed assignment is not the active
        # source of the calcium reversal potential in this comparison.
        self.cell.paint(
            self.regions["soma"],
            mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca",
                temp=temp,
                Co=2.0 * u.mM,
                Ci_initializer=45e-6 * u.mM,
                TotalPump=ion.cdp_pump_soma * (u.mol / u.cm**2),
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
                TotalPump=ion.cdp_pump_dend * (u.mol / u.cm**2),
            ),
        )

    def _paint_channels(self) -> None:
        if self.cell is None:
            raise RuntimeError("Cell must be created before painting channels.")
        cable = self.params.cable
        ion = self.params.ion
        ch = self.params.channel
        temp = u.celsius2kelvin(self.temperature_celsius)

        # SOMA BLOCK
        #
        # This corresponds to the NEURON sequence:
        #
        #   soma.insert("pas"); soma.g_pas/e_pas = ...
        #   soma.insert("Nav1p6_MA24_PC"); soma.gbar_Nav1p6_MA24_PC = ...
        #   soma.insert("Kv..."); soma.g... = ...
        #
        # BrainCell combines each ``insert`` and parameter assignment into one
        # ``mech.Channel`` declaration.  The imported mechanism names use the
        # BrainCell registry names, but the order follows ``pc_neuron.py``.
        self.cell.paint(
            self.regions["soma"],
            mech.Channel("IL", g_max=cable.leak_g_soma_mS_cm2 * (u.mS / u.cm**2), E=cable.leak_e_mV * u.mV),
            mech.Channel("Nav1p6_MA2024_PC", g_max=ch.soma.nav1p6 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kv1p1_MA2024_PC", g_max=ch.soma.kv1p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kv1p5_MA2024_PC", g_max=ch.soma.kv1p5 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kv3p4_MA2024_PC", g_max=ch.soma.kv3p4 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kir2p3_MA2024_PC", g_max=ch.soma.kir2p3 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca1p1_MA2024_PC", g_max=ch.soma.kca1p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca2p2_MA2024_PC", g_max=ch.soma.kca2p2 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca3p1_MA2024_PC", g_max=ch.soma.kca3p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Cav2p1_MA2024_PC_Frozen", g_max=ch.soma.cav2p1_perm * (u.cm / u.second), temp=temp),
            mech.Channel("Cav3p1_MA2024_PC_Frozen", g_max=ch.soma.cav3p1_perm * (u.cm / u.second), temp=temp),
            mech.Channel("Cav3p2_MA2024_PC", g_max=ch.soma.cav3p2 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel(
                "Cav3p3_MA2024_PC_Frozen",
                perm=ch.soma.cav3p3_perm * (u.cm / u.second),
                g_scale=ch.cav3p3_g_scale,
                temp=temp,
            ),
            mech.Channel("HCN1_MA2024_PC", g_max=ch.soma.hcn1 * (u.siemens / u.cm**2), E=ion.h_e_mV * u.mV, temp=temp),
        )

        # DENDRITE BASE BLOCK
        #
        # This is the body of ``for dend in self.dend`` before the NEURON
        # ``if dend.diam >= THICK_DEND_DIAM_UM`` branch.  These channels are
        # painted on every dendrite.
        self.cell.paint(
            self.regions["dend"],
            mech.Channel("IL", g_max=cable.leak_g_dend_mS_cm2 * (u.mS / u.cm**2), E=cable.leak_e_mV * u.mV),
            mech.Channel("Kv3p3_MA2024_PC", g_max=ch.dend.kv3p3 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kv4p3_MA2024_PC", g_max=ch.dend.kv4p3 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca1p1_MA2024_PC", g_max=ch.dend.kca1p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca2p2_MA2024_PC", g_max=ch.dend.kca2p2 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Cav2p1_MA2024_PC_Frozen", g_max=ch.dend.cav2p1_perm * (u.cm / u.second), temp=temp),
            mech.Channel(
                "Cav3p3_MA2024_PC_Frozen",
                perm=ch.dend.cav3p3_perm * (u.cm / u.second),
                g_scale=ch.cav3p3_g_scale,
                temp=temp,
            ),
            mech.Channel("HCN1_MA2024_PC", g_max=ch.dend.hcn1 * (u.siemens / u.cm**2), E=ion.h_e_mV * u.mV, temp=temp),
        )

        # THICK-DENDRITE BLOCK
        #
        # NEURON:
        #
        #   if dend.diam >= THICK_DEND_DIAM_UM:
        #       dend.cm = SOMA_CM_UF_CM2
        #       insert Kv1p1/Kv1p5/Kir2p3/Cav3p1/Cav3p2/Kca3p1
        #
        # The capacitance overwrite is handled in ``pc24_dend_cm``.  This paint
        # call handles only the extra thick-dendrite mechanisms.
        self.cell.paint(
            self.regions["thick_dend"],
            mech.Channel("Kv1p1_MA2024_PC", g_max=ch.dend.kv1p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kv1p5_MA2024_PC", g_max=ch.dend.kv1p5 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kir2p3_MA2024_PC", g_max=ch.dend.kir2p3 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Kca3p1_MA2024_PC", g_max=ch.dend.kca3p1 * (u.siemens / u.cm**2), temp=temp),
            mech.Channel("Cav3p1_MA2024_PC_Frozen", g_max=ch.dend.cav3p1_perm * (u.cm / u.second), temp=temp),
            mech.Channel("Cav3p2_MA2024_PC", g_max=ch.dend.cav3p2 * (u.siemens / u.cm**2), temp=temp),
        )

        # NAV-DENDRITE BLOCK
        #
        # NEURON nests Nav1.6 inside the thick-dendrite block:
        #
        #   if dend.diam >= NAV_DEND_DIAM_UM:
        #       insert Nav1p6_MA24_PC
        #
        # In BrainCell this becomes a separate region paint.  It can reuse the
        # same channel class name as soma; runtime keeps distinct same-name
        # layouts from overwriting each other.
        self.cell.paint(
            self.regions["nav_dend"],
            mech.Channel("Nav1p6_MA2024_PC", g_max=ch.dend.nav1p6 * (u.siemens / u.cm**2), temp=temp),
        )


def _pc24_cv_count(branch: Any, cable: PCCableParameters) -> int:
    length_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    return pc24_nseg_rule(length_um, max_len_um=cable.cv_max_len_um)
