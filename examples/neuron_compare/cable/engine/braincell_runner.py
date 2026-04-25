#!/usr/bin/env python3
"""Run one braincell-side case for the multi-compartment cable template."""



from collections import defaultdict
from pathlib import Path
from typing import Any

import brainstate
import brainunit as u
import numpy as np

try:
    import braincell
    from braincell import CableProperty, Cell, CVPerBranch, Morphology
    from braincell.filter import AllRegion, at
    from braincell.mech import StateProbe

    from .case_schema import MultiCompartmentCableCase
    from .morphology_io import load_braincell_morphology
    from .stimulus import build_braincell_stimulus
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    _templates_root = _here.parent
    for candidate in (_here, _templates_root, Path(__file__).resolve().parents[4]):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import braincell  # type: ignore
    from braincell import CableProperty, Cell, CVPerBranch, Morphology  # type: ignore
    from braincell.filter import AllRegion, at  # type: ignore
    from braincell.mech import StateProbe  # type: ignore

    from case_schema import MultiCompartmentCableCase  # type: ignore
    from morphology_io import load_braincell_morphology  # type: ignore
    from stimulus import build_braincell_stimulus  # type: ignore


def run_case(case: MultiCompartmentCableCase) -> dict[str, Any]:
    brainstate.environ.set(precision=64)
    morpho = load_braincell_morphology(case)
    cell = Cell(
        morpho,
        solver="staggered",
        cv_policy=CVPerBranch(cv_per_branch=case.cv_policy.cv_per_branch),
    )
    cell.paint(
        AllRegion(),
        CableProperty(
            resting_potential=case.simulation.v_init_mV * u.mV,
            membrane_capacitance=case.cable.cm_uF_cm2 * (u.uF / u.cm ** 2),
            axial_resistivity=case.cable.ra_ohm_cm * (u.ohm * u.cm),
        ),
    )
    cell.place(at("soma", 0.5), build_braincell_stimulus(case.stimulus))
    probe_names = []
    for cv in cell.cvs:
        midpoint_x = 0.5 * (float(cv.prox) + float(cv.dist))
        probe_name = f"cv_{int(cv.id)}_v"
        probe_names.append(probe_name)
        cell.place(at(int(cv.branch_id), midpoint_x), StateProbe(name=probe_name))
    cell.init_state()
    result = cell.run(
        dt=float(case.simulation.dt_ms) * u.ms,
        duration=float(case.simulation.duration_ms) * u.ms,
    )
    times_ms = np.asarray(result.time.to_decimal(u.ms), dtype=float).reshape(-1)
    voltage_columns = [
        np.asarray(result.traces[probe_name].to_decimal(u.mV), dtype=float).reshape(-1)
        for probe_name in probe_names
    ]
    voltage_mV = np.stack(voltage_columns, axis=1)

    return {
        "time_ms": times_ms,
        "voltage_mV": voltage_mV,
        "compartment_labels": _compartment_labels(cell),
        "branch_order": _branch_order(morpho),
    }


def _branch_order(morpho) -> list[dict[str, Any]]:
    return [
        {
            "branch_id": int(branch.index),
            "branch_name": branch.name,
            "branch_type": branch.type,
        }
        for branch in morpho.branches
    ]


def _compartment_labels(cell: Cell) -> list[dict[str, Any]]:
    local_index_by_branch: dict[int, int] = defaultdict(int)
    labels: list[dict[str, Any]] = []
    for cv in cell.cvs:
        local_index = local_index_by_branch[int(cv.branch_id)]
        local_index_by_branch[int(cv.branch_id)] += 1
        labels.append(
            {
                "compartment_index": int(cv.id),
                "branch_id": int(cv.branch_id),
                "branch_name": cell.morpho.branch(index=int(cv.branch_id)).name,
                "branch_type": cv.branch_type,
                "local_index": int(local_index),
                "prox": float(cv.prox),
                "dist": float(cv.dist),
            }
        )
    return labels


def main() -> int:
    raise NotImplementedError("Run this module through run_case(case) for now.")


if __name__ == "__main__":
    raise SystemExit(main())
