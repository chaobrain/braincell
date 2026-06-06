"""Run minimal BrainCell `NetStim + ExpSyn/Exp2Syn` comparisons.

The returned ``current_nA`` trace is a **signed total synaptic current**
in nanoamps. For excitatory synapses under the NEURON sign convention,
the peak may be negative, so callers should inspect ``min`` or
``abs(current_nA).max()`` instead of only ``max``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import brainunit as u
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (
        candidate
        for candidate in (_HERE, *_HERE.parents)
        if (candidate / "braincell").exists() and (candidate / "examples").exists()
    ),
    _HERE,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_demo_morphology():
    from braincell import Branch, Morphology

    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def _step_indices(*, start_ms: float, number: int, interval_ms: float, dt_ms: float) -> tuple[int, ...]:
    """Return integer step indices for a deterministic NetStim schedule."""
    if number <= 0:
        return ()
    return tuple(
        int(round((start_ms + (index * interval_ms)) / dt_ms))
        for index in range(int(number))
    )

def run_braincell_synapse_case(
    *,
    synapse_type: str,
    synapse_name: str,
    synapse_params: dict[str, Any],
    netstim_start_ms: float,
    netstim_number: int,
    netstim_interval_ms: float,
    dt_ms: float,
    duration_ms: float,
    leak_g_max = 0.1 * (u.mS / u.cm ** 2),
    leak_E = -65.0 * u.mV,
) -> dict[str, np.ndarray]:
    """Run one BrainCell `ExpSyn` / `Exp2Syn` toy case.

    Returns
    -------
    dict[str, np.ndarray]
        A dict containing at least:

        - ``time_ms``
        - ``pre_spike`` (reference event train derived from the NetStim config)
        - ``voltage_mV``
        - ``current_nA`` (signed total current)
    """
    import braincell
    import braincell.mech as mech
    from braincell import Cell, CVPerBranch
    from braincell.filter import at

    cell = Cell(
        build_demo_morphology(),
        cv_policy=CVPerBranch(),
        solver="staggered",
        V_init=-65.0 * u.mV,
    )
    cell.paint(
        braincell.filter.AllRegion(),
        mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        ),
        mech.Channel("IL", g_max=leak_g_max, E=leak_E),
    )
    cell.place(
        at("soma", 0.5),
        mech.NetStim(
            name="stim",
            start=netstim_start_ms * u.ms,
            number=netstim_number,
            interval=netstim_interval_ms * u.ms,
            noise=0.0,
            weight=1.0,
        ),
    )
    cell.place(at("soma", 0.5), mech.StateProbe(name="v", field="v"))
    cell.place(at("soma", 0.5), mech.CurrentProbe(name="i_syn", mechanism=synapse_name))
    if synapse_type == "ExpSyn":
        cell.place(at("soma", 0.5), mech.MechanismProbe(name="g", mechanism=synapse_name, field="g"))
    elif synapse_type == "Exp2Syn":
        cell.place(at("soma", 0.5), mech.MechanismProbe(name="A", mechanism=synapse_name, field="A"))
        cell.place(at("soma", 0.5), mech.MechanismProbe(name="B", mechanism=synapse_name, field="B"))
    else:
        raise ValueError(f"Unsupported synapse_type {synapse_type!r}.")
    cell.place(
        at("soma", 0.5),
        mech.Synapse(synapse_type, name=synapse_name, **synapse_params),
    )
    cell.init_state()

    dt = dt_ms * u.ms
    n_steps = int(round(duration_ms / dt_ms))
    spike_steps = set(
        _step_indices(
            start_ms=netstim_start_ms,
            number=netstim_number,
            interval_ms=netstim_interval_ms,
            dt_ms=dt_ms,
        )
    )
    result = cell.run(
        dt=dt,
        duration=duration_ms * u.ms,
    )

    time_ms = np.asarray(result.time.to_decimal(u.ms), dtype=float)[:-1]
    voltage_mV = np.asarray(result.traces["v"].to_decimal(u.mV), dtype=float)[:-1]
    current_nA = np.asarray(result.traces["i_syn"].to_decimal(u.nA), dtype=float)[:-1]
    pre_spike = np.asarray(
        [1.0 if step in spike_steps else 0.0 for step in range(n_steps - 1)],
        dtype=float,
    )

    data = {
        "time_ms": time_ms,
        "pre_spike": pre_spike,
        "voltage_mV": voltage_mV,
        "current_nA": current_nA,
    }
    if synapse_type == "ExpSyn":
        data["g_uS"] = np.asarray(result.traces["g"].to_decimal(u.uS), dtype=float)[:-1]
    if synapse_type == "Exp2Syn":
        data["A_uS"] = np.asarray(result.traces["A"].to_decimal(u.uS), dtype=float)[:-1]
        data["B_uS"] = np.asarray(result.traces["B"].to_decimal(u.uS), dtype=float)[:-1]
        data["g_uS"] = data["B_uS"] - data["A_uS"]
    return data
