# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
        braincell.NetStim(
            name="stim",
            start=netstim_start_ms * u.ms,
            number=netstim_number,
            interval=netstim_interval_ms * u.ms,
            noise=0.0,
            weight=1.0,
        ),
    )
    if synapse_type == "ExpSyn":
        synapse_fields = ("g",)
    elif synapse_type == "Exp2Syn":
        synapse_fields = ("A", "B")
    else:
        raise ValueError(f"Unsupported synapse_type {synapse_type!r}.")
    cell.place(
        at("soma", 0.5),
        mech.Synapse(synapse_type, name=synapse_name, **synapse_params),
    )
    soma = cell.loc(at("soma", 0.5))
    synapse = braincell.observe.synapse(name=synapse_name)
    soma.record("v", braincell.observe.state("V"))
    soma.record("i_syn", synapse.current())
    for field in synapse_fields:
        soma.record(field, synapse.state(field))
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

    # A recording samples before each step: row k is the state at t = k dt,
    # row 0 the initialized state. Rows 1 .. n_steps - 1 are the states at
    # t = dt .. duration - dt, the sample times of the NEURON trace.
    def series(name, unit):
        return np.asarray(result.traces[name].to_decimal(unit), dtype=float)[1:]

    time_ms = np.asarray(result.time.to_decimal(u.ms), dtype=float)[1:]
    voltage_mV = series("v", u.mV)
    current_nA = series("i_syn", u.nA)
    # pre_spike[i] is 1 when a NetStim event falls at time_ms[i].
    pre_spike = np.asarray(
        [1.0 if step in spike_steps else 0.0 for step in range(1, n_steps)],
        dtype=float,
    )

    data = {
        "time_ms": time_ms,
        "pre_spike": pre_spike,
        "voltage_mV": voltage_mV,
        "current_nA": current_nA,
    }
    if synapse_type == "ExpSyn":
        data["g_uS"] = series("g", u.uS)
    if synapse_type == "Exp2Syn":
        data["A_uS"] = series("A", u.uS)
        data["B_uS"] = series("B", u.uS)
        data["g_uS"] = data["B_uS"] - data["A_uS"]
    return data
