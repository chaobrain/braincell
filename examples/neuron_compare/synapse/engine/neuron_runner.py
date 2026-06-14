"""Run minimal NEURON `NetStim + ExpSyn/Exp2Syn` comparisons.

The returned ``current_nA`` trace is the signed total synaptic current
recorded from the point process. Under NEURON's default sign convention,
an excitatory current is often negative because ``i = g * (v - e)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import brainunit as u


def run_neuron_synapse_case(
    *,
    synapse_type: str,
    synapse_params: dict[str, Any],
    netstim_start_ms: float,
    netstim_number: int,
    netstim_interval_ms: float,
    dt_ms: float,
    duration_ms: float,
    leak_g_max = 0.1 * (u.mS / u.cm ** 2),
    leak_E = -65.0 * u.mV,
) -> dict[str, np.ndarray]:
    """Run one NEURON `NetStim + ExpSyn/Exp2Syn` toy case."""
    from neuron import h

    h.load_file("stdrun.hoc")
    soma = h.Section(name="soma")
    soma.L = 20.0
    soma.diam = 20.0
    soma.nseg = 1
    soma.cm = 1.0
    soma.Ra = 100.0
    soma.insert("pas")
    for seg_pas in soma:
        seg_pas.pas.g = float(leak_g_max.to_decimal(u.mS / u.cm ** 2)) / 1000.0
        seg_pas.pas.e = float(leak_E.to_decimal(u.mV))
    seg = soma(0.5)

    if synapse_type == "ExpSyn":
        syn = h.ExpSyn(seg)
        syn.tau = float(synapse_params["tau"].to_decimal(u.ms))
        syn.e = float(synapse_params["e"].to_decimal(u.mV))
    elif synapse_type == "Exp2Syn":
        syn = h.Exp2Syn(seg)
        syn.tau1 = float(synapse_params["tau1"].to_decimal(u.ms))
        syn.tau2 = float(synapse_params["tau2"].to_decimal(u.ms))
        syn.e = float(synapse_params["e"].to_decimal(u.mV))
    else:
        raise ValueError(f"Unsupported synapse_type {synapse_type!r}.")

    stim = h.NetStim()
    stim.start = float(netstim_start_ms)
    stim.number = int(netstim_number)
    stim.interval = float(netstim_interval_ms)
    stim.noise = 0.0

    nc = h.NetCon(stim, syn)
    nc.weight[0] = float(synapse_params["weight"].to_decimal(u.uS))
    nc.delay = 0.0

    h.cvode_active(0)
    h.dt = float(dt_ms)
    h.steps_per_ms = 1.0 / float(dt_ms)
    h.v_init = -65.0
    h.finitialize(h.v_init)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(seg._ref_v)
    i_vec = h.Vector().record(syn._ref_i)

    extra = {}
    for field in ("g", "A", "B"):
        ref_name = f"_ref_{field}"
        if hasattr(syn, ref_name):
            extra[field] = h.Vector().record(getattr(syn, ref_name))

    try:
        h.tstop = float(duration_ms)
        h.run()
    finally:
        h.delete_section(sec=soma)

    time_ms = np.asarray(t_vec, dtype=float)[1:]
    voltage_mV = np.asarray(v_vec, dtype=float)[1:]
    # NEURON point-process current recording can expose the step-start cached
    # event-boundary current, while BrainCell comparison traces are sampled at
    # step end. Keep this phase convention explicit for ExpSyn peak comparisons.
    current_nA = np.asarray(i_vec, dtype=float)[2:]

    target_len = min(len(time_ms), len(voltage_mV), len(current_nA))
    data = {
        "time_ms": time_ms[:target_len],
        "voltage_mV": voltage_mV[:target_len],
        "current_nA": current_nA[:target_len],
    }
    if "g" in extra:
        data["g_uS"] = np.asarray(extra["g"], dtype=float)[1:1 + target_len]
    if "A" in extra:
        data["A_uS"] = np.asarray(extra["A"], dtype=float)[1:1 + target_len]
    if "B" in extra:
        data["B_uS"] = np.asarray(extra["B"], dtype=float)[1:1 + target_len]
        data["g_uS"] = data["B_uS"] - data["A_uS"]
    return data
