#!/usr/bin/env python3
"""Run one NEURON-side case for the single-compartment HH + fixed-ion template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .case_schema import SingleCompartmentChannelHHFixedIonCase
    from .discovery import discover_neuron_channel_metadata
    from .pair_manifest import get_pair_entry
    from .stimulus_utils import current_at_ms
except ImportError:  # pragma: no cover
    import importlib.util
    import sys

    def _load_local_module(module_name: str, path: Path):
        module_key = f"sc_channel_hh_fixed_ion__{module_name}"
        module = sys.modules.get(module_key)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(module_key, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        return module

    _here = Path(__file__).resolve().parent
    _case_schema = _load_local_module("case_schema", _here / "case_schema.py")
    SingleCompartmentChannelHHFixedIonCase = _case_schema.SingleCompartmentChannelHHFixedIonCase
    _discovery = _load_local_module("discovery", _here / "discovery.py")
    discover_neuron_channel_metadata = _discovery.discover_neuron_channel_metadata
    _pair_manifest = _load_local_module("pair_manifest", _here / "pair_manifest.py")
    get_pair_entry = _pair_manifest.get_pair_entry
    _stimulus_utils = _load_local_module("stimulus_utils", _here / "stimulus_utils.py")
    current_at_ms = _stimulus_utils.current_at_ms


def run_case(case: SingleCompartmentChannelHHFixedIonCase) -> dict[str, Any]:
    pair = get_pair_entry(case.pair_id)
    meta = discover_neuron_channel_metadata(case.mod_dir, pair.neuron_mechanism_name)
    ion_type = case.ion.ion_type or pair.ion_type_override or meta.ion_type
    if ion_type is None:
        raise ValueError(
            f"Could not determine ion type for pair_id {case.pair_id!r}; provide ion.ion_type explicitly."
        )
    if case.ion.ion_type is not None and meta.ion_type is not None and case.ion.ion_type != meta.ion_type:
        raise ValueError(
            f"Configured ion_type {case.ion.ion_type!r} disagrees with discovered NEURON ion_type {meta.ion_type!r}."
        )
    h, section, segment, mechanism_name = _build_neuron_model(
        case,
        mechanism_name=pair.neuron_mechanism_name,
        ion_type=ion_type,
    )

    current_field = meta.current_field
    current_owner = meta.current_owner
    gate_names = _resolve_neuron_gate_names(case, meta=meta)

    h.cvode_active(0)
    h.dt = float(case.simulation.dt_ms)
    h.celsius = float(case.simulation.temperature_celsius)
    h.v_init = float(case.simulation.v_init_mV)
    h.finitialize(h.v_init)
    sample_times_ms = np.arange(0.0, float(case.simulation.duration_ms), float(case.simulation.dt_ms), dtype=float)
    if sample_times_ms.size == 0:
        raise ValueError("simulation.duration_ms must produce at least one sample.")

    stim = h.IClamp(segment)
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0
    amp_values = np.asarray(
        [current_at_ms(case.stimulus, float(t_ms)) for t_ms in sample_times_ms],
        dtype=float,
    )
    amp_time_vec = h.Vector(sample_times_ms)
    amp_value_vec = h.Vector(amp_values)
    amp_value_vec.play(stim._ref_amp, amp_time_vec, 1)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(segment._ref_v)
    mech_obj = getattr(segment, mechanism_name)
    if current_owner == "mechanism":
        current_ref = getattr(mech_obj, f"_ref_{current_field}")
    else:
        current_ref = getattr(segment, f"_ref_{current_field}")
    current_vec = h.Vector().record(current_ref)
    gate_vecs = {
        gate_name: h.Vector().record(getattr(mech_obj, f"_ref_{gate_name}"))
        for gate_name in gate_names
    }

    try:
        h.tstop = float(case.simulation.duration_ms)
        h.run()
    finally:
        amp_value_vec.play_remove()
        h.delete_section(sec=section)

    time_ms = _ensure_1d(t_vec, name="neuron.time_ms")[1:]
    voltage_mV = _ensure_1d(v_vec, name="neuron.voltage_mV")[1:]
    # NEURON ionic/mechanism currents are outward-positive; convert to the
    # channel contribution form used on the braincell side, g * (E - V).
    current_ix = -_ensure_1d(current_vec, name="neuron.current.ix")[1:]
    gates = {
        gate_name: _ensure_1d(gate_vec, name=f"neuron.gates.{gate_name}")[1:]
        for gate_name, gate_vec in gate_vecs.items()
    }

    return {
        "time_ms": time_ms,
        "voltage_mV": voltage_mV,
        "current": {"ix": current_ix},
        "gates": gates,
    }


def _build_neuron_model(
    case: SingleCompartmentChannelHHFixedIonCase,
    *,
    mechanism_name: str,
    ion_type: str,
):
    from neuron import h, load_mechanisms

    load_mechanisms(str(Path(case.mod_dir).resolve()))
    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = float(case.morphology.length_um)
    soma.diam = float(2.0 * case.morphology.radius_um)
    soma.nseg = 1
    soma.cm = float(case.morphology.cm_uF_cm2)

    if case.leak.enabled:
        soma.insert("pas")
        for seg in soma:
            seg.pas.g = float(case.leak.g_S_cm2)
            seg.pas.e = float(case.leak.e_mV)

    soma.insert(mechanism_name)
    ion_field = _resolve_neuron_ion_field(ion_type)
    setattr(soma, ion_field, float(case.ion.fixed_E_mV))

    for seg in soma:
        mech_obj = getattr(seg, mechanism_name)
        for key, value in case.channel_overrides.items():
            attr_name = {
                "g_max_S_cm2": "gbar",
                "V_sh_mV": "V_sh",
                "v12_mV": "v12",
            }.get(key, key)
            setattr(mech_obj, attr_name, float(value))

    return h, soma, soma(0.5), mechanism_name


def _ensure_1d(vec: object, *, name: str) -> np.ndarray:
    value = np.asarray(vec, dtype=float)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape!r}.")
    return value.reshape(-1)
def _resolve_neuron_ion_field(ion_type: str) -> str:
    return {
        "na": "ena",
        "k": "ek",
        "ca": "eca",
    }[ion_type]


def _resolve_neuron_gate_names(
    case: SingleCompartmentChannelHHFixedIonCase,
    *,
    meta,
) -> tuple[str, ...]:
    if case.compare.gate_names is not None:
        return case.compare.gate_names
    if len(meta.gate_names) == 0:
        raise ValueError(
            "Could not auto-discover gate names; "
            "provide compare.gate_names explicitly."
        )
    return meta.gate_names


def main() -> int:
    raise NotImplementedError("Run this module through run_case(case) for now.")


if __name__ == "__main__":
    raise SystemExit(main())
