#!/usr/bin/env python3
"""Run one braincell-side case for the single-compartment HH + fixed-ion template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import braintools
import brainunit as u
import numpy as np

try:
    from .case_schema import SingleCompartmentChannelHHFixedIonCase
    from .discovery import discover_braincell_channel_metadata
    from .pair_manifest import get_pair_entry
    from .stimulus_utils import build_braincell_stimulus
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
    discover_braincell_channel_metadata = _discovery.discover_braincell_channel_metadata
    _pair_manifest = _load_local_module("pair_manifest", _here / "pair_manifest.py")
    get_pair_entry = _pair_manifest.get_pair_entry
    _stimulus_utils = _load_local_module("stimulus_utils", _here / "stimulus_utils.py")
    build_braincell_stimulus = _stimulus_utils.build_braincell_stimulus


def run_case(case: SingleCompartmentChannelHHFixedIonCase) -> dict[str, Any]:
    import braincell
    from braincell.filter import AllRegion, at
    from braincell.mech import Channel, CurrentProbe, MechanismProbe, StateProbe
    from braincell.morph import Branch, Morphology

    pair = get_pair_entry(case.pair_id)
    meta = discover_braincell_channel_metadata(pair.braincell_channel_name)
    if meta.channel_kind != "hh":
        raise ValueError(
            f"braincell channel {pair.braincell_channel_name!r} is not detected as HH, got {meta.channel_kind!r}."
        )

    ion_type = case.ion.ion_type or pair.ion_type_override or meta.ion_type
    if ion_type is None:
        raise ValueError(
            f"Could not determine ion type for pair_id {case.pair_id!r}; provide ion.ion_type explicitly."
        )
    if case.ion.ion_type is not None and meta.ion_type is not None and case.ion.ion_type != meta.ion_type:
        raise ValueError(
            f"Configured ion_type {case.ion.ion_type!r} disagrees with discovered braincell ion_type {meta.ion_type!r}."
        )

    gate_names = case.compare.gate_names or meta.gate_names
    if len(gate_names) == 0:
        raise ValueError(
            f"Could not determine gate names for braincell channel {pair.braincell_channel_name!r}."
        )

    channel_kwargs = _convert_channel_overrides_for_braincell(meta, case.channel_overrides)

    soma = Branch.from_lengths(
        lengths=[float(case.morphology.length_um)] * u.um,
        radii=[float(case.morphology.radius_um), float(case.morphology.radius_um)] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    cell = braincell.Cell(
        morpho,
        cv_policy=braincell.CVPerBranch(),
        V_initializer=braintools.init.Uniform(
            float(case.simulation.v_init_mV) * u.mV,
            float(case.simulation.v_init_mV) * u.mV,
        ),
    )

    region = AllRegion()
    cell.paint(
        region,
        braincell.mech.CableProperty(
            resting_potential=float(case.simulation.v_init_mV) * u.mV,
            membrane_capacitance=float(case.morphology.cm_uF_cm2) * (u.uF / (u.cm ** 2)),
            axial_resistivity=100.0 * (u.ohm * u.cm),
            temperature=u.celsius2kelvin(float(case.simulation.temperature_celsius)),
        ),
    )

    cell.paint(region, Channel(pair.braincell_channel_name, **channel_kwargs))

    if case.leak.enabled:
        cell.paint(
            region,
            Channel(
                "IL",
                g_max=float(case.leak.g_S_cm2) * (u.siemens / (u.cm ** 2)),
                E=float(case.leak.e_mV) * u.mV,
            ),
        )

    probe_loc = at("soma", 0.5)
    cell.place(probe_loc, build_braincell_stimulus(case.stimulus))
    cell.place(probe_loc, StateProbe())
    cell.place(
        probe_loc,
        *(MechanismProbe(mechanism=pair.braincell_channel_name, field=gate_name) for gate_name in gate_names),
    )
    cell.place(probe_loc, CurrentProbe(ion=ion_type, mechanism=pair.braincell_channel_name))

    cell.init_state()
    ion = cell.get_ion(ion_type)
    ion.E = float(case.ion.fixed_E_mV) * u.mV
    cell.reset_state()

    result = cell.run(
        dt=float(case.simulation.dt_ms) * u.ms,
        duration=float(case.simulation.duration_ms) * u.ms,
    )

    current_probe_name = f"soma(0.5)_{pair.braincell_channel_name}_current"
    return {
        "time_ms": _ensure_1d(result.time.to_decimal(u.ms), "braincell_time_ms"),
        "voltage_mV": _ensure_1d(result.traces["soma(0.5)_v"].to_decimal(u.mV), "braincell_voltage_mV"),
        "current": {
            "ix": _ensure_1d(
                result.traces[current_probe_name].to_decimal(u.mA / (u.cm ** 2)),
                "braincell_current_ix",
            )
        },
        "gates": {
            gate_name: _ensure_1d(
                np.asarray(result.traces[f"soma(0.5)_{pair.braincell_channel_name}_{gate_name}"]),
                f"braincell_gate_{gate_name}",
            )
            for gate_name in gate_names
        },
    }


def _convert_channel_overrides_for_braincell(meta, channel_overrides: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in channel_overrides.items():
        if key == "g_max_S_cm2":
            converted["g_max"] = float(value) * (u.siemens / (u.cm ** 2))
            continue
        if key == "V_sh_mV":
            converted["V_sh"] = float(value) * u.mV
            continue
        if key == "v12_mV":
            converted["v12"] = float(value) * u.mV
            continue
        if key in meta.constructor_param_names:
            converted[key] = value
            continue
        raise ValueError(
            f"channel_overrides contains unsupported param for {meta.channel_name!r}: {key!r}."
        )
    return converted


def _ensure_1d(arr: object, name: str) -> np.ndarray:
    value = np.asarray(arr)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape!r}.")
    return value.astype(float, copy=False).reshape(-1)


def main() -> int:
    raise NotImplementedError("Run this module through run_case(case) for now.")


if __name__ == "__main__":
    raise SystemExit(main())
