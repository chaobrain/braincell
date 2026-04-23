#!/usr/bin/env python3
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

"""Single-channel point-neuron comparison between NEURON and braincell."""



import inspect
import math
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

FIGURE_NAME = "channel_validate.png"


def ensure_1d(arr: Any, name: str) -> np.ndarray:
    value = np.asarray(arr)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape}")
    return value.astype(float, copy=False).reshape(-1)


def align_pair(ref: Any, pred: Any, ref_name: str, pred_name: str) -> tuple[np.ndarray, np.ndarray]:
    ref_1d = ensure_1d(ref, ref_name)
    pred_1d = ensure_1d(pred, pred_name)
    n = min(ref_1d.shape[0], pred_1d.shape[0])
    if n == 0:
        raise ValueError(f"Empty trace after alignment: {ref_name}, {pred_name}")
    return ref_1d[:n], pred_1d[:n]


def compute_error_metrics(ref: Any, pred: Any) -> dict[str, float]:
    ref_1d, pred_1d = align_pair(ref, pred, "ref", "pred")
    err = ref_1d - pred_1d
    eps = 1e-10
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_bias": float(np.mean(err)),
        "rel_mae_pct": float(np.mean(np.abs(err) / (np.abs(ref_1d) + eps)) * 100.0),
        "n_samples": int(err.shape[0]),
    }


def resolve_neuron_ion_field(ion_type: str) -> str:
    ion_field_map = {
        "na": "ena",
        "k": "ek",
        "ca": "eca",
    }
    try:
        return ion_field_map[ion_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ion_type for NEURON: {ion_type!r}") from exc


def resolve_braincell_ion_class(ion_type: str):
    import braincell

    ion_class_map = {
        "na": braincell.ion.SodiumFixed,
        "k": braincell.ion.PotassiumFixed,
        "ca": braincell.ion.CalciumFixed,
    }
    try:
        return ion_class_map[ion_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ion_type for braincell: {ion_type!r}") from exc


def resolve_braincell_channel_class(channel_name: str):
    import braincell

    if hasattr(braincell.channel, channel_name):
        return getattr(braincell.channel, channel_name)

    alias_map = {
        "Kv": "K_Kv_test",
    }
    if channel_name in alias_map and hasattr(braincell.channel, alias_map[channel_name]):
        return getattr(braincell.channel, alias_map[channel_name])

    raise ValueError(f"Unsupported braincell channel name: {channel_name!r}")


def _ensure_neuron_mechanism_available(mod_dir: Path, mechanism: str) -> None:
    from neuron import h, load_mechanisms

    probe = h.Section(name="mech_probe")
    try:
        probe.insert(mechanism)
        return
    except Exception:
        pass
    finally:
        h.delete_section(sec=probe)

    load_mechanisms(str(mod_dir))

    probe = h.Section(name="mech_probe")
    try:
        probe.insert(mechanism)
    except Exception as exc:
        raise RuntimeError(
            f"Mechanism '{mechanism}' is unavailable after load_mechanisms({mod_dir})."
        ) from exc
    finally:
        h.delete_section(sec=probe)


def map_channel_params_to_braincell(
    channel_cls,
    params: dict[str, Any],
    *,
    temperature_celsius: float,
) -> dict[str, Any]:
    import brainunit as u

    params = dict(params)
    if "gbar" in params and "g_max" not in params:
        params["g_max"] = params.pop("gbar")
    if "gbar_S_cm2" in params and "g_max" not in params:
        params["g_max"] = params.pop("gbar_S_cm2")
    if "g_max_S_cm2" in params and "g_max" not in params:
        params["g_max"] = params.pop("g_max_S_cm2")

    signature = inspect.signature(channel_cls.__init__)
    valid_names = set(signature.parameters)

    if "temp" in valid_names and "temp" not in params:
        params["temp"] = u.celsius2kelvin(float(temperature_celsius))

    mapped: dict[str, Any] = {}
    for key, value in params.items():
        if key in valid_names:
            mapped[key] = value
    return mapped


def _convert_channel_params_for_braincell(
    channel_cls,
    params: dict[str, Any],
    *,
    temperature_celsius: float,
):
    import brainunit as u

    mapped = map_channel_params_to_braincell(channel_cls, params, temperature_celsius=temperature_celsius)
    converted: dict[str, Any] = {}
    for key, value in mapped.items():
        if key == "g_max":
            converted[key] = float(value) * (u.siemens / (u.cm ** 2))
        elif key in {"V_sh", "v12", "q"}:
            converted[key] = float(value) * u.mV
        elif key == "temp":
            converted[key] = value
        else:
            converted[key] = value
    return converted


def _convert_leak_params_for_braincell(params: dict[str, Any]) -> dict[str, Any]:
    import brainunit as u

    gbar = params.get("gbar", params.get("g", 0.0))
    return {
        "g_max": float(gbar) * (u.siemens / (u.cm ** 2)),
    }


def _resolve_length_and_diameter_um(common_params: dict[str, Any]) -> tuple[float, float]:
    if "L_um" in common_params:
        length_um = float(common_params["L_um"])
    else:
        length_um = float(common_params["length_um"])

    if "diam_um" in common_params:
        diam_um = float(common_params["diam_um"])
    else:
        diam_um = 2.0 * float(common_params["radius_um"])

    return length_um, diam_um


def _normalize_leak_spec(leak_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if leak_spec is None:
        return None

    if "enabled" in leak_spec or "params" in leak_spec or "rev_mV" in leak_spec:
        legacy = dict(leak_spec)
        if not legacy.get("enabled", True):
            return None
        params = legacy.get("params", {})
        return {
            "g_S_cm2": float(params.get("gbar", params.get("g", 0.0))),
            "e_mV": float(legacy["rev_mV"]),
        }

    return {
        "g_S_cm2": float(leak_spec["g_S_cm2"]),
        "e_mV": float(leak_spec["e_mV"]),
    }


def create_cell_neuron(
    common_params: dict[str, Any],
    ion_spec: dict[str, Any],
    channel_spec: dict[str, Any],
    leak_spec: dict[str, Any] | None = None,
    mod_dir: Path | None = None,
):
    from neuron import h

    mod_dir = (Path(__file__).resolve().parent / "mods") if mod_dir is None else Path(mod_dir)
    mechanism = channel_spec.get("mechanism", channel_spec.get("neuron_name"))
    if mechanism is None:
        raise ValueError("channel_spec must provide 'mechanism' or 'neuron_name'.")
    _ensure_neuron_mechanism_available(mod_dir, mechanism)
    h.load_file("stdrun.hoc")

    length_um, diam_um = _resolve_length_and_diameter_um(common_params)
    soma = h.Section(name="soma")
    soma.L = length_um
    soma.diam = diam_um
    soma.nseg = 1
    soma.cm = float(common_params["cm_uF_cm2"])

    leak_cfg = _normalize_leak_spec(leak_spec)
    if leak_cfg is not None:
        soma.insert("pas")
        for seg in soma:
            seg.pas.g = float(leak_cfg["g_S_cm2"])
            seg.pas.e = float(leak_cfg["e_mV"])

    soma.insert(mechanism)
    channel_params = channel_spec.get("params", {})
    for seg in soma:
        mech_obj = getattr(seg, mechanism)
        for key, value in channel_params.items():
            attr_name = {
                "gbar_S_cm2": "gbar",
                "g_S_cm2": "g",
            }.get(key, key)
            setattr(mech_obj, attr_name, float(value))

    ion_field = resolve_neuron_ion_field(ion_spec["ion_type"])
    setattr(soma, ion_field, float(ion_spec["ion_rev_mV"]))

    return {
        "h": h,
        "section": soma,
        "segment": soma(0.5),
        "mechanism_name": mechanism,
    }


def create_cell_braincell(
    common_params: dict[str, Any],
    ion_spec: dict[str, Any],
    channel_spec: dict[str, Any],
    leak_spec: dict[str, Any] | None = None,
    *,
    solver: str = "rk4",
):
    import braincell
    import braintools
    import brainunit as u

    ion_cls = resolve_braincell_ion_class(ion_spec["ion_type"])
    channel_name = channel_spec.get("channel_name", channel_spec.get("braincell_name"))
    if channel_name is None:
        raise ValueError("channel_spec must provide 'channel_name' or 'braincell_name'.")
    channel_cls = resolve_braincell_channel_class(channel_name)
    channel_kwargs = _convert_channel_params_for_braincell(
        channel_cls,
        channel_spec.get("params", {}),
        temperature_celsius=float(common_params["temperature_celsius"]),
    )

    leak_cfg = _normalize_leak_spec(leak_spec)
    leak_kwargs = _convert_leak_params_for_braincell({"gbar": leak_cfg["g_S_cm2"]}) if leak_cfg else None
    leak_rev_mV = float(leak_cfg["e_mV"]) if leak_cfg else None
    length_um, diam_um = _resolve_length_and_diameter_um(common_params)

    class GenericSingleChannelCell(braincell.SingleCompartment):
        def __init__(self, size: int = 1):
            super().__init__(
                size,
                length=length_um * u.um,
                radius=0.5 * diam_um * u.um,
                C=float(common_params["cm_uF_cm2"]) * u.uF / (u.cm ** 2),
                solver=solver,
                V_initializer=braintools.init.Uniform(
                    float(common_params["v_init_mV"]) * u.mV,
                    float(common_params["v_init_mV"]) * u.mV,
                ),
            )

            ion_group = ion_cls(size, E=float(ion_spec["ion_rev_mV"]) * u.mV)
            ion_group.add(main_channel=channel_cls(size, **channel_kwargs))
            setattr(self, ion_spec["ion_type"].lower(), ion_group)

            if leak_cfg is not None:
                self.IL = braincell.channel.IL(
                    size,
                    E=leak_rev_mV * u.mV,
                    **leak_kwargs,
                )

    return GenericSingleChannelCell(1)


def run_neuron(
    cell_info,
    I_ext: Callable[[float], float],
    *,
    dt_ms: float,
    tstop_ms: float,
    v_init_mV: float,
    temperature_celsius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = cell_info["h"]
    soma = cell_info["section"]
    segment = cell_info["segment"]

    stim = h.IClamp(segment)
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0

    h.cvode_active(0)
    h.dt = float(dt_ms)
    h.celsius = float(temperature_celsius)
    h.finitialize(float(v_init_mV))

    times_ms = np.arange(0.0, float(tstop_ms), float(dt_ms), dtype=float)
    currents_nA = np.empty_like(times_ms)
    voltages_mV = np.empty_like(times_ms)

    try:
        for idx, t_ms in enumerate(times_ms):
            i_nA = float(I_ext(float(t_ms)))
            stim.amp = i_nA
            currents_nA[idx] = i_nA
            voltages_mV[idx] = float(segment.v)
            h.fadvance()
    finally:
        h.delete_section(sec=soma)

    return times_ms, currents_nA, voltages_mV


def run_braincell(
    cell,
    I_ext: Callable[[float], float],
    *,
    dt_ms: float,
    tstop_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import brainstate
    import brainunit as u

    cell.init_state()
    cell.reset_state()
    times_ms = np.arange(0.0, float(tstop_ms), float(dt_ms), dtype=float)
    currents_nA = np.asarray([float(I_ext(float(t_ms))) for t_ms in times_ms], dtype=float)
    times = times_ms * u.ms

    def step_fun(t, i_nA):
        v_now = cell.V.value
        with brainstate.environ.context(t=t):
            cell.update(i_nA * u.nA)
        return v_now

    with brainstate.environ.context(dt=dt_ms * u.ms):
        voltages = brainstate.transform.for_loop(step_fun, times, currents_nA)

    return times_ms, currents_nA, ensure_1d(voltages / u.mV, "braincell_v_mV")


def plot_compare(
    t_neuron_ms: np.ndarray,
    i_neuron_nA: np.ndarray,
    v_neuron_mV: np.ndarray,
    t_braincell_ms: np.ndarray,
    i_braincell_nA: np.ndarray,
    v_braincell_mV: np.ndarray,
    *,
    out_path: Path | None = None,
) -> Path:
    out_path = Path(__file__).resolve().parent / FIGURE_NAME if out_path is None else Path(out_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t_neuron_ms, i_neuron_nA, label="NEURON current", linewidth=2)
    axes[0].plot(t_braincell_ms, i_braincell_nA, label="braincell current", linewidth=2, linestyle="--")
    axes[0].set_ylabel("I_ext (nA)")
    axes[0].legend()

    axes[1].plot(t_neuron_ms, v_neuron_mV, label="NEURON voltage", linewidth=2)
    axes[1].plot(t_braincell_ms, v_braincell_mV, label="braincell voltage", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Membrane potential (mV)")
    axes[1].legend()

    fig.suptitle("Single-neuron current and voltage comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    common_params = {
        "dt_ms": 0.025,
        "v_init_mV": -65.0,
        "temperature_celsius": 25.0,
        "L_um": 10.0,
        "diam_um": 100.0 / math.pi,
        "cm_uF_cm2": 1.0,
        "tstop_ms": 100.0,
    }

    ion_spec = {
        "ion_type": "k",
        "ion_rev_mV": -80.0,
    }

    channel_spec = {
        "mechanism": "Kv",
        "channel_name": "K_Kv_test",
        "params": {
            "gbar": 0.0,
            "v12": 25.0,
            "q": 9.0,
        },
    }

    leak_spec = {"g_S_cm2": 1e-4, "e_mV": -65.0}

    def I_ext(t_ms: float) -> float:
        return 0.01 if 0.0 <= t_ms < 100.0 else 0.0

    neuron_cell = create_cell_neuron(common_params, ion_spec, channel_spec, leak_spec=leak_spec)
    braincell_cell = create_cell_braincell(common_params, ion_spec, channel_spec, leak_spec=leak_spec)

    t_neuron_ms, i_neuron_nA, v_neuron_mV = run_neuron(
        neuron_cell,
        I_ext,
        dt_ms=float(common_params["dt_ms"]),
        tstop_ms=float(common_params["tstop_ms"]),
        v_init_mV=float(common_params["v_init_mV"]),
        temperature_celsius=float(common_params["temperature_celsius"]),
    )
    t_braincell_ms, i_braincell_nA, v_braincell_mV = run_braincell(
        braincell_cell,
        I_ext,
        dt_ms=float(common_params["dt_ms"]),
        tstop_ms=float(common_params["tstop_ms"]),
    )

    metrics = compute_error_metrics(v_neuron_mV, v_braincell_mV)
    figure_path = plot_compare(
        t_neuron_ms,
        i_neuron_nA,
        v_neuron_mV,
        t_braincell_ms,
        i_braincell_nA,
        v_braincell_mV,
    )

    print(f"figure={figure_path}")
    print(f"n_samples={metrics['n_samples']}")
    print(f"mae_mV={metrics['mae']:.6f}")
    print(f"rmse_mV={metrics['rmse']:.6f}")
    print(f"max_abs_mV={metrics['max_abs']:.6f}")


if __name__ == "__main__":
    main()
