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

"""Single-ion comparison template between NEURON and braincell.

This template currently targets the Kv channel:
- NEURON side: `Kv` mechanism from compiled mod files.
- braincell side: `braincell.channel.IK_Kv_test`.

The script sweeps over a Cartesian grid of simulation settings and reports
per-case + aggregated errors for selected observables.
"""



import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Keep the template CPU-safe by default in mixed CUDA environments.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


DEFAULT_CONFIG: Dict[str, Any] = {
    "global": {
        "dt_ms": 0.025,
        "v_init_mV": -65.0,
        "temperature_celsius": 25.0,
    },
    "geometry": {
        "L_um": 10.0,
        "diam_um": 100.0 / math.pi,
        "cm_uF_cm2": 1.0,
        "Ra_ohm_cm": 100.0,
    },
    "neuron": {
        "mod_dir": ".",
        "mechanism": "Kv",
        "pas": {
            "g_S_cm2": 1e-4,
            "e_mV": -65.0,
        },
        "channel_params": {
            "gbar_S_cm2": 0.0,
        },
        "ek_mV": -80.0,
        "gate_var": "n",
    },
    "braincell": {
        "solver": "rk4",
        "ion_E_mV": -80.0,
        "leak": {
            "g_S_cm2": 1e-4,
            "e_mV": -65.0,
        },
        "channel_params": {
            "g_max_S_cm2": 0.0,
        },
    },
    "sweep_grid": {
        "tstop_ms": [100.0],
        "amp_nA": [0.01],
        "delay_ms": [0.0],
        "dur_ms": [100.0],
    },
    "metrics": {
        "observables": ["v_mV", "ik_mA_cm2", "gate_n"],
    },
}

EPS = 1e-10
_LOADED_MECHANISMS: set[Tuple[str, str]] = set()


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = merge_dicts(DEFAULT_CONFIG, user_cfg)
    mod_dir = Path(cfg["neuron"]["mod_dir"])
    if not mod_dir.is_absolute():
        mod_dir = (config_path.parent / mod_dir).resolve()
    cfg["neuron"]["mod_dir"] = str(mod_dir)
    return cfg


def ensure_1d(arr: Any, name: str) -> np.ndarray:
    value = np.asarray(arr)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape}")
    return value.astype(float, copy=False)


def align_pair(ref: Any, pred: Any, ref_name: str, pred_name: str) -> Tuple[np.ndarray, np.ndarray]:
    ref_1d = ensure_1d(ref, ref_name)
    pred_1d = ensure_1d(pred, pred_name)
    n = min(ref_1d.shape[0], pred_1d.shape[0])
    if n == 0:
        raise ValueError(f"Empty trace after alignment: {ref_name}, {pred_name}")
    return ref_1d[:n], pred_1d[:n]


def compute_error_metrics(ref: Any, pred: Any) -> Dict[str, float]:
    ref_1d, pred_1d = align_pair(ref, pred, "ref", "pred")
    err = ref_1d - pred_1d
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_bias": float(np.mean(err)),
        "rel_mae_pct": float(np.mean(np.abs(err) / (np.abs(ref_1d) + EPS)) * 100.0),
        "n_samples": int(err.shape[0]),
    }


def amp_nA_to_density_nA_cm2(amp_nA: float, L_um: float, diam_um: float) -> float:
    area_cm2 = math.pi * L_um * diam_um * 1e-8
    if area_cm2 <= 0:
        raise ValueError(f"Invalid membrane area: {area_cm2}")
    return amp_nA / area_cm2


def step_current_density_nA_cm2(t_ms: float, amp_nA_cm2: float, delay_ms: float, dur_ms: float) -> float:
    """Return step current density at scalar time t."""
    return float(amp_nA_cm2 if (delay_ms <= t_ms < (delay_ms + dur_ms)) else 0.0)


def expand_cases(sweep_grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    required = ["tstop_ms", "amp_nA", "delay_ms", "dur_ms"]
    for key in required:
        if key not in sweep_grid:
            raise ValueError(f"sweep_grid is missing key: {key}")

    cases: List[Dict[str, float]] = []
    case_id = 0
    for tstop_ms, amp_nA, delay_ms, dur_ms in itertools.product(
        sweep_grid["tstop_ms"],
        sweep_grid["amp_nA"],
        sweep_grid["delay_ms"],
        sweep_grid["dur_ms"],
    ):
        tstop_ms = float(tstop_ms)
        amp_nA = float(amp_nA)
        delay_ms = float(delay_ms)
        dur_ms = float(dur_ms)
        if delay_ms + dur_ms > tstop_ms:
            continue
        cases.append(
            {
                "case_id": case_id,
                "tstop_ms": tstop_ms,
                "amp_nA": amp_nA,
                "delay_ms": delay_ms,
                "dur_ms": dur_ms,
            }
        )
        case_id += 1
    return cases


def ensure_mechanism_available(mod_dir: Path, mechanism: str) -> None:
    from neuron import h, load_mechanisms

    key = (str(mod_dir.resolve()), mechanism)
    if key in _LOADED_MECHANISMS:
        return

    probe = h.Section(name="mech_probe")
    try:
        probe.insert(mechanism)
        _LOADED_MECHANISMS.add(key)
        return
    except Exception:
        pass
    finally:
        h.delete_section(sec=probe)

    try:
        load_mechanisms(str(mod_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load mechanisms from {mod_dir}. "
            f"Please compile .mod files with nrnivmodl first."
        ) from exc

    probe = h.Section(name="mech_probe")
    try:
        probe.insert(mechanism)
    except Exception as exc:
        raise RuntimeError(
            f"Mechanism '{mechanism}' is still unavailable after load_mechanisms({mod_dir})."
        ) from exc
    finally:
        h.delete_section(sec=probe)

    _LOADED_MECHANISMS.add(key)


def build_neuron_model(cfg: Dict[str, Any], case: Dict[str, float]) -> Dict[str, Any]:
    from neuron import h

    h.load_file("stdrun.hoc")

    gcfg = cfg["global"]
    geom = cfg["geometry"]
    ncfg = cfg["neuron"]
    mechanism = ncfg["mechanism"]

    ensure_mechanism_available(Path(ncfg["mod_dir"]), mechanism)

    soma = h.Section(name=f"soma_case_{case['case_id']}")
    soma.L = float(geom["L_um"])
    soma.diam = float(geom["diam_um"])
    soma.nseg = 1
    soma.cm = float(geom["cm_uF_cm2"])
    soma.Ra = float(geom["Ra_ohm_cm"])

    soma.insert("pas")
    for seg in soma:
        seg.pas.g = float(ncfg["pas"]["g_S_cm2"])
        seg.pas.e = float(ncfg["pas"]["e_mV"])

    soma.insert(mechanism)
    channel = ncfg["channel_params"]
    for seg in soma:
        mech_obj = getattr(seg, mechanism)
        if mechanism == "Kv":
            if "gbar_S_cm2" in channel:
                mech_obj.gbar = float(channel["gbar_S_cm2"])
        else:
            raise ValueError(f"Unsupported mechanism in template: {mechanism}")

    if "ek_mV" in ncfg:
        soma.ek = float(ncfg["ek_mV"])

    stim = h.IClamp(soma(0.5))
    stim.delay = float(case["delay_ms"])
    stim.dur = float(case["dur_ms"])
    stim.amp = float(case["amp_nA"])

    gate_var = ncfg.get("gate_var", "n")
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    ik_outward_vec = h.Vector().record(soma(0.5)._ref_ik)
    gate_ref = getattr(getattr(soma(0.5), mechanism), f"_ref_{gate_var}")
    gate_vec = h.Vector().record(gate_ref)

    h.cvode_active(0)
    h.dt = float(gcfg["dt_ms"])
    h.celsius = float(gcfg["temperature_celsius"])
    h.finitialize(float(gcfg["v_init_mV"]))
    h.tstop = float(case["tstop_ms"])

    return {
        "h": h,
        "soma": soma,
        "stim": stim,
        "t_vec": t_vec,
        "v_vec": v_vec,
        "ik_outward_vec": ik_outward_vec,
        "gate_vec": gate_vec,
    }


def run_case_neuron(cfg: Dict[str, Any], case: Dict[str, float]) -> Dict[str, np.ndarray]:
    model = build_neuron_model(cfg, case)
    h = model["h"]
    soma = model["soma"]
    try:
        h.run()
        t_ms = ensure_1d(model["t_vec"], "neuron_t_ms")
        v_mV = ensure_1d(model["v_vec"], "neuron_v_mV")
        ik_outward = ensure_1d(model["ik_outward_vec"], "neuron_ik_outward")
        gate_n = ensure_1d(model["gate_vec"], "neuron_gate")
    finally:
        h.delete_section(sec=soma)

    # NEURON `ik` is outward-positive. Convert to channel contribution form g*(E-V).
    ik_mA_cm2 = -ik_outward
    return {
        "t_ms": t_ms,
        "v_mV": v_mV,
        "ik_mA_cm2": ik_mA_cm2,
        "gate_n": gate_n,
    }


def build_braincell_model(cfg: Dict[str, Any]):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import braintools
    import brainunit as u
    import braincell

    gcfg = cfg["global"]
    bcfg = cfg["braincell"]
    ch = bcfg["channel_params"]
    leak = bcfg["leak"]
    neuron_gbar = float(cfg["neuron"]["channel_params"]["gbar_S_cm2"])
    g_max = float(ch.get("g_max_S_cm2", neuron_gbar))

    class KvCell(braincell.SingleCompartment):
        def __init__(self, size: int = 1):
            v_init = float(gcfg["v_init_mV"]) * u.mV
            super().__init__(
                size,
                solver=bcfg["solver"],
                V_initializer=braintools.init.Uniform(v_init, v_init),
            )

            self.k = braincell.ion.PotassiumFixed(size, E=float(bcfg["ion_E_mV"]) * u.mV)
            self.k.add(
                IK=braincell.channel.IK_Kv_test(
                    size,
                    g_max=g_max * (u.siemens / (u.cm ** 2)),
                )
            )

            self.IL = braincell.channel.IL(
                size,
                E=float(leak["e_mV"]) * u.mV,
                g_max=float(leak["g_S_cm2"]) * (u.siemens / (u.cm ** 2)),
            )

    cell = KvCell(1)
    return cell


def run_case_braincell(cfg: Dict[str, Any], case: Dict[str, float]) -> Dict[str, np.ndarray]:
    import brainstate
    import brainunit as u

    geom = cfg["geometry"]
    dt_ms = float(cfg["global"]["dt_ms"])
    amp_nA = float(case["amp_nA"])
    delay_ms = float(case["delay_ms"])
    dur_ms = float(case["dur_ms"])
    density_nA_cm2 = amp_nA_to_density_nA_cm2(
        amp_nA=amp_nA,
        L_um=float(geom["L_um"]),
        diam_um=float(geom["diam_um"]),
    )

    cell = build_braincell_model(cfg)
    cell.init_state()
    cell.reset_state()

    def step_fun(t):
        t_ms = t / u.ms
        active = u.math.logical_and(t_ms >= delay_ms, t_ms < (delay_ms + dur_ms))
        i_nA_cm2 = u.math.where(active, density_nA_cm2, 0.0)
        with brainstate.environ.context(t=t):
            cell.update(i_nA_cm2 * u.nA / (u.cm ** 2))
        v_now = cell.V.value
        ik_now = cell.k.current(v_now)
        gate_now = cell.k.IK.n.value
        return v_now, ik_now, gate_now

    with brainstate.environ.context(dt=dt_ms * u.ms):
        times = u.math.arange(0.0 * u.ms, float(case["tstop_ms"]) * u.ms, brainstate.environ.get_dt())
        v_seq, ik_seq, gate_seq = brainstate.transform.for_loop(step_fun, times)

    return {
        "t_ms": ensure_1d(times / u.ms, "braincell_t_ms"),
        "v_mV": ensure_1d(v_seq / u.mV, "braincell_v_mV"),
        "ik_mA_cm2": ensure_1d(ik_seq / (u.mA / (u.cm ** 2)), "braincell_ik_mA_cm2"),
        "gate_n": ensure_1d(gate_seq, "braincell_gate"),
    }


def aggregate_case_metrics(
    successful_cases: List[Dict[str, Any]], observables: Iterable[str]
) -> Dict[str, Any]:
    observables = list(observables)
    if not successful_cases:
        return {
            "n_success_cases": 0,
            "per_observable": {},
            "overall": {},
            "worst_case_by_mean_rmse": None,
        }

    per_observable: Dict[str, Dict[str, float]] = {}
    for obs in observables:
        obs_rows = [c["metrics"][obs] for c in successful_cases if obs in c["metrics"]]
        if not obs_rows:
            continue
        per_observable[obs] = {
            "mae_mean": float(np.mean([r["mae"] for r in obs_rows])),
            "rmse_mean": float(np.mean([r["rmse"] for r in obs_rows])),
            "max_abs_max": float(np.max([r["max_abs"] for r in obs_rows])),
            "rel_mae_pct_mean": float(np.mean([r["rel_mae_pct"] for r in obs_rows])),
        }

    case_mean_rmse: List[Tuple[int, float]] = []
    case_mean_mae: List[Tuple[int, float]] = []
    for case in successful_cases:
        rmses = [case["metrics"][obs]["rmse"] for obs in observables if obs in case["metrics"]]
        maes = [case["metrics"][obs]["mae"] for obs in observables if obs in case["metrics"]]
        if not rmses or not maes:
            continue
        case_mean_rmse.append((case["case_id"], float(np.mean(rmses))))
        case_mean_mae.append((case["case_id"], float(np.mean(maes))))

    worst_case = None
    if case_mean_rmse:
        worst_case_id, worst_rmse = max(case_mean_rmse, key=lambda x: x[1])
        worst_case = {"case_id": worst_case_id, "mean_rmse": worst_rmse}

    overall = {
        "case_weighted_mean_rmse": float(np.mean([x[1] for x in case_mean_rmse])) if case_mean_rmse else None,
        "case_weighted_mean_mae": float(np.mean([x[1] for x in case_mean_mae])) if case_mean_mae else None,
    }
    return {
        "n_success_cases": len(successful_cases),
        "per_observable": per_observable,
        "overall": overall,
        "worst_case_by_mean_rmse": worst_case,
    }


def save_case_plot(
    out_path: Path,
    case: Dict[str, float],
    t_ms: np.ndarray,
    neuron_trace: Dict[str, np.ndarray],
    braincell_trace: Dict[str, np.ndarray],
    observables: List[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "v_mV": "V (mV)",
        "ik_mA_cm2": "I_k (mA/cm2)",
        "gate_n": "Gate n",
    }

    fig, axes = plt.subplots(len(observables), 1, figsize=(10, 3.0 * len(observables)), sharex=True)
    if len(observables) == 1:
        axes = [axes]

    for idx, obs in enumerate(observables):
        ax = axes[idx]
        ax.plot(t_ms, neuron_trace[obs], label="NEURON", linewidth=1.5)
        ax.plot(t_ms, braincell_trace[obs], label="braincell", linewidth=1.2, linestyle="--")
        ax.set_ylabel(labels.get(obs, obs))
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.set_title(
                f"Case {case['case_id']} | tstop={case['tstop_ms']} ms, "
                f"amp={case['amp_nA']} nA, delay={case['delay_ms']} ms, dur={case['dur_ms']} ms"
            )
    axes[-1].set_xlabel("Time (ms)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_case(
    cfg: Dict[str, Any], case: Dict[str, float], observables: List[str]
) -> Dict[str, Any]:
    neuron_trace = run_case_neuron(cfg, case)
    braincell_trace = run_case_braincell(cfg, case)

    aligned_neuron: Dict[str, np.ndarray] = {}
    aligned_braincell: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    t_ref, t_pred = align_pair(neuron_trace["t_ms"], braincell_trace["t_ms"], "neuron_t_ms", "braincell_t_ms")
    t_ms = t_ref[: min(t_ref.shape[0], t_pred.shape[0])]
    if t_ms.shape[0] == 0:
        raise ValueError("No aligned time points")

    for obs in observables:
        if obs not in neuron_trace or obs not in braincell_trace:
            raise ValueError(f"Observable '{obs}' is missing in traces.")
        ref, pred = align_pair(neuron_trace[obs], braincell_trace[obs], f"neuron_{obs}", f"braincell_{obs}")
        n = min(t_ms.shape[0], ref.shape[0], pred.shape[0])
        aligned_neuron[obs] = ref[:n]
        aligned_braincell[obs] = pred[:n]
        metrics[obs] = compute_error_metrics(aligned_neuron[obs], aligned_braincell[obs])
        t_ms = t_ms[:n]

    return {
        "case_id": int(case["case_id"]),
        "case": dict(case),
        "time_ms": t_ms,
        "metrics": metrics,
        "neuron": aligned_neuron,
        "braincell": aligned_braincell,
    }


def write_case_metrics_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "case_id",
        "status",
        "tstop_ms",
        "amp_nA",
        "delay_ms",
        "dur_ms",
        "observable",
        "n_samples",
        "mae",
        "rmse",
        "max_abs",
        "mean_bias",
        "rel_mae_pct",
        "error_message",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(cfg: Dict[str, Any], out_dir: Path, do_plot: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    if do_plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    observables = list(cfg.get("metrics", {}).get("observables", ["v_mV", "ik_mA_cm2", "gate_n"]))
    cases = expand_cases(cfg["sweep_grid"])
    if not cases:
        raise ValueError("No valid cases generated from sweep_grid.")

    (out_dir / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows: List[Dict[str, Any]] = []
    success_cases: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    for case in cases:
        try:
            result = compare_case(cfg, case, observables)
            success_cases.append(result)

            for obs in observables:
                m = result["metrics"][obs]
                csv_rows.append(
                    {
                        "case_id": case["case_id"],
                        "status": "ok",
                        "tstop_ms": case["tstop_ms"],
                        "amp_nA": case["amp_nA"],
                        "delay_ms": case["delay_ms"],
                        "dur_ms": case["dur_ms"],
                        "observable": obs,
                        "n_samples": m["n_samples"],
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "max_abs": m["max_abs"],
                        "mean_bias": m["mean_bias"],
                        "rel_mae_pct": m["rel_mae_pct"],
                        "error_message": "",
                    }
                )

            if do_plot:
                save_case_plot(
                    out_path=plots_dir / f"case_{case['case_id']:03d}.png",
                    case=case,
                    t_ms=result["time_ms"],
                    neuron_trace=result["neuron"],
                    braincell_trace=result["braincell"],
                    observables=observables,
                )
        except Exception as exc:
            failed_cases.append(
                {
                    "case_id": case["case_id"],
                    "error_message": str(exc),
                    "case": dict(case),
                }
            )
            csv_rows.append(
                {
                    "case_id": case["case_id"],
                    "status": "failed",
                    "tstop_ms": case["tstop_ms"],
                    "amp_nA": case["amp_nA"],
                    "delay_ms": case["delay_ms"],
                    "dur_ms": case["dur_ms"],
                    "observable": "",
                    "n_samples": "",
                    "mae": "",
                    "rmse": "",
                    "max_abs": "",
                    "mean_bias": "",
                    "rel_mae_pct": "",
                    "error_message": str(exc),
                }
            )

    write_case_metrics_csv(csv_rows, out_dir / "case_metrics.csv")

    aggregate = aggregate_case_metrics(success_cases, observables)
    aggregate["n_total_cases"] = len(cases)
    aggregate["n_failed_cases"] = len(failed_cases)
    aggregate["failed_cases"] = failed_cases
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Finished {len(cases)} cases: {len(success_cases)} success, {len(failed_cases)} failed.")
    if aggregate.get("overall"):
        print(json.dumps(aggregate["overall"], indent=2, ensure_ascii=False))
    print(f"Outputs written to: {out_dir}")
    return 0 if len(success_cases) > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-ion NEURON vs braincell comparison helper.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--out", default=None, help="Output directory (default: <config_dir>/results).")
    parser.set_defaults(plot=True)
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-case comparison plots.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    out_dir = Path(args.out).resolve() if args.out else (config_path.parent / "results").resolve()
    return run(cfg, out_dir=out_dir, do_plot=bool(args.plot))


if __name__ == "__main__":
    raise SystemExit(main())
