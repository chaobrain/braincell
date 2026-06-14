from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import brainunit as u
import numpy as np
import pandas as pd
from neuron import h


def _find_repo_root(start: Path | None = None) -> Path:
    cwd = Path.cwd().resolve() if start is None else start.resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "braincell").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Run from inside the braincell-ion_dyn repository.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import brainstate
from braincell import mech
from braincell.filter import at

from examples.neuron_compare.cell.pc_ma2024.debug.pc_parameters import (
    PCConfig,
    PCToggles,
    load_pc24_params,
)
from examples.neuron_compare.cell.pc_ma2024.debug.pc_neuron_debug import PC as NeuronPC
from examples.neuron_compare.cell.pc_ma2024.debug.pc_braincell_debug import PC as BrainCellPC


brainstate.environ.set(precision=64)


DT_MS = 0.1
DURATION_MS = 100.0
DELAY_MS = 10.0
STIM_DUR_MS = 80.0
AMP_NA = 0.5
TEMP_C = 36.0
V_INIT_MV = -65.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=Path(__file__).resolve().parent / "results" / "kca2p2_coupling",
        type=Path,
    )
    parser.add_argument("--duration-ms", default=DURATION_MS, type=float)
    parser.add_argument("--dt-ms", default=DT_MS, type=float)
    parser.add_argument(
        "--case",
        action="append",
        help="Run only named case(s). May be repeated.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("full_no_kca2p2_family", _toggles(kca2p2=False), "family"),
        ("full_with_kca2p2_family", _toggles(kca2p2=True), "family"),
        ("all_no_kca2p2_family", _toggles(kca1p1=True, kca2p2=False), "family"),
        ("all_with_kca2p2_family", _toggles(kca1p1=True, kca2p2=True), "family"),
        ("full_with_kca2p2_integration", _toggles(kca2p2=True), "integration"),
        ("all_with_kca2p2_integration", _toggles(kca1p1=True, kca2p2=True), "integration"),
        ("fixed_ca_with_kca2p2_family", _toggles(kca2p2=True, cdp=False), "family"),
        ("all_fixed_ca_no_kca2p2_family", _toggles(kca1p1=True, kca2p2=False, cdp=False), "family"),
        ("all_fixed_ca_with_kca2p2_family", _toggles(kca1p1=True, kca2p2=True, cdp=False), "family"),
        (
            "no_ca_channel_with_kca2p2_family",
            _toggles(kca2p2=True, cav21=False, cav31=False, cav32=False, cav33=False),
            "family",
        ),
        (
            "all_no_ca_channel_with_kca2p2_family",
            _toggles(kca1p1=True, kca2p2=True, cav21=False, cav31=False, cav32=False, cav33=False),
            "family",
        ),
        (
            "no_ca_channel_with_kca2p2_no_cdp_family",
            _toggles(kca2p2=True, cav21=False, cav31=False, cav32=False, cav33=False, cdp=False),
            "family",
        ),
        (
            "all_no_ca_channel_with_kca2p2_no_cdp_family",
            _toggles(
                kca1p1=True,
                kca2p2=True,
                cav21=False,
                cav31=False,
                cav32=False,
                cav33=False,
                cdp=False,
            ),
            "family",
        ),
        (
            "all_only_cav21_with_kca2p2_family",
            _toggles(kca1p1=True, kca2p2=True, cav21=True, cav31=False, cav32=False, cav33=False),
            "family",
        ),
        (
            "all_only_cav31_with_kca2p2_family",
            _toggles(kca1p1=True, kca2p2=True, cav21=False, cav31=True, cav32=False, cav33=False),
            "family",
        ),
        (
            "all_only_cav32_with_kca2p2_family",
            _toggles(kca1p1=True, kca2p2=True, cav21=False, cav31=False, cav32=True, cav33=False),
            "family",
        ),
        (
            "all_only_cav33_with_kca2p2_family",
            _toggles(kca1p1=True, kca2p2=True, cav21=False, cav31=False, cav32=False, cav33=True),
            "family",
        ),
    ]
    if args.case:
        requested = set(args.case)
        cases = [item for item in cases if item[0] in requested]
        missing = requested - {item[0] for item in cases}
        if missing:
            raise ValueError(f"Unknown case(s): {sorted(missing)!r}")

    rows = []
    for case_name, toggles, order in cases:
        print(f"running {case_name}", flush=True)
        result = run_case(case_name, toggles, order, dt_ms=args.dt_ms, duration_ms=args.duration_ms)
        rows.append(result["summary"])
        (out_dir / f"{case_name}.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"wrote {out_dir}", flush=True)
    return 0


def _toggles(**overrides: bool) -> PCToggles:
    base = PCToggles(
        leak=True,
        nav=True,
        kv1p1=True,
        kv1p5=True,
        kv3p3=True,
        kv3p4=True,
        kv4p3=True,
        kir2p3=True,
        kca1p1=False,
        kca2p2=False,
        kca3p1=True,
        cav21=True,
        cav31=True,
        cav32=True,
        cav33=True,
        hcn1=True,
        cdp=True,
    )
    return replace(base, **overrides)


def run_case(case_name: str, toggles: PCToggles, order: str, *, dt_ms: float, duration_ms: float) -> dict[str, Any]:
    params = load_pc24_params()
    config = PCConfig(toggles=toggles, temperature_celsius=TEMP_C, v_init_mV=V_INIT_MV)

    neuron_pc = NeuronPC(params=params, config=config).build()
    braincell_pc = BrainCellPC(params=params, config=config, ion_channel_update_order=order).build()
    try:
        nrn_voltage_probes = neuron_pc.attach_voltage_probes(all_compartments=True, soma=True)
        bc_voltage_probes = braincell_pc.attach_voltage_probes(all_compartments=True, soma=True)
        diag_handles = attach_soma_diagnostics(neuron_pc, braincell_pc, toggles)

        stim = h.IClamp(neuron_pc.root_soma(0.5))
        stim.delay = DELAY_MS
        stim.dur = STIM_DUR_MS
        stim.amp = AMP_NA
        h.cvode_active(0)
        h.dt = dt_ms
        h.steps_per_ms = 1.0 / h.dt
        h.celsius = config.temperature_celsius
        h.tstop = duration_ms
        h.v_init = config.v_init_mV
        t_neuron = h.Vector().record(h._ref_t)
        h.finitialize(h.v_init)
        neuron_init = sample_neuron_soma(neuron_pc, toggles)
        h.run()

        braincell_pc.cell.place(
            at("soma", 0.5),
            mech.CurrentClamp(delay=DELAY_MS * u.ms, durations=STIM_DUR_MS * u.ms, amplitudes=AMP_NA * u.nA),
        )
        braincell_pc.cell.init_state()
        braincell_pc.cell.reset_state()
        braincell_init = sample_braincell_init(braincell_pc)
        bc_run = braincell_pc.cell.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)

        nrn_v = neuron_pc.collect_voltage_results(nrn_voltage_probes)
        bc_v = braincell_pc.collect_voltage_results(bc_voltage_probes, bc_run)
        metrics = voltage_metrics(nrn_v, bc_v, t_neuron, dt_ms=dt_ms, duration_ms=duration_ms)
        diag_metrics = diagnostic_metrics(diag_handles, bc_run, dt_ms=dt_ms, duration_ms=duration_ms)

        summary = {
            "case": case_name,
            "ion_channel_update_order": order,
            "kca2p2": bool(toggles.kca2p2),
            "cdp": bool(toggles.cdp),
            "ca_channels": bool(toggles.cav21 or toggles.cav31 or toggles.cav32 or toggles.cav33),
            **metrics["summary"],
            **diag_metrics["summary"],
        }
        return {
            "summary": summary,
            "max_compartment": metrics["max_compartment"],
            "threshold_crossings": metrics["threshold_crossings"],
            "neuron_init": neuron_init,
            "braincell_init": braincell_init,
            "diagnostics": diag_metrics["diagnostics"],
        }
    finally:
        neuron_pc.cleanup()


def attach_soma_diagnostics(neuron_pc: NeuronPC, braincell_pc: BrainCellPC, toggles: PCToggles) -> dict[str, Any]:
    handles: dict[str, Any] = {}
    nrn_seg = neuron_pc.root_soma(0.5)
    if toggles.cdp or any((toggles.cav21, toggles.cav31, toggles.cav32, toggles.cav33, toggles.kca2p2, toggles.kca3p1)):
        if hasattr(nrn_seg, "_ref_cai"):
            handles["cai_neuron"] = h.Vector().record(nrn_seg._ref_cai)
            braincell_pc.cell.place(at("soma", 0.5), mech.MechanismProbe(mechanism="ca", field="Ci", name="soma_ca_Ci"))
            handles["cai_braincell_trace"] = "soma_ca_Ci"
        if hasattr(nrn_seg, "_ref_eca"):
            handles["eca_neuron"] = h.Vector().record(nrn_seg._ref_eca)
            braincell_pc.cell.place(at("soma", 0.5), mech.MechanismProbe(mechanism="ca", field="E", name="soma_ca_E"))
            handles["eca_braincell_trace"] = "soma_ca_E"
    if toggles.kca2p2:
        mech_nrn = nrn_seg.Kca2p2_MA24_PC
        handles["kca2p2_current_neuron"] = h.Vector().record(mech_nrn._ref_ik)
        braincell_pc.cell.place(
            at("soma", 0.5),
            mech.CurrentProbe(ion="k", mechanism="Kca2p2_MA2024_PC", name="soma_kca2p2_current"),
        )
        handles["kca2p2_current_braincell_trace"] = "soma_kca2p2_current"
        for neuron_field, braincell_field in (("c2", "C2"), ("c3", "C3"), ("c4", "C4"), ("o1", "O1"), ("o2", "O2")):
            handles[f"kca2p2_{braincell_field}_neuron"] = h.Vector().record(getattr(mech_nrn, f"_ref_{neuron_field}"))
            braincell_pc.cell.place(
                at("soma", 0.5),
                mech.MechanismProbe(
                    mechanism="Kca2p2_MA2024_PC",
                    field=braincell_field,
                    name=f"soma_kca2p2_{braincell_field}",
                ),
            )
            handles[f"kca2p2_{braincell_field}_braincell_trace"] = f"soma_kca2p2_{braincell_field}"
    return handles


def sample_neuron_soma(neuron_pc: NeuronPC, toggles: PCToggles) -> dict[str, float | None]:
    seg = neuron_pc.root_soma(0.5)
    out = {
        "v_mV": _maybe_float(seg, "v"),
        "cai_mM": _maybe_float(seg, "cai"),
        "eca_mV": _maybe_float(seg, "eca"),
    }
    if toggles.kca2p2:
        mech_nrn = seg.Kca2p2_MA24_PC
        for field in ("c1", "c2", "c3", "c4", "o1", "o2", "ik"):
            out[f"kca2p2_{field}"] = _maybe_float(mech_nrn, field)
    return out


def sample_braincell_init(braincell_pc: BrainCellPC) -> dict[str, float | None]:
    samples = braincell_pc.cell.sample_probes()
    out: dict[str, float | None] = {}
    for name, value in samples.items():
        out[name] = quantity_scalar(value)
    return out


def voltage_metrics(nrn_v: dict[str, Any], bc_v: dict[str, Any], t_neuron: Any, *, dt_ms: float, duration_ms: float) -> dict[str, Any]:
    reference_time_ms = np.round(np.arange(0.0, duration_ms, dt_ms, dtype=float), decimals=12)
    neuron_soma = trim_neuron_trace(nrn_v["soma_voltage_mV"], reference_time_ms)
    neuron_comp = nrn_v["compartment_voltage_mV"]
    if neuron_comp.shape[0] == reference_time_ms.shape[0] + 1:
        neuron_comp = neuron_comp[1:, :]
    braincell_soma = np.asarray(bc_v["soma_voltage_mV"], dtype=float)
    braincell_comp = np.asarray(bc_v["compartment_voltage_mV"], dtype=float)
    n_time = min(neuron_comp.shape[0], braincell_comp.shape[0], reference_time_ms.shape[0])
    neuron_soma = neuron_soma[:n_time]
    braincell_soma = braincell_soma[:n_time]
    neuron_comp = neuron_comp[:n_time, :]
    braincell_comp = braincell_comp[:n_time, :]
    time = reference_time_ms[:n_time]

    pair_table = pd.merge(
        bc_v["compartment_table"],
        nrn_v["compartment_table"],
        on=["branch_index", "branch_type", "local_index"],
        suffixes=("_braincell", "_neuron"),
    )
    max_rows = []
    global_max = -1.0
    global_info: dict[str, Any] = {}
    for row in pair_table.itertuples(index=False):
        bc_idx = int(row.compartment_index_braincell)
        nrn_idx = int(row.compartment_index_neuron)
        delta = braincell_comp[:, bc_idx] - neuron_comp[:, nrn_idx]
        abs_delta = np.abs(delta)
        idx = int(np.argmax(abs_delta))
        value = float(abs_delta[idx])
        item = {
            "branch_index": int(row.branch_index),
            "branch_name": row.branch_name_braincell,
            "branch_type": row.branch_type,
            "local_index": int(row.local_index),
            "time_ms": float(time[idx]),
            "max_abs_mV": value,
            "signed_delta_mV": float(delta[idx]),
        }
        max_rows.append(item)
        if value > global_max:
            global_max = value
            global_info = item

    soma_delta = braincell_soma - neuron_soma
    soma_abs = np.abs(soma_delta)
    soma_idx = int(np.argmax(soma_abs))
    comp_abs = np.abs(braincell_comp[:, pair_table["compartment_index_braincell"].to_numpy(dtype=int)] - neuron_comp[:, pair_table["compartment_index_neuron"].to_numpy(dtype=int)])
    comp_max_by_time = comp_abs.max(axis=1)
    return {
        "summary": {
            "soma_max_abs_mV": float(soma_abs[soma_idx]),
            "soma_max_time_ms": float(time[soma_idx]),
            "soma_signed_delta_mV": float(soma_delta[soma_idx]),
            "compartment_max_abs_mV": float(global_max),
            "compartment_max_time_ms": float(global_info.get("time_ms", np.nan)),
            "compartment_max_branch_index": int(global_info.get("branch_index", -1)),
            "compartment_max_local_index": int(global_info.get("local_index", -1)),
            "compartment_max_branch_name": global_info.get("branch_name", ""),
        },
        "max_compartment": global_info,
        "threshold_crossings": {
            str(threshold): first_crossing(time, comp_max_by_time, threshold)
            for threshold in (0.1, 1.0, 5.0)
        },
    }


def diagnostic_metrics(handles: dict[str, Any], bc_run: Any, *, dt_ms: float, duration_ms: float) -> dict[str, Any]:
    summary: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    reference_time_ms = np.round(np.arange(0.0, duration_ms, dt_ms, dtype=float), decimals=12)
    for key, neuron_vec in handles.items():
        if not key.endswith("_neuron"):
            continue
        prefix = key.removesuffix("_neuron")
        trace_key = f"{prefix}_braincell_trace"
        if trace_key not in handles:
            continue
        neuron_values = trim_neuron_trace(neuron_vec, reference_time_ms)
        braincell_values = trace_to_numpy(bc_run.traces[handles[trace_key]], key=prefix)
        n = min(neuron_values.shape[0], braincell_values.shape[0])
        if n == 0:
            continue
        delta = braincell_values[:n] - neuron_values[:n]
        abs_delta = np.abs(delta)
        idx = int(np.argmax(abs_delta))
        diagnostics[prefix] = {
            "max_abs": float(abs_delta[idx]),
            "time_ms": float(reference_time_ms[idx]),
            "signed_delta": float(delta[idx]),
        }
        summary[f"{prefix}_max_abs"] = float(abs_delta[idx])
        summary[f"{prefix}_max_time_ms"] = float(reference_time_ms[idx])
    return {"summary": summary, "diagnostics": diagnostics}


def trim_neuron_trace(values: Any, reference_time_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] == reference_time_ms.shape[0] + 1:
        return values[1:]
    return values


def trace_to_numpy(value: Any, *, key: str) -> np.ndarray:
    if key == "eca":
        return np.asarray(value.to_decimal(u.mV), dtype=float).reshape(-1)
    if key == "cai":
        return np.asarray(value.to_decimal(u.mM), dtype=float).reshape(-1)
    if key == "kca2p2_current":
        return -np.asarray(value.to_decimal(u.mA / (u.cm**2)), dtype=float).reshape(-1)
    if hasattr(value, "to_decimal"):
        return np.asarray(value.to_decimal(u.mM), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def first_crossing(time: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    hits = np.flatnonzero(values > threshold)
    if len(hits) == 0:
        return None
    return float(time[int(hits[0])])


def quantity_scalar(value: Any) -> float | None:
    try:
        if hasattr(value, "to_decimal"):
            for unit in (u.mV, u.mM, u.mA / (u.cm**2)):
                try:
                    arr = np.asarray(value.to_decimal(unit), dtype=float).reshape(-1)
                    return float(arr[0])
                except Exception:
                    pass
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0])
    except Exception:
        return None


def _maybe_float(owner: Any, field: str) -> float | None:
    if not hasattr(owner, field):
        return None
    try:
        return float(getattr(owner, field))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
