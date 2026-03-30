# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.

import importlib.util
import math
from pathlib import Path

import numpy as np


def _load_template_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "examples" / "ion_test" / "single_ion_compare_template.py"
    spec = importlib.util.spec_from_file_location("single_ion_compare_template", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_helper_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "examples" / "ion_test" / "single_ion_compare_helper.py"
    spec = importlib.util.spec_from_file_location("single_ion_compare_helper", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_error_metrics_squeezes_and_avoids_broadcast():
    mod = _load_template_module()
    ref = np.array([1.0, 2.0, 3.0], dtype=float)
    pred = np.array([[1.1], [1.9], [3.2]], dtype=float)

    m = mod.compute_error_metrics(ref, pred)
    expected_err = np.array([-0.1, 0.1, -0.2], dtype=float)

    assert m["n_samples"] == 3
    assert math.isclose(m["mae"], np.mean(np.abs(expected_err)))
    assert math.isclose(m["rmse"], np.sqrt(np.mean(expected_err ** 2)))
    assert math.isclose(m["max_abs"], np.max(np.abs(expected_err)))
    assert math.isclose(m["mean_bias"], np.mean(expected_err))


def test_expand_cases_filters_invalid_duration():
    mod = _load_template_module()
    cases = mod.expand_cases(
        {
            "tstop_ms": [20.0],
            "amp_nA": [0.01],
            "delay_ms": [0.0, 15.0],
            "dur_ms": [10.0],
        }
    )
    # only delay=0,dur=10 is valid for tstop=20
    assert len(cases) == 1
    assert cases[0]["delay_ms"] == 0.0


def test_aggregate_case_metrics_case_weighted_average():
    mod = _load_template_module()
    successful_cases = [
        {
            "case_id": 0,
            "metrics": {
                "v_mV": {"mae": 1.0, "rmse": 2.0, "max_abs": 3.0, "rel_mae_pct": 1.0},
                "ik_mA_cm2": {"mae": 2.0, "rmse": 4.0, "max_abs": 5.0, "rel_mae_pct": 2.0},
            },
        },
        {
            "case_id": 1,
            "metrics": {
                "v_mV": {"mae": 3.0, "rmse": 6.0, "max_abs": 7.0, "rel_mae_pct": 3.0},
                "ik_mA_cm2": {"mae": 4.0, "rmse": 8.0, "max_abs": 9.0, "rel_mae_pct": 4.0},
            },
        },
    ]
    agg = mod.aggregate_case_metrics(successful_cases, ["v_mV", "ik_mA_cm2"])

    assert agg["n_success_cases"] == 2
    assert math.isclose(agg["per_observable"]["v_mV"]["rmse_mean"], 4.0)
    assert math.isclose(agg["per_observable"]["ik_mA_cm2"]["mae_mean"], 3.0)
    # case mean rmse: case0=(2+4)/2=3, case1=(6+8)/2=7
    assert math.isclose(agg["overall"]["case_weighted_mean_rmse"], 5.0)
    assert agg["worst_case_by_mean_rmse"]["case_id"] == 1


def test_amp_density_conversion():
    mod = _load_template_module()
    density = mod.amp_nA_to_density_nA_cm2(amp_nA=0.01, L_um=10.0, diam_um=100.0 / math.pi)
    # area = pi * 10 * (100/pi) * 1e-8 = 1e-5 cm2, so density = 0.01 / 1e-5 = 1000 nA/cm2
    assert math.isclose(density, 1000.0, rel_tol=1e-12, abs_tol=1e-12)


def test_step_current_density_boundaries():
    mod = _load_template_module()
    amp = 123.0
    delay = 10.0
    dur = 20.0
    assert mod.step_current_density_nA_cm2(9.999, amp, delay, dur) == 0.0
    assert mod.step_current_density_nA_cm2(10.0, amp, delay, dur) == amp
    assert mod.step_current_density_nA_cm2(29.999, amp, delay, dur) == amp
    assert mod.step_current_density_nA_cm2(30.0, amp, delay, dur) == 0.0


def test_run_writes_expected_outputs_for_single_case(tmp_path, monkeypatch):
    mod = _load_template_module()

    cfg = mod.merge_dicts(
        mod.DEFAULT_CONFIG,
        {
            "sweep_grid": {
                "tstop_ms": [25.0],
                "amp_nA": [0.01],
                "delay_ms": [5.0],
                "dur_ms": [10.0],
            }
        },
    )

    def fake_compare_case(_cfg, case, observables):
        t_ms = np.array([0.0, 0.025, 0.050], dtype=float)
        neuron = {}
        braincell = {}
        metrics = {}
        for idx, obs in enumerate(observables):
            ref = np.array([0.0, 1.0, 2.0], dtype=float) + idx
            pred = ref + 0.1
            neuron[obs] = ref
            braincell[obs] = pred
            metrics[obs] = mod.compute_error_metrics(ref, pred)
        return {
            "case_id": int(case["case_id"]),
            "case": dict(case),
            "time_ms": t_ms,
            "metrics": metrics,
            "neuron": neuron,
            "braincell": braincell,
        }

    def fake_save_case_plot(out_path, **_kwargs):
        out_path.write_text("synthetic plot", encoding="utf-8")

    monkeypatch.setattr(mod, "compare_case", fake_compare_case)
    monkeypatch.setattr(mod, "save_case_plot", fake_save_case_plot)

    out_dir = tmp_path / "results"
    exit_code = mod.run(cfg, out_dir=out_dir, do_plot=True)

    assert exit_code == 0
    assert (out_dir / "cases.json").exists()
    assert (out_dir / "case_metrics.csv").exists()
    assert (out_dir / "aggregate.json").exists()
    assert (out_dir / "plots" / "case_000.png").exists()


def test_helper_module_reexports_core_interface():
    helper = _load_helper_module()

    assert callable(helper.main)
    assert "global" in helper.DEFAULT_CONFIG
    cases = helper.expand_cases(
        {
            "tstop_ms": [20.0],
            "amp_nA": [0.01],
            "delay_ms": [0.0],
            "dur_ms": [10.0],
        }
    )
    assert len(cases) == 1
