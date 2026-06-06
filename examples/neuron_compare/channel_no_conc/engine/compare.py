"""Compare one channel_no_conc case between braincell and NEURON."""



from typing import Any

import numpy as np

try:
    from .braincell_runner import run_case as run_braincell_case
    from .experiment_schema import ChannelNoConcCase
    from .metrics import ensure_1d, metric_record_for_pair
    from .neuron_runner import run_case as run_neuron_case
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from braincell_runner import run_case as run_braincell_case  # type: ignore
    from experiment_schema import ChannelNoConcCase  # type: ignore
    from metrics import ensure_1d, metric_record_for_pair  # type: ignore
    from neuron_runner import run_case as run_neuron_case  # type: ignore


def compare_case(case: ChannelNoConcCase) -> dict[str, Any]:
    mapping_spec = case.mapping_spec
    braincell_result = _normalize_runner_result(
        run_braincell_case(case),
        side="braincell",
        expected_gate_names=mapping_spec.braincell.gate_names,
    )
    neuron_result = _normalize_runner_result(
        run_neuron_case(case),
        side="neuron",
        expected_gate_names=mapping_spec.neuron.gate_names,
    )
    neuron_trimmed = _align_time_axes(
        braincell_result=braincell_result,
        neuron_result=neuron_result,
        dt_ms=float(case.simulation.dt_ms),
    )
    aligned_current = _align_current_traces(
        braincell_result=braincell_result,
        neuron_result=neuron_result,
    )

    metrics = {
        "voltage": metric_record_for_pair(braincell_result["voltage_mV"], neuron_result["voltage_mV"]),
        "current": {
            "ix": metric_record_for_pair(
                aligned_current["braincell_ix"],
                aligned_current["neuron_ix"],
            ),
        },
        "ion_state": {
            ion_name: metric_record_for_pair(
                braincell_result["ion_state"][ion_name],
                neuron_result["ion_state"][ion_name],
            )
            for ion_name in sorted(set(braincell_result["ion_state"]) & set(neuron_result["ion_state"]))
        },
        "gates": {
            gate_pair.canonical_name: metric_record_for_pair(
                braincell_result["gates"][gate_pair.braincell],
                neuron_result["gates"][gate_pair.neuron],
            )
            for gate_pair in mapping_spec.gate_map
        },
    }

    return {
        "case_id": case.case_id,
        "run_id": case.run_id,
        "config_name": case.config_name,
        "template_name": case.template_name,
        "time_ms": neuron_result["time_ms"].tolist(),
        "braincell": _serialize_runner_result(braincell_result),
        "neuron": _serialize_runner_result(neuron_result),
        "aligned": {
            "current": {
                "time_ms": aligned_current["time_ms"].tolist(),
                "braincell_ix": aligned_current["braincell_ix"].tolist(),
                "neuron_ix": aligned_current["neuron_ix"].tolist(),
            },
            "ion_state": {
                ion_name: {
                    "time_ms": neuron_result["time_ms"].tolist(),
                    "braincell": braincell_result["ion_state"][ion_name].tolist(),
                    "neuron": neuron_result["ion_state"][ion_name].tolist(),
                }
                for ion_name in sorted(set(braincell_result["ion_state"]) & set(neuron_result["ion_state"]))
            },
        },
        "alignment": {
            "time_axis_trimmed_neuron_initial_sample": neuron_trimmed,
            "current": {
                "neuron_shift_steps": aligned_current["neuron_shift_steps"],
                "braincell_drop_tail_steps": aligned_current["braincell_drop_tail_steps"],
            },
            "gates": [
                {
                    "canonical_name": gate_pair.canonical_name,
                    "braincell_gate": gate_pair.braincell,
                    "neuron_gate": gate_pair.neuron,
                }
                for gate_pair in mapping_spec.gate_map
            ],
        },
        "metrics": metrics,
    }


def _normalize_runner_result(
    result: dict[str, Any],
    *,
    side: str,
    expected_gate_names: tuple[str, ...],
) -> dict[str, Any]:
    time_ms = ensure_1d(result.get("time_ms"), name=f"{side}.time_ms")
    voltage_mV = ensure_1d(result.get("voltage_mV"), name=f"{side}.voltage_mV")
    current_data = result.get("current")
    if not isinstance(current_data, dict):
        raise ValueError(f"{side}.current must be a mapping.")
    current_ix = ensure_1d(current_data.get("ix"), name=f"{side}.current.ix")
    ion_state_data = result.get("ion_state", {})
    if not isinstance(ion_state_data, dict):
        raise ValueError(f"{side}.ion_state must be a mapping.")
    ion_state = {
        str(name): ensure_1d(trace, name=f"{side}.ion_state.{name}")
        for name, trace in ion_state_data.items()
    }
    gates_data = result.get("gates")
    if not isinstance(gates_data, dict):
        raise ValueError(f"{side}.gates must be a mapping.")

    gates = {str(name): ensure_1d(trace, name=f"{side}.gates.{name}") for name, trace in gates_data.items()}
    actual_gate_names = tuple(sorted(gates))
    expected_gate_names = tuple(sorted(expected_gate_names))
    if actual_gate_names != expected_gate_names:
        raise ValueError(
            f"{side}.gates do not match mapping configuration: "
            f"{actual_gate_names!r} vs {expected_gate_names!r}."
        )

    if voltage_mV.shape != time_ms.shape:
        raise ValueError(
            f"{side}.voltage_mV shape must match {side}.time_ms: "
            f"{voltage_mV.shape!r} vs {time_ms.shape!r}."
        )
    if current_ix.shape != time_ms.shape:
        raise ValueError(
            f"{side}.current.ix shape must match {side}.time_ms: "
            f"{current_ix.shape!r} vs {time_ms.shape!r}."
        )
    for ion_name, ion_trace in ion_state.items():
        if ion_trace.shape != time_ms.shape:
            raise ValueError(
                f"{side}.ion_state.{ion_name} shape must match {side}.time_ms: "
                f"{ion_trace.shape!r} vs {time_ms.shape!r}."
            )
    for gate_name, gate_trace in gates.items():
        if gate_trace.shape != time_ms.shape:
            raise ValueError(
                f"{side}.gates.{gate_name} shape must match {side}.time_ms: "
                f"{gate_trace.shape!r} vs {time_ms.shape!r}."
            )

    return {
        "time_ms": time_ms,
        "voltage_mV": voltage_mV,
        "current": {"ix": current_ix},
        "ion_state": ion_state,
        "gates": gates,
    }


def _align_time_axes(
    *,
    braincell_result: dict[str, Any],
    neuron_result: dict[str, Any],
    dt_ms: float,
) -> bool:
    braincell_time = braincell_result["time_ms"]
    neuron_time = neuron_result["time_ms"]
    if _same_time_grid(braincell_time, neuron_time, dt_ms=dt_ms):
        return False
    if _is_constant_step_shift(braincell_time, neuron_time, dt_ms=dt_ms):
        return False
    if neuron_time.shape[0] == braincell_time.shape[0] + 1 and _same_time_grid(
        neuron_time[1:],
        braincell_time,
        dt_ms=dt_ms,
    ):
        _trim_initial_sample(neuron_result)
        return True
    raise ValueError("braincell and NEURON time axes do not match.")


def _trim_initial_sample(result: dict[str, Any]) -> None:
    result["time_ms"] = result["time_ms"][1:]
    result["voltage_mV"] = result["voltage_mV"][1:]
    result["current"]["ix"] = result["current"]["ix"][1:]
    result["ion_state"] = {
        ion_name: ion_trace[1:]
        for ion_name, ion_trace in result["ion_state"].items()
    }
    result["gates"] = {gate_name: gate_trace[1:] for gate_name, gate_trace in result["gates"].items()}


def _time_step_indices(time_ms: np.ndarray, *, dt_ms: float) -> np.ndarray | None:
    if dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}.")
    time_ms = np.asarray(time_ms, dtype=float)
    rounded_steps = np.rint(time_ms / float(dt_ms))
    grid_time_ms = rounded_steps * float(dt_ms)
    abs_tol = max(1e-6, abs(float(dt_ms)) * 1e-4)
    if not np.allclose(time_ms, grid_time_ms, rtol=0.0, atol=abs_tol):
        return None
    return rounded_steps.astype(np.int64)


def _same_time_grid(braincell_time: np.ndarray, neuron_time: np.ndarray, *, dt_ms: float) -> bool:
    if braincell_time.shape != neuron_time.shape:
        return False
    if np.allclose(braincell_time, neuron_time):
        return True
    braincell_steps = _time_step_indices(braincell_time, dt_ms=dt_ms)
    neuron_steps = _time_step_indices(neuron_time, dt_ms=dt_ms)
    if braincell_steps is None or neuron_steps is None:
        return False
    return np.array_equal(braincell_steps, neuron_steps)


def _is_constant_step_shift(braincell_time: np.ndarray, neuron_time: np.ndarray, *, dt_ms: float) -> bool:
    if braincell_time.shape != neuron_time.shape or braincell_time.shape[0] < 2:
        return False
    braincell_steps = _time_step_indices(braincell_time, dt_ms=dt_ms)
    neuron_steps = _time_step_indices(neuron_time, dt_ms=dt_ms)
    if braincell_steps is None or neuron_steps is None:
        return False
    return np.array_equal(neuron_steps, braincell_steps + 1)


def _align_current_traces(*, braincell_result: dict[str, Any], neuron_result: dict[str, Any]) -> dict[str, Any]:
    braincell_current = braincell_result["current"]["ix"]
    neuron_current = neuron_result["current"]["ix"]
    braincell_time = braincell_result["time_ms"]
    if braincell_current.shape != neuron_current.shape:
        raise ValueError(
            "braincell and NEURON current shapes do not match after time-axis alignment: "
            f"{braincell_current.shape!r} vs {neuron_current.shape!r}."
        )
    if braincell_current.shape[0] < 2:
        return {
            "time_ms": braincell_time,
            "braincell_ix": braincell_current,
            "neuron_ix": neuron_current,
            "neuron_shift_steps": 0,
            "braincell_drop_tail_steps": 0,
        }
    return {
        "time_ms": braincell_time[:-1],
        "braincell_ix": braincell_current[:-1],
        "neuron_ix": neuron_current[1:],
        "neuron_shift_steps": 1,
        "braincell_drop_tail_steps": 1,
    }


def _serialize_runner_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "voltage_mV": result["voltage_mV"].tolist(),
        "current": {"ix": result["current"]["ix"].tolist()},
        "ion_state": {ion_name: ion_trace.tolist() for ion_name, ion_trace in result["ion_state"].items()},
        "gates": {gate_name: gate_trace.tolist() for gate_name, gate_trace in result["gates"].items()},
    }
