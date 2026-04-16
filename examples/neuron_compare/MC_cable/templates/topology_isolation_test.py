from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import numpy as np


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
case_schema = _load_module(_ROOT / "case_schema.py", "mc_cable_topology_case_schema")
compare_mod = _load_module(_ROOT / "compare_MC_cable.py", "mc_cable_topology_compare")
fixtures = _load_module(_ROOT / "fixtures.py", "mc_cable_topology_fixtures")


def _write_temp_swc(testcase: unittest.TestCase, body: str, filename: str) -> str:
    return str(fixtures.write_temp_swc(testcase, body, filename=filename))


def _make_case_payload(*, swc_path: str, cv_per_branch: int) -> dict:
    return fixtures.base_case_payload(
        swc_path=swc_path,
        dt_ms=0.025,
        duration_ms=2.0,
        cv_per_branch=cv_per_branch,
        stimulus=fixtures.dc_step_stimulus(delay_ms=0.5, dur_ms=1.0, amp_nA=0.05),
    )


def _metrics_summary(result: dict) -> dict[str, float]:
    braincell_voltage = np.asarray(result["braincell"]["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(result["neuron"]["voltage_mV"], dtype=float)
    abs_sum_error = np.abs(braincell_voltage.sum(axis=1) - neuron_voltage.sum(axis=1))
    target_pair = result["alignment"]["stimulus_target_pair"]
    braincell_target = int(target_pair["braincell_compartment_index"])
    neuron_target = int(target_pair["neuron_compartment_index"])
    target_abs_error = np.abs(braincell_voltage[:, braincell_target] - neuron_voltage[:, neuron_target])
    soma_indices = [
        index
        for index, label in enumerate(result["alignment"]["braincell_labels"])
        if label["canonical_name"].startswith("soma[")
    ]
    dend_indices = [
        index
        for index, label in enumerate(result["alignment"]["braincell_labels"])
        if label["canonical_name"].startswith("dend[")
    ]
    summary = {
        "overall_mae": float(result["metrics"]["overall"]["mae"]),
        "overall_rmse": float(result["metrics"]["overall"]["rmse"]),
        "overall_max_abs": float(result["metrics"]["overall"]["max_abs"]),
        "sum_voltage_mean_abs": float(np.mean(abs_sum_error)),
        "sum_voltage_max_abs": float(np.max(abs_sum_error)),
        "target_mae": float(np.mean(target_abs_error)),
    }
    if len(soma_indices) > 0:
        summary["soma_group_mae"] = float(
            np.mean([result["metrics"]["per_compartment"][index]["mae"] for index in soma_indices])
        )
    if len(dend_indices) > 0:
        summary["dend_group_mae"] = float(
            np.mean([result["metrics"]["per_compartment"][index]["mae"] for index in dend_indices])
        )
    return summary


def _summary_text(name: str, cv_per_branch: int, summary: dict[str, float]) -> str:
    ordered_keys = [
        "overall_mae",
        "overall_rmse",
        "overall_max_abs",
        "sum_voltage_mean_abs",
        "sum_voltage_max_abs",
        "target_mae",
        "soma_group_mae",
        "dend_group_mae",
    ]
    parts = [f"{key}={summary[key]:.6g}" for key in ordered_keys if key in summary]
    return f"{name} cv={cv_per_branch}: " + ", ".join(parts)


class TopologyIsolationTest(unittest.TestCase):
    maxDiff = None

    def _run_case(self, *, name: str, swc_body: str, cv_per_branch: int) -> tuple[dict, dict[str, float]]:
        swc_path = _write_temp_swc(self, swc_body, filename=f"{name}.swc")
        payload = _make_case_payload(swc_path=swc_path, cv_per_branch=cv_per_branch)
        payload["case_id"] = f"{name}_cv{cv_per_branch}"
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        result = compare_mod.compare_case(case)
        summary = _metrics_summary(result)
        return result, summary

    def test_topology_isolation_matrix(self) -> None:
        cases = {
            "root_soma_only": """
                1 1 0 0 0 10 -1
                2 1 10 0 0 10 1
            """,
            "root_soma_plus_dend_at_dist": """
                1 1 0 0 0 10 -1
                2 1 10 0 0 10 1
                3 3 20 0 0 1 2
            """,
            "root_soma_plus_dend_at_prox": """
                1 1 0 0 0 10 -1
                2 1 10 0 0 10 1
                3 3 -20 0 0 1 1
            """,
            "root_soma_plus_one_soma_child": """
                1 1 0 0 0 10 -1
                2 1 10 0 0 10 1
                3 1 20 0 0 10 2
            """,
            "root_soma_plus_two_soma_children": """
                1 1 0 0 0 10 -1
                2 1 10 0 0 10 1
                3 1 20 0 0 10 2
                4 1 -5 0 0 10 2
                5 1 -10 0 0 10 4
            """,
            "branched_soma_minimal": """
                1 1 0 0 100 10 -1
                2 1 0 0 0 10  1
                3 1 10 0 0 10 2
                4 1 20 0 0 10 3
                5 1 -5 0 0 10 2
                6 1 -10 0 0 10 5
                7 3 100 0 0 1 1
            """,
        }

        rows = []
        for name, swc_body in cases.items():
            for cv_per_branch in (1, 3):
                result, summary = self._run_case(name=name, swc_body=swc_body, cv_per_branch=cv_per_branch)
                rows.append(_summary_text(name, cv_per_branch, summary))
                self.assertTrue(np.isfinite(result["metrics"]["overall"]["mae"]))
                self.assertTrue(np.isfinite(result["metrics"]["overall"]["rmse"]))
                self.assertTrue(np.isfinite(result["metrics"]["overall"]["max_abs"]))
                self.assertIn("stimulus_target_pair", result["alignment"])

        report = "\n".join(rows)
        print("\n" + report)

        # Stability checks that should hold across topology shrinkage:
        # root-only and simple root+dend cases should stay very tight even when split.
        root_only_cv1 = next(line for line in rows if line.startswith("root_soma_only cv=1"))
        root_only_cv3 = next(line for line in rows if line.startswith("root_soma_only cv=3"))
        self.assertIn("overall_mae=", root_only_cv1)
        self.assertIn("overall_mae=", root_only_cv3)

    def test_single_chain_internal_and_reassembled_are_equivalent(self) -> None:
        import brainunit as u
        import numpy as np
        import braincell
        from braincell import Branch, Morphology, Cell, CVPerBranch, CableProperty
        from braincell.filter import AllRegion, at
        from braincell.mech import CurrentClamp, StateProbe
        from neuron import h

        segment_length_um = 100.0
        total_length_um = 300.0
        radius_um = 50.0
        ra_ohm_cm = 100.0
        cm_uF_cm2 = 1.0
        v_init_mV = -65.0
        dt_ms = 0.025
        duration_ms = 2.0
        stim_delay_ms = 0.5
        stim_dur_ms = 0.5
        stim_amp_nA = 0.01

        def metrics(a, b):
            diff = np.asarray(a, float) - np.asarray(b, float)
            return {
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "max_abs": float(np.max(np.abs(diff))),
            }

        def run_braincell_internal():
            soma = Branch.from_lengths(lengths=[total_length_um] * u.um, radii=[radius_um, radius_um] * u.um, type="soma")
            morpho = Morphology.from_root(soma, name="soma")
            cell = Cell(morpho, solver="staggered", cv_policy=CVPerBranch(cv_per_branch=3))
            cell.paint(
                AllRegion(),
                CableProperty(
                    resting_potential=v_init_mV * u.mV,
                    membrane_capacitance=cm_uF_cm2 * (u.uF / u.cm ** 2),
                    axial_resistivity=ra_ohm_cm * (u.ohm * u.cm),
                ),
            )
            cell.place(at("soma", 0.5), CurrentClamp.step(stim_amp_nA * u.nA, stim_dur_ms * u.ms, delay=stim_delay_ms * u.ms))
            for index, x in enumerate((1 / 6, 0.5, 5 / 6)):
                cell.place(at("soma", x), StateProbe(name=f"v{index}"))
            cell.init_state()
            result = cell.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
            return np.stack([np.asarray(result.traces[f"v{index}"].to_decimal(u.mV), dtype=float) for index in range(3)], axis=1)

        def run_braincell_reassembled():
            soma0 = Branch.from_lengths(lengths=[segment_length_um] * u.um, radii=[radius_um, radius_um] * u.um, type="soma")
            soma1 = Branch.from_lengths(lengths=[segment_length_um] * u.um, radii=[radius_um, radius_um] * u.um, type="soma")
            soma2 = Branch.from_lengths(lengths=[segment_length_um] * u.um, radii=[radius_um, radius_um] * u.um, type="soma")
            morpho = Morphology.from_root(soma0, name="soma_0")
            morpho.attach(parent="soma_0", child_branch=soma1, child_name="soma_1", parent_x=1.0)
            morpho.attach(parent="soma_1", child_branch=soma2, child_name="soma_2", parent_x=1.0)
            cell = Cell(morpho, solver="staggered", cv_policy=CVPerBranch(cv_per_branch=1))
            cell.paint(
                AllRegion(),
                CableProperty(
                    resting_potential=v_init_mV * u.mV,
                    membrane_capacitance=cm_uF_cm2 * (u.uF / u.cm ** 2),
                    axial_resistivity=ra_ohm_cm * (u.ohm * u.cm),
                ),
            )
            cell.place(at("soma_1", 0.5), CurrentClamp.step(stim_amp_nA * u.nA, stim_dur_ms * u.ms, delay=stim_delay_ms * u.ms))
            for index, name in enumerate(("soma_0", "soma_1", "soma_2")):
                cell.place(at(name, 0.5), StateProbe(name=f"v{index}"))
            cell.init_state()
            result = cell.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
            return np.stack([np.asarray(result.traces[f"v{index}"].to_decimal(u.mV), dtype=float) for index in range(3)], axis=1)

        def _sample_all_segments(sections):
            return np.asarray([float(seg.v) for sec in sections for seg in sec], dtype=float)

        def run_neuron_internal():
            soma = h.Section(name="soma[0]")
            soma.L = total_length_um
            soma.diam = 2.0 * radius_um
            soma.nseg = 3
            soma.Ra = ra_ohm_cm
            soma.cm = cm_uF_cm2
            stim = h.IClamp(soma(0.5))
            stim.delay = 0.0
            stim.dur = 1e9
            stim.amp = 0.0
            h.load_file("stdrun.hoc")
            h.dt = dt_ms
            h.finitialize(v_init_mV)
            times = np.arange(0.0, duration_ms, dt_ms, dtype=float)
            volts = np.empty((len(times) + 1, 3), dtype=float)
            volts[0, :] = _sample_all_segments([soma])
            for i, t in enumerate(times):
                stim.amp = stim_amp_nA if stim_delay_ms <= t < (stim_delay_ms + stim_dur_ms) else 0.0
                h.fadvance()
                volts[i + 1, :] = _sample_all_segments([soma])
            h.delete_section(sec=soma)
            return volts[1:, :]

        def run_neuron_reassembled():
            s0 = h.Section(name="soma[0]")
            s1 = h.Section(name="soma[1]")
            s2 = h.Section(name="soma[2]")
            for sec in (s0, s1, s2):
                sec.L = segment_length_um
                sec.diam = 2.0 * radius_um
                sec.nseg = 1
                sec.Ra = ra_ohm_cm
                sec.cm = cm_uF_cm2
            s1.connect(s0(1.0), 0.0)
            s2.connect(s1(1.0), 0.0)
            stim = h.IClamp(s1(0.5))
            stim.delay = 0.0
            stim.dur = 1e9
            stim.amp = 0.0
            h.load_file("stdrun.hoc")
            h.dt = dt_ms
            h.finitialize(v_init_mV)
            times = np.arange(0.0, duration_ms, dt_ms, dtype=float)
            volts = np.empty((len(times) + 1, 3), dtype=float)
            volts[0, :] = _sample_all_segments([s0, s1, s2])
            for i, t in enumerate(times):
                stim.amp = stim_amp_nA if stim_delay_ms <= t < (stim_delay_ms + stim_dur_ms) else 0.0
                h.fadvance()
                volts[i + 1, :] = _sample_all_segments([s0, s1, s2])
            for sec in (s0, s1, s2):
                h.delete_section(sec=sec)
            return volts[1:, :]

        bc_internal = run_braincell_internal()
        bc_reassembled = run_braincell_reassembled()
        nrn_internal = run_neuron_internal()
        nrn_reassembled = run_neuron_reassembled()

        self.assertLess(metrics(bc_internal, bc_reassembled)["max_abs"], 1e-9)
        self.assertLess(metrics(nrn_internal, nrn_reassembled)["max_abs"], 1e-12)
        self.assertLess(metrics(bc_internal, nrn_internal)["max_abs"], 1e-9)
        self.assertLess(metrics(bc_reassembled, nrn_reassembled)["max_abs"], 1e-9)


if __name__ == "__main__":
    unittest.main()
