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

"""Compare saved population timings and voltages without running simulations."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def read_reports(paths):
    reports = {}
    for path in paths:
        report = json.loads(path.read_text())
        plan = report["plan"]
        if report["status"] != "complete" or len(plan["sizes"]) != 1:
            raise ValueError(f"Expected a completed, single-CV-size scan: {path}")
        key = (plan["platform"], "baseline" if report["label"].startswith("baseline") else "arrays")
        if key in reports:
            raise ValueError(f"Duplicate configuration: {key}")
        expected = (["compile"] + ["warmup"] * plan["extra_warmup_rollouts_per_case"]
                    + ["timed"] * plan["timed_rollouts_per_case"])
        if [case["pop_size"] for case in report["cases"]] != plan["pop_sizes"]:
            raise ValueError(f"Incomplete population scan: {path}")
        for case in report["cases"]:
            if case["status"] != "complete" or [run["kind"] for run in case["runs"]] != expected:
                raise ValueError(f"Incomplete execution counts: {path}")
            if case["n_cv"] != plan["sizes"][0] or case["voltage_shape"] != [case["pop_size"], case["n_cv"]]:
                raise ValueError(f"Unexpected dimensions: {path}")
            if key[1] == "arrays" and case["cable_array_shapes"] != [[case["n_cv"]]] * 4:
                raise ValueError(f"Geometry was not shared: {path}")
        reports[key] = report
    if set(reports) != {(backend, revision) for backend in ("cpu", "gpu") for revision in ("baseline", "arrays")}:
        raise ValueError("Provide exactly baseline/arrays on CPU/GPU")
    reference = reports["cpu", "baseline"]
    for key, report in reports.items():
        for field in ("sizes", "pop_sizes", "steps_per_rollout", "precision_bits", "dt_ms",
                      "extra_warmup_rollouts_per_case", "timed_rollouts_per_case"):
            if report["plan"][field] != reference["plan"][field]:
                raise ValueError(f"Mismatched protocol field {field}: {key}")
        for field in ("driver_sha256", "shared_driver_sha256", "versions"):
            if report["environment"][field] != reference["environment"][field]:
                raise ValueError(f"Mismatched driver/environment {field}: {key}")
    for revision in ("baseline", "arrays"):
        if reports["cpu", revision]["environment"]["source_sha256"] != reports["gpu", revision]["environment"]["source_sha256"]:
            raise ValueError(f"Source changed between backends: {revision}")
    return reports


def compare_voltages(reports):
    checks = []
    pairs = [((backend, "baseline"), (backend, "arrays")) for backend in ("cpu", "gpu")]
    pairs += [(("cpu", revision), ("gpu", revision)) for revision in ("baseline", "arrays")]
    for left, right in pairs:
        for a, b in zip(reports[left]["cases"], reports[right]["cases"], strict=True):
            with np.load(a["voltage_file"]) as data:
                va = data["final_voltage_mv"]
            with np.load(b["voltage_file"]) as data:
                vb = data["final_voltage_mv"]
            np.testing.assert_allclose(va, vb, rtol=1e-9, atol=1e-7)
            checks.append({"left": list(left), "right": list(right), "pop_size": a["pop_size"],
                           "max_abs_difference_mv": float(np.max(np.abs(va - vb)))})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = read_reports(args.inputs)
    checks = compare_voltages(reports)
    output = args.output or args.inputs[0].parent / "population_report"
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = reports["cpu", "baseline"]["plan"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    rows = ["| Backend | Revision | Population | First init (ms) | First JIT + rollout (s) | Steady median [min, max] (ms) | Reset median (ms) | Cell-step (us) | Logical State (bytes) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for column, backend in enumerate(("cpu", "gpu")):
        for revision, label, color in (("baseline", "Baseline", "#355c9c"), ("arrays", "Array candidate", "#d47b27")):
            report = reports[backend, revision]
            cases = report["cases"]
            populations = [case["pop_size"] for case in cases]
            axes[0, column].plot(populations, [case["init_state_s"] * 1000 for case in cases], "o-", color=color, label=label)
            axes[1, column].plot(populations, [case["steady"]["median_s"] * 1000 for case in cases], "o-", color=color, label=label)
            axes[1, column].fill_between(populations, [case["steady"]["min_s"] * 1000 for case in cases],
                                         [case["steady"]["max_s"] * 1000 for case in cases], color=color, alpha=0.15)
            axes[2, column].plot(populations, [case["steady"]["cell_steps_per_second"] for case in cases], "o-", color=color, label=label)
            for case in cases:
                steady = case["steady"]
                reset = statistics.median(run["reset_s"] for run in case["runs"] if run["kind"] == "timed")
                rows.append(f"| {backend.upper()} | {label} | {case['pop_size']} | {case['init_state_s'] * 1000:.2f} | "
                            f"{case['runs'][0]['rollout_s']:.3f} | {steady['median_s'] * 1000:.3f} "
                            f"[{steady['min_s'] * 1000:.3f}, {steady['max_s'] * 1000:.3f}] | "
                            f"{reset * 1000:.3f} | {steady['cell_step_us']:.3f} | {case['logical_state_bytes']} |")
        for row, ylabel in enumerate(("First init_state (ms)", f"Steady {plan['steps_per_rollout']}-step rollout (ms)", "Throughput (cell-steps / s)")):
            axis = axes[row, column]
            axis.set(xscale="log", yscale="log", xlabel="Population size", ylabel=ylabel,
                     xticks=plan["pop_sizes"], xticklabels=plan["pop_sizes"], title=backend.upper())
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    # Match backend axes so CPU/GPU magnitudes can be compared directly.
    for row in range(3):
        limits = [axis.get_ylim() for axis in axes[row]]
        for axis in axes[row]:
            axis.set_ylim(min(limit[0] for limit in limits), max(limit[1] for limit in limits))
    fig.suptitle(f"{plan['sizes'][0]} CV, shared fixed geometry, float64 passive forward\n"
                 f"One process per curve; steady median and min-max of {plan['timed_rollouts_per_case']} runs")
    fig.savefig(output / "population.png", dpi=160)
    plt.close(fig)
    table = "\n".join(rows) + "\n"
    (output / "timings.md").write_text(table)
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(table)
    print(f"Saved-voltage checks: {len(checks)} passed; max difference {max(c['max_abs_difference_mv'] for c in checks):.3g} mV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
