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

"""Plot saved initialization samples; never import or execute the simulator."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in args.inputs]
    output = args.output or args.inputs[0].parent / "report"
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    rows = ["| Backend | Revision | CV | First init (ms) | Warm init median [min, max] (ms) | First JIT median (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for column, backend in enumerate(("cpu", "gpu")):
        available = [report for report in reports if report["plan"]["platform"] == backend
                     and report["status"] == "complete"]
        for report in available:
            baseline = report["label"].startswith("baseline")
            label = "Baseline" if baseline else "Array candidate"
            color = "#355c9c" if baseline else "#d47b27"
            sizes = report["plan"]["sizes"]
            for row, (metric, factor, ylabel) in enumerate((
                    ("init_state_s", 1000, "init_state (ms)"),
                    ("first_jit_step_s", 1, "First JIT + one step (s)"))):
                axis = axes[row, column]
                values = [report["summary"][str(size)][metric] for size in sizes]
                first = [next(s[metric] for s in report["samples"] if s["size"] == size) * factor for size in sizes]
                x = list(range(len(sizes)))
                axis.plot(x, [v["median"] * factor for v in values], "o-", color=color, label=label + " warm median")
                axis.fill_between(x, [v["min"] * factor for v in values],
                                  [v["max"] * factor for v in values], color=color, alpha=0.15)
                axis.plot(x, first, "D:", color=color, mfc="white", alpha=0.8, label=label + " first use")
                axis.set(xticks=x, xticklabels=sizes, xlabel="CV count", ylabel=ylabel, yscale="log")
                axis.grid(alpha=0.2)
            for size in sizes:
                init = report["summary"][str(size)]["init_state_s"]
                first = next(s["init_state_s"] for s in report["samples"] if s["size"] == size)
                jit = report["summary"][str(size)]["first_jit_step_s"]["median"]
                rows.append(f"| {backend.upper()} | {label} | {size} | {first * 1000:.2f} | "
                            f"{init['median'] * 1000:.2f} [{init['min'] * 1000:.2f}, {init['max'] * 1000:.2f}] | {jit:.3f} |")
        for row in range(2):
            axes[row, column].set_title(backend.upper())
            if available:
                axes[row, column].legend(fontsize=8)
            if not any(report["label"].startswith("baseline") for report in available):
                axes[row, column].text(0.03, 0.97, "Baseline unavailable", transform=axes[row, column].transAxes,
                                       va="top", fontsize=9, color="#9c3535")
    fig.suptitle("Fresh-cell initialization: CPU/GPU float64\nBands: five warm trials; diamonds: first use at each scale")
    fig.savefig(output / "initialization.png", dpi=160)
    plt.close(fig)
    table = "\n".join(rows) + "\n"
    (output / "timings.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
