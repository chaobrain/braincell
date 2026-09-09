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

"""Run regular whole-cell comparisons in isolated worker processes.

Python: ``run_case("pc_ma2024", protocol=...)``.
CLI: ``python -m validation.neuron.cell.run --cell pc_ma2024``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .protocol import MODELS, Protocol, get_model
from .results import ComparisonResult, Trace, load_result, save_result


def run_case(model: str, *, protocol: Protocol | None = None, output_dir: str | Path | None = None) -> ComparisonResult:
    """Run both simulators in a fresh Python process and return persisted raw traces.

    Parameters
    ----------
    model : str
        A name in ``protocol.MODELS``, including ``grc_ma2020_full``.
    protocol : Protocol, optional
        Experiment settings. Omission uses the model's original notebook defaults.
    output_dir : str or pathlib.Path, optional
        Parent for a unique run directory. Defaults to the model's ``artifacts/``.

    Returns
    -------
    ComparisonResult
        Original traces, configuration, versions and the unique ``output_dir``.
        Nonfinite voltages remain available for diagnosis; analysis rejects them.

    Raises
    ------
    RuntimeError
        The worker failed; its output directory contains ``run.log`` and ``request.json``.
    """
    spec = get_model(model)
    protocol = spec.default_protocol if protocol is None else protocol
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol.")
    parent = Path(output_dir) if output_dir is not None else Path(__file__).parent / spec.folder / "artifacts"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = (parent / f"{model}-{stamp}-{uuid4().hex[:8]}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    request.write_text(json.dumps({"model": model, "protocol": asdict(protocol)}, indent=2) + "\n")
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))
    with (directory / "run.log").open("w") as log:
        completed = subprocess.run(
            [sys.executable, "-m", "validation.neuron.cell.run", "--worker-request", str(request)],
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        tail = "\n".join((directory / "run.log").read_text().splitlines()[-20:])
        raise RuntimeError(f"Comparison failed for {model}; see {directory / 'run.log'}\n{tail}")
    return load_result(directory)


def _run_worker(model: str, protocol: Protocol, directory: Path) -> None:
    """Build models and retain all simulator-owned objects through data collection."""
    import platform

    import braincell
    import brainstate
    import brainunit as u
    import jax
    import neuron
    from neuron import h
    import numpy as np

    from braincell import mech
    from braincell.filter import at
    from validation.neuron._mechanisms import compile_mechanisms
    from ._nrnmech import nrnmech_path

    spec = get_model(model)
    brainstate.environ.set(precision=protocol.precision)
    build_dir = compile_mechanisms(spec.folder, kind="cell")
    params = spec.parameters(protocol)
    reference = spec.create("neuron", params, protocol, nrnmech_path=nrnmech_path(build_dir / "x86_64"))
    try:
        reference.build()
        stim = h.IClamp(reference.root_soma(0.5))
        stim.delay = protocol.delay_ms
        stim.dur = protocol.stim_dur_ms
        stim.amp = protocol.amp_nA
        time = h.Vector().record(h._ref_t)
        voltage = h.Vector().record(reference.root_soma(0.5)._ref_v)
        h.cvode_active(0)
        h.dt = protocol.dt_ms
        h.steps_per_ms = 1.0 / h.dt
        h.celsius = protocol.temperature_celsius
        h.tstop = protocol.duration_ms
        h.v_init = protocol.v_init_mV
        h.finitialize(h.v_init)
        h.run()
        neuron_trace = Trace(np.asarray(time), np.asarray(voltage))

        assembly = spec.create("braincell", params, protocol).build()
        cell = assembly.cell
        cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma"))
        cell.place(
            at("soma", 0.5),
            mech.CurrentClamp(
                delay=protocol.delay_ms * u.ms,
                durations=protocol.stim_dur_ms * u.ms,
                amplitudes=protocol.amp_nA * u.nA,
            ),
        )
        cell.init_state()
        cell.reset_state()
        result = cell.run(dt=protocol.dt_ms * u.ms, duration=protocol.duration_ms * u.ms)
        values = np.asarray(result.traces["v_soma"].to_decimal(u.mV), dtype=float).reshape(-1)
        # StateProbe samples after each update; RunResult.time labels step starts.
        times = (np.arange(values.size, dtype=float) + 1.0) * protocol.dt_ms
        braincell_trace = Trace(times, values)
        metadata = {
            "format_version": 1,
            "model": model,
            "protocol": asdict(protocol),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units": {"time": "ms", "voltage": "mV"},
            "recording": {"location": "soma(0.5)", "braincell": "post-step", "neuron": "initial-and-post-step"},
            "solvers": {"braincell": "staggered", "neuron": "fixed-step"},
            "versions": {
                "python": platform.python_version(),
                "braincell": braincell.__version__,
                "brainstate": brainstate.__version__,
                "brainunit": u.__version__,
                "jax": jax.__version__,
                "numpy": np.__version__,
                "neuron": neuron.__version__,
            },
            "backend": jax.default_backend(),
            "finite_voltage": {
                "neuron": bool(np.isfinite(neuron_trace.voltage_mV).all()),
                "braincell": bool(np.isfinite(braincell_trace.voltage_mV).all()),
            },
        }
        save_result(ComparisonResult(neuron_trace, braincell_trace, metadata), directory)
    finally:
        reference.cleanup()


def main(argv=None) -> int:
    """Run a case or regenerate metrics and plots from a saved run."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--cell", choices=tuple(MODELS), help="Model to simulate.")
    source.add_argument("--load", type=Path, help="Existing run directory; analyze and plot without simulation.")
    parser.add_argument("--output-dir", type=Path, help="Parent directory for a new run.")
    parser.add_argument("--threshold-mv", type=float, default=0.0, help="Spike threshold in mV.")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    for field in fields(Protocol):
        option = "--" + field.name.lower().replace("_", "-")
        parser.add_argument(option, dest=field.name, type=int if field.name == "precision" else float)
    args = parser.parse_args(argv)
    if args.worker_request is not None:
        request = json.loads(args.worker_request.read_text())
        _run_worker(request["model"], Protocol(**request["protocol"]), args.worker_request.parent)
        return 0
    overrides = {
        field.name: getattr(args, field.name) for field in fields(Protocol) if getattr(args, field.name) is not None
    }
    if args.load is not None:
        if overrides or args.output_dir:
            parser.error("--load reuses the saved protocol and output directory; omit simulation overrides.")
        result = load_result(args.load)
    elif args.cell is not None:
        protocol = replace(get_model(args.cell).default_protocol, **overrides)
        result = run_case(args.cell, protocol=protocol, output_dir=args.output_dir)
    else:
        parser.error("Choose --cell to simulate or --load to analyze a saved run.")
    from .analysis import analyze, save_analysis
    from .plotting import plot_analysis, save_plot

    analysis = analyze(result, threshold_mV=args.threshold_mv)
    save_analysis(analysis, result.output_dir)
    figure, _ = plot_analysis(analysis)
    save_plot(figure, result.output_dir / "comparison.png")
    import matplotlib.pyplot as plt

    plt.close(figure)
    print(json.dumps(analysis.summary, indent=2, allow_nan=False))
    print(f"Results: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
