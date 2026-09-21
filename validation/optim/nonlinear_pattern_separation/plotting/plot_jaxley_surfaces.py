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
# ===============================================================================

"""Plot all saved Jaxley restart voltage and spike surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.optim.nonlinear_pattern_separation.plotting.compare_results import (
    _input_legend_handles,
    _plot_input_samples,
)


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JAXLEY_DIR = (
    WORKFLOW_ROOT
    / "artifacts"
    / "nonlinear_pattern_separation_2026-09-15"
    / "raw"
    / "jaxley"
)
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "artifacts"
    / "nonlinear_pattern_separation_2026-09-15"
    / "analysis"
    / "figures"
)

DT_MS = 0.025
T_MAX_MS = 5.95
CURRENT_DELAY_MS = 1.0
CURRENT_DURATION_MS = 0.9
CURRENT_SCALE_NA = 0.05
READOUT_INDEX = 120
NSEG_PER_BRANCH = 4
DEPTH = 2
PARAMETER_BOUNDS = {
    "radius": (0.1, 5.0),
    "length": (1.0, 20.0),
    "axial_resistivity": (500.0, 5500.0),
    "HH_gNa": (0.05, 1.1),
    "HH_gK": (0.01, 0.3),
    "HH_gLeak": (0.0001, 0.001),
}


def _build_simulator(jx, jnp, jax):
    """Build the saved experiment cell and a batched forward simulator."""
    from jaxley.channels import HH

    parents = jnp.asarray([-1] + [branch // 2 for branch in range(0, 2**DEPTH - 2)])
    num_branches = len(parents)
    stimulus_branches = np.concatenate(
        [
            np.arange(2 ** (DEPTH + layer) - 1, 2 ** (DEPTH + layer + 1) - 1)
            for layer in range(-1, 0)
        ]
    )
    num_repeats = len(stimulus_branches) // 2
    stimulus_locations = np.ones(2 * num_repeats)

    compartment = jx.Compartment().initialize()
    branch = jx.Branch([compartment for _ in range(NSEG_PER_BRANCH)]).initialize()
    cell = jx.Cell([branch for _ in range(num_branches)], parents=parents).initialize()
    cell.insert(HH())
    cell.set("v", -70.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)
    cell.branch(0).loc(0.0).record()
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        initial = (lower + upper) / 2.0
        cell.branch("all").comp("all").make_trainable(name, initial, verbose=False)

    def simulate(parameters, image):
        currents = jx.datapoint_to_step_currents(
            CURRENT_DELAY_MS,
            CURRENT_DURATION_MS,
            CURRENT_SCALE_NA * image,
            DT_MS,
            T_MAX_MS,
        )
        data_stimuli = None
        for index, current in zip(range(2 * num_repeats), currents):
            data_stimuli = cell.branch(stimulus_branches[index]).loc(
                stimulus_locations[index]
            ).data_stimulate(current, data_stimuli=data_stimuli)
        return jx.integrate(cell, params=parameters, data_stimuli=data_stimuli)

    return jax.jit(jax.vmap(simulate, in_axes=(None, 0)))


def _load_saved_data(jaxley_dir):
    with np.load(jaxley_dir / "history.npz") as history:
        train_inputs = np.asarray(history["train_inputs"])
        test_inputs = np.asarray(history["test_inputs"])
        axis = np.asarray(history["sweep_axis"], dtype=float)
    with (jaxley_dir / "parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)["final"]
    if train_inputs.ndim != 3 or test_inputs.ndim != 3:
        raise ValueError("Jaxley train/test inputs must have shape (restart, sample, 2).")
    if len(parameters) != train_inputs.shape[0] or len(parameters) != test_inputs.shape[0]:
        raise ValueError("Saved Jaxley parameters and input samples have different restart counts.")
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("Saved Jaxley sweep_axis must be a one-dimensional grid.")
    return parameters, train_inputs, test_inputs, axis


def plot_jaxley_surfaces(jaxley_dir, output_dir):
    """Generate the 2x5 Jaxley 3 ms voltage and spike-surface figures."""
    import jax
    import jax.numpy as jnp
    import jaxley as jx

    parameters, train_inputs, test_inputs, axis = _load_saved_data(jaxley_dir)
    if len(parameters) != 10:
        raise ValueError(f"Expected 10 Jaxley restarts, found {len(parameters)}.")

    simulator = _build_simulator(jx, jnp, jax)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    sweep_inputs = jnp.asarray(np.stack((grid_x.ravel(), grid_y.ravel()), axis=1))
    voltage_surfaces = []
    spike_surfaces = []
    for restart, saved_parameters in enumerate(parameters):
        jax_parameters = jax.tree_util.tree_map(jnp.asarray, saved_parameters)
        traces = np.asarray(simulator(jax_parameters, sweep_inputs))
        soma = traces[:, 0, :]
        voltage_surfaces.append(soma[:, READOUT_INDEX].reshape(grid_x.shape))
        valid = np.all(np.isfinite(soma), axis=1)
        spike_count = np.sum(
            (soma[:, :-1] < 0.0) & (soma[:, 1:] >= 0.0), axis=1
        ).astype(float)
        spike_count[~valid] = np.nan
        spike_surfaces.append(spike_count.reshape(grid_x.shape))
        print(f"restart {restart}: surface {grid_x.shape[0]}x{grid_x.shape[1]}", flush=True)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    voltage_cmap = plt.get_cmap("coolwarm").copy()
    voltage_cmap.set_bad("#d9d9d9")
    for axis_plot, restart, voltage in zip(axes.flat, range(10), voltage_surfaces):
        nonfinite = int(np.count_nonzero(~np.isfinite(voltage)))
        if nonfinite:
            print(f"restart {restart}: masking {nonfinite} non-finite voltage values", flush=True)
        image = axis_plot.imshow(
            np.ma.masked_invalid(voltage),
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            aspect="equal",
            cmap=voltage_cmap,
            vmin=-75,
            vmax=45,
        )
        _plot_input_samples(axis_plot, train_inputs[restart], test_inputs[restart])
        axis_plot.set_title(f"Jaxley restart {restart}")
        axis_plot.set_xlim(axis[0], axis[-1])
        axis_plot.set_ylim(axis[0], axis[-1])
        axis_plot.set_xlabel("x1")
        axis_plot.set_ylabel("x2")
    fig.legend(
        handles=_input_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        title="Input regions and samples",
    )
    fig.colorbar(image, ax=axes, label="V(3 ms), mV", shrink=0.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "jaxley_all_seed_voltage_surfaces.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    spike_figure, spike_axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    spike_cmap = plt.get_cmap("gray_r").copy()
    spike_cmap.set_bad("#d9d9d9")
    for axis_plot, restart, spike_count in zip(
        spike_axes.flat, range(10), spike_surfaces
    ):
        nonfinite = int(np.count_nonzero(~np.isfinite(spike_count)))
        if nonfinite:
            print(f"restart {restart}: masking {nonfinite} non-finite spike values", flush=True)
        spike_image = axis_plot.imshow(
            np.ma.masked_invalid(spike_count),
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            aspect="equal",
            cmap=spike_cmap,
            vmin=0,
            vmax=1,
        )
        _plot_input_samples(axis_plot, train_inputs[restart], test_inputs[restart])
        axis_plot.set_title(f"Jaxley restart {restart}")
        axis_plot.set_xlim(axis[0], axis[-1])
        axis_plot.set_ylim(axis[0], axis[-1])
        axis_plot.set_xlabel("x1")
        axis_plot.set_ylabel("x2")
    spike_figure.legend(
        handles=_input_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        title="Input regions and samples",
    )
    spike_figure.colorbar(
        spike_image,
        ax=spike_axes,
        ticks=[0, 1],
        label="spike presence",
        shrink=0.8,
    )
    spike_output = output_dir / "jaxley_all_seed_spike_surfaces.png"
    spike_figure.savefig(spike_output, dpi=180)
    plt.close(spike_figure)
    return output, spike_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaxley-dir", type=Path, default=DEFAULT_JAXLEY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    voltage_output, spike_output = plot_jaxley_surfaces(args.jaxley_dir, args.output_dir)
    print(voltage_output)
    print(spike_output)


if __name__ == "__main__":
    main()
