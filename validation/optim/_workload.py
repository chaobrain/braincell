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

"""Shared HH workloads for gradient agreement and scaling measurements."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import NamedTuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import AllRegion, at
from braincell.experimental.optim import build_rollout_value_and_grad

DT_MS = 0.025



RNG_SEED = 20260828



METHODS = ("bptt", "rtrl")



BACKSUBS = ("recursive", "ordinary")



_BASE_G_MAX = {
    "leak": 0.1 * u.mS / u.cm**2,
    "na": 120.0 * u.mS / u.cm**2,
    "k": 10.0 * u.mS / u.cm**2,
}



_CHANNEL_ORDER = tuple(_BASE_G_MAX)



_STATE_VARIABLES_PER_CHANNEL = {"leak": 0, "na": 2, "k": 1}



@dataclass(frozen=True)
class MechanismSpec:
    """Static channel and optimizer subset for one factorial case."""

    name: str
    painted_channels: tuple[str, ...]
    trainable_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        painted_channels = tuple(self.painted_channels)
        trainable_channels = tuple(self.trainable_channels)
        object.__setattr__(self, "painted_channels", painted_channels)
        object.__setattr__(self, "trainable_channels", trainable_channels)
        painted = set(painted_channels)
        trainable = set(trainable_channels)
        known = set(_CHANNEL_ORDER)
        if not self.name:
            raise ValueError("MechanismSpec.name must be non-empty.")
        if not painted or not painted.issubset(known):
            raise ValueError(f"painted_channels must be a non-empty subset of {_CHANNEL_ORDER!r}.")
        if not trainable or not trainable.issubset(painted):
            raise ValueError("trainable_channels must be a non-empty subset of painted_channels.")
        if tuple(channel for channel in _CHANNEL_ORDER if channel in painted) != painted_channels:
            raise ValueError(f"painted_channels must follow {_CHANNEL_ORDER!r} order.")
        if tuple(channel for channel in _CHANNEL_ORDER if channel in trainable) != trainable_channels:
            raise ValueError(f"trainable_channels must follow {_CHANNEL_ORDER!r} order.")

    @property
    def state_variables_per_cv(self) -> int:
        """Return voltage plus painted-channel gate states per CV."""
        return 1 + sum(_STATE_VARIABLES_PER_CHANNEL[channel] for channel in self.painted_channels)

    @property
    def trainable_channels_per_cv(self) -> int:
        """Return the number of independent per-CV channel scales."""
        return len(self.trainable_channels)

    @property
    def painted_label(self) -> str:
        """Return a compact CSV-safe painted-channel label."""
        return "+".join(self.painted_channels)

    @property
    def trainable_label(self) -> str:
        """Return a compact CSV-safe trainable-channel label."""
        return "+".join(self.trainable_channels)



FULL_HH_SPEC = MechanismSpec("lkn_fit_lkn", ("leak", "na", "k"), ("leak", "na", "k"))



MECHANISM_FACTORIAL_SPECS = (
    MechanismSpec("l_fit_l", ("leak",), ("leak",)),
    MechanismSpec("lk_fit_l", ("leak", "k"), ("leak",)),
    MechanismSpec("lk_fit_lk", ("leak", "k"), ("leak", "k")),
    MechanismSpec("lkn_fit_l", ("leak", "na", "k"), ("leak",)),
    MechanismSpec("lkn_fit_lk", ("leak", "na", "k"), ("leak", "k")),
    FULL_HH_SPEC,
)



@dataclass(frozen=True, order=True)
class BenchmarkConfig:
    """One static benchmark configuration."""

    n_cv: int
    duration_ms: float
    batch_size: int
    n_seed: int

    def __post_init__(self) -> None:
        if self.n_cv < 1 or self.n_cv % 2 == 0:
            raise ValueError("n_cv must be a positive odd integer.")
        if self.duration_ms <= 0.0:
            raise ValueError("duration_ms must be positive.")
        if self.batch_size < 1 or self.n_seed < 1:
            raise ValueError("batch_size and n_seed must be positive.")
        steps = self.duration_ms / DT_MS
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-10):
            raise ValueError(f"duration_ms must be an integer multiple of {DT_MS} ms.")

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_ms / DT_MS))

    @property
    def id(self) -> str:
        duration = f"{self.duration_ms:g}".replace(".", "p")
        return f"c{self.n_cv}_t{duration}_b{self.batch_size}_s{self.n_seed}"



class PreparedBenchmark(NamedTuple):
    """Compiled-input metadata for one method/configuration."""

    function: object
    seed_roots: object
    state_scalar_count_per_seed: int
    parameter_count_per_seed: int
    active_state_count_per_trajectory: int
    rtrl_carry_bytes: int | None



def build_morphology(n_cv: int) -> braincell.Morphology:
    """Build a soma with two equally segmented dendritic arms."""
    if n_cv < 1 or n_cv % 2 == 0:
        raise ValueError("n_cv must be a positive odd integer.")
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    segments = (n_cv - 1) // 2
    arm_specs = (
        ("dend_a", "basal_dendrite", 80.0, 2.0, 1.2),
        ("dend_b", "apical_dendrite", 120.0, 2.5, 1.0),
    )
    for prefix, branch_type, total_length, proximal_radius, terminal_radius in arm_specs:
        parent = "soma"
        for segment in range(segments):
            lo = segment / segments
            hi = (segment + 1) / segments
            branch = braincell.Branch.from_lengths(
                lengths=[total_length / segments] * u.um,
                radii=[
                    proximal_radius + lo * (terminal_radius - proximal_radius),
                    proximal_radius + hi * (terminal_radius - proximal_radius),
                ]
                * u.um,
                type=branch_type,
            )
            name = f"{prefix}_{segment}"
            morphology.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
    return morphology



def target_row_scales(n_cv: int) -> dict[str, np.ndarray]:
    """Return smooth deterministic per-CV target conductance scales."""
    position = np.linspace(0.0, 1.0, n_cv, dtype=np.float64)
    return {
        "leak": 1.05 + 0.15 * np.cos(np.pi * position),
        "na": 0.95 + 0.20 * np.sin(np.pi * (position + 0.15)),
        "k": 1.10 - 0.15 * np.cos(np.pi * position),
    }



def current_amplitudes(batch_size: int) -> object:
    """Return one DC-current protocol per batch member."""
    return u.Quantity(np.linspace(0.03, 0.08, batch_size, dtype=np.float64), u.nA)



def build_cell(
    config: BenchmarkConfig,
    *,
    trainable: bool,
    mechanism: MechanismSpec = FULL_HH_SPEC,
    trainable_channels: tuple[str, ...] | None = None,
    trainable_group_by: str = "cv",
) -> braincell.Cell:
    """Build one batch-population Cell whose parameters are shared over batch."""
    selected_trainables = mechanism.trainable_channels if trainable_channels is None else tuple(trainable_channels)
    if not set(selected_trainables).issubset(mechanism.painted_channels):
        raise ValueError("trainable_channels must be a subset of the painted mechanism channels.")
    scales = target_row_scales(config.n_cv)
    cell = braincell.Cell(
        build_morphology(config.n_cv),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(config.batch_size,),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    channel_scales = {
        name: 1.0 if trainable else float(scales[name][0]) if config.n_cv == 1 else scales[name]
        for name in mechanism.painted_channels
    }
    channel_mechanisms = {
        "leak": braincell.mech.Channel(
            "IL",
            name="leak",
            g_max=channel_scales.get("leak", 1.0) * _BASE_G_MAX["leak"],
        ),
        "na": braincell.mech.Channel(
            "Na_HH1952",
            name="na",
            g_max=channel_scales.get("na", 1.0) * _BASE_G_MAX["na"],
        ),
        "k": braincell.mech.Channel(
            "K_HH1952",
            name="k",
            g_max=channel_scales.get("k", 1.0) * _BASE_G_MAX["k"],
        ),
    }
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        *(channel_mechanisms[name] for name in mechanism.painted_channels),
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.CurrentClamp(
            delay=0.0 * u.ms,
            durations=config.duration_ms * u.ms,
            amplitudes=current_amplitudes(config.batch_size),
        ),
    )
    if trainable:
        for name in selected_trainables:
            cell.channels[name].trainable(
                g_max=braincell.trainable.scale(group_by=trainable_group_by, name=f"{name}.scale")
            )
    cell.init_state()
    if cell.n_cv != config.n_cv:
        raise RuntimeError(f"Requested {config.n_cv} CVs but built {cell.n_cv}.")
    return cell



def simulate_voltage(cell: braincell.Cell, times_ms) -> object:
    """Return pre-step voltage samples with shape ``(time, batch, cv)``."""
    cell.reset_state()

    def step(time_ms):
        voltage = cell.V.value.to_decimal(u.mV)
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return voltage

    return brainstate.transform.for_loop(step, times_ms)



def seed_parameter_roots(parameter_states, *, n_seed: int) -> tuple[object, ...]:
    """Return deterministic seed-leading optimizer roots."""
    random = brainstate.random.RandomState(RNG_SEED)
    roots = []
    for state in parameter_states.values():
        shape = tuple(state.value.shape)
        roots.append(random.uniform(0.75, 1.25, size=(n_seed,) + shape))
    return tuple(roots)



def prepare_benchmark(
    config: BenchmarkConfig,
    method: str,
    *,
    mechanism: MechanismSpec = FULL_HH_SPEC,
) -> PreparedBenchmark:
    """Build one seed-blocked gradient kernel and its static inputs."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}.")
    times_ms = jnp.arange(config.num_steps, dtype=jnp.float64) * DT_MS
    measured_backsub = os.environ.get("BRAINCELL_DHS_BACKSUB", "recursive")
    os.environ["BRAINCELL_DHS_BACKSUB"] = "recursive"
    try:
        target_cell = build_cell(config, trainable=False, mechanism=FULL_HH_SPEC)
        target_voltage = simulate_voltage(target_cell, times_ms)
    finally:
        os.environ["BRAINCELL_DHS_BACKSUB"] = measured_backsub
    candidate = build_cell(config, trainable=True, mechanism=mechanism)

    def rollout_step(data):
        time_ms, target_mv = data
        voltage = candidate.V.value.to_decimal(u.mV)
        local_loss = jnp.mean((voltage - target_mv) ** 2) / config.num_steps
        with brainstate.environ.context(t=time_ms * u.ms):
            candidate.update()
        return local_loss

    engine = build_rollout_value_and_grad(candidate, step=rollout_step, method=method)
    engine.prepare((times_ms[0], target_voltage[0]))
    parameter_states = engine.parameter_states
    seed_roots = seed_parameter_roots(parameter_states, n_seed=config.n_seed)
    names = engine.parameter_names

    if method == "bptt":
        one_seed = engine._bptt
        carry_bytes = None
    else:
        one_seed = engine._rtrl
        one_roots = tuple(root[0] for root in seed_roots)
        _values, one_tangents = engine._initial_full_carry(one_roots)
        carry_bytes = config.n_seed * _tree_nbytes(one_tangents)

    def seed_step(roots):
        result = one_seed(roots, (times_ms, target_voltage))
        gradient = jnp.concatenate([jnp.ravel(result.gradients[name]) for name in names])
        return result.loss, result.losses, gradient

    function = jax.vmap(seed_step)
    initial_values = engine._initial_primal_values(tuple(root[0] for root in seed_roots))
    state_count = sum(
        int(np.prod(np.shape(leaf), dtype=np.int64)) if np.shape(leaf) else 1
        for leaf in jax.tree.leaves(initial_values)
    )
    parameter_count = sum(int(np.prod(root.shape[1:], dtype=np.int64)) for root in seed_roots)
    return PreparedBenchmark(
        function=function,
        seed_roots=seed_roots,
        state_scalar_count_per_seed=state_count,
        parameter_count_per_seed=parameter_count,
        active_state_count_per_trajectory=mechanism.state_variables_per_cv * config.n_cv,
        rtrl_carry_bytes=carry_bytes,
    )



def _tree_nbytes(tree) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree)
    )
