#!/usr/bin/env python3
"""Measure dense channel-layout costs and a compact packed reference.

The benchmark contract and interpretation rules live in
``docs/specs/2026-08-16-channel-layout-cost-benchmark.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import platform
import sys
import time
from typing import Callable

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import mech
from braincell.filter import AllRegion, BranchSlice, at
from braincell._multi_compartment.run import _make_run_loop
from braincell.quad import ind_exp_euler_step


CV_COUNTS = (4, 32, 32, 32)
POP_SIZES = (1, 10, 100, 1000)
CHANNEL_NAMES = ("na", "k", "leak")
PROFILE_REGIONS = {
    "soma": ("soma", "soma", "soma"),
    "half_dend": ("half_dend", "half_dend", "half_dend"),
    "one_dend": ("one_dend", "one_dend", "one_dend"),
    "mixed": ("soma", "half_dend", "one_dend"),
    "global": ("global", "global", "global"),
}


def build_morphology() -> braincell.Morphology:
    """Build the deterministic four-branch benchmark morphology.

    Returns
    -------
    braincell.Morphology
        Morphology containing one soma, two dendrites, and one axon.
    """
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = braincell.Branch.from_lengths(
        lengths=[200.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    dend_b = braincell.Branch.from_lengths(
        lengths=[200.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    axon = braincell.Branch.from_lengths(
        lengths=[300.0] * u.um,
        radii=[1.0, 0.5] * u.um,
        type="axon",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend_a = dend_a
    morphology.soma.dend_b = dend_b
    morphology.soma.axon = axon
    return morphology


def _region(name: str):
    if name == "soma":
        return BranchSlice(branch_index=0, prox=0.0, dist=1.0)
    if name == "half_dend":
        return BranchSlice(branch_index=1, prox=0.0, dist=0.5)
    if name == "one_dend":
        return BranchSlice(branch_index=2, prox=0.0, dist=1.0)
    if name == "global":
        return AllRegion()
    raise ValueError(f"Unknown coverage region {name!r}.")


def build_cell(*, pop_size: int, profile: str = "mixed") -> braincell.Cell:
    """Build and initialize the full 100-CV benchmark cell.

    Parameters
    ----------
    pop_size : int
        Number of homogeneous cells in the population axis.
    profile : str, optional
        Coverage profile from :data:`PROFILE_REGIONS`.

    Returns
    -------
    braincell.Cell
        Initialized cell with HH sodium, HH potassium, and leak channels.

    Raises
    ------
    ValueError
        If ``profile`` is not a registered coverage profile.
    """
    if profile not in PROFILE_REGIONS:
        raise ValueError(f"Unknown profile {profile!r}.")
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranchList(CV_COUNTS),
        pop_size=(int(pop_size),),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
    )
    na_region, k_region, leak_region = PROFILE_REGIONS[profile]
    cell.paint(
        _region(na_region),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
    )
    cell.paint(
        _region(k_region),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
    )
    cell.paint(
        _region(leak_region),
        mech.Channel(
            "IL",
            name="leak",
            g_max=0.3 * u.mS / u.cm**2,
            E=-54.387 * u.mV,
        ),
    )
    cell.place(at("soma", 0.5), mech.StateProbe(name="v", field="v"))
    cell.init_state()
    return cell


def _array(value):
    return value.mantissa if isinstance(value, u.Quantity) else value


def _array_descriptor(value) -> tuple[tuple[int, ...], np.dtype, int] | None:
    array = _array(value)
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return None
    shape = tuple(int(dim) for dim in shape)
    dtype = np.dtype(dtype)
    return shape, dtype, int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _compact_nbytes(value, *, n_point: int, n_active: int) -> int:
    descriptor = _array_descriptor(value)
    if descriptor is None:
        return 0
    shape, dtype, dense_nbytes = descriptor
    if not shape or shape[-1] != int(n_point):
        return dense_nbytes
    compact_shape = shape[:-1] + (int(n_active),)
    return int(np.prod(compact_shape, dtype=np.int64)) * dtype.itemsize


def channel_layout_rows(cell: braincell.Cell) -> list[dict[str, object]]:
    """Return layout and persistent-memory rows for every density channel.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized benchmark cell.

    Returns
    -------
    list of dict
        One layout, shape, and memory-accounting record per channel.
    """
    rows = []
    for layout in cell.layouts:
        if not layout.kind.startswith("channel:"):
            continue
        declaration = cell.runtime.get_layout_mechanism(layout.id)
        node = cell.get_runtime_node(layout.id)

        declaration_bytes = 0
        projected_declaration_bytes = 0
        for (layout_id, var_name), value in cell.runtime.state_buffers.items():
            if int(layout_id) != int(layout.id) or str(var_name).startswith("_mask_"):
                continue
            descriptor = _array_descriptor(value)
            if descriptor is None:
                continue
            declaration_bytes += descriptor[2]
            projected_declaration_bytes += _compact_nbytes(
                value,
                n_point=cell.n_point,
                n_active=layout.n_active,
            )

        parameter_bytes = 0
        projected_parameter_bytes = 0
        for param_name in declaration.params.keys():
            if not hasattr(node, param_name):
                continue
            value = getattr(node, param_name)
            descriptor = _array_descriptor(value)
            if descriptor is None:
                continue
            parameter_bytes += descriptor[2]
            projected_parameter_bytes += _compact_nbytes(
                value,
                n_point=cell.n_point,
                n_active=layout.n_active,
            )

        gate_bytes = 0
        projected_gate_bytes = 0
        gate_shapes = {}
        for path, state in brainstate.graph.states(node).items():
            descriptor = _array_descriptor(state.value)
            if descriptor is None:
                continue
            gate_name = ".".join(str(part) for part in path)
            gate_shapes[gate_name] = descriptor[0]
            gate_bytes += descriptor[2]
            projected_gate_bytes += _compact_nbytes(
                state.value,
                n_point=cell.n_point,
                n_active=layout.n_active,
            )

        mask_bytes = int(np.asarray(layout.point_mask, dtype=bool).nbytes)
        packed_index_bytes = int(layout.n_active) * np.dtype(np.int32).itemsize
        dense_total = declaration_bytes + parameter_bytes + gate_bytes + mask_bytes
        packed_total = (
            projected_declaration_bytes + projected_parameter_bytes + projected_gate_bytes + packed_index_bytes
        )
        hybrid_total = min(dense_total, packed_total)
        rows.append(
            {
                "layout_id": int(layout.id),
                "kind": layout.kind,
                "name": declaration.instance_name,
                "n_active": int(layout.n_active),
                "active_fraction_cv": float(layout.n_active / cell.n_cv),
                "active_fraction_point": float(layout.n_active / cell.n_point),
                "node_shape": tuple(int(dim) for dim in node.varshape),
                "gate_shapes": gate_shapes,
                "declaration_bytes": declaration_bytes,
                "runtime_parameter_bytes": parameter_bytes,
                "gate_value_bytes": gate_bytes,
                "mask_bytes": mask_bytes,
                "dense_total_bytes": dense_total,
                "projected_declaration_bytes": projected_declaration_bytes,
                "projected_parameter_bytes": projected_parameter_bytes,
                "projected_gate_value_bytes": projected_gate_bytes,
                "packed_index_bytes": packed_index_bytes,
                "projected_packed_total_bytes": packed_total,
                "projected_saved_bytes": dense_total - packed_total,
                "projected_hybrid_total_bytes": hybrid_total,
                "hybrid_layout_choice": "packed" if packed_total < dense_total else "dense",
                "projected_hybrid_saved_bytes": dense_total - hybrid_total,
            }
        )
    return rows


def memory_summary(cell: braincell.Cell) -> dict[str, object]:
    """Summarize current and projected compact channel-owned storage.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized benchmark cell.

    Returns
    -------
    dict
        Per-channel rows plus dense, packed, and hybrid aggregate bytes.
    """
    rows = channel_layout_rows(cell)
    dense = sum(int(row["dense_total_bytes"]) for row in rows)
    packed = sum(int(row["projected_packed_total_bytes"]) for row in rows)
    hybrid = sum(int(row["projected_hybrid_total_bytes"]) for row in rows)
    return {
        "rows": rows,
        "dense_total_bytes": dense,
        "projected_packed_total_bytes": packed,
        "projected_saved_bytes": dense - packed,
        "projected_saved_fraction": 0.0 if dense == 0 else float((dense - packed) / dense),
        "projected_hybrid_total_bytes": hybrid,
        "projected_hybrid_saved_bytes": dense - hybrid,
        "projected_hybrid_saved_fraction": 0.0 if dense == 0 else float((dense - hybrid) / dense),
    }


def environment_summary() -> dict[str, object]:
    """Return reproducibility metadata for one benchmark process.

    Returns
    -------
    dict
        Python, JAX, backend, device, and precision metadata.
    """
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "precision": int(brainstate.environ.get_precision()),
        "x64_enabled": bool(jax.config.jax_enable_x64),
    }


def _cost_dict(cost: dict[str, float]) -> dict[str, float]:
    keys = ("flops", "transcendentals", "bytes accessed")
    return {key: float(cost.get(key, 0.0)) for key in keys}


def _memory_analysis_dict(stats) -> dict[str, int]:
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {name: int(getattr(stats, name, 0)) for name in names}


def _hlo_summary(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        "text_bytes": len(text.encode("utf-8")),
        "fused_computations": lower.count("%fused_computation"),
        "fusion_instructions": lower.count("_fusion ="),
    }


def current_compiler_analysis(cell: braincell.Cell, *, steps: int, dt) -> dict[str, object]:
    """Lower and compile the current full-cell run loop.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized benchmark cell.
    steps : int
        Static number of simulation steps in the lowered loop.
    dt : brainunit.Quantity
        Simulation timestep.

    Returns
    -------
    dict
        Compiler cost, memory, and optimized-HLO summaries.
    """
    probe_names = tuple(sorted(cell.sample_probes()))
    with brainstate.environ.context(dt=dt):
        relative_times = u.math.arange(0.0 * u.ms, int(steps) * dt, dt)
        lowered = _make_run_loop(cell, dt=dt, ordered_names=probe_names).lower(relative_times)
        cost = _cost_dict(lowered.cost_analysis())
        compiled = lowered.compile()
    return {
        "cost": cost,
        "memory": _memory_analysis_dict(compiled.memory_analysis()),
        "hlo": _hlo_summary(compiled.as_text()),
    }


def _block_run(result) -> None:
    jax.block_until_ready(result.time)
    for value in result.traces.values():
        jax.block_until_ready(value)


def _time_calls(call: Callable[[], object], *, discard: int, repeats: int) -> dict[str, object]:
    discarded = []
    for _ in range(int(discard)):
        started = time.perf_counter()
        result = call()
        _block_value(result)
        discarded.append(time.perf_counter() - started)
    samples = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        result = call()
        _block_value(result)
        samples.append(time.perf_counter() - started)
    return {
        "discarded_s": discarded,
        "samples_s": samples,
        "median_s": float(np.median(samples)),
        "min_s": float(np.min(samples)),
        "max_s": float(np.max(samples)),
    }


def _block_value(value) -> None:
    if hasattr(value, "traces"):
        _block_run(value)
        return
    value = _array(value)
    jax.block_until_ready(value)


def benchmark_current(
    *,
    pop_size: int,
    profile: str,
    steps: int,
    discard: int,
    repeats: int,
    include_compiler: bool,
) -> dict[str, object]:
    """Benchmark the current full-cell dense runtime.

    Parameters
    ----------
    pop_size : int
        Population size.
    profile : str
        Coverage profile name.
    steps : int
        Steps per timed call.
    discard : int
        Initial synchronized calls excluded from statistics.
    repeats : int
        Number of measured synchronized calls.
    include_compiler : bool
        Whether to include compiler analysis from an independent cell.

    Returns
    -------
    dict
        Environment, layout, memory, timing, and optional compiler data.
    """
    cell = build_cell(pop_size=pop_size, profile=profile)
    dt = 0.025 * u.ms
    duration = int(steps) * dt
    result = {
        "environment": environment_summary(),
        "mode": "current",
        "profile": profile,
        "pop_size": int(pop_size),
        "n_cv": int(cell.n_cv),
        "n_point": int(cell.n_point),
        "active_counts": [int(row["n_active"]) for row in channel_layout_rows(cell)],
        "memory": memory_summary(cell),
        "timing": _time_calls(
            lambda: cell.run(dt=dt, duration=duration),
            discard=discard,
            repeats=repeats,
        ),
    }
    if include_compiler:
        compiler_cell = build_cell(pop_size=pop_size, profile=profile)
        result["compiler"] = current_compiler_analysis(compiler_cell, steps=steps, dt=dt)
    return result


def _ion_info(shape: tuple[int, ...], *, reversal) -> braincell.IonInfo:
    return braincell.IonInfo(
        Ci=jnp.full(shape, 10.0) * u.mM,
        Co=jnp.full(shape, 140.0) * u.mM,
        E=jnp.full(shape, reversal.to_decimal(u.mV)) * u.mV,
        valence=1,
    )


def _take_info(info: braincell.IonInfo, indices: np.ndarray) -> braincell.IonInfo:
    return braincell.IonInfo(
        Ci=info.Ci[..., indices],
        Co=info.Co[..., indices],
        E=info.E[..., indices],
        valence=info.valence,
    )


def _masked_quantity(value, mask: np.ndarray):
    unit = value.unit
    mantissa = np.asarray(value.to_decimal(unit))
    return u.Quantity(np.where(mask, mantissa, 0.0), unit)


@dataclass
class ChannelMicroSystem:
    """Hold exact channel-class microbenchmark state.

    Parameters
    ----------
    layout : {"dense", "packed"}
        State layout used by :meth:`step`.
    n_point : int
        Full point-space width.
    indices : tuple of ndarray
        Active point indices for sodium, potassium, and leak channels.
    na, k, leak : object
        Runtime channel instances.
    na_info, k_info : braincell.IonInfo
        Full point-space ion information.
    """

    layout: str
    n_point: int
    indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    na: object
    k: object
    leak: object
    na_info: braincell.IonInfo
    k_info: braincell.IonInfo

    def step(self, point_v):
        """Advance channel gates once and return point-space current.

        Parameters
        ----------
        point_v : brainunit.Quantity
            Full point-space voltage.

        Returns
        -------
        brainunit.Quantity
            Total sodium, potassium, and leak current density in point space.
        """
        if self.layout == "dense":
            ind_exp_euler_step(self.na, point_v, self.na_info)
            ind_exp_euler_step(self.k, point_v, self.k_info)
            na_mask, k_mask, leak_mask = self._masks()
            na_v = u.Quantity(
                u.math.where(na_mask, point_v.to_decimal(u.mV), -65.0),
                u.mV,
            )
            k_v = u.Quantity(
                u.math.where(k_mask, point_v.to_decimal(u.mV), -65.0),
                u.mV,
            )
            na_current = u.math.where(na_mask, self.na.current(na_v, self.na_info), 0.0 * u.nA / u.cm**2)
            k_current = u.math.where(k_mask, self.k.current(k_v, self.k_info), 0.0 * u.nA / u.cm**2)
            leak_current = u.math.where(
                leak_mask,
                self.leak.current(point_v),
                0.0 * u.nA / u.cm**2,
            )
            return na_current + k_current + leak_current

        na_index, k_index, leak_index = self.indices
        na_v = point_v[..., na_index]
        k_v = point_v[..., k_index]
        leak_v = point_v[..., leak_index]
        na_info = _take_info(self.na_info, na_index)
        k_info = _take_info(self.k_info, k_index)
        ind_exp_euler_step(self.na, na_v, na_info)
        ind_exp_euler_step(self.k, k_v, k_info)
        total = u.math.zeros(point_v.shape) * (u.nA / u.cm**2)
        total = total.at[..., na_index].add(self.na.current(na_v, na_info))
        total = total.at[..., k_index].add(self.k.current(k_v, k_info))
        total = total.at[..., leak_index].add(self.leak.current(leak_v))
        return total

    def _masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = []
        for index in self.indices:
            mask = np.zeros((self.n_point,), dtype=bool)
            mask[index] = True
            masks.append(mask)
        return tuple(masks)


def _profile_indices(profile: str) -> tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cell = build_cell(pop_size=1, profile=profile)
    rows = []
    for layout in cell.layouts:
        if layout.kind.startswith("channel:"):
            rows.append(np.asarray(layout.point_index, dtype=np.int32))
    return cell.n_point, tuple(rows)


def build_micro_system(*, pop_size: int, profile: str, layout: str) -> tuple[ChannelMicroSystem, object]:
    """Build dense or packed exact-channel microbenchmark state.

    Parameters
    ----------
    pop_size : int
        Population size.
    profile : str
        Coverage profile name.
    layout : {"dense", "packed"}
        Microbenchmark state layout.

    Returns
    -------
    ChannelMicroSystem
        Initialized channel system.
    brainunit.Quantity
        Full point-space voltage supplied to the rollout.

    Raises
    ------
    ValueError
        If ``layout`` is not ``"dense"`` or ``"packed"``.
    """
    if layout not in {"dense", "packed"}:
        raise ValueError(f"Unknown micro layout {layout!r}.")
    n_point, indices = _profile_indices(profile)
    full_shape = (int(pop_size), int(n_point))
    point_v = jnp.full(full_shape, -65.0) * u.mV
    na_info = _ion_info(full_shape, reversal=50.0 * u.mV)
    k_info = _ion_info(full_shape, reversal=-77.0 * u.mV)

    if layout == "dense":
        masks = []
        for index in indices:
            mask = np.zeros((n_point,), dtype=bool)
            mask[index] = True
            masks.append(mask)
        na_g = _masked_quantity(jnp.full(full_shape, 120.0) * u.mS / u.cm**2, masks[0])
        k_g = _masked_quantity(jnp.full(full_shape, 36.0) * u.mS / u.cm**2, masks[1])
        leak_g = _masked_quantity(jnp.full(full_shape, 0.3) * u.mS / u.cm**2, masks[2])
        na = braincell.channel.Na_HH1952(size=full_shape, g_max=na_g)
        k = braincell.channel.K_HH1952(size=full_shape, g_max=k_g)
        leak = braincell.channel.IL(size=full_shape, g_max=leak_g, E=-54.387 * u.mV)
        na.init_state(point_v, na_info)
        k.init_state(point_v, k_info)
        na.reset_state(point_v, na_info)
        k.reset_state(point_v, k_info)
    else:
        na_shape = (int(pop_size), len(indices[0]))
        k_shape = (int(pop_size), len(indices[1]))
        leak_shape = (int(pop_size), len(indices[2]))
        na = braincell.channel.Na_HH1952(size=na_shape, g_max=120.0 * u.mS / u.cm**2)
        k = braincell.channel.K_HH1952(size=k_shape, g_max=36.0 * u.mS / u.cm**2)
        leak = braincell.channel.IL(size=leak_shape, g_max=0.3 * u.mS / u.cm**2, E=-54.387 * u.mV)
        na_v = point_v[..., indices[0]]
        k_v = point_v[..., indices[1]]
        na_local_info = _take_info(na_info, indices[0])
        k_local_info = _take_info(k_info, indices[1])
        na.init_state(na_v, na_local_info)
        k.init_state(k_v, k_local_info)
        na.reset_state(na_v, na_local_info)
        k.reset_state(k_v, k_local_info)

    return (
        ChannelMicroSystem(
            layout=layout,
            n_point=n_point,
            indices=indices,
            na=na,
            k=k,
            leak=leak,
            na_info=na_info,
            k_info=k_info,
        ),
        point_v,
    )


def make_micro_rollout(system: ChannelMicroSystem, *, steps: int, dt):
    """Build one compiled multi-step microbenchmark rollout.

    Parameters
    ----------
    system : ChannelMicroSystem
        Stateful channel microbenchmark.
    steps : int
        Static number of channel update steps.
    dt : brainunit.Quantity
        Timestep captured by the returned transform's execution context.

    Returns
    -------
    Callable
        Jitted function accepting full point-space voltage.
    """
    xs = jnp.arange(int(steps), dtype=jnp.int32)

    def rollout(point_v):
        with brainstate.environ.context(dt=dt):

            def one_step(_):
                return system.step(point_v)

            currents = brainstate.transform.for_loop(one_step, xs)
            return currents[-1]

    return brainstate.transform.jit(rollout)


def micro_compiler_analysis(system: ChannelMicroSystem, point_v, *, steps: int, dt) -> dict[str, object]:
    """Lower and compile one dense or packed micro rollout.

    Parameters
    ----------
    system : ChannelMicroSystem
        Stateful channel microbenchmark.
    point_v : brainunit.Quantity
        Full point-space voltage.
    steps : int
        Static number of update steps.
    dt : brainunit.Quantity
        Simulation timestep.

    Returns
    -------
    dict
        Compiler cost, memory, and optimized-HLO summaries.
    """
    with brainstate.environ.context(dt=dt):
        lowered = make_micro_rollout(system, steps=steps, dt=dt).lower(point_v)
        cost = _cost_dict(lowered.cost_analysis())
        compiled = lowered.compile()
    return {
        "cost": cost,
        "memory": _memory_analysis_dict(compiled.memory_analysis()),
        "hlo": _hlo_summary(compiled.as_text()),
    }


def micro_parity(*, pop_size: int = 2, profile: str = "mixed", steps: int = 3) -> dict[str, float]:
    """Compare dense and packed current and gates after identical rollouts.

    Parameters
    ----------
    pop_size : int, optional
        Population size used by both systems.
    profile : str, optional
        Coverage profile name.
    steps : int, optional
        Number of channel update steps.

    Returns
    -------
    dict
        Maximum absolute point-current and active-gate errors.
    """
    dt = 0.025 * u.ms
    dense, dense_v = build_micro_system(pop_size=pop_size, profile=profile, layout="dense")
    packed, packed_v = build_micro_system(pop_size=pop_size, profile=profile, layout="packed")
    with brainstate.environ.context(dt=dt):
        dense_current = make_micro_rollout(dense, steps=steps, dt=dt)(dense_v)
        packed_current = make_micro_rollout(packed, steps=steps, dt=dt)(packed_v)
    jax.block_until_ready(dense_current.mantissa)
    jax.block_until_ready(packed_current.mantissa)

    current_error = float(
        np.max(
            np.abs(
                np.asarray(dense_current.to_decimal(u.nA / u.cm**2))
                - np.asarray(packed_current.to_decimal(u.nA / u.cm**2))
            )
        )
    )
    gate_error = 0.0
    for dense_node, packed_node, index in (
        (dense.na, packed.na, dense.indices[0]),
        (dense.k, packed.k, dense.indices[1]),
    ):
        dense_states = brainstate.graph.states(dense_node)
        packed_states = brainstate.graph.states(packed_node)
        for path in dense_states:
            dense_value = np.asarray(dense_states[path].value)[..., index]
            packed_value = np.asarray(packed_states[path].value)
            gate_error = max(gate_error, float(np.max(np.abs(dense_value - packed_value))))
    return {"max_current_error": current_error, "max_gate_error": gate_error}


def benchmark_micro(
    *,
    pop_size: int,
    profile: str,
    layout: str,
    steps: int,
    discard: int,
    repeats: int,
) -> dict[str, object]:
    """Benchmark one exact-channel dense or packed micro rollout.

    Parameters
    ----------
    pop_size : int
        Population size.
    profile : str
        Coverage profile name.
    layout : {"dense", "packed"}
        Microbenchmark state layout.
    steps : int
        Steps per compiled rollout.
    discard : int
        Initial synchronized calls excluded from statistics.
    repeats : int
        Number of measured synchronized calls.

    Returns
    -------
    dict
        Environment, compiler, and timing data.
    """
    dt = 0.025 * u.ms
    compiler_system, compiler_v = build_micro_system(pop_size=pop_size, profile=profile, layout=layout)
    compiler = micro_compiler_analysis(compiler_system, compiler_v, steps=steps, dt=dt)
    timing_system, timing_v = build_micro_system(pop_size=pop_size, profile=profile, layout=layout)
    rollout = make_micro_rollout(timing_system, steps=steps, dt=dt)

    def call():
        with brainstate.environ.context(dt=dt):
            return rollout(timing_v)

    return {
        "environment": environment_summary(),
        "mode": "micro",
        "layout": layout,
        "profile": profile,
        "pop_size": int(pop_size),
        "steps": int(steps),
        "n_point": int(timing_system.n_point),
        "active_counts": [int(len(index)) for index in timing_system.indices],
        "compiler": compiler,
        "timing": _time_calls(call, discard=discard, repeats=repeats),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "micro", "parity"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_REGIONS), default="mixed")
    parser.add_argument("--pop-size", type=int, default=1)
    parser.add_argument("--layout", choices=("dense", "packed"), default="dense")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--discard", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--compiler-analysis", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one isolated configuration and print one JSON record.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments. ``None`` reads :data:`sys.argv`.

    Returns
    -------
    int
        Zero on success.

    Raises
    ------
    ValueError
        If numeric benchmark arguments are outside their valid ranges.
    """
    args = _parse_args(argv)
    if args.pop_size <= 0 or args.steps <= 0 or args.discard < 0 or args.repeats <= 0:
        raise ValueError("pop-size, steps, and repeats must be positive; discard must be non-negative.")
    if args.mode == "current":
        result = benchmark_current(
            pop_size=args.pop_size,
            profile=args.profile,
            steps=args.steps,
            discard=args.discard,
            repeats=args.repeats,
            include_compiler=args.compiler_analysis,
        )
    elif args.mode == "micro":
        result = benchmark_micro(
            pop_size=args.pop_size,
            profile=args.profile,
            layout=args.layout,
            steps=args.steps,
            discard=args.discard,
            repeats=args.repeats,
        )
    else:
        result = {
            "environment": environment_summary(),
            "mode": "parity",
            "profile": args.profile,
            "pop_size": args.pop_size,
            **micro_parity(pop_size=args.pop_size, profile=args.profile, steps=args.steps),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
