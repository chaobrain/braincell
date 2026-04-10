from __future__ import annotations

from dataclasses import asdict, dataclass, field
import cProfile
import io
import pstats
import time
from typing import Any

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel
from braincell.quad import _voltage_solver as voltage_solver

if False:  # pragma: no cover
    from .cell import Cell

__all__ = ["CellProfileReport", "profile_cell"]

# Profiling is intentionally out-of-band: it clones and replays a declaration
# cell so timing/inspection does not mutate the user's live simulation object.


@dataclass(frozen=True)
class CellProfileReport:
    """Structured profile output for one declaration-level cell run.

    This report is a pure analysis artifact. It does not participate in runtime
    execution; instead :func:`profile_cell` clones a cell, runs a controlled
    measurement sequence, and stores the resulting timing breakdown here.

    Field groups:

    - overall timings such as ``init_state_ms`` and ``avg_update_ms``
    - optional staggered-solver breakdown fields for voltage-step internals
    - metadata describing the measured cell shape and solver choice
    - optional cProfile text and summarized top entries

    ``Cell.profile(...)`` returns this type directly so callers can print it,
    turn it into a dict, or persist profiling results without depending on the
    timing helpers themselves.
    """
    init_state_ms: float
    warmup_update_ms: float
    avg_update_ms: float
    channel_update_ms: float | None
    dhs_voltage_step_ms: float | None
    independent_channel_integrate_ms: float | None
    linear_and_const_ms: float | None
    point_tree_linear_system_ms: float | None
    dhs_pack_and_convert_ms: float | None
    triang_ms: float | None
    backsub_ms: float | None
    dhs_finalize_ms: float | None
    residual_ms: float | None
    n_cv: int
    n_point: int | None
    solver: str
    steps_measured: int
    dt_ms: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    cprofile_text: str | None = None
    cprofile_top_entries: tuple[tuple[str, float, float, int], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        lines = [
            "Cell Profile Report",
            f"solver={self.solver} n_cv={self.n_cv} n_point={self.n_point} dt_ms={self.dt_ms:.6f}",
            f"init_state_ms={self.init_state_ms:.3f}",
            f"warmup_update_ms={self.warmup_update_ms:.3f}",
            f"avg_update_ms={self.avg_update_ms:.3f} steps={self.steps_measured}",
        ]
        if self.channel_update_ms is not None:
            lines.extend([
                f"channel_update_ms={self.channel_update_ms:.3f}",
                f"dhs_voltage_step_ms={self.dhs_voltage_step_ms:.3f}",
                f"independent_channel_integrate_ms={self.independent_channel_integrate_ms:.3f}",
                f"linear_and_const_ms={self.linear_and_const_ms:.3f}",
                f"point_tree_linear_system_ms={self.point_tree_linear_system_ms:.3f}",
                f"dhs_pack_and_convert_ms={self.dhs_pack_and_convert_ms:.3f}",
                f"triang_ms={self.triang_ms:.3f}",
                f"backsub_ms={self.backsub_ms:.3f}",
                f"dhs_finalize_ms={self.dhs_finalize_ms:.3f}",
                f"residual_ms={self.residual_ms:.3f}",
            ])
        if self.notes:
            lines.append("notes=" + "; ".join(self.notes))
        return "\n".join(lines)


def profile_cell(
    cell: "Cell",
    *,
    steps: int = 20,
    warmup_steps: int = 1,
    repeat_init: int = 3,
    I_ext=0.0 * u.nA,
    include_cprofile: bool = False,
    top_k: int = 20,
) -> CellProfileReport:
    if steps < 1:
        raise ValueError("Cell.profile(...) requires steps >= 1.")
    if warmup_steps < 0:
        raise ValueError("Cell.profile(...) requires warmup_steps >= 0.")
    if repeat_init < 1:
        raise ValueError("Cell.profile(...) requires repeat_init >= 1.")

    dt = brainstate.environ.get("dt")
    if dt is None:
        raise ValueError("Cell.profile(...) requires brainstate.environ['dt'] to be set.")
    dt_ms = float(dt.to_decimal(u.ms))

    init_state_ms = _measure_init_state_ms(cell, repeat_init=repeat_init)
    probe = _clone_profile_cell(cell)
    probe.init_state()

    with brainstate.environ.context(t=brainstate.environ.get("t", 0.0), dt=dt):
        warmup_update_ms = _measure_warmup(probe, warmup_steps=warmup_steps, I_ext=I_ext)
        breakdown = _measure_solver_breakdown(probe, dt=dt, I_ext=I_ext)
        avg_update_ms = _measure_avg_update(probe, steps=steps, I_ext=I_ext)
        cprofile_text, cprofile_top_entries = _measure_cprofile(
            probe,
            I_ext=I_ext,
            include_cprofile=include_cprofile,
            top_k=top_k,
        )

    known_parts = [
        value
        for value in (
            breakdown["channel_update_ms"],
            breakdown["dhs_voltage_step_ms"],
            breakdown["independent_channel_integrate_ms"],
        )
        if value is not None
    ]
    residual_ms = None if not known_parts else max(0.0, avg_update_ms - sum(known_parts))
    notes = (
        "timings are measured on a declaration-level clone and do not mutate the original cell",
        "init_state_ms includes fresh Cell construction, declaration replay, and init_state()",
    )
    return CellProfileReport(
        init_state_ms=init_state_ms,
        warmup_update_ms=warmup_update_ms,
        avg_update_ms=avg_update_ms,
        channel_update_ms=breakdown["channel_update_ms"],
        dhs_voltage_step_ms=breakdown["dhs_voltage_step_ms"],
        independent_channel_integrate_ms=breakdown["independent_channel_integrate_ms"],
        linear_and_const_ms=breakdown["linear_and_const_ms"],
        point_tree_linear_system_ms=breakdown["point_tree_linear_system_ms"],
        dhs_pack_and_convert_ms=breakdown["dhs_pack_and_convert_ms"],
        triang_ms=breakdown["triang_ms"],
        backsub_ms=breakdown["backsub_ms"],
        dhs_finalize_ms=breakdown["dhs_finalize_ms"],
        residual_ms=residual_ms,
        n_cv=probe.n_cv,
        n_point=breakdown["n_point"],
        solver=probe.solver_name,
        steps_measured=steps,
        dt_ms=dt_ms,
        notes=notes,
        cprofile_text=cprofile_text,
        cprofile_top_entries=cprofile_top_entries,
    )


def _measure_init_state_ms(cell: "Cell", *, repeat_init: int) -> float:
    start = time.perf_counter()
    for _ in range(repeat_init):
        probe = _clone_profile_cell(cell)
        probe.init_state()
    return (time.perf_counter() - start) * 1000.0 / repeat_init


def _measure_warmup(cell: "Cell", *, warmup_steps: int, I_ext) -> float:
    if warmup_steps == 0:
        return 0.0
    start = time.perf_counter()
    for _ in range(warmup_steps):
        cell.update(I_ext)
    return (time.perf_counter() - start) * 1000.0 / warmup_steps


def _measure_avg_update(cell: "Cell", *, steps: int, I_ext) -> float:
    start = time.perf_counter()
    for _ in range(steps):
        cell.update(I_ext)
    return (time.perf_counter() - start) * 1000.0 / steps


def _measure_solver_breakdown(cell: "Cell", *, dt, I_ext) -> dict[str, float | int | None]:
    if cell.solver_name != "staggered":
        return {
            "channel_update_ms": None,
            "dhs_voltage_step_ms": None,
            "independent_channel_integrate_ms": None,
            "linear_and_const_ms": None,
            "point_tree_linear_system_ms": None,
            "dhs_pack_and_convert_ms": None,
            "triang_ms": None,
            "backsub_ms": None,
            "dhs_finalize_ms": None,
            "n_point": None,
        }

    point_tree = cell.point_tree()
    scheduling = cell.point_scheduling(algorithm="dhs")
    V_n = cell.V.value
    point_V = cell._point_voltage(V_n)

    start = time.perf_counter()
    for _, node in cell.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
        node.update(point_V)
    channel_update_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    linear, const = voltage_solver._linear_and_const_term(cell, V_n, I_ext)
    linear_and_const_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    system = voltage_solver._point_tree_linear_system(cell, point_tree=point_tree, scheduling=scheduling, dt=dt)
    point_tree_linear_system_ms = (time.perf_counter() - start) * 1000.0

    V_n, linear, const = [x.reshape((-1, V_n.shape[-1])) for x in (V_n, linear, const)]
    n_point = int(system["n_point"])

    start = time.perf_counter()
    static_cache = voltage_solver._build_dhs_static_cache(system)
    numeric = voltage_solver._build_dhs_numeric_state(
        V_n,
        linear,
        const,
        dt=dt,
        static_cache=static_cache,
    )
    dhs_pack_and_convert_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    diags, solves = voltage_solver.comp_triang_raw(
        numeric.diags,
        numeric.solves,
        numeric.lowers,
        numeric.uppers,
        numeric.edges,
        static_cache.level_offsets_np,
    )
    triang_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    voltage_solver.comp_backsub_raw(
        diags,
        solves,
        numeric.lowers,
        static_cache.backsub_indices_jnp,
    )
    backsub_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    cell.V.value = voltage_solver._restore_midpoint_voltage(
        solves,
        dynamic_rows=numeric.dynamic_rows,
        target_shape=cell.V.value.shape,
    )
    dhs_finalize_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    from braincell.quad._exp_euler import ind_exp_euler_step
    ind_exp_euler_step(cell, I_ext, excluded_paths=[("V",)])
    independent_channel_integrate_ms = (time.perf_counter() - start) * 1000.0

    dhs_voltage_step_ms = (
        linear_and_const_ms
        + point_tree_linear_system_ms
        + dhs_pack_and_convert_ms
        + triang_ms
        + backsub_ms
        + dhs_finalize_ms
    )

    return {
        "channel_update_ms": channel_update_ms,
        "dhs_voltage_step_ms": dhs_voltage_step_ms,
        "independent_channel_integrate_ms": independent_channel_integrate_ms,
        "linear_and_const_ms": linear_and_const_ms,
        "point_tree_linear_system_ms": point_tree_linear_system_ms,
        "dhs_pack_and_convert_ms": dhs_pack_and_convert_ms,
        "triang_ms": triang_ms,
        "backsub_ms": backsub_ms,
        "dhs_finalize_ms": dhs_finalize_ms,
        "n_point": n_point,
    }


def _measure_cprofile(
    cell: "Cell",
    *,
    I_ext,
    include_cprofile: bool,
    top_k: int,
) -> tuple[str | None, tuple[tuple[str, float, float, int], ...]]:
    if not include_cprofile:
        return None, ()

    profiler = cProfile.Profile()
    profiler.enable()
    cell.update(I_ext)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(top_k)

    entries = []
    for func_key, stat in list(stats.stats.items())[:top_k]:
        cc, nc, tt, ct, _ = stat
        filename, lineno, name = func_key
        entries.append((f"{filename}:{lineno}({name})", float(tt), float(ct), int(nc)))
    return buffer.getvalue(), tuple(entries)


def _clone_profile_cell(cell: "Cell") -> "Cell":
    from .cell import Cell

    probe = Cell(
        cell.morpho,
        cv_policy=cell.cv_policy,
        V_th=cell._V_th_value,
        V_initializer=cell._V_initializer_spec,
        spk_fun=cell._spk_fun,
        solver=cell.solver_name,
        name=getattr(cell, "name", None),
    )
    for rule in cell.paint_rules[1:]:
        probe.paint(rule.region, rule.mechanism)
    for rule in cell.place_rules:
        probe.place(rule.locset, *rule.mechanisms)
    return probe
