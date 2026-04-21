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

from dataclasses import dataclass
from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel
from braincell.compute._assignment_table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell.compute._point_tree import (
    PointScheduling,
    PointTree,
    build_point_scheduling,
    build_point_tree,
)
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
    clone_morpho,
    cv_value_vector,
    gather_midpoint_values,
    install_cell_runtime,
    is_python_zero,
    matches_last_dim,
    mechanism_signature,
    scatter_midpoint_values,
)
from braincell.cv._cv import CV, assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    PaintRule,
    PlaceRule,
    apply_paint_rules,
    apply_place_rules,
    default_paint_rules,
    init_cv_mech,
    merge_paint_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell.cv._policy import CVPerBranch, CVPolicy
from braincell.filter import LocsetExpr, RegionExpr
from braincell.mech import CurrentProbe, Density, MechanismProbe, StateProbe
from braincell.morph.morphology import Morphology
from braincell.quad.protocol import DiffEqState, IndependentIntegration
from braincell.quad import _staggered as voltage_solver, get_integrator

__all__ = ["Cell", "RunResult"]


def _cast_like(value, like):
    dtype = jnp.asarray(u.get_magnitude(like)).dtype
    if isinstance(value, u.Quantity):
        unit = u.get_unit(value)
        return jnp.asarray(value.to_decimal(unit), dtype=dtype) * unit
    return jnp.asarray(value, dtype=dtype)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunResult:
    time: object
    traces: dict[str, object]


class Cell(HHTypedNeuron):
    """Main cell object owning declaration, compilation, and runtime state.

    ``Cell`` is the public orchestration object for the multi-compartment stack.
    It plays three roles at once:

    - declaration frontend: collect ``paint`` / ``place`` requests
    - lazy rebuild owner: lower morphology + policy + rules into immutable CVs
    - runtime facade: compile layouts into runtime nodes and expose simulation APIs

    The key design choice is that declaration is cheap while rebuild/compile is
    deferred. Mutating declarations only flips dirty flags; the expensive
    geometry/mechanism/runtime work happens later when a derived view is needed.

    Core owned state:

    - a cloned morphology snapshot that no longer follows outside edits
    - one active :class:`CVPolicy`
    - normalized ``paint`` and ``place`` declarations
    - cached immutable :class:`CV` objects and derived :class:`PointTree`
    - cached compiled :class:`CellRuntimeState`
    - dirty flags tracking whether structure, mechanisms, or installed runtime
      must be rebuilt

    Main method groups:

    - declaration: :meth:`paint`, :meth:`place`
    - structural inspection: :attr:`cvs`, :meth:`point_tree`,
      :meth:`point_scheduling`
    - runtime inspection: layout/state/runtime-node getters
    - state lifecycle: :meth:`init_state`, :meth:`reset_state`
    - solver path: :meth:`compute_derivative`, :meth:`update`,
      :meth:`compute_membrane_derivative`, :meth:`compute_axial_derivative`

    Main collaborators:

    - :func:`braincell.cv._geo.build_cv_geo` decides CV geometry from
      morphology and policy
    - :mod:`braincell.cv._mech` maps declaration rules onto those CVs
    - :class:`braincell.compute.PointTree` provides the merged point-edge
      execution view
    - :class:`braincell.compute._runtime.CellRuntimeState` lowers immutable
      declarations into runtime layouts, state buffers, and installed
      mechanism nodes

    The result is that users interact almost entirely with ``Cell``, while the
    lower layers stay explicit and inspectable for debugging or solver work.
    """

    __module__ = "braincell"

    def __init__(
        self,
        morpho: Morphology,
        *,
        cv_policy: CVPolicy | None = None,
        V_th: object = -75 * u.mV,
        V_initializer: object | None = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        name: str | None = None,
    ) -> None:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"Cell expects Morpho, got {type(morpho).__name__!s}.")

        super().__init__(
            size=(1,),
            name=name,
            **build_placeholder_ions(),
        )

        self._morpho = clone_morpho(morpho)
        self._cv_policy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}.")

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()
        self._cvs: tuple[CV, ...] | None = None
        self._point_tree: PointTree | None = None
        self._point_scheduling: dict[tuple[str, int], PointScheduling] = {}
        self._compiled_runtime: CellRuntimeState | None = None
        self._cached_voltage_linearizer = None
        self._cached_axial_operator = None
        self._frontend_dirty = True
        self._structure_dirty = True
        self._mechanism_dirty = True
        self._value_dirty = False
        self._state_initialized = False
        self._current_time = brainstate.ShortTermState(0.0 * u.ms)
        self._V_th_value = V_th
        self._V_initializer_spec = V_initializer
        self._spk_fun = spk_fun
        self.solver_name, self.solver = _resolve_solver(solver)

        # Build the first immutable CV view eagerly so basic inspection APIs are
        # available right after construction. Runtime installation still stays lazy.
        self._rebuild_if_needed()

    @property
    def pop_size(self) -> tuple[int, ...]:
        return ()

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        if not isinstance(value, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(value).__name__!s}.")
        self._cv_policy = value
        self._mark_dirty(structure=True, mechanism=True)

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    def paint(self, region: RegionExpr, *mechanisms: object) -> "Cell":
        # ``paint`` only records normalized declarations. No runtime mutation
        # happens here; the next rebuild will replay all rules onto fresh CVs.
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._mark_dirty(mechanism=True)
        return self

    def place(self, locset: LocsetExpr, *mechanisms: object) -> "Cell":
        # Point mechanisms are stored as declarations on the frontend, then
        # attached to the CV that owns the evaluated location during rebuild.
        self._place_rules = self._place_rules + (
            normalize_place_rule(locset, mechanisms),
        )
        self._mark_dirty(mechanism=True)
        return self

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self) -> tuple[CV, ...]:
        return self._rebuild_if_needed()

    @property
    def layouts(self):
        return self._ensure_runtime_compiled().layouts

    @property
    def voltage_shape(self) -> tuple[int, ...]:
        return self._ensure_runtime_compiled().voltage_shape

    @property
    def current_time(self):
        return self._current_time.value

    @property
    def _dirty(self) -> bool:
        return self._frontend_dirty

    def __repr__(self) -> str:
        return (
            f"Cell(root={self.morpho.root.name!r}, n_branches={len(self.morpho.branches)!r}, "
            f"n_cv={self.n_cv!r}, n_paint_rules={len(self.paint_rules)!r}, "
            f"n_place_rules={len(self.place_rules)!r})"
        )

    def __str__(self) -> str:
        return (
            f"{'-' * 35}\n"
            f"{'root':<14} | {self.morpho.root.name}\n"
            f"{'n_branches':<14} | {len(self.morpho.branches)}\n"
            f"{'n_cv':<14} | {self.n_cv}\n"
            f"{'n_paint_rules':<14} | {len(self.paint_rules)}\n"
            f"{'n_place_rules':<14} | {len(self.place_rules)}\n"
            f"{'-' * 35}\n"
        )

    def point_tree(self) -> PointTree:
        self._rebuild_if_needed()
        cached = self._point_tree
        if cached is not None:
            return cached
        # ``PointTree`` is the execution view derived from the current immutable
        # CV list. It is not a second morphology object; it is a merged point-edge
        # graph used by runtime lowering and matrix assembly.
        tree = build_point_tree(
            self._morpho,
            cvs=self.cvs,
        )
        self._point_tree = tree
        return tree

    def point_scheduling(
        self,
        *,
        max_group_size: int = 32,
        algorithm: str = "dhs",
    ) -> PointScheduling:
        self._rebuild_if_needed()
        cache_key = (algorithm, max_group_size)
        cached = self._point_scheduling.get(cache_key)
        if cached is not None:
            return cached
        tree = self.point_tree()
        # Scheduling is derived entirely from the point tree and cached
        # separately because callers may request different algorithms/group sizes.
        scheduling = build_point_scheduling(
            tree,
            max_group_size=max_group_size,
            algorithm=algorithm,
        )
        self._point_scheduling[cache_key] = scheduling
        return scheduling

    def get_point_layouts(self, point_id: int):
        return self._ensure_runtime_compiled().get_point_layouts(point_id)

    def get_cv_layouts(self, cv_id: int):
        return self._ensure_runtime_compiled().get_cv_layouts(cv_id)

    def expected_state_shape(self, layout_id: int, var_name: str) -> tuple[int, ...]:
        return self._ensure_runtime_compiled().expected_state_shape(layout_id, var_name)

    def get_state(self, layout_id: int, var_name: str):
        return self._ensure_runtime_compiled().get_state(layout_id, var_name)

    def set_state(self, layout_id: int, var_name: str, value: object) -> None:
        runtime = self._ensure_runtime_compiled()
        runtime.set_state(layout_id, var_name, value)
        self._value_dirty = False

    def get_point_state(self, point_id: int) -> dict[int, dict[str, object]]:
        return self._ensure_runtime_compiled().get_point_state(point_id)

    def get_cv_state(self, cv_id: int) -> dict[int, dict[str, object]]:
        return self._ensure_runtime_compiled().get_cv_state(cv_id)

    def get_runtime_node(self, layout_id: int) -> object:
        return self._ensure_runtime_compiled().get_runtime_node(layout_id)

    def get_ion(self, name: str) -> object:
        return self._ensure_runtime_compiled().get_ion(name)

    def sample_probe(self, name: str) -> object:
        self._ensure_runtime_ready()
        runtime = self._ensure_runtime_compiled()
        matches: list[tuple[object, object]] = []
        for layout in runtime.layouts:
            declaration = runtime.get_layout_mechanism(layout.id)
            if isinstance(declaration, (StateProbe, MechanismProbe, CurrentProbe)) and _probe_name(declaration) == name:
                matches.append((layout, declaration))
        if len(matches) == 0:
            raise KeyError(f"No probe is registered with name {name!r}.")
        if len(matches) > 1:
            raise ValueError(f"Multiple probes share the same name {name!r}; probe names must be unique.")
        layout, declaration = matches[0]
        return self._sample_probe_layout(runtime, layout=layout, declaration=declaration)

    def sample_probes(self) -> dict[str, object]:
        self._ensure_runtime_ready()
        runtime = self._ensure_runtime_compiled()
        sampled: dict[str, object] = {}
        for layout in runtime.layouts:
            declaration = runtime.get_layout_mechanism(layout.id)
            if not isinstance(declaration, (StateProbe, MechanismProbe, CurrentProbe)):
                continue
            probe_name = _probe_name(declaration)
            if probe_name in sampled:
                raise ValueError(
                    f"Multiple probes share the same name {probe_name!r}; probe names must be unique."
                )
            sampled[probe_name] = self._sample_probe_layout(
                runtime,
                layout=layout,
                declaration=declaration,
            )
        return sampled

    def run(self, *, dt, duration) -> RunResult:
        self._ensure_runtime_ready()
        _validate_time_quantity(dt, name="dt")
        _validate_time_quantity(duration, name="duration")

        initial_samples = self.sample_probes()
        if len(initial_samples) == 0:
            raise ValueError("Cell.run(...) requires at least one placed probe.")
        ordered_names = tuple(sorted(initial_samples))

        with brainstate.environ.context(dt=dt):
            start_t = self.current_time
            relative_times = u.math.arange(0.0 * u.ms, duration, brainstate.environ.get_dt())
            if int(relative_times.shape[0]) == 0:
                raise ValueError("Cell.run(...) produced no timesteps; ensure duration > 0 and dt > 0.")
            times = start_t + relative_times

            def _step_run(t):
                self._current_time.value = t
                self.update()
                snapshot = self.sample_probes()
                return tuple(snapshot[name] for name in ordered_names)

            traces_over_time = brainstate.transform.for_loop(_step_run, times)
            self._current_time.value = start_t + (int(times.shape[0]) * brainstate.environ.get_dt())

        traces_tuple = _normalize_run_traces(traces_over_time, n_traces=len(ordered_names))
        traces = {
            name: trace
            for name, trace in zip(ordered_names, traces_tuple)
        }
        return RunResult(time=times, traces=traces)

    def mech_table(self) -> MechanismObjectTable:
        self._ensure_runtime_compiled()
        return self._mech_table()

    def init_state(self, batch_size=None) -> None:
        # ``init_state`` is the boundary where declaration-level state becomes a
        # runnable neuron object with concrete voltage/spike/channel states.
        self._ensure_runtime_compiled()
        self.V = DiffEqState(braintools.init.param(self._resolve_V_initializer(), self.varshape, batch_size))
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        self._current_time.value = 0.0 * u.ms
        point_V = self._point_voltage(self.V.value)
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        for channel in nodes:
            channel.init_state(point_V, batch_size=batch_size)
        self._state_initialized = True
        self._value_dirty = False

    def reset_state(self, batch_size=None) -> None:
        if self._compiled_runtime is None or self._structure_dirty or self._mechanism_dirty or not hasattr(self, "V"):
            self.init_state(batch_size=batch_size)
            return
        if not self._state_initialized:
            self.init_state(batch_size=batch_size)
            return
        self.V.value = braintools.init.param(self._resolve_V_initializer(), self.varshape, batch_size)
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time.value = 0.0 * u.ms
        point_V = self._point_voltage(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.reset_state(point_V, batch_size=batch_size)
        self._value_dirty = False

    def pre_integral(self, I_ext=0.0):
        self._ensure_runtime_ready()
        point_V = self._point_voltage(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(point_V)

    def compute_derivative(self, I_ext=0.0):
        self._ensure_runtime_ready()
        if is_python_zero(I_ext):
            self.V.derivative = self.compute_voltage_derivative(self.V.value)
        else:
            self.V.derivative = self.compute_voltage_derivative(self.V.value, I_ext)
        point_V = self._point_voltage(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.compute_derivative(point_V)

    def compute_membrane_derivative(self, V, I_ext=0.0):
        point_V = self._point_voltage(V)
        # Point clamps are intrinsic declarations owned by the cell. ``I_ext`` is
        # the caller-supplied external current pathway. Both are converted into
        # point-level current density before they are summed with channel currents.
        point_clamp_input = self._point_clamp_input()
        point_external_input = self._point_external_input(I_ext)
        summed_inputs = self.sum_current_inputs(point_clamp_input, point_external_input, point_V)
        I_total = None if is_python_zero(summed_inputs) else summed_inputs
        for key, ch in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            try:
                current = ch.current(point_V)
                if current is None:
                    continue
                I_total = current if I_total is None else (I_total + current)
            except Exception as exc:
                raise ValueError(
                    f"Error in computing current for ion channel '{key}': \n"
                    f"{ch}\n"
                    f"Error: {exc}"
                ) from exc
        if I_total is None:
            I_total = 0.0 * (u.nA / (u.cm ** 2))
        return self._midpoint_current(I_total) / self.C

    def compute_axial_derivative(self, V):
        operator = jnp.asarray(self._axial_operator(), dtype=jnp.float64)
        V_decimal = jnp.asarray(V.to_decimal(u.mV), dtype=jnp.float64)
        axial_decimal = -jnp.matmul(V_decimal, operator.T)
        return axial_decimal * (u.mV / u.ms)

    def compute_voltage_derivative(self, V, I_ext=0.0):
        if is_python_zero(I_ext):
            return self.compute_membrane_derivative(V) + self.compute_axial_derivative(V)
        return self.compute_membrane_derivative(V, I_ext) + self.compute_axial_derivative(V)

    def post_integral(self, I_ext=0.0):
        self._ensure_runtime_ready()
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._point_voltage(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.post_integral(point_V)

    def update(self, I_ext=0.0):
        self._ensure_runtime_ready()
        point_V = self._point_voltage(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.update(point_V)

        last_V = self.V.value
        dt = brainstate.environ.get("dt")
        if dt is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")
        # The solver always advances the installed runtime object. The frontend
        # declaration layer should already be frozen at this point. Point-clamp
        # evaluation reads either ``brainstate.environ['t']`` or the cell-owned
        # ``current_time`` fallback, so ``update()`` itself does not need to
        # mutate the global time environment.
        if is_python_zero(I_ext):
            self.solver(self)
        else:
            self.solver(self, I_ext)
        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    @property
    def V_initializer(self) -> object:
        return self._resolve_V_initializer()

    def get_spike(self, last_V, next_V):
        denom = _cast_like(20.0 * u.mV, next_V)
        V_th = _cast_like(self.V_th, next_V)
        return (
            self._spk_fun((next_V - V_th) / denom) *
            self._spk_fun((V_th - last_V) / denom)
        )

    def _rebuild_if_needed(self) -> tuple[CV, ...]:
        if not self._frontend_dirty and self._cvs is not None:
            return self._cvs

        # Rebuild pipeline:
        # 1. split morphology into CV geometry from the active CV policy
        # 2. accumulate rule effects into mutable per-CV mechanism buckets
        # 3. freeze those buckets into immutable user-facing ``CV`` objects
        cv_geo, cv_ids_by_branch = build_cv_geo(
            self._morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
        )
        cv_mech = init_cv_mech(len(cv_geo))

        apply_paint_rules(
            self._morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            paint_rules=self._paint_rules,
            mechs=cv_mech,
        )
        apply_place_rules(
            self._morpho,
            cvs=cv_geo,
            cv_ids_by_branch=cv_ids_by_branch,
            place_rules=self._place_rules,
            mechs=cv_mech,
        )

        self._cvs = tuple(
            assemble_cv(cv_geo=piece, mech=cv_mech[piece.id])
            for piece in cv_geo
        )
        # Any frontend rebuild invalidates all derived execution views and the
        # compiled runtime bridge, because point ids/layout ids may have changed.
        self._frontend_dirty = False
        self._point_tree = None
        self._point_scheduling = {}
        self._compiled_runtime = None
        self._cached_voltage_linearizer = None
        self._cached_axial_operator = None
        self._state_initialized = False
        return self._cvs

    def _ensure_runtime_compiled(self) -> CellRuntimeState:
        self._rebuild_if_needed()
        cached = self._compiled_runtime
        if cached is not None and not self._structure_dirty and not self._mechanism_dirty:
            return cached

        # Runtime compilation lowers immutable CV declarations into layout/state
        # buffers plus installed runtime nodes. This is the bridge from frontend
        # declarations to simulation-ready objects.
        compiled = CellRuntimeState.from_cell(self)
        self._compiled_runtime = compiled
        install_cell_runtime(self, compiled)
        self._structure_dirty = False
        self._mechanism_dirty = False
        self._value_dirty = False
        self._state_initialized = False
        return compiled

    def _ensure_runtime_ready(self) -> None:
        if self._compiled_runtime is None or self._structure_dirty or self._mechanism_dirty:
            raise ValueError("Cell runtime is stale; call Cell.init_state() after changing cv_policy, paint, or place.")
        if not self._state_initialized or not hasattr(self, "V") or not hasattr(self, "spike"):
            raise ValueError("Cell state is not initialized; call Cell.init_state() first.")

    def _mark_dirty(self, *, structure: bool = False, mechanism: bool = False) -> None:
        # Dirty flags separate "what changed" so callers can reason about why a
        # later ``init_state`` or runtime compile is required.
        # - structure: CV splitting / point tree / axial operator may change
        # - mechanism: layouts / state buffers / runtime nodes may change
        if structure:
            self._structure_dirty = True
        if mechanism:
            self._mechanism_dirty = True
        self._frontend_dirty = True
        self._value_dirty = False
        self._compiled_runtime = None
        self._cached_voltage_linearizer = None
        self._cached_axial_operator = None
        self._state_initialized = False

    def _resolve_V_initializer(self) -> object:
        if self._V_initializer_spec is not None:
            return self._V_initializer_spec
        runtime = self._compiled_runtime
        if runtime is None:
            self._ensure_runtime_compiled()
            runtime = self._compiled_runtime
        if runtime is None:
            raise ValueError("Cell runtime has not been compiled; cannot resolve V initializer.")
        return cv_value_vector(self, attr_name="v")

    def _point_voltage(self, cv_voltage: object) -> object:
        runtime = self._ensure_runtime_compiled()
        point_ids = runtime.point_tree.cv_midpoint_point_id
        # Runtime channels and clamps operate on point-space arrays, while the
        # public membrane state is CV-shaped. Midpoint scattering bridges them.
        return scatter_midpoint_values(
            values=cv_voltage,
            point_ids=point_ids,
            n_point=runtime.n_point,
        )

    def _midpoint_current(self, point_values: object) -> object:
        runtime = self._ensure_runtime_compiled()
        point_ids = runtime.point_tree.cv_midpoint_point_id
        return gather_midpoint_values(point_values, point_ids=point_ids)

    def _point_external_input(self, value: object) -> object:
        runtime = self._ensure_runtime_compiled()
        value = self._normalize_external_current(value)
        if matches_last_dim(value, runtime.n_cv):
            # CV-shaped inputs are only defined on CV midpoints, so they need the
            # same midpoint scatter that voltage uses.
            return scatter_midpoint_values(
                values=value,
                point_ids=runtime.point_tree.cv_midpoint_point_id,
                n_point=runtime.n_point,
            )
        return value

    def _normalize_external_current(self, value: object) -> object:
        runtime = self._ensure_runtime_compiled()
        if not isinstance(value, u.Quantity):
            return value

        current_density_unit = 1.0 * u.nA / (u.cm ** 2)
        total_current_unit = 1.0 * u.nA
        if value.has_same_unit(current_density_unit):
            return value.in_unit(u.nA / (u.cm ** 2))
        if value.has_same_unit(total_current_unit):
            area = cv_value_vector(self, attr_name="area")
            # Total current inputs are normalized into current density because the
            # channel solver sums everything in ``nA / cm^2`` at point space.
            if matches_last_dim(value, runtime.n_point):
                point_area = self._point_area()
                return (value / point_area).in_unit(u.nA / (u.cm ** 2))
            if matches_last_dim(value, runtime.n_cv):
                return (value / area).in_unit(u.nA / (u.cm ** 2))
            if getattr(value, "shape", ()) == ():
                return (value / area).in_unit(u.nA / (u.cm ** 2))
        return value

    def _point_area(self) -> object:
        runtime = self._ensure_runtime_compiled()
        area = cv_value_vector(self, attr_name="area")
        return scatter_midpoint_values(
            values=area,
            point_ids=runtime.point_tree.cv_midpoint_point_id,
            n_point=runtime.n_point,
        )

    def _point_area_decimal(self, *, unit) -> np.ndarray:
        runtime = self._ensure_runtime_compiled()
        point_area = np.zeros((runtime.n_point,), dtype=float)
        for cv in self.cvs:
            point_id = int(runtime.point_tree.cv_midpoint_point_id[cv.id])
            point_area[point_id] = float(np.asarray(cv.area.to_decimal(unit), dtype=float))
        return point_area

    def _mech_table(self) -> MechanismObjectTable:
        runtime = self._ensure_runtime_compiled()
        point_tree = self.point_tree()
        column_ids = tuple(range(len(point_tree.points)))

        row_keys: list[tuple[str, str]] = []
        row_labels: list[str] = []
        row_index_by_key: dict[tuple[str, str], int] = {}
        pending_cells: list[tuple[int, int, MechanismObjectCell]] = []
        layout_id_by_signature = {
            (layout.target,) + mechanism_signature(runtime.get_layout_mechanism(layout.id)): layout.id
            for layout in runtime.layouts
        }

        def ensure_row(mechanism: object) -> int:
            row_key = mechanism_cell_key(mechanism)
            row_index = row_index_by_key.get(row_key)
            if row_index is not None:
                return row_index
            row_index = len(row_keys)
            row_keys.append(row_key)
            class_name, instance_name = row_key
            row_labels.append(class_name if class_name == instance_name else f"{instance_name}:{class_name}")
            row_index_by_key[row_key] = row_index
            return row_index

        for cv in self.cvs:
            midpoint_point_id = int(point_tree.cv_midpoint_point_id[cv.id])
            for mechanism in cv.density_mech:
                row_key = mechanism_cell_key(mechanism)
                row_index = ensure_row(mechanism)
                layout_id = layout_id_by_signature[("density",) + mechanism_signature(mechanism)]
                column_id = midpoint_point_id
                pending_cells.append(
                    (
                        row_index,
                        int(column_id),
                        MechanismObjectCell(
                            runtime=runtime,
                            layout_id=int(layout_id),
                            class_name=row_key[0],
                            instance_name=row_key[1],
                            column_id=int(column_id),
                            domain="point",
                            cv_id=None,
                            point_id=midpoint_point_id,
                        ),
                    )
                )
            for mechanism in cv.point_mech:
                row_key = mechanism_cell_key(mechanism)
                row_index = ensure_row(mechanism)
                layout_id = layout_id_by_signature[("point",) + mechanism_signature(mechanism)]
                column_id = midpoint_point_id
                pending_cells.append(
                    (
                        row_index,
                        int(column_id),
                        MechanismObjectCell(
                            runtime=runtime,
                            layout_id=int(layout_id),
                            class_name=row_key[0],
                            instance_name=row_key[1],
                            column_id=int(column_id),
                            domain="point",
                            cv_id=None,
                            point_id=midpoint_point_id,
                        ),
                    )
                )

        values = np.full((len(row_keys), len(column_ids)), None, dtype=object)
        for row_index, column_id, cell in pending_cells:
            values[row_index, int(column_id)] = cell

        return MechanismObjectTable(
            domain="point",
            row_keys=tuple(row_keys),
            row_labels=tuple(row_labels),
            column_ids=column_ids,
            values=values,
        )

    def _point_clamp_input(self) -> object:
        runtime = self._ensure_runtime_compiled()
        try:
            t = brainstate.environ.get("t")
        except KeyError:
            t = self.current_time
        # Clamp current is evaluated in point space, but only active midpoint
        # points have membrane area. Normalize only those active clamp points
        # instead of dividing the full point vector, which would hit 0-area
        # topology points (root/terminal connectors) and create 0/0 -> NaN.
        point_current = runtime.evaluate_point_clamps(t=t)
        active_point_ids: list[int] = []
        for layout in runtime.layouts:
            if layout.target != "point" or layout.point_index is None:
                continue
            if layout.kind not in {"CurrentClamp", "SineClamp", "FunctionClamp"}:
                continue
            active_point_ids.extend(int(point_id) for point_id in layout.point_index.tolist())

        point_density_decimal = u.math.zeros((runtime.n_point,), dtype=float)
        if len(active_point_ids) == 0:
            return u.Quantity(point_density_decimal, u.nA / (u.cm ** 2))

        unique_active_point_ids = np.asarray(sorted(set(active_point_ids)), dtype=np.int32)
        point_current_decimal = u.math.asarray(point_current.to_decimal(u.nA))
        point_area_decimal = self._point_area_decimal(unit=u.cm ** 2)
        active_area_decimal = point_area_decimal[unique_active_point_ids]
        if np.any(active_area_decimal <= 0.0):
            bad_ids = unique_active_point_ids[active_area_decimal <= 0.0].tolist()
            raise ValueError(
                "Point clamp active points must have positive membrane area, "
                f"got non-positive area at point ids {bad_ids!r}."
            )
        active_density_decimal = point_current_decimal[unique_active_point_ids] / active_area_decimal
        point_density_decimal = point_density_decimal.at[unique_active_point_ids].set(active_density_decimal)
        return u.Quantity(point_density_decimal, u.nA / (u.cm ** 2))

    def _voltage_linearizer(self):
        self._ensure_runtime_compiled()
        cached = self._cached_voltage_linearizer
        if cached is not None:
            return cached
        cached = brainstate.transform.vector_grad(
            self.compute_membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )
        self._cached_voltage_linearizer = cached
        return cached

    def _axial_operator(self):
        self._ensure_runtime_compiled()
        cached = self._cached_axial_operator
        if cached is not None:
            return cached
        # The axial operator depends on the current point-tree lowering and
        # scheduling, so it is cached only after runtime/structure are up to date.
        cached = voltage_solver._build_cv_axial_operator(
            self,
            point_tree=self.point_tree(),
            scheduling=self.point_scheduling(algorithm="dhs"),
        )
        self._cached_axial_operator = cached
        return cached

    def _sample_probe_layout(self, runtime: CellRuntimeState, *, layout: object, declaration: object) -> object:
        point_ids = getattr(layout, "point_index", None)
        if point_ids is None:
            raise ValueError(f"Probe layout {getattr(layout, 'id', None)!r} is missing point_index.")
        samples = []
        for point_id in point_ids.tolist():
            if isinstance(declaration, StateProbe):
                samples.append(self._sample_state_probe_point(runtime, declaration=declaration, point_id=int(point_id)))
            elif isinstance(declaration, MechanismProbe):
                samples.append(
                    self._sample_mechanism_probe_point(
                        runtime,
                        declaration=declaration,
                        point_id=int(point_id),
                    )
                )
            elif isinstance(declaration, CurrentProbe):
                samples.append(
                    self._sample_current_probe_point(
                        runtime,
                        declaration=declaration,
                        point_id=int(point_id),
                    )
                )
            else:  # pragma: no cover
                raise TypeError(f"Unsupported probe declaration type {type(declaration).__name__!s}.")
        return _pack_probe_samples(samples)

    def _sample_state_probe_point(
        self,
        runtime: CellRuntimeState,
        *,
        declaration: StateProbe,
        point_id: int,
    ) -> object:
        if declaration.field != "v":
            raise ValueError(f"Unsupported StateProbe field {declaration.field!r}.")
        cv_id = _midpoint_cv_id(runtime, point_id=point_id)
        return _select_last_axis(self.V.value, cv_id)

    def _sample_mechanism_probe_point(
        self,
        runtime: CellRuntimeState,
        *,
        declaration: MechanismProbe,
        point_id: int,
    ) -> object:
        matched_layouts = []
        for layout in runtime.get_point_layouts(point_id):
            mechanism = runtime.get_layout_mechanism(layout.id)
            if isinstance(mechanism, Density) and mechanism.instance_name == declaration.mechanism:
                matched_layouts.append(layout)
        if len(matched_layouts) > 1:
            raise ValueError(
                f"Probe {_probe_name(declaration)!r} matched multiple mechanisms named "
                f"{declaration.mechanism!r} at point {point_id!r}."
            )
        if len(matched_layouts) == 1:
            node = runtime.get_runtime_node(matched_layouts[0].id)
            raw = _probe_state_attr(node, declaration.field, probe_name=_probe_name(declaration))
            return _select_last_axis(raw.value, point_id)

        try:
            ion = runtime.get_ion(declaration.mechanism)
        except KeyError:
            ion = None
        if ion is not None:
            raw = _probe_state_attr(ion, declaration.field, probe_name=_probe_name(declaration))
            return _select_last_axis(raw.value, point_id)

        raise KeyError(
            f"Probe {_probe_name(declaration)!r} could not find a mechanism or ion named "
            f"{declaration.mechanism!r} at point {point_id!r}."
        )

    def _sample_current_probe_point(
        self,
        runtime: CellRuntimeState,
        *,
        declaration: CurrentProbe,
        point_id: int,
    ) -> object:
        point_V = self._point_voltage(self.V.value)
        if declaration.mechanism is not None:
            matched_layouts = []
            for layout in runtime.get_point_layouts(point_id):
                mechanism = runtime.get_layout_mechanism(layout.id)
                if isinstance(mechanism, Density) and mechanism.instance_name == declaration.mechanism:
                    matched_layouts.append(layout)
            if len(matched_layouts) > 1:
                raise ValueError(
                    f"Probe {_probe_name(declaration)!r} matched multiple mechanisms named "
                    f"{declaration.mechanism!r} at point {point_id!r}."
                )
            if len(matched_layouts) == 0:
                raise KeyError(
                    f"Probe {_probe_name(declaration)!r} could not find a mechanism named "
                    f"{declaration.mechanism!r} at point {point_id!r}."
                )
            layout_id = matched_layouts[0].id
            node = runtime.get_runtime_node(layout_id)
            bound_ion_keys = runtime.bound_ion_keys.get(layout_id, ())
            if len(bound_ion_keys) > 1:
                current = node.current(
                    point_V,
                    *tuple(runtime.get_ion(ion_key).pack_info() for ion_key in bound_ion_keys),
                )
            else:
                ion_info = _probe_current_ion_info(
                    runtime,
                    declaration=declaration,
                    mechanism=node,
                    layout_id=layout_id,
                )
                current = _probe_current_value(node, point_V, ion_info, probe_name=_probe_name(declaration))
            return _select_last_axis(current, point_id)

        if declaration.ion is None:
            raise ValueError(
                f"Probe {_probe_name(declaration)!r} must define 'ion' when 'mechanism' is omitted."
            )
        ion = runtime.get_ion(declaration.ion)
        current = ion.current(point_V, include_external=False)
        return _select_last_axis(current, point_id)


def _resolve_solver(solver: str | Callable) -> tuple[str, Callable]:
    if isinstance(solver, str):
        solver_name = str(solver)
        # The "explicit" → euler mapping is now an alias on the registered
        # euler integrator, so a plain registry lookup is sufficient.
        return solver_name, get_integrator(solver_name)
    if callable(solver):
        solver_name = getattr(solver, "__name__", type(solver).__name__)
        return solver_name, solver
    raise TypeError(f"solver must be str or callable, got {type(solver).__name__!s}.")


def _midpoint_cv_id(runtime: CellRuntimeState, *, point_id: int) -> int:
    matches = np.flatnonzero(runtime.point_tree.cv_midpoint_point_id == int(point_id))
    if len(matches) != 1:
        raise ValueError(
            f"Point {point_id!r} is not a unique CV midpoint; got CV matches {matches.tolist()!r}."
        )
    return int(matches[0])


def _select_last_axis(value: object, index: int) -> object:
    value = value.value if isinstance(value, brainstate.State) else value
    shape = getattr(value, "shape", ())
    if shape in (None, ()):
        return value
    return value[..., int(index)]


def _probe_state_attr(owner: object, field: str, *, probe_name: str) -> brainstate.State:
    if not hasattr(owner, field):
        raise KeyError(f"Probe {probe_name!r} field {field!r} was not found on {type(owner).__name__!s}.")
    raw = getattr(owner, field)
    if not isinstance(raw, brainstate.State):
        raise ValueError(
            f"Probe {probe_name!r} field {field!r} on {type(owner).__name__!s} is not a runtime state."
        )
    return raw


def _probe_current_ion_info(
    runtime: CellRuntimeState,
    *,
    declaration: CurrentProbe,
    mechanism: object,
    layout_id: int | None = None,
) -> object | None:
    if declaration.ion is not None:
        return runtime.get_ion(declaration.ion).pack_info()

    if layout_id is not None:
        owner_key = runtime.current_owner_keys.get(int(layout_id))
        if owner_key is not None:
            return runtime.get_ion(owner_key).pack_info()

    mechanism_name = declaration.mechanism
    if mechanism_name is None:
        return None

    for ion in runtime.ions.values():
        channels = getattr(ion, "channels", None)
        if isinstance(channels, dict) and mechanism_name in channels and channels[mechanism_name] is mechanism:
            return ion.pack_info()
    return None


def _probe_current_value(owner: object, point_V: object, ion_info: object | None, *, probe_name: str) -> object:
    if not hasattr(owner, "current"):
        raise KeyError(f"Probe {probe_name!r} target {type(owner).__name__!s} has no current(...) method.")
    if ion_info is None:
        return owner.current(point_V)
    try:
        return owner.current(point_V, ion_info)
    except TypeError:
        return owner.current(point_V)


def _pack_probe_samples(samples: list[object]) -> object:
    if len(samples) == 0:
        return u.math.asarray([])
    if len(samples) == 1:
        return samples[0]
    first = samples[0]
    if hasattr(first, "unit"):
        unit = first.unit
        decimals = [sample.to_decimal(unit) for sample in samples]
        return u.Quantity(u.math.asarray(decimals), unit)
    return u.math.asarray(samples)


def _probe_name(declaration: StateProbe | MechanismProbe | CurrentProbe) -> str:
    if declaration.name is None:
        raise ValueError(
            f"Probe declaration {type(declaration).__name__!s} has no resolved name."
        )
    return declaration.name


def _validate_time_quantity(value: object, *, name: str) -> None:
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"Cell.run(...) {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimal.shape not in ((), (1,)):
        raise ValueError(f"Cell.run(...) {name} must be scalar, got shape {decimal.shape!r}.")
    if float(decimal.reshape(())) <= 0.0:
        raise ValueError(f"Cell.run(...) {name} must be > 0, got {value!r}.")


def _normalize_run_traces(values: object, *, n_traces: int) -> tuple[object, ...]:
    if n_traces == 1:
        return values if isinstance(values, tuple) else (values,)
    if not isinstance(values, tuple):
        raise TypeError(f"Cell.run(...) expected {n_traces} trace arrays, got {type(values).__name__!s}.")
    if len(values) != n_traces:
        raise ValueError(
            f"Cell.run(...) expected {n_traces} trace arrays, got {len(values)!r}."
        )
    return values
