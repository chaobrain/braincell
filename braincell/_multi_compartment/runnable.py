"""``RunnableCell`` — the runtime facade produced by :meth:`Cell.build`.

The object is constructed only through the internal staging API
(``__new__`` + :meth:`_preinit` + :meth:`_attach_runtime`). User code
gets one by calling :meth:`Cell.build`.

All mutable per-step state is held in :mod:`brainstate` ``State``
objects (``V``, ``spike``, ``_current_time_state``). Topology
(``morpho``, ``cvs``, ``runtime``, ``point_tree``) is plain attributes
that never change after ``_attach_runtime`` returns. The object has
no dirty flags.
"""

from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel
from braincell.compute._assignment_table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell.compute._point_tree import build_point_scheduling
from braincell.compute._runtime import (
    CellRuntimeState,
    build_placeholder_ions,
    mechanism_signature,
)
from braincell.compute._runtime import cv_value_vector
from braincell.morph.morphology import Morphology
from braincell.quad.protocol import IndependentIntegration
from . import bridge, currents, probes, run as run_module

__all__ = ["RunnableCell"]


def _cast_like(value, like):
    dtype = jnp.asarray(u.get_magnitude(like)).dtype
    if isinstance(value, u.Quantity):
        unit = u.get_unit(value)
        return jnp.asarray(value.to_decimal(unit), dtype=dtype) * unit
    return jnp.asarray(value, dtype=dtype)


class RunnableCell(HHTypedNeuron):
    """Frozen runtime facade for a multi-compartment cell."""

    __module__ = "braincell"

    # ------------------------------------------------------------------
    # Staging API (internal; only :func:`build.build` calls these)

    def _preinit(
        self,
        *,
        name: str | None,
        V_th_value,
        V_initializer_spec,
        spk_fun: Callable,
        solver_name: str,
        solver: Callable,
        morpho: Morphology,
        cvs,
    ) -> None:
        HHTypedNeuron.__init__(
            self,
            size=(1,),
            name=name,
            **build_placeholder_ions(),
        )
        self._morpho = morpho
        self._cvs = cvs
        self._V_th_value = V_th_value
        self._V_initializer_spec = V_initializer_spec
        self._spk_fun = spk_fun
        self.solver_name = solver_name
        self.solver = solver
        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._point_scheduling_cache: dict[tuple[str, int], object] = {}
        self._axial_jax = None
        self._runtime: CellRuntimeState | None = None
        self._point_tree = None

    def _attach_runtime(self, *, runtime: CellRuntimeState, point_tree) -> None:
        self._runtime = runtime
        self._point_tree = point_tree

    # ------------------------------------------------------------------
    # Topology (read-only)

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cvs(self):
        return self._cvs

    @property
    def runtime(self) -> CellRuntimeState:
        return self._runtime

    @property
    def n_cv(self) -> int:
        return self._runtime.n_cv

    @property
    def n_point(self) -> int:
        return self._runtime.n_point

    @property
    def pop_size(self) -> tuple[int, ...]:
        return ()

    @property
    def varshape(self) -> tuple[int, ...]:
        return (self.n_cv,)

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    def point_tree(self):
        return self._point_tree

    def point_scheduling(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        key = (algorithm, int(max_group_size))
        cached = self._point_scheduling_cache.get(key)
        if cached is not None:
            return cached
        scheduling = build_point_scheduling(
            self._point_tree,
            max_group_size=max_group_size,
            algorithm=algorithm,
        )
        self._point_scheduling_cache[key] = scheduling
        return scheduling

    # ------------------------------------------------------------------
    # Time

    @property
    def current_time(self):
        return self._current_time_state.value

    def _set_current_time(self, value) -> None:
        self._current_time_state.value = value

    # ------------------------------------------------------------------
    # Repr

    def __repr__(self) -> str:
        return (
            f"RunnableCell(root={self._morpho.root.name!r}, "
            f"n_cv={self.n_cv!r}, n_point={self.n_point!r})"
        )

    # ------------------------------------------------------------------
    # Bridging

    def _cv_to_point(self, cv_values):
        return bridge.cv_to_point(cv_values, self._runtime)

    def _point_to_cv(self, point_values):
        return bridge.point_to_cv(point_values, self._runtime)

    # ------------------------------------------------------------------
    # Solver path

    def _resolve_t(self):
        try:
            return brainstate.environ.get("t")
        except KeyError:
            return self.current_time

    def pre_integral(self, I_ext=0.0):
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(point_V)

    def compute_derivative(self, I_ext=0.0):
        self.V.derivative = self.compute_voltage_derivative(self.V.value, I_ext)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.compute_derivative(point_V)

    def compute_membrane_derivative(self, V, I_ext=0.0):
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, I_ext=I_ext, t=t)
        return I_total / self.C

    def compute_axial_derivative(self, V):
        V_decimal = jnp.asarray(V.to_decimal(u.mV), dtype=jnp.float64)
        axial = -jnp.matmul(V_decimal, self._axial_jax.T)
        return axial * (u.mV / u.ms)

    def compute_voltage_derivative(self, V, I_ext=0.0):
        return (
            self.compute_membrane_derivative(V, I_ext)
            + self.compute_axial_derivative(V)
        )

    def post_integral(self, I_ext=0.0):
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.post_integral(point_V)

    def update(self, I_ext=None):
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.update(point_V)

        last_V = self.V.value
        if brainstate.environ.get("dt", None) is None:
            raise ValueError("RunnableCell.update(...) requires brainstate.environ['dt'] to be set.")

        if I_ext is None:
            self.solver(self)
        else:
            self.solver(self, I_ext)

        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    # ------------------------------------------------------------------
    # Spike / reset

    def get_spike(self, last_V, next_V):
        denom = _cast_like(20.0 * u.mV, next_V)
        V_th = _cast_like(self.V_th, next_V)
        return (
            self._spk_fun((next_V - V_th) / denom)
            * self._spk_fun((V_th - last_V) / denom)
        )

    def reset_state(self, batch_size=None) -> None:
        v_init = self._V_initializer_spec
        if v_init is None:
            v_init = cv_value_vector(self, attr_name="v")
        self.V.value = braintools.init.param(v_init, self.varshape, batch_size)
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time_state.value = 0.0 * u.ms
        point_V = self._cv_to_point(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.reset_state(point_V, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Inspection forwards

    @property
    def layouts(self):
        return self._runtime.layouts

    @property
    def voltage_shape(self):
        return self._runtime.voltage_shape

    def get_point_layouts(self, point_id):
        return self._runtime.get_point_layouts(point_id)

    def get_cv_layouts(self, cv_id):
        return self._runtime.get_cv_layouts(cv_id)

    def expected_state_shape(self, layout_id, var_name):
        return self._runtime.expected_state_shape(layout_id, var_name)

    def get_state(self, layout_id, var_name):
        return self._runtime.get_state(layout_id, var_name)

    def set_state(self, layout_id, var_name, value) -> None:
        self._runtime.set_state(layout_id, var_name, value)

    def get_point_state(self, point_id):
        return self._runtime.get_point_state(point_id)

    def get_cv_state(self, cv_id):
        return self._runtime.get_cv_state(cv_id)

    def get_runtime_node(self, layout_id):
        return self._runtime.get_runtime_node(layout_id)

    def get_ion(self, name):
        return self._runtime.get_ion(name)

    # ------------------------------------------------------------------
    # Probes + mech_table

    def sample_probe(self, name: str):
        return probes.sample_probe(self, name)

    def sample_probes(self) -> dict[str, object]:
        return probes.sample_probes(self)

    def mech_table(self) -> MechanismObjectTable:
        runtime = self._runtime
        point_tree = self._point_tree
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
            row_labels.append(
                class_name if class_name == instance_name else f"{instance_name}:{class_name}"
            )
            row_index_by_key[row_key] = row_index
            return row_index

        for cv in self.cvs:
            midpoint_point_id = int(point_tree.cv_midpoint_point_id[cv.id])
            for target_label, mech_list in (("density", cv.density_mech), ("point", cv.point_mech)):
                for mechanism in mech_list:
                    row_key = mechanism_cell_key(mechanism)
                    row_index = ensure_row(mechanism)
                    layout_id = layout_id_by_signature[
                        (target_label,) + mechanism_signature(mechanism)
                        ]
                    pending_cells.append(
                        (
                            row_index,
                            midpoint_point_id,
                            MechanismObjectCell(
                                runtime=runtime,
                                layout_id=int(layout_id),
                                class_name=row_key[0],
                                instance_name=row_key[1],
                                column_id=midpoint_point_id,
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

    # ------------------------------------------------------------------
    # Run

    def run(self, *, dt, duration):
        return run_module.run(self, dt=dt, duration=duration)
