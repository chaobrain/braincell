# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Voltage solver for the active point-tree runtime.

Type responsibilities in this file:

- ``u.Quantity`` is retained in the numerical hot path when the value carries
  physical meaning, such as membrane voltage or ``dt * conductance`` factors.
- ``np.ndarray`` is used for static topology metadata and static float64 source
  coefficients that are assembled once from the point tree.
- ``jnp.ndarray`` mantissas are produced through ``brainstate.environ`` so the
  JAX runtime follows the current precision without hard-coded ``float32`` /
  ``float64`` annotations.
"""

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import set_module_as
from ._exp_euler import ind_exp_euler_step
from ._registry import register_integrator
from .protocol import DiffEqModule

__all__ = [
    'staggered_step',
]


def _is_traced_value(value) -> bool:
    if isinstance(value, u.Quantity):
        value = u.get_mantissa(value)
    return isinstance(value, jax.core.Tracer)


@register_integrator(
    "staggered",
    aliases=("stagger",),
    category="staggered",
    description="Staggered voltage / ion-channel splitting using DHS + ind_exp_euler.",
)
@set_module_as('braincell.quad')
def staggered_step(
    target: DiffEqModule,
    *args
):
    r"""Advance a multi-compartment cell by one *staggered* time step.

    The staggered (operator-splitting) scheme separates the membrane voltage
    update from the ion-channel gating-variable update so that each
    sub-system can use the integrator best suited to it. Within a single
    time step ``dt``:

    1. The cable voltage is advanced with an implicit Euler step solved on
       the point-tree by :func:`dhs_voltage_step` (the dendritic hierarchical
       solver, DHS). This is unconditionally stable for the linear axial
       block and lets ``dt`` exceed the explicit-stability limit.
    2. All remaining differential states (typically Hodgkin-Huxley gating
       variables and ion concentrations) are then advanced by
       :func:`ind_exp_euler_step`, with the voltage path ``('V',)`` excluded
       so the new midpoint voltage from step 1 is not overwritten.

    Splitting the cable problem from the channel problem is the same trick
    used by NEURON and many other compartmental simulators: it preserves
    second-order accuracy when the channel kinetics are smooth, while
    keeping the linear cable solve cheap.

    Parameters
    ----------
    target : DiffEqModule
        A multi-compartment cell exposing ``point_tree``,
        ``point_scheduling``, and a voltage state ``V``. In practice this is
        a :class:`braincell.Cell` instance whose membrane state is laid
        out on a point tree compatible with the DHS scheduler.
    *args
        Forwarded verbatim to :meth:`DiffEqModule.compute_derivative` and
        the underlying voltage and channel solvers (typically the input
        currents being injected this step).

    Returns
    -------
    None
        The state of *target* — voltage, gating variables, and any auxiliary
        ion states — is updated in place.

    Raises
    ------
    AssertionError
        If *target* is not a :class:`DiffEqModule`.
    TypeError
        If *target* does not expose the point-tree machinery required by
        :func:`dhs_voltage_step`.

    See Also
    --------
    dhs_voltage_step : Single implicit-Euler DHS step for the cable voltage.
    ind_exp_euler_step : Independent exponential-Euler update for the
        non-voltage states.

    Notes
    -----
    The staggered scheme is registered as both ``"staggered"`` (canonical)
    and ``"stagger"`` (alias), and can be selected with
    :func:`braincell.quad.get_integrator`.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import staggered_step
        >>> # Inside a simulation loop with a Cell instance ``cell``:
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     staggered_step(cell, input_current)        # doctest: +SKIP
    """
    if not isinstance(target, DiffEqModule):
        raise TypeError(
            f"The stagger integrator only support {DiffEqModule.__name__}, "
            f"but we got {type(target)} instead."
        )
    t = brainstate.environ.get('t', 0.0)
    dt = brainstate.environ.get('dt')

    # voltage integration
    dhs_voltage_step(target, t, dt, *args)

    # ind_exp_euler for ion channels
    ind_exp_euler_step(target, *args, excluded_paths=[('V',)])


@dataclass(frozen=True)
class DHSStaticSource:
    n_point: int
    dynamic_rows_np: np.ndarray
    diag_ms_inv_np: np.ndarray
    lowers_ms_inv_np: np.ndarray
    uppers_ms_inv_np: np.ndarray
    edges_np: np.ndarray
    level_offsets_np: np.ndarray
    backsub_indices_np: np.ndarray


@dataclass(frozen=True)
class DHSStaticCache:
    float_dtype: jnp.dtype
    diag_ms_inv: object
    lowers_ms_inv: object
    uppers_ms_inv: object


@dataclass(frozen=True)
class DHSNumericState:
    diags: object
    solves: object
    lowers: object
    uppers: object


@register_integrator(
    "dhs_voltage",
    category="voltage",
    description="Implicit-Euler dendritic hierarchical solver (DHS) voltage step.",
)
@set_module_as("braincell")
def dhs_voltage_step(target, t, dt, *args):
    r"""Advance the membrane voltage by one implicit-Euler DHS step.

    Solves the linearized cable equation on a multi-compartment cell using
    the **dendritic hierarchical solver** (DHS): the axial coupling matrix is
    cast onto the point-tree representation of the morphology, the membrane
    derivative is linearized around the current voltage, and one implicit
    Euler update of the form

    .. math::

        (I - \Delta t \, J)\, V_{n+1} = V_n + \Delta t \, b

    is solved by a leaf-to-root forward elimination followed by a
    recursive-doubling back substitution. Both phases are pure ``jax.numpy``
    kernels and run inside ``jit``/``vmap`` without dynamic shapes.

    The public cell voltage lives on CV midpoints with shape
    ``[..., n_cv]``. DHS solves the linear system on point-tree rows with
    shape ``[batch, n_point]`` plus one sentinel row used by the recursive
    doubling back-substitution; the result is restored back to the original
    voltage shape on exit.

    Parameters
    ----------
    target : DiffEqModule
        A point-tree aware cell that exposes ``point_tree()``,
        ``point_scheduling("dhs")``, a voltage state ``V``, and a per-CV
        capacitance/area description. In practice this is a
        :class:`braincell.Cell` instance.
    t : Quantity[time]
        Current simulation time. Used by ``compute_membrane_derivative`` and
        any time-dependent input bound through ``args``.
    dt : Quantity[time]
        Numerical time step for the implicit Euler update. Must carry units
        of time (e.g. ``0.025 * u.ms``).
    *args
        Extra arguments forwarded to ``target.compute_membrane_derivative``
        (typically the injected currents).

    Returns
    -------
    None
        ``target.V.value`` is updated in place with the new midpoint
        voltages.

    Raises
    ------
    TypeError
        If *target* does not expose the ``point_tree`` / ``point_scheduling``
        attributes required by the DHS solver.

    See Also
    --------
    staggered_step : Combines this DHS voltage step with an exponential
        Euler update for ion channels.

    Notes
    -----
    The static topology metadata produced by ``_build_dhs_static_source``
    (row lookup tables, edge ordering, recursive-doubling jump table) is
    assembled as NumPy ``float64`` / ``int32`` data and cached on the
    runtime. Per-step numerical operands are then materialized into the
    current JAX precision while keeping physical units on values such as
    voltage and ``dt * conductance`` factors.
    """
    if not hasattr(target, "point_tree") or not hasattr(target, "point_scheduling"):
        raise TypeError(f"dhs_voltage_step(...) requires a point-tree aware target, got {type(target)}.")

    point_tree = target.point_tree()
    scheduling = target.point_scheduling(algorithm="dhs")
    static_source = _get_dhs_static_source(target, point_tree=point_tree, scheduling=scheduling)
    static_cache = _get_dhs_static_cache(target, static_source)
    V_n = target.V.value
    linear, const = _linear_and_const_term(target, V_n, *args)
    numeric = _build_dhs_numeric_state(
        V_n,
        linear,
        const,
        dt=dt,
        static_source=static_source,
        static_cache=static_cache,
    )
    diags, solves = comp_triang_raw(
        numeric.diags,
        numeric.solves,
        numeric.lowers,
        numeric.uppers,
        static_source.edges_np,
        static_source.level_offsets_np,
    )
    solves = comp_backsub_raw(
        diags,
        solves,
        numeric.lowers,
        static_source.backsub_indices_np,
    )
    target.V.value = _restore_midpoint_voltage(
        solves,
        dynamic_rows=static_source.dynamic_rows_np,
        target_shape=target.V.value.shape,
    )


def _build_dhs_static_source(target, *, point_tree, scheduling) -> DHSStaticSource:
    """Build the static NumPy DHS source data from the point tree."""
    n_point, dynamic_rows, axial_matrix = _build_point_tree_axial_matrix(
        target,
        point_tree=point_tree,
        point_id_to_row=scheduling.point_id_to_row,
    )
    diag_ms_inv = np.asarray(np.diag(axial_matrix), dtype=np.float64)
    lowers_ms_inv = np.zeros((n_point,), dtype=np.float64)
    uppers_ms_inv = np.zeros((n_point,), dtype=np.float64)
    for row, parent_row in enumerate(scheduling.parent_rows.tolist()):
        if parent_row < 0:
            continue
        lowers_ms_inv[row] = axial_matrix[row, parent_row]
        uppers_ms_inv[row] = axial_matrix[parent_row, row]

    parent_lookup = np.empty((n_point + 1,), dtype=np.int32)
    spurious_row = n_point
    parent_lookup[:n_point] = np.where(scheduling.parent_rows >= 0, scheduling.parent_rows, spurious_row)
    parent_lookup[spurious_row] = spurious_row
    edges, level_size = _build_dhs_edge_order(scheduling)
    backsub_indices = _build_backsub_indices(parent_lookup, n_nodes=n_point)
    level_offsets_np = np.cumsum(np.insert(level_size, 0, 0)).astype(np.int32, copy=False)
    return DHSStaticSource(
        n_point=n_point,
        dynamic_rows_np=dynamic_rows,
        diag_ms_inv_np=diag_ms_inv,
        lowers_ms_inv_np=lowers_ms_inv,
        uppers_ms_inv_np=uppers_ms_inv,
        edges_np=edges,
        level_offsets_np=level_offsets_np,
        backsub_indices_np=backsub_indices,
    )


def _build_point_tree_axial_matrix(target, *, point_tree, point_id_to_row) -> tuple[int, np.ndarray, np.ndarray]:
    """Assemble the mixed point-tree axial operator in ``ms^-1``."""
    n_point = len(point_tree.points)
    point_id_to_row = np.asarray(point_id_to_row, dtype=np.int32)
    cv_row_by_cv = point_id_to_row[point_tree.cv_midpoint_point_id]
    dynamic_rows = np.asarray([int(cv_row_by_cv[cv_id]) for cv_id in range(len(target.cvs))], dtype=np.int32)
    row_capacitance = _row_capacitance_scale(target, dynamic_rows=dynamic_rows, n_point=n_point)
    axial_matrix = np.zeros((n_point, n_point), dtype=np.float64)

    for edge in point_tree.edges:
        parent_row = int(point_id_to_row[edge.parent_point_id])
        child_row = int(point_id_to_row[edge.child_point_id])
        conductance = _edge_conductance(edge=edge, cvs=target.cvs)

        # Dynamic rows use physical membrane capacitance. Algebraic boundary rows
        # use an arbitrary nonzero scale because the row is only used as a
        # constraint during static reduction.
        parent_coeff = _scalar_decimal(conductance / row_capacitance[parent_row], u.ms ** -1)
        child_coeff = _scalar_decimal(conductance / row_capacitance[child_row], u.ms ** -1)

        axial_matrix[parent_row, parent_row] += parent_coeff
        axial_matrix[parent_row, child_row] -= parent_coeff
        axial_matrix[child_row, child_row] += child_coeff
        axial_matrix[child_row, parent_row] -= child_coeff
    return n_point, dynamic_rows, axial_matrix


def build_cv_axial_operator(target, *, point_tree, scheduling) -> np.ndarray:
    """Reduce the mixed point-tree axial system to a CV-midpoint operator."""
    _, dynamic_rows, axial_matrix = _build_point_tree_axial_matrix(
        target,
        point_tree=point_tree,
        point_id_to_row=scheduling.point_id_to_row,
    )
    dynamic_rows = np.asarray(dynamic_rows, dtype=np.int32)
    algebraic_rows = np.asarray(
        [row for row in range(axial_matrix.shape[0]) if row not in set(dynamic_rows.tolist())],
        dtype=np.int32,
    )
    if algebraic_rows.size == 0:
        reduced = axial_matrix[np.ix_(dynamic_rows, dynamic_rows)]
    else:
        dynamic_dynamic = axial_matrix[np.ix_(dynamic_rows, dynamic_rows)]
        dynamic_algebraic = axial_matrix[np.ix_(dynamic_rows, algebraic_rows)]
        algebraic_dynamic = axial_matrix[np.ix_(algebraic_rows, dynamic_rows)]
        algebraic_algebraic = axial_matrix[np.ix_(algebraic_rows, algebraic_rows)]
        reduced = dynamic_dynamic - dynamic_algebraic @ np.linalg.solve(algebraic_algebraic, algebraic_dynamic)
    return np.asarray(reduced, dtype=np.float64)


def _get_dhs_static_source(target, *, point_tree, scheduling) -> DHSStaticSource:
    runtime = getattr(target, "_runtime", getattr(target, "_compiled_runtime", None))
    source = getattr(runtime, "dhs_static_source_np", None)
    if source is not None:
        return source
    source = _build_dhs_static_source(target, point_tree=point_tree, scheduling=scheduling)
    if runtime is not None:
        object.__setattr__(runtime, "dhs_static_source_np", source)
    return source


def _build_dhs_static_cache(source: DHSStaticSource) -> DHSStaticCache:
    float_dtype = jnp.asarray(0.0).dtype
    return DHSStaticCache(
        float_dtype=float_dtype,
        diag_ms_inv=jnp.asarray(source.diag_ms_inv_np, dtype=brainstate.environ.dftype()) * (u.ms ** -1),
        lowers_ms_inv=jnp.asarray(source.lowers_ms_inv_np, dtype=brainstate.environ.dftype()) * (u.ms ** -1),
        uppers_ms_inv=jnp.asarray(source.uppers_ms_inv_np, dtype=brainstate.environ.dftype()) * (u.ms ** -1),
    )


def _get_dhs_static_cache(target, source: DHSStaticSource) -> DHSStaticCache:
    runtime = getattr(target, "_runtime", getattr(target, "_compiled_runtime", None))
    cache = getattr(runtime, "dhs_static_cache", None)
    float_dtype = jnp.asarray(0.0).dtype
    if cache is not None and getattr(cache, "float_dtype", None) == float_dtype:
        return cache
    cache = _build_dhs_static_cache(source)
    if runtime is not None and not _is_traced_value(cache.diag_ms_inv):
        object.__setattr__(runtime, "dhs_static_cache", cache)
    return cache


def _build_dhs_numeric_state(V_n, linear, const, *, dt, static_source: DHSStaticSource,
                             static_cache: DHSStaticCache) -> DHSNumericState:
    V_n, linear, const = [x.reshape((-1, V_n.shape[-1])) for x in (V_n, linear, const)]
    batch_size = V_n.shape[0]
    n_point = static_source.n_point

    rhs_midpoint_mv = _to_jax_quantity(V_n + dt * const, u.mV)
    linear_ms_inv = _to_jax_quantity(linear, u.ms ** -1)
    dt_ms = _to_jax_quantity(dt, u.ms)

    diag_base = static_cache.diag_ms_inv * dt_ms
    lower_base = static_cache.lowers_ms_inv * dt_ms
    upper_base = static_cache.uppers_ms_inv * dt_ms
    diag_base_mantissa = u.get_mantissa(diag_base)
    diag_base_with_sentinel = jnp.concatenate([diag_base_mantissa, jnp.ones_like(diag_base_mantissa[:1])], axis=0) * u.UNITLESS

    diags = u.math.broadcast_to(diag_base_with_sentinel[None, :], (batch_size, n_point + 1))
    diag_update = jnp.ones_like(u.get_mantissa(linear_ms_inv)) * u.UNITLESS - dt_ms * linear_ms_inv
    diags = diags.at[:, static_source.dynamic_rows_np].add(diag_update)

    solves = u.Quantity(jnp.zeros((batch_size, n_point + 1), dtype=rhs_midpoint_mv.dtype), u.mV)
    solves = solves.at[:, static_source.dynamic_rows_np].set(rhs_midpoint_mv)

    return DHSNumericState(
        diags=diags,
        solves=solves,
        lowers=jnp.concatenate([u.get_mantissa(lower_base), jnp.zeros_like(u.get_mantissa(lower_base[:1]))], axis=0) * u.UNITLESS,
        uppers=jnp.concatenate([u.get_mantissa(upper_base), jnp.zeros_like(u.get_mantissa(upper_base[:1]))], axis=0) * u.UNITLESS,
    )


def _restore_midpoint_voltage(solves: object, *, dynamic_rows: np.ndarray,
                              target_shape: tuple[int, ...]) -> object:
    return solves[:, dynamic_rows].reshape(target_shape)


def _edge_conductance(*, edge, cvs) -> object:
    """Sum all half-CV conductances attached to one point-tree edge."""
    if len(edge.cv_edges) == 0:
        raise ValueError(f"Point-tree edge {edge.id!r} has no CV edge roles.")

    resistances = []
    branch_ids = set()
    for role in edge.cv_edges:
        cv = cvs[role.cv_id]
        branch_ids.add(int(cv.branch_id))
        resistance = cv.r_axial_prox if role.half == "prox" else cv.r_axial_dist
        resistances.append(resistance)

    # When two adjacent CV halves come from the same branch interior, the two
    # half-segment resistances sit in series between the midpoint voltages.
    # Explicitly reassembled one-CV-per-branch morphologies model that internal
    # boundary with an algebraic point; after elimination the equivalent
    # midpoint conductance is 1 / (R_left + R_right), not 1/R_left + 1/R_right.
    if len(resistances) > 1 and len(branch_ids) == 1:
        total_resistance = None
        for resistance in resistances:
            total_resistance = resistance if total_resistance is None else (total_resistance + resistance)
        if total_resistance is None:  # pragma: no cover
            raise ValueError(f"Point-tree edge {edge.id!r} did not produce a total resistance.")
        return 1.0 / total_resistance

    conductance = None
    for resistance in resistances:
        value = 1.0 / resistance
        conductance = value if conductance is None else (conductance + value)
    return conductance


def _row_capacitance_scale(target, *, dynamic_rows: np.ndarray, n_point: int) -> list[object]:
    """Return per-row membrane capacitance used in static axial assembly.

    Only CV midpoint rows carry membrane capacitance. Boundary rows are algebraic
    rows, so a dummy capacitance is used there because the corresponding axial
    coefficients are never consumed as membrane rows.
    """
    midpoint_capacitances = [cv.area * cv.cm for cv in target.cvs]
    if len(midpoint_capacitances) == 0:
        raise ValueError("Point-tree linear system requires at least one CV.")

    row_capacitance: list[object] = [1.0 * u.uF for _ in range(n_point)]
    for cv_id, row in enumerate(dynamic_rows.tolist()):
        row_capacitance[int(row)] = midpoint_capacitances[cv_id]
    return row_capacitance


def _build_dhs_edge_order(scheduling) -> tuple[np.ndarray, np.ndarray]:
    """Build leaf-to-root DHS elimination groups as static integer arrays."""
    edge_pairs: list[list[int]] = []
    level_size: list[int] = []
    for group in reversed(scheduling.groups):
        level_edges = []
        for row in group.tolist():
            parent_row = int(scheduling.parent_rows[row])
            if parent_row >= 0:
                level_edges.append([int(row), parent_row])
        if level_edges:
            edge_pairs.extend(level_edges)
            level_size.append(len(level_edges))

    if edge_pairs:
        return np.asarray(edge_pairs, dtype=np.int32).reshape((-1, 2)), np.asarray(level_size, dtype=np.int32)
    return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)


def _check_comp_triang(diags, solves, lowers, uppers, edges):
    """Kernel contract check for the quantity-aware DHS forward pass."""
    if isinstance(edges, u.Quantity):
        raise ValueError("edges must be a plain array, not a Quantity")
    if diags.ndim != 2:
        raise ValueError(f"diags must be 2D, got ndim={diags.ndim}")
    if solves.ndim != 2:
        raise ValueError(f"solves must be 2D, got ndim={solves.ndim}")
    if lowers.ndim != 1:
        raise ValueError(f"lowers must be 1D, got ndim={lowers.ndim}")
    if uppers.ndim != 1:
        raise ValueError(f"uppers must be 1D, got ndim={uppers.ndim}")
    if isinstance(diags, u.Quantity) and not u.get_unit(diags).is_unitless:
        raise ValueError(f"diags must be unitless, got unit={u.get_unit(diags)}")
    if isinstance(lowers, u.Quantity) and not u.get_unit(lowers).is_unitless:
        raise ValueError(f"lowers must be unitless, got unit={u.get_unit(lowers)}")
    if isinstance(uppers, u.Quantity) and not u.get_unit(uppers).is_unitless:
        raise ValueError(f"uppers must be unitless, got unit={u.get_unit(uppers)}")
    if lowers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"lowers.shape[0]={lowers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if uppers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"uppers.shape[0]={uppers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (_, 2), got {edges.shape}")


def comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets):
    """DHS forward elimination on quantity-aware JAX inputs."""
    _check_comp_triang(diags, solves, lowers, uppers, edges)
    for i in range(level_offsets.shape[0] - 1):
        children = edges[level_offsets[i]:level_offsets[i + 1], 0]
        parent = edges[level_offsets[i]:level_offsets[i + 1], 1]
        lower_val = lowers[children]
        upper_val = uppers[children]
        child_diag = diags[:, children]
        child_solve = solves[:, children]

        multiplier = upper_val / child_diag
        diags = diags.at[:, parent].add(-lower_val * multiplier)
        solves = solves.at[:, parent].add(-child_solve * multiplier)
    return diags, solves


def _check_comp_backsub(diags, solves, lowers, backsub_indices):
    """Kernel contract check for quantity-aware recursive doubling."""
    if isinstance(backsub_indices, u.Quantity):
        raise ValueError("backsub_indices must be a plain array, not a Quantity")
    if diags.ndim != 2:
        raise ValueError(f"diags must be 2D, got ndim={diags.ndim}")
    if solves.ndim != 2:
        raise ValueError(f"solves must be 2D, got ndim={solves.ndim}")
    if lowers.ndim != 1:
        raise ValueError(f"lowers must be 1D, got ndim={lowers.ndim}")
    if isinstance(diags, u.Quantity) and not u.get_unit(diags).is_unitless:
        raise ValueError(f"diags must be unitless, got unit={u.get_unit(diags)}")
    if isinstance(lowers, u.Quantity) and not u.get_unit(lowers).is_unitless:
        raise ValueError(f"lowers must be unitless, got unit={u.get_unit(lowers)}")
    if diags.shape != solves.shape:
        raise ValueError(
            f"diags.shape={diags.shape} must equal solves.shape={solves.shape}"
        )
    if lowers.shape[0] != diags.shape[1]:
        raise ValueError(
            f"lowers.shape[0]={lowers.shape[0]} must equal diags.shape[1]={diags.shape[1]}"
        )
    if backsub_indices.ndim != 2:
        raise ValueError(f"backsub_indices must be 2D, got ndim={backsub_indices.ndim}")
    if backsub_indices.shape[1] != diags.shape[1]:
        raise ValueError(
            f"backsub_indices.shape[1]={backsub_indices.shape[1]} "
            f"must equal diags.shape[1]={diags.shape[1]}"
        )


def _build_backsub_indices(parent_lookup: np.ndarray, *, n_nodes: int) -> np.ndarray:
    """Precompute recursive-doubling ancestor jumps as static metadata."""
    parent_lookup = np.asarray(parent_lookup, dtype=np.int32)
    indices = []
    old_step = 0
    new_step = 1
    k_step_parent = np.arange(n_nodes + 1, dtype=np.int32)
    while new_step <= max(1, n_nodes):
        for _ in range(new_step - old_step):
            k_step_parent = parent_lookup[k_step_parent]
        old_step = new_step
        new_step = 2 * new_step
        indices.append(k_step_parent)
    return np.asarray(indices, dtype=np.int32)


def comp_backsub_raw(
    diags,
    solves,
    lowers,
    backsub_indices,
):
    """DHS recursive-doubling back substitution on quantity-aware inputs."""
    _check_comp_backsub(diags, solves, lowers, backsub_indices)
    zero = 0.0 * u.UNITLESS if isinstance(lowers, u.Quantity) else 0.0
    lowers = lowers.at[0].set(zero)
    lower_effect = -lowers / diags
    solve_effect = solves / diags

    for i in range(backsub_indices.shape[0]):
        k_step_parent = backsub_indices[i]
        solve_effect = solve_effect + lower_effect * solve_effect[:, k_step_parent]
        lower_effect = lower_effect * lower_effect[:, k_step_parent]

    return solve_effect


def _linear_and_const_term(target, V_n, *args):
    """Linearize membrane dynamics around ``V_n``.

    Returns two boundary quantities with units:

    - ``linear`` in ``ms^-1``
    - ``const`` in voltage/time
    """
    if hasattr(target, "_voltage_linearizer"):
        linearizer = target._voltage_linearizer()
    else:
        linearizer = brainstate.transform.vector_grad(
            target.compute_membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )
    linear, derivative = linearizer(V_n, *args)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    const = derivative - V_n * linear
    return linear, const


def _to_jax_quantity(value: object, unit: object) -> u.Quantity:
    """Convert a quantity-like value while preserving any existing JAX dtype."""
    return u.Quantity(u.math.asarray(value.to_decimal(unit)), unit)


def _scalar_decimal(value: object, unit: object) -> float:
    """Convert a scalar quantity to Python float for static assembly."""
    return float(np.asarray(value.to_decimal(unit), dtype=float))
