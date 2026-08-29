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

"""Voltage solver for the active node-tree runtime.

Type responsibilities in this file:

- ``u.Quantity`` is retained in the numerical hot path when the value carries
  physical meaning, such as membrane voltage or ``dt * conductance`` factors.
- ``np.ndarray`` is used for static topology metadata and static float64 source
  coefficients that are assembled once from the node tree.
- ``jnp.ndarray`` mantissas are produced through ``brainstate.environ`` so the
  JAX runtime follows the current precision without hard-coded ``float32`` /
  ``float64`` annotations.
"""

import functools
import operator
import os
from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import is_traced_value, scalar_decimal as _scalar_decimal, set_module_as
from braincell._typing import DT, T
from ._registry import register_integrator
from ._util import environ_time
from .protocol import DiffEqModule

__all__ = [
    'staggered_step',
    'dhs_voltage_step',
]


def _with_sentinel(base, fill: float):
    """Append the spurious trailing row the DHS back substitution expects.

    Recursive doubling indexes a sentinel row past the end of the tree, so
    every operand carries one extra entry: ``1`` for ``diags`` (so dividing by
    it is a no-op) and ``0`` for ``lowers`` / ``uppers`` (so gathering through
    it contributes nothing). Naming the fill keeps that difference — the only
    thing that varies between the three call sites — visible.
    """
    mantissa = u.get_mantissa(base)
    return jnp.concatenate([mantissa, jnp.full_like(mantissa[:1], fill)], axis=0) * u.UNITLESS


@register_integrator(
    "staggered",
    aliases=("stagger",),
    category="staggered",
    description="Staggered voltage / ion-channel splitting using DHS + ind_exp_euler.",
)
@set_module_as('braincell.quad')
def staggered_step(target: DiffEqModule, *args):
    r"""Advance a multi-compartment cell by one *staggered* time step.

    The staggered (operator-splitting) scheme separates the membrane voltage
    update from the ion-channel gating-variable update so that each
    sub-system can use the integrator best suited to it. Within a single
    time step ``dt``:

    1. When required by an ion model, its total source current is cached at
       the old voltage before any state advances.
    2. The cable voltage is advanced with a locally linearized implicit Euler
       step solved on the node-tree by :func:`dhs_voltage_step` (the
       dendritic hierarchical solver, DHS). The axial block is implicit, so
       it is not subject to the corresponding explicit-stability limit.
    3. Continuous runtime-synapse states and ion/channel states are advanced
       at the new voltage. Dependent states use independent exponential Euler
       updates; submodules marked for independent integration dispatch to
       their configured solvers. The exact family ordering is selected by
       ``target.ion_channel_update_order``.

    This is a first-order Lie/semi-implicit split: the voltage phase reads the
    old mechanism state, and the mechanism phase reads the new voltage. It is
    not a symmetric Strang split, so no general second-order accuracy claim is
    made. The benefit is a cheap, stable implicit solve for the linear axial
    block while retaining specialized updates for channel kinetics.

    Parameters
    ----------
    target : DiffEqModule
        A multi-compartment cell exposing ``node_tree()``,
        ``node_scheduling()``, and a voltage state ``V``. In practice this is
        a :class:`braincell.Cell` instance whose membrane state is laid
        out on a node tree compatible with the DHS scheduler.
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
    TypeError
        If *target* is not a :class:`DiffEqModule`, or does not expose the
        node-tree machinery required by :func:`dhs_voltage_step`.

    See Also
    --------
    dhs_voltage_step : Single implicit-Euler DHS step for the cable voltage.
    ind_exp_euler_step : Independent exponential-Euler kernel used for
        dependent non-voltage states.

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
            f"The stagger integrator only support {DiffEqModule.__name__}, but we got {type(target)} instead."
        )
    t, dt = environ_time(target)

    if hasattr(target, "cache_ion_total_currents"):
        with jax.named_scope("braincell:staggered:cache_ion_total_currents"):
            target.cache_ion_total_currents(target.V.value)

    # voltage integration
    with jax.named_scope("braincell:staggered:dhs_voltage_step"):
        dhs_voltage_step(target, *args, t=t, dt=dt)

    with jax.named_scope("braincell:staggered:cv_to_point_after_voltage"):
        point_V = target._cv_to_point(target.V.value)
    if target.ion_channel_update_order == "family":
        with jax.named_scope("braincell:staggered:synapse_dynamics"):
            target._integrate_runtime_synapse_dynamics(point_V)
        with jax.named_scope("braincell:staggered:ion_channel_update"):
            target._update_ion_channel_families(target.V.value)
    elif target.ion_channel_update_order == "integration":
        with jax.named_scope("braincell:staggered:ion_channel_update"):
            target._update_ion_channels_by_integration(target.V.value)
    else:
        raise ValueError(
            f"ion_channel_update_order must be 'family' or 'integration', got {target.ion_channel_update_order!r}."
        )


@dataclass(frozen=True)
class DHSStaticSource:
    n_point: int
    dynamic_rows_np: np.ndarray
    row_to_point_id_np: np.ndarray
    row_capacitance_uF_np: np.ndarray
    diag_ms_inv_np: np.ndarray
    lowers_ms_inv_np: np.ndarray
    uppers_ms_inv_np: np.ndarray
    edges_np: np.ndarray
    level_offsets_np: np.ndarray
    backsub_indices_np: np.ndarray
    ordinary_backsub_edges_np: np.ndarray
    ordinary_backsub_level_offsets_np: np.ndarray


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
@set_module_as("braincell.quad")
def dhs_voltage_step(target, *args, t: T = None, dt: DT = None):
    r"""Advance the membrane voltage by one implicit-Euler DHS step.

    Solves the linearized cable equation on a multi-compartment cell using
    the **dendritic hierarchical solver** (DHS): the axial coupling matrix is
    cast onto the node-tree representation of the morphology, the membrane
    derivative is linearized around the current voltage, and one implicit
    Euler update of the form

    .. math::

        (I - \Delta t \, J)\, V_{n+1} = V_n + \Delta t \, b

    is solved by a leaf-to-root forward elimination followed by a
    recursive-doubling back substitution. Both phases are pure ``jax.numpy``
    kernels and run inside ``jit``/``vmap`` without dynamic shapes.

    The public cell voltage lives on CV midpoints with shape
    ``[..., n_cv]``. DHS solves the linear system on node-tree rows with
    shape ``[batch, n_point]`` plus one sentinel row used by the recursive
    doubling back-substitution; the result is restored back to the original
    voltage shape on exit.

    Parameters
    ----------
    target : DiffEqModule
        A node-tree aware cell that exposes ``node_tree()``,
        ``node_scheduling("dhs")``, a voltage state ``V``, and a per-CV
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
        If *target* does not expose the ``node_tree`` / ``node_scheduling``
        attributes required by the DHS solver.

    See Also
    --------
    staggered_step : Combines this DHS voltage step with an exponential
        Euler update for ion channels.

    Notes
    -----
    ``t`` and ``dt`` are keyword-only and optional, defaulting to the active
    :mod:`brainstate.environ` context, so this step matches the
    ``(target, *args)`` convention the hosts call and is selectable through
    ``Cell(solver="dhs_voltage")``. :func:`staggered_step` passes both
    explicitly because it has already read them.

    Note that this advances the voltage *only*; selecting it directly leaves
    gating variables and ion concentrations frozen. ``"staggered"`` is the
    solver that pairs it with a channel update.

    The static topology metadata produced by ``_build_dhs_static_source``
    (row lookup tables, edge ordering, recursive-doubling jump table) is
    assembled as NumPy ``float64`` / ``int32`` data and cached on the
    runtime. Per-step numerical operands are then materialized into the
    current JAX precision while keeping physical units on values such as
    voltage and ``dt * conductance`` factors.
    """
    if not hasattr(target, "node_tree") or not hasattr(target, "node_scheduling"):
        raise TypeError(f"dhs_voltage_step(...) requires a node-tree aware target, got {type(target)}.")

    t, dt = environ_time(target, t, dt)
    node_tree = target.node_tree
    scheduling = target.node_scheduling(algorithm="dhs")
    static_source = _get_dhs_static_source(target, node_tree=node_tree, scheduling=scheduling)
    static_cache = _get_dhs_static_cache(target, static_source)
    V_n = target.V.value
    with jax.named_scope("braincell:dhs:linearize_membrane_current"):
        linear, const = _linear_and_const_term(target, V_n, *args)
    with jax.named_scope("braincell:dhs:edge_current"):
        edge_point_current = _edge_point_current(target, t=t)
    with jax.named_scope("braincell:dhs:build_numeric_state"):
        numeric = _build_dhs_numeric_state(
            V_n,
            linear,
            const,
            dt=dt,
            static_source=static_source,
            static_cache=static_cache,
            edge_point_current=edge_point_current,
        )
    with jax.named_scope("braincell:dhs:forward_elimination"):
        diags, solves = comp_triang_raw(
            numeric.diags,
            numeric.solves,
            numeric.lowers,
            numeric.uppers,
            static_source.edges_np,
            static_source.level_offsets_np,
        )
    with jax.named_scope("braincell:dhs:backsubstitution"):
        if _dhs_backsub_mode() == "ordinary":
            solves = comp_backsub_hines_raw(
                diags,
                solves,
                numeric.lowers,
                static_source.ordinary_backsub_edges_np,
                static_source.ordinary_backsub_level_offsets_np,
            )
        else:
            solves = comp_backsub_raw(
                diags,
                solves,
                numeric.lowers,
                static_source.backsub_indices_np,
            )
    with jax.named_scope("braincell:dhs:restore_voltage"):
        target.V.value = _restore_midpoint_voltage(
            solves,
            dynamic_rows=static_source.dynamic_rows_np,
            target_shape=target.V.value.shape,
        )


def _build_dhs_static_source(target, *, node_tree, scheduling) -> DHSStaticSource:
    """Build the static NumPy DHS source data from the node tree."""
    n_point, dynamic_rows, axial_matrix, row_capacitance = _build_node_tree_axial_matrix(
        target,
        node_tree=node_tree,
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
    ordinary_backsub_edges, ordinary_backsub_level_size = _build_dhs_ordinary_backsub_order(scheduling)
    backsub_indices = _build_backsub_indices(parent_lookup, n_nodes=n_point)
    level_offsets_np = np.cumsum(np.insert(level_size, 0, 0)).astype(np.int32, copy=False)
    ordinary_backsub_level_offsets = np.cumsum(np.insert(ordinary_backsub_level_size, 0, 0)).astype(
        np.int32, copy=False
    )
    return DHSStaticSource(
        n_point=n_point,
        dynamic_rows_np=dynamic_rows,
        row_to_point_id_np=np.asarray(scheduling.row_to_point_id, dtype=np.int32),
        row_capacitance_uF_np=row_capacitance,
        diag_ms_inv_np=diag_ms_inv,
        lowers_ms_inv_np=lowers_ms_inv,
        uppers_ms_inv_np=uppers_ms_inv,
        edges_np=edges,
        level_offsets_np=level_offsets_np,
        backsub_indices_np=backsub_indices,
        ordinary_backsub_edges_np=ordinary_backsub_edges,
        ordinary_backsub_level_offsets_np=ordinary_backsub_level_offsets,
    )


def _build_node_tree_axial_matrix(
    target, *, node_tree, point_id_to_row
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the mixed node-tree axial operator in ``ms^-1``."""
    n_point = len(node_tree.nodes)
    point_id_to_row = np.asarray(point_id_to_row, dtype=np.int32)
    cv_row_by_cv = point_id_to_row[node_tree.cv_to_mid_node_id]
    dynamic_rows = np.asarray([int(cv_row_by_cv[cv_id]) for cv_id in range(len(target.cvs))], dtype=np.int32)
    row_capacitance = _row_capacitance_scale(target, dynamic_rows=dynamic_rows, n_point=n_point)
    row_capacitance_uF = np.asarray(
        [_scalar_decimal(value, u.uF) for value in row_capacitance],
        dtype=np.float64,
    )
    axial_matrix = np.zeros((n_point, n_point), dtype=np.float64)

    for edge in node_tree.edges:
        parent_row = int(point_id_to_row[edge.parent_node_id])
        child_row = int(point_id_to_row[edge.child_node_id])
        conductance = _edge_conductance(edge=edge, cvs=target.cvs)

        # Dynamic rows use physical membrane capacitance. Algebraic boundary rows
        # use an arbitrary nonzero scale because the row is only used as a
        # constraint during static reduction.
        parent_coeff = _scalar_decimal(conductance / row_capacitance[parent_row], u.ms**-1)
        child_coeff = _scalar_decimal(conductance / row_capacitance[child_row], u.ms**-1)

        axial_matrix[parent_row, parent_row] += parent_coeff
        axial_matrix[parent_row, child_row] -= parent_coeff
        axial_matrix[child_row, child_row] += child_coeff
        axial_matrix[child_row, parent_row] -= child_coeff
    return n_point, dynamic_rows, axial_matrix, row_capacitance_uF


def build_cv_axial_operator(target, *, node_tree, scheduling) -> np.ndarray:
    """Reduce the mixed node-tree axial system to a CV-midpoint operator."""
    _, dynamic_rows, axial_matrix, _row_capacitance = _build_node_tree_axial_matrix(
        target,
        node_tree=node_tree,
        point_id_to_row=scheduling.point_id_to_row,
    )
    dynamic_rows = np.asarray(dynamic_rows, dtype=np.int32)
    dynamic_row_set = set(dynamic_rows.tolist())
    algebraic_rows = np.asarray(
        [row for row in range(axial_matrix.shape[0]) if row not in dynamic_row_set],
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


def _get_dhs_static_source(target, *, node_tree, scheduling) -> DHSStaticSource:
    runtime = target._runtime
    source = getattr(runtime, "dhs_static_source_np", None)
    if source is not None:
        return source
    source = _build_dhs_static_source(target, node_tree=node_tree, scheduling=scheduling)
    if runtime is not None:
        runtime.dhs_static_source_np = source
    return source


def _build_dhs_static_cache(source: DHSStaticSource) -> DHSStaticCache:
    # ``float_dtype`` must record the dtype the arrays below are actually
    # built with, so that _get_dhs_static_cache can detect a precision
    # change and rebuild instead of handing back a stale-precision cache.
    float_dtype = jnp.dtype(brainstate.environ.dftype())
    return DHSStaticCache(
        float_dtype=float_dtype,
        diag_ms_inv=jnp.asarray(source.diag_ms_inv_np, dtype=float_dtype) * (u.ms**-1),
        lowers_ms_inv=jnp.asarray(source.lowers_ms_inv_np, dtype=float_dtype) * (u.ms**-1),
        uppers_ms_inv=jnp.asarray(source.uppers_ms_inv_np, dtype=float_dtype) * (u.ms**-1),
    )


def _get_dhs_static_cache(target, source: DHSStaticSource) -> DHSStaticCache:
    runtime = target._runtime
    cache = getattr(runtime, "dhs_static_cache", None)
    float_dtype = jnp.dtype(brainstate.environ.dftype())
    if cache is not None and getattr(cache, "float_dtype", None) == float_dtype:
        return cache
    cache = _build_dhs_static_cache(source)
    if runtime is not None and not is_traced_value(cache.diag_ms_inv):
        runtime.dhs_static_cache = cache
    return cache


def _build_dhs_numeric_state(
    V_n, linear, const, *, dt, static_source: DHSStaticSource, static_cache: DHSStaticCache, edge_point_current=None
) -> DHSNumericState:
    """Assemble the numeric DHS solve state for one timestep.

    Parameters
    ----------
    V_n, linear, const : object
        Voltage, linear term, and constant term in CV space. Any
        leading population/batch axes are flattened into one solve batch.
    dt : Quantity[time]
        Timestep.
    static_source : DHSStaticSource
        Static DHS topology metadata.
    static_cache : DHSStaticCache
        Precision-specific cached diagonal/off-diagonal factors.
    edge_point_current : object, optional
        Optional point-space clamp current with shape
        ``(..., n_point)``. Leading axes are flattened alongside
        ``V_n``.

    Returns
    -------
    DHSNumericState
        Numeric buffers ready for forward elimination and back-substitution.
    """
    V_n, linear, const = [x.reshape((-1, V_n.shape[-1])) for x in (V_n, linear, const)]
    batch_size = V_n.shape[0]
    n_point = static_source.n_point

    rhs_midpoint_mv = u.math.asarray(V_n + dt * const, unit=u.mV)
    linear_ms_inv = u.math.asarray(linear, unit=u.ms**-1)
    dt_ms = u.math.asarray(dt, unit=u.ms)

    diag_base = static_cache.diag_ms_inv * dt_ms
    lower_base = static_cache.lowers_ms_inv * dt_ms
    upper_base = static_cache.uppers_ms_inv * dt_ms
    diags = u.math.broadcast_to(_with_sentinel(diag_base, 1.0)[None, :], (batch_size, n_point + 1))
    diag_update = jnp.ones_like(u.get_mantissa(linear_ms_inv)) * u.UNITLESS - dt_ms * linear_ms_inv
    diags = diags.at[:, static_source.dynamic_rows_np].add(diag_update)

    solves = u.Quantity(jnp.zeros((batch_size, n_point + 1), dtype=rhs_midpoint_mv.dtype), u.mV)
    solves = solves.at[:, static_source.dynamic_rows_np].set(rhs_midpoint_mv)
    if edge_point_current is not None:
        edge_rhs = _edge_current_voltage_delta(
            edge_point_current,
            dt=dt,
            static_source=static_source,
        )
        edge_rhs = edge_rhs.reshape((batch_size, n_point))
        solves = solves.at[:, :n_point].add(edge_rhs)

    return DHSNumericState(
        diags=diags,
        solves=solves,
        lowers=_with_sentinel(lower_base, 0.0),
        uppers=_with_sentinel(upper_base, 0.0),
    )


def _edge_point_current(target, *, t):
    """Return boundary point-clamp current for the DHS point-tree RHS."""

    runtime = getattr(target, "_runtime", None)
    if runtime is None or not hasattr(runtime, "evaluate_point_clamps"):
        return None
    table = getattr(runtime, "clamp_routing_table", None)
    if table is None or len(table.boundary_ids) == 0:
        return None
    return runtime.evaluate_point_clamps(t=t, point_ids=table.boundary_ids)


def _edge_current_voltage_delta(edge_point_current, *, dt, static_source: DHSStaticSource):
    current_by_point = u.math.asarray(edge_point_current.to_decimal(u.nA))
    current_by_row = current_by_point[..., static_source.row_to_point_id_np]
    capacitance = (
        jnp.asarray(
            static_source.row_capacitance_uF_np,
            dtype=brainstate.environ.dftype(),
        )
        * u.uF
    )
    rate = (u.Quantity(current_by_row, u.nA) / capacitance).in_unit(u.mV / u.ms)
    return (dt * rate).in_unit(u.mV)


def _restore_midpoint_voltage(solves: object, *, dynamic_rows: np.ndarray, target_shape: tuple[int, ...]) -> object:
    return solves[:, dynamic_rows].reshape(target_shape)


def _edge_conductance(*, edge, cvs) -> object:
    """Sum all half-CV conductances attached to one node-tree edge."""
    if len(edge.roles) == 0:
        raise ValueError(f"Point-tree edge {edge.id!r} has no CV edge roles.")

    resistances = []
    branch_ids = set()
    for role in edge.roles:
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
        return 1.0 / functools.reduce(operator.add, resistances)

    return functools.reduce(operator.add, (1.0 / r for r in resistances))


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


def _build_dhs_ordinary_backsub_order(scheduling) -> tuple[np.ndarray, np.ndarray]:
    """Build root-to-leaf edge groups for work-efficient Hines backsub."""
    edge_pairs: list[list[int]] = []
    level_size: list[int] = []
    for group in scheduling.groups:
        level_edges = []
        for row in group.tolist():
            parent_row = int(scheduling.parent_rows[row])
            if parent_row >= 0:
                level_edges.append([int(row), parent_row])
        edge_pairs.extend(level_edges)
        level_size.append(len(level_edges))
    edges = np.asarray(edge_pairs, dtype=np.int32).reshape((-1, 2)) if edge_pairs else np.empty((0, 2), dtype=np.int32)
    return edges, np.asarray(level_size, dtype=np.int32)


def _require_ndim(name, value, ndim):
    """Raise unless ``value`` has exactly ``ndim`` dimensions."""
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got ndim={value.ndim}")


def _require_unitless(name, value):
    """Raise unless ``value`` is a plain array or a unitless Quantity."""
    if isinstance(value, u.Quantity) and not u.get_unit(value).is_unitless:
        raise ValueError(f"{name} must be unitless, got unit={u.get_unit(value)}")


def _require_plain_array(name, value):
    """Raise if ``value`` is a Quantity where an index array is expected."""
    if isinstance(value, u.Quantity):
        raise ValueError(f"{name} must be a plain array, not a Quantity")


def _require_row_length(name, value, diags):
    """Raise unless ``value`` has one entry per column of ``diags``."""
    if value.shape[0] != diags.shape[1]:
        raise ValueError(f"{name}.shape[0]={value.shape[0]} must equal diags.shape[1]={diags.shape[1]}")


def _check_dhs_operands(diags, solves, lowers):
    """Check the operand contract shared by both DHS kernels."""
    _require_ndim("diags", diags, 2)
    _require_ndim("solves", solves, 2)
    _require_ndim("lowers", lowers, 1)
    _require_unitless("diags", diags)
    _require_unitless("lowers", lowers)
    _require_row_length("lowers", lowers, diags)


def _check_comp_triang(diags, solves, lowers, uppers, edges):
    """Kernel contract check for the quantity-aware DHS forward pass."""
    _require_plain_array("edges", edges)
    _check_dhs_operands(diags, solves, lowers)
    _require_ndim("uppers", uppers, 1)
    _require_unitless("uppers", uppers)
    _require_row_length("uppers", uppers, diags)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (_, 2), got {edges.shape}")


def comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets):
    """DHS forward elimination on quantity-aware JAX inputs."""
    _check_comp_triang(diags, solves, lowers, uppers, edges)
    if _profile_dhs_levels_enabled():
        return _comp_triang_raw_profiled(diags, solves, lowers, uppers, edges, level_offsets)
    for i in range(level_offsets.shape[0] - 1):
        level_edges = edges[level_offsets[i] : level_offsets[i + 1]]
        diags, solves = _comp_triang_level(diags, solves, lowers, uppers, level_edges)
    return diags, solves


def _profile_dhs_levels_enabled() -> bool:
    """Return whether profiler-only DHS level scopes should be emitted."""
    return os.environ.get("BRAINCELL_PROFILE_DHS_LEVELS") == "1"


def _comp_triang_raw_profiled(diags, solves, lowers, uppers, edges, level_offsets):
    """DHS forward elimination with profiler-visible level scopes."""
    batch_size = int(diags.shape[0])
    for i in range(level_offsets.shape[0] - 1):
        start = int(level_offsets[i])
        stop = int(level_offsets[i + 1])
        edge_count = stop - start
        scope = f"braincell:dhs:forward_level:i={i:03d}:edges={edge_count:06d}:batch={batch_size}"
        level_edges = edges[start:stop]
        with jax.named_scope(scope):
            diags, solves = _comp_triang_level(diags, solves, lowers, uppers, level_edges)
    return diags, solves


def _comp_triang_level(diags, solves, lowers, uppers, level_edges):
    """Apply one DHS forward elimination level."""
    children = level_edges[:, 0]
    parent = level_edges[:, 1]
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
    _require_plain_array("backsub_indices", backsub_indices)
    _check_dhs_operands(diags, solves, lowers)
    if diags.shape != solves.shape:
        raise ValueError(f"diags.shape={diags.shape} must equal solves.shape={solves.shape}")
    if backsub_indices.ndim != 2:
        raise ValueError(f"backsub_indices must be 2D, got ndim={backsub_indices.ndim}")
    if backsub_indices.shape[1] != diags.shape[1]:
        raise ValueError(
            f"backsub_indices.shape[1]={backsub_indices.shape[1]} must equal diags.shape[1]={diags.shape[1]}"
        )


def _build_backsub_indices(parent_lookup: np.ndarray, *, n_nodes: int) -> np.ndarray:
    """Precompute recursive-doubling ancestor jumps as static metadata.

    Row ``i`` of the result is the ``2**i``-th ancestor of every row, i.e.
    the map :math:`A_{2^i}` obtained by following ``parent_lookup`` that
    many times. Rather than walking the tree one parent at a time, each
    level is built by composing the previous one with itself
    (:math:`A_{2k}[j] = A_k[A_k[j]]`), which turns the construction from
    :math:`O(n^2)` gathers into :math:`O(n \\log n)`.

    The table is sized by the tree's *depth*, not its node count. Recursive
    doubling only needs enough levels to span the longest ancestor chain, so
    the loop stops at the first level that maps every row to the sentinel:
    from there on ``lowers[sentinel]`` and ``solves[sentinel]`` are both zero,
    making each further round an arithmetic no-op that still costs two gathers
    and two multiplies per step. A deep-but-narrow morphology is unaffected;
    a wide one sheds several rounds. The level count can never exceed the
    ``floor(log2(n_nodes)) + 1`` the node-count bound would have produced.
    """
    parent_lookup = np.asarray(parent_lookup, dtype=np.int32)
    indices = []
    # A_1: one parent hop from every row (fancy-indexed rather than sliced so
    # that a too-short ``parent_lookup`` still raises instead of truncating).
    k_step_parent = parent_lookup[np.arange(n_nodes + 1, dtype=np.int32)]
    max_levels = max(1, int(n_nodes)).bit_length()
    while True:
        indices.append(k_step_parent)
        if len(indices) >= max_levels or bool(np.all(k_step_parent == n_nodes)):
            break
        k_step_parent = k_step_parent[k_step_parent]
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


def comp_backsub_hines_raw(diags, solves, lowers, edges, level_offsets):
    """Hines root-to-leaf back substitution with linear total work."""
    _check_comp_triang(diags, solves, lowers, lowers, edges)
    solution = solves / diags
    for i in range(level_offsets.shape[0] - 1):
        children = edges[level_offsets[i] : level_offsets[i + 1], 0]
        parents = edges[level_offsets[i] : level_offsets[i + 1], 1]
        child_solution = solution[:, children] - (lowers[children] / diags[:, children]) * solution[:, parents]
        solution = solution.at[:, children].set(child_solution)
    return solution


def _dhs_backsub_mode() -> str:
    value = os.environ.get("BRAINCELL_DHS_BACKSUB", "recursive")
    if value not in {"recursive", "ordinary"}:
        raise ValueError(f"BRAINCELL_DHS_BACKSUB must be 'recursive' or 'ordinary', got {value!r}.")
    return value


def _linear_and_const_term(target, V_n, *args):
    """Linearize membrane dynamics around ``V_n``.

    Returns two boundary quantities with units:

    - ``linear`` in ``ms^-1``
    - ``const`` in voltage/time
    """
    if hasattr(target, "_voltage_linearizer"):
        linearizer = target._voltage_linearizer()
    else:
        membrane_derivative = jax.named_call(
            target.compute_membrane_derivative,
            name="braincell_dhs_compute_membrane_derivative",
        )
        linearizer = brainstate.transform.vector_grad(
            membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )
    linear, derivative = linearizer(V_n, *args)
    linear_mantissa = u.get_mantissa(linear)
    linear_unit = u.get_unit(derivative) / u.get_unit(V_n)
    if getattr(linear_mantissa, "dtype", None) == jax.dtypes.float0:
        linear = u.Quantity(
            jnp.zeros_like(u.get_mantissa(derivative)),
            linear_unit,
        )
    else:
        linear = u.Quantity(linear_mantissa, linear_unit)
    const = derivative - V_n * linear
    return linear, const
