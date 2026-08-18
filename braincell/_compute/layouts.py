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

"""Mechanism layout records, clamp routing, and state-buffer allocation.

This module owns the leaf-level lowering vocabulary that
:class:`braincell._compute.state.CellRuntimeState` builds on:

- :class:`MechanismLayout` — the record describing where one mechanism
  declaration landed in point space, and whether it is stored densely
  (every point, with an active mask) or sparsely (active point ids only).
- :func:`choose_layout`, :func:`mechanism_kind`, :func:`mechanism_signature` —
  the grouping rules that decide which declarations merge into one layout.
- :class:`ClampRoutingTable` and :func:`build_clamp_routing_table` — routing of
  point clamps to CV-midpoint nodes (converted to current density through the
  membrane area) versus boundary nodes (consumed as total point current).
- The ``_allocate_*`` / ``_write_state_buffer`` / ``_extract_point_value``
  helpers that turn declared parameters into rectangular
  :class:`brainunit.Quantity` buffers, ragged tuple buffers, or padded
  step-protocol buffers.
- The ``_eval_*`` / ``_evaluate_*`` helpers that evaluate clamp and ``NetStim``
  layouts at a simulation time.

Nothing here holds runtime state of its own; every function takes the data it
needs as an argument. That keeps this module a leaf: it imports no other
``braincell._compute`` module at runtime.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Literal

import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._discretization.base import NodeTree
from braincell.mech import (
    CVContext,
    CurrentProbe,
    CurrentClamp,
    Density,
    FunctionClamp,
    Junction,
    MechanismProbe,
    NetStim,
    Point,
    ProbeMechanism,
    SineClamp,
    StateProbe,
    Synapse as SynapsePlacement,
)
from braincell.mech._params import _to_hashable

if TYPE_CHECKING:
    from .state import CellRuntimeState

__all__ = [
    "CLAMP_KINDS",
    "ClampRoutingTable",
    "MechanismLayout",
    "build_clamp_routing_table",
    "choose_layout",
    "mechanism_kind",
    "mechanism_signature",
]


Target = Literal["density", "point"]
Layout = Literal["dense", "sparse"]


@dataclass(frozen=True)
class MechanismLayout:
    """Internal layout decision for one mechanism instance lowered onto points.

    Layouts are the runtime bridge format: they describe where one declaration
    ended up in point space, whether it is stored densely or sparsely, and which
    state buffers/runtime node belong to it.

    Important fields:

    - ``kind`` identifies the lowered mechanism family, such as a named channel
      or one of the point clamp kinds
    - ``target`` separates density-like layouts from point-only layouts
    - ``layout`` distinguishes dense storage over all points from sparse storage
      over just the active point ids
    - ``point_index`` records which points are active for this layout
    - ``source_cv_ids`` remembers which CV declarations contributed to it

    ``CellRuntimeState`` uses these records to allocate state buffers, answer
    inspection queries, and instantiate runtime nodes with the correct shapes.
    """

    id: int
    kind: str
    target: Target
    layout: Layout
    point_index: np.ndarray | None
    point_mask: np.ndarray | None
    n_active: int
    source_cv_ids: tuple[int, ...]
    source_rule: str | None = None

    @property
    def point_axis_len(self) -> int:
        """Length of the axis this layout's buffers index points along.

        A dense layout stores every point of the cell and is indexed by
        the full mask, while a sparse layout stores only its active points.
        Buffer allocation and buffer interpretation must agree on which,
        so both read it from here rather than repeating the decision.

        Returns
        -------
        int
            ``point_mask`` length when dense, otherwise ``n_active``.
        """
        if self.layout == "dense" and self.point_mask is not None:
            return int(np.asarray(self.point_mask).shape[0])
        return int(self.n_active)


#: Clamp layout kinds that contribute point-space current via
#: :meth:`CellRuntimeState.evaluate_point_clamps`.
CLAMP_KINDS = frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"})


@dataclass(frozen=True)
class ClampRoutingTable:
    """Point-clamp routing metadata for membrane and boundary currents.

    Point clamps can land either on CV midpoint nodes or on boundary nodes.
    Midpoint clamps have a membrane area and are converted from total current
    to current density before entering the membrane equation. Boundary clamps
    do not have membrane area and are consumed as total point currents by the
    node-tree voltage solver.

    Attributes
    ----------
    midpoint_ids : np.ndarray
        Sorted unique midpoint point ids that carry clamp layouts.
    midpoint_area : np.ndarray
        Membrane area in ``cm^2`` for ``midpoint_ids``.
    boundary_ids : np.ndarray
        Sorted unique non-midpoint point ids that carry clamp layouts.
    """

    midpoint_ids: np.ndarray
    midpoint_area: np.ndarray
    boundary_ids: np.ndarray


def build_clamp_routing_table(
    *,
    layouts: "tuple[MechanismLayout, ...]",
    cvs,
    node_tree: "NodeTree",
    n_point: int,
) -> ClampRoutingTable | None:
    """Return point-clamp routing metadata or ``None`` if no clamps exist.

    Parameters
    ----------
    layouts : tuple[MechanismLayout, ...]
        All mechanism layouts from :class:`CellRuntimeState`.
    cvs : Sequence[CV]
        The cell's control volumes — source of per-CV membrane area.
    node_tree : NodeTree
        Carries ``cv_to_mid_node_id`` for CV→node mapping.
    n_point : int
        Number of nodes in ``node_tree``.

    Raises
    ------
    ValueError
        If any active midpoint clamp point has non-positive membrane area
        (would produce NaN in ``I_total / area`` division).
    """
    active: set[int] = set()
    for layout in layouts:
        if layout.target != "point" or layout.point_index is None:
            continue
        if layout.kind not in CLAMP_KINDS:
            continue
        active.update(int(pid) for pid in layout.point_index.tolist())

    if not active:
        return None

    point_area = np.zeros((n_point,), dtype=float)
    midpoint_ids: set[int] = set()
    for cv in cvs:
        pid = int(node_tree.cv_to_mid_node_id[cv.id])
        midpoint_ids.add(pid)
        point_area[pid] = float(np.asarray(cv.area.to_decimal(u.cm**2), dtype=float))

    active_midpoints = sorted(pid for pid in active if pid in midpoint_ids)
    ids = np.asarray(active_midpoints, dtype=np.int32)
    area = point_area[ids]
    if np.any(area <= 0.0):
        bad = ids[area <= 0.0].tolist()
        raise ValueError(
            "Midpoint clamp active points must have positive membrane area; "
            f"got non-positive area at point ids {bad!r}."
        )
    boundary_ids = np.asarray(
        sorted(pid for pid in active if pid not in midpoint_ids),
        dtype=np.int32,
    )
    return ClampRoutingTable(
        midpoint_ids=ids,
        midpoint_area=area.astype(np.float64, copy=False),
        boundary_ids=boundary_ids,
    )


def _source_cv_ids_for_point(node_tree: NodeTree, *, point_id: int) -> tuple[int, ...]:
    """Return CV ids whose local roles collapsed into ``point_id``."""

    point = node_tree.nodes[int(point_id)]
    cv_ids = sorted({int(role.cv_id) for role in point.roles})
    return tuple(cv_ids)


def choose_layout(*, target: Target) -> Layout:
    """Choose the runtime storage layout for a mechanism target.

    Parameters
    ----------
    target : {"point", "density"}
        Mechanism target category. Point mechanisms are stored only at
        explicitly selected points, while density mechanisms use dense
        point-shaped state with an active-point mask.

    Returns
    -------
    {"sparse", "dense"}
        Runtime layout name used by mechanism lowering.

    Raises
    ------
    ValueError
        If ``target`` is not a supported mechanism target.
    """
    if target == "point":
        return "sparse"
    if target == "density":
        return "dense"
    raise ValueError(f"Unsupported target {target!r}.")


def mechanism_kind(mechanism: object) -> str:
    """Return a stable string tag describing the mechanism's type.

    For :class:`Density` mechanisms the tag is
    ``"{category}:{class_name}"``. :class:`Point` mechanisms use their
    class ``__name__`` (``CurrentClamp``, ``SineClamp``,
    ``StateProbe`` / ``MechanismProbe`` / ``CurrentProbe`` / ``ProbeMechanism`` appending
    their selector fields for debuggability.
    """
    if isinstance(mechanism, Density):
        return f"{mechanism.category}:{mechanism.class_name}"
    if isinstance(mechanism, SynapsePlacement):
        return f"synapse:{mechanism.synapse_type}"
    if isinstance(mechanism, StateProbe):
        return f"state_probe:{mechanism.field}:{mechanism.name}"
    if isinstance(mechanism, MechanismProbe):
        return f"mechanism_probe:{mechanism.mechanism}:{mechanism.field}:{mechanism.name}"
    if isinstance(mechanism, CurrentProbe):
        target = mechanism.mechanism if mechanism.mechanism is not None else mechanism.ion
        return f"current_probe:{target}:{mechanism.name}"
    if isinstance(mechanism, ProbeMechanism):
        return f"probe:{mechanism.variable}:{mechanism.target}"
    if isinstance(mechanism, Point):
        return type(mechanism).__name__
    return type(mechanism).__name__


_opaque_warned: set = set()


def _fn_fingerprint(fn) -> tuple:
    """Produce a hashable fingerprint for a callable ``fn``.

    Structurally identical lambdas (same bytecode, consts, varnames,
    and closure-cell contents) yield the same fingerprint, so two
    separately-constructed ``lambda`` objects can merge into one
    :class:`MechanismLayout` when used inside :class:`FunctionClamp`.
    Non-hashable / opaque closure cells fall back to ``id(value)``;
    such lambdas will not dedup with textually identical siblings, so
    a :class:`RuntimeWarning` is emitted once per call-site pointing
    the user at the `hoist to module level` fix.
    """
    code = fn.__code__
    closure_cells: list[object] = []
    opaque_hit = False
    for cell in fn.__closure__ or ():
        v = cell.cell_contents
        if hasattr(v, "to_decimal") and hasattr(v, "unit"):
            closure_cells.append(("quantity", float(v.to_decimal(v.unit)), str(v.unit)))
        elif isinstance(v, (int, float, str, bytes, bool)) or v is None:
            closure_cells.append(("prim", v))
        else:
            closure_cells.append(("id", id(v)))
            opaque_hit = True
    if opaque_hit:
        site = (code.co_filename, code.co_firstlineno)
        if site not in _opaque_warned:
            _opaque_warned.add(site)
            warnings.warn(
                f"FunctionClamp.fn at {site[0]}:{site[1]} has an opaque closure "
                "cell; two textually identical lambdas will dedup as distinct. "
                "Hoist to module level with a named function to recover dedup.",
                RuntimeWarning,
                stacklevel=2,
            )
    return (code.co_code, code.co_consts, code.co_varnames, tuple(closure_cells))


def mechanism_signature(mechanism: object) -> tuple[object, ...]:
    """Return a hashable signature used to group declarations.

    Most supported mechanism types are frozen dataclasses with
    structural equality, so the signature reduces to
    ``(type_name, hashable_field_view)``. :class:`FunctionClamp` is
    special-cased: its ``fn`` field is compared by identity under the
    dataclass-generated ``__eq__``, so we fingerprint the callable by
    bytecode + normalized closure so structurally identical lambdas merge
    into one layout.
    """
    if isinstance(mechanism, FunctionClamp):
        return (
            "FunctionClamp",
            _fn_fingerprint(mechanism.fn),
        )
    return (type(mechanism).__qualname__, _to_hashable(mechanism))


def _mechanism_var_names(mechanism: object) -> tuple[str, ...]:
    """Return the state-buffer variable names for a mechanism.

    For :class:`Density` this is the declared ``params`` keys. For
    synapses and junctions it is the parameter keys (or a single
    default name when empty). For clamps it is the concrete dataclass
    field names. The v1 probe declarations do not allocate their own
    state buffers; they are read through explicit sampling helpers.
    """
    if isinstance(mechanism, Density):
        return tuple(mechanism.params.keys())
    if isinstance(mechanism, SynapsePlacement):
        names = tuple(mechanism.params.keys())
        base = names if names else ("g_max", "E_rev")
        return ("pre_spike",) + tuple(base)
    if isinstance(mechanism, Junction):
        names = tuple(mechanism.params.keys())
        return names if names else ("conductance",)
    if isinstance(mechanism, CurrentClamp):
        return ("delay", "durations", "amplitudes")
    if isinstance(mechanism, NetStim):
        return ("start", "number", "interval", "noise", "weight")
    if isinstance(mechanism, SineClamp):
        return ("amplitude", "frequency", "phase", "offset", "delay", "duration")
    if isinstance(mechanism, FunctionClamp):
        return ("fn",)
    if isinstance(mechanism, (StateProbe, MechanismProbe, CurrentProbe)):
        return ()
    if isinstance(mechanism, ProbeMechanism):
        return (mechanism.variable,)
    if is_dataclass(mechanism):
        return tuple(field.name for field in fields(mechanism))
    return ("value",)


def _mechanism_var_value(mechanism: object, var_name: str) -> object:
    if isinstance(mechanism, SynapsePlacement) and var_name == "pre_spike":
        if "weight" in mechanism.params:
            weight = mechanism.params["weight"]
            if isinstance(weight, u.Quantity):
                return 0.0 * weight.unit
        return 0.0
    if isinstance(mechanism, Density):
        if var_name not in mechanism.params:
            raise KeyError(f"Mechanism has no parameter {var_name!r}.")
        return mechanism.params[var_name]
    if isinstance(mechanism, (SynapsePlacement, Junction)):
        if var_name in mechanism.params:
            return mechanism.params[var_name]
    if hasattr(mechanism, var_name):
        return getattr(mechanism, var_name)
    raise KeyError(f"Mechanism {type(mechanism).__name__} has no attribute {var_name!r}.")


def _allocate_clamp_ragged_buffer(
    *,
    per_point_sequences: list,
    unit,
    pop_size: tuple[int, ...] = (),
    n_active: int | None = None,
) -> tuple:
    """Pack ragged per-point sequences into ``(Quantity 2D, bool mask 2D)``.

    Each row ``i`` is zero-padded up to ``max_steps``; ``mask[i, j]`` is
    ``True`` where the original sequence had a value. :func:`_eval_current_clamp`
    multiplies through the mask so padded slots contribute nothing.
    """
    if not per_point_sequences:
        raise ValueError("Ragged clamp buffer requires at least one sequence.")
    max_steps = max(len(seq) for seq in per_point_sequences)
    if n_active is None:
        n_active = len(per_point_sequences)
    n_pop = int(np.prod(pop_size, dtype=int)) if len(pop_size) > 0 else 1
    if len(per_point_sequences) == n_active:
        per_point_sequences = per_point_sequences * n_pop
    if len(per_point_sequences) != n_pop * n_active:
        raise ValueError(
            "Ragged clamp buffer expected either one sequence per active point "
            "or one sequence per population-active-point combination."
        )
    mantissa = np.zeros((n_pop, n_active, max_steps), dtype=np.float64)
    mask = np.zeros((n_pop, n_active, max_steps), dtype=bool)
    for flat_idx, seq in enumerate(per_point_sequences):
        pop_idx = flat_idx // n_active
        local_idx = flat_idx % n_active
        for j, item in enumerate(seq):
            mantissa[pop_idx, local_idx, j] = float(item.to_decimal(unit))
            mask[pop_idx, local_idx, j] = True
    if len(pop_size) == 0:
        mantissa = mantissa.reshape((n_active, max_steps))
        mask = mask.reshape((n_active, max_steps))
    else:
        mantissa = mantissa.reshape(pop_size + (n_active, max_steps))
        mask = mask.reshape(pop_size + (n_active, max_steps))
    return u.Quantity(mantissa, unit), mask


def _allocate_clamp_sequence_buffer(
    *,
    value: object,
    unit,
    pop_size: tuple[int, ...],
    n_active: int,
) -> tuple[object, np.ndarray]:
    """Pack a clamp value into ``target_shape + (n_step,)`` storage."""
    target_prefix = pop_size + (n_active,)
    decimals = np.asarray(value.to_decimal(unit), dtype=np.float64)

    if decimals.shape == ():
        mantissa = np.broadcast_to(decimals, target_prefix + (1,)).copy()
    elif decimals.shape == target_prefix:
        mantissa = decimals[..., None].copy()
    elif decimals.shape == pop_size and n_active == 1:
        mantissa = decimals[..., None, None].copy()
    elif decimals.shape == (n_active,) and len(pop_size) == 0:
        mantissa = decimals[..., None].copy()
    elif len(decimals.shape) >= 1 and decimals.shape[:-1] == target_prefix:
        mantissa = decimals.copy()
    elif len(pop_size) > 0 and decimals.shape[:-1] == pop_size and n_active == 1:
        mantissa = decimals[..., None, :].copy()
    elif len(pop_size) == 0 and decimals.ndim == 1:
        mantissa = np.broadcast_to(
            decimals[None, :],
            target_prefix + (decimals.shape[-1],),
        ).copy()
    else:
        try:
            mantissa = np.broadcast_to(decimals[..., None], target_prefix + (1,)).copy()
        except ValueError as exc:
            raise ValueError(
                f"CurrentClamp value with shape {decimals.shape!r} cannot be broadcast "
                f"to target shape {target_prefix!r} or target+step shape."
            ) from exc

    max_steps = int(mantissa.shape[-1])
    mask = np.ones(target_prefix + (max_steps,), dtype=bool)
    if len(pop_size) == 0:
        mantissa = mantissa.reshape((n_active, max_steps))
        mask = mask.reshape((n_active, max_steps))
    return u.Quantity(mantissa, unit), mask


def _allocate_current_clamp_buffer(
    *,
    mechanism: CurrentClamp,
    var_name: str,
    pop_size: tuple[int, ...],
    n_active: int,
) -> tuple[object, np.ndarray]:
    target_prefix = pop_size + (n_active,)
    durations_dec = np.asarray(mechanism.durations.to_decimal(u.ms), dtype=np.float64)
    n_step = int(durations_dec.shape[-1]) if durations_dec.shape != () else 1

    value = mechanism.durations if var_name == "durations" else mechanism.amplitudes
    unit = u.ms if var_name == "durations" else u.nA
    decimals = np.asarray(value.to_decimal(unit), dtype=np.float64)

    if var_name == "durations" and durations_dec.shape != ():
        return _pack_clamp_steps(decimals, unit=unit, target_prefix=target_prefix)
    if var_name == "amplitudes" and n_step > 1:
        return _pack_clamp_steps(decimals, unit=unit, target_prefix=target_prefix)
    return _allocate_clamp_sequence_buffer(
        value=value,
        unit=unit,
        pop_size=pop_size,
        n_active=n_active,
    )


def _allocate_current_clamp_delay_buffer(
    *,
    mechanism: CurrentClamp,
    pop_size: tuple[int, ...],
    n_active: int,
) -> object:
    target_prefix = pop_size + (n_active,)
    decimals = np.asarray(mechanism.delay.to_decimal(u.ms), dtype=np.float64)

    if decimals.shape == ():
        mantissa = np.broadcast_to(decimals, target_prefix).copy()
    elif decimals.shape == target_prefix:
        mantissa = decimals.copy()
    elif decimals.shape == pop_size and n_active == 1:
        mantissa = decimals[..., None].copy()
    elif decimals.shape == (n_active,) and len(pop_size) == 0:
        mantissa = decimals.copy()
    else:
        try:
            mantissa = np.broadcast_to(decimals, target_prefix).copy()
        except ValueError as exc:
            raise ValueError(
                f"CurrentClamp.delay with shape {decimals.shape!r} cannot be broadcast "
                f"to target shape {target_prefix!r}."
            ) from exc

    return u.Quantity(mantissa, u.ms)


def _pack_clamp_steps(decimals: np.ndarray, *, unit, target_prefix: tuple[int, ...]) -> tuple[object, np.ndarray]:
    if decimals.shape == ():
        step_values = np.broadcast_to(decimals, target_prefix + (1,)).copy()
    elif decimals.ndim == 1:
        step_values = np.broadcast_to(
            decimals.reshape((1,) * len(target_prefix) + (decimals.shape[0],)),
            target_prefix + (decimals.shape[0],),
        ).copy()
    elif decimals.shape[:-1] == target_prefix:
        step_values = decimals.copy()
    elif len(target_prefix) > 1 and decimals.shape[:-1] == target_prefix[:-1] and target_prefix[-1] == 1:
        step_values = decimals[..., None, :].copy()
    else:
        try:
            step_values = np.broadcast_to(decimals, target_prefix + (decimals.shape[-1],)).copy()
        except ValueError as exc:
            raise ValueError(
                f"CurrentClamp step value with shape {decimals.shape!r} cannot be broadcast "
                f"to target+step shape {target_prefix + (decimals.shape[-1],)!r}."
            ) from exc
    mask = np.ones(step_values.shape, dtype=bool)
    return u.Quantity(step_values, unit), mask


def _is_ragged_param(value: object) -> bool:
    """True for callable / tuple / list param values.

    Ragged params include :class:`CurrentClamp` ``durations`` /
    ``amplitudes`` (tuple of Quantity) and :class:`FunctionClamp`
    ``fn`` (callable). These are stored per-point as a Python tuple
    buffer rather than a rectangular :class:`u.Quantity` array.
    """
    if callable(value):
        return True
    if isinstance(value, (tuple, list)):
        return True
    return False


def _constant_quantity_value(value: object) -> u.Quantity | None:
    """Extract a Quantity from a deterministic constant initializer.

    Parameters
    ----------
    value : object
        Mechanism parameter or runtime ion constructor value.

    Returns
    -------
    Quantity or None
        The wrapped quantity when ``value`` is ``braintools.init.Constant``
        over a :mod:`brainunit` quantity; otherwise ``None``.

    Notes
    -----
    Runtime ion parameters use rectangular Quantity buffers for fast
    broadcast/scatter.  Treating ``Constant(Quantity)`` as an opaque callable
    would allocate a large Python object tuple, which is both slower and
    unnecessary because the initializer is deterministic.
    """
    if isinstance(value, braintools.init.Constant) and isinstance(value.value, u.Quantity):
        return value.value
    return None


def _allocate_spatial_density_buffer(
    *,
    mechanism: Density,
    var_name: str,
    value: object,
    layout: MechanismLayout,
    shape: tuple[int, ...],
    cv_contexts: tuple[CVContext, ...],
    node_tree: NodeTree,
) -> object:
    """Evaluate one callable density parameter at active CV midpoints.

    The callable is evaluated once per source CV during ``init_state()``.
    Results must all be scalar numeric values or mutually compatible scalar
    :class:`brainunit.Quantity` values.
    """
    if layout.target != "density" or layout.layout != "dense":
        raise ValueError(
            "Callable density parameters currently require a dense density "
            f"layout, got target={layout.target!r}, layout={layout.layout!r}."
        )
    if not callable(value):  # pragma: no cover - guarded by caller
        raise TypeError(f"Spatial density parameter {var_name!r} is not callable.")

    evaluated: list[tuple[int, int, object]] = []
    for cv_id in layout.source_cv_ids:
        context = cv_contexts[int(cv_id)]
        point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
        try:
            result = value(context)
        except Exception as exc:
            raise ValueError(
                _spatial_density_error_prefix(mechanism, var_name, context)
                + f" callable raised {type(exc).__name__}: {exc}"
            ) from exc
        evaluated.append((int(cv_id), point_id, result))

    if len(evaluated) == 0:  # pragma: no cover - layouts always have a source CV
        raise ValueError(f"Callable density parameter {var_name!r} has no source CVs to evaluate.")

    first_result = evaluated[0][2]
    quantity_result = isinstance(first_result, u.Quantity)
    unit = first_result.unit if quantity_result else None
    scalar_values: list[tuple[int, float]] = []

    for cv_id, point_id, result in evaluated:
        context = cv_contexts[cv_id]
        if isinstance(result, u.Quantity) != quantity_result:
            expected = "a Quantity" if quantity_result else "a unitless number"
            raise TypeError(
                _spatial_density_error_prefix(mechanism, var_name, context)
                + f" returned {type(result).__name__}; expected {expected} "
                "consistently across all CVs."
            )
        try:
            raw = result.to_decimal(unit) if quantity_result else result
            decimal = np.asarray(raw, dtype=np.float64)
        except Exception as exc:
            expected = f"a Quantity compatible with {unit!s}" if quantity_result else "a numeric scalar"
            raise TypeError(
                _spatial_density_error_prefix(mechanism, var_name, context)
                + f" must return {expected}, got {result!r}."
            ) from exc
        if decimal.shape not in ((), (1,)):
            raise TypeError(
                _spatial_density_error_prefix(mechanism, var_name, context)
                + f" must return a scalar, got shape {decimal.shape!r}."
            )
        scalar_values.append((point_id, float(decimal.reshape(()))))

    mantissa = np.zeros(shape, dtype=np.float64)
    for point_id, scalar in scalar_values:
        mantissa[..., point_id] = scalar
    if quantity_result:
        return u.Quantity(mantissa, unit)
    return mantissa


def _spatial_density_error_prefix(
    mechanism: Density,
    var_name: str,
    context: CVContext,
) -> str:
    return (
        f"Spatial callable for {mechanism.category} {mechanism.class_name!r} "
        f"parameter {var_name!r} at CV {context.cv_id!r} "
        f"(branch {context.branch_id!r}, {context.branch_name!r})"
    )


def _allocate_state_buffer(
    mechanism: object,
    *,
    var_name: str,
    shape: tuple[int, ...],
) -> object:
    """Allocate a state buffer for one mechanism parameter.

    Returns a :class:`u.Quantity` whose mantissa is a :class:`jnp.ndarray`
    when the declared value carries a unit. For ragged sequence / callable
    values (``CurrentClamp.durations`` / ``.amplitudes`` / ``FunctionClamp.fn``)
    the buffer is a Python tuple of length ``shape[0]`` (handled in Task 13).
    Plain numeric values (no unit) become a :class:`jnp.ndarray`.
    """
    value = _mechanism_var_value(mechanism, var_name)
    constant_quantity = _constant_quantity_value(value)
    if constant_quantity is not None:
        value = constant_quantity

    if _is_ragged_param(value):
        n = int(np.prod(shape, dtype=int)) if shape else 1
        return tuple(value for _ in range(n))

    if hasattr(value, "unit") and hasattr(value, "to_decimal"):
        unit = value.unit
        mantissa = np.full(shape, float(value.to_decimal(unit)), dtype=np.float64)
        return u.Quantity(mantissa, unit)

    return np.full(shape, value, dtype=np.float64)


def _write_state_buffer(layout: "MechanismLayout", buffer: object, value: object) -> object:
    """Write ``value`` into ``buffer``; return the possibly-new buffer.

    - Quantity buffer: broadcast scalar Quantity, validate unit and shape.
    - Tuple buffer (ragged): replace whole tuple, or fill every slot with a
      scalar.
    - Plain ``jnp.ndarray`` buffer: broadcast scalar, validate shape.
    """
    constant_quantity = _constant_quantity_value(value)
    if constant_quantity is not None:
        value = constant_quantity

    if isinstance(buffer, u.Quantity):
        target_shape = buffer.mantissa.shape
        target_unit = buffer.unit

        if isinstance(value, u.Quantity):
            mantissa = np.asarray(value.to_decimal(target_unit), dtype=np.float64)
            if mantissa.ndim == 0:
                mantissa = np.broadcast_to(mantissa, target_shape).copy()
            if mantissa.shape != target_shape:
                raise ValueError(f"State assignment shape mismatch: expected {target_shape!r}, got {mantissa.shape!r}.")
            return u.Quantity(mantissa, target_unit)

        if isinstance(value, (list, tuple)):
            if len(target_shape) == 2 and value and isinstance(value[0], (list, tuple)):
                rows = [[float(np.asarray(q.to_decimal(target_unit))) for q in row] for row in value]
                arr = np.asarray(rows, dtype=np.float64)
            else:
                decimals = [float(np.asarray(q.to_decimal(target_unit))) for q in value]
                arr = np.asarray(decimals, dtype=np.float64)
                if (
                    len(target_shape) == 2
                    and arr.ndim == 1
                    and target_shape[0] == 1
                    and arr.shape[0] == target_shape[1]
                ):
                    arr = arr.reshape(target_shape)
            if arr.shape != target_shape:
                raise ValueError(f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}.")
            return u.Quantity(arr, target_unit)

        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, target_shape).copy()
        if arr.shape != target_shape:
            raise ValueError(f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}.")
        return u.Quantity(arr, target_unit)

        raise TypeError(
            f"State buffer for layout {layout.id!r} expects a Quantity or sequence of Quantities, "
            f"got {type(value).__name__!r}."
        )

    if isinstance(buffer, tuple):
        if isinstance(value, (tuple, list)):
            if len(buffer) == 1:
                return (tuple(value),)
            if len(value) != len(buffer):
                raise ValueError(
                    f"State assignment shape mismatch for ragged buffer: expected length {len(buffer)}, got {len(value)}."
                )
            return tuple(value)
        return tuple(value for _ in buffer)

    arr = np.asarray(value, dtype=np.float64)
    target_shape = np.asarray(buffer).shape
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, target_shape).copy()
    if arr.shape != target_shape:
        raise ValueError(f"State assignment shape mismatch: expected {target_shape!r}, got {arr.shape!r}.")
    return arr


def _extract_point_value(layout: MechanismLayout, *, point_id: int, buffer: object) -> object:
    # Any leading axes are homogeneous-population dimensions, so a buffer
    # whose trailing axis is *not* the layout's point axis carries an extra
    # per-point axis (the ragged ``durations`` / ``amplitudes`` step axis).
    point_axis_len = layout.point_axis_len

    def _pick_ragged(index: int) -> object:
        if isinstance(buffer, u.Quantity):
            mantissa = np.asarray(buffer.mantissa)
            if mantissa.ndim >= 2:
                return (
                    tuple(u.Quantity(item, buffer.unit) for item in mantissa[..., index, :].reshape(-1))
                    if mantissa.ndim == 2
                    else u.Quantity(buffer.mantissa[..., index, :], buffer.unit)
                )
            return u.Quantity(buffer.mantissa[..., index], buffer.unit)
        return None

    def _pick(index: int) -> object:
        if isinstance(buffer, u.Quantity):
            mantissa = np.asarray(buffer.mantissa)
            if mantissa.ndim >= 2 and int(mantissa.shape[-1]) != point_axis_len:
                ragged = _pick_ragged(index)
                if ragged is not None:
                    return ragged
            return u.Quantity(buffer.mantissa[..., index], buffer.unit)
        if isinstance(buffer, tuple):
            return buffer[index]
        return buffer[..., index]

    if layout.layout == "dense":
        return _pick(int(point_id))
    if layout.point_index is None:
        raise ValueError(f"Sparse layout {layout.id!r} is missing point_index.")
    matches = np.flatnonzero(layout.point_index == int(point_id))
    if len(matches) == 0:
        raise KeyError(f"Point {point_id!r} is not active in layout {layout.id!r}.")
    return _pick(int(matches[0]))


def _evaluate_clamp_layout(
    runtime: CellRuntimeState, *, layout: MechanismLayout, t, local_indices=None
) -> tuple[object, ...]:
    """Evaluate one sparse clamp layout at time ``t``.

    Parameters
    ----------
    runtime : CellRuntimeState
        Runtime state object.
    layout : MechanismLayout
        Sparse point-layout to evaluate.
    t : Quantity[time]
        Absolute simulation time.
    local_indices : iterable of int or None, optional
        Optional local active-point indices to evaluate. ``None`` evaluates
        every active point in ``layout``.

    Returns
    -------
    tuple
        One current contribution per active point. Each item may carry
        population-shaped leading dimensions.
    """
    if layout.layout != "sparse" or layout.point_index is None:
        raise ValueError(f"Clamp layout {layout.id!r} must be sparse with point_index.")
    out: list[object] = []
    indices = range(layout.n_active) if local_indices is None else local_indices
    for local_index in indices:
        if layout.kind == "CurrentClamp":
            local_t = (
                t
                - _scalar_state_value(
                    runtime,
                    layout_id=layout.id,
                    var_name="delay",
                    local_index=local_index,
                )
            ).in_unit(u.ms)
            out.append(_eval_current_clamp(runtime, layout_id=layout.id, local_index=local_index, local_t=local_t))
            continue
        if layout.kind == "SineClamp":
            local_t = (t - _scalar_state_value(runtime, layout_id=layout.id, var_name="delay")).in_unit(u.ms)
            out.append(_eval_sine_clamp(runtime, layout_id=layout.id, local_index=local_index, local_t=local_t))
            continue
        if layout.kind == "FunctionClamp":
            out.append(_eval_function_clamp(runtime, layout_id=layout.id, local_index=local_index, t=t))
            continue
        raise ValueError(f"Unsupported clamp layout kind {layout.kind!r}.")
    return tuple(out)


def _evaluate_netstim_layout(runtime: CellRuntimeState, *, layout: MechanismLayout, t) -> object:
    """Evaluate one sparse `NetStim` layout at time `t`.

    Returns a point-local `pre_spike` drive with shape `(..., n_active)`.
    """
    if layout.layout != "sparse" or layout.point_index is None:
        raise ValueError(f"NetStim layout {layout.id!r} must be sparse with point_index.")
    start = _scalar_state_value(runtime, layout_id=layout.id, var_name="start")
    interval = _scalar_state_value(runtime, layout_id=layout.id, var_name="interval")
    number = _scalar_state_value(runtime, layout_id=layout.id, var_name="number")
    noise = _scalar_state_value(runtime, layout_id=layout.id, var_name="noise")
    weight = _scalar_state_value(runtime, layout_id=layout.id, var_name="weight")

    if float(np.asarray(noise).reshape(())) != 0.0:
        raise ValueError("NetStim.noise != 0.0 is not supported yet.")

    local_t = (t - start).in_unit(u.ms) if hasattr(start, "in_unit") else (t - u.Quantity(start, u.ms)).in_unit(u.ms)
    local_t_ms = u.math.asarray(local_t.to_decimal(u.ms))
    interval_ms = (
        u.math.asarray(interval.to_decimal(u.ms)) if hasattr(interval, "to_decimal") else u.math.asarray(interval)
    )
    number_arr = u.math.asarray(number)
    weight_arr = u.math.asarray(weight)

    if getattr(local_t_ms, "shape", ()) == ():
        local_t_ms = u.math.broadcast_to(local_t_ms, weight_arr.shape)
    if getattr(interval_ms, "shape", ()) == ():
        interval_ms = u.math.broadcast_to(interval_ms, weight_arr.shape)
    if getattr(number_arr, "shape", ()) == ():
        number_arr = u.math.broadcast_to(number_arr, weight_arr.shape)

    event_index = u.math.round(local_t_ms / interval_ms)
    on_grid = u.math.abs(local_t_ms - (event_index * interval_ms)) <= 1e-9
    fired = (local_t_ms >= 0.0) & on_grid & (event_index >= 0) & (event_index < number_arr)
    return u.math.where(fired, weight_arr, 0.0)


def _scalar_state_value(runtime: CellRuntimeState, *, layout_id: int, var_name: str, local_index: int = 0) -> object:
    buffer = runtime.state_buffers[(int(layout_id), str(var_name))]
    index = int(local_index)
    if isinstance(buffer, u.Quantity):
        return u.Quantity(buffer.mantissa[..., index], buffer.unit)
    if isinstance(buffer, tuple):
        return buffer[index]
    return buffer[..., index]


def _quantity_sequence_to_decimal_vector(values: object, *, unit: object) -> object:
    if hasattr(values, "to_decimal"):
        return u.math.asarray(values.to_decimal(unit))
    decimals = [item.to_decimal(unit) for item in values]
    return u.math.stack([u.math.asarray(item) for item in decimals], axis=-1)


def _eval_current_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    """Evaluate a :class:`CurrentClamp` step protocol at a padded local row.

    Uses the padded ``durations`` / ``amplitudes`` state buffers paired
    with the bool ``_mask_durations`` buffer. Population-leading axes
    are preserved; only the final step axis is reduced.
    """
    durations_q = runtime.state_buffers[(int(layout_id), "durations")]
    amplitudes_q = runtime.state_buffers[(int(layout_id), "amplitudes")]
    mask = runtime.state_buffers[(int(layout_id), "_mask_durations")]

    dur_row = jnp.asarray(durations_q.mantissa[..., int(local_index), :])
    amp_row = jnp.asarray(amplitudes_q.mantissa[..., int(local_index), :])
    mask_row = jnp.asarray(mask[..., int(local_index), :])

    local_t_ms = local_t.to_decimal(u.ms)
    if getattr(local_t_ms, "shape", ()) != ():
        local_t_ms = u.math.expand_dims(local_t_ms, axis=-1)
    ends = jnp.cumsum(dur_row, axis=-1)
    starts = ends - dur_row
    is_active = (local_t_ms >= 0.0) & (local_t_ms >= starts) & (local_t_ms < ends) & mask_row
    current = jnp.sum(jnp.where(is_active, amp_row, 0.0), axis=-1)
    return u.Quantity(current, u.nA)


def _eval_sine_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, local_t) -> object:
    duration = _scalar_state_value(runtime, layout_id=layout_id, var_name="duration", local_index=local_index)
    amplitude_decimal = _scalar_state_value(
        runtime, layout_id=layout_id, var_name="amplitude", local_index=local_index
    ).to_decimal(u.nA)
    offset_decimal = _scalar_state_value(
        runtime, layout_id=layout_id, var_name="offset", local_index=local_index
    ).to_decimal(u.nA)
    frequency = _scalar_state_value(runtime, layout_id=layout_id, var_name="frequency", local_index=local_index)
    phase = u.math.asarray(_scalar_state_value(runtime, layout_id=layout_id, var_name="phase", local_index=local_index))
    local_t_ms = local_t.to_decimal(u.ms)
    active = u.math.logical_and(local_t_ms >= 0.0, local_t < duration)
    angle = 2.0 * np.pi * frequency.to_decimal(u.Hz) * local_t.to_decimal(u.second) + phase
    current_decimal = offset_decimal + (u.math.sin(angle) * amplitude_decimal)
    return u.Quantity(u.math.where(active, current_decimal, 0.0), u.nA)


def _eval_function_clamp(runtime: CellRuntimeState, *, layout_id: int, local_index: int, t) -> object:
    fn = _scalar_state_value(runtime, layout_id=layout_id, var_name="fn", local_index=local_index)
    value = fn(t)
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"FunctionClamp fn must return a current quantity, got {value!r}.")
    shape = getattr(value, "shape", ())
    if shape not in ((), None):
        raise ValueError(f"FunctionClamp fn must return a scalar current, got shape {shape!r}.")
    return value.in_unit(u.nA)
