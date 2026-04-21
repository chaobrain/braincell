"""Single lowering pipeline: :class:`Cell` → :class:`RunnableCell`.

This replaces the legacy cascade of dirty-flag guards
(``_rebuild_if_needed`` / ``_ensure_runtime_compiled`` /
``_ensure_runtime_ready`` / ``install_cell_runtime`` / ``init_state``)
with one linear function. Every call produces a fresh, independent
``RunnableCell``; calling :meth:`Cell.build` twice yields two runnables
with no shared state.
"""

from typing import TYPE_CHECKING

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel
from braincell.compute._point_tree import build_point_tree
from braincell.compute._runtime import (
    CellRuntimeState,
    clone_morpho,
    cv_value_vector,
    install_cell_runtime,
)
from braincell.cv._cv import assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    apply_paint_rules,
    apply_place_rules,
    init_cv_mech,
)
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqState
from .runnable import RunnableCell

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["build"]


def build(cell: "Cell") -> RunnableCell:
    """Lower ``cell`` into a fresh :class:`RunnableCell`."""

    # 1. Clone the morphology so the runnable owns its own topology.
    morpho = clone_morpho(cell.morpho)

    # 2. Compute geometry + apply paint / place rules → immutable CVs.
    cv_geo, cv_ids_by_branch = build_cv_geo(
        morpho,
        policy=cell.cv_policy,
        paint_rules=cell.paint_rules,
    )
    cv_mech = init_cv_mech(len(cv_geo))
    apply_paint_rules(
        morpho,
        cvs=cv_geo,
        cv_ids_by_branch=cv_ids_by_branch,
        paint_rules=cell.paint_rules,
        mechs=cv_mech,
    )
    apply_place_rules(
        morpho,
        cvs=cv_geo,
        cv_ids_by_branch=cv_ids_by_branch,
        place_rules=cell.place_rules,
        mechs=cv_mech,
    )
    cvs = tuple(
        assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
    )

    # 3. Stage the RunnableCell. ``__new__`` + ``_preinit`` gets us a
    # partially-populated instance that CellRuntimeState.from_cell can
    # read ``cvs`` / ``point_tree()`` from.
    rcell = RunnableCell.__new__(RunnableCell)
    rcell._preinit(
        name=cell.name,
        V_th_value=cell.V_th,
        V_initializer_spec=cell.V_init,
        spk_fun=cell.spk_fun,
        solver_name=cell.solver_name,
        solver=cell.solver,
        morpho=morpho,
        cvs=cvs,
    )

    # 4. Build the point tree and attach it so ``from_cell`` can consult
    # ``rcell.point_tree()``.
    point_tree = build_point_tree(morpho, cvs=cvs)
    rcell._point_tree = point_tree

    # 5. Lower CV declarations into a runtime state.
    runtime = CellRuntimeState.from_cell(rcell)
    rcell._attach_runtime(runtime=runtime, point_tree=point_tree)

    # 6. Install runtime nodes + C + V_th onto the runnable.
    install_cell_runtime(rcell, runtime)

    # 7. Allocate V / spike / current_time states.
    v_initializer = (
        cell.V_init if cell.V_init is not None else cv_value_vector(rcell, attr_name="v")
    )
    rcell.V = DiffEqState(braintools.init.param(v_initializer, rcell.varshape))
    rcell.spike = brainstate.ShortTermState(rcell.get_spike(rcell.V.value, rcell.V.value))
    rcell._current_time_state.value = 0.0 * u.ms

    # 8. Seed channel states at point-space voltage.
    point_V = rcell._cv_to_point(rcell.V.value)
    for channel in rcell.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
        channel.init_state(point_V, batch_size=None)

    # 9. Pre-cache the JAX axial operator once.
    rcell._axial_jax = jnp.asarray(
        build_cv_axial_operator(
            rcell,
            point_tree=point_tree,
            scheduling=rcell.point_scheduling(algorithm="dhs"),
        ),
        dtype=jnp.float64,
    )

    return rcell
