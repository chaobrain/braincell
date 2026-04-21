"""``Cell`` — the mutable declaration frontend.

``Cell`` collects morphology, CV policy, and paint/place rules; it
owns no runtime state. Call :meth:`Cell.build` to produce a frozen
:class:`RunnableCell` once declaration is complete. Calling
``build`` twice produces independent runnables — the ``Cell`` itself
remains mutable and safe to re-paint or re-place afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import braintools
import brainunit as u

from braincell.cv._cv import assemble_cv
from braincell.cv._geo import build_cv_geo
from braincell.cv._mech import (
    PaintRule,
    PlaceRule,
    apply_paint_rules,
    apply_place_rules,
    default_paint_rules,
    init_cv_mech,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell.cv._policy import CVPerBranch, CVPolicy
from braincell.filter import LocsetExpr, RegionExpr
from braincell.morph.morphology import Morphology
from braincell.quad import get_integrator

if TYPE_CHECKING:
    from .runnable import RunnableCell

__all__ = ["Cell"]


class Cell:
    """Mutable declaration object. Build a runnable cell via :meth:`build`.

    Parameters
    ----------
    morpho : Morphology
        The morphology tree.
    cv_policy : CVPolicy, optional
        Control-volume splitting policy; defaults to
        :class:`CVPerBranch`.
    V_th : Quantity
        Spike-detection threshold (default ``-75 mV``).
    V_init : Quantity or Callable or None
        Initial voltage. ``None`` means "use per-CV resting potential".
    spk_fun : Callable
        Surrogate-gradient spike function.
    solver : str or Callable
        Integrator name (registry lookup) or a callable step function.
    name : str, optional
        Cell name, propagated to the runnable.
    """

    __module__ = "braincell"

    def __init__(
        self,
        morpho: Morphology,
        *,
        cv_policy: CVPolicy | None = None,
        V_th=-75 * u.mV,
        V_init=None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        name: str | None = None,
    ) -> None:
        if not isinstance(morpho, Morphology):
            raise TypeError(
                f"Cell expects Morphology, got {type(morpho).__name__!s}."
            )

        self._morpho = morpho
        self._cv_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}."
            )

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_init = V_init
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)

        self._cvs_cache: tuple | None = None
        self._cvs_cache_key: object = None

    # ------------------------------------------------------------------
    # Read-only accessors / mutable config

    @property
    def morpho(self) -> Morphology:
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        if not isinstance(value, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(value).__name__!s}."
            )
        self._cv_policy = value
        self._cvs_cache = None

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    @property
    def V_th(self):
        return self._V_th

    @V_th.setter
    def V_th(self, value) -> None:
        self._V_th = value

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._V_init = value

    @property
    def solver(self):
        return self._solver_fn

    @solver.setter
    def solver(self, value) -> None:
        self._solver_name, self._solver_fn = _resolve_solver(value)

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def spk_fun(self):
        return self._spk_fun

    @property
    def name(self) -> str | None:
        return self._name

    # ------------------------------------------------------------------
    # Declaration

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        """Paint mechanisms onto ``region``. Returns ``self`` for chaining."""
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._cvs_cache = None
        return self

    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":
        """Place point mechanisms at ``locset``. Returns ``self`` for chaining."""
        self._place_rules = merge_place_rules(
            self._place_rules,
            (normalize_place_rule(locset, mechanisms),),
        )
        self._cvs_cache = None
        return self

    # ------------------------------------------------------------------
    # Preview — CV tuple without runtime lowering

    @property
    def cvs(self):
        """Return the declared CV tuple. Memoized on declaration state."""
        key = (
            id(self._morpho),
            self._cv_policy,
            self._paint_rules,
            self._place_rules,
        )
        if self._cvs_cache is not None and self._cvs_cache_key == key:
            return self._cvs_cache

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
        cvs = tuple(
            assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo
        )
        self._cvs_cache = cvs
        self._cvs_cache_key = key
        return cvs

    # ------------------------------------------------------------------
    # Terminal — produce a runnable

    def build(self) -> "RunnableCell":
        """Lower this declaration into a frozen :class:`RunnableCell`."""
        from .build import build as _build_pipeline
        return _build_pipeline(self)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Cell(root={self._morpho.root.name!r}, "
            f"n_branches={len(self._morpho.branches)}, "
            f"n_paint_rules={len(self._paint_rules)}, "
            f"n_place_rules={len(self._place_rules)})"
        )


def _resolve_solver(solver):
    if isinstance(solver, str):
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(
        f"solver must be str or callable, got {type(solver).__name__!s}."
    )
