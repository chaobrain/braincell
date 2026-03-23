from __future__ import annotations

from numbers import Integral

from ..filter import LocsetExpr, RegionExpr
from ..morpho import Morpho
from .cv import CV, CVPolicy, assemble_cv
from .cv_geo import build_cv_geo
from .cv_mech import (
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


class Cell:
    """Frontend-only cell object built around CV and paint/place declarations."""

    def __init__(self, morpho: Morpho, *, cv_policy: CVPolicy | None = None) -> None:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"Cell expects Morpho, got {type(morpho).__name__!s}.")
        self._morpho = _clone_morpho(morpho)
        self._cv_policy = CVPolicy() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}.")

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()
        self._cvs: tuple[CV, ...] | None = None
        self._dirty = True
        self._rebuild_if_needed()

    @property
    def morpho(self) -> Morpho:
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        if not isinstance(value, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(value).__name__!s}.")
        self._cv_policy = value
        self._dirty = True

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    def paint(self, region: RegionExpr, *mechanisms: object) -> "Cell":
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._dirty = True
        return self

    def place(self, locset: LocsetExpr, *mechanisms: object) -> "Cell":
        self._place_rules = self._place_rules + (
            normalize_place_rule(locset, mechanisms),
        )
        self._dirty = True
        return self

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self) -> tuple[CV, ...]:
        return self._rebuild_if_needed()

    def cv(self, index: int) -> CV:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(f"Cell.cv(...) index must be int, got {index!r}.")
        idx = int(index)
        cvs = self.cvs
        if idx < 0 or idx >= len(cvs):
            raise IndexError(f"CV index {idx!r} is out of range [0, {len(cvs)}).")
        return cvs[idx]

    def summary(self) -> dict[str, object]:
        return {
            "n_cv": self.n_cv,
            "n_paint_rules": len(self._paint_rules),
            "n_place_rules": len(self._place_rules),
        }

    def _rebuild_if_needed(self) -> tuple[CV, ...]:
        if not self._dirty and self._cvs is not None:
            return self._cvs

        cv_geo, cv_ids_by_branch = build_cv_geo(self._morpho, policy=self._cv_policy)
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
        self._dirty = False
        return self._cvs


def _clone_morpho(morpho: Morpho) -> Morpho:
    cloned = Morpho.from_root(morpho.root.branch, name=morpho.root.name)
    for index in range(1, len(morpho.branches)):
        branch = morpho.branch_by_index(index)
        parent = branch.parent
        if parent is None:
            continue
        cloned.attach(
            parent=parent.name,
            child=f"child_{index}",
            branch=branch.branch,
            parent_x=float(branch.parent_x),  # type: ignore[arg-type]
            child_x=float(branch.child_x),  # type: ignore[arg-type]
        )
    return cloned
