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


"""Memoization shared by every leaf of one region / locset evaluation."""

from dataclasses import dataclass, field
from typing import Callable

__all__ = ["SelectionCache", "evaluate_cached"]

_MISS = object()


@dataclass
class SelectionCache:
    """Scratch space threaded through a whole selection expression tree.

    Memoization keyed by expression value is the whole of it:
    :meth:`evaluated` lets a composite reuse an operand's mask instead of
    walking the morphology again, so ``(A | B) & (A | C)`` evaluates ``A``
    once rather than twice. Consumers that derive something *from* a mask
    can memoize that too, by giving it its own cache instance rather than
    sharing the keyspace holding the masks -- see ``_RegionCache`` in
    ``braincell._discretization.mechanism``.

    Notes
    -----
    Memoized entries are valid only for the morphology and structural
    revision they were produced from. Handing the same cache to a second
    morphology, or reusing it after :meth:`Morphology.attach`, drops the
    stored masks rather than returning a stale answer.

    This class previously also carried ``tree_distance_to_root``,
    ``euclidean_distance_to_root``, and ``branch_radius_summary``, reserved
    for the region types that still raise ``NotImplementedError``. Nothing
    ever wrote them. Whoever implements one of those types should add the
    field it actually needs rather than inherit three guesses -- note that
    per-node tree distance is already computed by
    :class:`braincell.morph._spatial.MorphologySpatialGeometry`.
    """

    _masks: dict[object, object] = field(default_factory=dict, repr=False, compare=False)
    _morpho: object | None = field(default=None, repr=False, compare=False)
    _revision: int = field(default=-1, repr=False, compare=False)

    def evaluated(self, expr: object, morpho: object, evaluate: Callable[[], object]) -> object:
        """Return ``evaluate()``, reusing an earlier result for ``expr``.

        Parameters
        ----------
        expr : RegionExpr or LocsetExpr or LocsetMask
            The expression being evaluated. Used as the memo key, so the
            result is shared between operands that compare equal -- and,
            just as importantly, *not* shared between two that do not.
        morpho : braincell.Morphology
            Morphology the result belongs to.
        evaluate : callable
            Zero-argument thunk producing the value when there is no hit.

        Returns
        -------
        object
            The freshly computed or previously memoized value -- a mask for
            the sub-expression cache, whatever the thunk returns otherwise.

        Notes
        -----
        Expressions carrying an unhashable payload (a list of branch
        indices, an array of bounds) simply do not memoize; they fall
        through to ``evaluate()`` unchanged.
        """
        revision = getattr(morpho, "_revision", None)
        if self._morpho is not morpho or self._revision != revision:
            self._masks.clear()
            self._morpho = morpho
            self._revision = revision
        try:
            cached = self._masks.get(expr, _MISS)
        except (TypeError, ValueError):
            return evaluate()
        if cached is not _MISS:
            return cached
        result = evaluate()
        try:
            self._masks[expr] = result
        except (TypeError, ValueError):
            pass
        return result


def evaluate_cached(expr: object, morpho: object, cache: SelectionCache | None) -> object:
    """Evaluate one operand of a composite, memoizing through ``cache``.

    Parameters
    ----------
    expr : RegionExpr or LocsetExpr
        Operand to evaluate.
    morpho : braincell.Morphology
        Morphology to evaluate against.
    cache : SelectionCache or None
        Shared cache. ``None`` disables memoization; the operand is still
        handed the same ``None`` so leaves cannot silently create their own.

    Returns
    -------
    RegionMask or LocsetMask
        The operand's mask.
    """
    if cache is None:
        return expr.evaluate(morpho, cache)
    return cache.evaluated(expr, morpho, lambda: expr.evaluate(morpho, cache))
