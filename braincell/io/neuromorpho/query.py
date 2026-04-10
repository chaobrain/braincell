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

"""Typed query builder for NeuroMorpho.Org Solr search."""



from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["NeuroMorphoQuery"]


def _normalize(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _build_or_clause(field_name: str, values: tuple[str, ...]) -> str | None:
    if not values:
        return None
    if len(values) == 1:
        return f"{field_name}:{values[0]}"
    joined = " OR ".join(f"{field_name}:{value}" for value in values)
    return f"({joined})"


@dataclass(frozen=True)
class NeuroMorphoQuery:
    """Typed builder for NeuroMorpho.Org search queries.

    Each non-``raw`` field becomes a Solr clause that is ANDed together.
    Multi-value fields produce ``(field:a OR field:b)`` clauses. Use
    :attr:`raw_q` and :attr:`raw_fq` to append filter strings verbatim
    when you need a Solr feature this builder doesn't expose.

    Parameters
    ----------
    species : str or tuple of str or None
        Filter on the ``species`` field.
    brain_region : str or tuple of str or None
        Filter on the ``brain_region`` field.
    cell_type : str or tuple of str or None
        Filter on the ``cell_type`` field.
    archive : str or tuple of str or None
        Filter on the ``archive`` field.
    original_format : str or tuple of str or None
        Filter on the ``original_format`` field.
    stain : str or tuple of str or None
        Filter on the ``stain`` field.
    age_classification : str or tuple of str or None
        Filter on the ``age_classification`` field.
    gender : str or tuple of str or None
        Filter on the ``gender`` field.
    raw_q : tuple of str
        Extra clauses appended verbatim to the ``q`` parameter.
    raw_fq : tuple of str
        Extra filter strings appended verbatim to ``fq``.

    Examples
    --------

    .. code-block:: python

        >>> q = NeuroMorphoQuery(species="mouse", brain_region="cerebellum")
        >>> q.to_q()
        'species:mouse AND brain_region:cerebellum'
        >>> q.to_fq()
        []

        >>> multi = NeuroMorphoQuery(species=("mouse", "rat"))
        >>> multi.to_q()
        '(species:mouse OR species:rat)'
    """

    species: str | tuple[str, ...] | None = None
    brain_region: str | tuple[str, ...] | None = None
    cell_type: str | tuple[str, ...] | None = None
    archive: str | tuple[str, ...] | None = None
    original_format: str | tuple[str, ...] | None = None
    stain: str | tuple[str, ...] | None = None
    age_classification: str | tuple[str, ...] | None = None
    gender: str | tuple[str, ...] | None = None
    raw_q: tuple[str, ...] = field(default_factory=tuple)
    raw_fq: tuple[str, ...] = field(default_factory=tuple)

    def to_q(self) -> str:
        """Return the assembled ``q`` parameter for ``/api/neuron/select``.

        Returns
        -------
        str
            ``"*:*"`` when no filters were specified, otherwise an
            ``AND``-joined string of clauses.
        """

        clauses: list[str] = []
        for attr in (
            "species",
            "brain_region",
            "cell_type",
            "archive",
            "original_format",
            "stain",
            "age_classification",
            "gender",
        ):
            clause = _build_or_clause(attr, _normalize(getattr(self, attr)))
            if clause is not None:
                clauses.append(clause)
        clauses.extend(item for item in self.raw_q if item)
        if not clauses:
            return "*:*"
        return " AND ".join(clauses)

    def to_fq(self) -> list[str]:
        """Return the ``fq`` filter list for ``/api/neuron/select``.

        Currently only ``raw_fq`` populates this list; the typed fields
        all flow through ``q``. The method exists so callers and the
        CLI can pass through ``fq`` strings without special-casing.

        Returns
        -------
        list of str
        """

        return [item for item in self.raw_fq if item]

    def to_params(self) -> dict[str, object]:
        """Return ``q``/``fq`` packaged as a kwargs dict.

        Returns
        -------
        dict
            ``{"q": ..., "fq": [...]}`` ready to be unpacked into
            :meth:`NeuroMorphoClient.search`.
        """

        return {"q": self.to_q(), "fq": self.to_fq()}
