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

"""Endpoint pairing rules for direct connections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Callable

import brainstate
import brainunit as u
import numpy as np

from braincell._multi_compartment.synapses import SynapseView
from .event import EventSourceView
from braincell.morph._spatial import MorphologySpatialGeometry, interpolate_branch

Score = Callable[["PairingContext"], object]
Degree = int | np.ndarray | Callable[["PairingContext", brainstate.random.RandomState], object]

__all__ = [
    "PairingContext",
    "PairingSpec",
    "degree",
    "independent",
    "match_degrees",
    "by_source",
    "by_synapse",
    "source_first",
    "synapse_first",
]


@dataclass(frozen=True)
class GroupContext:
    """Describe the current target-cell partition."""

    index: int
    target_population_index: int
    size: int
    n_groups: int


@dataclass(frozen=True)
class PairingContext:
    """Data visible to score and degree callables."""

    source: "SourceContext | None"
    synapse: "SynapseContext | None"
    group: GroupContext | None


class _EndpointContext:
    """Read-only shaped projection over one endpoint column store."""

    __slots__ = ("_data", "_positions", "_shape")

    def __init__(self, data, positions, *, fixed: bool | None) -> None:
        self._data = data
        self._positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        self._shape = (-1,) if fixed is None else ((-1, 1) if fixed else (1, -1))

    def _column(self, name: str):
        value = getattr(self._data, name)
        if isinstance(value, u.Quantity):
            return u.Quantity(np.asarray(value.mantissa)[self._positions].reshape(self._shape), value.unit)
        return np.asarray(value)[self._positions].reshape(self._shape)


class SourceContext(_EndpointContext):
    """Source endpoint columns visible to a pairing callable."""

    @property
    def id(self):
        return self._column("id")

    @property
    def type(self) -> str:
        return self._data.type

    @property
    def name(self) -> str | None:
        return self._data.name

    @property
    def owner(self):
        return self._data.owner

    @property
    def population_index(self):
        if self._data.population_index is None:
            raise AttributeError("This source does not expose population_index.")
        return self._column("population_index")

    def get(self, field: str):
        """Return a scalar or source-aligned owner field."""
        if not isinstance(field, str) or not field:
            raise ValueError("SourceContext.get field must be a non-empty string.")
        return self._data.get(field, self._positions, self._shape)


class SynapseContext(_EndpointContext):
    """Synapse endpoint and morphology columns visible to a pairing callable."""

    @property
    def id(self):
        return self._column("id")

    @property
    def population_index(self):
        return self._column("population_index")

    @property
    def placement_id(self):
        return self._column("placement_id")

    @property
    def point_id(self):
        return self._column("point_id")

    @property
    def cv_id(self):
        return self._column("cv_id")

    @property
    def branch_id(self):
        return self._column("branch_id")

    @property
    def branch_x(self):
        return self._column("branch_x")

    @property
    def branch_name(self):
        return self._column("branch_name")

    @property
    def branch_type(self):
        return self._column("branch_type")

    @property
    def radius(self):
        return self._column("radius")

    @property
    def path_distance_to_root(self):
        return self._column("path_distance_to_root")

    @property
    def path_distance_from_soma(self):
        return self._column("path_distance_from_soma")

    @property
    def type(self):
        return self._column("type")

    @property
    def name(self):
        return self._column("name")

    @property
    def cell(self):
        return self._data.cell

    @property
    def local_position(self):
        if self._data.local_position is None:
            raise ValueError("SynapseContext.local_position requires full 3-D point geometry.")
        return self._position_column(self._data.local_position)

    @property
    def position(self):
        if self._data.position is None:
            raise ValueError("SynapseContext.position requires full 3-D point geometry.")
        return self._position_column(self._data.position)

    def _position_column(self, value):
        values = np.asarray(value.mantissa)[self._positions]
        if len(self._shape) == 1:
            shape = (len(self._positions), 3)
        else:
            shape = (len(self._positions), 1, 3) if self._shape[1] == 1 else (1, len(self._positions), 3)
        return u.Quantity(values.reshape(shape), value.unit)

    def get(self, parameter: str):
        """Return one synapse parameter with the endpoint orientation."""
        return self._data.get(parameter, self._positions, self._shape)


class _SourceData:
    __slots__ = ("view", "id", "type", "name", "owner", "population_index")

    def __init__(self, view: EventSourceView) -> None:
        self.view = view
        self.id = view.source_id
        self.owner = view.owner
        self.type = view.owner.source_type
        self.name = view.owner.source_name
        population_index = getattr(view.owner, "population_index", None)
        if population_index is None and view.owner.source_type == "CellSpikeSource":
            population_index = np.arange(view.owner.size, dtype=np.int64)
        self.population_index = None if population_index is None else np.asarray(population_index)[self.id]

    def get(self, field: str, positions, shape):
        if not hasattr(self.owner, field):
            raise KeyError(f"Source type {self.type!r} has no field {field!r}.")
        value = getattr(self.owner, field)
        if callable(value):
            raise KeyError(f"Source field {field!r} is callable and cannot be used as data.")
        if isinstance(value, u.Quantity):
            mantissa = np.asarray(value.mantissa)
            if mantissa.ndim == 0:
                return value
            if mantissa.shape[0] != self.owner.size:
                raise ValueError(f"Source field {field!r} is not scalar or source-aligned.")
            selected = mantissa[self.id[positions]].reshape(shape + mantissa.shape[1:])
            return u.Quantity(selected, value.unit)
        values = np.asarray(value)
        if values.ndim == 0:
            return value
        if values.shape[0] != self.owner.size:
            raise ValueError(f"Source field {field!r} is not scalar or source-aligned.")
        return values[self.id[positions]].reshape(shape + values.shape[1:])


class _SynapseData:
    __slots__ = (
        "view",
        "cell",
        "id",
        "population_index",
        "placement_id",
        "point_id",
        "cv_id",
        "branch_id",
        "branch_x",
        "type",
        "name",
        "_geometry",
    )

    def __init__(self, view: SynapseView) -> None:
        self.view = view
        self.cell = view.cell
        self.id = view.id
        self.population_index = view.population_index
        self.placement_id = view.placement_id
        self.point_id = view.point_id
        self.cv_id = view.cv_id
        self.branch_id = view.branch_id
        self.branch_x = view.branch_x
        self.type = view.synapse_type
        self.name = view.name
        self._geometry = None

    @property
    def branch_name(self):
        return self._geometry_columns[0]

    @property
    def branch_type(self):
        return self._geometry_columns[1]

    @property
    def radius(self):
        return self._geometry_columns[2]

    @property
    def path_distance_to_root(self):
        return self._geometry_columns[3]

    @property
    def path_distance_from_soma(self):
        return self._geometry_columns[4]

    @property
    def local_position(self):
        return self._geometry_columns[5]

    @property
    def position(self):
        return self._geometry_columns[5]

    @property
    def _geometry_columns(self):
        if self._geometry is None:
            self._geometry = _synapse_geometry(self.view)
        return self._geometry

    def get(self, parameter: str, positions, shape):
        value = self.view[positions].get(parameter)
        if isinstance(value, u.Quantity):
            return u.Quantity(np.asarray(value.mantissa).reshape(shape), value.unit)
        return np.asarray(value).reshape(shape)


def _synapse_geometry(view: SynapseView):
    morpho = view.cell.morpho
    geometry = MorphologySpatialGeometry.build(morpho)
    names = []
    types = []
    radius = []
    root_distance = []
    soma_distance = []
    positions = []
    has_positions = True
    for branch_id, x in zip(view.branch_id.tolist(), view.branch_x.tolist()):
        branch_view = morpho.branch(index=int(branch_id))
        names.append(str(branch_view.name))
        types.append(str(branch_view.type))
        radius_value, point_value = interpolate_branch(morpho, int(branch_id), float(x))
        radius.append(float(radius_value.to_decimal(u.um)))
        root_distance.append(float(geometry.path_distance_to_root(int(branch_id), float(x)).to_decimal(u.um)))
        soma_distance.append(float(geometry.path_distance_from_soma(int(branch_id), float(x)).to_decimal(u.um)))
        if point_value is None:
            has_positions = False
            positions.append(None)
        else:
            positions.append(np.asarray(point_value.to_decimal(u.um), dtype=float))
    local_position = None
    if has_positions:
        local_position = u.Quantity(np.asarray(positions, dtype=float), u.um)
    return (
        np.asarray(names, dtype=object),
        np.asarray(types, dtype=object),
        u.Quantity(np.asarray(radius), u.um),
        u.Quantity(np.asarray(root_distance), u.um),
        u.Quantity(np.asarray(soma_distance), u.um),
        local_position,
    )


@dataclass(frozen=True)
class PairingSpec:
    """Immutable base declaration for an endpoint-pairing strategy."""

    group_by: str | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.group_by not in (None, "target_cell"):
            raise ValueError("group_by must be None or 'target_cell'.")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, Integral)):
            raise TypeError("pairing seed must be an integer or None.")


@dataclass(frozen=True)
class _Independent(PairingSpec):
    number: object = 1
    source_score: object = None
    synapse_score: object = None
    source_replace: bool = True
    synapse_replace: bool = True


@dataclass(frozen=True)
class _First(PairingSpec):
    side: str = "source"
    number: object = 1
    source_score: object = None
    synapse_score: object = None
    first_replace: bool = True
    partner_replace: bool = True


@dataclass(frozen=True)
class _ByEndpoint(PairingSpec):
    side: str = "source"
    endpoint_degree: object = 1
    partner_score: object = None
    replace: bool = True


@dataclass(frozen=True)
class _MatchDegrees(PairingSpec):
    source_degree: object = 1
    synapse_degree: object = 1


def independent(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace: bool = True,
    synapse_replace: bool = True,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Sample both endpoints independently for a fixed number of rows."""
    return _Independent(group_by, seed, number, source_score, synapse_score, source_replace, synapse_replace)


def source_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace: bool = True,
    replace: bool = True,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Sample sources first, then conditionally sample synapses."""
    return _First(group_by, seed, "source", number, source_score, synapse_score, source_replace, replace)


def synapse_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    synapse_replace: bool = True,
    replace: bool = True,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Sample synapses first, then conditionally sample sources."""
    return _First(group_by, seed, "synapse", number, source_score, synapse_score, synapse_replace, replace)


def by_source(
    degree,
    *,
    synapse_score=None,
    replace: bool = True,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Draw a partner count for every source and sample its synapses."""
    return _ByEndpoint(group_by, seed, "source", degree, synapse_score, replace)


def by_synapse(
    degree,
    *,
    source_score=None,
    replace: bool = True,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Draw a partner count for every synapse and sample its sources."""
    return _ByEndpoint(group_by, seed, "synapse", degree, source_score, replace)


def match_degrees(
    source_degree,
    synapse_degree,
    *,
    group_by: str | None = None,
    seed: int | None = None,
) -> PairingSpec:
    """Randomly match exact source and synapse stub counts."""
    return _MatchDegrees(group_by, seed, source_degree, synapse_degree)


@dataclass(frozen=True)
class _PoissonDegree:
    lam: object

    def __call__(self, ctx, rng):
        return rng.poisson(lam=self.lam, size=_degree_context_size(ctx))


@dataclass(frozen=True)
class _BinomialDegree:
    n: object
    p: object

    def __call__(self, ctx, rng):
        return rng.binomial(self.n, self.p, size=_degree_context_size(ctx))


@dataclass(frozen=True)
class _NegativeBinomialDegree:
    n: object
    p: object

    def __call__(self, ctx, rng):
        size = _degree_context_size(ctx)
        p = np.asarray(self.p, dtype=float)
        if np.any((p <= 0.0) | (p > 1.0)):
            raise ValueError("negative_binomial p must be within (0, 1].")
        rate = rng.gamma(self.n, scale=(1.0 - p) / p, size=size)
        return rng.poisson(rate, size=size)


@dataclass(frozen=True)
class _EmpiricalDegree:
    values: object
    probabilities: object

    def __call__(self, ctx, rng):
        values = _integer_counts(self.values, name="degree.empirical values")
        probabilities = _probabilities(self.probabilities, len(values), "degree.empirical probabilities")
        draws = np.asarray(rng.random(size=_degree_context_size(ctx)), dtype=float)
        return values[np.searchsorted(np.cumsum(probabilities), draws, side="right")]


class _DegreeNamespace:
    @staticmethod
    def poisson(lam) -> Callable:
        """Return a Poisson degree callable."""
        return _PoissonDegree(lam)

    @staticmethod
    def binomial(n, p) -> Callable:
        """Return a binomial degree callable."""
        return _BinomialDegree(n, p)

    @staticmethod
    def negative_binomial(n, p) -> Callable:
        """Return a negative-binomial degree callable."""
        return _NegativeBinomialDegree(n, p)

    @staticmethod
    def empirical(values, probabilities) -> Callable:
        """Return a callable sampling one explicit empirical PMF."""
        return _EmpiricalDegree(values, probabilities)


degree = _DegreeNamespace()


@dataclass(frozen=True)
class _EndpointPairs:
    source_position: np.ndarray
    synapse_position: np.ndarray


def materialize_pairing(
    spec: PairingSpec,
    source: EventSourceView,
    synapse: SynapseView,
    *,
    seed_root: int = 0,
    seed_path: tuple[str, ...] = (),
) -> _EndpointPairs:
    """Materialize temporary local endpoint positions from one pairing spec."""
    if not isinstance(spec, PairingSpec):
        raise TypeError("pairing must be returned by braincell.network.connection pairing helpers.")
    _require_unique(source.source_id, "source")
    _require_unique(synapse.id, "synapse")
    source_data = _SourceData(source)
    synapse_data = _SynapseData(synapse)
    base_seed = int(spec.seed) if spec.seed is not None else _derived_seed(seed_root, *seed_path)
    source_positions = np.arange(len(source), dtype=np.int64)
    if spec.group_by is None:
        groups = [(None, np.arange(len(synapse), dtype=np.int64))]
    else:
        population_indices = synapse_data.population_index
        owners = np.unique(population_indices)
        groups = [
            (
                GroupContext(index, int(owner), int(np.count_nonzero(population_indices == owner)), len(owners)),
                np.flatnonzero(population_indices == owner).astype(np.int64),
            )
            for index, owner in enumerate(owners.tolist())
        ]
    source_rows = []
    synapse_rows = []
    for group_index, (group, synapse_positions) in enumerate(groups):
        group_seed = _derived_seed(base_seed, "group", _group_key(group, group_index))
        result = _materialize_group(
            spec,
            source_data,
            synapse_data,
            source_positions,
            synapse_positions,
            group,
            group_index,
            len(groups),
            group_seed,
        )
        source_rows.append(result.source_position)
        synapse_rows.append(result.synapse_position)
    result = _EndpointPairs(np.concatenate(source_rows), np.concatenate(synapse_rows))
    if len(result.source_position) == 0:
        raise ValueError("pairing produced zero connection rows.")
    return result


def _materialize_group(spec, source_data, synapse_data, sources, synapses, group, group_index, n_groups, seed):
    if isinstance(spec, _Independent):
        count = _group_number(spec.number, group_index, n_groups)
        if count == 0:
            return _EndpointPairs(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
        source_ctx = _context(source_data, synapse_data, sources, synapses, group)
        source_weight = _score(spec.source_score, source_ctx, (1, len(sources)), side="source", positions=sources)
        synapse_weight = _score(spec.synapse_score, source_ctx, (1, len(synapses)), side="synapse", positions=synapses)
        return _EndpointPairs(
            _sample(sources, source_weight[0], count, spec.source_replace, _derived_seed(seed, "source")),
            _sample(synapses, synapse_weight[0], count, spec.synapse_replace, _derived_seed(seed, "synapse")),
        )
    if isinstance(spec, _First):
        count = _group_number(spec.number, group_index, n_groups)
        if count == 0:
            return _EndpointPairs(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
        ctx = _context(source_data, synapse_data, sources, synapses, group)
        if spec.side == "source":
            first_weight = _score(spec.source_score, ctx, (1, len(sources)), side="source", positions=sources)
            first = _sample(sources, first_weight[0], count, spec.first_replace, _derived_seed(seed, "first"))
            second = _conditional_sample(
                fixed_side="source",
                fixed=first,
                candidates=synapses,
                score=spec.synapse_score,
                replace=spec.partner_replace,
                source_data=source_data,
                synapse_data=synapse_data,
                group=group,
                seed=_derived_seed(seed, "conditional"),
            )
            return _EndpointPairs(first, second)
        first_weight = _score(spec.synapse_score, ctx, (1, len(synapses)), side="synapse", positions=synapses)
        first = _sample(synapses, first_weight[0], count, spec.first_replace, _derived_seed(seed, "first"))
        second = _conditional_sample(
            fixed_side="synapse",
            fixed=first,
            candidates=sources,
            score=spec.source_score,
            replace=spec.partner_replace,
            source_data=source_data,
            synapse_data=synapse_data,
            group=group,
            seed=_derived_seed(seed, "conditional"),
        )
        return _EndpointPairs(second, first)
    if isinstance(spec, _ByEndpoint):
        if spec.side == "source":
            ctx = PairingContext(SourceContext(source_data, sources, fixed=None), None, group)
            raw_degree = _group_degree(spec.endpoint_degree, group_index, n_groups, len(sources))
            counts = _degree_counts(raw_degree, ctx, len(sources), _derived_seed(seed, "degree"))
            fixed = np.repeat(sources, counts)
            partner = _conditional_sample(
                fixed_side="source",
                fixed=fixed,
                candidates=synapses,
                score=spec.partner_score,
                replace=spec.replace,
                source_data=source_data,
                synapse_data=synapse_data,
                group=group,
                seed=_derived_seed(seed, "partners"),
            )
            return _EndpointPairs(fixed, partner)
        ctx = PairingContext(None, SynapseContext(synapse_data, synapses, fixed=None), group)
        raw_degree = _group_degree(spec.endpoint_degree, group_index, n_groups, len(synapses), positions=synapses)
        counts = _degree_counts(raw_degree, ctx, len(synapses), _derived_seed(seed, "degree"))
        fixed = np.repeat(synapses, counts)
        partner = _conditional_sample(
            fixed_side="synapse",
            fixed=fixed,
            candidates=sources,
            score=spec.partner_score,
            replace=spec.replace,
            source_data=source_data,
            synapse_data=synapse_data,
            group=group,
            seed=_derived_seed(seed, "partners"),
        )
        return _EndpointPairs(partner, fixed)
    if isinstance(spec, _MatchDegrees):
        source_ctx = PairingContext(SourceContext(source_data, sources, fixed=None), None, group)
        synapse_ctx = PairingContext(None, SynapseContext(synapse_data, synapses, fixed=None), group)
        source_raw = _group_degree(spec.source_degree, group_index, n_groups, len(sources))
        synapse_raw = _group_degree(spec.synapse_degree, group_index, n_groups, len(synapses), positions=synapses)
        source_count = _degree_counts(source_raw, source_ctx, len(sources), _derived_seed(seed, "source_degree"))
        synapse_count = _degree_counts(synapse_raw, synapse_ctx, len(synapses), _derived_seed(seed, "synapse_degree"))
        if int(source_count.sum()) != int(synapse_count.sum()):
            label = "global" if group is None else f"target cell {group.target_population_index}"
            raise ValueError(
                f"match_degrees requires equal stub sums for {label}; got "
                f"source={int(source_count.sum())}, synapse={int(synapse_count.sum())}."
            )
        source_stub = np.repeat(sources, source_count)
        synapse_stub = np.repeat(synapses, synapse_count)
        if len(source_stub) == 0:
            return _EndpointPairs(source_stub, synapse_stub)
        rng = brainstate.random.RandomState(seed)
        synapse_stub = np.asarray(rng.permutation(synapse_stub), dtype=np.int64)
        return _EndpointPairs(source_stub, synapse_stub)
    raise TypeError(f"Unsupported pairing spec {type(spec).__name__!s}.")


def _context(source_data, synapse_data, sources, synapses, group):
    return PairingContext(
        SourceContext(source_data, sources, fixed=False),
        SynapseContext(synapse_data, synapses, fixed=False),
        group,
    )


def _conditional_sample(
    *,
    fixed_side,
    fixed,
    candidates,
    score,
    replace,
    source_data,
    synapse_data,
    group,
    seed,
):
    fixed = np.asarray(fixed, dtype=np.int64)
    if fixed.size == 0:
        return np.asarray([], dtype=np.int64)
    unique, inverse, counts = np.unique(fixed, return_inverse=True, return_counts=True)
    if fixed_side == "source":
        ctx = PairingContext(
            SourceContext(source_data, unique, fixed=True),
            SynapseContext(synapse_data, candidates, fixed=False),
            group,
        )
        side = "synapse"
    else:
        ctx = PairingContext(
            SourceContext(source_data, candidates, fixed=False),
            SynapseContext(synapse_data, unique, fixed=True),
            group,
        )
        side = "source"
    weights = _score(score, ctx, (len(unique), len(candidates)), side=side, positions=candidates)
    sampled_by_fixed = []
    for row, (endpoint, count) in enumerate(zip(unique.tolist(), counts.tolist())):
        endpoint_id = source_data.id[int(endpoint)] if fixed_side == "source" else synapse_data.id[int(endpoint)]
        sampled_by_fixed.append(
            _sample(
                candidates,
                weights[row],
                count,
                replace,
                _derived_seed(seed, fixed_side, int(endpoint_id)),
            )
        )
    offsets = np.zeros(len(unique), dtype=np.int64)
    result = np.empty(len(fixed), dtype=np.int64)
    for row, unique_index in enumerate(inverse.tolist()):
        result[row] = sampled_by_fixed[unique_index][offsets[unique_index]]
        offsets[unique_index] += 1
    return result


def _score(score, ctx, shape, *, side, positions):
    if score is None:
        result = np.ones(shape, dtype=float)
    else:
        value = score(ctx) if callable(score) else score
        if isinstance(value, u.Quantity):
            if not u.get_unit(value).is_unitless:
                raise TypeError(f"{side}_score must be dimensionless.")
            value = value.mantissa
        values = np.asarray(value, dtype=float)
        if not callable(score) and values.ndim == 1:
            full_size = len(ctx.source._data.id) if side == "source" else len(ctx.synapse._data.id)
            if values.shape[0] == full_size and values.shape[0] != shape[-1]:
                values = values[np.asarray(positions, dtype=np.int64)]
        try:
            result = np.asarray(np.broadcast_to(values, shape), dtype=float)
        except ValueError as exc:
            raise ValueError(f"{side}_score must broadcast to {shape!r}, got {values.shape!r}.") from exc
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{side}_score must contain finite non-negative values.")
    if np.any(np.sum(result, axis=1) <= 0.0):
        raise ValueError(f"{side}_score must have positive support in every normalization row.")
    return result


def _sample(candidates, weights, count, replace, seed):
    candidates = np.asarray(candidates, dtype=np.int64)
    if count == 0:
        return np.asarray([], dtype=np.int64)
    if isinstance(replace, np.bool_):
        replace = bool(replace)
    if not isinstance(replace, bool):
        raise TypeError("replace flags must be bool values.")
    support = int(np.count_nonzero(weights > 0.0))
    if not replace and count > support:
        raise ValueError(f"Sampling without replacement requires at least {count} positive candidates; got {support}.")
    rng = brainstate.random.RandomState(seed)
    probabilities = np.asarray(weights, dtype=float) / float(np.sum(weights))
    return np.asarray(rng.choice(candidates, size=count, replace=replace, p=probabilities), dtype=np.int64)


def _degree_counts(value, ctx, size, seed):
    if callable(value):
        value = value(ctx, brainstate.random.RandomState(seed))
    values = np.asarray(value)
    if values.ndim == 0:
        values = np.broadcast_to(values, (size,))
    elif values.shape != (size,):
        raise ValueError(f"degree must be scalar or have shape {(size,)!r}, got {values.shape!r}.")
    return _integer_counts(values, name="degree")


def _integer_counts(value, *, name):
    values = np.asarray(value)
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers.")
    values = values.astype(np.int64, copy=False).reshape(-1)
    if np.any(values < 0):
        raise ValueError(f"{name} must contain non-negative values.")
    return values


def _group_number(value, group_index, n_groups):
    values = np.asarray(value)
    if values.ndim == 0:
        selected = values
    elif values.shape == (n_groups,):
        selected = values[group_index]
    else:
        raise ValueError(f"number must be scalar or have shape {(n_groups,)!r}, got {values.shape!r}.")
    result = _integer_counts(np.asarray([selected]), name="number")
    if result[0] < 0:
        raise ValueError("number must be >= 0 for every pairing group.")
    return int(result[0])


def _group_degree(value, group_index, n_groups, endpoint_size, positions=None):
    if callable(value) or np.asarray(value).ndim == 0:
        return value
    values = np.asarray(value)
    if values.shape == (n_groups, endpoint_size):
        return values[group_index]
    if positions is not None and values.ndim == 1 and values.shape[0] > endpoint_size:
        return values[np.asarray(positions, dtype=np.int64)]
    return value


def _degree_context_size(ctx):
    endpoint = ctx.source if ctx.source is not None else ctx.synapse
    return len(endpoint._positions)


def _probabilities(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)!r}, got {result.shape!r}.")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0) or np.sum(result) <= 0.0:
        raise ValueError(f"{name} must be finite, non-negative, and have positive mass.")
    return result / np.sum(result)


def _require_unique(values, label):
    if len(np.unique(values)) != len(values):
        raise ValueError(f"pairing candidate {label} view must contain unique IDs.")


def _group_key(group, fallback):
    return fallback if group is None else group.target_population_index


def _derived_seed(seed, *parts):
    payload = "\x1f".join((str(int(seed)), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")
