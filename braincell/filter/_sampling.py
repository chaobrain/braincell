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

"""Continuous morphology sampling backend."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import brainstate
import brainunit as u
import numpy as np
from scipy.integrate import quad
from scipy.stats.sampling import NumericalInversePolynomial

from braincell.morph.morphology import Morphology
from . import helper

Density = Callable[["SamplingContext"], object]
_MEASURES = {"normalized", "length", "lateral_area", "area"}

__all__ = ["SamplingContext", "sample_locations_from_region"]


@dataclass(frozen=True)
class SamplingContext:
    """Geometry visible to a continuous sampling-density callable.

    ``branch_x``, ``radius`` and the distance fields are scalar during
    numerical inversion and may be arrays when a density is inspected or
    evaluated in bulk. Positions use morphology-local coordinates in v1.
    """

    branch_id: int
    branch_name: str
    branch_type: str
    branch_x: object
    radius: object
    path_distance_to_root: object
    path_distance_from_soma: object
    _local_position: object | None = None

    @property
    def local_position(self) -> object:
        """Morphology-local 3-D position.

        Raises
        ------
        ValueError
            If the morphology does not contain full 3-D point geometry.
        """
        if self._local_position is None:
            raise ValueError("SamplingContext.local_position requires full 3-D point geometry.")
        return self._local_position

    @property
    def position(self) -> object:
        """World position, equal to ``local_position`` until transforms land."""
        if self._local_position is None:
            raise ValueError("SamplingContext.position requires full 3-D point geometry.")
        return self._local_position


@dataclass(frozen=True)
class _ContinuousComponent:
    branch_id: int
    x0: float
    x1: float
    branch_length_um: float
    segment_x0: float
    segment_x1: float
    radius0_um: float
    radius1_um: float
    point0_um: np.ndarray | None
    point1_um: np.ndarray | None

    def radius_um(self, x: object) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        fraction = (values - self.segment_x0) / (self.segment_x1 - self.segment_x0)
        return self.radius0_um + fraction * (self.radius1_um - self.radius0_um)

    def position_um(self, x: object) -> np.ndarray | None:
        if self.point0_um is None or self.point1_um is None:
            return None
        values = np.asarray(x, dtype=float)
        fraction = (values - self.segment_x0) / (self.segment_x1 - self.segment_x0)
        return self.point0_um + np.expand_dims(fraction, axis=-1) * (self.point1_um - self.point0_um)

    def jacobian(self, measure: str, x: object) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if measure == "normalized":
            return np.ones_like(values)
        if measure == "length":
            return np.full_like(values, self.branch_length_um)
        dr_dx = (self.radius1_um - self.radius0_um) / (self.segment_x1 - self.segment_x0)
        slant_dx = np.sqrt(self.branch_length_um**2 + dr_dx**2)
        return 2.0 * np.pi * self.radius_um(values) * slant_dx


@dataclass(frozen=True)
class _AtomComponent:
    branch_id: int
    x: float
    radius_um: float
    area_um2: float
    point_um: np.ndarray | None


@dataclass(frozen=True)
class _PreparedComponent:
    source: _ContinuousComponent | _AtomComponent
    log_mass: float
    inverse: object | None = None
    q0: float | None = None
    q1: float | None = None


def _coerce_inputs(number: object, seed: object, measure: object, u_resolution: object) -> tuple[int, int, str, float]:
    if isinstance(number, bool) or not isinstance(number, Integral):
        raise TypeError(f"number must be an integer, got {number!r}.")
    n = int(number)
    if n <= 0:
        raise ValueError(f"number must be > 0, got {n!r}.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    if not isinstance(measure, str) or measure not in _MEASURES:
        raise ValueError(f"measure must be one of {sorted(_MEASURES)!r}, got {measure!r}.")
    if isinstance(u_resolution, bool) or not isinstance(u_resolution, (int, float)):
        raise TypeError("u_resolution must be a real scalar.")
    resolution = float(u_resolution)
    if not np.isfinite(resolution) or not 1e-12 <= resolution <= 1e-5:
        raise ValueError("u_resolution must be within [1e-12, 1e-5].")
    return n, int(seed), measure, resolution


def _inside_interval(x: float, prox: float, dist: float) -> bool:
    return prox <= x < dist or (x == 1.0 and dist == 1.0 and prox <= x)


def _build_components(
    morpho: Morphology,
    intervals: tuple[helper.Interval, ...],
    *,
    measure: str,
) -> list[_ContinuousComponent | _AtomComponent]:
    components: list[_ContinuousComponent | _AtomComponent] = []
    for branch_id, prox, dist in helper.normalize_region_intervals(intervals):
        branch = morpho.branch(index=branch_id).branch
        lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        r0 = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
        r1 = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
        total = float(np.sum(lengths))
        if total <= 0.0:
            continue
        p0 = (
            None if branch.points_proximal is None else np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
        )
        p1 = None if branch.points_distal is None else np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)

        cursor = 0.0
        for index, segment_length in enumerate(lengths):
            if segment_length > 0.0:
                segment_x0 = cursor / total
                cursor += float(segment_length)
                segment_x1 = cursor / total
                x0 = max(prox, segment_x0)
                x1 = min(dist, segment_x1)
                if x1 > x0:
                    components.append(
                        _ContinuousComponent(
                            branch_id=branch_id,
                            x0=x0,
                            x1=x1,
                            branch_length_um=total,
                            segment_x0=segment_x0,
                            segment_x1=segment_x1,
                            radius0_um=float(r0[index]),
                            radius1_um=float(r1[index]),
                            point0_um=None if p0 is None else p0[index],
                            point1_um=None if p1 is None else p1[index],
                        )
                    )
            elif measure == "area":
                x = cursor / total
                if _inside_interval(x, prox, dist):
                    area = np.pi * (float(r0[index]) + float(r1[index])) * abs(float(r1[index]) - float(r0[index]))
                    if area > 0.0:
                        point = None
                        if p0 is not None and p1 is not None:
                            point = 0.5 * (p0[index] + p1[index])
                        components.append(
                            _AtomComponent(
                                branch_id=branch_id,
                                x=x,
                                radius_um=0.5 * (float(r0[index]) + float(r1[index])),
                                area_um2=area,
                                point_um=point,
                            )
                        )
    return components


def _branch_bases(morpho: Morphology) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    lengths = {
        index: float(morpho.branch(index=index).branch.length.to_decimal(u.um)) for index in range(len(morpho.branches))
    }
    edges = {int(edge.child.index): edge for edge in morpho.edges}
    root_id = int(morpho.root.index)
    entry = {root_id: 0.0}
    root_base = {root_id: 0.0}
    soma_base = {root_id: 0.0}
    resolving: set[int] = set()

    def resolve(branch_id: int) -> None:
        if branch_id in root_base:
            return
        if branch_id in resolving:
            raise ValueError(f"Morphology contains a cycle at branch {branch_id!r}.")
        resolving.add(branch_id)
        edge = edges[branch_id]
        parent = int(edge.parent.index)
        resolve(parent)
        entry[branch_id] = float(edge.child_x)
        step = abs(float(edge.parent_x) - entry[parent]) * lengths[parent]
        root_base[branch_id] = root_base[parent] + step
        soma_base[branch_id] = 0.0 if str(edge.parent.type) == "soma" else soma_base[parent] + step
        resolving.remove(branch_id)

    for branch_id in range(len(morpho.branches)):
        resolve(branch_id)
    return entry, root_base, soma_base


def _context(
    morpho: Morphology,
    component: _ContinuousComponent | _AtomComponent,
    x: object,
    bases: tuple[dict[int, float], dict[int, float], dict[int, float]],
) -> SamplingContext:
    branch_id = component.branch_id
    branch_view = morpho.branch(index=branch_id)
    values = np.asarray(x, dtype=float)
    entry, root_base, soma_base = bases
    length_um = float(branch_view.branch.length.to_decimal(u.um))
    local_distance = np.abs(values - entry[branch_id]) * length_um
    if isinstance(component, _ContinuousComponent):
        radius = component.radius_um(values)
        position = component.position_um(values)
    else:
        radius = np.full_like(values, component.radius_um)
        position = component.point_um
        if position is not None and values.ndim > 0:
            position = np.broadcast_to(position, values.shape + (3,))
    soma_distance = np.zeros_like(values) if str(branch_view.type) == "soma" else soma_base[branch_id] + local_distance
    return SamplingContext(
        branch_id=branch_id,
        branch_name=str(branch_view.name),
        branch_type=str(branch_view.type),
        branch_x=x,
        radius=u.Quantity(radius, u.um),
        path_distance_to_root=u.Quantity(root_base[branch_id] + local_distance, u.um),
        path_distance_from_soma=u.Quantity(soma_distance, u.um),
        _local_position=None if position is None else u.Quantity(position, u.um),
    )


def _density_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, u.Quantity):
        if not u.get_unit(value).is_unitless:
            raise TypeError("density must return dimensionless values.")
        value = u.get_mantissa(value)
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.broadcast_to(values, shape)
    elif values.shape != shape:
        raise ValueError(f"density must return a scalar or shape {shape!r}, got {values.shape!r}.")
    if np.any(~np.isfinite(values)):
        raise ValueError("density must return finite values.")
    if np.any(values < 0.0):
        raise ValueError("density must return non-negative values.")
    return values


def _log_density_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, u.Quantity):
        if not u.get_unit(value).is_unitless:
            raise TypeError("log density must be dimensionless.")
        value = u.get_mantissa(value)
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.broadcast_to(values, shape)
    elif values.shape != shape:
        raise ValueError(f"log density must return a scalar or shape {shape!r}, got {values.shape!r}.")
    if np.any(~np.isfinite(values)):
        raise ValueError("log density must return finite values.")
    return values


def _density_value(density: Density, context: SamplingContext, *, log_shift: float | None) -> float:
    if log_shift is not None:
        log_value = _log_density_array(density._log_density(context), shape=())  # type: ignore[attr-defined]
        value = float(np.exp(float(log_value) - log_shift))
    else:
        value = float(_density_array(density(context), shape=()))
    return value


def _builtin_log_shift(
    density: Density,
    morpho: Morphology,
    components: list[_ContinuousComponent | _AtomComponent],
    bases: tuple[dict[int, float], dict[int, float], dict[int, float]],
) -> float | None:
    if not hasattr(density, "_log_density"):
        return None
    maxima: list[float] = []
    for component in components:
        xs = (
            np.asarray([component.x0, component.x1])
            if isinstance(component, _ContinuousComponent)
            else np.asarray(component.x)
        )
        context = _context(morpho, component, xs, bases)
        values = _log_density_array(density._log_density(context), shape=xs.shape)  # type: ignore[attr-defined]
        maxima.append(float(np.max(values)))
    return max(0.0, max(maxima, default=0.0))


def _analytic_mass(component: _ContinuousComponent, measure: str) -> tuple[float, float, float]:
    q0 = float(component.jacobian(measure, component.x0))
    q1 = float(component.jacobian(measure, component.x1))
    mass = 0.5 * (q0 + q1) * (component.x1 - component.x0)
    return mass, q0, q1


def _prepare_components(
    morpho: Morphology,
    components: list[_ContinuousComponent | _AtomComponent],
    *,
    measure: str,
    density: Density | None,
    u_resolution: float,
) -> list[_PreparedComponent]:
    bases = _branch_bases(morpho)
    log_shift = None if density is None else _builtin_log_shift(density, morpho, components, bases)
    prepared: list[_PreparedComponent] = []
    for component in components:
        if isinstance(component, _AtomComponent):
            preference = (
                1.0
                if density is None
                else _density_value(density, _context(morpho, component, component.x, bases), log_shift=log_shift)
            )
            mass = preference * component.area_um2
            if mass > 0.0:
                prepared.append(_PreparedComponent(component, float(np.log(mass))))
            continue

        if density is None:
            mass, q0, q1 = _analytic_mass(component, measure)
            if mass > 0.0:
                prepared.append(_PreparedComponent(component, float(np.log(mass)), q0=q0, q1=q1))
            continue

        def pdf(x: float, *, source: _ContinuousComponent = component) -> float:
            context = _context(morpho, source, float(x), bases)
            return _density_value(density, context, log_shift=log_shift) * float(source.jacobian(measure, x))

        mass, _ = quad(pdf, component.x0, component.x1, epsabs=0.0, epsrel=u_resolution, limit=200)
        if mass <= 0.0:
            continue
        try:
            inverse = NumericalInversePolynomial(
                type("_Distribution", (), {"pdf": staticmethod(pdf)})(),
                domain=(component.x0, component.x1),
                center=0.5 * (component.x0 + component.x1),
                u_resolution=u_resolution,
            )
        except Exception as exc:
            raise ValueError(
                "density could not be inverted on a morphology segment; split the Region around disconnected "
                "or non-smooth support."
            ) from exc
        prepared.append(_PreparedComponent(component, float(np.log(mass)), inverse=inverse))
    if not prepared:
        raise ValueError("sample() density has zero mass over the selected region.")
    return prepared


def _inverse_linear(component: _ContinuousComponent, u_value: object, q0: float, q1: float) -> object:
    values = np.asarray(u_value, dtype=float)
    delta_x = component.x1 - component.x0
    if np.isclose(q0, q1, rtol=1e-14, atol=0.0):
        result = component.x0 + values * delta_x
        return float(result) if result.ndim == 0 else result
    target = values * 0.5 * (q0 + q1)
    discriminant = np.maximum(0.0, q0 * q0 + 2.0 * (q1 - q0) * target)
    denominator = q0 + np.sqrt(discriminant)
    fraction = np.divide(2.0 * target, denominator, out=np.array(values, copy=True), where=denominator != 0.0)
    result = component.x0 + np.clip(fraction, 0.0, 1.0) * delta_x
    return float(result) if result.ndim == 0 else result


def sample_locations_from_region(
    morpho: Morphology,
    *,
    intervals: tuple[helper.Interval, ...],
    number: object,
    seed: object,
    measure: object = "length",
    density: Density | None = None,
    u_resolution: object = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize continuous random locations from a morphology region."""
    n, seed_value, measure_name, resolution = _coerce_inputs(number, seed, measure, u_resolution)
    if density is not None and not callable(density):
        raise TypeError("density must be callable or None.")
    components = _build_components(morpho, intervals, measure=measure_name)
    if not components:
        raise ValueError("sample() cannot sample from an empty region under the selected measure.")
    prepared = _prepare_components(
        morpho,
        components,
        measure=measure_name,
        density=density,
        u_resolution=resolution,
    )

    log_masses = np.asarray([item.log_mass for item in prepared], dtype=float)
    masses = np.exp(log_masses - np.max(log_masses))
    cumulative = np.cumsum(masses)
    total = float(cumulative[-1])
    uniforms = np.asarray(brainstate.random.RandomState(seed_value).random(n), dtype=float)
    targets = uniforms * total
    component_ids = np.minimum(np.searchsorted(cumulative, targets, side="right"), len(prepared) - 1)
    branch_ids = np.empty(n, dtype=np.int64)
    branch_x = np.empty(n, dtype=float)
    if len(prepared) == 1:
        groups = ((0, slice(None)),)
    elif len(prepared) <= 8:
        groups = tuple((index, np.flatnonzero(component_ids == index)) for index in range(len(prepared)))
    else:
        order = np.argsort(component_ids, kind="stable")
        counts = np.bincount(component_ids, minlength=len(prepared))
        offsets = np.concatenate(([0], np.cumsum(counts)))
        groups = tuple((index, order[offsets[index] : offsets[index + 1]]) for index in range(len(prepared)))
    for index, selected in groups:
        if not isinstance(selected, slice) and len(selected) == 0:
            continue
        item = prepared[index]
        lower = 0.0 if index == 0 else float(cumulative[index - 1])
        local_u = np.clip((targets[selected] - lower) / float(masses[index]), 0.0, 1.0)
        branch_ids[selected] = item.source.branch_id
        if isinstance(item.source, _AtomComponent):
            branch_x[selected] = item.source.x
        elif item.inverse is not None:
            branch_x[selected] = np.asarray(item.inverse.ppf(local_u), dtype=float)
        else:
            branch_x[selected] = np.asarray(
                _inverse_linear(item.source, local_u, float(item.q0), float(item.q1)),
                dtype=float,
            )
    return branch_ids, branch_x
