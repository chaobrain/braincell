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

"""Validation and lowering for network connections."""

from __future__ import annotations

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell._discretization.base import locate_cv_on_branch
from braincell.filter import LocsetExpr

from .core import Population


@dataclass(frozen=True)
class ConnectionBlock:
    """Runtime-ready sparse contact block."""

    pre_population: str
    post_population: str
    synapse: str
    layout_id: int
    n_active: int
    pre_index: np.ndarray
    post_index: np.ndarray
    synapse_index: np.ndarray
    weight: object
    delay_steps: np.ndarray
    buffer_size: int
    source_cv_id: int
    packed: bool = False
    event_source: object | None = None


def lower_direct_connections(
    populations: dict[str, Population],
    *,
    dt,
    delay_quantization: str = "nearest",
) -> tuple[ConnectionBlock, ...]:
    """Lower target-owned live ``connect()`` rows into network route blocks."""
    _validate_time_quantity(dt, name="dt")
    delay_quantization = _normalize_delay_quantization(delay_quantization)
    cell_populations = tuple(population for population in populations.values() if population.kind == "cell")
    population_by_cell = {id(population.cell): population for population in cell_populations}
    blocks: list[ConnectionBlock] = []
    for post in cell_populations:
        for connection in post.cell.connections._call_views(scheduled=False):
            source = connection.source
            owner = source.execution_owner
            pre = population_by_cell.get(id(owner))
            if pre is None:
                source_name = source.source_name or source.source_type
                raise RuntimeError(
                    f"Live EventSource {source_name!r} is outside this Network execution scope; "
                    "add its owning Cell as a population."
                )
            synapse_types = tuple(dict.fromkeys(connection.synapse_type.tolist()))
            if len(synapse_types) != 1:
                raise TypeError("One direct Connection block must target exactly one synapse type.")
            synapse_type = str(synapse_types[0])
            layout_id = post.cell._get_synapse_store().layout_id(synapse_type)
            layout = post.cell.runtime.layouts[layout_id]
            synapse_index = post.cell._get_synapse_store().runtime_rows(connection.synapse_id).astype(np.int32)
            delay_steps = _expand_delay_steps(
                connection.delay,
                dt=dt,
                n_contact=len(connection),
                quantization=delay_quantization,
            )
            blocks.append(
                ConnectionBlock(
                    pre_population=pre.name,
                    post_population=post.name,
                    synapse=str(connection.synapse_name[0]),
                    layout_id=int(layout_id),
                    n_active=int(layout.n_active),
                    pre_index=connection.source_index.astype(np.int32),
                    post_index=connection.synapse.population_index.astype(np.int32),
                    synapse_index=synapse_index,
                    weight=connection.weight,
                    delay_steps=delay_steps,
                    buffer_size=int(np.max(delay_steps, initial=1)) + 1,
                    # Live EventSource routes read their own endpoint mapping;
                    # this legacy scalar is only used by raw Cell spike blocks.
                    source_cv_id=0,
                    packed=True,
                    event_source=source,
                )
            )
    return tuple(blocks)


def resolve_source_cv(cell, source: LocsetExpr) -> int:
    """Resolve a single continuous presynaptic location to its owning CV."""
    if not isinstance(source, LocsetExpr):
        raise TypeError(
            f"Connection source must be a LocsetExpr resolving to one location, got {type(source).__name__!s}."
        )
    mask = source.evaluate(cell.morpho)
    if len(mask) != 1:
        raise ValueError(f"Connection source must resolve to exactly one presynaptic location; got {len(mask)!r}.")
    branch_id = int(mask.branch_id[0])
    branch_x = float(mask.branch_x[0])
    ids = cell.cv_tree.branch_to_cv_ids[int(branch_id)]
    return locate_cv_on_branch(
        ids,
        cell.cvs,
        x=float(branch_x),
    )


def _expand_delay_steps(delay, *, dt, n_contact: int, quantization: str = "nearest") -> np.ndarray:
    """Return fixed-step delay offsets for one connection table.

    Parameters
    ----------
    delay : Quantity[time]
        Scalar or per-edge delay.
    dt : Quantity[time]
        Fixed simulation step.
    n_contact : int
        Number of contacts.
    quantization : {"nearest", "ceil", "floor", "strict"}
        Policy for delays that do not fall on the fixed-step grid.
    """
    _validate_time_quantity(delay, name="delay")
    quantization = _normalize_delay_quantization(quantization)
    delay_ms = np.asarray(delay.to_decimal(u.ms), dtype=float)
    if delay_ms.shape == ():
        delay_ms = np.broadcast_to(delay_ms, (n_contact,)).copy()
    if delay_ms.shape != (n_contact,):
        raise ValueError(f"Connection delay must be scalar or shape {(n_contact,)!r}, got {delay_ms.shape!r}.")
    if np.any(delay_ms < 0.0):
        raise ValueError("Connection delay must be >= 0.")
    dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
    raw_steps = delay_ms / dt_ms
    rounded_raw_steps = np.rint(raw_steps)
    raw_steps = np.where(
        np.isclose(raw_steps, rounded_raw_steps, rtol=1e-7, atol=1e-7),
        rounded_raw_steps,
        raw_steps,
    )
    if quantization == "strict":
        rounded = np.rint(raw_steps).astype(np.int32)
        if not np.allclose(raw_steps, rounded, rtol=1e-9, atol=1e-9):
            raise ValueError("Connection delay must be an integer multiple of dt when delay_quantization='strict'.")
        steps = rounded
    elif quantization == "nearest":
        steps = _round_half_up_steps(raw_steps).astype(np.int32)
    elif quantization == "ceil":
        steps = np.ceil(raw_steps - 1e-12).astype(np.int32)
    elif quantization == "floor":
        steps = np.floor(raw_steps + 1e-12).astype(np.int32)
    else:  # pragma: no cover
        raise ValueError("Connection delay_quantization must be 'nearest', 'ceil', 'floor', or 'strict'.")
    return np.maximum(steps, 0)


def _round_half_up_steps(values: np.ndarray) -> np.ndarray:
    """Round non-negative step ratios, snapping numerical half ties upward."""
    half = np.floor(values) + 0.5
    magnitude = np.abs(values)
    ulp = np.nextafter(magnitude, np.inf) - magnitude
    snapped = np.where(np.abs(values - half) <= 4.0 * ulp, half, values)
    return np.floor(snapped + 0.5)


def _normalize_delay_quantization(value: str) -> str:
    if value not in ("nearest", "ceil", "floor", "strict"):
        raise ValueError(f"Network delay_quantization must be 'nearest', 'ceil', 'floor', or 'strict', got {value!r}.")
    return value


def _validate_time_quantity(value, *, name: str) -> None:
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"Network {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if name == "dt" and decimal.shape not in ((), (1,)):
        raise ValueError(f"Network dt must be scalar, got shape {decimal.shape!r}.")
    if name == "dt" and float(decimal.reshape(())) <= 0.0:
        raise ValueError(f"Network dt must be > 0, got {value!r}.")
