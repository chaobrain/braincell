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

from braincell._misc import validate_time_quantity

from .core import Population
from .event import round_half_up_steps_host as _round_half_up_steps


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
    event_source: object


def lower_direct_connections(
    populations: dict[str, Population],
    *,
    dt,
    delay_quantization: str = "nearest",
) -> tuple[ConnectionBlock, ...]:
    """Lower target-owned live ``connect()`` rows into network route blocks."""
    validate_time_quantity(dt, name="dt", prefix="Network")
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
                    event_source=source,
                )
            )
    return tuple(blocks)


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
    validate_time_quantity(
        delay,
        name="delay",
        prefix="Network",
        # A delay may be a per-contact vector, and zero means immediate
        # delivery, so neither the scalar nor the positivity rule applies.
        require_scalar=False,
        require_positive=False,
    )
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


def _normalize_delay_quantization(value: str) -> str:
    if value not in ("nearest", "ceil", "floor", "strict"):
        raise ValueError(f"Network delay_quantization must be 'nearest', 'ceil', 'floor', or 'strict', got {value!r}.")
    return value
