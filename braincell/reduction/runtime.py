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

"""Lightweight packed synapse-input runtime for reduced Cells."""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np

from braincell.mech import NoEventInput, ScalarEventInput, TriggerEventInput
from braincell.reduction.core import (
    ReductionContext,
    ReductionInputGroup,
    ReductionInputGroupSchema,
    ReductionInputs,
)

from braincell.reduction.runtime_utils import _build_reduction_context

__all__ = ["ReductionInputLayout", "ReductionInputRuntime", "build_reduction_input_runtime"]


@dataclass(frozen=True)
class ReductionInputLayout:
    """Describe one packed event-buffer layout needed by connection lowering."""

    id: int
    kind: str
    n_active: int
    placement_index: np.ndarray
    synapse_index: np.ndarray | None
    schema: ReductionInputGroupSchema


@dataclass
class ReductionInputRuntime:
    """Own the common event-input boundary for every reduced Cell.

    The default adapter retains the detailed synapse payload contract,
    including physical units and weighted aggregation. Model adapters may
    override connection routing and delivery while retaining the same public
    Connection and event-source declarations.

    Adapters own input preparation, enqueueing, and queue reset. Pending
    events survive consecutive runs and are cleared by ``reset_state``.
    """

    layouts: tuple[ReductionInputLayout, ...]
    event_buffers: dict[int, brainstate.State]
    context: ReductionContext

    def connection_route(self, connection):
        """Map declared contacts to this adapter's input rows and weights.

        Parameters
        ----------
        connection : braincell.network.connection.ConnectionView
            Contacts targeting one synapse type on the owning Cell.

        Returns
        -------
        tuple
            Layout id, one input-row index per contact, and contact weights.
            The default keeps physical synapse weights. An adapter may encode
            a weight into its row identity and return unit event counts.
        """
        store = self.context.cell._get_synapse_store()
        layout_id = store.layout_id(str(connection.synapse_type[0]))
        return layout_id, store.runtime_rows(connection.synapse_id).astype(np.int32), connection.weight

    def prepare_delivery(self, blocks):
        """Prepare incoming live routes and return those for Network delivery.

        Parameters
        ----------
        blocks : tuple of braincell.network.lowering.ConnectionBlock
            Incoming routes with this adapter's rows, weights and quantized
            delays. Their event sources may be detailed or reduced Cells.

        Returns
        -------
        tuple of braincell.network.lowering.ConnectionBlock
            Routes still handled by Network. The default returns all blocks.
            Adapters retain any privately handled routes and enqueue their
            current source events in ``enqueue_events``.

        Notes
        -----
        Called once per Network run setup in the automatic event backend.
        An explicit Network event backend uses ordinary delivery instead.
        """
        return blocks

    def scheduled_inputs(self, layout, *, t, template):
        """Return scheduled arrivals for one layout at the current step.

        Adapters consuming schedules internally return a zero payload here.
        The default evaluates the Cell's scheduled Connections, preserving
        weights, delays, and duplicate events.

        Parameters
        ----------
        layout : ReductionInputLayout
            Packed input layout receiving the arrivals.
        t : brainunit.Quantity
            Current simulation time.
        template : array or brainunit.Quantity
            Buffer defining the payload shape, dtype, and physical unit.

        Returns
        -------
        array or brainunit.Quantity
            Current scheduled payload matching the template.
        """
        return self.context.cell._evaluate_contact_inputs(layout, t=t, template=template, scheduled_only=True)

    def take_inputs(self) -> ReductionInputs:
        """Snapshot all current payloads and clear their backing buffers."""
        groups = tuple(
            ReductionInputGroup(layout.schema, self.get_event_buffer(layout.id))
            for layout in self.layouts
            if layout.id in self.event_buffers
        )
        self.clear_event_buffers()
        return ReductionInputs(groups)

    def enqueue_events(self) -> None:
        """Queue private live arrivals after all populations have updated.

        The default has no private queue. Zero-delay events target the next
        postsynaptic update, matching ordinary Network delivery.
        """

    def reset_state(self) -> None:
        """Clear input buffers and private queues before the model is reset."""
        self.clear_event_buffers()

    def get_event_buffer(self, layout_id: int):
        """Return the current payload for one input layout."""
        return self.event_buffers[int(layout_id)].value

    def clear_event_buffer(self, layout_id: int) -> None:
        """Set one input buffer to zero without changing its unit or dtype."""
        state = self.event_buffers[int(layout_id)]
        state.value = u.math.zeros_like(state.value)

    def clear_event_buffers(self) -> None:
        """Clear every packed input buffer."""
        for layout_id in self.event_buffers:
            self.clear_event_buffer(layout_id)


def build_reduction_input_runtime(cell) -> ReductionInputRuntime:
    """Build a packed input-only runtime from the Cell's current synapses."""
    # 1. Bind the declared synapses to common reduction input schemas.
    context = _build_reduction_context(cell)
    store = cell._get_synapse_store()
    layouts = []
    event_buffers = {}

    # 2. Allocate each layout's payload with its physical unit and event dtype.
    for schema in context.input_groups:
        rows = store.row_indices(schema.synapse_id)
        layout = ReductionInputLayout(
            id=schema.layout_id,
            kind=f"synapse:{schema.synapse_type}",
            n_active=schema.size,
            placement_index=np.asarray(store.placement_id[rows], dtype=np.int64),
            synapse_index=schema.synapse_id,
            schema=schema,
        )
        layouts.append(layout)
        event_input = schema.event_input
        if isinstance(event_input, ScalarEventInput):
            zero = u.Quantity(np.zeros((schema.size,), dtype=float), event_input.unit)
        elif isinstance(event_input, TriggerEventInput):
            zero = np.zeros((schema.size,), dtype=np.int32)
        elif isinstance(event_input, NoEventInput):
            continue
        else:
            raise TypeError(f"Unsupported event input {type(event_input).__name__!r} for {schema.synapse_type!r}.")
        event_buffers[schema.layout_id] = brainstate.ShortTermState(zero)
    return ReductionInputRuntime(tuple(layouts), event_buffers, context)
