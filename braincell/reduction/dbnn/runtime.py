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

"""Current reduction-runtime adapter for DBNN models."""

from __future__ import annotations

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.mech import ScalarEventInput
from braincell.reduction.core import ReductionContext, ReductionInputs, ReductionModel, ReductionOutput
from braincell.reduction.dbnn.model import DBNN, DBNNGIF


class DBNNReduction(ReductionModel):
    """Run one recurrent DBNN or DBNN-GIF through a Cell reduction runtime.

    The model channel order is the initial ``ReductionContext.synapses`` order.
    A model is therefore valid only for that exact logical-synapse layout.
    """

    def __init__(self, model: DBNN | DBNNGIF) -> None:
        if not isinstance(model, (DBNN, DBNNGIF)):
            raise TypeError("model must be a DBNN or DBNNGIF.")
        if model.mode != "r":
            raise ValueError("DBNNReduction requires a model with mode='r'.")
        self.model = model
        self._context = None
        self._channel_by_synapse_id = None
        self._reference_weight_by_synapse_id = None
        self._population_by_group = None
        self._channel_by_group = None
        self._reference_weight_by_group = None

    def init_state(self, context: ReductionContext, batch_size=None) -> ReductionOutput:
        if batch_size is not None:
            raise NotImplementedError("DBNNReduction does not yet support Cell batch execution.")
        expected_synapses = context.population_size * self.model.n_channels
        if len(context.synapses) != expected_synapses:
            raise ValueError(
                f"DBNN requires {self.model.n_channels} logical channels per Cell member but Cell exposes "
                f"{len(context.synapses)} synapses for population size {context.population_size}."
            )
        expected_channels = tuple(range(self.model.n_channels))
        for population_index in range(context.population_size):
            channels = tuple(
                item.synapse_index for item in context.synapses if item.population_index == population_index
            )
            if tuple(sorted(channels)) != expected_channels:
                raise ValueError(
                    f"DBNN channels for population member {population_index} do not match its logical synapse layout."
                )
        for group in context.input_groups:
            if not isinstance(group.event_input, ScalarEventInput):
                raise TypeError(
                    "DBNNReduction supports only ScalarEventInput; "
                    f"{group.synapse_type!r} declares {type(group.event_input).__name__}."
                )
        self._context = context
        self._channel_by_synapse_id = {int(item.id): int(item.synapse_index) for item in context.synapses}
        self._reference_weight_by_synapse_id = {
            int(item.id): self.model.layout.reference_weight_us(item.synapse_index)
            for item in context.synapses
        }
        self._population_by_group = tuple(
            jnp.asarray(group.population_index, dtype=jnp.int32)
            for group in context.input_groups
        )
        self._channel_by_group = tuple(
            jnp.asarray(
                [self._channel_by_synapse_id[int(value)] for value in group.synapse_id],
                dtype=jnp.int32,
            )
            for group in context.input_groups
        )
        self._reference_weight_by_group = tuple(
            jnp.asarray(
                [self._reference_weight_by_synapse_id[int(value)] for value in group.synapse_id],
                dtype=jnp.float32,
            )
            for group in context.input_groups
        )
        self.model.init_state(batch_size=context.population_size)
        return self._output()

    def update(self, inputs: ReductionInputs) -> ReductionOutput:
        self._require_initialized()
        events = jnp.zeros((self._context.population_size, self.model.n_channels), dtype=jnp.float32)
        groups = inputs.groups
        if groups:
            values = jnp.concatenate(
                tuple(
                    jnp.asarray(u.get_magnitude(group.payload), dtype=events.dtype).reshape(-1)
                    for group in groups
                )
            )
            population = jnp.concatenate(self._population_by_group)
            channel = jnp.concatenate(self._channel_by_group)
            reference_weight = jnp.concatenate(self._reference_weight_by_group)
            events = events.at[population, channel].add(values / reference_weight)
        self.model.update(events)
        return self._output()

    def reset_state(self, batch_size=None) -> ReductionOutput:
        self._require_initialized()
        if batch_size is not None:
            raise NotImplementedError("DBNNReduction does not yet support Cell batch execution.")
        self.model.reset_state(batch_size=self._context.population_size)
        return self._output()

    def reset(self) -> None:
        self._context = None
        self._channel_by_synapse_id = None
        self._reference_weight_by_synapse_id = None
        self._population_by_group = None
        self._channel_by_group = None
        self._reference_weight_by_group = None

    def _output(self) -> ReductionOutput:
        values = {"voltage": self.model.voltage.value * u.mV}
        if isinstance(self.model, DBNNGIF):
            values["threshold"] = self.model.threshold.value * u.mV
        return ReductionOutput(values=values, event=self.model.spike.value.astype(jnp.int32))

    def _require_initialized(self) -> None:
        if (
            self._context is None
            or self._channel_by_synapse_id is None
            or self._reference_weight_by_synapse_id is None
        ):
            raise RuntimeError("DBNNReduction requires init_state() first.")


__all__ = ["DBNNReduction"]
