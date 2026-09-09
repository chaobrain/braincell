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

"""Adapt the real Cell input stage for fixed-window delivery experiments."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._multi_compartment.cell import Cell, _connection_event_weight, _rewrap_event_template
from braincell.experimental.scheduled_delivery import prepare_delivery
from braincell.experimental.scheduled_events import prepare_events
from braincell.network.event import _round_half_up_steps


class ScheduledDeliveryCell(Cell):
    """Use opt-in experimental scheduled delivery with the production solver.

    Notes
    -----
    Construction matches Cell. Call ``prepare_scheduled_delivery`` after network
    setup and before the first compiled rollout. The plan is immutable for this
    instance; build a new instance to change method, schedule, topology or grid.
    Only physical additive event buffers are supported by this experiment.
    """

    def prepare_scheduled_delivery(self, *, method, dt, n_steps, start_step=0, block_size=128):
        """Install fixed-window input plans before tracing a network rollout.

        Parameters
        ----------
        method : str
            ``current``, ``scan``, ``bucket``, ``direct`` or ``padded``.
        dt : brainunit.Quantity
            Network time step.
        n_steps, start_step : int
            Valid prepared absolute step window.
        block_size : int
            Compact consumption block size.

        Returns
        -------
        tuple
            Prepared plans, for timing and storage reporting.
        """
        if hasattr(self, "_scheduled_delivery_routes"):
            raise RuntimeError("build a fresh experimental Cell to change its delivery plan")
        if method not in ("current", "scan", "bucket", "direct", "padded"):
            raise ValueError("unknown experimental delivery method")
        self._raise_if_not_initialized("prepare_scheduled_delivery")
        routes, plans, arrays = {}, [], []
        store = self._get_synapse_store()
        for layout in self._event_layouts():
            template = self._event_runtime().get_event_buffer(layout.id)
            if not isinstance(template, u.Quantity):
                raise TypeError("experimental adapter requires physical event buffers")
            entries = []
            for connection in self.connections._call_views(scheduled=True):
                if store.layout_id(str(connection.synapse_type[0])) != int(layout.id):
                    continue
                targets = store.runtime_rows(connection.synapse_id).astype(np.int32)
                kwargs = dict(
                    delay=connection.delay,
                    dt=dt,
                    n_steps=n_steps,
                    start_step=start_step,
                    method=method,
                    block_size=block_size,
                )
                if method in ("direct", "padded"):
                    plan = prepare_delivery(
                        connection.source, connection.source_index, targets, n_targets=template.size, **kwargs
                    )
                else:
                    plan = prepare_events(connection.source, connection.source_index, **kwargs)
                entries.append((len(plans), connection, plan, jnp.asarray(targets)))
                plans.append(plan)
                arrays.append(plan.arrays)
            routes[int(layout.id)] = tuple(entries)
        # Read-only State makes arrays dynamic inputs to brainstate's compiled
        # network runner, avoiding closed-over precomputed event trajectories.
        self._scheduled_delivery_arrays = brainstate.ShortTermState(tuple(arrays))
        self._scheduled_delivery_routes = routes
        self._scheduled_delivery_dt = dt
        return tuple(plans)

    def _evaluate_contact_inputs(self, layout, *, t, template, scheduled_only=True):
        routes = getattr(self, "_scheduled_delivery_routes", None)
        if routes is None or not scheduled_only:
            return super()._evaluate_contact_inputs(layout, t=t, template=template, scheduled_only=scheduled_only)
        step = _round_half_up_steps(t.to_decimal(u.ms) / self._scheduled_delivery_dt.to_decimal(u.ms)).astype(jnp.int32)
        output = jnp.zeros(template.size, dtype=u.get_mantissa(template).dtype)
        for index, connection, plan, targets in routes.get(int(layout.id), ()):
            raw_weight = jnp.asarray(_connection_event_weight(template, connection.weight))
            weights = jnp.broadcast_to(raw_weight, (len(connection),))
            arrays = self._scheduled_delivery_arrays.value[index]
            if plan.method in ("direct", "padded"):
                drive = plan.deliver(arrays, step, u.Quantity(weights, template.unit)).to_decimal(template.unit)
            else:
                _, count = plan.query(arrays, jnp.empty(0, dtype=jnp.int32), step)
                drive = jnp.zeros_like(output).at[targets].add(count * weights)
            output = output + drive
        return _rewrap_event_template(template, output.reshape(template.shape))
