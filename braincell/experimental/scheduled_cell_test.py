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

"""Check the adapter against the real Network timestep and Cell solver."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.performance.synapse_events.benchmark import Case, build_workload
from benchmarks.performance.synapse_events.delivery_benchmark import full_runner
from braincell.experimental.scheduled_cell import ScheduledDeliveryCell


class ScheduledCellTest(unittest.TestCase):
    def test_solver_traces_layouts_declarations_and_reset(self):
        cfg = Case(n=2, m=3, number=8, duration_ms=4.0, dt_ms=0.25, rate_hz=2000, pattern="sync")
        for layout, declaration in (("independent", "batched"), ("shared", "per_cell")):
            case = replace(cfg, layout=layout, declaration=declaration)
            reference = None
            for method in ("production", "current", "scan", "bucket", "direct", "padded"):
                with self.subTest(layout=layout, method=method):
                    work = build_workload(case, cell_type=None if method == "production" else ScheduledDeliveryCell)
                    if method != "production":
                        work["cell"].prepare_scheduled_delivery(method=method, dt=case.dt_ms * u.ms, n_steps=case.steps)
                    reset, run = full_runner(work, record=True)
                    jax.block_until_ready(reset())
                    result = jax.device_get(run(jnp.arange(case.steps))[2])
                    if reference is None:
                        reference = result
                    for actual, expected in zip(result, reference):
                        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-6)
                    self.assertGreater(np.max(result[1]), 0)
                    jax.block_until_ready(reset())
                    again = jax.device_get(run(jnp.arange(case.steps))[2])
                    for a, b in zip(result, again):
                        np.testing.assert_array_equal(a, b)

    def test_configuration_is_immutable_and_missing_layout_is_zero(self):
        cfg = Case(n=1, m=2, number=3, duration_ms=2.0, dt_ms=0.25, rate_hz=2000, pattern="sync")
        work = build_workload(cfg, cell_type=ScheduledDeliveryCell)
        cell = work["cell"]
        with self.assertRaises(ValueError):
            cell.prepare_scheduled_delivery(method="invalid", dt=0.25 * u.ms, n_steps=8)
        cell.prepare_scheduled_delivery(method="direct", dt=0.25 * u.ms, n_steps=8)
        with self.assertRaises(RuntimeError):
            cell.prepare_scheduled_delivery(method="direct", dt=0.25 * u.ms, n_steps=8)
        with brainstate.environ.context(dt=0.25 * u.ms):
            live = cell._evaluate_contact_inputs(
                work["layout"],
                t=0 * u.ms,
                template=cell.runtime.get_event_buffer(work["layout"].id),
                scheduled_only=False,
            )
        np.testing.assert_array_equal(u.get_mantissa(live), np.zeros(live.shape))

    def test_trainable_weights_tau_and_unconnected_layout(self):
        import braincell as bc
        from braincell.filter import AllRegion, LocsetMask

        results = []
        for method in ("production", "direct", "padded"):
            cell_type = bc.Cell if method == "production" else ScheduledDeliveryCell
            branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
            cell = cell_type(
                bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV
            )
            cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
            site = LocsetMask.from_columns(np.array([0]), np.array([0.5]))
            cell.place(site, bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV))
            cell.place(site, bc.mech.Synapse("Exp2Syn", name="other", tau1=0.5 * u.ms, tau2=3.0 * u.ms, e=0.0 * u.mV))
            cell.synapses["exp"].trainable(tau=bc.trainable.scale(name="tau_scale"))
            net = bc.Network("gradient_delivery")
            post = net.add_population("post", cell)
            stim = net.add_population("stim", bc.NetStim(size=1, start=0 * u.ms, interval=0.5 * u.ms, number=8))
            net.connect(
                "drive",
                source=stim.event_outputs["spike"],
                synapse=post.synapses["exp"],
                weight=0.001 * u.uS,
                delay=0 * u.ms,
            )
            cell.connections["drive"].trainable(weight=bc.trainable.scale(name="weight_scale"))
            net.prepare_run(dt=0.25 * u.ms, event_backend="scatter")
            if method != "production":
                cell.prepare_scheduled_delivery(method=method, dt=0.25 * u.ms, n_steps=16)
            layout_id = cell.synapses["exp"]._store.layout_id("ExpSyn")
            node = cell.runtime.get_runtime_node(layout_id)

            @brainstate.transform.jit
            def observe():
                net.reset_state()
                return brainstate.transform.for_loop(
                    lambda _: (net.update(), node.g.value.to_decimal(u.uS).sum())[1], jnp.arange(16)
                ).sum()

            value = observe()
            grad = brainstate.transform.jit(
                brainstate.transform.grad(observe, grad_states=net.trainables.parameters().states())
            )()
            self.assertNotEqual(float(grad["post.weight_scale"]), 0.0)
            self.assertNotEqual(float(grad["post.tau_scale"]), 0.0)
            results.append((value, grad))
            params = net.trainables.parameters().physical_values()
            params["post.weight_scale"] = 2.0
            net.trainables.parameters().set_physical_values(params)
            np.testing.assert_allclose(observe(), 2 * value, rtol=3e-5)
        for value, grad in results[1:]:
            np.testing.assert_allclose(value, results[0][0], rtol=3e-5)
            for key in grad:
                np.testing.assert_allclose(grad[key], results[0][1][key], rtol=3e-5)

    def test_adapter_rejects_dimensionless_buffers(self):
        cfg = Case(n=1, m=1, duration_ms=2.0, dt_ms=0.25)
        work = build_workload(cfg, cell_type=ScheduledDeliveryCell)
        cell = work["cell"]
        with patch.object(type(cell.runtime), "get_event_buffer", return_value=jnp.zeros(1)):
            with self.assertRaises(TypeError):
                cell.prepare_scheduled_delivery(method="direct", dt=0.25 * u.ms, n_steps=8)
