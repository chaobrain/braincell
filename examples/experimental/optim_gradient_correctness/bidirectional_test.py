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

"""Coupled population gradients must include both directions and all roots."""

import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from examples.experimental.optim_gradient_correctness.bidirectional import (
    ATOL,
    DT,
    RTOL,
    STEPS,
    build,
    engine_for,
    fit,
    gradient_comparison,
)


class BidirectionalTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(brainstate.environ.context(dt=DT, precision=64))

    def test_activity_and_independent_population_contact_coordinates(self):
        experiment = build()
        values = experiment.network.trainables.parameters().physical_values()
        for population, size in (("A", 2), ("B", 3)):
            for field in ("gmax", "shift", "ion", "reversal", "threshold"):
                self.assertEqual(values[f"{population}.{field}"].shape, (size,))
            self.assertEqual(values[f"{population}.weight"].shape, (6,))
        activity = brainstate.transform.jit(experiment.rollout)()
        spikes = np.asarray(activity["spike"])
        conductance = np.asarray(activity["conductance"])
        self.assertTrue(np.all(spikes.sum(axis=0) >= 2), spikes.sum(axis=0))
        for member in range(5):
            arrivals = np.flatnonzero(conductance[:, member] > 0.0)
            self.assertGreater(len(arrivals), 0)
            self.assertGreater(spikes[arrivals[0] + 1 :, member].sum(), 0.0)

    def test_all_parameter_gradients_across_losses_delays_and_backends(self):
        for loss, delay, backend in (
            ("joint", "heterogeneous", "scatter"),
            ("A", "positive", "scatter"),
            ("B", "heterogeneous", "brainevent"),
            ("spike", "zero", "brainevent"),
        ):
            with self.subTest(loss=loss, delay=delay, backend=backend):
                result = gradient_comparison(loss_kind=loss, delay=delay, backend=backend)
                self.assertTrue(all(np.isfinite(g).all() for g in result["gradients"].values()))
                if loss in ("A", "B"):
                    upstream = "B" if loss == "A" else "A"
                    for field in ("gmax", "threshold"):
                        self.assertGreater(np.linalg.norm(result["gradients"][f"{upstream}.{field}"]), 1e-10)

    def test_detached_events_preserve_forward_but_remove_cross_population_gradients(self):
        traces = []
        gradients = []
        for detached in (False, True):
            experiment = build(detached=detached)
            traces.append(brainstate.transform.jit(experiment.rollout)())
            losses = {}
            for loss in ("A", "B"):
                engine = engine_for(experiment, "bptt", loss_kind=loss)
                target = jnp.full((STEPS, 5), -60.0)
                engine.prepare(target[0])
                losses[loss] = brainstate.transform.jit(engine)(target).gradients
            gradients.append(losses)
        for key in traces[0]:
            np.testing.assert_array_equal(traces[0][key], traces[1][key])
        for loss, upstream in (("A", "B"), ("B", "A")):
            for field in ("gmax", "threshold"):
                name = f"{upstream}.{field}"
                self.assertGreater(np.linalg.norm(gradients[0][loss][name]), 1e-10)
                np.testing.assert_array_equal(gradients[1][loss][name], 0.0)

    def test_shared_root_gradient_is_sum_of_independent_roots(self):
        independent = gradient_comparison(grouped=True)
        shared = gradient_comparison(grouped=True, shared=True)
        self.assertNotIn("B.gmax", shared["gradients"])
        np.testing.assert_allclose(
            shared["gradients"]["A.gmax"],
            independent["gradients"]["A.gmax"] + independent["gradients"]["B.gmax"],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_repeated_compiled_reset_and_root_updates(self):
        experiment = build(grouped=True)
        target = jnp.full((160, 5), -60.0)
        engine = engine_for(experiment, "rtrl")
        engine.prepare(target[0])
        run = brainstate.transform.jit(engine)
        first, repeated = run(target), run(target)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(repeated)):
            np.testing.assert_array_equal(a, b)
        parameters = experiment.network.trainables.parameters()
        original = parameters.physical_values()
        changed = dict(original)
        changed["A.weight"] = changed["A.weight"] * 1.1
        changed["B.tau2"] = changed["B.tau2"] * 0.9
        parameters.set_physical_values(changed)
        self.assertNotEqual(float(run(target).loss), float(first.loss))
        parameters.set_physical_values(original)
        restored = run(target)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(restored)):
            np.testing.assert_array_equal(a, b)

    def test_prefix_gradients_and_queue_cross_population_sensitivities(self):
        experiment = build(grouped=True)
        trajectory = brainstate.transform.jit(experiment.rollout)()
        spikes = np.asarray(trajectory["spike"])
        first_a = int(np.flatnonzero(spikes[:, :2].sum(axis=1))[0])
        first_b = int(np.flatnonzero(spikes[:, 2:].sum(axis=1))[0])
        at = tuple(sorted({0, first_a, first_a + 8, first_b, first_b + 8, STEPS - 1}))
        target = jnp.full((STEPS, 5), -60.0)
        engine = engine_for(experiment, "rtrl")
        engine.prepare(target[0])
        diagnostic = brainstate.transform.jit(lambda: engine.diagnose(target, at=at))()
        coordinates = engine._parameter_coordinates
        reference = build(grouped=True)
        bptt = engine_for(reference, "bptt")
        bptt.prepare(target[0])
        for sample, index in enumerate(at):
            result = brainstate.transform.jit(bptt)(target[: index + 1])
            np.testing.assert_allclose(
                diagnostic.prefix_gradients[sample],
                coordinates.flatten({name: result.gradients[name] for name in coordinates.names}),
                rtol=RTOL,
                atol=ATOL,
            )
        states = {id(state): i for i, state in enumerate(engine._functional_step.state_trace.states)}
        for receiver, sender in (("A", "B"), ("B", "A")):
            state_index = states[id(experiment.cells[receiver].V)]
            directions = coordinates.slices[coordinates.names.index(f"{sender}.gmax")]
            sensitivity = np.asarray(brainstate.maybe_state(diagnostic.sensitivity[state_index]).mantissa)
            np.testing.assert_array_equal(sensitivity[0, directions], 0.0)
            self.assertGreater(np.linalg.norm(sensitivity[1:, directions]), 1e-10)
        delivery = experiment.network._prepared_run[1].delivery_state
        for queue in delivery.ring_buffers:
            sensitivity = diagnostic.sensitivity[states[id(queue)]]
            self.assertGreater(np.linalg.norm(np.asarray(sensitivity.mantissa)), 1e-10)
        roots = tuple(state.value for state in engine.parameter_states.values())
        shapes = tuple(np.shape(x) for x in jax.tree.leaves(engine._initial_full_carry(roots)))
        brainstate.transform.jit(engine)(target[:100])
        short_shapes = tuple(np.shape(x) for x in jax.tree.leaves(engine._initial_full_carry(roots)))
        self.assertEqual(shapes, short_shapes)

    def test_both_gradient_methods_train_both_populations_and_connections(self):
        for method in ("bptt", "rtrl"):
            with self.subTest(method=method):
                result = fit(method=method)
                self.assertTrue(np.isfinite(result["history"]).all())
                self.assertLess(result["final_mse"], result["initial_mse"] * 0.1)
                for population in ("A", "B"):
                    for field in ("gmax", "ion", "reversal", "weight"):
                        name = f"{population}.{field}"
                        self.assertNotEqual(float(result["initial_roots"][name]), float(result["fitted_roots"][name]))
