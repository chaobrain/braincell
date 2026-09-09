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

"""Check scheduled query semantics, continuation and differentiable consumers."""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.experimental.scheduled_events import METHODS, prepare_events
from braincell.network.event import NetStim
from braincell.synapse import ExpSyn


def source_with_times(times, mask=None):
    source = NetStim(size=len(times), number=0, seed=7)
    object.__setattr__(source, "_event_times_ms", np.asarray(times))
    object.__setattr__(source, "_event_mask", np.ones_like(times, dtype=bool) if mask is None else np.asarray(mask))
    return source


def trajectory(plan, start=None, stop=None):
    start = plan.start_step if start is None else start
    stop = plan.start_step + plan.n_steps if stop is None else stop

    @brainstate.transform.jit
    def run(arrays, state, steps):
        return brainstate.transform.scan(lambda p, k: plan.query(arrays, p, k), state, steps)

    return run(plan.arrays, plan.reset(start), jnp.arange(start, stop))


class QueryTest(unittest.TestCase):
    def test_explicit_example_repetitions_fanout_and_padding(self):
        source = source_with_times([[1, 1.2, 5], [2, 7, 99]], [[True] * 3, [True, True, False]])
        expected = np.zeros((9, 3), dtype=int)
        expected[2] = [2, 1, 0]
        expected[6, 0] = expected[7, 1] = expected[3, 2] = 1
        expected[3, 2] = 2
        expected[7, 2] = 1
        for method in METHODS:
            for block in (1, 2, 128):
                with self.subTest(method=method, block=block):
                    plan = prepare_events(
                        source,
                        [0, 1, 0],
                        delay=np.array([1, 0, 2]) * u.ms,
                        dt=1 * u.ms,
                        n_steps=9,
                        method=method,
                        block_size=block,
                    )
                    np.testing.assert_array_equal(trajectory(plan)[1], expected)

    def test_empty_silent_exhaustion_large_future_and_zero_delay(self):
        for source in (NetStim(size=2, number=0), source_with_times([[0.0, 0.0, 1e20], [8.0, 9.0, 1e20]])):
            results = []
            for method in METHODS:
                plan = prepare_events(source, [0, 1], delay=0 * u.ms, dt=0.25 * u.ms, n_steps=8, method=method)
                results.append(trajectory(plan)[1])
            for result in results[1:]:
                np.testing.assert_array_equal(result, results[0])

    def test_half_ties_and_precision_match_actual_production_expression(self):
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision):
                dtype = np.dtype(f"float{precision}")
                tie = dtype.type(0.5)
                nearby = [np.nextafter(tie, dtype.type(0)), tie, np.nextafter(tie, dtype.type(1)), 0.49, 1.49]
                source = source_with_times([sorted(nearby), sorted(nearby)])
                # round(.49 + .49) must be 1, not round(.49)+round(.49)=0.
                results = []
                for method in METHODS:
                    plan = prepare_events(
                        source,
                        [0, 1],
                        delay=np.array([0, 0.49]) * u.ms,
                        dt=1 * u.ms,
                        n_steps=4,
                        method=method,
                        block_size=2,
                    )
                    results.append(trajectory(plan)[1])
                for result in results[1:]:
                    np.testing.assert_array_equal(result, results[0])

    def test_split_resume_seek_reset_and_nonzero_window(self):
        source = source_with_times([[0.0, 2.0, 2.0, 3.0, 7.0, 20.0]])
        for method in METHODS:
            plan = prepare_events(source, [0], delay=0 * u.ms, dt=1 * u.ms, start_step=2, n_steps=7, method=method)

            @brainstate.transform.jit
            def run(arrays, state, ks):
                return brainstate.transform.scan(lambda p, k: plan.query(arrays, p, k), state, ks)

            full = run(plan.arrays, plan.reset(), jnp.arange(2, 9))[1]
            state, a = run(plan.arrays, plan.reset(), jnp.arange(2, 4))
            _, b = run(plan.arrays, state, jnp.arange(4, 9))
            np.testing.assert_array_equal(jnp.concatenate((a, b)), full)
            np.testing.assert_array_equal(trajectory(plan)[1], full)
            np.testing.assert_array_equal(trajectory(plan, 4)[1], full[2:])
            for invalid in (-1, 9, 2.1):
                with self.assertRaises(ValueError):
                    plan.reset(invalid)

    def test_actual_expsyn_response_weight_and_tau_gradients(self):
        tolerance = 2e-12 if jax.config.jax_enable_x64 else 2e-6
        source = source_with_times([[0.0, 2.0, 2.0, 6.0], [1.0, 2.0, 4.0, 7.0]])
        values = []
        for method in METHODS:
            plan = prepare_events(source, [0, 1], delay=0 * u.ms, dt=1 * u.ms, n_steps=8, method=method, block_size=2)
            node = ExpSyn(size=1, tau=2 * u.ms)
            node.init_state()

            @brainstate.transform.jit
            def response(arrays, weights, tau):
                node.g.value = jnp.zeros(1) * u.uS

                def step(p, k):
                    p, count = plan.query(arrays, p, k)
                    node.apply_events(jnp.sum(count * weights, keepdims=True) * u.uS)
                    # Exact ExpSyn decay; event then decay at every boundary.
                    node.g.value = node.g.value * jnp.exp(-1 / tau)
                    return p, node.g.value.to_decimal(u.uS)

                return brainstate.transform.scan(step, plan.reset(), jnp.arange(8))[1]

            weights = jnp.array([0.2, 0.3])
            result = response(plan.arrays, weights, jnp.array(2.0))
            grads = brainstate.transform.grad(lambda w, tau: response(plan.arrays, w, tau).sum(), argnums=(0, 1))(
                weights, jnp.array(2.0)
            )
            values.append((result, *grads))
        for candidate in values[1:]:
            for actual, expected in zip(candidate, values[0]):
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance / 10)
        # Independently computed impulse response and analytic derivatives.
        arrivals = np.array([[0, 2, 2, 6], [1, 2, 4, 7]])
        ages = np.arange(8)[:, None, None] - arrivals[None, :, :] + 1
        impulse = np.where(ages > 0, np.exp(-np.maximum(ages, 0) / 2), 0)
        np.testing.assert_allclose(values[0][0][:, 0], (impulse.sum(axis=2) * [0.2, 0.3]).sum(axis=1), rtol=tolerance)
        np.testing.assert_allclose(values[0][1], impulse.sum(axis=(0, 2)), rtol=tolerance)
        np.testing.assert_allclose(
            values[0][2], (impulse * ages / 4 * np.array([0.2, 0.3])[None, :, None]).sum(), rtol=tolerance
        )

    def test_float64_expsyn_response_and_gradients(self):
        with brainstate.environ.context(precision=64):
            self.test_actual_expsyn_response_weight_and_tau_gradients()

    def test_invalid_inputs(self):
        source = NetStim(size=1, number=1)
        base = dict(source=source, source_index=[0], delay=0 * u.ms, dt=1 * u.ms, n_steps=2)
        for changes, error in [
            ({"source": None}, TypeError),
            ({"method": "bad"}, ValueError),
            ({"source_index": [0.1]}, TypeError),
            ({"source_index": []}, TypeError),
            ({"source_index": [[0]]}, ValueError),
            ({"source_index": [1]}, ValueError),
            ({"source_index": [-1]}, ValueError),
            ({"delay": 0}, TypeError),
            ({"dt": 1}, TypeError),
            ({"dt": 0 * u.ms}, ValueError),
            ({"dt": np.array([1]) * u.ms}, ValueError),
            ({"dt": np.nan * u.ms}, ValueError),
            ({"dt": 1e-300 * u.ms}, ValueError),
            ({"dt": 1e300 * u.ms}, ValueError),
            ({"dt": 2**31 * u.ms}, ValueError),
            ({"delay": -1 * u.ms}, ValueError),
            ({"delay": np.inf * u.ms}, ValueError),
            ({"delay": np.array([1, 2]) * u.ms}, ValueError),
            ({"n_steps": 0}, ValueError),
            ({"n_steps": 2**31}, ValueError),
            ({"start_step": -1}, ValueError),
            ({"start_step": True}, ValueError),
            ({"block_size": 0}, ValueError),
            ({"n_steps": 2.5}, ValueError),
        ]:
            with self.subTest(changes=changes), self.assertRaises(error):
                prepare_events(**(base | changes))
        for times, mask in [
            ([[2.0, 1.0]], None),
            ([[np.nan]], None),
            ([[-1.0]], None),
            ([[0.0, 1.0]], [[False, True]]),
        ]:
            with self.assertRaises(ValueError):
                prepare_events(**(base | {"source": source_with_times(times, mask)}))

    def test_reject_float32_window_beyond_half_tie_resolution(self):
        # At 2**20 the production float32 four-ULP snapping can map an
        # integer step to the next integer. A cursor indexed by raw steps
        # must reject that grid rather than silently shift delivery.
        with brainstate.environ.context(precision=32):
            source = source_with_times([[float(2**20)]])
            for method in METHODS:
                with self.subTest(method=method), self.assertRaises(ValueError):
                    prepare_events(source, [0], delay=0 * u.ms, dt=1 * u.ms, start_step=2**20, n_steps=2, method=method)
        with brainstate.environ.context(precision=64):
            for method in METHODS:
                plan = prepare_events(
                    source, [0], delay=0 * u.ms, dt=1 * u.ms, start_step=2**20, n_steps=2, method=method
                )
                np.testing.assert_array_equal(trajectory(plan)[1][:, 0], [1, 0])
