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

"""Validate sparse delivery, resource limits and its explicit transpose."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.experimental.scheduled_delivery import PaddingLimitError, prepare_delivery
from braincell.experimental.scheduled_events import prepare_events
from braincell.network.event import NetStim


def _source(times):
    source = NetStim(size=len(times), number=0, seed=7)
    object.__setattr__(source, "_event_times_ms", np.asarray(times))
    object.__setattr__(source, "_event_mask", np.ones(np.shape(times), dtype=bool))
    return source


def _trace(plan, weights, start=None, stop=None):
    start = plan.start_step if start is None else start
    stop = plan.start_step + plan.n_steps if stop is None else stop

    @brainstate.transform.jit
    def run(arrays, w, ks):
        return brainstate.transform.for_loop(lambda k: plan.deliver(arrays, k, w * u.uS).to_decimal(u.uS), ks)

    return run(plan.arrays, weights, jnp.arange(start, stop))


class DeliveryTest(unittest.TestCase):
    def test_counts_targets_duplicates_and_tail_blocks(self):
        source = _source([[0.0, 0.0, 1.0, 2.0, 2.0, 4.0], [0.0, 1.0, 1.0, 3.0, 4.0, 8.0]])
        indices = [0, 1, 0, 1]
        targets = np.array([1, 0, 1, 1])
        delays = np.array([0.0, 0.0, 1.0, 0.49]) * u.ms
        reference = prepare_events(source, indices, delay=delays, dt=1 * u.ms, n_steps=9, method="current")
        weights = jnp.array([0.2, -0.3, 0.4, 0.0])

        @brainstate.transform.jit
        def expected():
            return brainstate.transform.for_loop(
                lambda k: jnp.zeros(3).at[targets].add(reference.query((), reference.reset(), k)[1] * weights),
                jnp.arange(9),
            )

        expected = expected()
        for method in ("direct", "padded"):
            for block in (1, 2, 4, 128):
                with self.subTest(method=method, block=block):
                    plan = prepare_delivery(
                        source,
                        indices,
                        targets,
                        n_targets=3,
                        delay=delays,
                        dt=1 * u.ms,
                        n_steps=9,
                        method=method,
                        block_size=block,
                    )
                    np.testing.assert_allclose(_trace(plan, weights), expected, rtol=2e-6, atol=1e-7)
                    np.testing.assert_array_equal(_trace(plan, jnp.ones(4)), _trace(plan, jnp.ones(4)).astype(int))
                    np.testing.assert_allclose(_trace(plan, weights * 2), expected * 2, rtol=2e-6, atol=1e-7)
                    self.assertLess(plan.preparation["compressed_records"], plan.preparation["arrivals"])

    def test_empty_future_and_stateless_restart(self):
        for source in (_source([[], []]), _source([[100.0, 1e20], [200.0, 1e20]])):
            for method in ("direct", "padded"):
                plan = prepare_delivery(
                    source, [0, 1], [0, 0], n_targets=1, delay=0 * u.ms, dt=1 * u.ms, n_steps=5, method=method
                )
                np.testing.assert_array_equal(_trace(plan, jnp.ones(2)), np.zeros((5, 1)))
        source = _source([[0.0, 2.0, 2.0, 3.0, 7.0, 20.0]])
        for method in ("direct", "padded"):
            plan = prepare_delivery(
                source, [0], [0], n_targets=1, delay=0 * u.ms, dt=1 * u.ms, start_step=2, n_steps=7, method=method
            )
            full = _trace(plan, jnp.ones(1))
            split = jnp.concatenate((_trace(plan, jnp.ones(1), 2, 4), _trace(plan, jnp.ones(1), 4, 9)))
            np.testing.assert_array_equal(full, split)
            np.testing.assert_array_equal(full, _trace(plan, jnp.ones(1)))

    def test_half_ties_both_precisions(self):
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision):
                dtype = np.dtype(f"float{precision}")
                tie = dtype.type(0.5)
                times = sorted([0.49, np.nextafter(tie, dtype.type(0)), tie, np.nextafter(tie, dtype.type(1)), 1.49])
                source = _source([times, times])
                kwargs = dict(delay=np.array([0, 0.49]) * u.ms, dt=1 * u.ms, n_steps=4)
                ref = prepare_events(source, [0, 1], method="scan", **kwargs)
                expected = brainstate.transform.jit(
                    lambda arrays: brainstate.transform.for_loop(
                        lambda k: ref.query(arrays, ref.reset(), k)[1], jnp.arange(4)
                    )
                )(ref.arrays)
                for method in ("direct", "padded"):
                    plan = prepare_delivery(source, [0, 1], [0, 1], n_targets=2, method=method, **kwargs)
                    np.testing.assert_array_equal(_trace(plan, jnp.ones(2)), expected)

    def test_weight_tau_reverse_gradients(self):
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision):
                source = _source([[0.0, 2.0, 2.0, 6.0], [1.0, 2.0, 4.0, 7.0]])
                values = []
                for method in ("direct", "padded"):
                    plan = prepare_delivery(
                        source,
                        [0, 1],
                        [0, 0],
                        n_targets=1,
                        delay=0 * u.ms,
                        dt=1 * u.ms,
                        n_steps=8,
                        method=method,
                        block_size=2,
                    )

                    @brainstate.transform.jit
                    def response(weights, tau):
                        def step(g, k):
                            drive = plan.deliver(plan.arrays, k, weights * u.uS).to_decimal(u.uS)
                            g = (g + drive) * jnp.exp(-1 / tau)
                            return g, g

                        return brainstate.transform.scan(step, jnp.zeros(1), jnp.arange(8))[1].sum()

                    weights = jnp.array([0.2, 0.3])
                    grad = brainstate.transform.grad(response, argnums=(0, 1))(weights, jnp.array(2.0))
                    values.append((response(weights, jnp.array(2.0)), *grad))
                arrivals = np.array([[0, 2, 2, 6], [1, 2, 4, 7]])
                ages = np.arange(8)[:, None, None] - arrivals[None, :, :] + 1
                impulse = np.where(ages > 0, np.exp(-np.maximum(ages, 0) / 2), 0)
                expected = (
                    (impulse * np.array([0.2, 0.3])[None, :, None]).sum(),
                    impulse.sum(axis=(0, 2)),
                    (impulse * ages / 4 * np.array([0.2, 0.3])[None, :, None]).sum(),
                )
                for result in values:
                    for a, b in zip(result, expected):
                        np.testing.assert_allclose(a, b, rtol=3e-6 if precision == 32 else 2e-12)

    def test_padding_limits_are_explicit(self):
        source = _source([[0.0]])
        args = dict(
            source=source,
            source_index=[0],
            target_index=[0],
            n_targets=1,
            delay=0 * u.ms,
            dt=1 * u.ms,
            n_steps=100,
            method="padded",
        )
        with self.assertRaises(PaddingLimitError):
            prepare_delivery(**args)
        with self.assertRaises(PaddingLimitError):
            prepare_delivery(**(args | dict(n_steps=2, max_bytes=1)))
        plan = prepare_delivery(**(args | dict(method="direct", max_bytes=1)))
        self.assertEqual(plan.preparation["arrivals"], 1)

    def test_invalid_inputs_and_units(self):
        args = dict(
            source=_source([[0.0]]),
            source_index=[0],
            target_index=[0],
            n_targets=1,
            delay=0 * u.ms,
            dt=1 * u.ms,
            n_steps=2,
        )
        for changes, exception in [
            ({"method": "x"}, ValueError),
            ({"n_targets": 0}, ValueError),
            ({"n_targets": True}, ValueError),
            ({"target_index": [0.5]}, TypeError),
            ({"target_index": [[0]]}, TypeError),
            ({"target_index": [-1]}, ValueError),
            ({"target_index": [1]}, ValueError),
            ({"target_index": [0, 0]}, ValueError),
            ({"max_bytes": 0}, ValueError),
            ({"max_padding_ratio": 0.5}, ValueError),
            ({"max_padding_ratio": float("nan")}, ValueError),
            ({"delay": 0}, TypeError),
            ({"start_step": 2**20}, ValueError),
        ]:
            with self.subTest(changes=changes), self.assertRaises(exception):
                prepare_delivery(**(args | changes))
        plan = prepare_delivery(**args)
        with self.assertRaises(TypeError):
            plan.deliver(plan.arrays, 0, jnp.ones(1))
        with self.assertRaises(ValueError):
            plan.deliver(plan.arrays, 0, jnp.ones(2) * u.uS)
