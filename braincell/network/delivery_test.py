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

"""Tests for :mod:`braincell.network.delivery`.

Only the behaviours this module exposes directly are covered here; the
ring-buffer arrival machinery is exercised end-to-end from ``engine_test.py``
via :meth:`Network.run`."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np


class ZerosLikeTest(unittest.TestCase):
    """The one shape helper every buffer in the module goes through.

    Arrival vectors, ring-buffer queues, and the scatter accumulator inside
    a delivery op each used to inline this ``isinstance`` fork; the unit
    branch is the half that a plain ``jnp.zeros`` would silently drop.
    """

    def test_a_quantity_payload_keeps_its_unit(self) -> None:
        from braincell.network.delivery import zeros_like

        result = zeros_like(np.asarray([1.0, 2.0]) * u.uS, shape=(3, 4))

        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(u.get_unit(result), u.uS)
        np.testing.assert_array_equal(np.asarray(result.mantissa), np.zeros((3, 4)))

    def test_a_plain_payload_keeps_its_dtype(self) -> None:
        from braincell.network.delivery import zeros_like

        result = zeros_like(np.asarray([1.0, 2.0], dtype=np.float32), shape=(2,))

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, np.float32)
        self.assertNotIsInstance(result, u.Quantity)


class DeliveryTest(unittest.TestCase):
    def test_brainevent_batched_weight_and_event_derivatives(self) -> None:
        import brainevent

        from braincell.network.delivery import DeliveryBlock, make_delivery_op

        # Unsorted presynaptic indices, repeated destinations, and a
        # presynaptic member without contacts exercise the CSR conversion.
        pre_index = np.asarray([2, 0, 2, 0, 2])
        target_index = np.asarray([0, 0, 1, 1, 0])
        for weights in (jnp.asarray([0.2, 0.3, 0.4, 0.5, 0.7]), jnp.asarray([0.2])):
            for unit in (u.UNITLESS, u.uS):
                with self.subTest(weights=weights.shape, unit=unit):
                    source = SimpleNamespace(n_active=2)

                    def evaluate(weight, event, backend):
                        block = DeliveryBlock(source, 0, pre_index, target_index, weight * unit)
                        result = make_delivery_op(block, pre_size=3, backend=backend)(event)
                        self.assertEqual(u.get_unit(result), unit)
                        return u.get_mantissa(result)

                    event = jnp.asarray([1.0, 0.25, 0.5])
                    directions = jnp.eye(weights.size + event.size)
                    results = []
                    for backend in ("scatter", "brainevent"):
                        with patch.object(brainevent, "csrmv", wraps=brainevent.csrmv) as csrmv:
                            call = lambda w, e: evaluate(w, e, backend)
                            primal, linear = jax.linearize(call, weights, event)
                            tangent = jax.jit(jax.vmap(linear))(
                                directions[:, : weights.size], directions[:, weights.size :]
                            )
                            reverse = jax.jit(jax.grad(lambda w, e: call(w, e).sum(), argnums=(0, 1)))(weights, event)
                            bool_event = jax.jit(lambda w: call(w, jnp.asarray([True, False, True])))(weights)
                            bool_gradient = jax.jit(
                                jax.grad(lambda w: call(w, jnp.asarray([True, False, True])).sum())
                            )(weights)
                        self.assertEqual(csrmv.called, backend == "brainevent")
                        results.append((primal, tangent, reverse, bool_event, bool_gradient))
                    for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_event_backend_selection(self) -> None:
        import braincell.network.delivery as delivery

        self.assertEqual(delivery.resolve_event_backend("auto"), "scatter")
        self.assertEqual(delivery.resolve_event_backend("scatter"), "scatter")
        self.assertEqual(delivery.resolve_event_backend("brainevent"), "brainevent")

    def test_event_backend_selection_without_csrmv(self) -> None:
        import braincell.network.delivery as delivery

        with patch.dict("sys.modules", {"brainevent": ModuleType("brainevent")}):
            self.assertEqual(delivery.resolve_event_backend("auto"), "scatter")
            self.assertEqual(delivery.resolve_event_backend("scatter"), "scatter")
            with self.assertRaisesRegex(RuntimeError, "brainevent.csrmv.*event_backend='scatter'"):
                delivery.resolve_event_backend("brainevent")

    def test_event_backend_selection_without_brainevent(self) -> None:
        import braincell.network.delivery as delivery

        # A ``None`` entry makes ``import brainevent`` raise ImportError.
        with patch.dict("sys.modules", {"brainevent": None}):
            self.assertEqual(delivery.resolve_event_backend("auto"), "scatter")
            with self.assertRaisesRegex(RuntimeError, "brainevent package.*event_backend='scatter'"):
                delivery.resolve_event_backend("brainevent")



def _one_route_per_block(blocks):
    """Routes as ``delivery_blocks(..., group_by_delay=False)`` built them before CELL-18b."""
    from braincell.network.delivery import DeliveryBlock

    return tuple(
        DeliveryBlock(
            source=block,
            delay_steps=np.asarray(block.delay_steps, dtype=np.int32),
            pre_index=np.asarray(block.pre_index, dtype=np.int32),
            flat_target_index=np.asarray(block.synapse_index, dtype=np.int32),
            weight=block.weight,
        )
        for block in blocks
    )

class _RecordingCell:
    """Stands in for a Cell: keeps what delivery writes during one step."""

    def __init__(self) -> None:
        self.arrivals, self.drives = {}, {}

    def _write_event_arrival(self, layout_id, arrival) -> None:
        self.arrivals[int(layout_id)] = arrival

    def _apply_synapse_layout_event_drive(self, layout_id, drive) -> None:
        self.drives[int(layout_id)] = drive


class _StepSource:
    """A live source whose event vector at each step comes from a table."""

    def __init__(self, table) -> None:
        self.table = jnp.asarray(table)
        self.size = self.table.shape[1]
        self.step = None

    def current_event_count(self, ids):
        return self.table[self.step][jnp.asarray(ids)]


class DeliveryOrderTest(unittest.TestCase):
    """CELL-18b: every queue receives the same values, summed in the same order.

    Each case runs a compiled loop of queue writes, zero-delay drives and
    enqueues over the routes ``delivery_blocks`` returns and over one route
    per block, and compares every value written. The weights span seven
    orders of magnitude, so a float32 sum taken in another order differs in
    its last bits.
    """

    N_STEPS = 12

    def _block(self, source, post, layout, pre_index, target, delay, weight, n_active=6):
        from braincell.network.lowering import ConnectionBlock

        pre_index = np.asarray(pre_index, dtype=np.int32)
        return ConnectionBlock(
            pre_population="pre",
            post_population=post,
            synapse="syn",
            layout_id=layout,
            n_active=n_active,
            pre_index=pre_index,
            post_index=np.asarray(target, dtype=np.int32),
            synapse_index=np.asarray(target, dtype=np.int32),
            weight=weight,
            delay_steps=np.broadcast_to(np.asarray(delay, dtype=np.int32), pre_index.shape).copy(),
            event_source=source,
        )

    def _weights(self, rng, n, unit=u.uS):
        return u.Quantity(np.asarray(10.0 ** rng.uniform(-6, 1, n), dtype=np.float32), unit)

    def _deliver(self, routed, blocks, sources):
        from braincell.network.delivery import (
            advance_delivery_state,
            apply_immediate_events,
            create_delivery_state,
            enqueue_future_events,
            write_arrivals,
        )

        state = create_delivery_state(routed, delivery_ops=(None,) * len(routed))
        cells = {name: _RecordingCell() for name in {block.post_population for block in blocks}}
        populations = {name: SimpleNamespace(cell=cell) for name, cell in cells.items()}

        def step(index):
            for source in sources:
                source.step = index
            for cell in cells.values():
                cell.arrivals, cell.drives = {}, {}
            write_arrivals(state, populations=populations)
            apply_immediate_events(routed, state, populations=populations)
            enqueue_future_events(routed, state)
            advance_delivery_state(state)
            return (
                {(name, layout): value for name, cell in cells.items() for layout, value in cell.arrivals.items()},
                {(name, layout): value for name, cell in cells.items() for layout, value in cell.drives.items()},
            )

        return brainstate.transform.for_loop(step, jnp.arange(self.N_STEPS))

    def _assert_same_delivery(self, blocks, sources) -> None:
        from braincell.network.delivery import delivery_blocks

        expected = self._deliver(_one_route_per_block(blocks), blocks, sources)
        actual = self._deliver(delivery_blocks(blocks, group_by_delay=False), blocks, sources)
        written = 0
        for expected_values, actual_values in zip(expected, actual):
            self.assertEqual(set(actual_values), set(expected_values))
            for key, values in actual_values.items():
                self.assertEqual(u.get_unit(values), u.get_unit(expected_values[key]))
                values = np.asarray(u.get_mantissa(values))
                np.testing.assert_array_equal(values, np.asarray(u.get_mantissa(expected_values[key])))
                written += int(np.count_nonzero(values))
        self.assertGreater(written, 0)

    def test_blocks_of_one_queue_keep_their_order(self) -> None:
        rng = np.random.default_rng(18)
        spikes = (rng.random((self.N_STEPS, 4)) < 0.6).astype(np.float32)
        first, second = _StepSource(spikes), _StepSource(spikes[::-1])
        cases = {
            "disjoint targets, delayed": [
                self._block(first, "post", 0, [0, 1, 2], [0, 1, 2], 2, self._weights(rng, 3)),
                self._block(first, "post", 0, [3, 1, 0], [3, 4, 5], 3, self._weights(rng, 3)),
            ],
            "shared targets, mixed delays": [
                self._block(first, "post", 0, [0, 1, 2, 3, 0], [0, 0, 1, 1, 0], [0, 2, 0, 2, 0], self._weights(rng, 5)),
                self._block(first, "post", 0, [3, 2, 1, 0, 1], [0, 0, 1, 5, 0], [0, 2, 0, 0, 2], self._weights(rng, 5)),
                self._block(first, "post", 0, [1, 1, 2], [0, 0, 1], 0, self._weights(rng, 3)),
            ],
            "interleaved sources and queues": [
                self._block(first, "post", 0, [0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1], self._weights(rng, 4)),
                self._block(second, "post", 0, [0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1], self._weights(rng, 4)),
                self._block(first, "post", 1, [0, 2], [2, 2], 0, self._weights(rng, 2)),
                self._block(first, "post", 0, [3, 0, 1], [1, 0, 0], [1, 0, 0], self._weights(rng, 3)),
                self._block(first, "other", 0, [1, 2], [3, 3], [0, 4], self._weights(rng, 2)),
            ],
            "weights in two units": [
                self._block(first, "post", 0, [0, 1], [0, 0], [0, 1], self._weights(rng, 2)),
                self._block(first, "post", 0, [2, 3], [0, 0], [0, 1], self._weights(rng, 2, unit=u.nS)),
                self._block(first, "post", 0, [1, 3], [0, 0], [0, 1], self._weights(rng, 2)),
            ],
        }
        for name, blocks in cases.items():
            with self.subTest(name):
                self._assert_same_delivery(blocks, (first, second))

    def test_two_contacts_summing_into_a_synapse_an_earlier_block_reached(self) -> None:
        # In float32 1 + (2**-24 + 2**-24) is 1 + 2**-23, while adding the two
        # halves of an ulp to 1 one at a time rounds back to 1 each time. The
        # compiled zero-delay path adds them one at a time: XLA turns
        # ``drive + zeros.at[i].add(x)`` into ``drive.at[i].add(x)``.
        always = _StepSource(np.ones((self.N_STEPS, 2), dtype=np.float32))
        half_ulp = np.float32(2.0**-24)
        for delay in (0, 3):
            with self.subTest(delay=delay):
                blocks = [
                    self._block(always, "post", 0, [0], [0], delay, np.asarray([1.0], np.float32) * u.uS),
                    self._block(always, "post", 0, [0, 1], [0, 0], delay, np.asarray([half_ulp] * 2) * u.uS),
                ]
                self._assert_same_delivery(blocks, (always,))

    def test_a_route_spans_blocks_that_reach_different_synapses(self) -> None:
        from braincell.network.delivery import delivery_blocks

        rng = np.random.default_rng(7)
        first, second = _StepSource(np.ones((1, 4))), _StepSource(np.ones((1, 4)))
        cases = {
            "different synapses": ([[0, 1], [2, 3], [4]], [first] * 3, [u.uS] * 3, [5]),
            "a synapse reached twice": ([[0, 1], [2, 3], [1, 5]], [first] * 3, [u.uS] * 3, [4, 2]),
            "another source between": ([[0], [1], [2]], [first, second, first], [u.uS] * 3, [1, 1, 1]),
            "another unit": ([[0], [1]], [first, first], [u.uS, u.nS], [1, 1]),
        }
        for name, (targets, sources, units, contacts) in cases.items():
            with self.subTest(name):
                blocks = tuple(
                    self._block(source, "post", 0, np.zeros(len(target)), target, 1, self._weights(rng, len(target), unit))
                    for target, source, unit in zip(targets, sources, units)
                )
                routes = delivery_blocks(blocks, group_by_delay=False)
                self.assertEqual([len(route.pre_index) for route in routes], contacts)
                np.testing.assert_array_equal(
                    np.concatenate([route.flat_target_index for route in routes]), np.concatenate(targets)
                )


if __name__ == "__main__":
    unittest.main()
