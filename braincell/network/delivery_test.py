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

The delayed-event histories are covered here on their own and through
:meth:`Network.run`; the rest of the arrival machinery is exercised
end-to-end from ``engine_test.py``."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell


class ZerosLikeTest(unittest.TestCase):
    """The one shape helper every buffer in the module goes through.

    Arrival vectors, event histories, and the scatter accumulator inside
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
    history writes over the routes ``delivery_blocks`` returns and over one route
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

    def _deliver(self, routed, blocks, sources, delivery_ops=None, n_steps=None):
        from braincell.network.delivery import (
            advance_delivery_state,
            apply_immediate_events,
            create_delivery_state,
            record_source_events,
            write_arrivals,
        )

        if delivery_ops is None:
            delivery_ops = (None,) * len(routed)
        state = create_delivery_state(routed, delivery_ops=delivery_ops)
        cells = {name: _RecordingCell() for name in {block.post_population for block in blocks}}
        populations = {name: SimpleNamespace(cell=cell) for name, cell in cells.items()}

        def step(index):
            for source in sources:
                source.step = index
            for cell in cells.values():
                cell.arrivals, cell.drives = {}, {}
            write_arrivals(routed, state, populations=populations)
            apply_immediate_events(routed, state, populations=populations)
            record_source_events(state)
            advance_delivery_state(state)
            return (
                {(name, layout): value for name, cell in cells.items() for layout, value in cell.arrivals.items()},
                {(name, layout): value for name, cell in cells.items() for layout, value in cell.drives.items()},
            )

        return brainstate.transform.for_loop(step, jnp.arange(self.N_STEPS if n_steps is None else n_steps))

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

    def _queue_of_weighted_events(self, routes, n_steps):
        """The delayed arrivals of each queue, from a float32 ring of weighted events written here.

        Each step reads and clears its slot first; then each route adds the
        events of this step, weighted now, into the slot of their arrival,
        ``k + 1`` steps on for ``k`` delay steps (CELL-75):
        contact by contact on the scatter path, as one summed vector per route
        on the grouped path. A slot thus sums its events longest delay first,
        then by route and by contact.
        """
        rings, sizes = {}, {}
        for route in routes:
            key = (route.source.post_population, int(route.source.layout_id))
            sizes[key] = int(route.source.n_active)
        depth = 2 + max(int(np.max(route.delay_steps)) for route in routes)
        rings = {key: np.zeros((depth, size), np.float32) for key, size in sizes.items()}
        arrivals = {key: np.zeros((n_steps, size), np.float32) for key, size in sizes.items()}
        for step in range(n_steps):
            for key, ring in rings.items():
                arrivals[key][step] = ring[step % depth]
                ring[step % depth] = 0.0
            for route in routes:
                ring = rings[(route.source.post_population, int(route.source.layout_id))]
                counts = np.asarray(route.source.event_source.table[step], np.float32)[np.asarray(route.pre_index)]
                events = counts * np.asarray(route.weight.to_decimal(u.uS), np.float32)
                delays = np.broadcast_to(np.asarray(route.delay_steps), events.shape)
                if np.ndim(route.delay_steps) == 0:
                    if int(route.delay_steps) == 0:
                        continue
                    summed = np.zeros(ring.shape[1], np.float32)
                    np.add.at(summed, np.asarray(route.flat_target_index), events)
                    ring[(step + int(route.delay_steps) + 1) % depth] += summed
                    continue
                for target, delay, event in zip(route.flat_target_index, delays, events):
                    if delay:
                        ring[(step + int(delay) + 1) % depth, int(target)] += event
        return arrivals

    def test_delayed_arrivals_sum_as_a_queue_of_weighted_events(self) -> None:
        # CELL-69 weights a delayed event at its arrival; with the weights
        # unchanged, each arrival holds the bits a queue of weighted events
        # held, on both paths.
        from braincell.network.delivery import delivery_blocks, make_delivery_op

        rng = np.random.default_rng(69)
        n_steps = 40
        first = _StepSource(rng.integers(0, 3, (n_steps, 8)).astype(np.float32))
        second = _StepSource((rng.random((n_steps, 8)) < 0.5).astype(np.float32))
        blocks = []
        for source, post in ((first, "post"), (second, "post"), (first, "post"), (second, "other")):
            n = 30
            blocks.append(
                self._block(
                    source,
                    post,
                    0,
                    rng.integers(0, 8, n),
                    rng.integers(0, 4, n),
                    rng.integers(0, 5, n),
                    self._weights(rng, n),
                    n_active=4,
                )
            )
        blocks = tuple(blocks)
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                routes = delivery_blocks(blocks, group_by_delay=grouped)
                ops = None
                if grouped:
                    ops = tuple(make_delivery_op(route, pre_size=8, backend="scatter") for route in routes)
                arrivals, _ = self._deliver(routes, blocks, (first, second), delivery_ops=ops, n_steps=n_steps)
                expected = self._queue_of_weighted_events(routes, n_steps)
                self.assertEqual(set(arrivals), set(expected))
                for key, values in expected.items():
                    self.assertGreater(np.count_nonzero(values), n_steps)
                    actual = np.asarray(arrivals[key].to_decimal(u.uS))
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(actual, values, err_msg=str(key))

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
                    self._block(
                        source, "post", 0, np.zeros(len(target)), target, 1, self._weights(rng, len(target), unit)
                    )
                    for target, source, unit in zip(targets, sources, units)
                )
                routes = delivery_blocks(blocks, group_by_delay=False)
                self.assertEqual([len(route.pre_index) for route in routes], contacts)
                np.testing.assert_array_equal(
                    np.concatenate([route.flat_target_index for route in routes]), np.concatenate(targets)
                )

    def test_a_shared_route_reads_its_members_weights(self) -> None:
        from braincell.network.delivery import delivery_blocks

        source = _StepSource(np.ones((1, 4)))
        plain = (
            self._block(source, "post", 0, [0, 1], [0, 1], 1, np.asarray([0.5, 0.25], np.float32)),
            self._block(source, "post", 0, [2], [2], 1, np.asarray([2.0], np.float32)),
        )
        (route,) = delivery_blocks(plain, group_by_delay=False)
        self.assertEqual(route.members, plain)
        np.testing.assert_array_equal(np.asarray(route.weight), np.asarray([0.5, 0.25, 2.0], np.float32))
        trigger_only = (
            self._block(source, "post", 0, [0], [0], 1, None),
            self._block(source, "post", 0, [1], [1], 1, None),
        )
        routes = delivery_blocks(trigger_only, group_by_delay=False)
        self.assertEqual([route.source for route in routes], list(trigger_only))
        self.assertEqual([route.weight for route in routes], [None, None])


def _reference_delay_groups(blocks):
    """``delivery_blocks(..., group_by_delay=True)`` before CELL-18g: one scan per delay."""
    from braincell.network.delivery import DeliveryBlock, slice_weight

    delivery = []
    for block in blocks:
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        for delay_step in sorted(set(delay_steps.tolist())):
            contact_indices = np.nonzero(delay_steps == int(delay_step))[0]
            delivery.append(
                DeliveryBlock(
                    source=block,
                    delay_steps=int(delay_step),
                    pre_index=block.pre_index[contact_indices],
                    flat_target_index=block.synapse_index[contact_indices].astype(np.int32, copy=False),
                    weight=slice_weight(block.weight, contact_indices),
                    weight_indices=contact_indices,
                )
            )
    return tuple(delivery)


class DelayGroupTest(unittest.TestCase):
    """CELL-18g: each delay group lists its contacts, weights and indices as before."""

    @staticmethod
    def _block(rng, n, delays, weight_kind):
        from braincell.network.lowering import ConnectionBlock

        mantissa = rng.uniform(0.0, 1.0, n).astype(np.float32)
        getter = None
        if weight_kind == "quantity":
            weight = u.Quantity(mantissa, u.uS)
        elif weight_kind == "array":
            weight = mantissa
        else:
            weight = None
            getter = lambda: u.Quantity(jnp.asarray(mantissa), u.nS)  # noqa: E731
        return ConnectionBlock(
            pre_population="pre",
            post_population="post",
            synapse="syn",
            layout_id=0,
            n_active=n,
            pre_index=rng.integers(0, 50, n).astype(np.int32),
            post_index=rng.integers(0, 40, n).astype(np.int64),
            synapse_index=rng.integers(0, 40, n).astype(np.int64),
            weight=weight,
            delay_steps=np.asarray(delays, dtype=np.int64),
            event_source=None,
            weight_getter=getter,
        )

    def test_groups_match_one_scan_per_delay(self) -> None:
        from braincell.network.delivery import delivery_blocks

        rng = np.random.default_rng(18)
        blocks = (
            self._block(rng, 200, rng.integers(0, 12, 200), "quantity"),
            self._block(rng, 7, [3, 3, 3, 3, 3, 3, 3], "array"),
            self._block(rng, 0, [], "quantity"),
            self._block(rng, 30, rng.integers(0, 4, 30)[::-1], "getter"),
            self._block(rng, 9, [5, 0, 5, 1, 0, 9, 9, 1, 5], "array"),
        )
        actual, expected = delivery_blocks(blocks, group_by_delay=True), _reference_delay_groups(blocks)
        self.assertEqual(len(actual), len(expected))
        for index, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(index=index):
                self.assertIs(got.source, want.source)
                self.assertEqual(type(got.delay_steps), type(want.delay_steps))
                self.assertEqual(got.delay_steps, want.delay_steps)
                for name in ("pre_index", "flat_target_index", "weight_indices"):
                    np.testing.assert_array_equal(getattr(got, name), getattr(want, name), err_msg=name)
                    self.assertEqual(getattr(got, name).dtype, getattr(want, name).dtype, name)
                got_weight, want_weight = got.weight, want.weight
                self.assertEqual(type(got_weight), type(want_weight))
                if isinstance(want_weight, u.Quantity):
                    self.assertEqual(got_weight.unit, want_weight.unit)
                    got_weight, want_weight = got_weight.mantissa, want_weight.mantissa
                np.testing.assert_array_equal(np.asarray(got_weight), np.asarray(want_weight))
                self.assertEqual(np.asarray(got_weight).dtype, np.asarray(want_weight).dtype)


def _float64(test):
    """Run ``test`` in float64."""

    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    wrapped.__doc__ = test.__doc__
    return wrapped


class InFlightWeightTest(unittest.TestCase):
    """CELL-69: a delayed live event takes the weight at its delivery, as NEURON's ``NetCon`` does.

    A toy Cell connects its own spike port to its ExpSyn. Its solver sets V
    and leaves the synapse alone, so the increment of g over step ``b`` is
    the payload delivered at boundary ``b``. It crosses threshold in the
    steps that end at 3.0, 4.8, 5.0 and 5.2 ms (dt 0.1 ms, steps
    ``m`` = 29, 47, 49, 51); with ``k >= 1`` delay steps an event of step
    ``m`` arrives at boundary ``m + k + 1`` (CELL-75). The first run ends
    at boundary 50, ``set(weight=)`` then changes the weights, and the
    second run takes 20 steps: the events sent before boundary 50 and
    delivered from it on were in flight across the change. The expected payloads are a host rule
    written here: the weight is the new one from boundary 50 on.
    """

    DT_MS = 0.1
    N1, N2 = 50, 20
    SPIKE_STEPS = (29, 47, 49, 51)

    def _cell(self):
        from braincell import CVPerBranch, Cell
        from braincell.filter import at
        from braincell.network._testing import make_soma_tree

        ends = (np.asarray(self.SPIKE_STEPS, dtype=float) + 1.0) * self.DT_MS

        def solver(cell):
            t = brainstate.environ.get("t").to_decimal(u.ms)
            hit = jnp.any(jnp.abs(t - (ends - self.DT_MS)) < 0.05)
            cell.V.value = u.math.where(hit, 40.0, -65.0) * u.mV + 0.0 * cell.V.value

        cell = Cell(
            make_soma_tree(),
            cv_policy=CVPerBranch(),
            pop_size=(1,),
            V_init=-65.0 * u.mV,
            V_th=0.0 * u.mV,
            solver=solver,
        )
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
        cell.soma.record("g", braincell.observe.synapse(name="s").state("g"))
        return cell

    def _model(self, rows):
        """``rows``: ``(name, delay ms, weight before, weight after)``."""
        cell = self._cell()
        for name, delay, before, _ in rows:
            braincell.connect(
                name,
                source=cell.event_outputs["spike"],
                synapse=cell.synapses["s"],
                weight=before * u.uS,
                delay=delay * u.ms,
            )
        net = braincell.Network("in_flight")
        net.add_population("post", cell)
        return net, cell

    @staticmethod
    def _set_weights(cell, rows) -> None:
        for name, _, _, after in rows:
            cell.connections[name].set(weight=after * u.uS)

    @staticmethod
    def _g(cell) -> float:
        return float(np.asarray(cell.synapses["s"].g.to_decimal(u.uS)).reshape(-1)[0])

    def _payloads_of_runs(self, rows, **run_kwargs):
        """The payload of boundaries 0..69 over two runs with ``set(weight=)`` between them."""
        model, cell = self._model(rows)
        dt = self.DT_MS * u.ms
        g = []
        for steps in (self.N1, self.N2):
            result = model.run(dt=dt, duration=steps * dt, **run_kwargs)
            values = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS), dtype=np.float64)
            self.assertEqual(values.shape[0], steps)
            g.append(values.reshape(steps, -1)[:, 0])
            if steps == self.N1:
                self._set_weights(cell, rows)
        g.append([self._g(cell)])
        return np.diff(np.concatenate(g))

    def _payloads_of_updates(self, rows):
        model, cell = self._model(rows)
        model.prepare_run(dt=self.DT_MS * u.ms)
        g = [self._g(cell)]
        for step in range(self.N1 + self.N2):
            if step == self.N1:
                self._set_weights(cell, rows)
            model.update()
            g.append(self._g(cell))
        return np.diff(np.asarray(g))

    def _expected(self, rows):
        payload = np.zeros(self.N1 + self.N2)
        for _, delay, before, after in rows:
            k = int(round(delay / self.DT_MS))
            for m in self.SPIKE_STEPS:
                # A zero-delay event is added at the end of its own step; a
                # delayed one at the start of step m + k + 1 (CELL-75).
                b = m if k == 0 else m + k + 1
                payload[b] += after if b >= self.N1 else before
        return payload

    def _assert_payloads(self, actual, rows) -> None:
        expected = self._expected(rows)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    @_float64
    def test_an_event_in_flight_takes_the_weight_at_its_delivery(self) -> None:
        for delay in (0.0, 0.1, 0.2, 0.3):
            rows = (("c", delay, 0.05, 0.08),)
            for backend in ("scatter", "brainevent"):
                with self.subTest(delay=delay, event_backend=backend):
                    network = self._payloads_of_runs(rows, event_backend=backend)
                    self._assert_payloads(network, rows)
            with self.subTest(delay=delay, driver="Network.update"):
                self._assert_payloads(self._payloads_of_updates(rows), rows)

    @_float64
    def test_rows_of_one_synapse_each_take_their_own_weight_at_delivery(self) -> None:
        # Boundary 51 receives the event of step 49 over "near" and the one
        # of step 47 over "far", both sent before the change: 0.08 + 0.03.
        rows = (("near", 0.1, 0.05, 0.08), ("far", 0.3, 0.02, 0.03))
        self.assertAlmostEqual(self._expected(rows)[self.N1 + 1], 0.11, places=15)
        for backend in ("scatter", "brainevent"):
            with self.subTest(event_backend=backend):
                self._assert_payloads(self._payloads_of_runs(rows, event_backend=backend), rows)
        with self.subTest(driver="Network.update"):
            self._assert_payloads(self._payloads_of_updates(rows), rows)

    @_float64
    def test_reset_state_drops_the_events_in_flight(self) -> None:
        # The events of steps 47 and 49 are in flight when the first run
        # ends; kept, they would arrive at boundaries 1 and 3 of the next.
        model, cell = self._model((("c", 0.3, 0.05, 0.05),))
        dt = self.DT_MS * u.ms
        runs = []
        for _ in range(2):
            result = model.run(dt=dt, duration=self.N1 * dt)
            runs.append(np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS)))
            model.reset_state()
        self.assertGreater(np.ptp(runs[0]), 0.0)
        np.testing.assert_array_equal(runs[1], runs[0])

    def test_the_loop_compiles_once_over_runs_a_weight_change_and_a_reset(self) -> None:
        from braincell._multi_compartment.lifecycle_test import loop_compiles

        rows = (("near", 0.1, 0.05, 0.08), ("far", 0.3, 0.02, 0.03))
        for backend in ("scatter", "brainevent"):
            with self.subTest(event_backend=backend):
                model, cell = self._model(rows)
                dt = self.DT_MS * u.ms
                counts = []
                for action in ("run", "run", "set", "reset"):
                    if action == "set":
                        self._set_weights(cell, rows)
                    elif action == "reset":
                        model.reset_state()
                    with loop_compiles() as names:
                        model.run(dt=dt, duration=self.N1 * dt, event_backend=backend)
                    counts.append(names.count("jit(jaxpr_call)"))
                self.assertGreaterEqual(counts[0], 1)
                self.assertEqual(counts[1:], [0, 0, 0])


class LiveArrivalStepTest(unittest.TestCase):
    """CELL-75: a delayed live event is first seen in the same step on every path, as NEURON's ``NetCon``.

    Seven identical HH members (a 1-CV cable with ``staggered``, or a lumped
    Cell with ``rk4``), 0.1 nA from 0.25 ms, each driving its own ExpSyn
    (tau 2 ms, e 0 mV, 0.002 uS) through a live connection of ``k`` steps of
    0.025 ms. All cross 0 mV in step ``m`` (V before it below, V after it at
    or above). The first step whose V after it differs from the same path
    without the connection is ``m + 1 + k``; so in NEURON 9.0.2 with a fixed
    step, whose ``NetCon`` delivers at ``t_{m+1+k}`` (the NetCon delay
    comparison in ``validation/neuron/synapse/netcon_delay.py``).
    """

    DT = 0.025 * u.ms
    N = 100
    KS = (0, 1, 2, 3, 4, 11, 12)
    W_US, TAU_MS = 0.002, 2.0
    # NEURON 9.0.2, fixed step 0.025 ms: first seen in step m + 1 + k for every k (validation/neuron/synapse/netcon_delay.py).
    NEURON_FIRST_SEEN_AFTER_CROSSING = 1

    def _cell(self, kind, *, connect=True, rule=False):
        from braincell import mech
        from braincell.filter import AllRegion, RootLocation

        n = len(self.KS)
        channels = (
            mech.Ion("SodiumFixed", name="sodium", E=50.0 * u.mV),
            mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            mech.Channel("IL", name="leak"),
            mech.Channel("Na_HH1952", name="na"),
            mech.Channel("K_HH1952", name="k"),
        )
        clamp = mech.CurrentClamp(delay=0.25 * u.ms, durations=8.0 * u.ms, amplitudes=0.1 * u.nA)
        synapse = mech.Synapse("ExpSyn", name="syn", tau=self.TAU_MS * u.ms, e=0.0 * u.mV)
        if kind == "cable":
            branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
            cell = braincell.Cell(
                braincell.Morphology.from_root(branch),
                cv_policy=braincell.CVPerBranch(1),
                pop_size=(n,),
                V_init=-65.0 * u.mV,
                solver="staggered",
            )
            cell.paint(
                AllRegion(),
                mech.CableProperty(
                    resting_potential=-65.0 * u.mV,
                    membrane_capacitance=1.0 * u.uF / u.cm**2,
                    axial_resistivity=100.0 * u.ohm * u.cm,
                ),
                *channels,
            )
            cell.place(RootLocation(0.5), clamp)
            cell.place(RootLocation(0.5), synapse)
        else:
            cell = braincell.Cell.point(n, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
            cell.paint(AllRegion(), *channels)
            cell.place(clamp)
            cell.place(synapse)
        cell.record("v", braincell.observe.state("V"))
        cell.record("g", braincell.observe.synapse(name="syn").state("g"))
        if connect:
            plasticity = None
            if rule:
                plasticity = mech.ConnectionRule(
                    "PairSTDP", A_plus=1e-9 * u.uS, A_minus=1e-9 * u.uS, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms
                )
            braincell.connect(
                "self",
                source=cell.event_outputs["spike"],
                synapse=cell.synapses["syn"],
                weight=self.W_US * u.uS,
                delay=np.asarray(self.KS) * self.DT,
                plasticity=plasticity,
            )
        return cell

    def _after(self, samples, cell, name):
        """``name`` after each step (rows) per member: the samples are taken before the steps."""
        values = np.asarray(u.get_mantissa(samples[name].values), dtype=np.float64).reshape(self.N, -1)
        last = np.asarray(u.get_mantissa(getattr(cell, "V").value if name == "v" else cell.synapses["syn"].g))
        return np.concatenate([values[1:], last.reshape(1, -1)]), values

    def _drive(self, kind, path, *, connect=True):
        """V after each step and the g samples (before each step), per member, by one path."""
        cell = self._cell(kind, connect=connect, rule=path == "network rule")
        if path == "Cell.run":
            cell.init_state()
            samples = cell.run(dt=self.DT, duration=self.N * self.DT).samples
        elif path == "update":
            cell.init_state()
            if connect:
                cell.connections.prepare_runtime(self.DT)

            def body(t):
                with brainstate.environ.context(t=t):
                    cell.update()
                return cell.V.value.to_decimal(u.mV)

            with brainstate.environ.context(dt=self.DT):
                v = brainstate.transform.for_loop(body, jnp.arange(self.N) * self.DT)
            return np.asarray(v, dtype=np.float64).reshape(self.N, -1), None
        else:
            net = braincell.Network("arrival")
            net.add_population("c", cell)
            net.init_state()
            samples = net.run(dt=self.DT, duration=self.N * self.DT).samples["c"]
        v, _ = self._after(samples, cell, "v")
        g = np.asarray(samples["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(self.N, -1)
        return v, g

    def _first_seen(self, v, reference):
        changed = v != reference
        self.assertTrue(np.all(changed.any(axis=0)))
        return changed.argmax(axis=0)

    def _check_kind(self, kind):
        runs = {}
        for path in ("Cell.run", "update", "network", "network rule"):
            reference = runs.get(("reference", path if path != "network rule" else "network"))
            if reference is None:
                reference = self._drive(kind, path if path != "network rule" else "network", connect=False)[0]
                runs[("reference", path if path != "network rule" else "network")] = reference
            column = reference[:, 0]
            np.testing.assert_array_equal(reference, np.repeat(column[:, None], len(self.KS), axis=1))
            v_before = np.concatenate([[-65.0], column[:-1]])
            m = int(np.flatnonzero((v_before < 0.0) & (column >= 0.0))[0])
            v, g = self._drive(kind, path)
            runs[path] = (v, g)
            with self.subTest(kind=kind, path=path):
                np.testing.assert_array_equal(
                    self._first_seen(v, reference) - m, np.asarray(self.KS) + self.NEURON_FIRST_SEEN_AFTER_CROSSING
                )
        return runs, m

    @_float64
    def test_every_path_first_sees_an_event_k_steps_after_the_crossing_step(self) -> None:
        for kind in ("cable", "lumped"):
            runs, m = self._check_kind(kind)
            (net_v, net_g), (run_v, _) = runs["network"], runs["Cell.run"]
            with self.subTest(kind=kind, check="Network V is Cell.run V"):
                np.testing.assert_array_equal(net_v, run_v)
                # Until CELL-75 a Network row of k steps was bitwise a Cell.run row of k - 1.
                for j in range(1, len(self.KS)):
                    if self.KS[j - 1] == self.KS[j] - 1:
                        self.assertFalse(np.array_equal(net_v[:, j], run_v[:, j - 1]), self.KS[j])
            with self.subTest(kind=kind, check="recorded g as NEURON"):
                # Sample j is g at t_j: NEURON records 0 at the delivery boundary t_{m+1+k}
                # and w exp(-dt/tau) at the next one.
                for j, k in enumerate(self.KS):
                    if k == 0:
                        continue
                    np.testing.assert_array_equal(net_g[: m + 2 + k, j], 0.0)
                    np.testing.assert_allclose(
                        net_g[m + 2 + k, j], self.W_US * np.exp(-0.025 / self.TAU_MS), rtol=0.0, atol=1e-12
                    )

    @_float64
    def test_a_delayed_event_between_two_populations_and_across_a_split_run(self) -> None:
        # One HH source and three passive targets at rest (their V stays at -65 mV
        # until an event arrives), with 0, 1 and 3 steps of delay. The runs split
        # at the boundary after the crossing step and at the boundaries the
        # 1- and 3-step events are delivered at, t_{m+2} and t_{m+4}.
        from braincell import mech

        ks = (0, 1, 3)

        def network():
            pre = self._cell("lumped", connect=False)
            post = braincell.Cell.point(3, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
            post.paint(mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
            post.place(mech.Synapse("ExpSyn", name="syn", tau=self.TAU_MS * u.ms, e=0.0 * u.mV))
            post.record("v", braincell.observe.state("V"))
            net = braincell.Network("pair")
            net.add_population("pre", pre)
            net.add_population("post", post)
            net.connect(
                "pp",
                source=net.populations["pre"].event_outputs["spike"][[0, 0, 0]],
                synapse=net.populations["post"].synapses["syn"],
                weight=self.W_US * u.uS,
                delay=np.asarray(ks) * self.DT,
            )
            return net, pre, post

        net, pre, post = network()
        result = net.run(dt=self.DT, duration=self.N * self.DT)
        pre_v, _ = self._after(result.samples["pre"], pre, "v")
        post_v, _ = self._after(result.samples["post"], post, "v")
        v_before = np.concatenate([[-65.0], pre_v[:-1, 0]])
        m = int(np.flatnonzero((v_before < 0.0) & (pre_v[:, 0] >= 0.0))[0])
        first = (post_v != -65.0).argmax(axis=0)
        np.testing.assert_array_equal(first - m, np.asarray(ks) + self.NEURON_FIRST_SEEN_AFTER_CROSSING)
        net, _, split_post = network()
        pieces = []
        for steps in (m + 1, 1, 2, self.N - m - 4):
            part = net.run(dt=self.DT, duration=steps * self.DT).samples["post"]["v"].values
            pieces.append(np.asarray(part.to_decimal(u.mV), dtype=np.float64).reshape(steps, -1))
        np.testing.assert_array_equal(
            np.concatenate(pieces),
            np.asarray(result.samples["post"]["v"].values.to_decimal(u.mV), dtype=np.float64).reshape(self.N, -1),
        )


if __name__ == "__main__":
    unittest.main()
