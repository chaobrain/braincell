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


if __name__ == "__main__":
    unittest.main()
