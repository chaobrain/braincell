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

import brainunit as u
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
    def test_event_backend_brainevent_requires_coomv(self) -> None:
        import braincell.network.delivery as delivery

        try:
            import brainevent
        except Exception:
            return
        if hasattr(brainevent, "coomv"):
            return

        with self.assertRaisesRegex(RuntimeError, "brainevent.coomv"):
            delivery.resolve_event_backend("brainevent")


if __name__ == "__main__":
    unittest.main()
