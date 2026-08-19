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

Only the two behaviours this module exposes directly are covered here; the
ring-buffer arrival machinery is exercised end-to-end from ``engine_test.py``
via :meth:`Network.run`."""

import unittest

import numpy as np

import braincell
from braincell.network._testing import (
    make_spiking_cell,
)


class DeliveryTest(unittest.TestCase):
    def test_population_spike_reduces_multicompartment_spike_to_cell_level_events(self) -> None:
        cell = make_spiking_cell(size=2)
        spike = np.asarray([[False, True, False], [False, False, False]])

        reduced = braincell.network.delivery.population_spike(spike)

        np.testing.assert_array_equal(np.asarray(reduced), [True, False])
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
