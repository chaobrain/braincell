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

"""A delayed live self-connection in a Network against NEURON's ``NetCon`` (CELL-75).

See :mod:`validation.neuron.synapse.netcon_delay`. For every delay of ``k``
steps both simulators first see the event in step ``m + 1 + k`` after their
crossing step ``m``, record a zero conductance at the delivery boundary
``t_{m+1+k}`` and ``w exp(-dt / tau)`` at the next sample; the two values
agree within 1e-12 uS. NEURON 9.0.2 gave first seen ``m + 1 + k`` for k = 0,
1, 2, 3, 4 and 12, with ``m`` = 98 (BrainCell: 84). Only built-in mechanisms are used.
"""

import unittest

import numpy as np

from validation.neuron.synapse.netcon_delay import DELAY_STEPS, DT_MS, TAU_MS, WEIGHT_US, run_braincell, run_neuron

try:
    import neuron
except ImportError:  # pragma: no cover - depends on the environment
    neuron = None


class NetConDelayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if neuron is None:
            raise unittest.SkipTest("NEURON is not installed; the NetCon delay comparison needs it.")
        cls.braincell = run_braincell()
        cls.neuron = run_neuron()

    def test_both_first_see_the_event_k_steps_after_the_step_after_the_crossing(self):
        for k in DELAY_STEPS:
            with self.subTest(k=k):
                for side in (self.braincell[k], self.neuron[k]):
                    self.assertEqual(side["first_seen"] - side["crossing"], 1 + k)

    def test_both_record_the_delivery_boundary_alike(self):
        expected = WEIGHT_US * np.exp(-DT_MS / TAU_MS)
        for k in DELAY_STEPS:
            if k == 0:
                continue  # BrainCell adds a zero-delay event at the end of its step: CELL-77
            with self.subTest(k=k):
                values = []
                for side in (self.braincell[k], self.neuron[k]):
                    boundary = side["crossing"] + 1 + k
                    g = side["g"]
                    np.testing.assert_array_equal(g[: boundary + 1], 0.0)
                    values.append(g[boundary + 1])
                np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
