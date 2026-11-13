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

"""BrainCell's spike times against NEURON's ``NetCon.record``, spike by spike (CELL-80).

See :mod:`validation.neuron.synapse.spike_times`. With 0.1, 0.2 and 0.3 nA the
soma fires 3, 4 and 5 spikes in 50 ms on both sides, and each spike of
``RunResult.spikes`` and of the Network's ``events["c"]["spike"]`` is within
1e-9 ms of NEURON 9.0.2's: both stamp a crossing in the step from ``t_m`` to
``t_{m+1}`` with ``t_{m+1}``. NEURON accumulates ``t``, so its times sit up to
about 1e-10 ms off the step grid.
"""

import unittest

import numpy as np

from validation.neuron.synapse.spike_times import AMPLITUDES_NA, run_braincell, run_neuron

try:
    import neuron
except ImportError:  # pragma: no cover - depends on the environment
    neuron = None

SPIKE_COUNTS = {0.1: 3, 0.2: 4, 0.3: 5}


class SpikeTimesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if neuron is None:
            raise unittest.SkipTest("NEURON is not installed; the spike time comparison needs it.")
        cls.braincell = run_braincell()
        cls.neuron = run_neuron()

    def test_both_fire_the_same_number_of_spikes(self):
        for amplitude in AMPLITUDES_NA:
            with self.subTest(amplitude=amplitude):
                self.assertEqual(self.neuron[amplitude].size, SPIKE_COUNTS[amplitude])
                for times in self.braincell[amplitude].values():
                    self.assertEqual(times.size, SPIKE_COUNTS[amplitude])

    def test_each_spike_time_is_neurons(self):
        for amplitude in AMPLITUDES_NA:
            for driver, times in self.braincell[amplitude].items():
                with self.subTest(amplitude=amplitude, driver=driver):
                    np.testing.assert_allclose(times, self.neuron[amplitude], rtol=0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
