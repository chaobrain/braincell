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

"""Autaptic feedback must retain the full forward sensitivity."""

import unittest
import brainstate
import brainunit as u

from validation.optim.gradient_correctness.autapse import DT, compare


class AutapseTest(unittest.TestCase):
    def test_voltage_and_spike_losses_with_event_feedback(self):
        with brainstate.environ.context(dt=DT, precision=64):
            for kind in ("voltage", "spike"):
                for delay in (0.0 * u.ms, 0.1 * u.ms):
                    with self.subTest(kind=kind, delay=delay):
                        result = compare(loss_kind=kind, delay=delay)
                        self.assertGreater(result["loss"], 0.0)
                        if kind == "voltage":
                            self.assertNotEqual(result["gradients"]["weight"], 0.0)
                        else:
                            self.assertNotEqual(result["gradients"]["threshold"], 0.0)

    def test_network_autapse_uses_the_same_full_state_recurrence(self):
        with brainstate.environ.context(dt=DT, precision=64):
            result = compare(network=True)
            self.assertNotEqual(result["gradients"]["cell.weight"], 0.0)

    def test_cross_cell_sensitivity_and_fixed_size_carry(self):
        with brainstate.environ.context(dt=DT, precision=64):
            prefix = compare(network=True, paired=True, steps=100)
            full = compare(network=True, paired=True, steps=160)
            self.assertNotEqual(full["gradients"]["cell.na"], 0.0)
            self.assertNotEqual(full["gradients"]["cell.threshold"], 0.0)
            self.assertEqual(prefix["state_shapes"], full["state_shapes"])
            self.assertEqual(prefix["sensitivity_shapes"], full["sensitivity_shapes"])
