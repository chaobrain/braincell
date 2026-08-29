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

"""Tests for optimizer-facing parameter collections."""

import unittest

import brainstate
import brainunit as u

from braincell.trainable import ParameterSet


class ParameterSetTest(unittest.TestCase):
    def test_physical_setter_is_complete_and_unit_aware(self) -> None:
        root = brainstate.nn.Param(1.0 * u.mV)
        parameters = ParameterSet({"voltage": root})
        parameters.set_physical_values({"voltage": 2.0 * u.mV})
        self.assertEqual(root.value(), 2.0 * u.mV)
        with self.assertRaises(TypeError):
            parameters.set_physical_values({"voltage": 3.0})
        self.assertEqual(root.value(), 2.0 * u.mV)
        with self.assertRaises(KeyError):
            parameters.set_physical_values({})

    def test_optimizer_setter_preserves_state_identity(self) -> None:
        root = brainstate.nn.Param(2.0)
        parameters = ParameterSet({"factor": root})
        state = parameters.states()["factor"]
        parameters.set_optimizer_values({"factor": 3.0})
        self.assertIs(parameters.states()["factor"], state)
        self.assertEqual(root.value(), 3.0)

    def test_physical_transform_failure_does_not_partially_update(self) -> None:
        good = brainstate.nn.Param(1.0)
        bounded = brainstate.nn.Param(0.5, t=brainstate.nn.SigmoidT(0.0, 1.0))
        parameters = ParameterSet({"good": good, "bounded": bounded})
        with self.assertRaises(Exception):
            parameters.set_physical_values({"good": 2.0, "bounded": 2.0})
        self.assertEqual(good.value(), 1.0)
        self.assertAlmostEqual(float(bounded.value()), 0.5, places=6)
