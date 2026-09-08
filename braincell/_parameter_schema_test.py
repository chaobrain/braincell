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

"""Tests for shared mechanism field declarations."""

import unittest

import brainunit as u

from braincell._parameter_schema import ParameterSpec, StateSpec


class ParameterSchemaTest(unittest.TestCase):
    def test_parameter_validates_units_and_finite_values(self) -> None:
        spec = ParameterSpec(1.0 * u.mV)
        spec.validate(2.0 * u.mV, "voltage")
        with self.assertRaises(TypeError):
            spec.validate(2.0, "voltage")
        with self.assertRaises(ValueError):
            spec.validate(float("nan") * u.mV, "voltage")

    def test_state_can_be_owner_managed(self) -> None:
        spec = StateSpec()
        self.assertTrue(spec.owner_managed)
        spec.validate(object(), "gate")
