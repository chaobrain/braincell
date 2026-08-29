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

"""Tests for schema-aware density parameter storage."""

import unittest

import brainunit as u
import numpy as np

from braincell._compute.parameters import make_runtime_parameter_state, set_parameter_row
from braincell._parameter_schema import ParameterSpec


class RuntimeParameterStateTest(unittest.TestCase):
    def test_uniform_state_stays_scalar_and_exposes_masked_rectangle(self) -> None:
        state = make_runtime_parameter_state(
            2.0 * u.mS / u.cm**2,
            full_shape=(2, 3),
            spec=ParameterSpec(1.0 * u.mS / u.cm**2),
            name="g_max",
            point_mask=np.asarray([False, True, False]),
        )
        self.assertEqual(state.value.shape, ())
        self.assertEqual(state.dense_value().shape, (2, 3))
        expected = np.asarray([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        np.testing.assert_allclose(state.dense_value(masked=True).to_decimal(u.mS / u.cm**2), expected)

    def test_row_write_promotes_without_replacing_state(self) -> None:
        state = make_runtime_parameter_state(
            -70.0 * u.mV,
            full_shape=(2, 3),
            spec=ParameterSpec(-70.0 * u.mV),
            name="E",
        )
        identity = id(state)
        set_parameter_row(
            state,
            population_index=1,
            point_id=2,
            population_size=2,
            point_size=3,
            value=-60.0 * u.mV,
        )
        self.assertEqual(id(state), identity)
        self.assertEqual(state.axis, "row")
        self.assertEqual(state.value.shape, (2, 3))
        self.assertEqual(state.value[1, 2], -60.0 * u.mV)
