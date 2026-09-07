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

from braincell._compute.parameters import (
    density_parameter_schema,
    density_parameter_value,
    make_runtime_parameter_state,
    set_parameter_row,
)
from braincell._parameter_schema import ParameterSpec
from braincell.mech import Channel, get_registry


class SignatureParameterTest(unittest.TestCase):
    def test_previously_unclassified_channel_has_signature_defaults(self):
        mechanism = Channel("Na_TM1991")
        schema = density_parameter_schema(mechanism)
        self.assertIn("V_sh", schema)
        self.assertIn("name", schema)
        self.assertNotIn("arbitrary_keyword", schema)
        self.assertEqual(density_parameter_value(mechanism, "V_sh"), schema["V_sh"].default)

    def test_forwarded_signature_exposes_parent_defaults(self):
        schema = density_parameter_schema(Channel("Ca_ZH2019_IO_Frozen"))
        self.assertIn("mMidV", schema)
        self.assertIn("freeze_m_inf", schema)

    def test_registered_channel_numeric_defaults_are_valid(self):
        from braincell._compute.parameters import density_parameter_names

        for class_name in get_registry().names("channel"):
            if class_name.startswith("_"):
                continue
            mechanism = Channel(class_name)
            schema = density_parameter_schema(mechanism)
            for field in density_parameter_names(mechanism):
                with self.subTest(channel=class_name, field=field):
                    value = density_parameter_value(mechanism, field)
                    if not callable(value):
                        make_runtime_parameter_state(value, full_shape=(1, 2), spec=schema[field], name=field)


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
