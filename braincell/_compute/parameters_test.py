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

import inspect
import os
import unittest

import braintools
import brainunit as u
import numpy as np

import braincell
from braincell._compute.parameters import (
    density_error_context,
    density_parameter_schema,
    density_parameter_value,
    make_runtime_parameter_state,
    set_parameter_row,
    spatial_callable_source,
)
from braincell._discretization.mechanism import PaintRule
from braincell._parameter_schema import ParameterSpec
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, Ion, Synapse, get_registry


class SignatureParameterTest(unittest.TestCase):
    def test_registered_ion_numeric_defaults_are_valid(self):
        from braincell._compute.parameters import density_parameter_names

        for class_name in get_registry().names("ion"):
            if class_name.startswith("_"):
                continue
            mechanism = Ion(class_name)
            schema = density_parameter_schema(mechanism)
            for field in density_parameter_names(mechanism):
                with self.subTest(ion=class_name, field=field):
                    value = density_parameter_value(mechanism, field)
                    make_runtime_parameter_state(value, full_shape=(1, 2), spec=schema[field], name=field)

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


def _one_cv_cell():
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um)
    return braincell.Cell(braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1))


def _next_line() -> str:
    """``<file>:<line>`` of the line after the caller's."""
    frame = inspect.currentframe().f_back
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno + 1}"


class ParameterUnitMessageTest(unittest.TestCase):
    """MECH-25: init_state unit errors name the mechanism, value and paint/place call."""

    def _init_error(self, declaration, expected_type, *, place=False):
        cell = _one_cv_cell()
        if place:
            site = _next_line()
            cell.place(RootLocation(0.5), declaration)
        else:
            site = _next_line()
            cell.paint(AllRegion(), declaration)
        with self.assertRaises(expected_type) as caught:
            cell.init_state()
        return str(caught.exception), site

    def test_channel_ion_and_synapse_literals(self):
        callable_line = f"parameters_test.py:{inspect.currentframe().f_lineno + 1}"
        no_unit = lambda ctx: 0.1  # noqa: E731
        cases = (
            (
                Channel("IL", name="leak", g_max=0.1),
                TypeError,
                "Channel 'leak' (IL)",
                "g_max",
                "mS / cm^2",
                "0.1 (no unit)",
            ),
            (
                Channel("IL", name="leak", g_max=0.1 * u.mV),
                ValueError,
                "Channel 'leak' (IL)",
                "g_max",
                "mS / cm^2",
                "0.1 mV",
            ),
            (
                Channel("IL", name="leak", g_max=no_unit),
                TypeError,
                "Channel 'leak' (IL)",
                "g_max",
                "mS / cm^2",
                "0.1 (no unit)",
            ),
            (Ion("SodiumFixed", E=50.0), TypeError, "Ion 'SodiumFixed'", "E", "mV", "50.0 (no unit)"),
            (Ion("SodiumFixed", E=50.0 * u.mM), ValueError, "Ion 'SodiumFixed'", "E", "mV", "50.0 mM"),
            (Synapse("ExpSyn", name="syn", tau=5.0), TypeError, "Synapse 'syn' (ExpSyn)", "tau", "ms", "5.0 (no unit)"),
            (Synapse("ExpSyn", name="syn", tau=5 * u.mV), ValueError, "Synapse 'syn' (ExpSyn)", "tau", "ms", "5 mV"),
        )
        for declaration, expected_type, owner, field, unit, got in cases:
            with self.subTest(declaration=declaration):
                place = isinstance(declaration, Synapse)
                message, site = self._init_error(declaration, expected_type, place=place)
                self.assertIn(f"{owner} parameter '{field}'", message)
                self.assertIn(
                    f"compatible with {unit};" if expected_type is TypeError else f"incompatible with {unit};", message
                )
                self.assertIn(f"got {got}.", message)
                region = "Placed on RootLocation(x=0.5)" if place else "ainted on AllRegion()"
                self.assertIn(f"{region} at {site}.", message)
                if declaration.params.get(field) is no_unit:
                    self.assertIn(f"spatial callable <lambda> ({callable_line})", message)

    def test_channel_initializer_is_rejected_with_a_specific_message(self):
        initializer = braintools.init.Constant(0.1 * u.mS / u.cm**2)
        message, site = self._init_error(Channel("IL", name="leak", g_max=initializer), TypeError)
        self.assertIn("Channel 'leak' (IL) parameter 'g_max'", message)
        self.assertIn("braintools initializers are not supported for Channel parameters in a Cell", message)
        self.assertIn(site, message)

    def test_unknown_paint_origin_and_unmatched_declarations_are_omitted(self):
        declaration = Channel("IL", name="leak", g_max=0.1)
        rule = PaintRule(region=AllRegion(), mechanism=declaration)
        self.assertEqual(
            density_error_context(declaration, (rule,)), ("Channel 'leak' (IL)", "painted on AllRegion().")
        )
        self.assertEqual(density_error_context(declaration, ()), ("Channel 'leak' (IL)", None))
        self.assertEqual(spatial_callable_source(abs), "The value came from the spatial callable abs")

    def test_view_set_and_trainable_name_the_channel(self):
        cell = _one_cv_cell()
        cell.paint(AllRegion(), Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2))
        cell.init_state()
        leak = cell.channels["leak"]
        with self.assertRaises(u.UnitMismatchError) as caught:
            leak.set(g_max=0.1 * u.mV)
        self.assertIn(
            "Channel 'leak' (IL) parameter 'g_max' has units incompatible with mS / cm^2; got 0.1 mV.",
            str(caught.exception),
        )
        with self.assertRaisesRegex(
            TypeError, r"Channel 'leak' \(IL\) parameter 'g_max' requires .* got 0.1 \(no unit\)"
        ):
            leak.set(g_max=0.1)
        with self.assertRaisesRegex(TypeError, r"Channel 'leak' \(IL\) parameter 'g_max' .* trainable"):
            leak.trainable(g_max=braincell.trainable.parameter(0.1))
