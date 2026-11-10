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

"""Field-declaration contract for vectorized runtime synapse models.

These specs are what a synapse author writes (see
:mod:`braincell.synapse.exponential`), so their validation is the first thing a
malformed declaration hits. Until this module moved into
:mod:`braincell.mech` it had no direct tests at all -- every branch below was
reachable only through a concrete synapse.
"""

import unittest

import brainunit as u
import jax
import numpy as np

from braincell.mech import ParameterSpec, StateSpec, positive
from braincell.mech._synapse_schema import describe_mechanism, describe_value


class ParameterSpecTest(unittest.TestCase):
    def test_quantity_default_accepts_a_compatible_quantity(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms)
        spec.validate(5.0 * u.ms, "tau")
        spec.validate(0.003 * u.second, "tau")
        spec.validate(np.asarray([1.0, 2.0]) * u.ms, "tau")

    def test_quantity_default_rejects_a_bare_number(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms)
        with self.assertRaisesRegex(TypeError, "requires a quantity compatible with"):
            spec.validate(5.0, "tau")

    def test_quantity_default_rejects_an_incompatible_unit(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "units incompatible with"):
            spec.validate(5.0 * u.mV, "tau")

    def test_dimensionless_default_rejects_a_quantity(self) -> None:
        spec = ParameterSpec(default=0.5)
        spec.validate(0.25, "ratio")
        with self.assertRaisesRegex(TypeError, "must be dimensionless"):
            spec.validate(0.25 * u.ms, "ratio")

    def test_empty_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            ParameterSpec(default=2.0 * u.ms).validate(np.asarray([]) * u.ms, "tau")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            ParameterSpec(default=0.5).validate(np.asarray([]), "ratio")

    def test_non_finite_values_are_rejected(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms)
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "only finite values"):
                spec.validate(bad * u.ms, "tau")

    def test_non_numeric_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            ParameterSpec(default=0.5).validate("not-a-number", "ratio")

    def test_extra_validator_runs_after_the_shape_check(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms, validator=positive)
        spec.validate(1.0 * u.ms, "tau")
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            spec.validate(-1.0 * u.ms, "tau")
        # The dimensional check still precedes the custom validator.
        with self.assertRaisesRegex(TypeError, "requires a quantity"):
            spec.validate(-1.0, "tau")


class StateSpecTest(unittest.TestCase):
    def test_state_validates_against_its_initial_value(self) -> None:
        spec = StateSpec(initial=0.0 * u.uS)
        spec.validate(1.0 * u.uS, "g")
        with self.assertRaisesRegex(TypeError, "requires a quantity compatible with"):
            spec.validate(1.0, "g")

    def test_state_spec_is_frozen(self) -> None:
        spec = StateSpec(initial=0.0 * u.uS)
        with self.assertRaises(Exception):
            spec.initial = 1.0 * u.uS


class PositiveValidatorTest(unittest.TestCase):
    def test_strictly_positive_values_pass(self) -> None:
        positive(1e-9 * u.ms, "tau")
        positive(np.asarray([1.0, 2.0]) * u.ms, "tau")
        positive(0.5, "ratio")

    def test_zero_and_negative_are_rejected(self) -> None:
        for bad in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "must be > 0"):
                positive(bad * u.ms, "tau")

    def test_one_bad_entry_rejects_the_whole_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            positive(np.asarray([1.0, -1.0]) * u.ms, "tau")


class UnitErrorMessageTest(unittest.TestCase):
    def test_without_context_the_message_is_unchanged(self) -> None:
        with self.assertRaises(TypeError) as caught:
            ParameterSpec(default=2.0 * u.ms).validate(5.0, "tau")
        self.assertEqual(str(caught.exception), "Mechanism field 'tau' requires a quantity compatible with ms.")

    def test_context_names_owner_value_and_source(self) -> None:
        spec = ParameterSpec(default=2.0 * u.ms)
        owner = describe_mechanism("synapse", "ExpSyn", "syn")
        with self.assertRaises(ValueError) as caught:
            spec.validate(5.0 * u.mV, "tau", owner=owner, source="placed on X at f.py:3.")
        self.assertEqual(
            str(caught.exception),
            "Synapse 'syn' (ExpSyn) parameter 'tau' has units incompatible with ms; got 5.0 mV. Placed on X at f.py:3.",
        )
        with self.assertRaisesRegex(TypeError, "^Mechanism parameter 'ratio' must be dimensionless; got 1.0 ms.$"):
            ParameterSpec(default=0.5).validate(1.0 * u.ms, "ratio", source="")
        with self.assertRaisesRegex(
            ValueError, "^Ion 'Na' parameter 'E' must contain only finite values; got nan mV.$"
        ):
            ParameterSpec(default=1.0 * u.mV).validate(np.nan * u.mV, "E", owner="Ion 'Na'")

    def test_owner_omits_an_instance_name_equal_to_the_class(self) -> None:
        self.assertEqual(describe_mechanism("channel", "IL", "IL"), "Channel 'IL'")
        self.assertEqual(describe_mechanism("ion", "SodiumFixed"), "Ion 'SodiumFixed'")
        self.assertEqual(describe_mechanism("channel", "IL", "leak"), "Channel 'leak' (IL)")

    def test_values_are_described_by_scalar_or_shape(self) -> None:
        self.assertEqual(describe_value(0.1), "0.1 (no unit)")
        self.assertEqual(describe_value(np.asarray([[0.1]]) * u.mV), "0.1 mV")
        self.assertEqual(describe_value(np.zeros((2, 3))), "an array of shape (2, 3) (no unit)")
        self.assertEqual(describe_value(np.zeros(2) * u.mV), "an array of shape (2,) in mV")
        seen = []
        jax.jit(lambda x: seen.append(describe_value(x * u.mV)) or x)(1.0)
        self.assertIn("Tracer", seen[0])
        self.assertTrue(seen[0].endswith(" mV"))


if __name__ == "__main__":
    unittest.main()
