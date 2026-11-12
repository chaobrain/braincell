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

"""Tests for shared mechanism field declarations and unit prototypes."""

import inspect
import unittest

import brainunit as u

import braincell
from braincell._compute.parameters import _ion_default
from braincell._parameter_schema import (
    ParameterSpec,
    SignatureParameterSpec,
    StateSpec,
    _has_unit_prototype,
    constructor_parameters,
)
from braincell.mech import get_registry, register_channel

# Constructor arguments that configure the node rather than carry a value.
_NOT_PARAMETERS = {"size", "name", "solver", "substeps", "species_initializers"}

# Bare-number defaults that are dimensionless by design.
_DIMENSIONLESS = {
    "q10",
    "q10_base",
    "q10_p",
    "q10_q",
    "gateCurrent",
    "Tauact",
    "Tauinactf",
    "Tauinacts",
    "diff",
    "qdeltat",
    "g_scale",
    "n",
    "phi",
    "Nannuli",
    "valence",
    "P",  # GabaA_JE2011: bicarbonate share of the conductance
    "U",  # ExpSynSTD: fraction of the resource one event releases
}

# Bare-number defaults of physical quantities whose unit is only documented.
_IMPLIED_UNITS = {
    ("AHP_De1994", "alpha"): "mM^-n ms^-1; the unit varies with the trainable exponent n.",
    ("AHP_De1994", "beta"): "ms^-1; kept bare so existing bare-number calls keep working.",
}

# Parameter counts in the registry when this test was written; the scan must
# not inspect fewer, or it silently stopped checking something.
_MINIMUM_COUNTS = {"channel": 375, "ion": 298, "synapse": 5}


def _signature_prototype(category, cls, name, default):
    """The prototype used before ``parameter_info()`` carried one."""
    return _ion_default(cls, name, default) if category == "ion" else default


def _prototype(spec: SignatureParameterSpec):
    return spec.prototype if _has_unit_prototype(spec.prototype) else spec.default


def _dimension(value):
    if isinstance(value, u.Quantity):
        return u.get_dim(value)
    return None if isinstance(value, (bool, int, float)) else type(value).__name__


def unit_prototype_problems(registry):
    """Compare each registered parameter's prototype with the signature one.

    Returns
    -------
    problems : list of str
        Mismatched dimensions and unexplained bare-number prototypes.
    counts : dict
        Parameters inspected per category.
    """
    problems, counts = [], {}
    for category in ("channel", "ion", "synapse"):
        counts[category] = 0
        for class_name, cls in registry.items(category):
            info = cls.parameter_info()
            for name, parameter in constructor_parameters(cls).items():
                if name in _NOT_PARAMETERS:
                    continue
                counts[category] += 1
                old = _signature_prototype(category, cls, name, parameter.default)
                new = _prototype(info[name])
                where = f"{category} {class_name}.{name}"
                if _dimension(old) != _dimension(new):
                    problems.append(f"{where}: prototype {new!r} differs from signature prototype {old!r}")
                bare = isinstance(new, (int, float)) and not isinstance(new, bool)
                if bare and name not in _DIMENSIONLESS and (class_name, name) not in _IMPLIED_UNITS:
                    problems.append(f"{where}: bare-number default {new!r} is neither dimensionless nor listed")
    return problems, counts


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


class UnitPrototypeConsistencyTest(unittest.TestCase):
    def test_every_registered_parameter_keeps_its_prototype_dimension(self):
        problems, counts = unit_prototype_problems(get_registry())
        self.assertEqual(problems, [])
        for category, minimum in _MINIMUM_COUNTS.items():
            self.assertGreaterEqual(counts[category], minimum, category)

    def test_implied_unit_entries_are_live_and_explained(self):
        registry = get_registry()
        for (class_name, name), reason in _IMPLIED_UNITS.items():
            with self.subTest(parameter=f"{class_name}.{name}"):
                self.assertTrue(reason)
                self.assertIn(name, registry.get("channel", class_name).parameter_info())

    def test_an_unexplained_bare_number_default_is_reported(self):
        @register_channel("_Mech25BareTau")
        class BareTau(braincell.Channel):
            def __init__(self, size, tau=5.0, name=None):
                super().__init__(size, name=name)
                self.tau = tau

        try:
            problems, _ = unit_prototype_problems(get_registry())
        finally:
            get_registry().unregister("channel", "_Mech25BareTau")
        self.assertEqual(
            problems, ["channel _Mech25BareTau.tau: bare-number default 5.0 is neither dimensionless nor listed"]
        )


class SignatureParameterSpecTest(unittest.TestCase):
    def test_prototype_falls_back_to_default_then_to_the_value(self):
        declared = SignatureParameterSpec(None, prototype=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "units incompatible with ms"):
            declared.validate(1.0 * u.mV, "tau")
        from_default = SignatureParameterSpec(2.0 * u.ms)
        with self.assertRaisesRegex(TypeError, "requires a quantity compatible with ms"):
            from_default.validate(1.0, "tau")
        unchecked = SignatureParameterSpec(inspect.Parameter.empty)
        unchecked.validate(1.0 * u.mV, "tau")
        unchecked.validate(1.0, "tau")

    def test_validator_runs_after_the_unit_check(self):
        def negative(value, name):
            raise ValueError(f"{name} rejected by validator")

        with self.assertRaisesRegex(ValueError, "tau rejected by validator"):
            SignatureParameterSpec(1.0 * u.ms, validator=negative).validate(2.0 * u.ms, "tau")


if __name__ == "__main__":
    unittest.main()
