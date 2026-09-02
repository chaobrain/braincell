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

import unittest

import brainunit as u

import numpy as np

from braincell.mech import CableProperty
from braincell.mech._testing import make_cable


class CablePropertyTest(unittest.TestCase):
    def test_fields_round_trip(self) -> None:
        cp = make_cable()
        self.assertEqual(cp.resting_potential.to_decimal(u.mV), -65.0)
        self.assertEqual(cp.membrane_capacitance.to_decimal(u.uF / u.cm**2), 1.0)
        self.assertEqual(cp.axial_resistivity.to_decimal(u.ohm * u.cm), 100.0)

    def test_default_temperature_is_309_15K(self) -> None:
        self.assertAlmostEqual(
            make_cable().temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_explicit_temperature(self) -> None:
        cp = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
            temperature=u.celsius2kelvin(22.0),
        )
        self.assertAlmostEqual(
            cp.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(22.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_bad_temperature_raises(self) -> None:
        for bad in (310.0, np.array([310.0, 311.0]) * u.kelvin):
            with self.subTest(temperature=type(bad).__name__):
                with self.assertRaises(TypeError):
                    CableProperty(
                        resting_potential=-65.0 * u.mV,
                        membrane_capacitance=1.0 * (u.uF / u.cm**2),
                        axial_resistivity=100.0 * (u.ohm * u.cm),
                        temperature=bad,  # type: ignore[arg-type]
                    )

    def test_already_canonical_temperature_is_passed_through(self) -> None:
        # The paint lowering pass rebuilds a CableProperty per control
        # volume from fields it has already canonicalized, so the
        # coercion recognises that form and returns it untouched.
        canonical = make_cable().temperature
        rebuilt = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
            temperature=canonical,
        )
        self.assertIs(rebuilt.temperature, canonical)

    def test_equality_and_hash(self) -> None:
        a, b = make_cable(), make_cable()
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, make_cable(cm=2.0))


if __name__ == "__main__":
    unittest.main()
