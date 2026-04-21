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

from braincell.mech import CableProperty


class CablePropertyTest(unittest.TestCase):
    def test_fields_round_trip(self) -> None:
        cp = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        self.assertEqual(cp.resting_potential.to_decimal(u.mV), -65.0)
        self.assertEqual(
            cp.membrane_capacitance.to_decimal(u.uF / u.cm ** 2), 1.0
        )
        self.assertEqual(
            cp.axial_resistivity.to_decimal(u.ohm * u.cm), 100.0
        )

    def test_default_temperature_is_309_15K(self) -> None:
        cp = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        self.assertAlmostEqual(
            cp.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_explicit_temperature(self) -> None:
        cp = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
            temperature=u.celsius2kelvin(22.0),
        )
        self.assertAlmostEqual(
            cp.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(22.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_temperature_not_quantity_raises(self) -> None:
        with self.assertRaises(TypeError):
            CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
                axial_resistivity=100.0 * (u.ohm * u.cm),
                temperature=310.0,  # type: ignore[arg-type]
            )

    def test_non_scalar_temperature_raises(self) -> None:
        import numpy as np

        with self.assertRaises(TypeError):
            CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
                axial_resistivity=100.0 * (u.ohm * u.cm),
                temperature=np.array([310.0, 311.0]) * u.kelvin,
            )

    def test_with_updates_non_mutating(self) -> None:
        original = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        updated = original.with_updates(resting_potential=-70.0 * u.mV)
        self.assertEqual(original.resting_potential.to_decimal(u.mV), -65.0)
        self.assertEqual(updated.resting_potential.to_decimal(u.mV), -70.0)

    def test_equality_and_hash(self) -> None:
        a = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        b = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main()
