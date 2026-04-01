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

from __future__ import annotations

import unittest

from braincell._test_support import u

from braincell import CableProperties, CurrentClamp
from braincell.mech.point import GapJunctionMechanism, ProbeMechanism, SynapseMechanism


class MechanismTest(unittest.TestCase):
    def test_cable_properties_store_quantity_fields(self) -> None:
        cable = CableProperties(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )

        self.assertEqual(cable.resting_potential.to_decimal(u.mV), -65.0)
        self.assertEqual(cable.membrane_capacitance.to_decimal(u.uF / u.cm**2), 1.0)
        self.assertEqual(cable.axial_resistivity.to_decimal(u.ohm * u.cm), 100.0)
        self.assertAlmostEqual(
            cable.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_point_mechanism_dataclasses_are_constructible(self) -> None:
        clamp = CurrentClamp(amplitude=0.2 * u.nA, delay=2.0 * u.ms, duration=1.0 * u.ms)
        synapse = SynapseMechanism(synapse_type="exp2syn", params=(("tau", 5.0),))
        gap = GapJunctionMechanism(params=(("g", 1.0),))
        probe = ProbeMechanism(variable="v", target="soma")

        self.assertEqual(clamp.amplitude.to_decimal(u.nA), 0.2)
        self.assertEqual(clamp.delay.to_decimal(u.ms), 2.0)
        self.assertEqual(clamp.duration.to_decimal(u.ms), 1.0)
        self.assertEqual(synapse.synapse_type, "exp2syn")
        self.assertEqual(gap.params[0], ("g", 1.0))
        self.assertEqual(probe.variable, "v")
