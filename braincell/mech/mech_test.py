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


import math
import unittest

import braintools
import brainunit as u

import braincell
from braincell import CableProperty, CurrentClamp, FunctionClamp, SineClamp
from braincell.mech.point import GapJunctionMechanism, ProbeMechanism, SynapseMechanism


class MechanismTest(unittest.TestCase):
    def test_cable_properties_store_quantity_fields(self) -> None:
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )

        self.assertEqual(cable.resting_potential.to_decimal(u.mV), -65.0)
        self.assertEqual(cable.membrane_capacitance.to_decimal(u.uF / u.cm ** 2), 1.0)
        self.assertEqual(cable.axial_resistivity.to_decimal(u.ohm * u.cm), 100.0)
        self.assertAlmostEqual(
            cable.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_point_mechanism_dataclasses_are_constructible(self) -> None:
        clamp = CurrentClamp(amplitude=0.2 * u.nA, delay=2.0 * u.ms, duration=1.0 * u.ms)
        step_clamp = CurrentClamp(
            start=1.0 * u.ms,
            durations=(2.0 * u.ms, 3.0 * u.ms),
            amplitudes=(0.0 * u.nA, 0.3 * u.nA),
        )
        sine = SineClamp(amplitude=0.4 * u.nA, frequency=50.0 * u.Hz, duration=5.0 * u.ms)
        function = FunctionClamp(fn=lambda local_t: 0.1 * u.nA, duration=4.0 * u.ms)
        synapse = SynapseMechanism(synapse_type="exp2syn", params=(("tau", 5.0),))
        gap = GapJunctionMechanism(params=(("g", 1.0),))
        probe = ProbeMechanism(variable="v", target="soma")

        self.assertEqual(clamp.amplitude.to_decimal(u.nA), 0.2)
        self.assertEqual(clamp.delay.to_decimal(u.ms), 2.0)
        self.assertEqual(clamp.duration.to_decimal(u.ms), 1.0)
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in step_clamp.durations), (2.0, 3.0))
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in step_clamp.amplitudes), (0.0, 0.3))
        self.assertEqual(sine.frequency.to_decimal(u.Hz), 50.0)
        self.assertTrue(callable(function.fn))
        self.assertEqual(synapse.synapse_type, "exp2syn")
        self.assertEqual(gap.params[0], ("g", 1.0))
        self.assertEqual(probe.variable, "v")

    def test_single_compartment_default_area_uses_default_geometry(self) -> None:
        cell = braincell.SingleCompartment(1)
        expected_area = 2.0 * math.pi * 5.0 * 10.0
        self.assertAlmostEqual(cell.area.to_decimal(u.um ** 2), expected_area, places=12)

    def test_single_compartment_accepts_total_current_in_nA(self) -> None:
        cell = braincell.SingleCompartment(
            1,
            V_initializer=braintools.init.Uniform(-65.0 * u.mV, -65.0 * u.mV),
        )
        cell.init_state()
        cell.compute_derivative(1.0 * u.nA)
        derivative_from_total = cell.V.derivative

        density = (1.0 * u.nA / cell.area).in_unit(u.nA / (u.cm ** 2))
        cell.compute_derivative(density)
        derivative_from_density = cell.V.derivative

        self.assertTrue(
            u.math.allclose(
                derivative_from_total.to_decimal(u.mV / u.ms),
                derivative_from_density.to_decimal(u.mV / u.ms),
            )
        )

    def test_single_compartment_area_uses_explicit_geometry(self) -> None:
        cell = braincell.SingleCompartment(1, length=20.0 * u.um, radius=2.0 * u.um)
        expected_area = 2.0 * math.pi * 2.0 * 20.0
        self.assertAlmostEqual(cell.area.to_decimal(u.um ** 2), expected_area, places=12)
