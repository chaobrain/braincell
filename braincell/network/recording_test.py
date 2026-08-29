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

import braincell
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import BranchSlice, at


class RecordingTest(unittest.TestCase):
    def test_scalar_cell_spatial_recording(self) -> None:
        soma = braincell.Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"))
        cell.soma.record("v", braincell.observe.state("v"))

        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

        self.assertEqual(result.samples["v"].values.shape, (2, 1))

    def test_state_recording_uses_global_half_open_schedule(self) -> None:
        cell = _cell(2)
        before_placements = len(cell.point_placements)
        cell[0].soma.record("soma_v", braincell.observe.state("v"), period=0.1 * u.ms)
        self.assertEqual(len(cell.point_placements), before_placements)

        first = cell.run(dt=0.05 * u.ms, duration=0.2 * u.ms)
        second = cell.run(dt=0.05 * u.ms, duration=0.2 * u.ms)

        np.testing.assert_allclose(first.samples["soma_v"].time.to_decimal(u.ms), [0.0, 0.1])
        np.testing.assert_allclose(second.samples["soma_v"].time.to_decimal(u.ms), [0.2, 0.3])
        self.assertEqual(first.samples["soma_v"].values.shape, (2, 1))
        self.assertEqual(first.samples["soma_v"].schema.rows[0].population_index, 0)
        self.assertEqual(first.samples["soma_v"].schema.rows[0].branch_id, 0)

    def test_recordings_support_different_fixed_sampling_periods(self) -> None:
        cell = _cell(1)
        cell.soma.record("fast", braincell.observe.state("v"), period=0.1 * u.ms)
        cell.soma.record("slow", braincell.observe.state("v"), period=0.2 * u.ms)

        first = cell.run(dt=0.05 * u.ms, duration=0.4 * u.ms)
        second = cell.run(dt=0.05 * u.ms, duration=0.4 * u.ms)

        self.assertEqual(first.samples["fast"].values.shape, (4, 1))
        self.assertEqual(first.samples["slow"].values.shape, (2, 1))
        np.testing.assert_allclose(first.samples["fast"].time.to_decimal(u.ms), [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(second.samples["slow"].time.to_decimal(u.ms), [0.4, 0.6])

    def test_unaligned_eager_recording_warns_and_keeps_variable_length_results(self) -> None:
        cell = _cell(1)
        cell.soma.record("v", braincell.observe.state("v"), period=0.1 * u.ms)

        with self.assertWarnsRegex(RuntimeWarning, "unsupported under jax.jit/grad"):
            first = cell.run(dt=0.05 * u.ms, duration=0.15 * u.ms)
        with self.assertWarnsRegex(RuntimeWarning, "unsupported under jax.jit/grad"):
            second = cell.run(dt=0.05 * u.ms, duration=0.15 * u.ms)

        np.testing.assert_allclose(first.samples["v"].time.to_decimal(u.ms), [0.0, 0.1])
        np.testing.assert_allclose(second.samples["v"].time.to_decimal(u.ms), [0.2])
        self.assertEqual(first.samples["v"].values.shape, (2, 1))
        self.assertEqual(second.samples["v"].values.shape, (1, 1))

    def test_nonzero_start_remains_available_in_eager_mode(self) -> None:
        cell = _cell(1)
        cell.soma.record(
            "v",
            braincell.observe.state("v"),
            period=0.1 * u.ms,
            start=0.1 * u.ms,
        )

        result = cell.run(dt=0.05 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(result.samples["v"].time.to_decimal(u.ms), [0.1, 0.2])

    def test_split_result_concatenates_to_single_run(self) -> None:
        split = _cell(1)
        split.soma.record("v", braincell.observe.state("v"))
        parts = (
            split.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
            split.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
        )
        joined = braincell.RunResult.concat(parts)

        full = _cell(1)
        full.soma.record("v", braincell.observe.state("v"))
        one = full.run(dt=0.05 * u.ms, duration=0.2 * u.ms)

        np.testing.assert_allclose(joined.time.to_decimal(u.ms), one.time.to_decimal(u.ms))
        np.testing.assert_allclose(
            joined.samples["v"].values.to_decimal(u.mV),
            one.samples["v"].values.to_decimal(u.mV),
        )

    def test_recording_validation_and_frozen_mappings(self) -> None:
        cell = _cell(1)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            cell.record(
                "bad",
                braincell.observe.state("v"),
                period=0.1 * u.ms,
                frequency=10.0 * u.kHz,
            )
        cell.record("v", braincell.observe.state("v"))
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            cell.run(dt=0.05 * u.ms, duration=0.12 * u.ms)
        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)
        with self.assertRaises(TypeError):
            result.samples["new"] = result.samples["v"]

    def test_channel_and_synapse_state_and_current_schemas(self) -> None:
        cell = _cell(1)
        cell.paint(
            BranchSlice([0, 1], 0.0, 1.0),
            braincell.mech.Channel(
                "IL",
                name="leak",
                g_max=0.1 * u.mS / u.cm**2,
                E=-70.0 * u.mV,
            ),
        )
        cell.place(
            at("dend_a", 0.5),
            braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms),
        )
        cell.place(
            at("dend_a", 0.7),
            braincell.mech.Synapse("Exp2Syn", name="nmda", tau1=0.5 * u.ms, tau2=5.0 * u.ms),
        )
        cell.dendrite.record(
            "leak_current",
            braincell.observe.channel(name="leak").current(reduce="none"),
        )
        cell.record("tau", braincell.observe.synapse(name="ampa").state("tau"))
        cell.record("syn_current", braincell.observe.synapse(name="ampa").current(reduce="none"))
        cell.record("all_g", braincell.observe.synapse().state("g"))

        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

        self.assertEqual(result.samples["leak_current"].schema.size, 3)
        self.assertEqual(result.samples["tau"].schema.size, 1)
        self.assertEqual(result.samples["syn_current"].schema.size, 1)
        self.assertEqual(result.samples["all_g"].schema.size, 2)
        self.assertEqual(result.samples["tau"].schema.rows[0].mechanism_name, "ampa")

    def test_ion_and_total_membrane_current_recording(self) -> None:
        cell = _cell(1)
        region = BranchSlice([0, 1], 0.0, 1.0)
        cell.paint(region, braincell.mech.Ion("SodiumFixed", name="na_pool", E=50.0 * u.mV))
        cell.paint(
            region,
            braincell.mech.Channel(
                "Na_HH1952",
                name="nav",
                g_max=120.0 * u.mS / u.cm**2,
            ),
        )
        cell.soma.record("ina", braincell.observe.ion(name="na_pool").current())
        cell.soma.record("nav_p", braincell.observe.channel(name="nav").state("p"))
        cell.soma.record("imem", braincell.observe.membrane_current())

        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

        self.assertEqual(result.samples["ina"].schema.size, 1)
        self.assertEqual(result.samples["nav_p"].schema.size, 1)
        self.assertEqual(result.samples["imem"].schema.size, 1)
        self.assertTrue(np.isfinite(result.samples["ina"].values.to_decimal(u.mA / u.cm**2)).all())


if __name__ == "__main__":
    unittest.main()
