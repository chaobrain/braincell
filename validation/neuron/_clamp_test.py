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

"""Voltage-clamp commands, the NEURON clamp and the BrainCell replay solver.

Uses only NEURON's built-in mechanisms, so no compiled library is needed.
"""

import json
import math
import os
from pathlib import Path
import unittest

import numpy as np

from validation.neuron import _clamp
from validation.neuron._suite_loader import use_double_precision

os.environ.setdefault("JAX_PLATFORMS", "cpu")
use_double_precision()

TEMPLATES = Path(__file__).resolve().parent / "channel_no_conc" / "templates"
DT_MS = 0.025


def _template_commands(name: str) -> list[np.ndarray]:
    spec = json.loads((TEMPLATES / f"{name}.json").read_text())["voltage_clamp"]
    return _clamp.protocol_commands(spec, DT_MS)


class ProtocolCommandTest(unittest.TestCase):
    def test_vc_steps(self):
        commands = _template_commands("vc_steps")
        self.assertEqual(len(commands), 8)
        for test_mV, command in zip(range(-100, 41, 20), commands):
            self.assertEqual(command.shape, (1400,))
            np.testing.assert_array_equal(command[:200], -80.0)
            np.testing.assert_array_equal(command[200:1000], float(test_mV))
            np.testing.assert_array_equal(command[1000:], -80.0)

    def test_vc_ramp(self):
        (command,) = _template_commands("vc_ramp")
        self.assertEqual(command.shape, (6800,))
        np.testing.assert_array_equal(command[:201], -100.0)
        self.assertAlmostEqual(command[201], -100.0 + DT_MS, places=12)
        np.testing.assert_array_equal(command[6600:], 60.0)
        self.assertTrue(np.all(np.diff(command) >= 0.0))

    def test_rejects_unknown_kind_and_partial_steps(self):
        with self.assertRaises(ValueError):
            _clamp.protocol_commands({"kind": "sine"}, DT_MS)
        with self.assertRaises(ValueError):
            _clamp.protocol_commands(
                {"kind": "steps", "hold_mV": -80, "hold_ms": 5.01, "test_mV": [0], "test_ms": 1, "tail_ms": 1}, DT_MS
            )


def _hh_section(h, name):
    section = h.Section(name=name)
    section.L = 10.0
    section.diam = 31.830988618379067
    section.insert("hh")
    # Make NEURON derive ena from nai/nao at finitialize, as for a channel that reads cai.
    h.ion_style("na_ion", 1, 2, 1, 0, 0, sec=section)
    return section


def _nernst_mV(ci: float, co: float, celsius: float = 36.0) -> float:
    return 1e3 * 8.314462618 * (celsius + 273.15) / 96485.33212 * math.log(co / ci)


class NeuronClampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neuron import h

        cls.h = h

    def test_seclamp_follows_the_command(self):
        section = _hh_section(self.h, "clamp_follow")
        try:
            segment = section(0.5)
            for command in _template_commands("vc_steps"):
                traces = _clamp.run_neuron_seclamp(
                    self.h, segment, command, dt_ms=DT_MS, celsius=36.0, observables={"ina": lambda: segment.ina}
                )
                self.assertEqual(traces["v"].shape, (1401,))
                self.assertEqual(traces["ina"].shape, (1401,))
                self.assertLessEqual(float(np.max(np.abs(traces["v"][1:] - command))), 1e-7)
        finally:
            self.h.delete_section(sec=section)
        self.assertEqual(_clamp.SECLAMP_RS_MOHM, 1e-9)

    def test_concentrations_are_written_before_finitialize(self):
        section = _hh_section(self.h, "clamp_conc")
        try:
            segment = section(0.5)
            _clamp.initialize_neuron(
                self.h, segment, v_init_mV=-65.0, dt_ms=DT_MS, celsius=36.0, concentrations={"nai": 15.0, "nao": 140.0}
            )
            self.assertEqual(segment.nai, 15.0)
            self.assertAlmostEqual(segment.ena, _nernst_mV(15.0, 140.0), places=5)
            # Written after finitialize, a concentration does not reach ena.
            self.h.finitialize(-65.0)
            segment.nai = 20.0
            self.assertAlmostEqual(segment.ena, _nernst_mV(15.0, 140.0), places=5)
        finally:
            self.h.delete_section(sec=section)


class ReplaySolverTest(unittest.TestCase):
    def _passive_cell(self, voltages, order):
        import brainunit as u

        import braincell

        cell = braincell.Cell(
            braincell.Morphology.from_soma(),
            V_init=float(voltages[0]) * u.mV,
            solver=_clamp.voltage_replay_solver(voltages),
            ion_channel_update_order=order,
        )
        cell.record("v", braincell.observe.state("v"))
        result = cell.run(dt=DT_MS * u.ms, duration=(len(voltages) - 1) * DT_MS * u.ms)
        # The recording samples V(t_n) for n = 0 .. N - 1; the final V_N is the state after the run.
        recorded = np.asarray(result.samples["v"].values.to_decimal(u.mV)).reshape(-1)
        return np.append(recorded, np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1))

    def test_replayed_voltage_equals_the_table_step_by_step(self):
        voltages = np.concatenate([[-65.0], np.random.default_rng(0).uniform(-100.0, 40.0, 40)])
        for order in ("family", "integration"):
            with self.subTest(order=order):
                np.testing.assert_array_equal(self._passive_cell(voltages, order), voltages)


if __name__ == "__main__":
    unittest.main()
