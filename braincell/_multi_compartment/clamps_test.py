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

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell.filter import at


def _clamped_cell():
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"))
    first = braincell.CurrentClamp(durations=1.0 * u.ms, amplitudes=0.2 * u.nA)
    second = braincell.CurrentClamp(delay=1.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.3 * u.nA)
    sine = braincell.SineClamp(
        amplitude=0.0 * u.nA,
        frequency=1.0 * u.kHz,
        offset=0.4 * u.nA,
        duration=2.0 * u.ms,
    )
    cell.place(at("soma", 0.5), first, second, sine)
    return cell, first


class ClampViewTest(unittest.TestCase):
    def test_type_declaration_and_chained_position_selection(self) -> None:
        cell, declaration = _clamped_cell()

        self.assertEqual(cell.clamps.clamp_type.tolist(), ["CurrentClamp", "CurrentClamp", "SineClamp"])
        self.assertEqual(cell.clamps["CurrentClamp"].id.tolist(), [0, 1])
        self.assertEqual(cell.clamps.by_type(braincell.CurrentClamp)[1].id.tolist(), [1])
        self.assertEqual(cell.clamps[declaration].id.tolist(), [0])
        self.assertEqual(cell.clamps.by_id([2, 0]).id.tolist(), [2, 0])

    def test_prepared_point_current_is_held_across_solver_stage_times(self) -> None:
        cell, _ = _clamped_cell()
        cell.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=1.0 * u.ms):
            cell._prepare_step_clamps(t=0.0 * u.ms, dt=1.0 * u.ms)
            with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                at_start = cell._solver_clamp_point_current(t=0.0 * u.ms)
                at_late_stage = cell._solver_clamp_point_current(t=100.0 * u.ms)

        np.testing.assert_array_equal(at_start.to_decimal(u.nA), at_late_stage.to_decimal(u.nA))
        self.assertAlmostEqual(float(at_start[0, 1].to_decimal(u.nA)), 0.6, places=6)

    def test_delay_is_sampled_at_main_step_midpoint(self) -> None:
        soma = braincell.Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )

        def sample(delay_ms):
            cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"))
            cell.place(
                at("soma", 0.5),
                braincell.CurrentClamp(
                    delay=delay_ms * u.ms,
                    durations=2.0 * u.ms,
                    amplitudes=1.0 * u.nA,
                ),
            )
            cell.init_state()
            with brainstate.environ.context(t=4.0 * u.ms, dt=1.0 * u.ms):
                cell._prepare_step_clamps(t=4.0 * u.ms, dt=1.0 * u.ms)
            return float(cell._step_clamp_components.value[0].to_decimal(u.nA))

        self.assertEqual(sample(4.4), 1.0)
        self.assertEqual(sample(4.5), 1.0)
        self.assertEqual(sample(4.6), 0.0)


if __name__ == "__main__":
    unittest.main()
