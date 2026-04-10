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

import braincell
from braincell import Branch, Cell, Morpho
from braincell.filter import BranchSlice


def _build_soma_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    return Morpho.from_root(soma, name="soma")


class CellProfileTest(unittest.TestCase):
    def test_profile_reports_staggered_breakdown(self) -> None:
        cell = Cell(_build_soma_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            report = cell.profile(steps=2, warmup_steps=1, repeat_init=1)

        self.assertEqual(report.solver, "staggered")
        self.assertEqual(report.n_cv, 1)
        self.assertEqual(report.n_point, len(cell.point_tree().points))
        self.assertGreater(report.init_state_ms, 0.0)
        self.assertGreater(report.avg_update_ms, 0.0)
        self.assertIsNotNone(report.channel_update_ms)
        self.assertIsNotNone(report.dhs_voltage_step_ms)
        self.assertIsNotNone(report.independent_channel_integrate_ms)
        self.assertIsNotNone(report.linear_and_const_ms)
        self.assertIsNotNone(report.point_tree_linear_system_ms)
        self.assertIsNotNone(report.dhs_pack_and_convert_ms)
        self.assertIsNotNone(report.triang_ms)
        self.assertIsNotNone(report.backsub_ms)
        self.assertIsNotNone(report.dhs_finalize_ms)
        self.assertIsNotNone(report.residual_ms)
        self.assertGreater(report.channel_update_ms, 0.0)
        self.assertGreater(report.dhs_voltage_step_ms, 0.0)
        self.assertGreater(report.independent_channel_integrate_ms, 0.0)
        self.assertGreater(report.linear_and_const_ms, 0.0)
        self.assertGreater(report.triang_ms, 0.0)
        self.assertGreater(report.backsub_ms, 0.0)
        self.assertGreaterEqual(report.dhs_pack_and_convert_ms, 0.0)
        self.assertGreaterEqual(report.dhs_finalize_ms, 0.0)
        self.assertGreaterEqual(report.residual_ms, 0.0)
        self.assertAlmostEqual(
            report.avg_update_ms,
            report.channel_update_ms + report.dhs_voltage_step_ms + report.independent_channel_integrate_ms + report.residual_ms,
            delta=max(5.0, report.avg_update_ms),
        )
        self.assertIn("avg_update_ms", report.format_text())
        self.assertIn("solver", report.to_dict())

    def test_profile_reports_explicit_without_staggered_breakdown(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            report = cell.profile(steps=2, warmup_steps=1, repeat_init=1)

        self.assertEqual(report.solver, "explicit")
        self.assertGreater(report.avg_update_ms, 0.0)
        self.assertIsNone(report.n_point)
        self.assertIsNone(report.channel_update_ms)
        self.assertIsNone(report.linear_and_const_ms)
        self.assertIsNone(report.triang_ms)
        self.assertIsNone(report.backsub_ms)

    def test_profile_can_include_cprofile_text(self) -> None:
        cell = Cell(_build_soma_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            report = cell.profile(steps=1, warmup_steps=1, repeat_init=1, include_cprofile=True, top_k=5)

        self.assertIsNotNone(report.cprofile_text)
        self.assertTrue(len(report.cprofile_text) > 0)
        self.assertGreater(len(report.cprofile_top_entries), 0)
