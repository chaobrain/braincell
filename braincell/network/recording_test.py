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
import jax.numpy as jnp
import numpy as np

import braincell
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import BranchSlice, at, branch_in
from braincell.morph._testing import make_tree_s, make_tree_t, make_tree_u


class RecordingTest(unittest.TestCase):
    def test_synapse_current_uses_staggered_point_voltage(self) -> None:
        soma = braincell.Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = braincell.Cell(
            braincell.Morphology.from_root(soma, name="soma"),
            cv_policy=braincell.CVPerBranch(),
            solver="staggered",
        )
        synapse = braincell.mech.Synapse("ExpSyn", name="syn", e=0.0 * u.mV)
        cell.place(at("soma", 0.0), synapse)
        cell.record(
            "syn_current",
            braincell.observe.synapse(name="syn").current(reduce="none"),
        )
        cell.init_state()

        layout = next(layout for layout in cell.runtime.layouts if layout.kind == "synapse:ExpSyn")
        runtime_synapse = cell.runtime.get_runtime_node(layout.id)
        runtime_synapse.g.value = runtime_synapse.g.value + 0.01 * u.uS
        cell.V.value = cell.V.value.at[..., 0].set(-60.0 * u.mV)
        cell._set_dhs_point_voltage(jnp.asarray([[-40.0, -60.0, -60.0]]) * u.mV)

        sampled = cell._compiled_recordings(0.025 * u.ms)[0].sample()

        np.testing.assert_allclose(sampled.to_decimal(u.nA), [0.4], rtol=0.0, atol=1e-7)

    def test_clamp_view_recording_uses_midpoint_time_and_cached_components(self) -> None:
        cell = _cell(1)
        first = braincell.CurrentClamp(durations=0.1 * u.ms, amplitudes=0.2 * u.nA)
        second = braincell.CurrentClamp(durations=0.2 * u.ms, amplitudes=0.3 * u.nA)
        cell.place(at("soma", 0.5), first, second)
        cell.clamps["CurrentClamp"].record("clamps")
        cell.soma.record("external", braincell.observe.clamp_current())
        cell.soma.record("components", braincell.observe.clamp_current(reduce="none"))

        result = cell.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(result.samples["clamps"].time.to_decimal(u.ms), [0.05, 0.15, 0.25])
        np.testing.assert_allclose(
            result.samples["clamps"].values.to_decimal(u.nA),
            [[0.2, 0.3], [0.0, 0.3], [0.0, 0.0]],
        )
        np.testing.assert_allclose(
            result.samples["external"].values.to_decimal(u.nA)[:, 0],
            [0.5, 0.3, 0.0],
        )
        np.testing.assert_allclose(
            result.samples["components"].values.to_decimal(u.nA),
            result.samples["clamps"].values.to_decimal(u.nA),
        )
        self.assertEqual([row.clamp_id for row in result.samples["clamps"].schema.rows], [0, 1])

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


def _rows(result, name: str) -> list[tuple[int, int]]:
    return [(row.branch_id, row.cv_id) for row in result.samples[name].schema.rows]


def _run(cell):
    return cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)


def _soma_dend_tree() -> braincell.Morphology:
    """Soma 20 um (r 5 um) and a 100 um dendrite (r 1 um)."""
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
    tree = braincell.Morphology.from_root(soma, name="soma")
    tree.soma.dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"
    )
    return tree


def _basal(length: float = 50.0) -> braincell.Branch:
    return braincell.Branch.from_lengths(lengths=[length] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")


class RecordingFollowsItsDeclarationTest(unittest.TestCase):
    """MORPH-15: a recording declared before the grid or tree changed records what it named.

    Tree S: soma 0, dend 1, axon 2 (on the soma's proximal end). The
    recording used to keep the CV ids it resolved when declared, so after a
    new ``cv_policy`` or an edit it recorded other CVs without a word.
    """

    def test_a_region_recording_follows_a_new_cv_policy(self) -> None:
        cell = braincell.Cell(make_tree_s())
        cell.on(branch_in("type", "axon")).record("av", braincell.observe.state("v"))
        cell.cv_policy = braincell.MaxCVLen(40.0 * u.um)

        self.assertEqual(_rows(_run(cell), "av"), [(2, 4)])

    def test_a_region_recording_includes_a_branch_appended_later(self) -> None:
        morpho = make_tree_s()
        cell = braincell.Cell(morpho)
        cell.on(branch_in("type", "basal_dendrite")).record("dv", braincell.observe.state("v"))
        morpho.soma.dend2 = _basal()

        self.assertEqual(_rows(_run(cell), "dv"), [(1, 1), (3, 3)])

    def test_a_type_selection_follows_a_retyped_branch_on_the_same_grid(self) -> None:
        # Replacing dend by an apical branch of the same length keeps every
        # CV (CVPerBranch) but changes which CVs the type selection names.
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.apical_dendrite.record("ap", braincell.observe.state("v"))
        morpho.replace(
            "dend",
            braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite"),
        )

        self.assertEqual(_rows(_run(cell), "ap"), [(1, 1), (3, 3)])

    def test_population_and_location_selections_are_kept(self) -> None:
        morpho = make_tree_s()
        cell = braincell.Cell(morpho, pop_size=(3,))
        cell[[2, 0]].loc(at("axon", 0.5)).record("ax", braincell.observe.state("v"))
        cell.cv_policy = braincell.MaxCVLen(40.0 * u.um)

        rows = _run(cell).samples["ax"].schema.rows
        self.assertEqual([(row.population_index, row.branch_id, row.cv_id) for row in rows], [(2, 2, 4), (0, 2, 4)])

    def test_an_unchanged_grid_keeps_the_declared_selection(self) -> None:
        cell = braincell.Cell(make_tree_s())
        spec = cell.cv[1].record("c1", braincell.observe.state("v"))
        cell.paint(braincell.filter.AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2))

        self.assertEqual(_rows(_run(cell), "c1"), [(1, 1)])
        self.assertIs(cell.recordings["c1"], spec)

    def test_a_cv_recording_raises_on_a_new_grid(self) -> None:
        cell = braincell.Cell(_soma_dend_tree())
        cell.cv[1].record("c1", braincell.observe.state("v"))
        cell.cv_policy = braincell.MaxCVLen(20.0 * u.um)
        self.assertEqual(cell.n_cv, 6)

        pattern = r"Recording 'c1' .*cv\[\.\.\.\].*grid"
        with self.assertRaisesRegex(RuntimeError, pattern):
            cell.init_state()
        self.assertFalse(cell._initialized)
        with self.assertRaisesRegex(RuntimeError, pattern):
            _run(cell)
        # The recovery the message names: the grid it was declared on.
        cell.cv_policy = braincell.CVPerBranch()
        self.assertEqual(_rows(_run(cell), "c1"), [(1, 1)])

    def test_a_branch_recording_stays_on_its_branch_after_scaling(self) -> None:
        # MORPH-3b: scale(2) doubles every length, so MaxCVLen(40 um) cuts
        # more CVs and d1's CV ids move.
        morpho = make_tree_t()
        cell = braincell.Cell(morpho, cv_policy=braincell.MaxCVLen(40.0 * u.um))
        cell.branch["d1"].record("d1v", braincell.observe.state("v"))
        self.assertEqual([cv.id for cv in cell.cvs if cv.branch_id == 2], [4, 5, 6])

        morpho.scale(2)

        d1 = [cv.id for cv in cell.cvs if cv.branch_id == 2]
        self.assertEqual(d1, D1_AFTER_SCALE)
        self.assertEqual(_rows(_run(cell), "d1v"), [(2, cv_id) for cv_id in d1])

    def test_a_cv_recording_is_kept_when_an_edit_keeps_the_grid(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.cv[1].record("c1", braincell.observe.state("v"))
        morpho.replace(
            "dend",
            braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite"),
        )

        self.assertEqual(_rows(_run(cell), "c1"), [(1, 1)])

    def test_an_expression_on_a_removed_branch_names_the_recording(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.loc(at("dend", 0.5)).record("dv", braincell.observe.state("v"))
        morpho.remove("dend")

        with self.assertRaisesRegex(KeyError, r"Recording 'dv' selects branch 'dend'.*does not have"):
            cell.init_state()
        # Reusing the name gives the recording a branch to follow again.
        morpho.soma.dend = _basal()
        self.assertEqual(_rows(_run(cell), "dv"), [(3, 3)])

    def test_a_mask_recording_raises_once_a_removal_moves_its_indices(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.loc(at("axon", 0.5).evaluate(morpho)).record("ax", braincell.observe.state("v"))
        morpho.soma.extra = _basal()
        self.assertEqual(_rows(_run(cell), "ax"), [(2, 2)])

        cell.reset()
        morpho.remove("dend")
        with self.assertRaisesRegex(RuntimeError, r"Recording 'ax' .*LocsetMask.*removed or moved"):
            cell.init_state()


if __name__ == "__main__":
    unittest.main()
