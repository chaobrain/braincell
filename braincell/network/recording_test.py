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

import gc
import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell._multi_compartment.selection_test import _GRID_SELECTIONS, _cell
from braincell.filter import BranchSlice, RootLocation, at, branch_in
from braincell.morph._testing import make_tree_s, make_tree_t, make_tree_u
from braincell.network.recording import EventSeries, SampleBlock


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
        cell.soma.record("v", braincell.observe.state("V"))

        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

        self.assertEqual(result.samples["v"].values.shape, (2, 1))

    def test_state_recording_uses_global_half_open_schedule(self) -> None:
        cell = _cell(2)
        before_placements = len(cell.point_placements)
        cell[0].soma.record("soma_v", braincell.observe.state("V"), period=0.1 * u.ms)
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
        cell.soma.record("fast", braincell.observe.state("V"), period=0.1 * u.ms)
        cell.soma.record("slow", braincell.observe.state("V"), period=0.2 * u.ms)

        first = cell.run(dt=0.05 * u.ms, duration=0.4 * u.ms)
        second = cell.run(dt=0.05 * u.ms, duration=0.4 * u.ms)

        self.assertEqual(first.samples["fast"].values.shape, (4, 1))
        self.assertEqual(first.samples["slow"].values.shape, (2, 1))
        np.testing.assert_allclose(first.samples["fast"].time.to_decimal(u.ms), [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(second.samples["slow"].time.to_decimal(u.ms), [0.4, 0.6])

    def test_unaligned_eager_recording_warns_and_keeps_variable_length_results(self) -> None:
        cell = _cell(1)
        cell.soma.record("v", braincell.observe.state("V"), period=0.1 * u.ms)

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
            braincell.observe.state("V"),
            period=0.1 * u.ms,
            start=0.1 * u.ms,
        )

        result = cell.run(dt=0.05 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(result.samples["v"].time.to_decimal(u.ms), [0.1, 0.2])

    def test_split_result_concatenates_to_single_run(self) -> None:
        split = _cell(1)
        split.soma.record("v", braincell.observe.state("V"))
        parts = (
            split.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
            split.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
        )
        joined = braincell.RunResult.concat(parts)

        full = _cell(1)
        full.soma.record("v", braincell.observe.state("V"))
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
                braincell.observe.state("V"),
                period=0.1 * u.ms,
                frequency=10.0 * u.kHz,
            )
        cell.record("v", braincell.observe.state("V"))
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
        cell.on(branch_in("type", "axon")).record("av", braincell.observe.state("V"))
        cell.cv_policy = braincell.MaxCVLen(40.0 * u.um)

        self.assertEqual(_rows(_run(cell), "av"), [(2, 4)])

    def test_a_region_recording_includes_a_branch_appended_later(self) -> None:
        morpho = make_tree_s()
        cell = braincell.Cell(morpho)
        cell.on(branch_in("type", "basal_dendrite")).record("dv", braincell.observe.state("V"))
        morpho.soma.dend2 = _basal()

        self.assertEqual(_rows(_run(cell), "dv"), [(1, 1), (3, 3)])

    def test_a_type_selection_follows_a_retyped_branch_on_the_same_grid(self) -> None:
        # Replacing dend by an apical branch of the same length keeps every
        # CV (CVPerBranch) but changes which CVs the type selection names.
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.apical_dendrite.record("ap", braincell.observe.state("V"))
        morpho.replace(
            "dend",
            braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite"),
        )

        self.assertEqual(_rows(_run(cell), "ap"), [(1, 1), (3, 3)])

    def test_population_and_location_selections_are_kept(self) -> None:
        morpho = make_tree_s()
        cell = braincell.Cell(morpho, pop_size=(3,))
        cell[[2, 0]].loc(at("axon", 0.5)).record("ax", braincell.observe.state("V"))
        cell.cv_policy = braincell.MaxCVLen(40.0 * u.um)

        rows = _run(cell).samples["ax"].schema.rows
        self.assertEqual([(row.population_index, row.branch_id, row.cv_id) for row in rows], [(2, 2, 4), (0, 2, 4)])

    def test_an_unchanged_grid_keeps_the_declared_selection(self) -> None:
        cell = braincell.Cell(make_tree_s())
        spec = cell.cv[1].record("c1", braincell.observe.state("V"))
        cell.paint(braincell.filter.AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2))

        self.assertEqual(_rows(_run(cell), "c1"), [(1, 1)])
        self.assertIs(cell.recordings["c1"], spec)

    def test_a_cv_recording_raises_on_a_new_grid(self) -> None:
        cell = braincell.Cell(_soma_dend_tree())
        cell.cv[1].record("c1", braincell.observe.state("V"))
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
        cell.branch["d1"].record("d1v", braincell.observe.state("V"))
        self.assertEqual([cv.id for cv in cell.cvs if cv.branch_id == 2], [4, 5, 6])

        morpho.scale(2)

        d1 = [cv.id for cv in cell.cvs if cv.branch_id == 2]
        self.assertEqual(d1, [8, 9, 10])
        self.assertEqual(_rows(_run(cell), "d1v"), [(2, cv_id) for cv_id in d1])

    def test_a_cv_recording_is_kept_when_an_edit_keeps_the_grid(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.cv[1].record("c1", braincell.observe.state("V"))
        morpho.replace(
            "dend",
            braincell.Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite"),
        )

        self.assertEqual(_rows(_run(cell), "c1"), [(1, 1)])

    def test_an_expression_on_a_removed_branch_names_the_recording(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.loc(at("dend", 0.5)).record("dv", braincell.observe.state("V"))
        morpho.remove("dend")

        with self.assertRaisesRegex(KeyError, r"Recording 'dv' selects branch 'dend'.*does not have"):
            cell.init_state()
        # Reusing the name gives the recording a branch to follow again.
        morpho.soma.dend = _basal()
        self.assertEqual(_rows(_run(cell), "dv"), [(3, 3)])

    def test_a_mask_recording_raises_once_a_removal_moves_its_indices(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.loc(at("axon", 0.5).evaluate(morpho)).record("ax", braincell.observe.state("V"))
        morpho.soma.extra = _basal()
        self.assertEqual(_rows(_run(cell), "ax"), [(2, 2)])

        cell.reset()
        morpho.remove("dend")
        with self.assertRaisesRegex(RuntimeError, r"Recording 'ax' .*LocsetMask.*removed or moved"):
            cell.init_state()


class RecordingByIndexTest(unittest.TestCase):
    """MORPH-17: a recording whose expression names a branch by index raises once an edit shifts it.

    Tree U: soma 0, dend 1, axon 2, apical 3; removing ``dend`` puts apical
    on 2, where a recording of ``BranchSlice(2, ...)`` would have moved.
    """

    _STALE = r"Recording '{}' selects .*by default index 2, .*removed or moved.*Record on a new Cell"

    def test_a_removal_before_init_raises_at_init_state(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.on(BranchSlice(2, 0.0, 1.0)).record("ax", braincell.observe.state("V"))
        cell.loc(at(2, 0.5)).record("ax_mid", braincell.observe.state("V"))

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, self._STALE.format("ax")):
            cell.init_state()
        self.assertFalse(cell._initialized)
        self.assertIs(cell.morpho, morpho)
        # The recovery the message names: record on a new Cell, by name.
        fresh = braincell.Cell(morpho)
        fresh.branch["axon"].record("ax", braincell.observe.state("V"))
        self.assertEqual(_rows(_run(fresh), "ax"), [(1, 1)])

    def test_a_removal_after_init_raises_after_reset(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.branch[2].record("ax", braincell.observe.state("V"))
        self.assertEqual(_rows(_run(cell), "ax"), [(2, 2)])

        morpho.remove("dend")

        # The initialized Cell runs its snapshot, where index 2 is the axon.
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(_rows(_run(cell), "ax"), [(2, 2)])
        cell.reset()
        with self.assertRaisesRegex(ValueError, self._STALE.format("ax")):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_an_index_recording_survives_an_append_and_a_new_policy(self) -> None:
        morpho = make_tree_u()
        cell = braincell.Cell(morpho)
        cell.on(BranchSlice(2, 0.0, 1.0)).record("ax", braincell.observe.state("V"))

        morpho.soma.extra = _basal()
        cell.cv_policy = braincell.CVPerBranch(2)

        self.assertEqual(_rows(_run(cell), "ax"), [(2, 4), (2, 5)])

    def test_network_init_state_initializes_no_cell(self) -> None:
        morpho = make_tree_u()
        stale = braincell.Cell(morpho)
        stale.loc(at(2, 0.5)).record("ax", braincell.observe.state("V"))
        other = braincell.Cell(make_tree_u())
        network = braincell.Network("n")
        network.add_population("a", other)
        network.add_population("b", stale)

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, self._STALE.format("ax")):
            network.init_state()
        self.assertFalse(other._initialized)
        self.assertFalse(stale._initialized)


def _leaky(kind: str) -> braincell.Cell:
    """Three leaky members, lumped or on the two-branch ``_cell`` tree, away from rest."""
    if kind == "lumped":
        cell = braincell.Cell.point(3, V_init=-65.0 * u.mV)
    else:
        cell = _cell(3)
        cell.V_init = -65.0 * u.mV
    cell.paint(braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV))
    return cell


class VoltageNameTest(unittest.TestCase):
    """CELL-26: ``"V"``, as in ``cell.V``, is the one name of the membrane voltage in recordings."""

    def test_the_shorthand_records_what_observe_state_records(self) -> None:
        for kind in ("lumped", "cable"):
            blocks = {}
            for label, observable in (("state('V')", braincell.observe.state("V")), ("'V'", "V")):
                cell = _leaky(kind)
                cell.record("V", observable)
                blocks[label] = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms).samples["V"]
            reference, shorthand = blocks["state('V')"], blocks["'V'"]
            with self.subTest(kind=kind):
                self.assertEqual(reference.values.shape, (20, 3 if kind == "lumped" else 12))
                values = reference.values.to_decimal(u.mV)
                self.assertFalse(np.allclose(values[0], values[-1]))
                self.assertEqual({row.field for row in reference.schema.rows}, {"V"})
                self.assertEqual(shorthand.schema, reference.schema)
                np.testing.assert_array_equal(shorthand.values.to_decimal(u.mV), values)

    def test_views_take_the_shorthand(self) -> None:
        cell = _leaky("cable")
        cell.soma.record("soma", "V")
        cell[0].record("first", "V")
        cell.soma.record("soma_ref", braincell.observe.state("V"))
        cell[0].record("first_ref", braincell.observe.state("V"))
        result = cell.run(dt=0.05 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(_rows(result, "soma"), _rows(result, "soma_ref"))
        self.assertEqual(_rows(result, "first"), _rows(result, "first_ref"))
        self.assertEqual(result.samples["soma"].schema.size, 3)
        self.assertEqual(result.samples["first"].schema.size, 4)

    def test_the_spec_holds_the_observe_declaration(self) -> None:
        cell = _leaky("lumped")
        spec = cell.record("V", "V")
        self.assertEqual(spec.observable, braincell.observe.state("V"))
        self.assertEqual(cell.recordings["V"].observable, braincell.observe.state("V"))
        self.assertEqual(braincell.observe.state("V").field, "V")

    def test_the_removed_name_v_fails_with_the_new_name(self) -> None:
        cell = _leaky("lumped")
        removed = r"Cell state 'v' was removed in 0\.3; record the membrane voltage as 'V', as in cell\.V"
        with self.assertRaisesRegex(KeyError, removed):
            braincell.observe.state("v")
        with self.assertRaisesRegex(KeyError, removed):
            cell.record("v", "v")
        self.assertEqual(dict(cell.recordings), {})

    def test_other_names_fail_where_they_are_written(self) -> None:
        cell = _leaky("lumped")
        message = r"Cell state 'voltage' is not recordable; record the membrane voltage as 'V'"
        with self.assertRaisesRegex(KeyError, message):
            braincell.observe.state("voltage")
        with self.assertRaisesRegex(KeyError, "Cell state 'x'"):
            braincell.observe.state("x")
        with self.assertRaisesRegex(KeyError, message):
            cell.record("a", "voltage")
        with self.assertRaisesRegex(TypeError, r"'V' or a braincell.observe declaration, got int"):
            cell.record("a", 3)
        self.assertEqual(dict(cell.recordings), {})
        # The name stays free for a valid declaration.
        cell.record("a", "V")
        self.assertEqual(tuple(cell.recordings), ("a",))


class ResultPytreeTest(unittest.TestCase):
    """CELL-26: SampleBlock and EventSeries are JAX pytrees with a static schema or metadata."""

    def _block(self) -> SampleBlock:
        cell = _leaky("lumped")
        cell.record("v", "V")
        return cell.run(dt=0.05 * u.ms, duration=0.5 * u.ms).samples["v"]

    @staticmethod
    def _events(cv_id=(0, 0)) -> EventSeries:
        return EventSeries(
            time=np.asarray([0.1, 0.3]) * u.ms,
            source_id=np.asarray([1, 0]),
            count=np.asarray([1, 2]),
            metadata={"port": "spike", "cv_id": np.asarray(cv_id)},
        )

    def test_a_sample_block_round_trips(self) -> None:
        block = self._block()
        leaves, treedef = jax.tree.flatten(block)
        self.assertIs(leaves[0], u.get_mantissa(block.values))
        rebuilt = jax.tree.unflatten(treedef, leaves)
        self.assertIsInstance(rebuilt, SampleBlock)
        self.assertIs(rebuilt.schema, block.schema)
        np.testing.assert_array_equal(rebuilt.time.to_decimal(u.ms), block.time.to_decimal(u.ms))
        self.assertIs(jax.block_until_ready(block), block)
        doubled = jax.tree.map(lambda x: 2 * x, block)
        np.testing.assert_allclose(doubled.values.to_decimal(u.mV), 2 * block.values.to_decimal(u.mV))

    def test_an_event_series_round_trips(self) -> None:
        events = self._events()
        leaves, treedef = jax.tree.flatten(events)
        self.assertEqual([np.shape(leaf) for leaf in leaves], [(2,)] * 3)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        self.assertIsInstance(rebuilt, EventSeries)
        np.testing.assert_array_equal(rebuilt.count, [1, 2])
        self.assertEqual(rebuilt.metadata["port"], "spike")
        with self.assertRaises(TypeError):
            rebuilt.metadata["port"] = "other"

    def test_equal_static_parts_give_equal_tree_structures(self) -> None:
        # Equal structures let jit reuse a trace for a result argument; the
        # schema holds Quantities and the metadata arrays, so both compare by value.
        self.assertEqual(jax.tree.structure(self._block()), jax.tree.structure(self._block()))
        self.assertEqual(hash(jax.tree.structure(self._events())), hash(jax.tree.structure(self._events())))
        self.assertEqual(jax.tree.structure(self._events()), jax.tree.structure(self._events()))
        self.assertNotEqual(jax.tree.structure(self._events()), jax.tree.structure(self._events(cv_id=(0, 1))))
        cell = _leaky("lumped")
        cell.record("w", "V")
        other = cell.run(dt=0.05 * u.ms, duration=0.5 * u.ms).samples["w"]
        self.assertNotEqual(jax.tree.structure(self._block()), jax.tree.structure(other))


# CELL-47: a population with several axes. Configuration L is a lumped HH
# cell, C a 3-CV HH soma cable; both get one clamp per member at the root
# midpoint (CV 0 of L, CV 1 of C). The twin is the same Cell with pop_size
# (6,) and the amplitudes flattened in C order.
_GRID_AMPLITUDES = {
    "lumped": np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]),
    "cable": np.asarray([[0.0, 0.15, 0.3], [0.3, 0.0, 0.15]]),
}
_GRID_N_CV = {"lumped": 1, "cable": 3}
_GRID_MID_CV = {"lumped": 0, "cable": 1}
_GRID_DT = 0.025 * u.ms
_SOMA_AREA_CM2 = 2.0 * np.pi * 10e-4 * 60e-4

_GRID_OBSERVABLES = {
    "V": lambda: "V",
    "na_p": lambda: braincell.observe.channel(name="Na_HH1952").state("p"),
    "k_p": lambda: braincell.observe.channel(type="K_HH1952").state("p"),
    "na_i": lambda: braincell.observe.channel(name="Na_HH1952").current(),
    "channel_i": lambda: braincell.observe.channel().current(reduce="none"),
    "ion_i": lambda: braincell.observe.ion(species="na").current(),
    "ion_E": lambda: braincell.observe.ion(species="na").state("E"),
    "membrane_i": lambda: braincell.observe.membrane_current(),
    "clamp_i": lambda: braincell.observe.clamp_current(),
    "clamp_each": lambda: braincell.observe.clamp_current(reduce="none"),
}


def _hh_population(kind: str, pop_size, amplitudes) -> braincell.Cell:
    if kind == "lumped":
        cell = braincell.Cell.point(
            pop_size, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4"
        )
    else:
        soma = braincell.Branch.from_lengths(lengths=[60.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = braincell.Cell(
            braincell.Morphology.from_root(soma),
            cv_policy=braincell.CVPerBranch(3),
            pop_size=pop_size,
            V_init=-65.0 * u.mV,
        )
    cell.paint(
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
        braincell.mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
        braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(
        RootLocation(0.5),
        braincell.mech.CurrentClamp(delay=1.0 * u.ms, durations=18.0 * u.ms, amplitudes=np.asarray(amplitudes) * u.nA),
    )
    return cell


def _bias(kind: str, current):
    """An input of ``current`` nA per member: a lumped Cell takes nA, a cable one a density on every point."""
    if kind == "lumped":
        return lambda V: u.Quantity(current, u.nA)
    density = np.asarray(current) / _SOMA_AREA_CM2 * 1e-3
    return lambda V: u.Quantity(np.broadcast_to(density[..., None], density.shape + (V.shape[-1],)), u.uA / u.cm**2)


def _values(block) -> np.ndarray:
    values = block.values
    return np.asarray(values.mantissa if isinstance(values, u.Quantity) else values)


def _row_keys(block) -> list[tuple]:
    return [(row.population_index, row.cv_id, row.mechanism_name, row.clamp_id) for row in block.schema.rows]


def _assert_twin(test, got, expected) -> None:
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12 * scale)


class MultiDimensionalPopulationTest(unittest.TestCase):
    """CELL-47: recording a Cell whose pop_size has several axes."""

    @classmethod
    def setUpClass(cls) -> None:
        # One 20 ms run of each configuration and its twin, kept as NumPy
        # arrays: no float64 program outlives this method (see conftest).
        cls.runs = {}
        with brainstate.environ.context(precision=64):
            for kind in ("lumped", "cable"):
                for pop_size in ((2, 3), (6,)):
                    cell = _hh_population(kind, pop_size, _GRID_AMPLITUDES[kind].reshape(pop_size))
                    for name, observable in _GRID_OBSERVABLES.items():
                        cell.record(name, observable())
                    part = cell[:, 1:] if pop_size == (2, 3) else cell[[1, 2, 4, 5]]
                    part.record("part_p", _GRID_OBSERVABLES["na_p"]())
                    part.record("part_membrane_i", _GRID_OBSERVABLES["membrane_i"]())
                    part.clamps.record("part_clamps")
                    if pop_size == (2, 3):
                        for index, (select, _, _) in enumerate(_GRID_SELECTIONS.values()):
                            select(cell).record(f"selection_{index}", "V")
                    result = cell.run(dt=_GRID_DT, duration=20.0 * u.ms)
                    cls.runs[kind, pop_size] = {
                        "values": {name: _values(block) for name, block in result.samples.items()},
                        "rows": {name: _row_keys(block) for name, block in result.samples.items()},
                        "spike_step": np.rint(np.asarray(result.spikes.time.to_decimal(u.ms)) / 0.025).astype(int),
                        "spike_source": np.asarray(result.spikes.source_id),
                        "spike_cv": np.asarray(result.spikes.metadata["cv_id"]),
                    }
                    del cell, result
        gc.collect()

    def setUp(self) -> None:
        self.enterContext(brainstate.environ.context(precision=64))

    def test_every_observable_matches_the_one_dimensional_twin(self) -> None:
        for kind in ("lumped", "cable"):
            grid, twin = self.runs[kind, (2, 3)], self.runs[kind, (6,)]
            for name in tuple(_GRID_OBSERVABLES) + ("part_p", "part_membrane_i", "part_clamps"):
                with self.subTest(kind=kind, recording=name):
                    self.assertEqual(grid["rows"][name], twin["rows"][name])
                    self.assertGreater(len(grid["rows"][name]), 0)
                    _assert_twin(self, grid["values"][name], twin["values"][name])

    def test_rows_are_members_in_c_order_then_cvs(self) -> None:
        for kind in ("lumped", "cable"):
            n_cv = _GRID_N_CV[kind]
            with self.subTest(kind=kind):
                rows = [(population, cv) for population, cv, _, _ in self.runs[kind, (2, 3)]["rows"]["V"]]
                self.assertEqual(rows, [(member, cv) for member in range(6) for cv in range(n_cv)])
                self.assertEqual(self.runs[kind, (2, 3)]["values"]["V"].shape, (800, 6 * n_cv))

    def test_spikes_cross_the_threshold_in_the_rows_of_their_members(self) -> None:
        for kind in ("lumped", "cable"):
            run, n_cv = self.runs[kind, (2, 3)], _GRID_N_CV[kind]
            V = run["values"]["V"]
            with self.subTest(kind=kind):
                self.assertEqual(np.bincount(run["spike_source"], minlength=6).tolist(), [0, 1, 2, 2, 0, 1])
                self.assertTrue(np.all(run["spike_cv"] == _GRID_MID_CV[kind]))
                columns = np.arange(6) * n_cv + _GRID_MID_CV[kind]
                for step, member in zip(run["spike_step"], run["spike_source"]):
                    self.assertLess(V[step, columns[member]], 0.0)
                    self.assertGreaterEqual(V[step + 1, columns[member]], 0.0)
                crossings = np.count_nonzero((V[:-1, columns] < 0.0) & (V[1:, columns] >= 0.0))
                self.assertEqual(crossings, len(run["spike_source"]))

    def test_member_selections_record_columns_of_the_whole_cell(self) -> None:
        for kind in ("lumped", "cable"):
            run, n_cv = self.runs[kind, (2, 3)], _GRID_N_CV[kind]
            whole = run["values"]["V"]
            for index, (label, (_, members, _)) in enumerate(_GRID_SELECTIONS.items()):
                with self.subTest(kind=kind, selection=label):
                    if "loc" in label:
                        columns = [member * n_cv + _GRID_MID_CV[kind] for member in members]
                    else:
                        columns = [member * n_cv + cv for member in members for cv in range(n_cv)]
                    rows = [(population, cv) for population, cv, _, _ in run["rows"][f"selection_{index}"]]
                    self.assertEqual([population * n_cv + cv for population, cv in rows], columns)
                    np.testing.assert_array_equal(run["values"][f"selection_{index}"], whole[:, columns])

    def test_a_third_axis_matches_its_twin(self) -> None:
        amplitudes = np.linspace(0.0, 0.1, 8)
        grid = _hh_population("lumped", (2, 2, 2), amplitudes.reshape(2, 2, 2))
        twin = _hh_population("lumped", (8,), amplitudes)
        grid.record("V", "V")
        grid[1, :, 0].record("part", "V")
        twin.record("V", "V")
        on_grid = grid.run(dt=_GRID_DT, duration=20.0 * u.ms)
        on_twin = twin.run(dt=_GRID_DT, duration=20.0 * u.ms)
        _assert_twin(self, _values(on_grid.samples["V"]), _values(on_twin.samples["V"]))
        np.testing.assert_array_equal(_values(on_grid.samples["part"]), _values(on_grid.samples["V"])[:, [4, 6]])
        np.testing.assert_array_equal(on_grid.spikes.source_id, on_twin.spikes.source_id)

    def test_the_batch_axis_comes_before_the_rows(self) -> None:
        for kind in ("lumped", "cable"):
            n_cv, extra = _GRID_N_CV[kind], np.zeros((2, 2, 3))
            extra[1] = _GRID_AMPLITUDES[kind][::-1]
            results = {}
            for pop_size in ((2, 3), (6,)):
                cell = _hh_population(kind, pop_size, _GRID_AMPLITUDES[kind].reshape(pop_size))
                cell.add_current_input("bias", _bias(kind, extra.reshape((2,) + pop_size)))
                cell.record("V", "V")
                cell.record("na_p", _GRID_OBSERVABLES["na_p"]())
                cell.init_state(batch_size=2)
                results[pop_size] = cell.run(dt=_GRID_DT, duration=20.0 * u.ms)
            grid, twin = results[(2, 3)], results[(6,)]
            with self.subTest(kind=kind):
                for name in ("V", "na_p"):
                    self.assertEqual(grid.samples[name].values.shape, (800, 2, 6 * n_cv))
                    _assert_twin(self, _values(grid.samples[name]), _values(twin.samples[name]))
                spikes = grid.spikes
                self.assertEqual(
                    np.bincount(spikes.source_id, minlength=12).tolist(), [0, 1, 2, 2, 0, 1, 2, 1, 2, 2, 1, 2]
                )
                np.testing.assert_array_equal(
                    spikes.metadata["batch_index"] * 6 + spikes.metadata["population_index"], np.arange(12)
                )
                V = _values(grid.samples["V"])
                steps = np.rint(np.asarray(spikes.time.to_decimal(u.ms)) / 0.025).astype(int)
                for step, source in zip(steps, spikes.source_id):
                    batch, member = divmod(int(source), 6)
                    column = member * n_cv + _GRID_MID_CV[kind]
                    self.assertLess(V[step, batch, column], 0.0)
                    self.assertGreaterEqual(V[step + 1, batch, column], 0.0)

    def test_segments_and_reset_state_give_the_same_rows(self) -> None:
        for kind in ("lumped", "cable"):
            cell = _hh_population(kind, (2, 3), _GRID_AMPLITUDES[kind])
            cell.record("V", "V")
            cell[:, 1].record("column", "V")
            whole = cell.run(dt=_GRID_DT, duration=20.0 * u.ms)
            cell.reset_state()
            joined = braincell.RunResult.concat(
                [cell.run(dt=_GRID_DT, duration=duration * u.ms) for duration in (7.0, 13.0)]
            )
            cell.reset_state()
            again = cell.run(dt=_GRID_DT, duration=20.0 * u.ms)
            with self.subTest(kind=kind):
                for name in ("V", "column"):
                    _assert_twin(self, _values(joined.samples[name]), _values(whole.samples[name]))
                    _assert_twin(self, _values(again.samples[name]), _values(whole.samples[name]))
                n_cv = _GRID_N_CV[kind]
                columns = [member * n_cv + cv for member in (1, 4) for cv in range(n_cv)]
                np.testing.assert_array_equal(_values(whole.samples["column"]), _values(whole.samples["V"])[:, columns])

    def test_float32_long_run_equals_its_segments(self) -> None:
        # 32000 steps in float32 (L77): segments and one run give the same rows.
        with brainstate.environ.context(precision=32):
            cell = _hh_population("lumped", (2, 3), _GRID_AMPLITUDES["lumped"])
            cell.record("V", "V")
            cell[:, 1].record("column", "V")
            whole = cell.run(dt=_GRID_DT, duration=800.0 * u.ms)
            cell.reset_state()
            joined = braincell.RunResult.concat(
                [cell.run(dt=_GRID_DT, duration=duration * u.ms) for duration in (300.0, 500.0)]
            )
            V, joined_V = _values(whole.samples["V"]), _values(joined.samples["V"])
            self.assertEqual((V.shape, V.dtype), ((32000, 6), np.float32))
            np.testing.assert_allclose(joined_V, V, rtol=0.0, atol=1e-6 * float(np.max(np.abs(V))))
            np.testing.assert_array_equal(_values(whole.samples["column"]), V[:, [1, 4]])
            del cell, whole, joined
        gc.collect()

    def test_a_new_grid_selects_the_declared_members_again(self) -> None:
        cell = _hh_population("cable", (2, 3), _GRID_AMPLITUDES["cable"])
        cell.record("V", "V")
        cell[1, 0:2].loc(RootLocation(0.5)).record("selected", "V")
        cell[:, 2].soma.record("column", "V")
        cell.cv_policy = braincell.CVPerBranch(5)
        result = cell.run(dt=_GRID_DT, duration=2.0 * u.ms)
        members = {name: [(row[0], row[1]) for row in _row_keys(result.samples[name])] for name in ("selected", "column")}
        self.assertEqual(members["selected"], [(3, 2), (4, 2)])
        self.assertEqual(members["column"], [(member, cv) for member in (2, 5) for cv in range(5)])
        whole = _values(result.samples["V"])
        np.testing.assert_array_equal(_values(result.samples["selected"]), whole[:, [3 * 5 + 2, 4 * 5 + 2]])

    def test_empty_selections_and_absent_synapses_record_no_rows(self) -> None:
        cell = _hh_population("lumped", (2, 3), _GRID_AMPLITUDES["lumped"])
        cell[2:].record("none", "V")
        cell[:, 3:].record("no_column", "V")
        cell.record("synapses", braincell.observe.synapse().state("g"))
        result = cell.run(dt=_GRID_DT, duration=0.1 * u.ms)
        for name in ("none", "no_column", "synapses"):
            with self.subTest(recording=name):
                self.assertEqual(result.samples[name].values.shape, (4, 0))


class BatchChannelStateTest(unittest.TestCase):
    """CELL-47: with a batch axis, a channel-state row holds its own member in every batch element.

    On main ``b43893a`` a lumped Cell of 3 members with batch 2 recorded
    ``(200, 3, 2, 3)``, and a cable Cell raised ``ValueError`` at its first
    run: the member was taken from the leading (batch) axis.
    """

    def setUp(self) -> None:
        self.enterContext(brainstate.environ.context(precision=64))

    def test_each_batch_element_matches_its_own_run(self) -> None:
        for kind in ("lumped", "cable"):
            for n in (3, 2):
                amplitudes = np.linspace(0.0, 0.1 if kind == "lumped" else 0.3, n)
                extra = np.stack([np.zeros(n), amplitudes[::-1]])
                references, last = [], []
                for element in range(2):
                    cell = _hh_population(kind, (n,), amplitudes)
                    cell.add_current_input("bias", _bias(kind, extra[element]))
                    cell.record("na_p", _GRID_OBSERVABLES["na_p"]())
                    references.append(_values(cell.run(dt=_GRID_DT, duration=5.0 * u.ms).samples["na_p"]))
                    last.append(np.asarray(cell[[0, n - 1]].channels.by_name("Na_HH1952").get("p")))
                cell = _hh_population(kind, (n,), amplitudes)
                cell.add_current_input("bias", _bias(kind, extra))
                cell.record("na_p", _GRID_OBSERVABLES["na_p"]())
                cell.init_state(batch_size=2)
                got = _values(cell.run(dt=_GRID_DT, duration=5.0 * u.ms).samples["na_p"])
                read = np.asarray(cell[[0, n - 1]].channels.by_name("Na_HH1952").get("p"))
                with self.subTest(kind=kind, members=n):
                    self.assertEqual(got.shape, (200, 2, n * _GRID_N_CV[kind]))
                    self.assertEqual(read.shape, (2 * _GRID_N_CV[kind], 2))
                    self.assertGreater(float(np.max(np.abs(references[0] - references[1]))), 0.5)
                    for element in range(2):
                        np.testing.assert_allclose(got[:, element], references[element], rtol=1e-12, atol=0.0)
                        np.testing.assert_allclose(read[:, element], last[element], rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
