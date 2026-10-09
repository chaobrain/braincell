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

import dataclasses
import unittest

import braincell
import brainunit as u
import numpy as np

from braincell.filter import at
from braincell.reduction.dbnn import layout


def _cell(*synapses, pop_size=(1,)):
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        cv_policy=braincell.CVPerBranch(),
        pop_size=pop_size,
    )
    if synapses:
        cell.place(at("soma", 0.5), *synapses)
    return cell


def _layout(synapse_ids, *, branch_ids=(), branch_xs=()):
    spec = braincell.mech.Synapse("ExpSyn", name="E")
    count = len(synapse_ids)
    return layout.ChannelLayout(
        (spec,),
        tuple(synapse_ids),
        tuple(range(count)),
        (0,) * count,
        (0,) * count,
        branch_ids=tuple(branch_ids),
        branch_xs=tuple(branch_xs),
    )


class LayoutBoundaryTest(unittest.TestCase):
    def test_module_is_importable(self):
        self.assertIn("logical synapses", layout.__doc__)

    def test_build_layout_follows_logical_order_and_round_trips(self):
        excitatory = braincell.mech.Synapse("ExpSyn", name="E", tau=2.0 * u.ms, e=0.0 * u.mV)
        inhibitory = braincell.mech.Synapse("ExpSyn", name="I", tau=2.0 * u.ms, e=-80.0 * u.mV)
        cell = _cell(excitatory, inhibitory)
        actual = layout.build_channel_layout(cell, (excitatory, inhibitory))
        self.assertEqual(actual.n_channels, 2)
        point_id = int(cell.synapses.point_id[0])
        self.assertEqual(actual.synapse_ids, tuple(cell.synapses.id))
        self.assertEqual(actual.point_ids, (point_id, point_id))
        self.assertEqual(actual.branch_ids, tuple(int(value) for value in cell.synapses.branch_id))
        self.assertEqual(actual.branch_xs, (0.5, 0.5))
        self.assertEqual(actual.channel_id(int(cell.synapses.id[0])), 0)
        self.assertEqual(actual.channel_id(int(cell.synapses.id[1])), 1)
        with self.assertRaisesRegex(ValueError, "logical synapse"):
            actual.channel_id(int(max(cell.synapses.id)) + 1)
        self.assertEqual(actual.synapse_id(1), int(cell.synapses.id[1]))
        self.assertEqual(actual.placement_id(1), int(cell.synapses.placement_id[1]))
        self.assertEqual(actual.point_id(1), point_id)
        self.assertEqual(actual.spec_index(1), 1)
        self.assertEqual(actual.polarity(1), "I")
        restored = layout.ChannelLayout.from_dict(actual.to_dict())
        self.assertEqual(restored, actual)
        self.assertEqual(restored.fingerprint, actual.fingerprint)

        incomplete = actual.to_dict()
        del incomplete["branch_ids"]
        with self.assertRaises(KeyError):
            layout.ChannelLayout.from_dict(incomplete)

    def test_same_point_and_prototype_logical_synapses_remain_distinct_channels(self):
        synapse = braincell.mech.Synapse("ExpSyn", name="E")
        cell = _cell(synapse, synapse)
        actual = layout.build_channel_layout(cell, (synapse,))
        self.assertEqual(actual.n_channels, 2)
        self.assertEqual(actual.point_id(0), actual.point_id(1))
        self.assertNotEqual(actual.synapse_id(0), actual.synapse_id(1))
        teacher = _cell(synapse, synapse, pop_size=(2,))
        channel_ids = layout.validate_channel_coverage(actual, 0, teacher.synapses[synapse])
        np.testing.assert_array_equal(channel_ids, [0, 1, 0, 1])

    def test_unmatched_cell_synapse_is_rejected(self):
        expected = braincell.mech.Synapse("ExpSyn", name="E")
        cell = _cell(braincell.mech.Synapse("ExpSyn", name="unknown"))
        with self.assertRaisesRegex(ValueError, "no matching DBNN specs"):
            layout.build_channel_layout(cell, (expected,))

    def test_mismatched_synapse_setting_is_rejected(self):
        expected = braincell.mech.Synapse("ExpSyn", name="E", tau=2.0 * u.ms)
        cell = _cell(braincell.mech.Synapse("ExpSyn", name="E", tau=3.0 * u.ms))
        with self.assertRaisesRegex(ValueError, "does not match"):
            layout.build_channel_layout(cell, (expected,))

    def test_layout_resolves_registry_defaults_and_units(self):
        spec = braincell.mech.Synapse("ExpSyn", name="E", tau=2000.0 * u.us)
        actual = layout.build_channel_layout(_cell(spec), (spec,))
        self.assertEqual(actual.specs[0].params["e"], 0.0 * u.mV)
        self.assertEqual(actual.specs[0].params["tau"], 2.0 * u.ms)

    def test_reference_weights_must_be_strictly_positive(self):
        spec = braincell.mech.Synapse("ExpSyn", name="E")
        with self.assertRaisesRegex(ValueError, "positive"):
            layout.build_channel_layout(_cell(spec), (spec,), reference_weights_us=(0.0,))

    def test_polarity_is_derived_from_reversal_potential_threshold(self):
        boundary = braincell.mech.Synapse("ExpSyn", name="boundary", e=-50.0 * u.mV)
        inhibitory = braincell.mech.Synapse("ExpSyn", name="inhibitory", e=-50.1 * u.mV)
        actual = layout.build_channel_layout(_cell(boundary, inhibitory), (boundary, inhibitory))
        self.assertEqual(actual.polarity(0), "E")
        self.assertEqual(actual.polarity(1), "I")

    def test_layout_module_exports_alignment_api(self):
        self.assertNotIn("SynapsePrototype", layout.__all__)
        self.assertNotIn("SynapseChannel", layout.__all__)
        self.assertIn("ChannelAlignment", layout.__all__)
        self.assertIn("align_channels", layout.__all__)

    def test_channel_coverage_uses_resolved_population_and_point_columns(self):
        spec = braincell.mech.Synapse("ExpSyn", name="E")
        template = _cell(spec)
        channel_layout = layout.build_channel_layout(template, (spec,))
        cell = _cell(spec, pop_size=(2,))
        channel_ids = layout.validate_channel_coverage(channel_layout, 0, cell.synapses[spec])
        np.testing.assert_array_equal(channel_ids, [0, 0])
        with self.assertRaisesRegex(ValueError, "missing"):
            layout.validate_channel_coverage(channel_layout, 0, cell[0].synapses[spec])

    def test_build_layout_requires_single_member_template(self):
        spec = braincell.mech.Synapse("ExpSyn", name="E")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            layout.build_channel_layout(
                _cell(spec, pop_size=(2,)),
                (spec,),
            )

    def test_packed_pair_index_matches_upper_triangle_order(self):
        actual = [layout.packed_pair_index(i, j, 4) for i in range(4) for j in range(i + 1, 4)]
        self.assertEqual(actual, list(range(6)))

    def test_location_columns_must_be_empty_or_complete_and_valid(self):
        with self.assertRaisesRegex(ValueError, "both be empty or complete"):
            _layout((1,), branch_ids=(0,))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _layout((1,), branch_ids=(-1,), branch_xs=(0.5,))
        with self.assertRaisesRegex(TypeError, "only integers"):
            _layout((1,), branch_ids=(0.5,), branch_xs=(0.5,))
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                _layout((1,), branch_ids=(0,), branch_xs=(invalid,))
        with self.assertRaisesRegex(TypeError, "only integers"):
            layout.ChannelLayout(
                (braincell.mech.Synapse("ExpSyn", name="E"),),
                (1.5,),
                (0,),
                (0,),
                (0,),
            )


class ChannelAlignmentTest(unittest.TestCase):
    def test_exact_fingerprint_uses_identity(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 1), branch_xs=(0.2, 0.8))
        actual = layout.align_channels(checkpoint, layout.ChannelLayout.from_dict(checkpoint.to_dict()))
        self.assertEqual(actual.method, "identity")
        self.assertEqual(actual.permutation, (0, 1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            actual.method = "branch_position"

    def test_nearby_branch_positions_align_by_location(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 0), branch_xs=(0.2, 0.8))
        candidate = _layout((900, 700), branch_ids=(0, 0), branch_xs=(0.81, 0.21))
        actual = layout.align_channels(checkpoint, candidate)
        self.assertEqual(actual.method, "branch_position")
        self.assertEqual(actual.permutation, (1, 0))

    def test_duplicate_positions_and_ties_are_deterministic(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 0), branch_xs=(0.5, 0.5))
        candidate = _layout((20, 10), branch_ids=(0, 0), branch_xs=(0.5, 0.5))
        actual = layout.align_channels(checkpoint, candidate)
        self.assertEqual(actual.method, "branch_position")
        self.assertEqual(actual.permutation, (1, 0))

    def test_distant_positions_fall_back_to_identifiers(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 0), branch_xs=(0.2, 0.8))
        candidate = _layout((20, 10), branch_ids=(0, 0), branch_xs=(0.3, 0.7))
        actual = layout.align_channels(checkpoint, candidate)
        self.assertEqual(actual.method, "identifier_distance")
        self.assertEqual(actual.permutation, (1, 0))

    def test_branch_distance_threshold_is_inclusive_despite_float_roundoff(self):
        checkpoint = _layout((10,), branch_ids=(0,), branch_xs=(0.15,))
        candidate = _layout((20,), branch_ids=(0,), branch_xs=(0.20,))
        self.assertEqual(layout.align_channels(checkpoint, candidate).method, "branch_position")

    def test_group_mismatch_falls_back_to_identifiers(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 0), branch_xs=(0.2, 0.8))
        candidate = _layout((20, 10), branch_ids=(0, 1), branch_xs=(0.8, 0.2))
        actual = layout.align_channels(checkpoint, candidate)
        self.assertEqual(actual.method, "identifier_distance")
        self.assertEqual(actual.permutation, (1, 0))

    def test_missing_locations_fall_back_to_identifiers(self):
        checkpoint = _layout((10, 20), branch_ids=(0, 0), branch_xs=(0.2, 0.8))
        candidate = _layout((20, 10))
        actual = layout.align_channels(checkpoint, candidate)
        self.assertEqual(actual.method, "identifier_distance")
        self.assertEqual(actual.permutation, (1, 0))

    def test_counts_thresholds_and_permutations_are_validated(self):
        one = _layout((1,))
        two = _layout((1, 2))
        with self.assertRaisesRegex(ValueError, "equal channel counts"):
            layout.align_channels(one, two)
        with self.assertRaisesRegex(TypeError, "ChannelLayout"):
            layout.align_channels(object(), one)
        for invalid in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite and positive"):
                layout.align_channels(one, one, invalid)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            layout.ChannelAlignment("identity", (0, 0))
        with self.assertRaisesRegex(TypeError, "only integers"):
            layout.ChannelAlignment("identity", (0.5,))
