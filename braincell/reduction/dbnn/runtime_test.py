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
from unittest import mock

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.mech import ScalarEventInput, Synapse, TriggerEventInput
from braincell.reduction.core import (
    ReductionContext,
    ReductionInputGroup,
    ReductionInputGroupSchema,
    ReductionInputs,
    ReductionSynapse,
)
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import DBNN
from braincell.reduction.dbnn.runtime import DBNNReduction


class _Owner:
    pop_size = (1,)


def _context(event_input=ScalarEventInput(u.uS)):
    synapse = ReductionSynapse(0, 0, 0, 0, 0, 0, 0, 0.5, "input", "ExpSyn")
    schema = ReductionInputGroupSchema(0, "ExpSyn", event_input, np.asarray([0]), np.asarray([0]), np.asarray([0]))
    return ReductionContext.with_cell(_Owner(), synapses=(synapse,), input_groups=(schema,), fingerprint="dbnn-test")


class _TwoMemberOwner:
    pop_size = (2,)


def _two_member_context():
    synapses = tuple(
        ReductionSynapse(index, 0, index, 0, 0, 0, 0, 0.5, "input", "ExpSyn") for index in range(2)
    )
    schema = ReductionInputGroupSchema(
        0, "ExpSyn", ScalarEventInput(u.uS), np.asarray([0, 1]), np.asarray([0, 0]), np.asarray([0, 1])
    )
    return ReductionContext.with_cell(_TwoMemberOwner(), synapses=synapses, input_groups=(schema,), fingerprint="dbnn-test")


def _model(reference_weight_us=1.0):
    layout = ChannelLayout(
        (Synapse("ExpSyn", name="input"),),
        (0,),
        (0,),
        (0,),
        (0,),
        (reference_weight_us,),
    )
    return DBNN(layout, mode="r", dt=1.0 * u.ms)


class DBNNReductionTest(unittest.TestCase):
    def test_routes_scalar_payload_to_dbnn_channel(self):
        context = _context()
        reduction = DBNNReduction(_model())
        reduction.init_state(context)
        output = reduction.update(
            ReductionInputs((ReductionInputGroup(context.input_groups[0], jnp.asarray([1.0]) * u.uS),))
        )

        self.assertEqual(output.values["voltage"].shape, (1,))
        self.assertEqual(output.event.shape, (1,))

    def test_normalizes_payload_by_layout_reference_weight(self):
        context = _context()
        model = _model(reference_weight_us=0.2)
        reduction = DBNNReduction(model)
        reduction.init_state(context)

        with mock.patch.object(model, "update", wraps=model.update) as update:
            reduction.update(
                ReductionInputs((ReductionInputGroup(context.input_groups[0], jnp.asarray([0.2]) * u.uS),))
            )

        np.testing.assert_allclose(update.call_args.args[0], [[1.0]])

    def test_rejects_non_scalar_event_input(self):
        with self.assertRaisesRegex(TypeError, "ScalarEventInput"):
            DBNNReduction(_model()).init_state(_context(TriggerEventInput()))

    def test_requires_matching_logical_channel_count(self):
        context = _context()
        layout = ChannelLayout((Synapse("ExpSyn", name="first"), Synapse("ExpSyn", name="second")), (0, 1), (0, 1), (0, 1), (0, 1))
        with self.assertRaisesRegex(ValueError, "channels"):
            DBNNReduction(DBNN(layout, mode="r", dt=1.0 * u.ms)).init_state(context)

    def test_routes_each_population_member_to_its_shared_layout_channel(self):
        context = _two_member_context()
        model = _model(reference_weight_us=0.2)
        reduction = DBNNReduction(model)
        reduction.init_state(context)

        with mock.patch.object(model, "update", wraps=model.update) as update:
            reduction.update(
                ReductionInputs((ReductionInputGroup(context.input_groups[0], jnp.asarray([0.2, 0.4]) * u.uS),))
            )

        np.testing.assert_allclose(update.call_args.args[0], [[1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
