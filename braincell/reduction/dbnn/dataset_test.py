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
from pathlib import Path
import tempfile

import braincell
import brainunit as u
import numpy as np

from braincell.reduction.dbnn import dataset
from braincell.reduction.dbnn.layout import build_channel_layout
from braincell.reduction.dbnn.stimulus import generate_pair_protocol
from braincell.reduction.dbnn.stimulus import SparseEvents, StimulusPlan
from braincell.filter import at


def _batch():
    plan = StimulusPlan(
        layout_fingerprint="f" * 64,
        n_traces=1,
        events=SparseEvents(np.asarray([0, 0]), np.asarray([0, 1]), np.asarray([0.0, 1.0])),
        channel_weights=np.asarray([[2.0, 3.0]], dtype=np.float32),
        duration_ms=3.0,
        seeds=(1,),
        protocol_labels=("pair",),
        split="train",
    )
    return dataset.DatasetBatch(
        plan,
        np.asarray([[-70.0, -69.0, -68.0, -67.0]], dtype=np.float32),
        (np.asarray([2.0], dtype=np.float32),),
        np.arange(4, dtype=np.float32),
        {"input_alignment": "target-step"},
    )


class DatasetBoundaryTest(unittest.TestCase):
    def test_module_is_importable(self):
        self.assertIn("dataset", dataset.__doc__)

    def test_rasterize_alignment_and_padding(self):
        actual = np.asarray(dataset.rasterize_events(_batch().plan, dt_ms=1.0, input_alignment="target-step"))
        np.testing.assert_array_equal(actual[0, 0], [0.0, 2.0, 0.0, 0.0])
        np.testing.assert_array_equal(actual[0, 1], [0.0, 0.0, 3.0, 0.0])

    def test_dataset_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            dataset.save_dataset(path, _batch())
            actual = dataset.load_dataset(path)
        np.testing.assert_array_equal(actual.voltage_mv, _batch().voltage_mv)
        np.testing.assert_array_equal(actual.spike_times_ms[0], [2.0])
        self.assertEqual(actual.metadata, _batch().metadata)
        self.assertEqual(actual.layout_fingerprint, _batch().plan.layout_fingerprint)

    def test_validate_dataset_rejects_time_axis_shorter_than_stimulus(self):
        batch = _batch()
        spec = braincell.mech.Synapse("ExpSyn", name="E")
        channel_layout = dataset.ChannelLayout(
            (spec,),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 0),
        )
        plan = StimulusPlan(
            layout_fingerprint=channel_layout.fingerprint,
            n_traces=batch.plan.n_traces,
            events=batch.plan.events,
            channel_weights=batch.plan.channel_weights,
            duration_ms=batch.plan.duration_ms,
            seeds=batch.plan.seeds,
            protocol_labels=batch.plan.protocol_labels,
            split=batch.plan.split,
        )
        shortened = dataset.DatasetBatch(
            plan,
            batch.voltage_mv[:, :-1],
            batch.spike_times_ms,
            batch.time_ms[:-1],
            batch.metadata,
        )
        with self.assertRaisesRegex(ValueError, "time length"):
            dataset.validate_dataset(shortened, channel_layout, dt_ms=1.0)

    def test_teacher_runner_uses_existing_network_pipeline(self):
        excitatory = braincell.mech.Synapse("ExpSyn", name="E", tau=2.0 * u.ms, e=0.0 * u.mV)
        inhibitory = braincell.mech.Synapse("ExpSyn", name="I", tau=2.0 * u.ms, e=-80.0 * u.mV)

        def cell_factory(pop_size):
            soma = braincell.Branch.from_lengths(
                lengths=[20.0] * u.um,
                radii=[10.0, 10.0] * u.um,
                type="soma",
            )
            morphology = braincell.Morphology.from_root(soma, name="soma")

            def solver(cell):
                cell.V.value = cell.V.value - 0.01 * u.mV

            cell = braincell.Cell(
                morphology,
                cv_policy=braincell.CVPerBranch(),
                pop_size=pop_size,
                V_init=-65.0 * u.mV,
                V_th=-20.0 * u.mV,
                solver=solver,
            )
            cell.place(
                at("soma", 0.5),
                excitatory,
                inhibitory,
            )
            return cell

        specs = (excitatory, inhibitory)
        channel_layout = build_channel_layout(
            cell_factory((1,)),
            specs,
            reference_weights_us=(0.1, 0.1),
        )
        plan = generate_pair_protocol(
            channel_layout,
            event_times_ms=(0.1, 0.1),
            duration_ms=0.3,
        )
        teacher = dataset.TeacherSpec(cell_factory, at("soma", 0.5), "test-cell", 0.1 * u.ms, -20.0)
        actual = dataset.run_teacher_batch(teacher, channel_layout, plan)
        self.assertEqual(actual.voltage_mv.shape, (plan.n_traces, 4))
        self.assertEqual(len(actual.spike_times_ms), plan.n_traces)
        self.assertEqual(actual.layout_fingerprint, channel_layout.fingerprint)

        incompatible_layout = channel_layout.__class__(
            channel_layout.specs,
            channel_layout.synapse_ids,
            channel_layout.placement_ids,
            channel_layout.point_ids,
            channel_layout.spec_indices,
            (0.2, 0.1),
            channel_layout.training_ranges,
        )
        incompatible_plan = generate_pair_protocol(
            incompatible_layout,
            event_times_ms=(0.1, 0.1),
            duration_ms=0.3,
        )
        with self.assertRaisesRegex(ValueError, "different channel layout"):
            dataset.run_teacher_batch(teacher, channel_layout, incompatible_plan)

    def test_teacher_session_reuses_cell_and_restores_state(self):
        excitatory = braincell.mech.Synapse("ExpSyn", name="E", tau=2.0 * u.ms, e=0.0 * u.mV)
        inhibitory = braincell.mech.Synapse("ExpSyn", name="I", tau=2.0 * u.ms, e=-80.0 * u.mV)
        factory_calls = []

        def cell_factory(pop_size):
            factory_calls.append(pop_size)
            soma = braincell.Branch.from_lengths(
                lengths=[20.0] * u.um,
                radii=[10.0, 10.0] * u.um,
                type="soma",
            )
            morphology = braincell.Morphology.from_root(soma, name="soma")

            def solver(cell):
                _, synapse = next(cell.runtime.iter_synapse_layouts())
                conductance = u.math.reshape(
                    synapse.g.value.to_decimal(u.uS),
                    (cell.pop_size[0], -1),
                )
                cell.V.value = cell.V.value + u.math.sum(conductance, axis=1)[:, None] * u.mV

            cell = braincell.Cell(
                morphology,
                cv_policy=braincell.CVPerBranch(),
                pop_size=pop_size,
                V_init=-65.0 * u.mV,
                V_th=-20.0 * u.mV,
                solver=solver,
            )
            cell.place(at("soma", 0.5), excitatory, inhibitory)
            return cell

        channel_layout = build_channel_layout(
            cell_factory((1,)),
            (excitatory, inhibitory),
            reference_weights_us=(0.1, 0.1),
        )
        plan = generate_pair_protocol(
            channel_layout,
            event_times_ms=(0.1, 0.1),
            duration_ms=0.3,
        )
        teacher = dataset.TeacherSpec(cell_factory, at("soma", 0.5), "test-cell", 0.1 * u.ms, -20.0)
        session = dataset.TeacherSession(teacher, channel_layout, n_traces=plan.n_traces)

        first = session.run(plan)
        second = session.run(plan)
        network = dataset.run_teacher_batch(teacher, channel_layout, plan)

        self.assertEqual(factory_calls, [(1,), (plan.n_traces,), (plan.n_traces,)])
        np.testing.assert_allclose(first.voltage_mv, second.voltage_mv)
        np.testing.assert_allclose(first.voltage_mv, network.voltage_mv)
        self.assertEqual(first.metadata["input_alignment"], "target-step")
