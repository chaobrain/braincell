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

"""Tests for :mod:`braincell._compute.layouts`."""

import unittest
from dataclasses import dataclass as _dataclass

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import (
    CVPerBranch,
    Cell,
    CurrentClamp,
    FunctionClamp,
    SineClamp,
)
from braincell.filter import AllRegion, BranchSlice, RootLocation, at
from braincell.mech import CVContext, Channel, Ion
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology
from ._testing import _build_tree
from .layouts import (
    CLAMP_KINDS,
    ClampRoutingTable,
    build_clamp_routing_table,
)


@_dataclass
class _ClampStubLayout:
    target: str
    kind: str
    point_index: np.ndarray | None


def _clamp_areas(areas_cm2, *, n_point: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point_area_decimal, midpoint_ids)`` for ``len(areas_cm2)`` CVs.

    CV ``i`` owns point ``i``, mirroring ``cv_to_mid_node_id == arange(n_cv)``
    on a real node tree. Any remaining points are CV boundaries, which carry no
    membrane area of their own.
    """
    point_area = np.zeros((n_point,), dtype=float)
    point_area[: len(areas_cm2)] = np.asarray(areas_cm2, dtype=float)
    return point_area, np.arange(len(areas_cm2), dtype=np.int32)


class RuntimeLayoutTest(unittest.TestCase):
    """Layout construction, clamp evaluation, and probe layout allocation."""

    def test_density_mechanism_builds_dense_layout_with_global_shape(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2)),
        )

        self.assertEqual(cell.n_cv, 2)
        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.node_tree.nodes), 5)
        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.target, "density")
        self.assertEqual(layout.kind, "channel:leaky")
        self.assertEqual(layout.n_active, 2)
        self.assertEqual(layout.source_cv_ids, (0, 1))
        self.assertIsNone(layout.point_index)
        np.testing.assert_array_equal(layout.cv_mask, [True, True])
        self.assertEqual(rcell.expected_state_shape(layout.id, "g_max"), (1, 2))
        self.assertEqual(rcell.voltage_shape, (1, 2))
        self.assertEqual(rcell.get_state(layout.id, "g_max").shape, (1, 2))
        np.testing.assert_allclose(
            np.asarray(rcell.get_state(layout.id, "g_max").to_decimal(u.mS / u.cm**2))[0],
            [4.0, 4.0],
        )

    def test_region_limited_density_current_is_masked_outside_active_points(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="soma_leak", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="dend_leak", g_max=5.0 * (u.mS / u.cm**2), E=-67.0 * u.mV),
        )

        cell.init_state()
        rcell = cell
        soma_current = rcell.get_runtime_node(0).current(rcell.V.value)
        dend_current = rcell.get_runtime_node(1).current(rcell.V.value)
        np.testing.assert_allclose(
            np.asarray(soma_current.to_decimal(u.nA / u.cm**2))[0],
            [-12000.0, 0.0],
        )
        np.testing.assert_allclose(
            np.asarray(dend_current.to_decimal(u.nA / u.cm**2))[0],
            [0.0, -10000.0],
        )

    def test_partial_channel_coverage_keeps_fraction_metadata_and_full_cv_current(self) -> None:
        cell = Cell(_build_tree(), V_init=-65.0 * u.mV)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            braincell.mech.Channel("IL", name="half_leak", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()

        layout = cell.layouts[0]
        declaration = cell.runtime.get_layout_mechanism(layout.id)
        node = cell.runtime.get_runtime_node(layout.id)
        current = node.current(cell.V.value)

        self.assertAlmostEqual(declaration.coverage_area_fraction, 0.5, places=12)
        np.testing.assert_array_equal(layout.cv_mask, [True, False])
        np.testing.assert_allclose(
            np.asarray(current.to_decimal(u.nA / u.cm**2))[0],
            [-12000.0, 0.0],
        )

    def test_point_mechanism_builds_sparse_layout_with_local_shape(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.kind, "CurrentClamp")
        self.assertEqual(layout.n_active, 1)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertIsNone(layout.point_mask)
        self.assertEqual(rcell.expected_state_shape(layout.id, "amplitudes"), (1, 1, 1))
        self.assertEqual(len(rcell.get_state(layout.id, "amplitudes")), 1)
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in rcell.get_state(layout.id, "amplitudes")[0]), (0.1,))
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in rcell.get_state(layout.id, "durations")[0]), (2.0,))
        self.assertEqual(rcell.get_state(layout.id, "delay")[0], 1.0 * u.ms)
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(1)), (layout.id,))
        self.assertEqual(rcell.get_point_layouts(1), (layout,))

    def test_point_mechanism_can_land_on_root_endpoint(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.0),
            CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
        )

        cell.init_state()
        rcell = cell

        root_node_id = rcell.node_tree.root_node_id
        midpoint_id = int(rcell.node_tree.cv_to_mid_node_id[0])
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.point_index.tolist(), [root_node_id])
        self.assertEqual(tuple(item.id for item in rcell.get_point_layouts(root_node_id)), (layout.id,))
        self.assertEqual(tuple(item.id for item in rcell.get_point_layouts(midpoint_id)), ())
        self.assertEqual(tuple(item.id for item in rcell.get_cv_layouts(0)), (layout.id,))

    def test_channel_spec_builds_dense_layout_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.kind, "channel:IL")
        self.assertEqual(rcell.expected_state_shape(layout.id, "g_max"), (1, 2))
        self.assertEqual(rcell.expected_state_shape(layout.id, "E"), (1, 2))
        self.assertEqual(rcell.get_cv_state(0)[layout.id]["g_max"][0], 4.0 * (u.mS / u.cm**2))
        self.assertEqual(rcell.get_cv_state(1)[layout.id]["E"][0], -68.0 * u.mV)
        node = rcell.get_runtime_node(layout.id)
        self.assertIsInstance(node, braincell.channel.IL)
        self.assertEqual(node.varshape, (1, 2))
        self.assertAlmostEqual(float(node.g_max[0, 0].to_decimal(u.mS / u.cm**2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 1].to_decimal(u.mS / u.cm**2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.E[0, 0].to_decimal(u.mV)), -68.0, places=12)

    def test_named_channel_spec_merges_across_regions_when_identity_matches(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        self.assertEqual(rcell.layouts[0].source_cv_ids, (0, 1))

    def test_same_class_different_names_build_distinct_layouts(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
            braincell.mech.Channel("IL", name="leak_b", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 2)
        self.assertEqual({layout.kind for layout in rcell.layouts}, {"channel:IL"})
        self.assertTrue(all(layout.source_cv_ids == (0, 1) for layout in rcell.layouts))

    def test_runtime_state_keeps_dense_and_sparse_layouts_together(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2)),
        )
        clamp = CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA)
        cell.place(RootLocation(x=0.5), clamp)

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 2)
        dense = next(layout for layout in rcell.layouts if layout.layout == "dense")
        sparse = next(layout for layout in rcell.layouts if layout.layout == "sparse")
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(1)), (sparse.id,))
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(3)), ())
        self.assertEqual(tuple(layout.id for layout in rcell.get_cv_layouts(0)), (dense.id, sparse.id))
        self.assertEqual(tuple(layout.id for layout in rcell.get_cv_layouts(1)), (dense.id,))
        point_state = rcell.get_point_state(1)
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in point_state[sparse.id]["amplitudes"][0]), (0.1,))
        self.assertEqual(rcell.get_cv_state(0)[dense.id]["g_max"][0], 4.0 * (u.mS / u.cm**2))
        self.assertEqual({name for name in ("na", "k", "ca")}, {"na", "k", "ca"})

    def test_runtime_evaluates_step_sine_and_function_clamps_on_target_points(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=0.0 * u.ms, durations=(2.0 * u.ms, 2.0 * u.ms), amplitudes=(0.0 * u.nA, 0.3 * u.nA)),
            SineClamp(amplitude=0.2 * u.nA, frequency=500.0 * u.Hz, offset=0.1 * u.nA, duration=4.0 * u.ms),
            FunctionClamp(fn=lambda t: 0.4 * u.nA if t < 1.0 * u.ms else 0.0 * u.nA),
        )
        cell.init_state()
        rcell = cell

        runtime = rcell.runtime

        current_early = runtime.evaluate_point_clamps(t=0.5 * u.ms)
        current_late = runtime.evaluate_point_clamps(t=2.5 * u.ms)

        self.assertEqual(current_early.shape, (1, len(rcell.node_tree.nodes)))
        self.assertAlmostEqual(float(current_early[0, 1].to_decimal(u.nA)), 0.7, places=6)
        self.assertAlmostEqual(float(current_early[0, 0].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[0, 1].to_decimal(u.nA)), 0.6, places=6)

    def test_current_clamp_decimal_boundaries_have_no_gap_or_terminal_step(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(
                durations=tuple(0.1 * u.ms for _ in range(10)),
                amplitudes=tuple(float(index + 1) * u.nA for index in range(10)),
            ),
        )
        cell.init_state()

        at_internal_boundary = cell.runtime.evaluate_point_clamps(t=0.2 * u.ms).to_decimal(u.nA)
        at_total_duration = cell.runtime.evaluate_point_clamps(t=1.0 * u.ms).to_decimal(u.nA)

        self.assertAlmostEqual(float(at_internal_boundary[0, 1]), 3.0, places=6)
        self.assertAlmostEqual(float(at_total_duration[0, 1]), 0.0, places=6)

    def test_function_clamp_receives_absolute_time(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            FunctionClamp(fn=lambda t: 0.4 * u.nA if 2.0 * u.ms <= t < 3.0 * u.ms else 0.0 * u.nA),
        )
        cell.init_state()
        runtime = cell.runtime

        before = runtime.evaluate_point_clamps(t=1.0 * u.ms)
        active = runtime.evaluate_point_clamps(t=2.5 * u.ms)
        after = runtime.evaluate_point_clamps(t=4.0 * u.ms)

        self.assertAlmostEqual(float(before[0, 1].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(active[0, 1].to_decimal(u.nA)), 0.4, places=6)
        self.assertAlmostEqual(float(after[0, 1].to_decimal(u.nA)), 0.0, places=6)

    def test_sine_clamp_uses_delay_window(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            SineClamp(
                amplitude=0.0 * u.nA,
                frequency=500.0 * u.Hz,
                offset=0.2 * u.nA,
                delay=1.0 * u.ms,
                duration=2.0 * u.ms,
            ),
        )
        cell.init_state()
        runtime = cell.runtime

        before = runtime.evaluate_point_clamps(t=0.5 * u.ms)
        active = runtime.evaluate_point_clamps(t=1.5 * u.ms)
        after = runtime.evaluate_point_clamps(t=3.5 * u.ms)

        self.assertAlmostEqual(float(before[0, 1].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(active[0, 1].to_decimal(u.nA)), 0.2, places=6)
        self.assertAlmostEqual(float(after[0, 1].to_decimal(u.nA)), 0.0, places=6)

    def test_sine_clamp_reads_the_delay_of_each_point_not_of_point_zero(self) -> None:
        # Two identical SineClamps merge into one layout. The evaluator used
        # to fetch ``delay`` without ``local_index``, so every point in the
        # layout silently used point 0's delay -- an omitted argument that a
        # single-point test cannot see, because there index 0 is the only
        # index. CurrentClamp, eleven lines away, always passed it.
        cell = Cell(_build_tree())
        for branch_index in (0, 1):
            cell.place(
                at(branch_index, 0.5),
                SineClamp(
                    amplitude=0.0 * u.nA,
                    frequency=500.0 * u.Hz,
                    offset=0.2 * u.nA,
                    delay=0.0 * u.ms,
                    duration=100.0 * u.ms,
                ),
            )
        cell.init_state()
        runtime = cell.runtime

        layout = next(lay for lay in runtime.layouts if lay.kind == "SineClamp")
        self.assertEqual(layout.n_active, 2)
        # The two clamps must land on distinct points, or the scatter-add
        # below would hide a per-point difference behind a single total.
        self.assertEqual(len(set(layout.point_index.tolist())), 2)

        # Give the two points different delays: point 0 open, point 1 still
        # waiting at t = 1 ms.
        buffer = runtime.state_buffers[(int(layout.id), "delay")]
        delays = np.zeros_like(np.asarray(buffer.mantissa))
        delays[..., 1] = 5.0
        runtime.state_buffers[(int(layout.id), "delay")] = u.Quantity(delays, buffer.unit)

        current = runtime.evaluate_point_clamps(t=1.0 * u.ms).to_decimal(u.nA)
        per_point = [float(current[..., int(pid)].ravel()[0]) for pid in layout.point_index]

        self.assertAlmostEqual(per_point[0], 0.2, places=6)
        self.assertAlmostEqual(per_point[1], 0.0, places=6)

    def test_probe_layouts_are_sparse_and_allocate_no_state_buffers(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="p"),
            braincell.mech.CurrentProbe(ion="na", mechanism="Na_HH1952"),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 3)
        self.assertTrue(all(layout.layout == "sparse" for layout in rcell.layouts))
        self.assertTrue(all(layout.target == "point" for layout in rcell.layouts))
        resolved_names = []
        for layout in rcell.layouts:
            self.assertEqual(layout.point_index.tolist(), [1])
            self.assertEqual(rcell.get_point_state(1)[layout.id], {})
            declaration = rcell.runtime.get_layout_mechanism(layout.id)
            resolved_names.append(declaration.name)
        self.assertEqual(
            sorted(resolved_names),
            ["soma(0.5)_Na_HH1952_current", "soma(0.5)_Na_HH1952_p", "soma(0.5)_v"],
        )


class EvaluatePointClampsJitTest(unittest.TestCase):
    """Task 19: evaluate_point_clamps compiles under JAX without object dtype."""

    def test_evaluate_point_clamps_jit_compiles(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            CurrentClamp(delay=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.init_state()
        runtime = cell.runtime
        compiled = jax.jit(lambda t: runtime.evaluate_point_clamps(t=t))
        out = compiled(0.5 * u.ms)
        self.assertEqual(out.mantissa.shape, runtime.pop_size + (runtime.n_point,))

    def test_evaluate_point_clamps_supports_population_specific_amplitudes(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(durations=2.0 * u.ms, amplitudes=u.Quantity(jnp.asarray([0.1, 0.2]), u.nA)),
        )
        cell.init_state()
        runtime = cell.runtime
        out = runtime.evaluate_point_clamps(t=0.5 * u.ms)
        point_id = int(runtime.layouts[0].point_index[0])
        self.assertEqual(out.mantissa.shape, (2, runtime.n_point))
        self.assertAlmostEqual(float(out[0, point_id].to_decimal(u.nA)), 0.1, places=6)
        self.assertAlmostEqual(float(out[1, point_id].to_decimal(u.nA)), 0.2, places=6)

    def test_evaluate_point_clamps_can_filter_midpoint_and_boundary_points(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.0),
            CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA),
        )
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(durations=1.0 * u.ms, amplitudes=0.2 * u.nA),
        )
        cell.init_state()

        runtime = cell.runtime
        root_id = int(cell.node_tree.root_node_id)
        midpoint_id = int(cell.node_tree.cv_to_mid_node_id[0])

        all_current = runtime.evaluate_point_clamps(t=0.5 * u.ms).to_decimal(u.nA)
        midpoint_current = runtime.evaluate_point_clamps(
            t=0.5 * u.ms,
            point_ids=np.asarray([midpoint_id], dtype=np.int32),
        ).to_decimal(u.nA)
        boundary_current = runtime.evaluate_point_clamps(
            t=0.5 * u.ms,
            point_ids=np.asarray([root_id], dtype=np.int32),
        ).to_decimal(u.nA)

        self.assertAlmostEqual(float(all_current[0, root_id]), 0.1, places=6)
        self.assertAlmostEqual(float(all_current[0, midpoint_id]), 0.2, places=6)
        self.assertAlmostEqual(float(midpoint_current[0, root_id]), 0.0, places=6)
        self.assertAlmostEqual(float(midpoint_current[0, midpoint_id]), 0.2, places=6)
        self.assertAlmostEqual(float(boundary_current[0, root_id]), 0.1, places=6)
        self.assertAlmostEqual(float(boundary_current[0, midpoint_id]), 0.0, places=6)


class DensityLayoutMaskingUnderJit(unittest.TestCase):
    """Task 19 (C5-adjacent): density mantissa is JAX-friendly, no object dtype."""

    def test_state_buffer_mantissa_sums_under_jit(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        mantissa = cell.runtime.state_buffers[(layout.id, "g_max")].mantissa
        total = float(jax.jit(lambda x: jnp.asarray(x).sum())(mantissa))
        self.assertGreater(total, 0.0)


class TestBuildClampRoutingTable(unittest.TestCase):
    def test_no_clamp_layouts_returns_none(self):
        layouts = (
            _ClampStubLayout(
                target="density",
                kind="IL",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6], n_point=1)
        table = build_clamp_routing_table(
            layouts=layouts,
            point_area_decimal=area,
            midpoint_ids=midpoints,
        )
        self.assertIsNone(table)

    def test_midpoint_current_clamp_builds_midpoint_route(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([1], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6, 2e-6], n_point=2)
        table = build_clamp_routing_table(
            layouts=layouts,
            point_area_decimal=area,
            midpoint_ids=midpoints,
        )
        self.assertIsInstance(table, ClampRoutingTable)
        np.testing.assert_array_equal(table.midpoint_ids, np.asarray([1], dtype=np.int32))
        np.testing.assert_allclose(table.midpoint_area, np.asarray([2e-6]))
        np.testing.assert_array_equal(table.boundary_ids, np.asarray([], dtype=np.int32))

    def test_each_clamp_kind_is_recognized(self):
        self.assertEqual(CLAMP_KINDS, frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"}))

    def test_ids_are_sorted_and_unique(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([3, 1], dtype=np.int32),
            ),
            _ClampStubLayout(
                target="point",
                kind="SineClamp",
                point_index=np.asarray([1, 2], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6 * (i + 1) for i in range(4)], n_point=4)
        table = build_clamp_routing_table(
            layouts=layouts,
            point_area_decimal=area,
            midpoint_ids=midpoints,
        )
        np.testing.assert_array_equal(table.midpoint_ids, np.asarray([1, 2, 3], dtype=np.int32))

    def test_zero_area_raises(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([0.0], n_point=1)
        with self.assertRaises(ValueError):
            build_clamp_routing_table(
                layouts=layouts,
                point_area_decimal=area,
                midpoint_ids=midpoints,
            )

    def test_endpoint_clamp_builds_boundary_route(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([2], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6], n_point=3)
        table = build_clamp_routing_table(
            layouts=layouts,
            point_area_decimal=area,
            midpoint_ids=midpoints,
        )
        self.assertIsInstance(table, ClampRoutingTable)
        np.testing.assert_array_equal(table.midpoint_ids, np.asarray([], dtype=np.int32))
        np.testing.assert_array_equal(table.midpoint_area, np.asarray([], dtype=float))
        np.testing.assert_array_equal(table.boundary_ids, np.asarray([2], dtype=np.int32))

    def test_mixed_midpoint_and_endpoint_clamps_route_separately(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([0, 3], dtype=np.int32),
            ),
            _ClampStubLayout(
                target="point",
                kind="SineClamp",
                point_index=np.asarray([1, 4], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6, 2e-6], n_point=5)
        table = build_clamp_routing_table(
            layouts=layouts,
            point_area_decimal=area,
            midpoint_ids=midpoints,
        )
        np.testing.assert_array_equal(table.midpoint_ids, np.asarray([0, 1], dtype=np.int32))
        np.testing.assert_allclose(table.midpoint_area, np.asarray([1e-6, 2e-6]))
        np.testing.assert_array_equal(table.boundary_ids, np.asarray([3, 4], dtype=np.int32))

    def test_non_clamp_point_layout_ignored(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="Synapse",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        area, midpoints = _clamp_areas([1e-6], n_point=1)
        self.assertIsNone(
            build_clamp_routing_table(
                layouts=layouts,
                point_area_decimal=area,
                midpoint_ids=midpoints,
            )
        )


class StateBufferStorageTest(unittest.TestCase):
    """Task 12: rectangular params live as Quantity(jnp.ndarray, unit)."""

    def test_density_buffer_is_quantity_backed_by_jax(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2)),
        )
        cell.init_state()
        layout = cell.layouts[0]
        buffer = cell.runtime.state_buffers[(layout.id, "g_max")]
        self.assertTrue(hasattr(buffer, "unit"))
        self.assertTrue(hasattr(buffer, "mantissa"))
        self.assertTrue(isinstance(buffer.mantissa, (np.ndarray, jnp.ndarray)))

    def test_set_state_broadcast_scalar_and_readback(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        cell.runtime.set_state(layout.id, "g_max", 7.5 * (u.mS / u.cm**2))
        new = cell.runtime.get_state(layout.id, "g_max")
        self.assertAlmostEqual(float(new[0, 1].to_decimal(u.mS / u.cm**2)), 7.5, places=12)

    def test_set_state_shape_mismatch_raises(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        with self.assertRaises(ValueError):
            cell.runtime.set_state(
                layout.id,
                "g_max",
                u.Quantity(jnp.ones((99,)), u.mS / u.cm**2),
            )

    def test_no_object_dtype_in_density_buffer(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        buffer = cell.runtime.state_buffers[(layout.id, "g_max")]
        self.assertNotEqual(buffer.mantissa.dtype, np.dtype("O"))


class RaggedCurrentClampBufferTest(unittest.TestCase):
    """Task 13: CurrentClamp durations/amplitudes packed into padded + mask."""

    def test_identical_lambda_bodies_produce_one_layout(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            FunctionClamp(fn=lambda t: 0.1 * u.nA),
        )
        cell.place(
            at("soma", 0.75),
            FunctionClamp(fn=lambda t: 0.1 * u.nA),
        )
        cell.init_state()
        fn_clamp_layouts = [layout for layout in cell.layouts if layout.kind == "FunctionClamp"]
        self.assertEqual(len(fn_clamp_layouts), 1)

    def test_three_clamps_with_varying_step_counts_pad_and_mask(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.25),
            CurrentClamp(delay=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(1.0 * u.ms, 1.0 * u.ms),
                amplitudes=(0.1 * u.nA, 0.2 * u.nA),
            ),
        )
        cell.place(
            at("soma", 0.75),
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(0.5 * u.ms, 0.5 * u.ms, 1.0 * u.ms),
                amplitudes=(0.1 * u.nA, 0.2 * u.nA, 0.3 * u.nA),
            ),
        )
        cell.init_state()

        current_clamp_layouts = [layout for layout in cell.layouts if layout.kind == "CurrentClamp"]
        self.assertGreaterEqual(len(current_clamp_layouts), 1)
        for layout in current_clamp_layouts:
            dur = cell.runtime.state_buffers[(layout.id, "durations")]
            amp = cell.runtime.state_buffers[(layout.id, "amplitudes")]
            self.assertTrue(hasattr(dur, "unit"))
            self.assertEqual(dur.mantissa.ndim, 3)
            self.assertEqual(amp.mantissa.shape, dur.mantissa.shape)
            mask_key = (layout.id, "_mask_durations")
            self.assertIn(mask_key, cell.runtime.state_buffers)
            self.assertEqual(cell.runtime.state_buffers[mask_key].shape, dur.mantissa.shape)

    def test_population_multistep_current_clamp_preserves_step_axis(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(1.0 * u.ms, 1.0 * u.ms),
                amplitudes=(
                    u.Quantity(np.asarray([0.1, 0.0]), u.nA),
                    u.Quantity(np.asarray([0.0, 0.2]), u.nA),
                ),
            ),
        )
        cell.init_state()

        layout = next(layout for layout in cell.layouts if layout.kind == "CurrentClamp")
        durations = cell.runtime.state_buffers[(layout.id, "durations")]
        amplitudes = cell.runtime.state_buffers[(layout.id, "amplitudes")]
        self.assertEqual(durations.mantissa.shape, (2, 1, 2))
        self.assertEqual(amplitudes.mantissa.shape, (2, 1, 2))

        current0 = cell.runtime.evaluate_point_clamps(t=0.5 * u.ms).to_decimal(u.nA)
        current1 = cell.runtime.evaluate_point_clamps(t=1.5 * u.ms).to_decimal(u.nA)
        self.assertAlmostEqual(float(current0[0, 1]), 0.1, places=6)
        self.assertAlmostEqual(float(current0[1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current1[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current1[1, 1]), 0.2, places=6)


class FnFingerprintWarnsOnOpaqueClosureTest(unittest.TestCase):
    """MED-08: fingerprinting a lambda with opaque closure emits RuntimeWarning."""

    def test_opaque_closure_emits_warning(self) -> None:
        import warnings

        from braincell._compute.layouts import _fn_fingerprint, _opaque_warned

        class _Opaque:
            __slots__ = ("x",)

            def __init__(self) -> None:
                self.x = object()

        opaque = _Opaque()

        def _make():
            return lambda t: opaque

        fn = _make()

        # Drop any prior entry for this call-site so the test is independent
        # of test ordering.
        _opaque_warned.discard((fn.__code__.co_filename, fn.__code__.co_firstlineno))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _fn_fingerprint(fn)
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in captured),
            "expected RuntimeWarning for opaque closure cell",
        )


# =============================================================================
# Spatially-varying (callable) mechanism parameters
# =============================================================================


def _spatial_param_cell(*, pop_size=1) -> Cell:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(dend, name="dend", parent_x=0.5)
    return Cell(
        tree,
        pop_size=pop_size,
        cv_policy=CVPerBranch(cv_per_branch=2),
    )


class SpatialDensityParameterTest(unittest.TestCase):
    def test_channel_callable_resolves_once_per_cv_and_broadcasts_population(self) -> None:
        seen: list[CVContext] = []

        def g_max(context: CVContext):
            seen.append(context)
            distance = context.path_distance_from_soma.to_decimal(u.um)
            return (0.02 + 0.00008 * distance) * (u.mS / u.cm**2)

        cell = _spatial_param_cell(pop_size=(2,))
        cell.paint(
            AllRegion(),
            Channel(
                "IL",
                name="distance_leak",
                g_max=g_max,
                E=-70.0 * u.mV,
            ),
        )
        cell.init_state()

        layouts = [layout for layout in cell.layouts if layout.kind == "channel:IL"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual(len(seen), cell.n_cv)
        self.assertTrue(all(isinstance(context, CVContext) for context in seen))

        state = cell.get_state(layouts[0].id, "g_max")
        actual = np.asarray(state.to_decimal(u.mS / u.cm**2))
        expected = np.asarray([0.02, 0.02, 0.022, 0.026])
        np.testing.assert_allclose(actual, np.broadcast_to(expected, (2, 4)))
        self.assertEqual(cell.expected_state_shape(layouts[0].id, "g_max"), (2, 4))

        _ = cell.get_state(layouts[0].id, "g_max")
        self.assertEqual(len(seen), cell.n_cv)

    def test_ion_callable_resolves_with_units(self) -> None:
        def reversal(context: CVContext):
            distance = context.path_distance_from_soma.to_decimal(u.um)
            return (50.0 - 0.1 * distance) * u.mV

        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Ion("SodiumFixed", name="na_distance", E=reversal),
        )
        cell.init_state()

        layout = next(layout for layout in cell.layouts if layout.kind == "ion:SodiumFixed")
        state = cell.get_state(layout.id, "E")
        actual = state.to_decimal(u.mV)
        np.testing.assert_allclose(actual, [[50.0, 50.0, 47.5, 42.5]])

    def test_ion_callable_accepts_unitless_scalar(self) -> None:
        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Ion("SodiumFixed", name="na", valence=lambda context: 1.0),
        )
        cell.init_state()

        layout = next(layout for layout in cell.layouts if layout.kind == "ion:SodiumFixed")
        state = cell.get_state(layout.id, "valence")
        np.testing.assert_allclose(
            state,
            np.ones(cell.pop_size + (cell.n_cv,)),
        )

    def test_callable_rejects_non_scalar_result_with_cv_details(self) -> None:
        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Channel(
                "IL",
                g_max=lambda context: [0.1, 0.2] * (u.mS / u.cm**2),
                E=-70.0 * u.mV,
            ),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*CV 0.*must return a scalar",
        ):
            cell.init_state()

    def test_callable_rejects_mixed_unitful_and_unitless_results(self) -> None:
        def mixed(context: CVContext):
            if context.branch_type == "soma":
                return 0.02 * (u.mS / u.cm**2)
            return 0.03

        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=mixed, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*expected a Quantity consistently",
        ):
            cell.init_state()

    def test_callable_rejects_incompatible_quantity_units(self) -> None:
        def incompatible(context: CVContext):
            if context.branch_type == "soma":
                return 0.02 * (u.mS / u.cm**2)
            return -70.0 * u.mV

        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=incompatible, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*compatible with",
        ):
            cell.init_state()

    def test_callable_wraps_user_error_with_cv_details(self) -> None:
        def broken(context: CVContext):
            raise RuntimeError("bad spatial rule")

        cell = _spatial_param_cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=broken, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            ValueError,
            r"parameter 'g_max'.*CV 0.*bad spatial rule",
        ):
            cell.init_state()
