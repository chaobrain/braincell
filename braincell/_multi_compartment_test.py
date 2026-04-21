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


import math
import unittest
from unittest import mock

import brainstate
import brainunit as u
import numpy as np

brainstate.environ.set(precision=64)

import braincell
from braincell import (
    Branch,
    CVPerBranch,
    CableProperty,
    Cell,
    CurrentClamp,
    Morphology,
    RunResult,
)
from braincell.filter import (
    BranchSlice,
    RootLocation,
    Terminals,
    at,
)


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def _build_three_branch_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[120.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=0.0)
    return tree


def _point_id_by_role(point_tree, *, cv_id: int, position: str) -> int:
    matches = [
        point.id
        for point in point_tree.points
        if any(role.cv_id == cv_id and role.position == position for role in point.cv_points)
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one point with role {(cv_id, position)!r}, got {matches!r}.")
    return int(matches[0])


def _row_by_role(scheduling, point_tree, *, cv_id: int, position: str) -> int:
    point_id = _point_id_by_role(point_tree, cv_id=cv_id, position=position)
    return int(scheduling.point_id_to_row[point_id])


def _point_roles(point_tree, point_id: int) -> tuple[tuple[int, str], ...]:
    point = point_tree.points[point_id]
    return tuple((role.cv_id, role.position) for role in point.cv_points)


def _edge_roles(point_tree, *, parent_point_id: int, child_point_id: int) -> tuple[tuple[int, str], ...]:
    matches = [
        edge
        for edge in point_tree.edges
        if edge.parent_point_id == parent_point_id and edge.child_point_id == child_point_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one edge {(parent_point_id, child_point_id)!r}, got {[edge.id for edge in matches]!r}."
        )
    return tuple((role.cv_id, role.half) for role in matches[0].cv_edges)


def _build_soma_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    return Morphology.from_root(soma, name="soma")


class CellFacadeTest(unittest.TestCase):
    def test_default_cell_has_cv_and_default_paint_rules(self) -> None:
        cell = Cell(_build_tree())
        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(len(cell.paint_rules), 1)
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -65.0, places=12)
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm ** 2), 1.0, places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), 100.0, places=12)
        self.assertAlmostEqual(
            cv0.temp.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_repr_and_str_expose_compact_cell_summary(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms),
        )

        self.assertEqual(
            repr(cell),
            "Cell(root='soma', n_branches=2, n_cv=2, n_paint_rules=1, n_place_rules=1)",
        )
        self.assertEqual(
            str(cell),
            "-----------------------------------\n"
            "root           | soma\n"
            "n_branches     | 2\n"
            "n_cv           | 2\n"
            "n_paint_rules  | 1\n"
            "n_place_rules  | 1\n"
            "-----------------------------------\n",
        )

    def test_point_tree_counts_points_and_orders_root_first(self) -> None:
        cell = Cell(_build_tree())
        tree = cell.point_tree()

        self.assertEqual(len(tree.points), cell.n_cv + len(cell.morpho.branches) + 1)
        self.assertEqual(len(tree.edges), len(tree.points) - 1)
        self.assertEqual(_point_roles(tree, int(tree.matrix_index_to_point_id[0])), ((0, "proximal"),))
        self.assertEqual(
            [_point_roles(tree, int(point_id)) for point_id in tree.matrix_index_to_point_id.tolist()],
            [
                ((0, "proximal"),),
                ((0, "mid"),),
                ((0, "distal"), (1, "proximal")),
                ((1, "mid"),),
                ((1, "distal"),),
            ],
        )
        self.assertEqual(tree.point_parent.tolist(), [-1, 0, 1, 2, 3])

    def test_point_tree_repr_and_str_expose_compact_summary(self) -> None:
        point_tree = Cell(_build_tree()).point_tree()

        self.assertEqual(
            repr(point_tree),
            "PointTree(n_points=5, n_edges=4, root_point_id=0)",
        )
        self.assertEqual(
            str(point_tree),
            "-----------------------------------\n"
            "n_points       | 5\n"
            "n_edges        | 4\n"
            "root_point_id  | 0\n"
            "-----------------------------------\n",
        )

    def test_cell_freezes_morphology_snapshot(self) -> None:
        tree = _build_tree()
        cell = Cell(tree)
        self.assertEqual(cell.n_cv, 2)
        tree.soma.axon = Branch.from_lengths(lengths=[80.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon")
        self.assertEqual(cell.n_cv, 2)

    def test_cable_paint_hits_midpoint_only(self) -> None:
        cell = Cell(_build_tree())
        base = cell.cvs[0]
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.49),
            CableProperty(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm ** 2), base.cm.to_decimal(u.uF / u.cm ** 2), places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), base.ra.to_decimal(u.ohm * u.cm), places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), base.v.to_decimal(u.mV), places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), base.temp.to_decimal(u.kelvin), places=12)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.49, dist=0.51),
            CableProperty(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm ** 2), 2.0, places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), 200.0, places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -70.0, places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), u.celsius2kelvin(20.0).to_decimal(u.kelvin), places=12)

    def test_cable_paint_compacts_same_region_history(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        cell.paint(
            region,
            CableProperty(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        cell.paint(
            region,
            CableProperty(
                resting_potential=-60.0 * u.mV,
                membrane_capacitance=3.0 * (u.uF / u.cm ** 2),
                axial_resistivity=300.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(30.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        last = next(rule for rule in cell.paint_rules if rule.region == region)
        cable = last.mechanism
        self.assertIsInstance(cable, CableProperty)
        self.assertAlmostEqual(cable.resting_potential.to_decimal(u.mV), -60.0, places=12)
        self.assertAlmostEqual(cable.membrane_capacitance.to_decimal(u.uF / u.cm ** 2), 3.0, places=12)
        self.assertAlmostEqual(cable.axial_resistivity.to_decimal(u.ohm * u.cm), 300.0, places=12)
        self.assertAlmostEqual(
            cable.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(30.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_density_paint_channel_scales_by_area_fraction(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
            braincell.mech.Ion("SodiumFixed", Ci=12.0 * u.mM),
        )
        cv0 = cell.cvs[0]
        self.assertEqual(len(cv0.density_mech), 2)
        channel = next(mech for mech in cv0.density_mech if mech.category == "channel")
        ion = next(mech for mech in cv0.density_mech if mech.category == "ion")
        gmax = channel.params["g_max"]
        self.assertAlmostEqual(float(gmax.to_decimal(u.mS / u.cm ** 2)), 2.0, places=12)
        self.assertEqual(ion.params["Ci"], 12.0 * u.mM)

    def test_channel_spec_paint_scales_by_area_fraction(self) -> None:
        tree = Morphology.from_root(
            Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"),
            name="soma",
        )
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-72.0 * u.mV),
        )

        cv0 = cell.cvs[0]
        self.assertEqual(len(cv0.density_mech), 1)
        channel = cv0.density_mech[0]
        self.assertEqual(channel.category, "channel")
        self.assertEqual(channel.class_name, "IL")
        self.assertEqual(channel.instance_name, "IL")
        self.assertAlmostEqual(
            float(channel.params["g_max"].to_decimal(u.mS / u.cm ** 2)), 2.0, places=12
        )
        self.assertAlmostEqual(float(channel.params["E"].to_decimal(u.mV)), -72.0, places=12)

    def test_place_boundary_goes_to_right_cv_and_branch_endpoint_stays_local(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))

        stim_boundary = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms)
        stim_parent_end = CurrentClamp.step(0.2 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms)
        cell.place(RootLocation(x=0.5), stim_boundary)
        cell.place(RootLocation(x=1.0), stim_parent_end)

        # RootLocation(x=0.5) on branch 0 should map to right CV (id 1).
        self.assertEqual(len(cell.cvs[0].point_mech), 0)
        self.assertEqual(len(cell.cvs[1].point_mech), 2)
        # parent branch endpoint remains on branch 0 last CV, not child branch first CV.
        self.assertEqual(cell.cvs[1].point_mech[-1].amplitudes[0].to_decimal(u.nA), 0.2)
        self.assertEqual(len(cell.cvs[2].point_mech), 0)

    def test_place_at_location_follows_existing_boundary_mapping_rules(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))

        stim_boundary = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms)
        stim_parent_end = CurrentClamp.step(0.2 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms)
        cell.place(at("soma", 0.5), stim_boundary)
        cell.place(at(0, 1.0), stim_parent_end)

        # Single-point locsets reuse the existing point-to-CV mapping:
        # boundaries go to the right CV, while a branch endpoint stays on the
        # parent branch's terminal CV even when a child attaches there.
        self.assertEqual(len(cell.cvs[0].point_mech), 0)
        self.assertEqual(len(cell.cvs[1].point_mech), 2)
        self.assertEqual(cell.cvs[1].point_mech[-1].amplitudes[0].to_decimal(u.nA), 0.2)
        self.assertEqual(len(cell.cvs[2].point_mech), 0)

    def test_cv_geometry_and_split_axial_resistance(self) -> None:
        soma = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[1.0, 2.0, 3.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=1))
        cv0 = cell.cvs[0]
        self.assertFalse(hasattr(cv0, "mean_radius"))

        area1 = math.pi * (1.0 + 2.0) * math.sqrt(10.0 ** 2 + (2.0 - 1.0) ** 2)
        area2 = math.pi * (2.0 + 3.0) * math.sqrt(20.0 ** 2 + (3.0 - 2.0) ** 2)
        self.assertAlmostEqual(cv0.area.to_decimal(u.um ** 2), area1 + area2, places=9)

        expected_total = 100.0 * (
            (10.0e-4) / (math.pi * (1.0e-4) * (2.0e-4))
            + (20.0e-4) / (math.pi * (2.0e-4) * (3.0e-4))
        )
        expected_prox = 100.0 * (
            (10.0e-4) / (math.pi * (1.0e-4) * (2.0e-4))
            + (5.0e-4) / (math.pi * (2.0e-4) * (2.25e-4))
        )
        expected_dist = 100.0 * ((15.0e-4) / (math.pi * (2.25e-4) * (3.0e-4)))
        self.assertAlmostEqual(cv0.r_axial.to_decimal(u.ohm), expected_total, places=9)
        self.assertAlmostEqual(cv0.r_axial_prox.to_decimal(u.ohm), expected_prox, places=9)
        self.assertAlmostEqual(cv0.r_axial_dist.to_decimal(u.ohm), expected_dist, places=9)
        self.assertAlmostEqual(
            cv0.r_axial.to_decimal(u.ohm),
            cv0.r_axial_prox.to_decimal(u.ohm) + cv0.r_axial_dist.to_decimal(u.ohm),
            places=9,
        )
        self.assertAlmostEqual(cv0.radius_prox.to_decimal(u.um), 1.0, places=9)
        self.assertAlmostEqual(cv0.radius_dist.to_decimal(u.um), 3.0, places=9)
        self.assertAlmostEqual(cv0.diam_mid.to_decimal(u.um), 4.5, places=9)

    def test_zero_length_radius_jump_contributes_area_but_not_axial_resistance(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[1.0, 0.5] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        whole = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=1)).cvs[0]
        split = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))
        cv_left = split.cvs[0]
        cv_right = split.cvs[1]

        jump_area = math.pi * (3.0 ** 2 - 1.0 ** 2)
        seg1 = math.pi * (2.0 + 1.0) * math.sqrt(10.0 ** 2 + (1.0 - 2.0) ** 2)
        seg2 = math.pi * (3.0 + 0.5) * math.sqrt(10.0 ** 2 + (0.5 - 3.0) ** 2)

        self.assertAlmostEqual(whole.area.to_decimal(u.um ** 2), seg1 + jump_area + seg2, places=9)
        self.assertAlmostEqual(cv_left.area.to_decimal(u.um ** 2), seg1, places=9)
        self.assertAlmostEqual(cv_right.area.to_decimal(u.um ** 2), jump_area + seg2, places=9)
        self.assertAlmostEqual(whole.r_axial.to_decimal(u.ohm),
                               cv_left.r_axial.to_decimal(u.ohm) + cv_right.r_axial.to_decimal(u.ohm), places=9)

    def test_point_tree_internal_attachment_absorbs_to_parent_midpoint(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=0.5)

        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=1))
        point_tree = cell.point_tree()

        root_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 0)
        dend_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 1)
        root_mid_point_id = _point_id_by_role(point_tree, cv_id=root_cv_id, position="mid")
        dend_mid_point_id = _point_id_by_role(point_tree, cv_id=dend_cv_id, position="mid")
        self.assertEqual(len(point_tree.points), cell.n_cv + len(cell.morpho.branches) + 1)
        self.assertIn((dend_cv_id, "proximal"), _point_roles(point_tree, root_mid_point_id))
        self.assertEqual(int(point_tree.point_parent[dend_mid_point_id]), root_mid_point_id)

    def test_point_tree_reuses_non_root_attachment_endpoint(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        twig = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.attach(parent="dend", child_branch=twig, child_name="twig", parent_x=0.0)

        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=1))
        point_tree = cell.point_tree()

        soma_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 0)
        dend_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 1)
        twig_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 2)
        parent_point_id = _point_id_by_role(point_tree, cv_id=soma_cv_id, position="distal")
        twig_mid_point_id = _point_id_by_role(point_tree, cv_id=twig_cv_id, position="mid")
        self.assertEqual(len(point_tree.points), cell.n_cv + len(cell.morpho.branches) + 1)
        self.assertIn((dend_cv_id, "proximal"), _point_roles(point_tree, parent_point_id))
        self.assertIn((twig_cv_id, "proximal"), _point_roles(point_tree, parent_point_id))
        self.assertEqual(int(point_tree.point_parent[twig_mid_point_id]), parent_point_id)

    def test_point_tree_handles_child_x_one_by_reversing_branch_walk(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        twig = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0, child_x=1.0)
        tree.attach(parent="dend", child_branch=twig, child_name="twig", parent_x=0.0)

        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))
        point_tree = cell.point_tree()

        dend_cv_ids = [cv.id for cv in cell.cvs if cv.branch_id == 1]
        twig_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 2)
        dend_terminal_point_id = int(point_tree.branch_terminal_point_id[1])
        twig_mid_point_id = _point_id_by_role(point_tree, cv_id=twig_cv_id, position="mid")
        attachment_point_id = _point_id_by_role(point_tree, cv_id=dend_cv_ids[1], position="distal")
        inter_mid_edge_roles = _edge_roles(
            point_tree,
            parent_point_id=_point_id_by_role(point_tree, cv_id=dend_cv_ids[1], position="mid"),
            child_point_id=_point_id_by_role(point_tree, cv_id=dend_cv_ids[0], position="mid"),
        )

        self.assertIn((dend_cv_ids[1], "distal"), _point_roles(point_tree, attachment_point_id))
        self.assertIn((dend_cv_ids[0], "proximal"), _point_roles(point_tree, dend_terminal_point_id))
        self.assertEqual(int(point_tree.point_parent[twig_mid_point_id]), dend_terminal_point_id)
        self.assertEqual(inter_mid_edge_roles, ((dend_cv_ids[0], "dist"), (dend_cv_ids[1], "prox")))
        self.assertLess(
            int(point_tree.cv_id_to_matrix_index[dend_cv_ids[1]]),
            int(point_tree.cv_id_to_matrix_index[dend_cv_ids[0]]),
        )

    def test_point_tree_aggregates_adjacent_cv_halves_into_single_compute_edge(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))
        point_tree = cell.point_tree()

        left_cv_id, right_cv_id = [cv.id for cv in cell.cvs if cv.branch_id == 0]
        edge_roles = _edge_roles(
            point_tree,
            parent_point_id=_point_id_by_role(point_tree, cv_id=left_cv_id, position="mid"),
            child_point_id=_point_id_by_role(point_tree, cv_id=right_cv_id, position="mid"),
        )
        self.assertEqual(edge_roles, ((left_cv_id, "dist"), (right_cv_id, "prox")))

    def test_point_tree_resolves_roles_and_halves_for_attachment_combos(self) -> None:
        combos = (
            (0.0, 0.0, "proximal", "proximal", "prox", "dist"),
            (0.5, 1.0, "mid", "distal", "dist", "prox"),
            (1.0, 0.0, "distal", "proximal", "prox", "dist"),
            (1.0, 1.0, "distal", "distal", "dist", "prox"),
        )
        for parent_x, child_x, expected_parent_position, expected_child_attach_position, expected_entry_half, expected_exit_half in combos:
            with self.subTest(parent_x=parent_x, child_x=child_x):
                soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
                dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
                morpho = Morphology.from_root(soma, name="soma")
                morpho.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=parent_x, child_x=child_x)

                cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=1))
                point_tree = cell.point_tree()
                parent_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 0)
                child_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 1)
                child_mid_point_id = _point_id_by_role(point_tree, cv_id=child_cv_id, position="mid")
                expected_parent_point_id = _point_id_by_role(point_tree, cv_id=parent_cv_id,
                                                             position=expected_parent_position)
                child_terminal_point_id = int(point_tree.branch_terminal_point_id[1])

                self.assertEqual(len(point_tree.edges), len(point_tree.points) - 1)
                self.assertEqual(int(point_tree.point_parent[child_mid_point_id]), expected_parent_point_id)
                self.assertIn((child_cv_id, expected_child_attach_position),
                              _point_roles(point_tree, expected_parent_point_id))
                self.assertEqual(
                    _edge_roles(
                        point_tree,
                        parent_point_id=expected_parent_point_id,
                        child_point_id=child_mid_point_id,
                    ),
                    ((child_cv_id, expected_entry_half),),
                )
                self.assertEqual(
                    _edge_roles(
                        point_tree,
                        parent_point_id=child_mid_point_id,
                        child_point_id=child_terminal_point_id,
                    ),
                    ((child_cv_id, expected_exit_half),),
                )

    def test_point_scheduling_splits_dhs_groups_by_max_group_size(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        left = Branch.from_lengths(lengths=[20.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        right = Branch.from_lengths(lengths=[20.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=left, child_name="left", parent_x=0.5)
        tree.attach(parent="soma", child_branch=right, child_name="right", parent_x=0.5)

        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=1))
        point_tree = cell.point_tree()
        scheduling = cell.point_scheduling(max_group_size=1)

        self.assertEqual(scheduling.algorithm, "dhs")
        self.assertTrue(all(group.shape[0] == 1 for group in scheduling.groups))
        self.assertTrue(all(size == 1 for size in scheduling.level_size.tolist()))
        self.assertEqual(int(scheduling.row_to_point_id[0]), point_tree.root_point_id)
        self.assertEqual(int(scheduling.parent_rows[0]), -1)
        root_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 0)
        left_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 1)
        right_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 2)
        root_mid_row = _row_by_role(scheduling, point_tree, cv_id=root_cv_id, position="mid")
        left_mid_row = _row_by_role(scheduling, point_tree, cv_id=left_cv_id, position="mid")
        right_mid_row = _row_by_role(scheduling, point_tree, cv_id=right_cv_id, position="mid")
        self.assertEqual(int(scheduling.parent_rows[left_mid_row]), root_mid_row)
        self.assertEqual(int(scheduling.parent_rows[right_mid_row]), root_mid_row)

    def test_cv_as_branch(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))
        branch_slice = cell.cvs[1].as_branch()
        self.assertEqual(branch_slice.type, "soma")
        self.assertGreater(float(branch_slice.length.to_decimal(u.um)), 0.0)

    def test_type_validation_and_removed_legacy_exports(self) -> None:
        tree = _build_tree()
        cell = Cell(tree)
        with self.assertRaises(TypeError):
            Cell(tree.soma)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            # RootLocation is not a RegionExpr.
            cell.paint(
                RootLocation(x=0.5),
                braincell.mech.Channel("IL"),
            )  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            # CurrentClamp is a PointMechanism, not a density mechanism — rejected by place when passed a BranchSlice since the locset type is wrong.
            cell.place(
                BranchSlice(branch_index=0, prox=0.0, dist=1.0),  # type: ignore[arg-type]
                CurrentClamp.step(0.2 * u.nA, 1.0 * u.ms, delay=2.0 * u.ms),
            )
        with self.assertRaises(ValueError):
            cell.paint(BranchSlice(branch_index=0, prox=0.0, dist=1.0))
        with self.assertRaises(ValueError):
            cell.place(RootLocation(x=0.5))
        with self.assertRaises(TypeError):
            cell.paint(BranchSlice(branch_index=0, prox=0.0, dist=1.0), braincell.mech.Synapse("Foo"))
        self.assertFalse(hasattr(braincell, "DiscretizationPolicy"))

    def test_lazy_rebuild_only_happens_on_query(self) -> None:
        cell = Cell(_build_tree())
        original_cvs = cell._cvs
        self.assertIsNotNone(original_cvs)

        cell.cv_policy = CVPerBranch(cv_per_branch=2)
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.4, dist=0.6),
            CableProperty(
                resting_potential=-75.0 * u.mV,
                membrane_capacitance=1.2 * (u.uF / u.cm ** 2),
                axial_resistivity=120.0 * (u.ohm * u.cm),
            ),
        )
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.05 * u.nA, 1.0 * u.ms, delay=1.0 * u.ms),
        )
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)
        self.assertIsNone(cell._point_tree)
        self.assertEqual(cell._point_scheduling, {})

        self.assertEqual(cell.n_cv, 4)
        self.assertFalse(cell._dirty)
        self.assertIsNot(cell._cvs, original_cvs)

    def test_density_paint_channel_fallback_records_area_fraction(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            braincell.mech.Channel("leaky", g_max="not-scalable"),
            braincell.mech.Channel("leaky", tau=10.0),
        )
        cv0 = cell.cvs[0]
        self.assertEqual(len(cv0.density_mech), 2)

        with_gmax = next(mech for mech in cv0.density_mech if "g_max" in mech.params)
        self.assertEqual(with_gmax.params["g_max"], "not-scalable")
        self.assertAlmostEqual(with_gmax.coverage_area_fraction, 0.5, places=12)
        # coverage is a first-class field now, not a pseudo-parameter.
        self.assertNotIn("coverage_area_fraction", with_gmax.params)

        no_gmax = next(mech for mech in cv0.density_mech if "tau" in mech.params)
        self.assertEqual(no_gmax.params["tau"], 10.0)
        self.assertAlmostEqual(no_gmax.coverage_area_fraction, 0.5, places=12)
        self.assertNotIn("coverage_area_fraction", no_gmax.params)

    def test_mechanism_object_table_later_paint_overrides_same_row_same_cv(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            braincell.filter.AllRegion(),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70.0 * u.mV),
        )
        cell.paint(
            braincell.filter.BranchInFilter("type", "soma"),
            braincell.mech.Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        table = cell.mech_table()
        soma_point_ids = {
            int(cell.point_tree().cv_midpoint_point_id[cv.id])
            for cv in cell.cvs
            if cv.branch_type == "soma"
        }
        dend_point_ids = {
            int(cell.point_tree().cv_midpoint_point_id[cv.id])
            for cv in cell.cvs
            if cv.branch_type == "basal_dendrite"
        }

        for point_id in soma_point_ids:
            soma_cell = table.get_by_label("IL", point_id)
            self.assertIsNotNone(soma_cell)
            assert soma_cell is not None
            self.assertAlmostEqual(float(soma_cell.g_max.to_decimal(u.mS / u.cm ** 2)), 0.2, places=12)
        for point_id in dend_point_ids:
            dend_cell = table.get_by_label("IL", point_id)
            self.assertIsNotNone(dend_cell)
            assert dend_cell is not None
            self.assertAlmostEqual(float(dend_cell.g_max.to_decimal(u.mS / u.cm ** 2)), 0.1, places=12)

    def test_mech_table_uses_midpoint_points(self) -> None:
        cell = Cell(_build_three_branch_tree(), cv_policy=CVPerBranch(cv_per_branch=2), solver="staggered")
        cell.paint(
            braincell.filter.AllRegion(),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70.0 * u.mV),
        )
        cell.paint(
            braincell.filter.BranchInFilter("type", "soma"),
            braincell.mech.Channel("INa_HH1952", g_max=120.0 * (u.mS / u.cm ** 2)),
            braincell.mech.Channel("IK_HH1952", g_max=36.0 * (u.mS / u.cm ** 2)),
        )
        cell.paint(
            braincell.filter.BranchInFilter("type", "basal_dendrite"),
            braincell.mech.Channel("INa_HH1952", g_max=100.0 * (u.mS / u.cm ** 2)),
            braincell.mech.Channel("IK_HH1952", g_max=30.0 * (u.mS / u.cm ** 2)),
        )
        cell.paint(
            braincell.filter.BranchInFilter("type", "axon"),
            braincell.mech.Channel("INa_HH1952", g_max=140.0 * (u.mS / u.cm ** 2)),
            braincell.mech.Channel("IK_HH1952", g_max=40.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()

        table = cell.mech_table()
        self.assertEqual(table.row_labels, ("IL", "INa_HH1952", "IK_HH1952"))
        self.assertEqual(table.shape, (3, len(cell.point_tree().points)))

        midpoint_ids = {int(cell.point_tree().cv_midpoint_point_id[cv.id]) for cv in cell.cvs}
        for point_id in table.column_ids:
            point_cell = table.get_by_label("IL", point_id)
            if point_id in midpoint_ids:
                self.assertIsNotNone(point_cell)
                assert point_cell is not None
                self.assertEqual(point_cell.point_id, point_id)
                self.assertAlmostEqual(float(point_cell.g_max.to_decimal(u.mS / u.cm ** 2)), 0.1, places=12)
            else:
                self.assertIsNone(point_cell)

        soma_point_ids = {
            int(cell.point_tree().cv_midpoint_point_id[cv.id])
            for cv in cell.cvs
            if cv.branch_type == "soma"
        }
        dend_point_ids = {
            int(cell.point_tree().cv_midpoint_point_id[cv.id])
            for cv in cell.cvs
            if cv.branch_type == "basal_dendrite"
        }
        axon_point_ids = {
            int(cell.point_tree().cv_midpoint_point_id[cv.id])
            for cv in cell.cvs
            if cv.branch_type == "axon"
        }

        for point_id in soma_point_ids:
            ina = table.get_by_label("INa_HH1952", point_id)
            ik = table.get_by_label("IK_HH1952", point_id)
            self.assertIsNotNone(ina)
            self.assertIsNotNone(ik)
            assert ina is not None and ik is not None
            self.assertAlmostEqual(float(ina.g_max.to_decimal(u.mS / u.cm ** 2)), 120.0, places=12)
            self.assertAlmostEqual(float(ik.g_max.to_decimal(u.mS / u.cm ** 2)), 36.0, places=12)

        for point_id in dend_point_ids:
            ina = table.get_by_label("INa_HH1952", point_id)
            ik = table.get_by_label("IK_HH1952", point_id)
            self.assertIsNotNone(ina)
            self.assertIsNotNone(ik)
            assert ina is not None and ik is not None
            self.assertAlmostEqual(float(ina.g_max.to_decimal(u.mS / u.cm ** 2)), 100.0, places=12)
            self.assertAlmostEqual(float(ik.g_max.to_decimal(u.mS / u.cm ** 2)), 30.0, places=12)

        for point_id in axon_point_ids:
            ina = table.get_by_label("INa_HH1952", point_id)
            ik = table.get_by_label("IK_HH1952", point_id)
            self.assertIsNotNone(ina)
            self.assertIsNotNone(ik)
            assert ina is not None and ik is not None
            self.assertAlmostEqual(float(ina.g_max.to_decimal(u.mS / u.cm ** 2)), 140.0, places=12)
            self.assertAlmostEqual(float(ik.g_max.to_decimal(u.mS / u.cm ** 2)), 40.0, places=12)


class CellExecutionTest(unittest.TestCase):
    def test_cell_is_runtime_object_after_init_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm ** 2)),
        )

        cell.init_state()

        self.assertIsInstance(cell, braincell.HHTypedNeuron)
        self.assertEqual(cell.varshape, (2,))
        self.assertIs(cell.ion_channels["na"], cell.get_ion("na"))
        self.assertIs(cell.ion_channels["k"], cell.get_ion("k"))
        self.assertIs(cell.ion_channels["ca"], cell.get_ion("ca"))
        self.assertTrue(any(key.startswith("layout_") for key in cell.ion_channels))

    def test_reset_state_reinitializes_cell_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertEqual(cell.spike.value.shape, (2,))

        original = cell.V.value
        cell.V.value = cell.V.value + 1.0 * u.mV
        cell.reset_state()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertFalse(u.math.all(cell.V.value == original + 1.0 * u.mV))

    def test_compute_derivative_and_update_work_on_cell(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state()
        cell.compute_derivative()
        self.assertEqual(cell.V.derivative.shape, (2,))

        with brainstate.environ.context(dt=0.01 * u.ms):
            spike = cell.update()
        self.assertEqual(spike.shape, (2,))

    def test_run_returns_result_with_all_probe_traces(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm ** 2)),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="p"),
        )
        cell.init_state()

        result = cell.run(dt=0.01 * u.ms, duration=0.05 * u.ms)

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.time.shape, (5,))
        self.assertEqual(sorted(result.traces), ["soma(0.5)_INa_HH1952_p", "soma(0.5)_v"])
        self.assertEqual(result.traces["soma(0.5)_v"].shape, (5,))
        self.assertEqual(result.traces["soma(0.5)_INa_HH1952_p"].shape, (5,))
        self.assertEqual(cell.current_time, 0.05 * u.ms)

    def test_run_with_single_probe_returns_single_trace(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
        )
        cell.init_state()

        result = cell.run(dt=0.01 * u.ms, duration=0.03 * u.ms)

        self.assertEqual(result.time.shape, (3,))
        self.assertEqual(list(result.traces), ["soma(0.5)_v"])
        self.assertEqual(result.traces["soma(0.5)_v"].shape, (3,))

    def test_run_requires_initialized_state(self) -> None:
        cell = Cell(_build_tree())
        cell.place(at("soma", 0.5), braincell.mech.StateProbe())

        with self.assertRaises(ValueError):
            cell.run(dt=0.01 * u.ms, duration=0.03 * u.ms)

    def test_run_requires_at_least_one_probe(self) -> None:
        cell = Cell(_build_tree())
        cell.init_state()

        with self.assertRaises(ValueError):
            cell.run(dt=0.01 * u.ms, duration=0.03 * u.ms)

    def test_run_validates_dt_and_duration(self) -> None:
        cell = Cell(_build_tree())
        cell.place(at("soma", 0.5), braincell.mech.StateProbe())
        cell.init_state()

        with self.assertRaises(ValueError):
            cell.run(dt=0.0 * u.ms, duration=0.03 * u.ms)
        with self.assertRaises(ValueError):
            cell.run(dt=0.01 * u.ms, duration=0.0 * u.ms)
        with self.assertRaises(TypeError):
            cell.run(dt=0.01, duration=0.03 * u.ms)  # type: ignore[arg-type]

    def test_run_advances_absolute_time_across_calls_and_reset_state_resets_it(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm ** 2)),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="p"),
            braincell.CurrentClamp.step(0.2 * u.nA, 0.01 * u.ms, delay=0.0 * u.ms),
        )
        cell.init_state()

        first = cell.run(dt=0.01 * u.ms, duration=0.02 * u.ms)
        second = cell.run(dt=0.01 * u.ms, duration=0.02 * u.ms)

        self.assertEqual(first.time[0], 0.0 * u.ms)
        self.assertEqual(first.time[-1], 0.01 * u.ms)
        self.assertEqual(second.time[0], 0.02 * u.ms)
        self.assertEqual(second.time[-1], 0.03 * u.ms)
        self.assertEqual(cell.current_time, 0.04 * u.ms)
        runtime = cell._ensure_runtime_compiled()
        current_after_window = runtime.evaluate_point_clamps(t=second.time[0])
        self.assertAlmostEqual(float(current_after_window[1].to_decimal(u.nA)), 0.0, places=12)

        cell.reset_state()
        self.assertEqual(cell.current_time, 0.0 * u.ms)

    def test_run_is_make_jaxpr_safe(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm ** 2)),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="p"),
        )
        cell.init_state()

        jaxpr, _ = brainstate.transform.make_jaxpr(
            lambda: cell.run(dt=0.01 * u.ms, duration=0.02 * u.ms)
        )()

        self.assertIsNotNone(jaxpr)

    def test_staggered_solver_updates_cv_sized_voltage(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state()
        with brainstate.environ.context(dt=0.01 * u.ms):
            spike = cell.update()
        self.assertEqual(cell.V.value.shape, (2,))
        self.assertEqual(spike.shape, (2,))

    def test_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="exp_euler")
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()

        with mock.patch("braincell.quad._exp_euler.apply_standard_solver_step") as step_mock:
            with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
                cell.update()

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "concat")

    def test_splitting_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="splitting")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                braincell.quad.splitting_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)

    def test_cn_rk4_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="cn_rk4")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.rk4_step") as rk4_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap",
                                side_effect=RuntimeError("bridge reached")):
                    braincell.quad.cn_rk4_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(rk4_mock.called)

    def test_cn_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="cn_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap",
                                side_effect=RuntimeError("bridge reached")):
                    braincell.quad.cn_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_implicit_rk4_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="implicit_rk4")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.rk4_step") as rk4_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap",
                                side_effect=RuntimeError("bridge reached")):
                    braincell.quad.implicit_rk4_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(rk4_mock.called)

    def test_implicit_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="implicit_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap",
                                side_effect=RuntimeError("bridge reached")):
                    braincell.quad.implicit_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_exp_exp_euler_solver_uses_cell_bridge_without_legacy_multicompartment(self) -> None:
        cell = Cell(_build_tree(), solver="exp_exp_euler")
        cell.init_state()

        with mock.patch("braincell.quad._implicit.apply_standard_solver_step") as step_mock:
            with self.assertRaises(UnboundLocalError):
                with mock.patch("braincell.quad._implicit.brainstate.transform.vmap",
                                side_effect=RuntimeError("bridge reached")):
                    braincell.quad.exp_exp_euler_step(cell, 0.0 * u.ms, 0.01 * u.ms)

        self.assertTrue(step_mock.called)
        self.assertEqual(step_mock.call_args.kwargs["merging"], "stack")

    def test_staggered_solver_single_cv_leak_moves_toward_reversal(self) -> None:
        cell = Cell(_build_soma_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state()
        initial = cell.V.value
        with brainstate.environ.context(dt=0.01 * u.ms):
            cell.update()
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), float(initial[0].to_decimal(u.mV)))
        self.assertGreater(float(cell.V.value[0].to_decimal(u.mV)), -68.0)

    def test_total_current_input_matches_current_density_input(self) -> None:
        cell_total = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell_density = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        for cell in (cell_total, cell_density):
            cell.paint(
                BranchSlice(branch_index=0, prox=0.0, dist=1.0),
                braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
            )
            cell.init_state()

        initial = -65.0 * u.mV
        cell_total.V.value = u.math.asarray([initial.to_decimal(u.mV)]) * u.mV
        cell_density.V.value = u.math.asarray([initial.to_decimal(u.mV)]) * u.mV
        area = cell_total.cvs[0].area
        total_current = 0.05 * u.nA
        current_density = total_current / area

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            cell_total.update(total_current)
            cell_density.update(current_density)

        self.assertAlmostEqual(
            float(cell_total.V.value[0].to_decimal(u.mV)),
            float(cell_density.V.value[0].to_decimal(u.mV)),
            places=6,
        )

    def test_compute_axial_derivative_is_zero_for_single_cv(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.init_state()

        derivative = cell.compute_axial_derivative(u.math.asarray([-65.0]) * u.mV)
        self.assertEqual(derivative.shape, (1,))
        self.assertAlmostEqual(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_compute_axial_derivative_couples_two_cv_cable(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.init_state()

        derivative = cell.compute_axial_derivative(u.math.asarray([-60.0, -80.0]) * u.mV)
        self.assertEqual(derivative.shape, (2,))
        self.assertLess(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertGreater(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0)

    def test_compute_axial_derivative_is_zero_for_uniform_multi_cv_voltage(self) -> None:
        cell = Cell(_build_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.init_state()

        derivative = cell.compute_axial_derivative(u.math.asarray([-65.0, -65.0, -65.0, -65.0]) * u.mV)
        self.assertEqual(derivative.shape, (4,))
        for value in derivative:
            self.assertAlmostEqual(float(value.to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_explicit_solver_uses_axial_coupling(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.init_state()
        cell.V.value = u.math.asarray([-60.0, -80.0]) * u.mV

        with brainstate.environ.context(dt=1e-6 * u.ms, t=0.0 * u.ms):
            cell.update()

        self.assertGreater(float(cell.V.value[0].to_decimal(u.mV)), -80.0)
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), -60.0)
        self.assertGreater(float(cell.V.value[1].to_decimal(u.mV)), -80.0)
        self.assertLess(float(cell.V.value[1].to_decimal(u.mV)), -60.0)

    def test_staggered_solver_two_cv_passive_cable_stays_symmetric(self) -> None:
        tree = _build_soma_tree()
        cell = Cell(tree, solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=2))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state()
        initial = cell.V.value
        with brainstate.environ.context(dt=0.01 * u.ms):
            cell.update()

        self.assertEqual(cell.V.value.shape, (2,))
        self.assertLess(float(cell.V.value[0].to_decimal(u.mV)), float(initial[0].to_decimal(u.mV)))
        self.assertLess(float(cell.V.value[1].to_decimal(u.mV)), float(initial[1].to_decimal(u.mV)))
        self.assertAlmostEqual(
            float(cell.V.value[0].to_decimal(u.mV)),
            float(cell.V.value[1].to_decimal(u.mV)),
            places=6,
        )

    def test_staggered_solver_branched_passive_cell_matches_explicit_direction(self) -> None:
        tree = _build_tree()
        explicit = Cell(tree, solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        staggered = Cell(tree, solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        for cell in (explicit, staggered):
            cell.paint(
                BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
                braincell.mech.Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
            )
            cell.init_state()

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            explicit.update()
            staggered.update()

        self.assertEqual(staggered.V.value.shape, explicit.V.value.shape)
        for index in range(staggered.V.value.shape[0]):
            explicit_v = float(explicit.V.value[index].to_decimal(u.mV))
            staggered_v = float(staggered.V.value[index].to_decimal(u.mV))
            self.assertLess(staggered_v, -65.0)
            self.assertGreater(staggered_v, -68.0)
            self.assertAlmostEqual(staggered_v, explicit_v, places=3)

    def test_update_requires_init_state_after_declaration_change(self) -> None:
        cell = Cell(_build_tree())
        cell.init_state()
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        with self.assertRaisesRegex(ValueError, "Cell.init_state"):
            cell.compute_derivative()

    def test_placed_current_clamp_removes_manual_step_current_logic(self) -> None:
        cell = Cell(_build_soma_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp(
                start=1.0 * u.ms,
                durations=(2.0 * u.ms, 2.0 * u.ms),
                amplitudes=(0.0 * u.nA, 0.2 * u.nA),
            ),
        )
        cell.init_state()

        with brainstate.environ.context(t=0.5 * u.ms):
            early = cell.compute_membrane_derivative(cell.V.value)
        with brainstate.environ.context(t=3.5 * u.ms):
            late = cell.compute_membrane_derivative(cell.V.value)
        with brainstate.environ.context(t=5.5 * u.ms):
            after = cell.compute_membrane_derivative(cell.V.value)

        self.assertAlmostEqual(float(early[0].to_decimal(u.mV / u.ms)), 0.0, places=8)
        self.assertGreater(float(late[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertAlmostEqual(float(after[0].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_point_clamp_input_normalizes_only_active_midpoints(self) -> None:
        cell = Cell(_build_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp.step(0.2 * u.nA, 10.0 * u.ms, delay=0.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(t=1.0 * u.ms):
            point_clamp_input = np.asarray(cell._point_clamp_input().to_decimal(u.nA / (u.cm ** 2)), dtype=float)

        self.assertFalse(np.isnan(point_clamp_input).any())
        self.assertGreater(point_clamp_input[1], 0.0)
        self.assertAlmostEqual(point_clamp_input[0], 0.0, places=12)
        self.assertAlmostEqual(point_clamp_input[2], 0.0, places=12)
        self.assertAlmostEqual(point_clamp_input[4], 0.0, places=12)

    def test_placed_current_clamp_targets_only_selected_point(self) -> None:
        cell = Cell(_build_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp.step(0.2 * u.nA, 10.0 * u.ms, delay=0.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(t=1.0 * u.ms):
            derivative = cell.compute_membrane_derivative(cell.V.value)

        self.assertGreater(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0)
        self.assertAlmostEqual(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0, places=8)

    def test_default_zero_external_current_does_not_suppress_point_clamp(self) -> None:
        cell = Cell(_build_tree(), solver="staggered", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            RootLocation(x=0.5),
            braincell.CurrentClamp.step(0.2 * u.nA, 10.0 * u.ms, delay=0.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(dt=0.01 * u.ms, t=0.0 * u.ms):
            before = float(cell.V.value[0].to_decimal(u.mV))
            cell.update()
            after = float(cell.V.value[0].to_decimal(u.mV))

        self.assertGreater(after, before)

    def test_multiple_terminal_clamps_do_not_broadcast_to_all_points(self) -> None:
        cell = Cell(_build_tree(), solver="explicit", cv_policy=braincell.CVPerBranch(cv_per_branch=1))
        cell.place(
            Terminals(),
            braincell.CurrentClamp.step(0.15 * u.nA, 10.0 * u.ms, delay=0.0 * u.ms),
        )
        cell.init_state()

        with brainstate.environ.context(t=1.0 * u.ms):
            derivative = cell.compute_membrane_derivative(cell.V.value)

        self.assertAlmostEqual(float(derivative[0].to_decimal(u.mV / u.ms)), 0.0, places=8)
        self.assertGreater(float(derivative[1].to_decimal(u.mV / u.ms)), 0.0)
