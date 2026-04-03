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

import brainunit as u

import braincell

from braincell import (
    Branch,
    CVPolicy,
    CableProperties,
    Cell,
    CurrentClamp,
    DensityMechanism,
    Morpho,
)
from braincell.filter import BranchSlice, RootLocation


def _build_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend
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


class CellFacadeTest(unittest.TestCase):
    def test_default_cell_has_cv_and_default_paint_rules(self) -> None:
        cell = Cell(_build_tree())
        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(len(cell.paint_rules), 1)
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -65.0, places=12)
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), 1.0, places=12)
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
            CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms),
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
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), base.cm.to_decimal(u.uF / u.cm**2), places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), base.ra.to_decimal(u.ohm * u.cm), places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), base.v.to_decimal(u.mV), places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), base.temp.to_decimal(u.kelvin), places=12)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.49, dist=0.51),
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cvs[0]
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), 2.0, places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), 200.0, places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -70.0, places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), u.celsius2kelvin(20.0).to_decimal(u.kelvin), places=12)

    def test_cable_paint_compacts_same_region_history(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        cell.paint(
            region,
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        cell.paint(
            region,
            CableProperties(
                resting_potential=-60.0 * u.mV,
                membrane_capacitance=3.0 * (u.uF / u.cm**2),
                axial_resistivity=300.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(30.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        last = next(rule for rule in cell.paint_rules if rule.region == region)
        cable = last.mechanism
        self.assertIsInstance(cable, CableProperties)
        self.assertAlmostEqual(cable.resting_potential.to_decimal(u.mV), -60.0, places=12)
        self.assertAlmostEqual(cable.membrane_capacitance.to_decimal(u.uF / u.cm**2), 3.0, places=12)
        self.assertAlmostEqual(cable.axial_resistivity.to_decimal(u.ohm * u.cm), 300.0, places=12)
        self.assertAlmostEqual(
            cable.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(30.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_density_paint_channel_scales_by_area_fraction(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            DensityMechanism(channel_type="leaky", params=(("g_max", 4.0 * (u.mS / u.cm**2)),)),
            DensityMechanism(ion_type="sodium", params=(("c0", 12.0),)),
        )
        cv0 = cell.cvs[0]
        self.assertEqual(len(cv0.density_mech), 2)
        channel = next(mech for mech in cv0.density_mech if mech.channel_type is not None)
        ion = next(mech for mech in cv0.density_mech if mech.ion_type is not None)
        gmax = dict(channel.params)["g_max"]
        self.assertAlmostEqual(float(gmax.to_decimal(u.mS / u.cm**2)), 2.0, places=12)
        self.assertEqual(dict(ion.params)["c0"], 12.0)

    def test_place_boundary_goes_to_right_cv_and_branch_endpoint_stays_local(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))

        stim_boundary = CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms)
        stim_parent_end = CurrentClamp(amplitude=0.2 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms)
        cell.place(RootLocation(x=0.5), stim_boundary)
        cell.place(RootLocation(x=1.0), stim_parent_end)

        # RootLocation(x=0.5) on branch 0 should map to right CV (id 1).
        self.assertEqual(len(cell.cvs[0].point_mech), 0)
        self.assertEqual(len(cell.cvs[1].point_mech), 2)
        # parent branch endpoint remains on branch 0 last CV, not child branch first CV.
        self.assertEqual(cell.cvs[1].point_mech[-1].amplitude.to_decimal(u.nA), 0.2)
        self.assertEqual(len(cell.cvs[2].point_mech), 0)

    def test_cv_geometry_and_split_axial_resistance(self) -> None:
        soma = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[1.0, 2.0, 3.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1))
        cv0 = cell.cvs[0]
        self.assertFalse(hasattr(cv0, "mean_radius"))

        area1 = math.pi * (1.0 + 2.0) * math.sqrt(10.0**2 + (2.0 - 1.0) ** 2)
        area2 = math.pi * (2.0 + 3.0) * math.sqrt(20.0**2 + (3.0 - 2.0) ** 2)
        self.assertAlmostEqual(cv0.area.to_decimal(u.um**2), area1 + area2, places=9)

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

    def test_zero_length_radius_jump_contributes_area_but_not_axial_resistance(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[1.0, 0.5] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        whole = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1)).cvs[0]
        split = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
        cv_left = split.cvs[0]
        cv_right = split.cvs[1]

        jump_area = math.pi * (3.0**2 - 1.0**2)
        seg1 = math.pi * (2.0 + 1.0) * math.sqrt(10.0**2 + (1.0 - 2.0) ** 2)
        seg2 = math.pi * (3.0 + 0.5) * math.sqrt(10.0**2 + (0.5 - 3.0) ** 2)

        self.assertAlmostEqual(whole.area.to_decimal(u.um**2), seg1 + jump_area + seg2, places=9)
        self.assertAlmostEqual(cv_left.area.to_decimal(u.um**2), seg1, places=9)
        self.assertAlmostEqual(cv_right.area.to_decimal(u.um**2), jump_area + seg2, places=9)
        self.assertAlmostEqual(whole.r_axial.to_decimal(u.ohm), cv_left.r_axial.to_decimal(u.ohm) + cv_right.r_axial.to_decimal(u.ohm), places=9)

    def test_point_tree_internal_attachment_absorbs_to_parent_midpoint(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=0.5)

        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1))
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
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.attach(parent="dend", child_branch=twig, child_name="twig", parent_x=0.0)

        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1))
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
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0, child_x=1.0)
        tree.attach(parent="dend", child_branch=twig, child_name="twig", parent_x=0.0)

        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
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
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
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
                morpho = Morpho.from_root(soma, name="soma")
                morpho.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=parent_x, child_x=child_x)

                cell = Cell(morpho, cv_policy=CVPolicy(cv_per_branch=1))
                point_tree = cell.point_tree()
                parent_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 0)
                child_cv_id = next(cv.id for cv in cell.cvs if cv.branch_id == 1)
                child_mid_point_id = _point_id_by_role(point_tree, cv_id=child_cv_id, position="mid")
                expected_parent_point_id = _point_id_by_role(point_tree, cv_id=parent_cv_id, position=expected_parent_position)
                child_terminal_point_id = int(point_tree.branch_terminal_point_id[1])

                self.assertEqual(len(point_tree.edges), len(point_tree.points) - 1)
                self.assertEqual(int(point_tree.point_parent[child_mid_point_id]), expected_parent_point_id)
                self.assertIn((child_cv_id, expected_child_attach_position), _point_roles(point_tree, expected_parent_point_id))
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
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=left, child_name="left", parent_x=0.5)
        tree.attach(parent="soma", child_branch=right, child_name="right", parent_x=0.5)

        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1))
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
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
        branch_slice = cell.cvs[1].as_branch()
        self.assertEqual(branch_slice.type, "soma")
        self.assertGreater(float(branch_slice.length.to_decimal(u.um)), 0.0)

    def test_type_validation_and_removed_legacy_exports(self) -> None:
        tree = _build_tree()
        cell = Cell(tree)
        with self.assertRaises(TypeError):
            Cell(tree.soma)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cell.paint(RootLocation(x=0.5), DensityMechanism(channel_type="hh"))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cell.place(BranchSlice(branch_index=0, prox=0.0, dist=1.0), CurrentClamp(  # type: ignore[arg-type]
                amplitude=0.2 * u.nA,
                delay=2.0 * u.ms,
                duration=1.0 * u.ms,
            ))
        with self.assertRaises(ValueError):
            cell.paint(BranchSlice(branch_index=0, prox=0.0, dist=1.0))
        with self.assertRaises(ValueError):
            cell.place(RootLocation(x=0.5))
        self.assertFalse(hasattr(braincell, "DiscretizationPolicy"))

    def test_lazy_rebuild_only_happens_on_query(self) -> None:
        cell = Cell(_build_tree())
        original_cvs = cell._cvs
        self.assertIsNotNone(original_cvs)

        cell.cv_policy = CVPolicy(cv_per_branch=2)
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.4, dist=0.6),
            CableProperties(
                resting_potential=-75.0 * u.mV,
                membrane_capacitance=1.2 * (u.uF / u.cm**2),
                axial_resistivity=120.0 * (u.ohm * u.cm),
            ),
        )
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(amplitude=0.05 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms),
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
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            DensityMechanism(channel_type="leaky", params=(("g_max", "not-scalable"),)),
            DensityMechanism(channel_type="leaky", params=(("tau", 10.0),)),
        )
        cv0 = cell.cvs[0]
        self.assertEqual(len(cv0.density_mech), 2)

        with_gmax = next(
            mech for mech in cv0.density_mech if "g_max" in dict(mech.params)
        )
        self.assertEqual(dict(with_gmax.params)["g_max"], "not-scalable")
        self.assertAlmostEqual(dict(with_gmax.params)["coverage_area_fraction"], 0.5, places=12)

        no_gmax = next(
            mech for mech in cv0.density_mech if "tau" in dict(mech.params)
        )
        self.assertEqual(dict(no_gmax.params)["tau"], 10.0)
        self.assertAlmostEqual(dict(no_gmax.params)["coverage_area_fraction"], 0.5, places=12)

    def test_public_exports_remain_stable(self) -> None:
        from braincell import CVPolicy as PublicCVPolicy
        from braincell import Cell as PublicCell
        from braincell.cell import CV as PublicCV
        from braincell.cell import CVEdge as PublicCVEdge
        from braincell.cell import CVPoint as PublicCVPoint
        from braincell.cell import ComputeEdge as PublicComputeEdge
        from braincell.cell import ComputePoint as PublicComputePoint
        from braincell.cell import PaintRule as PublicPaintRule
        from braincell.cell import PointScheduling as PublicPointScheduling
        from braincell.cell import PointTree as PublicPointTree
        from braincell.cell import PlaceRule as PublicPlaceRule
        import braincell.cell as public_cell_module

        self.assertIs(PublicCell, Cell)
        self.assertIs(PublicCVPolicy, CVPolicy)
        self.assertEqual(PublicCV.__name__, "CV")
        self.assertEqual(PublicCVEdge.__name__, "CVEdge")
        self.assertEqual(PublicCVPoint.__name__, "CVPoint")
        self.assertEqual(PublicComputeEdge.__name__, "ComputeEdge")
        self.assertEqual(PublicComputePoint.__name__, "ComputePoint")
        self.assertEqual(PublicPaintRule.__name__, "PaintRule")
        self.assertEqual(PublicPointScheduling.__name__, "PointScheduling")
        self.assertEqual(PublicPointTree.__name__, "PointTree")
        self.assertEqual(PublicPlaceRule.__name__, "PlaceRule")
        self.assertFalse(hasattr(public_cell_module, "AxialEdge"))
