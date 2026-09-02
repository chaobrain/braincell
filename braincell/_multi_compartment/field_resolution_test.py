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

"""Tests for :mod:`braincell._multi_compartment.field_resolution`."""

import unittest

import brainunit as u
import numpy as np

from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell._multi_compartment import field_resolution as fr
from braincell.filter import AllRegion, BranchSlice, RootLocation
from braincell.mech import Channel


def _soma_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    return Morphology.from_root(soma, name="soma")


def _soma_dend_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class CoercersAgreeAcrossStorageFlavoursTest(unittest.TestCase):
    """United and unitless fields must coerce to the same numbers.

    Each coercer used to spell its length rules out twice, once for a
    ``Quantity`` and once for a bare array. They are now stated once and
    the unit is split off by :func:`~.field_resolution.split_unit`, so
    this pins the property that made the duplication removable in the
    first place.
    """

    #: Every coercer that maps a caller-supplied field into one space.
    #: Each is called as ``coerce(cell, value)``; the ``caller=`` string
    #: only reaches error messages, so it is supplied uniformly here.
    COERCERS = (
        "coerce_node_values",
        "coerce_cv_values",
        "coerce_named_node_values",
        "coerce_named_cv_values",
    )

    def setUp(self) -> None:
        self.cell = Cell(_soma_tree(), cv_policy=CVPerBranch(2))
        self.cell.init_state()

    def _coercers(self):
        yield "coerce_runtime_point_values", lambda value: fr.coerce_runtime_point_values(self.cell, value)
        for name in self.COERCERS:
            function = getattr(fr, name)
            yield name, lambda value, function=function: function(self.cell, value, caller="test")

    def _cases(self):
        # Scalar, point-length, and CV-length: the three shapes every
        # coercer branches on.
        return {
            "scalar": 1.5,
            "point": np.linspace(-70.0, -60.0, self.cell.n_point),
            "cv": np.linspace(-70.0, -60.0, self.cell.n_cv),
        }

    def test_unit_carrying_and_bare_inputs_give_the_same_numbers(self) -> None:
        for name, coerce in self._coercers():
            for label, plain in self._cases().items():
                with self.subTest(coercer=name, case=label):
                    bare = coerce(plain)
                    united = coerce(plain * u.mV)
                    self.assertTrue(u.get_unit(united).has_same_dim(u.mV))
                    np.testing.assert_allclose(
                        np.asarray(u.get_mantissa(bare), dtype=float),
                        np.asarray(united.to_decimal(u.mV), dtype=float),
                    )

    def test_a_length_that_is_neither_space_is_rejected_either_way(self) -> None:
        bad = np.zeros(self.cell.n_point + self.cell.n_cv + 1)
        for name, coerce in self._coercers():
            with self.subTest(coercer=name, storage="bare"):
                with self.assertRaises(ValueError):
                    coerce(bad)
            with self.subTest(coercer=name, storage="united"):
                with self.assertRaises(ValueError):
                    coerce(bad * u.mV)


class CallerAttributionTest(unittest.TestCase):
    """Messages name the entry point the user called, not the helper.

    Before these helpers were free functions, ``caller`` was baked in as
    the literal ``"Cell.vis_cv(...)"``, so a runtime ion read on a
    multi-member population reported a plotting function it had never
    gone near.
    """

    def setUp(self) -> None:
        self.cell = Cell(_soma_tree(), cv_policy=CVPerBranch(2), pop_size=4)
        self.cell.init_state()

    def test_population_refusal_quotes_the_supplied_caller(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^some caller addresses a single morphology"):
            fr.single_population_view(self.cell, self.cell.V.value, caller="some caller")

    def test_runtime_ion_inspection_does_not_mention_plotting(self) -> None:
        message = str(self._runtime_ion_error())
        self.assertIn("Runtime CV inspection", message)
        self.assertNotIn("vis", message)
        self.assertNotIn("plot", message)

    def _runtime_ion_error(self) -> Exception:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch(), pop_size=4)
        cell.init_state()
        # ``area`` is one of the geometry fields stored per population
        # member, so reading it through a CV view trips the single-
        # morphology guard. ``E`` would not — it is a bare scalar.
        with self.assertRaises(ValueError) as caught:
            cell.runtime_cvs[0].ions["na"].get("area")
        return caught.exception


class RegionAndLocsetResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())

    def test_cv_coverage_is_the_overlapped_area_fraction_of_each_cv(self) -> None:
        fractions = fr.cv_area_coverage_fractions(
            self.cell,
            BranchSlice(branch_index=1, prox=0.0, dist=0.5),
            caller="test",
        )
        # The dendrite tapers 2 um -> 1 um, so its proximal half carries
        # more than half the membrane: 0.5833... of the area against 0.5
        # of the length. The area is the number the physics uses.
        self.assertAlmostEqual(fractions[1], 0.5833333333333334)
        self.assertAlmostEqual(fractions[0], 0.0)

    def test_branch_coverage_measures_the_branch_not_its_cvs(self) -> None:
        fractions = fr.branch_coverage_fractions(
            self.cell,
            BranchSlice(branch_index=1, prox=0.25, dist=0.75),
            caller="test",
        )
        self.assertAlmostEqual(fractions[1], 0.5)

    def test_a_whole_region_covers_every_branch_fully(self) -> None:
        fractions = fr.branch_coverage_fractions(self.cell, AllRegion(), caller="test")
        self.assertEqual(set(fractions), {0, 1})
        for value in fractions.values():
            self.assertAlmostEqual(value, 1.0)

    def test_locset_resolves_to_the_owning_cv(self) -> None:
        self.assertEqual(fr.locset_cv_ids(self.cell, RootLocation(0.5), caller="test"), {0})

    def test_locset_highlights_are_full_intensity(self) -> None:
        fractions = fr.cv_highlight_fractions(
            self.cell,
            region=None,
            locset=RootLocation(0.5),
            caller="test",
        )
        self.assertEqual(fractions, {0: 1.0})

    def test_node_highlights_land_on_cv_midpoints(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        fractions = fr.node_highlight_fractions(
            cell,
            region=BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            locset=None,
            caller="test",
        )
        midpoint = int(cell.node_tree.cv_to_mid_node_id[1])
        self.assertAlmostEqual(fractions[midpoint], 1.0)

    def test_a_region_of_the_wrong_type_is_rejected_by_name(self) -> None:
        with self.assertRaisesRegex(TypeError, r"^caller X expects RegionExpr or RegionMask, got str"):
            fr.region_intervals(self.cell, "everything", caller="caller X")

    def test_a_locset_of_the_wrong_type_is_rejected_by_name(self) -> None:
        with self.assertRaisesRegex(TypeError, r"^caller X expects LocsetExpr or LocsetMask, got int"):
            fr.locset_cv_ids(self.cell, 3, caller="caller X")


class CoverageMatchesThePaintedFractionTest(unittest.TestCase):
    """CV coverage must report the fraction the physics actually applies.

    ``Cell.paint`` scales a density mechanism by the fraction of the CV's
    *lateral membrane area* a region covers, and stores it as
    ``Density.coverage_area_fraction``. ``Cell.on(region).cv.coverage_fraction``
    is the only way a user can read that number back, so the two must agree.

    They only diverge on a tapering branch: with constant radius the area
    fraction and the length fraction are the same number, which is why this
    went unnoticed. Both cases are pinned below.
    """

    #: ``BranchSlice(1, 0.0, 0.5)`` -- the proximal half of the dendrite.
    REGION = BranchSlice(branch_index=1, prox=0.0, dist=0.5)

    @staticmethod
    def _dendrite_cell(r_prox: float, r_dist: float) -> Cell:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[r_prox, r_dist] * u.um,
            type="basal_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend
        return Cell(tree, cv_policy=CVPerBranch(1))

    def _painted_fraction(self, cell: Cell) -> float:
        """Return the coverage the discretization baked into the dendrite CV."""
        cell.paint(self.REGION, Channel("IL", g_max=1.0 * (u.mS / u.cm**2), E=-70.0 * u.mV))
        painted = [mech.coverage_area_fraction for cv in cell.cvs for mech in cv.density_mech]
        self.assertEqual(len(painted), 1, "expected exactly one painted CV")
        return float(painted[0])

    def test_a_tapering_branch_reports_the_area_fraction_not_the_length_fraction(self) -> None:
        reported = self._dendrite_cell(4.0, 1.0).on(self.REGION).cv.coverage_fraction
        painted = self._painted_fraction(self._dendrite_cell(4.0, 1.0))
        # The length fraction here is 0.5; the area fraction is 0.65.
        self.assertAlmostEqual(painted, 0.65)
        self.assertEqual(reported.shape, (1,))
        self.assertAlmostEqual(float(reported[0]), painted)

    def test_an_untapered_branch_agrees_with_the_length_fraction(self) -> None:
        reported = self._dendrite_cell(2.0, 2.0).on(self.REGION).cv.coverage_fraction
        painted = self._painted_fraction(self._dendrite_cell(2.0, 2.0))
        self.assertAlmostEqual(painted, 0.5)
        self.assertAlmostEqual(float(reported[0]), 0.5)


class SpatialMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        self.cell.init_state()

    def test_cv_values_scatter_onto_midpoints_and_leave_the_rest_nan(self) -> None:
        mapped = fr.cv_to_node_values(self.cell, np.asarray([1.0, 2.0]))
        midpoints = np.asarray(self.cell.node_tree.cv_to_mid_node_id, dtype=int)
        self.assertEqual(mapped.shape, (self.cell.n_point,))
        np.testing.assert_allclose(mapped[midpoints], [1.0, 2.0])
        off_midpoint = np.setdiff1d(np.arange(self.cell.n_point), midpoints)
        self.assertTrue(np.all(np.isnan(mapped[off_midpoint])))

    def test_cv_values_keep_their_unit(self) -> None:
        mapped = fr.cv_to_node_values(self.cell, np.asarray([1.0, 2.0]) * u.mV)
        self.assertTrue(u.get_unit(mapped).has_same_dim(u.mV))

    def test_a_wrong_length_cv_vector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"cv_to_node_values expects shape"):
            fr.cv_to_node_values(self.cell, np.zeros(self.cell.n_cv + 1))

    def test_masking_blanks_every_non_midpoint(self) -> None:
        masked = fr.mask_non_midpoint_points(self.cell, np.ones(self.cell.n_point))
        midpoints = np.asarray(self.cell.node_tree.cv_to_mid_node_id, dtype=int)
        np.testing.assert_allclose(masked[midpoints], 1.0)
        self.assertEqual(int(np.count_nonzero(~np.isnan(masked))), len(set(midpoints.tolist())))

    def test_a_wrong_length_point_vector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"mask_non_midpoint_points expects shape"):
            fr.mask_non_midpoint_points(self.cell, np.zeros(self.cell.n_point + 1))


class LayoutLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        self.cell.init_state()

    def test_a_missing_layout_kind_is_reported_against_the_caller(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^caller Y found no runtime layout with kind"):
            fr.unique_layout_by_kind(self.cell, "channel:NotThere", caller="caller Y")

    def test_an_unknown_layout_id_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            fr.layout_field_to_cv_values(self.cell, 9999, "g_max", caller="test")


class RequireInitializedTest(unittest.TestCase):
    def test_a_declaration_only_cell_is_refused(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with self.assertRaisesRegex(RuntimeError, r"drawing requires init_state\(\) first"):
            fr.require_initialized(cell, "drawing")

    def test_an_initialized_cell_passes(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        self.assertIsNone(fr.require_initialized(cell, "drawing"))


if __name__ == "__main__":
    unittest.main()
