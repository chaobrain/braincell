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
import numpy as np
from scipy.special import ndtr, ndtri

import braincell
from braincell import Branch, Morphology
from braincell.filter._testing import make_soma
from braincell.filter import AllRegion, BranchSlice, RandomSamples, SamplingContext, density, metric, sample
from braincell.filter._sampling import (
    _AtomComponent,
    _ContinuousComponent,
    _analytic_mass,
    _build_components,
    _inverse_linear,
)


def _single_branch(
    *,
    lengths: list[float],
    radii: list[float],
    points: np.ndarray | None = None,
) -> Morphology:
    if points is None:
        branch = Branch.from_lengths(
            lengths=np.asarray(lengths) * u.um,
            radii=np.asarray(radii) * u.um,
            type="dendrite",
        )
    else:
        branch = Branch.from_points(
            points=points * u.um,
            radii=np.asarray(radii) * u.um,
            type="dendrite",
        )
    return Morphology.from_root(branch, name="dend")


def _uniforms(seed: int, number: int) -> np.ndarray:
    return np.asarray(brainstate.random.RandomState(seed).random(number), dtype=float)


class AnalyticMeasureTest(unittest.TestCase):
    def test_normalized_and_length_weight_branches_differently(self) -> None:
        root = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
        child = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
        morpho = Morphology.from_root(root, name="trunk")
        morpho.trunk.child = child
        region = AllRegion()

        normalized = sample(region, number=20_000, seed=7, measure="normalized").evaluate(morpho)
        by_length = sample(region, number=20_000, seed=7, measure="length").evaluate(morpho)

        self.assertAlmostEqual(np.mean(normalized.branch_id == 1), 0.5, delta=0.02)
        self.assertAlmostEqual(np.mean(by_length.branch_id == 1), 0.75, delta=0.02)

    def test_segment_splitting_does_not_change_constant_radius_samples(self) -> None:
        one = _single_branch(lengths=[100.0], radii=[2.0, 2.0])
        split = _single_branch(lengths=[40.0, 60.0], radii=[2.0, 2.0, 2.0])
        region = AllRegion()
        for measure in ("normalized", "length", "lateral_area", "area"):
            first = sample(region, number=100, seed=8, measure=measure).evaluate(one)
            second = sample(region, number=100, seed=8, measure=measure).evaluate(split)
            np.testing.assert_allclose(first.branch_x, second.branch_x, atol=2e-12, rtol=0.0)

    def test_frustum_area_mass_cdf_and_inverse_are_exact(self) -> None:
        morpho = _single_branch(lengths=[100.0], radii=[10.0, 20.0])
        components = _build_components(morpho, ((0, 0.0, 1.0),), measure="lateral_area")
        self.assertEqual(len(components), 1)
        component = components[0]
        self.assertIsInstance(component, _ContinuousComponent)
        assert isinstance(component, _ContinuousComponent)

        mass, q0, q1 = _analytic_mass(component, "lateral_area")
        slant = np.sqrt(100.0**2 + 10.0**2)
        self.assertAlmostEqual(mass, np.pi * (10.0 + 20.0) * slant)
        for probability in (0.0, 0.1, 0.5, 0.9, 1.0):
            x = _inverse_linear(component, probability, q0, q1)
            expected = (-10.0 + np.sqrt(100.0 + 300.0 * probability)) / 10.0
            self.assertAlmostEqual(x, expected, places=13)

    def test_area_includes_radius_jump_atom_and_lateral_area_excludes_it(self) -> None:
        morpho = _single_branch(lengths=[10.0, 0.0, 10.0], radii=[1.0, 1.0, 2.0, 2.0])
        area = _build_components(morpho, ((0, 0.0, 1.0),), measure="area")
        lateral = _build_components(morpho, ((0, 0.0, 1.0),), measure="lateral_area")
        atoms = [component for component in area if isinstance(component, _AtomComponent)]

        self.assertEqual(len(atoms), 1)
        self.assertAlmostEqual(atoms[0].x, 0.5)
        self.assertAlmostEqual(atoms[0].radius_um, 1.5)
        self.assertAlmostEqual(atoms[0].area_um2, 3.0 * np.pi)
        self.assertEqual(sum(isinstance(component, _AtomComponent) for component in lateral), 0)
        continuous_mass = sum(
            _analytic_mass(component, "area")[0] for component in area if isinstance(component, _ContinuousComponent)
        )
        self.assertAlmostEqual(continuous_mass, 60.0 * np.pi)
        self.assertAlmostEqual(atoms[0].area_um2 / (continuous_mass + atoms[0].area_um2), 3.0 / 63.0)

    def test_area_atoms_use_half_open_region_ownership(self) -> None:
        morpho = _single_branch(lengths=[10.0, 0.0, 10.0], radii=[1.0, 1.0, 2.0, 2.0])
        left = _build_components(morpho, ((0, 0.0, 0.5),), measure="area")
        right = _build_components(morpho, ((0, 0.5, 1.0),), measure="area")
        self.assertFalse(any(isinstance(component, _AtomComponent) for component in left))
        self.assertTrue(any(isinstance(component, _AtomComponent) for component in right))

    def test_terminal_area_atom_at_x_one_is_included(self) -> None:
        morpho = _single_branch(lengths=[10.0, 0.0], radii=[1.0, 1.0, 2.0])
        components = _build_components(morpho, ((0, 0.0, 1.0),), measure="area")
        atoms = [component for component in components if isinstance(component, _AtomComponent)]
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].x, 1.0)


class DensityInversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.morpho = _single_branch(lengths=[100.0], radii=[1.0, 1.0])
        self.region = AllRegion()
        self.seed = 13
        self.number = 128
        self.uniforms = _uniforms(self.seed, self.number)

    def _x(self, preference: object) -> np.ndarray:
        return (
            sample(
                self.region,
                number=self.number,
                seed=self.seed,
                measure="normalized",
                density=preference,
                u_resolution=1e-10,
            )
            .evaluate(self.morpho)
            .branch_x
        )

    def test_constant_density_matches_uniform_cdf(self) -> None:
        np.testing.assert_allclose(self._x(lambda context: 3.0), self.uniforms, atol=2e-9, rtol=0.0)

    def test_linear_density_matches_analytic_inverse(self) -> None:
        actual = self._x(lambda context: 2.0 * context.branch_x)
        np.testing.assert_allclose(actual, np.sqrt(self.uniforms), atol=2e-9, rtol=0.0)

    def test_composable_metric_density_matches_analytic_inverse(self) -> None:
        actual = self._x(lambda context: np.exp(metric.branch_x(context) / 0.2))
        expected = np.log1p(self.uniforms * np.expm1(5.0)) / 5.0
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0.0)

    def test_gaussian_helper_matches_truncated_gaussian_inverse(self) -> None:
        center = 0.4
        sigma = 0.15
        with self.assertWarns(DeprecationWarning):
            preference = density.gaussian("branch_x", center, sigma)
        actual = self._x(preference)
        low = ndtr((0.0 - center) / sigma)
        high = ndtr((1.0 - center) / sigma)
        expected = center + sigma * ndtri(low + self.uniforms * (high - low))
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0.0)

    def test_custom_density_is_invariant_to_source_segment_splitting(self) -> None:
        split = _single_branch(lengths=[40.0, 60.0], radii=[1.0, 1.0, 1.0])
        preference = lambda context: 1.0 + context.branch_x
        first = self._x(preference)
        second = (
            sample(
                AllRegion(),
                number=self.number,
                seed=self.seed,
                measure="normalized",
                density=preference,
            )
            .evaluate(split)
            .branch_x
        )
        np.testing.assert_allclose(first, second, atol=3e-9, rtol=0.0)

    def test_narrow_gaussian_remains_finite(self) -> None:
        with self.assertWarns(DeprecationWarning):
            preference = density.gaussian("branch_x", 0.5, 0.001)
        actual = self._x(preference)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertLess(np.max(np.abs(actual - 0.5)), 0.01)


class SamplingContextAndValidationTest(unittest.TestCase):
    def test_context_exposes_geometry_and_spatial_position(self) -> None:
        morpho = _single_branch(
            lengths=[10.0],
            radii=[2.0, 4.0],
            points=np.asarray([[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]),
        )
        seen: list[SamplingContext] = []

        def preference(context: SamplingContext) -> float:
            if not seen:
                seen.append(context)
            return float(np.asarray(context.position.to_decimal(u.um))[0]) + 1.0

        sample(AllRegion(), number=2, seed=2, density=preference).evaluate(morpho)
        context = seen[0]
        self.assertEqual(context.branch_id, 0)
        self.assertEqual(context.branch_name, "dend")
        self.assertEqual(context.branch_type, "dendrite")
        self.assertIsNotNone(context.local_position)
        self.assertIs(context.position, context.local_position)

    def test_spatial_density_requires_point_geometry(self) -> None:
        morpho = _single_branch(lengths=[10.0], radii=[1.0, 1.0])
        preference = density.spatial_gaussian(np.zeros(3) * u.um, 2.0 * u.um)
        with self.assertRaisesRegex(ValueError, "full 3-D point geometry"):
            sample(AllRegion(), number=1, seed=1, density=preference).evaluate(morpho)

    def test_spatial_gaussian_samples_along_point_geometry(self) -> None:
        morpho = _single_branch(
            lengths=[10.0],
            radii=[1.0, 1.0],
            points=np.asarray([[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]),
        )
        seed = 3
        number = 64
        actual = sample(
            AllRegion(),
            number=number,
            seed=seed,
            measure="normalized",
            density=density.spatial_gaussian(np.asarray([6.0, 2.0, 3.0]) * u.um, 2.0 * u.um),
        ).evaluate(morpho)
        uniforms = _uniforms(seed, number)
        low = ndtr(-2.5)
        high = ndtr(2.5)
        expected = 0.5 + 0.2 * ndtri(low + uniforms * (high - low))
        np.testing.assert_allclose(actual.branch_x, expected, atol=2e-9, rtol=0.0)

        with self.assertRaisesRegex(TypeError, "length quantity"):
            density.spatial_gaussian(np.zeros(3) * u.um, 2.0)

    def test_seed_order_duplicates_and_region_set_are_preserved(self) -> None:
        morpho = _single_branch(lengths=[10.0, 0.0, 10.0], radii=[1.0, 1.0, 1000.0, 1000.0])
        region = BranchSlice(0, 0.0, 0.6) | BranchSlice(0, 0.4, 1.0)
        first = sample(region, number=30, seed=4, measure="area").evaluate(morpho)
        second = sample(region, number=30, seed=4, measure="area").evaluate(morpho)
        self.assertEqual(first.points, second.points)
        self.assertGreater(first.points.count((0, 0.5)), 1)

        ordered = sample(AllRegion(), number=30, seed=4, measure="length").evaluate(morpho)
        other_seed = sample(AllRegion(), number=30, seed=5, measure="length").evaluate(morpho)
        self.assertNotEqual(ordered.points, tuple(sorted(ordered.points)))
        self.assertNotEqual(ordered.points, other_seed.points)

    def test_cell_placement_retains_distinct_continuous_samples_in_one_cv(self) -> None:
        morpho = _single_branch(lengths=[10.0], radii=[1.0, 1.0])
        locations = sample(AllRegion(), number=2, seed=5)
        expected_x = locations.evaluate(morpho).branch_x
        cell = braincell.Cell(morpho, cv_policy=braincell.CVPerBranch())
        cell.place(locations, braincell.mech.Synapse("ExpSyn", name="sampled", tau=2.0 * u.ms))

        placements = cell.point_placements
        np.testing.assert_allclose([placement.branch_x for placement in placements], expected_x)
        self.assertNotEqual(placements[0].branch_x, placements[1].branch_x)
        self.assertEqual(placements[0].point_id, placements[1].point_id)

    def test_invalid_inputs_and_density_results_fail_clearly(self) -> None:
        morpho = _single_branch(lengths=[10.0], radii=[1.0, 1.0])
        cases = [
            (sample(AllRegion(), number=0, seed=1), ValueError),
            (sample(AllRegion(), number=1, seed=True), TypeError),
            (sample(AllRegion(), number=1, seed=1, measure="volume"), ValueError),
            (sample(AllRegion(), number=1, seed=1, u_resolution=1e-14), ValueError),
            (sample(AllRegion(), number=1, seed=1, density=lambda context: -1.0), ValueError),
            (sample(AllRegion(), number=1, seed=1, density=lambda context: np.nan), ValueError),
            (sample(AllRegion(), number=1, seed=1, density=lambda context: np.ones(2)), ValueError),
            (sample(AllRegion(), number=1, seed=1, density=lambda context: 1.0 * u.um), TypeError),
            (sample(AllRegion(), number=1, seed=1, density=lambda context: 0.0), ValueError),
        ]
        for expression, error in cases:
            with self.subTest(expression=expression), self.assertRaises(error):
                expression.evaluate(morpho)

    def test_legacy_random_samples_keep_their_exact_numpy_sequence(self) -> None:
        root = make_soma(length=10.0)
        child = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="dendrite")
        morpho = Morphology.from_root(root, name="soma")
        morpho.soma.dend = child
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        actual = RandomSamples(region, 6, 123).evaluate(morpho).points
        self.assertEqual(
            actual,
            (
                (1, 0.923344998027),
                (0, 0.276574397797),
                (0, 0.819754561593),
                (0, 0.889892693111),
                (0, 0.51297045523),
                (1, 0.244964601069),
            ),
        )


if __name__ == "__main__":
    unittest.main()
