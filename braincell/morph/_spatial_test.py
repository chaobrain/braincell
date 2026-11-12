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
import warnings

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.morph._spatial import MorphologySpatialGeometry, _BranchGeometryUm, interpolate_branch


def _branch(length: float, branch_type: str = "dendrite") -> Branch:
    return Branch.from_lengths(
        lengths=[length] * u.um,
        radii=[2.0, 1.0] * u.um,
        type=branch_type,
    )


class MorphologySpatialGeometryTest(unittest.TestCase):
    def test_no_soma_uses_whole_root_branch_as_reference(self) -> None:
        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        morpho.trunk.attach(_branch(30.0), name="child", parent_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        np.testing.assert_array_equal(
            geometry.path_distance_from_soma(0, np.asarray([0.0, 0.4, 1.0])).to_decimal(u.um),
            [0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(geometry.path_distance_from_soma(1, 0.5).to_decimal(u.um), 15.0)
        self.assertAlmostEqual(geometry.path_distance_to_root(1, 0.5).to_decimal(u.um), 35.0)

    def test_all_soma_branches_form_one_zero_distance_region(self) -> None:
        morpho = Morphology.from_root(_branch(10.0, "soma"), name="soma_0")
        soma_1 = morpho.soma_0.attach(_branch(8.0, "soma"), name="soma_1", parent_x=1.0)
        soma_1.attach(_branch(40.0), name="dend", parent_x=0.5, child_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        self.assertEqual(geometry.path_distance_from_soma(0, 0.3).to_decimal(u.um), 0.0)
        self.assertEqual(geometry.path_distance_from_soma(1, 0.7).to_decimal(u.um), 0.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(2, 0.75).to_decimal(u.um), 10.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(2, 0.0).to_decimal(u.um), 40.0)

    def test_of_builds_once_per_revision(self) -> None:
        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        first = MorphologySpatialGeometry.of(morpho)
        self.assertIs(MorphologySpatialGeometry.of(morpho), first)

        morpho.trunk.attach(_branch(30.0), name="child", parent_x=1.0)
        edited = MorphologySpatialGeometry.of(morpho)
        self.assertIsNot(edited, first)
        self.assertIs(MorphologySpatialGeometry.of(morpho), edited)
        self.assertAlmostEqual(edited.path_distance_to_root(1, 0.5).to_decimal(u.um), 35.0)
        self.assertEqual(edited.branch_lengths_um.tolist(), MorphologySpatialGeometry.build(morpho).branch_lengths_um.tolist())

    def test_a_copy_reads_its_own_geometry(self) -> None:
        import copy

        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        morpho.trunk.attach(_branch(30.0), name="child", parent_x=1.0)
        original = MorphologySpatialGeometry.of(morpho)
        for duplicate in (copy.deepcopy(morpho), morpho.clone()):
            with self.subTest(type(duplicate).__name__):
                geometry = MorphologySpatialGeometry.of(duplicate)
                self.assertIsNot(geometry, original)
                self.assertIs(geometry.morpho, duplicate)
                self.assertAlmostEqual(geometry.path_distance_to_root(1, 0.5).to_decimal(u.um), 35.0)
        self.assertIs(MorphologySpatialGeometry.of(morpho), original)

    def test_distance_is_shortest_path_even_when_soma_is_below_root(self) -> None:
        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        morpho.trunk.attach(_branch(10.0, "soma"), name="soma", parent_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        self.assertAlmostEqual(geometry.path_distance_from_soma(0, 0.0).to_decimal(u.um), 20.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(0, 1.0).to_decimal(u.um), 0.0)

    def test_interpolation_returns_continuous_radius_and_position(self) -> None:
        root = Branch.from_points(
            points=np.asarray([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]]) * u.um,
            radii=np.asarray([3.0, 1.0]) * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(root, name="trunk")

        radius, position = interpolate_branch(morpho, 0, 0.25)
        self.assertAlmostEqual(radius.to_decimal(u.um), 2.5)
        assert position is not None
        np.testing.assert_allclose(position.to_decimal(u.um), [2.5, 1.0, 0.5])


def _reference_at(geometry: _BranchGeometryUm, x: float):
    """``_BranchGeometryUm.at`` before CELL-18c: walk the segments for one site."""
    lengths = geometry.lengths
    r0, r1, p0, p1 = geometry.r0, geometry.r1, geometry.p0, geometry.p1
    total = geometry.total

    cursor = 0.0
    last_positive = 0
    for index, segment_length in enumerate(lengths.tolist()):
        if segment_length <= 0.0:
            if np.isclose(x, cursor / total):
                point = None if p0 is None or p1 is None else 0.5 * (p0[index] + p1[index])
                return 0.5 * (r0[index] + r1[index]), point
            continue
        start = cursor / total
        cursor += segment_length
        end = cursor / total
        last_positive = index
        if x < end or (x == 1.0 and end == 1.0):
            fraction = (x - start) / (end - start)
            point = None if p0 is None or p1 is None else p0[index] + fraction * (p1[index] - p0[index])
            return r0[index] + fraction * (r1[index] - r0[index]), point
    point = None if p1 is None else p1[last_positive]
    return r1[last_positive], point


class InterpolateBranchTest(unittest.TestCase):
    """CELL-18c: every site of a branch at once, value for value as one site at a time."""

    def _assert_same_sites(self, branch, xs) -> None:
        morpho = Morphology.from_root(branch, name="trunk")
        geometry = _BranchGeometryUm(branch)
        xs = np.asarray(xs, dtype=float)
        radius, point = interpolate_branch(morpho, 0, xs)
        expected = [_reference_at(geometry, x) for x in xs.reshape(-1).tolist()]
        np.testing.assert_array_equal(
            radius.to_decimal(u.um), np.asarray([value for value, _ in expected], dtype=float).reshape(xs.shape)
        )
        if geometry.p0 is None:
            self.assertIsNone(point)
        else:
            np.testing.assert_array_equal(
                point.to_decimal(u.um),
                np.asarray([value for _, value in expected], dtype=float).reshape(xs.shape + (3,)),
            )

    @staticmethod
    def _sites(lengths) -> np.ndarray:
        # Every segment end, each nudged by a few ulp, isclose's reach, and a spread of sites.
        cursor = np.cumsum(np.asarray(lengths, dtype=float))
        ends = np.concatenate([[0.0], cursor / float(np.sum(lengths)), [1.0]])
        nudged = [np.nextafter(ends, 2.0), np.nextafter(ends, -1.0), ends + 1e-8, ends - 1e-8, ends + 2e-8]
        return np.clip(np.concatenate([ends, *nudged, np.linspace(0.0, 1.0, 37)]), 0.0, 1.0)

    def test_sites_match_one_walk_per_site(self) -> None:
        rng = np.random.default_rng(18)
        cases = {
            "one segment": [20.0],
            "uneven": [0.1, 0.2, 0.3, 0.7, 1.1],
            "zero-length first": [0.0, 10.0, 5.0],
            "zero-length middle": [30.0, 0.0, 45.0],
            "zero-length last": [3.0, 4.0, 0.0],
            "zero-length runs": [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
            "many": rng.uniform(0.01, 3.0, 300).round(3).tolist(),
        }
        for name, lengths in cases.items():
            lengths = np.asarray(lengths, dtype=float)
            radii = np.linspace(2.0, 0.5, lengths.size + 1)
            steps = np.cumsum(np.concatenate([[0.0], lengths]))
            xyz = np.stack([steps, 0.3 * steps, -0.1 * steps], axis=1)
            sites = self._sites(lengths)
            for kind in ("lengths", "points"):
                with self.subTest(name, kind=kind), warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if kind == "lengths":
                        branch = Branch.from_lengths(lengths=lengths * u.um, radii=radii * u.um, type="dendrite")
                    else:
                        branch = Branch.from_points(points=xyz * u.um, radii=radii * u.um, type="dendrite")
                    self._assert_same_sites(branch, sites)
                    self._assert_same_sites(branch, sites[: sites.size // 2 * 2].reshape(-1, 2)[:, ::-1])
                    self._assert_same_sites(branch, sites[:0])

    def test_reconstructed_branches_match_one_walk_per_site(self) -> None:
        from braincell.io._testing import FIXTURE_DIR

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            morpho = Morphology.from_swc(FIXTURE_DIR / "CA1.swc")
        for index in range(len(morpho.branches)):
            branch = morpho.branch(index=index).branch
            lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
            self._assert_same_sites(branch, self._sites(lengths))


if __name__ == "__main__":
    unittest.main()
