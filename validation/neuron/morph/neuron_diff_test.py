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

from braincell.io._testing import CEREBELLUM_FIXTURES

import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import neuron
from neuron import h

from neuron_diff import (
    check_soma_orientation,
    compare_asc_with_neuron,
    compare_morphology_with_neuron,
    compare_swc_with_neuron,
    compute_braincell_metrics,
    compute_neuron_metrics,
    load_asc_morphology,
    load_swc_morphology,
    supported_metric_names,
)


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "braincell").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Could not locate the braincell repository root.")


def _assert_zero_diff(testcase: unittest.TestCase, comparison: dict[str, object], metric_name: str) -> None:
    diff = comparison["diff"][metric_name]
    testcase.assertTrue(diff["available"], metric_name)
    testcase.assertLessEqual(diff["abs_diff"], 1e-4, metric_name)
    if diff["rel_diff"] is not None:
        testcase.assertLessEqual(diff["rel_diff"], 1e-6, metric_name)


def _assert_same_soma_orientation(testcase: unittest.TestCase, comparison: dict[str, object]) -> None:
    # Distances are measured from the root's first point on both sides, so check
    # that the two somata start at the same end before comparing any metric.
    record = comparison["soma_start_distance"]
    testcase.assertFalse(record["orientation_differs"], record["reason"])
    testcase.assertLessEqual(record["value"], 1e-4)


class NeuronDiffTest(unittest.TestCase):
    def _write_swc(self, body: str, *, filename: str = "sample.swc") -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / filename
        path.write_text(textwrap.dedent(body).strip() + "\n")
        return path

    def _write_asc(self, body: str, *, filename: str = "sample.asc") -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / filename
        path.write_text(textwrap.dedent(body).strip() + "\n")
        return path

    def test_load_swc_morphology_instantiates_sections(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 0 10 0 2 1
            3 3 0 20 0 1 2
            4 2 10 0 0 1 1
            5 2 20 0 0 0.5 4
            """
        )

        sections = load_swc_morphology(path)

        self.assertEqual(len(sections), 3)
        self.assertEqual(tuple(sec.name() for sec in sections), ("soma[0]", "axon[0]", "dend[0]"))

    def test_compare_swc_with_neuron_matches_simple_tree(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 0 10 0 2 1
            3 3 0 20 0 1 2
            4 2 10 0 0 1 1
            5 2 20 0 0 0.5 4
            """
        )

        comparison = compare_swc_with_neuron(path)

        self.assertEqual(comparison["soma_start_distance"]["value"], 0.0)
        self.assertIsNone(comparison["soma_start_distance"]["orientation_differs"])
        self.assertEqual(
            comparison["selected_metrics"],
            (
                "total_length",
                "total_area",
                "total_volume",
                "mean_radius",
                "n_branches",
                "n_stems",
                "n_bifurcations",
                "max_branch_order",
                "max_path_distance",
                "max_euclidean_distance",
            ),
        )
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_swc_with_neuron_matches_branching_tree(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 8 -1
            2 3 0 10 0 2 1
            3 3 0 20 0 1.5 2
            4 3 -10 30 0 1.2 3
            5 3 -20 40 0 1.0 4
            6 3 10 30 0 1.1 3
            7 3 20 40 0 0.9 6
            """
        )

        comparison = compare_swc_with_neuron(path)

        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)
        self.assertEqual(comparison["braincell"]["n_branches"]["value"], 4)
        self.assertEqual(comparison["braincell"]["n_bifurcations"]["value"], 1)
        self.assertEqual(comparison["braincell"]["max_branch_order"]["value"], 2)

    def test_compute_metrics_match_the_comparison(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 8 -1
            2 3 0 10 0 2 1
            3 3 10 20 0 1.5 2
            """
        )

        comparison = compare_swc_with_neuron(path)

        self.assertEqual(compute_braincell_metrics(path), comparison["braincell"])
        self.assertEqual(compute_neuron_metrics(load_swc_morphology(path)), comparison["neuron"])

    def test_compare_swc_with_neuron_reports_bent_tree_distance_metrics(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 8 -1
            2 3 0 10 0 2 1
            3 3 10 20 0 1.5 2
            """
        )

        comparison = compare_swc_with_neuron(path)

        _assert_zero_diff(self, comparison, "max_path_distance")
        _assert_zero_diff(self, comparison, "max_euclidean_distance")
        self.assertAlmostEqual(comparison["braincell"]["max_path_distance"]["value"], 8.0 + math.sqrt(200.0))
        self.assertAlmostEqual(comparison["neuron"]["max_path_distance"]["value"], 8.0 + math.sqrt(200.0))
        self.assertAlmostEqual(comparison["braincell"]["max_euclidean_distance"]["value"], math.sqrt(724.0))
        self.assertAlmostEqual(comparison["neuron"]["max_euclidean_distance"]["value"], math.sqrt(724.0))

    def test_load_asc_morphology_instantiates_sections(self) -> None:
        path = self._write_asc(
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            """
        )

        sections = load_asc_morphology(path)

        self.assertEqual(len(sections), 2)
        self.assertEqual(tuple(sec.name() for sec in sections), ("soma[0]", "dend[0]"))

    def test_compare_asc_with_neuron_matches_simple_tree(self) -> None:
        path = self._write_asc(
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            """
        )

        comparison = compare_asc_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_asc_with_neuron_matches_branching_tree(self) -> None:
        path = self._write_asc(
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (
                (0 10 0 1)
              |
                (5 5 0 1)
                (10 5 0 1)
              )
            )
            """
        )

        comparison = compare_asc_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_asc_with_neuron_matches_same_xyz_diameter_step_tree(self) -> None:
        path = self._write_asc(
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 2)
              (0 5 0 2)
              (
                (0 5 0 1)
                (0 10 0 1)
              )
            )
            """
        )

        comparison = compare_asc_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_asc_with_neuron_matches_spine_demo(self) -> None:
        path = self._write_asc(
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 4 "none")
                (Color Red)
                (Generated 0)
                (1 1 0 0.1)
              >
              (0 5 0 1)
            )
            """
        )

        comparison = compare_asc_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_morphology_with_neuron_supports_real_asc_fixture(self) -> None:
        path = _repo_root() / "data" / "morphology" / "goc.asc"

        comparison = compare_morphology_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        self.assertEqual(comparison["path"], str(path))
        self.assertIn("n_branches", comparison["diff"])
        self.assertTrue(comparison["diff"]["n_branches"]["available"])

    def test_compare_morphology_with_neuron_supports_real_sc_asc_fixture(self) -> None:
        path = CEREBELLUM_FIXTURES["SC"]

        comparison = compare_morphology_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            _assert_zero_diff(self, comparison, metric_name)

    def test_compare_morphology_with_neuron_supports_real_bc_asc_fixture(self) -> None:
        path = CEREBELLUM_FIXTURES["BC"]

        comparison = compare_morphology_with_neuron(path)

        self.assertEqual(comparison["kind"], "asc")
        _assert_same_soma_orientation(self, comparison)
        for metric_name in comparison["selected_metrics"]:
            diff = comparison["diff"][metric_name]
            self.assertTrue(diff["available"], metric_name)
            if diff["rel_diff"] is not None:
                self.assertLessEqual(diff["rel_diff"], 1e-6, metric_name)
            else:
                self.assertLessEqual(diff["abs_diff"], 1e-4, metric_name)

    def test_check_soma_orientation_reports_reversed_soma(self) -> None:
        # NEURON 8.2 laid the SC soma out from its far end; reversing NEURON's own
        # soma points reproduces that comparison without an older NEURON.
        sections = load_asc_morphology(CEREBELLUM_FIXTURES["SC"])
        soma = sections[0]
        self.assertEqual(soma.name(), "soma[0]")
        points = [
            (h.x3d(i, sec=soma), h.y3d(i, sec=soma), h.z3d(i, sec=soma))
            for i in range(int(h.n3d(sec=soma)))
        ]
        self.assertEqual(len(points), 21)

        same = check_soma_orientation(points, points)
        reversed_soma = check_soma_orientation(points, points[::-1])

        self.assertFalse(same["orientation_differs"])
        self.assertIsNone(same["reason"])
        self.assertEqual(same["value"], 0.0)
        self.assertTrue(reversed_soma["orientation_differs"])
        # The reversed start point is the far end of the straight soma.
        self.assertAlmostEqual(reversed_soma["value"], float(soma.L), delta=1e-5)
        self.assertEqual(reversed_soma["neuron_point"], points[0])
        self.assertEqual(reversed_soma["braincell_point"], points[-1])
        self.assertEqual(reversed_soma["neuron_version"], neuron.__version__)
        self.assertIn("soma orientation differs", reversed_soma["reason"])
        self.assertIn(f"NEURON {neuron.__version__}", reversed_soma["reason"])
        self.assertIn(f"{points[0][0]:.6f}", reversed_soma["reason"])
        self.assertIn(f"{points[-1][0]:.6f}", reversed_soma["reason"])

    def test_supported_metric_names_no_longer_exposes_n_3d_points(self) -> None:
        metric_names = supported_metric_names()
        self.assertEqual(metric_names, supported_metric_names(include_optional=True))
        self.assertNotIn("n_3d_points", metric_names)
        self.assertIn("max_euclidean_distance", metric_names)

    def test_compare_swc_with_neuron_rejects_non_swc_files(self) -> None:
        path = self._write_swc("1 1 0 0 0 1 -1", filename="sample.asc")

        with self.assertRaisesRegex(ValueError, "only supports \\.swc files"):
            compare_swc_with_neuron(path)

    def test_compare_asc_with_neuron_rejects_non_asc_files(self) -> None:
        path = self._write_swc("1 1 0 0 0 1 -1", filename="sample.swc")

        with self.assertRaisesRegex(ValueError, "only supports \\.asc files"):
            compare_asc_with_neuron(path)


if __name__ == "__main__":
    unittest.main()
