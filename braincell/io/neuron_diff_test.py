from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from braincell.io import compare_swc_with_neuron, load_swc_morphology, supported_metric_names


def _assert_zero_diff(testcase: unittest.TestCase, comparison: dict[str, object], metric_name: str) -> None:
    diff = comparison["diff"][metric_name]
    testcase.assertTrue(diff["available"], metric_name)
    testcase.assertLessEqual(diff["abs_diff"], 1e-4, metric_name)
    if diff["rel_diff"] is not None:
        testcase.assertLessEqual(diff["rel_diff"], 1e-6, metric_name)


class NeuronDiffTest(unittest.TestCase):
    def _write_swc(self, body: str, *, filename: str = "sample.swc") -> Path:
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

    def test_supported_metric_names_no_longer_exposes_n_3d_points(self) -> None:
        metric_names = supported_metric_names()
        self.assertEqual(metric_names, supported_metric_names(include_optional=True))
        self.assertNotIn("n_3d_points", metric_names)

    def test_compare_swc_with_neuron_rejects_non_swc_files(self) -> None:
        path = self._write_swc("1 1 0 0 0 1 -1", filename="sample.asc")

        with self.assertRaisesRegex(ValueError, "only supports \\.swc files"):
            compare_swc_with_neuron(path)


if __name__ == "__main__":
    unittest.main()
