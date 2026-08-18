"""Tests for the channel-layout cost benchmark helper."""

import unittest

import channel_layout_benchmark as benchmark


class ChannelLayoutFixtureTest(unittest.TestCase):
    def test_mixed_fixture_has_expected_topology_and_active_counts(self) -> None:
        cell = benchmark.build_cell(pop_size=1, profile="mixed")
        rows = benchmark.channel_layout_rows(cell)

        self.assertEqual(cell.n_cv, 100)
        self.assertEqual(cell.n_point, 105)
        self.assertEqual([row["name"] for row in rows], ["na", "k", "leak"])
        self.assertEqual([row["n_active"] for row in rows], [4, 16, 32])
        self.assertEqual([row["node_shape"] for row in rows], [(1, 105)] * 3)

    def test_population_scales_arrays_but_not_layout_masks(self) -> None:
        one = benchmark.channel_layout_rows(benchmark.build_cell(pop_size=1, profile="mixed"))
        ten = benchmark.channel_layout_rows(benchmark.build_cell(pop_size=10, profile="mixed"))

        for one_row, ten_row in zip(one, ten):
            self.assertEqual(ten_row["declaration_bytes"], 10 * one_row["declaration_bytes"])
            self.assertEqual(ten_row["runtime_parameter_bytes"], 10 * one_row["runtime_parameter_bytes"])
            self.assertEqual(ten_row["gate_value_bytes"], 10 * one_row["gate_value_bytes"])
            self.assertEqual(ten_row["mask_bytes"], one_row["mask_bytes"])
            self.assertEqual(ten_row["packed_index_bytes"], one_row["packed_index_bytes"])

    def test_projected_storage_tracks_coverage(self) -> None:
        totals = []
        expected_active = {
            "soma": [4, 4, 4],
            "half_dend": [16, 16, 16],
            "one_dend": [32, 32, 32],
            "global": [100, 100, 100],
        }
        for profile in expected_active:
            cell = benchmark.build_cell(pop_size=1, profile=profile)
            summary = benchmark.memory_summary(cell)
            self.assertEqual(
                [row["n_active"] for row in summary["rows"]],
                expected_active[profile],
            )
            totals.append(summary["projected_packed_total_bytes"])
            self.assertLessEqual(summary["projected_hybrid_total_bytes"], summary["dense_total_bytes"])
        self.assertEqual(totals, sorted(totals))
        self.assertLess(totals[0], benchmark.memory_summary(benchmark.build_cell(pop_size=1))["dense_total_bytes"])
        self.assertGreater(totals[-1], benchmark.memory_summary(benchmark.build_cell(pop_size=1))["dense_total_bytes"])


class ChannelLayoutMicroKernelTest(unittest.TestCase):
    def test_dense_and_packed_microkernels_match(self) -> None:
        parity = benchmark.micro_parity(pop_size=2, profile="mixed", steps=3)
        self.assertLess(parity["max_gate_error"], 1e-6)
        self.assertLess(parity["max_current_error"], 1e-3)

    def test_environment_and_summary_are_json_serializable(self) -> None:
        import json

        cell = benchmark.build_cell(pop_size=1, profile="mixed")
        payload = {
            "environment": benchmark.environment_summary(),
            "memory": benchmark.memory_summary(cell),
        }
        encoded = json.dumps(payload)
        self.assertIn('"backend": "cpu"', encoded)
        self.assertIn('"dense_total_bytes"', encoded)


if __name__ == "__main__":
    unittest.main()
