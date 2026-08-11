from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from examples.profiling.cases.cerebellar_probability_network import create_workload
from examples.profiling.profile_simulation import _parse_args, _profile_options, main


class TestProfileSimulation(unittest.TestCase):
    """Tests for developer profiling helpers."""

    def test_neuron_compare_cell_profile_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "profile_cell.json"
            main(
                [
                    "--case",
                    "neuron_compare_cell",
                    "--cell",
                    "grc_ma2020",
                    "--duration-ms",
                    "0.1",
                    "--dt-ms",
                    "0.1",
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                    "--out",
                    str(out),
                ]
            )
            payload = json.loads(out.read_text())

        phases = [row["phase"] for row in payload["phases"]]
        self.assertEqual(
            phases,
            [
                "import_case",
                "load_params",
                "import_model",
                "model_build",
                "place_probes",
                "init_reset",
                "steady_run",
                "materialize",
            ],
        )
        self.assertEqual(payload["metadata"]["cell"], "grc_ma2020")
        self.assertTrue(all(row["wall_time_s"] >= 0.0 for row in payload["phases"]))
        self.assertEqual(payload["materialized"]["voltage_shape"][0], 1)

    def test_cerebellar_probability_network_workload_metadata_subset(self) -> None:
        args = argparse.Namespace(
            scale="tiny",
            populations="GrC,GoC",
            precision=32,
            event_backend="auto",
            brainevent_backend="jax_raw",
            spike_recording="population",
            grc_size=None,
            goc_size=None,
            pc_size=None,
            sc_size=None,
            bc_size=None,
            dcn_size=None,
            io_size=None,
            dt_ms=0.1,
            duration_ms=0.1,
        )
        workload = create_workload(args)
        metadata = workload.metadata()
        self.assertEqual(metadata["scale"], "tiny")
        self.assertEqual(metadata["populations"], ("GrC", "GoC"))
        self.assertEqual(metadata["sizes"]["GrC"], 2)
        self.assertEqual(metadata["sizes"]["GoC"], 1)

    def test_parse_args_accepts_steady_device_cuda_trace(self) -> None:
        args = _parse_args(
            [
                "--platform",
                "cuda",
                "--trace-phase",
                "steady",
                "--trace-device-only",
            ]
        )

        self.assertEqual(args.platform, "cuda")
        self.assertEqual(args.trace_phase, "steady")
        self.assertIs(args.trace_device_only, True)

    def test_profile_options_device_only_sets_available_jax_fields(self) -> None:
        class Options:
            python_tracer_level = None
            host_tracer_level = None
            enable_hlo_proto = None

        class Profiler:
            @staticmethod
            def ProfileOptions():
                return Options()

        class Jax:
            profiler = Profiler()

        options = _profile_options(Jax(), device_only=True)

        self.assertEqual(options.python_tracer_level, 0)
        self.assertEqual(options.host_tracer_level, 2)
        self.assertIs(options.enable_hlo_proto, True)


if __name__ == "__main__":
    unittest.main()
