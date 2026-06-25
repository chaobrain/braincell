from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.profiling.cases.cerebellar_probability_network import create_workload
from examples.profiling.profile_simulation import main


def test_neuron_compare_cell_profile_smoke(tmp_path: Path) -> None:
    out = tmp_path / "profile_cell.json"
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
    assert phases == [
        "import_case",
        "load_params",
        "import_model",
        "model_build",
        "place_probes",
        "init_reset",
        "steady_run",
        "materialize",
    ]
    assert payload["metadata"]["cell"] == "grc_ma2020"
    assert all(row["wall_time_s"] >= 0.0 for row in payload["phases"])
    assert payload["materialized"]["voltage_shape"][0] == 1


def test_cerebellar_probability_network_workload_metadata_subset() -> None:
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
    assert metadata["scale"] == "tiny"
    assert metadata["populations"] == ("GrC", "GoC")
    assert metadata["sizes"]["GrC"] == 2
    assert metadata["sizes"]["GoC"] == 1
