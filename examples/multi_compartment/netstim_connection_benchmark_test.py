import netstim_connection_benchmark as benchmark


def test_build_model_has_two_synapses_and_connections_per_cell():
    cell, sources, connection = benchmark.build_model(4, heterogeneous=True)

    assert len(cell.synapses) == 8
    assert sources.size == 8
    assert len(connection) == 8


def test_small_benchmark_reports_all_cost_categories():
    result = benchmark.benchmark(2, heterogeneous=True, steps=1, repeats=1)

    assert result["platform"] in {"cpu", "gpu", "tpu"}
    assert result["synapse_instances"] == 4
    assert result["source_bytes"] > 0
    assert result["connection_bytes"] > 0
    assert result["synapse_bytes"] > 0
    assert result["first_run_s"] > 0.0
