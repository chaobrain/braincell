# Sampling Performance

Measure connection construction and continuous morphology-region sampling as input
size grows. Requires a BrainCell source checkout and its NumPy/BrainUnit dependencies.

| Script | Purpose and output |
| --- | --- |
| `connection_sampling_benchmark.py` | Compare independent, source-first and by-source construction; emits one JSON record |
| `continuous_region_sampling_benchmark.py` | Sweep segment and output counts with uniform or custom density; emits JSON lines |

Each selected configuration is evaluated once, without a separate warmup. Construction
and Python peak-memory measurements are not simulation-step timings. Capture stdout in
an ignored `artifacts/` directory; maintained summaries belong in `results/` when available.

```bash
mkdir -p benchmarks/performance/sampling/artifacts
python -m benchmarks.performance.sampling.connection_sampling_benchmark \
  --strategy <strategy> --source-size <sources> --target-size <targets> --rows <rows> \
  > benchmarks/performance/sampling/artifacts/connections.json
python -m benchmarks.performance.sampling.continuous_region_sampling_benchmark \
  --components <comma-separated-counts> --samples <comma-separated-counts> \
  > benchmarks/performance/sampling/artifacts/regions.jsonl
```

Use `--custom-density` for the continuous sampler's linear density profile.

Before measuring, follow the [execution rules](../../AGENTS.md): confirm the full
configuration matrix, independent rounds, first executions, warmups, timed repeats,
extra validation and time budget. Commands below are templates, not authorization.
Replace angle-bracket placeholders before running from the repository root.
