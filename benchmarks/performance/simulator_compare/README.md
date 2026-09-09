# BrainCell / Jaxley / NEURON Comparison

Compare simulation cost under a fixed morphology, stimulus and recording protocol.
The shared `n144.swc` fixture lives in `data/morphology/`; `common.py` verifies its checksum.
Backends use 217 branches, four CVs per branch, classic HH, dt=0.025 ms and 20 ms duration.

| Script | Purpose |
| --- | --- |
| `run_benchmark.py` | Coordinate isolated backend workers and accuracy checks |
| `backend_braincell.py`, `backend_jaxley.py`, `backend_neuron.py` | Execute one backend |
| `braincell_ablation.py` | Compare population/vmap and spike bookkeeping modes |
| `common.py` | Shared model protocol, GPU selection and result validation |
| `plot_results.py`, `plot_diagnostics.py` | Plot existing measurement records |

Install BrainCell and Jaxley in one Python/JAX environment with GPU support. NEURON may
use a separate interpreter selected by `--neuron-python`. Both interpreter options default
to the current Python. GPU backends use the same explicitly selected physical device;
`--gpu-candidates` accepts nonnegative physical indices and chooses an idle candidate.
For a NEURON-only run use `--backend neuron`; no GPU selection is required.

```bash
python benchmarks/performance/simulator_compare/run_benchmark.py \
  --batch-sizes <comma-separated-counts> --gpu-candidates <physical-indices> \
  --warmup <count> --repeat <count> --neuron-warmup <count> --neuron-repeat <count> \
  --transfer-repeat <count> --output benchmarks/performance/simulator_compare/artifacts/run.json
python benchmarks/performance/simulator_compare/braincell_ablation.py run \
  --batch-sizes <comma-separated-counts> --gpu <physical-index> \
  --warmup <count> --repeat <count> --transfer-repeat <count> \
  --output benchmarks/performance/simulator_compare/artifacts/ablation.json
python benchmarks/performance/simulator_compare/plot_diagnostics.py \
  <result-json> --output-prefix <output-prefix>
```

NEURON runs serially at at most ten cells; larger sizes are explicitly marked linear
projections. Steady timing includes state restoration, 800 updates and synchronization;
construction, compilation and output transfers are reported separately. Accuracy failure
prevents the performance plot. BrainCell uses the point-local linearizer by default;
`--braincell-linearizer generic` selects the alternative. Spike-bookkeeping `off` is
restricted to isolated cells without runtime synapses.

Generated backend JSON, aggregate tables and plots belong in ignored `artifacts/`.
[Historical setup](results/historical-protocol.md) preserves prior environment and command
examples; it is not a new performance result. Reviewed measurements belong in `results/`.

Before measuring, follow the [execution rules](../../AGENTS.md): confirm the full
configuration matrix, independent rounds, first executions, warmups, timed repeats,
extra validation and time budget. Commands below are templates, not authorization.
Replace angle-bracket placeholders before running from the repository root.
