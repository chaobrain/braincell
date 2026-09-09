# Examples

Runnable tutorials and task workflows, grouped by topic.

- [Cell](cell/README.md): one-CV models, stimulation and recording.
- [Synapse](synapse/README.md): placement, views and spatial sampling.
- [Network](network/README.md): event connections and population simulations.
- [IO](io/README.md): morphology checkpoints and NeuroMorpho downloads.
- [Integrators](quad/README.md): differential equations and solver selection.
- [Visualization](vis/README.md): morphology, fields, traces and export.
- [Reduction](reduction/README.md): reduced cells driven by network inputs.
- [Optimization](optim/README.md): parameter learning, fitting and stimulus design.
- [Single compartment](single_compartment): single-compartment models and teaching scripts.
- [MOD conversion](convert_mod/nmodl/README.md): NMODL conversion tools and walkthroughs.

Suggested reading: [Cell](cell/README.md) → [Synapse](synapse/README.md) →
[Connection](network/connection.ipynb) → [Network](network/network.ipynb) →
[Recording](cell/recording.ipynb).

Use a development installation (`pip install -e ".[dev]"`) and start notebooks
from the repository root or their own directory. Each topic lists additional
requirements and output locations. Generated files belong to local `artifacts/`.

[Validation](../validation/README.md), [benchmarks](../benchmarks/README.md) and
[shared data](../data/README.md) provide numerical checks, performance measurements
and reference inputs. Detailed API and architecture contracts live in
[Design](../docs/design/TODO.md).

Older plots with no confirmed workflow owner are preserved locally under
`artifacts/legacy_multi_compartment/plot/`.
