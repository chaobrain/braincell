# Repository Layout

Where the code, tests, examples and measurements live, and where to start for a given change.

## Top-level directories

| Directory | Holds |
| --- | --- |
| [braincell/](https://github.com/chaobrain/braincell/tree/main/braincell) | The library and its co-located `*_test.py` files, including `braincell.experimental`. |
| [examples/](https://github.com/chaobrain/braincell/tree/main/examples) | Usage examples, tutorials and task workflows. |
| [validation/](https://github.com/chaobrain/braincell/blob/main/validation/README.md) | Numerical accuracy and correctness comparisons, such as the NEURON suites. |
| [benchmarks/](https://github.com/chaobrain/braincell/blob/main/benchmarks/README.md) | Performance, memory and scaling measurements. Reviewed summaries are tracked in each experiment's `results/`; raw outputs go to ignored `artifacts/`. |
| [data/](https://github.com/chaobrain/braincell/blob/main/data/README.md) | Shared input data and reference material: morphologies, cerebellar models, MOD files. |
| [docs/](https://github.com/chaobrain/braincell/tree/main/docs) | User documentation, the API reference and this guide. |

Each directory's own README describes its internal structure, commands and data. Update this table when a top-level directory's role changes.

## Find code by task

Source paths are relative to the repository root. The [Architecture](../concepts/architecture.md) page shows how data flows across these modules.

| What to change | Source entry point | Public API |
| --- | --- | --- |
| Morphology construction, connections, statistics | `braincell/morph/` | [Morphology](../apis/morphology.rst) |
| SWC/ASC, checkpoints, online morphology readers | `braincell/io/` | [IO](../apis/io.rst) |
| Regions, locations, continuous sampling | `braincell/filter/` | [Filter](../apis/filter.rst) |
| Mechanism declarations and registration | `braincell/mech/` | [braincell.mech](../apis/braincell.mech.rst) |
| Channel and ion dynamics | `braincell/channel/`, `braincell/ion/` | [braincell.channel](../apis/braincell.channel.rst), [braincell.ion](../apis/braincell.ion.rst) |
| Cell, discretization, runtime binding | `braincell/_multi_compartment/`, `braincell/_discretization/`, `braincell/_compute/` | [braincell](../apis/braincell.rst) |
| Single-compartment models | `braincell/_single_compartment/` | [braincell](../apis/braincell.rst) |
| Synaptic dynamics | `braincell/synapse/` | [braincell.synapse](../apis/braincell.synapse.rst) |
| Networks, event sources, connections | `braincell/network/` | — |
| Numerical integration and voltage solvers | `braincell/quad/` | [Integration](../apis/integration.rst) |
| Parameter selection and learning mappings | `braincell/trainable/` | — |
| Model reduction | `braincell/reduction/` | — |
| Visualization | `braincell/vis/` | [Vis](../apis/vis.rst) |

Internal source paths differ from public import names: import `Branch` and `Morphology` from the top-level `braincell` package and mechanism declarations from `braincell.mech`. [Code Conventions](conventions.md#imports-and-package-layout) explains which packages are public.

## Experimental, validation, and performance workflows

Reusable experimental gradient interfaces live in `braincell.experimental.optim`, with usage workflows in `examples/optim/`.
Numerical accuracy comparisons live in `validation/`, performance and scaling measurements in `benchmarks/`, and shared references in `data/`.
Cerebellar models keep morphology, MOD files, and parameters under `data/cerebellum/<model>/`; compiled outputs go into validation artifacts.

## Read before you change

1. Start from the task or issue that describes the change.
2. Read the source and the adjacent `*_test.py` files to understand existing constraints and actual usage.
3. Review relevant examples or numerical comparisons to determine what results the change must validate.
4. After implementation, update the affected docstrings, documentation, examples, and `changelog.md` in the same PR.

See the [Extension Guide](extending.md) for adding mechanisms and the [Testing Guide](testing.md) for locating tests and using fixtures.
