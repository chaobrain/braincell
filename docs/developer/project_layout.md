# Code Navigation

Locate the source for the feature you want to change, then read its public API and the adjacent tests.
See [Repository Organization](../repository.md) for directory responsibilities and workflow entry points, and the
[Architecture](../concepts/architecture.md) page for data flow across modules.

## Find code by task

Source paths are relative to the repository root.

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
| Visualization | `braincell/vis/` | [Vis](../apis/vis.rst) |

Internal source paths differ from public import names. For example, import Branch and Morphology from the top-level `braincell` package and mechanism declarations from `braincell.mech`. Use the public entry points listed in the API reference.

## Experimental, validation, and performance workflows

Reusable experimental gradient interfaces live in `braincell.experimental.optim`, with usage workflows in `examples/optim/`.
Numerical accuracy comparisons live in `validation/`, performance and scaling measurements in `benchmarks/`, and shared references in `data/`.
Cerebellar models keep morphology, MOD files, and parameters under `data/cerebellum/<model>/`; compiled outputs go into validation artifacts.
See [Repository Organization](../repository.md) for entry points to specific commands.

## Read and update in order

1. Start from the task or issue that describes the change.
2. Compare the source and adjacent `*_test.py` files to understand existing constraints and actual usage.
3. Review relevant examples or numerical comparisons to determine what results the change must validate.
4. After implementation, update the affected docstrings, documentation, examples, and `changelog.md` in the same PR.

See the [Extension Guide](extending.md) for adding mechanisms and the [Testing Guide](testing.md) for locating tests and using fixtures.
