# Code and Design Navigation

Locate the source for the feature you want to change, then read that module's TODO and Current documents.
See [Repository Organization](../repository.md) for directory responsibilities and workflow entry points, and the
[System Overview](https://github.com/chaobrain/braincell/blob/main/docs/design/architecture/current/system-overview.md) for data flow across modules.

## Find code by task

Source paths are relative to the repository root. Each module's TODO lists current work and links to its API, architecture, and proposals.

| What to change | Source entry point | Design entry point |
| --- | --- | --- |
| Morphology construction, connections, statistics | `braincell/morph/` | [Morph](https://github.com/chaobrain/braincell/blob/main/docs/design/morph/TODO.md) |
| SWC/ASC, checkpoints, online morphology readers | `braincell/io/` | [IO](https://github.com/chaobrain/braincell/blob/main/docs/design/io/TODO.md) |
| Regions, locations, continuous sampling | `braincell/filter/` | [Filter](https://github.com/chaobrain/braincell/blob/main/docs/design/filter/TODO.md) |
| Mechanism declarations and registration | `braincell/mech/` | [Mech](https://github.com/chaobrain/braincell/blob/main/docs/design/mech/TODO.md) |
| Channel and ion dynamics | `braincell/channel/`, `braincell/ion/` | [Channel](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/TODO.md), [Ion](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/TODO.md) |
| Cell, discretization, runtime binding | `braincell/_multi_compartment/`, `braincell/_discretization/`, `braincell/_compute/` | [Cell](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/TODO.md) |
| Single-compartment models | `braincell/_single_compartment/` | [Single/Cell unification discussion](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/proposals/single-multi-compartment-unification.md) |
| Synaptic dynamics | `braincell/synapse/` | [Synapse](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/TODO.md) |
| Networks, event sources, connections | `braincell/network/` | [Network](https://github.com/chaobrain/braincell/blob/main/docs/design/network/TODO.md) |
| Numerical integration and voltage solvers | `braincell/quad/` | [Quad](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/TODO.md) |
| Parameter selection and learning mappings | `braincell/trainable/` | [Optim](https://github.com/chaobrain/braincell/blob/main/docs/design/optim/TODO.md) |
| Visualization | `braincell/vis/` | [Vis](https://github.com/chaobrain/braincell/blob/main/docs/design/vis/TODO.md) |

Internal source paths differ from public import names. For example, import Branch and Morphology from the top-level `braincell` package and mechanism declarations from `braincell.mech`. Use the public entry points described in the module's Current documents.

## Experimental, validation, and performance workflows

Reusable experimental gradient interfaces live in `braincell.experimental.optim`, with usage workflows in `examples/optim/`.
Numerical accuracy comparisons live in `validation/`, performance and scaling measurements in `benchmarks/`, and shared references in `data/`.
Cerebellar models keep morphology, MOD files, and parameters under `data/cerebellum/<model>/`; compiled outputs go into validation artifacts.
See [Repository Organization](../repository.md) for entry points to specific commands.

## Read and update in order

1. Find the task in the module TODO. Current documents describe current behavior; proposals preserve designs awaiting discussion or implementation.
2. Compare the source and adjacent `*_test.py` files to understand existing constraints and actual usage.
3. Review relevant examples or numerical comparisons to determine what results the change must validate.
4. After implementation, update affected Current documents, examples, and task status. Preserve useful historical decisions in dated specs.

See the [Extension Guide](extending.md) for adding mechanisms and the [Testing Guide](testing.md) for locating tests and using fixtures.
The [Design rules](https://github.com/chaobrain/braincell/blob/main/docs/design/AGENTS.md) maintain document responsibilities and status definitions.
