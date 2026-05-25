# Project Layout

`braincell` is organized so that the **public API is flat** (everything is
re-exported through the top-level `braincell` namespace) while the
**implementation is layered** into underscore-prefixed internal packages. This
page maps the package tree onto the {doc}`../concepts/architecture` layers.

## Naming convention

- **Internal packages** carry a leading underscore (`_base`, `_cv`, `_compute`,
  `_single_compartment`, `_multi_compartment`, `_misc`) because their *import
  paths* are not part of the supported public API.
- **Public re-exports** flow through `braincell/__init__.py`, plus the curated
  sub-namespaces `braincell.channel`, `braincell.ion`, `braincell.synapse`,
  `braincell.mech`, `braincell.quad`, `braincell.morph`, `braincell.filter`,
  `braincell.io`, and `braincell.vis`.
- **Modules inside** an internal package are unprefixed (`base.py`, `lower.py`,
  `runtime.py`) because they are import targets for sibling code in the same
  package.

## The map

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Package
  - Layer
  - Responsibility
* - `_base`, `_base_channel`, `_base_ion`
  - declaration
  - `HHTypedNeuron`, `IonChannel`, `Ion`, `MixIons`, `Channel`, `Synapse`
* - `_single_compartment`
  - declaration
  - the `SingleCompartment` class
* - `_multi_compartment`
  - declaration + runtime
  - `Cell`, `RunResult`, paint/place pipeline, probes, run loop
* - `mech`
  - declaration
  - the declarative mechanism specs (`Channel`, `Ion`, clamps, `Synapse`, …)
* - `filter`
  - declaration
  - region & locset selection algebra
* - `morph`
  - geometry
  - `Branch`, `Morphology`, typed branches, the mutable analysis tree
* - `_cv`
  - discretization
  - control volumes and CV policies
* - `_compute`
  - runtime
  - the execution graph and `CellRuntimeState`
* - `quad`
  - integration
  - the integrator protocol and solver registry
* - `channel`, `ion`, `synapse`
  - library
  - concrete, self-registering mechanism implementations
* - `io`
  - IO
  - SWC / ASC / NeuroML2 readers, NeuroMorpho client, checkpointing
* - `vis`
  - visualization
  - 2-D / 3-D rendering, morphometry, export
```

## Tests are co-located

Test files live **next to the source** they cover and are named `*_test.py`
(e.g. `braincell/io/neuromorpho/client.py` →
`braincell/io/neuromorpho/client_test.py`). This is the only naming pytest
discovers reliably in this repo. See {doc}`testing`.

## See also

- {doc}`../concepts/architecture` — the conceptual layers this maps onto.
- The repository's `CLAUDE.md` carries the authoritative, file-level layout.
