# Release Notes


## Unreleased

### Breaking Changes

- **`Cell` now always carries a population axis.** `pop_size` defaults to
  `1` and an explicitly empty `pop_size=()` raises `ValueError`. Runtime
  state is therefore always at least two-dimensional: `Cell.V` is shaped
  `pop_size + (n_cv,)` and point-space arrays `pop_size + (n_point,)`.
  Code that relied on the old rank-0 default sees one extra leading axis
  of length 1; `reshape(-1)` or indexing `[0]` recovers the previous view.
- `DiffEqState` is now a marker mixin rather than a concrete state class.
  `DiffEqState(value)` raises `TypeError`. The two concrete classes are
  `DiffEqSingleState` (over `brainstate.HiddenState`, used by
  `SingleCompartment`) and `DiffEqGroupState` (over
  `brainstate.HiddenGroupState`, used by `Cell`). Every
  `isinstance(x, DiffEqState)` check keeps working unchanged.
- `braincell.diffeq_state` is renamed to `braincell.state`. No alias is
  provided.

  ```python
  # before
  self.m = braincell.DiffEqState(init)

  # after (preferred — picks the right class for the host)
  self.m = braincell.state(init)

  # after (explicit)
  self.m = braincell.DiffEqSingleState(init)
  ```

### New Features

- **`braincell.MultiCompartment` is a new alias of `braincell.Cell`.** The
  same class object under a name that spells out what it models, which
  reads better next to `braincell.SingleCompartment`. Nothing about `Cell`
  changes; both names remain supported.
- **Hidden-state classes now match their host model.** Every hidden state
  owned by a `Cell` is a `brainstate.HiddenGroupState` — `Cell.V` is the
  new `braincell.DiffEqGroupState` — so the trailing compartment/point
  axis is exposed as a group of independently traced hidden states, which
  is what eligibility-trace learning requires. `SingleCompartment` has no
  spatial axis and uses `DiffEqSingleState`, which is itself a
  `brainstate.HiddenState`. `DiffEqGroupState` derives from `DiffEqState`,
  so every solver selects it unchanged.
- **`state` / `hidden_state` / `state_grouping`** are exported for
  custom mechanisms. Channel, ion, and synapse code is shared by both host
  models, so the right class cannot be chosen at the creation site.
  `DiffEqState(...)` now raises `TypeError`; write `state(...)` in a
  custom `init_state` instead, and the right class is chosen per host.

The remaining entries in this section harden the declarative channel
template layer that `braincell.channel` is built on. The catalogue's
numeric output is unchanged: each either adds a check that shipped
channels already pass, or moves code they already shared.

- **`OhmicHH`** — an `HH` subclass supplying the ohmic current
  `g_max * conductance_factor(...) * (E - V)`. 63 channels that restated
  it verbatim now inherit it; a channel driven by a fixed `self.E`
  overrides `reversal_potential()` instead. GHK-flux and
  permeability-scaled channels keep inheriting `HH` directly.
- **`Gate.time_unit`** — the unit a bare `f_*_tau` / `f_*_alpha` /
  `f_*_beta` return is read in, `u.ms` by default. Rate methods may now
  return properly united quantities instead.
- **`Gate.clip` / `Markov.clip_states`** — explicit, per-gate control of
  whether a state is projected into `[0, 1]` at the point of use.
  `Gate.clip` is off by default (NEURON does not clip HH gates);
  `clip_states` is on, preserving existing `Markov` behaviour.
- **String references in gate metadata** — `Gate(q10="q10")` resolves
  the named attribute off the instance, replacing 75
  `lambda self: self.q10` closures that were unpicklable and opaque to
  tooling. The callable form still works.
- `Gate`, `Transition`, `HH`, `OhmicHH`, `Markov` and `ghk_flux` are
  exported from `braincell.channel`; they previously had to be imported
  from the private `braincell.channel._base`.

### Changes & Improvements

- **Definition-time validation.** `HH` and `Markov` resolve and check
  `gates` / `pairs` in `__init_subclass__`. A mistyped gate name, a
  duplicate, a gate defining neither or both rate forms, a transition
  naming a missing rate method, a `dependent_state` outside the state
  set, and a Markov class with fewer than two states are now errors when
  the class is created rather than at `reset_state()` or not at all.
- **Unit checking on the derivative.** Gate and transition rates are
  dimension-dispatched, and every state derivative is asserted to be an
  inverse time before it reaches the integrator. A dimensioned `phi`
  used to yield a `mV / ms` derivative silently.
- HH gate methods receive only the ion arguments their signature
  declares, matching what `Markov` already did. This removed 178 lines
  of forwarding boilerplate from `potassium_sodium.py`.

### Bug Fixes

- `Cell.vis_cv(...)` / `Cell.vis_node(...)` and node-local runtime
  inspection (`cell.runtime_nodes[i].ions[...]`) rejected any field with a
  population axis, so they failed for every `pop_size` beyond the old
  rank-0 default. They now collapse a single-member population and
  otherwise raise naming the field and `pop_size`.
- Ion baseline broadcasting in `_sync_runtime_ion` ignored the population
  axis, disagreeing with the sibling code path.
- Dense channel construction dropped the population axis when a channel's
  parameters were all scalars, producing a rank-1 gate state that failed
  as a `jit`/`scan` carry mismatch.
- A kinetic-ion species declared with a scalar initializer (for example
  `CdpCR_MA2020_GrC.pumpca`) started rank-0 and was silently reshaped
  mid-simulation by the conservation write-back; it is now allocated at
  full shape.
- `set_module_as` assigned the public module path to `__name__` instead of
  `__module__`, so every function it decorated reported itself as being
  called `"braincell.quad"` while still advertising the private module it
  is defined in. Both are now correct: `braincell.quad.rk4_step.__name__`
  is `'rk4_step'` and its `__module__` is `'braincell.quad'`. This also
  changes `IntegratorEntry.module` from the private defining module to the
  public export path, and makes `dhs_voltage_step` consistent with the
  other integrators.
- `init_state()` refused nothing: a gate or Markov state whose name
  matched a constructor parameter silently replaced that parameter with
  a `DiffEqState`. It now raises, naming the collision.

### Deprecations

- `Markov` subclasses that do not declare `dependent_state` emit a
  `DeprecationWarning`. The implicit fallback — "the last state
  discovered while scanning `pairs`" — makes a reordering of `pairs`
  silently change which state is eliminated, and will be removed. All
  shipped channels now declare it explicitly.


## Version 0.1.0

This is a landmark release. BrainCell evolves from single-compartment
Hodgkin–Huxley modeling into a complete **multi-compartment, morphologically
detailed** neuron simulation framework in JAX. It introduces a morphology layer,
a control-volume discretization engine with pluggable policies, a compute
runtime, morphology IO (SWC / ASC / NeuroML2 / NeuroMorpho.Org), a declarative
mechanism system, location/region selection filters, and a 2D/3D visualization
stack.

### New Features

- **Multi-Compartment Morphological Modeling** (#69)
  - New `Cell` declaration frontend and frozen `RunnableCell` runtime for
    simulating branched morphologies.
  - High-level `rcell.run(dt=, duration=)` driver returning a structured
    `RunResult`.

- **Morphology Layer** (#68, #69)
  - Immutable `Branch` with typed subclasses (`Soma`, `Dendrite`, `Axon`,
    `BasalDendrite`, `ApicalDendrite`, `CustomBranch`).
  - Mutable `Morphology` tree with whole-morphology metric snapshots
    (`MorphoMetric`).

- **Control-Volume Discretization** (#72, #74, #88)
  - Pure-functional CV layer with composable policies: `CVPerBranch`,
    `DLambda`, `MaxCVLen`.
  - Paint/place rule machinery for mapping mechanisms onto morphology by region.

- **Compute Runtime** (#74, #88)
  - Execution-graph lowering (`NodeTree`), scheduling, runtime-state
    installation, and channel–ion binding resolution built on top of the
    discretization layer.

- **Declarative Mechanism System** (`braincell.mech`) (#69)
  - Hashable, order-insensitive `Density` / `Point` declarations and a
    mechanism registry (`@register_channel`, `@register_ion`,
    `@register_synapse`).
  - Point mechanisms: current/sine/function clamps, probes, synapses, and
    gap junctions.

- **Morphology IO** (`braincell.io`) (#68, #69)
  - Readers for **SWC**, **ASC**, and **NeuroML2**.
  - Three-tier **NeuroMorpho.Org** client with on-disk caching and a
    `braincell-neuromorpho` CLI.

- **Location/Region Filters** (`braincell.filter`) (#69)
  - Locset and region selection expressions (`ForkPoints`, `Terminals`,
    `UniformSamples`, `SubtreeRegion`, `BranchRangeFilter`, …) with selection
    caching.
  - Ordered, duplicate-preserving locsets with stable set algebra, explicit
    `unique()`, and read-only columnar `LocsetMask` storage. `BranchPoints`
    remains a compatibility alias for `ForkPoints`.

- **Visualization Stack** (`braincell.vis`) (#80, #82, #102)
  - 2D (matplotlib) and 3D (PyVista, Plotly) backends with a unified backend
    chooser.
  - 2D tree-layout engine, morphometry plots (dendrogram, Sholl, topology,
    branch-order histogram), trace panels, movies, and morphology/value
    comparators.

- **Cerebellum Dynamics** (#93)
  - Additional ion/channel dynamics and a Purkinje-cell MA2024 comparison
    scaffold.

### Breaking Changes

- **Removed the direct external-current injection path** for multi-compartment
  cells. The `Cell.update(I_ext)` path is gone; inject external current with
  placed point clamps instead — `CurrentClamp(...)`, `SineClamp`, or
  `FunctionClamp`.
- **Renamed the discretization package** `_cv` → `_discretization`, and
  `PointTree` → `NodeTree`, for clearer terminology (#88).

### Changes & Improvements

- Restructured the single-compartment module and import surface (#71).
- Refreshed the channel/ion public API and added deprecation aliases for the
  previous channel names (#97).
- Hardened multi-compartment dtype boundaries and runtime caching, including
  mixed-ion runtime fixes (#73, #82).
- Rendered the PyVista HTML backend as a static iframe for reliable notebook
  and docs embedding (#102).

### Bug Fixes

- Used `default_factory` for `brainunit.Quantity` dataclass field defaults to
  avoid shared mutable defaults (#92).

### Removed

- Dropped the `diffrax` dependency from the `quad` integrator stack (#94).

### Documentation

- Rebuilt the documentation around a layered, Arbor-inspired architecture;
  consolidated tutorials into runnable notebooks and expanded single-compartment
  examples (#101, #103, #104, #105).
- Added a top-level Numerical Integration tutorial and documented previously
  missing public APIs (#98, #99).
- Self-hosted the documentation at <https://brainx.chaobrain.com/braincell/>.

### Packaging & Tooling

- Marked the package as typed per **PEP 561** (added `py.typed`), so downstream
  type checkers consume BrainCell's inline annotations.
- Numerous CI workflow and dependency updates: deploy docs on release, GitHub
  Actions version bumps, and `brainx-sphinx-header` upgrades.

## Version 0.0.7

This release focuses on structural refactoring to improve codebase organization, specifically grouping morphology and integrator components into dedicated sub-packages (`braincell.morph` and `braincell.quad`).

### Refactoring & Code Organization

- **Morphology Sub-package** (`braincell.morph`)
  - Moved and renamed morphology-related modules into `braincell/morph/`:
    - `_morphology.py` -> `morph/_morphology.py`
    - `_morphology_branch_tree.py` -> `morph/_branch_tree.py`
    - `_morphology_from_asc.py` -> `morph/_from_asc.py`
    - `_morphology_from_swc.py` -> `morph/_from_swc.py`
    - `_morphology_utils.py` -> `morph/_utils.py`

- **Integrator Sub-package** (`braincell.quad`)
  - Moved all integrator and solver modules into `braincell/quad/`:
    - `_integrator*.py` files moved to `braincell/quad/`.
  - This improves the clarity of the top-level namespace.

### Documentation

- **Structure Updates**
  - Updated API documentation to reflect the new module structure.
  - Simplified `index.rst` and reorganized API reference pages.
  - Updated copyright to reflect membership in the BrainX Ecosystem.

### CI/CD

- **Workflow Updates**
  - Bumped versions for `actions/checkout`, `upload-artifact`, and `download-artifact`.

## Version 0.0.6

This release focuses on major dependency updates, code modernization, and extensive refactoring to improve compatibility with the latest BrainPy ecosystem.

### Breaking Changes

- **Dependency Version Updates**
  - Updated `brainstate` from `>=0.1.0` to `>=0.2.0`
  - Updated `brainpy` from `>=3.0.0` to `>=2.7.0`
  - These updates may require users to upgrade their BrainPy ecosystem packages

### Refactoring & Code Improvements

- **Core Architecture Simplification** (2acd212)
  - Refactored `HHTypedNeuron` to use `brainpy` directly for better integration
  - Simplified `_base.py` with significant code reduction (221 insertions, 282 deletions)
  - Removed deprecated `_integrator_diffrax.py` module (29 lines removed)
  - Streamlined integrator implementations in `_integrator_runge_kutta.py`
  - Cleaned up `_single_compartment.py` and integration protocol

- **Parameter Initialization Migration** (fa71171, a79c306, 18b053c, 77a11ac)
  - Migrated parameter initialization from `brainstate.nn` to `braintools` across the entire codebase
  - Updated parameter initialization in ion channels (calcium, potassium, sodium, hyperpolarization-activated)
  - Refactored parameter initialization in synapse models (markov)
  - Updated HTC and EINet classes to use `braintools`
  - Updated all example scripts and notebooks to use `braintools` for parameter initialization

- **API Migration** (e84351a, bf50e6e)
  - Migrated from `brainstate.nn` to `brainpy.state` and `braintools`
  - Fixed `_base` errors in brainpy integration
  - Updated `CurrentProj` references across the codebase

### Documentation

- **Updated Documentation** (#54, 2acd212)
  - Updated braincell logo image
  - Refreshed tutorial notebooks (cell, channel, ion tutorials in both English and Chinese)
  - Updated advanced tutorial examples (sc02-sc05 notebooks)
  - Revised quickstart concepts documentation
  - Updated all documentation to reflect API changes and new parameter initialization patterns

### Examples

- **Example Updates**
  - Updated all example scripts to use new APIs:
    - `SC01_fitting_a_hh_neuron.py`
    - `SC03_COBA_HH_2007_braincell.py`
    - `SC05_thalamus_single_compartment_neurons.py`
    - `SC06_unified_thalamus_model.py`
    - `SC07_Straital_beta_oscillation_2011.py`
    - `MC11_simple_dendrite_model.py`
    - `MC13_golgi_model/` simulations

### CI/CD

- **Publishing Workflow Enhancement** (2acd212)
  - Updated `.github/workflows/Publish.yml` with improved configuration

### Code Statistics

- Overall changes: 48 files changed, 1,307 insertions(+), 1,408 deletions(-)
- Net reduction of ~100 lines while improving code quality and maintainability

## Version 0.0.5

This release brings significant performance improvements, new integration methods, enhanced morphology support, expanded documentation, and modernized packaging infrastructure.

### New Features

- **Pallas Kernel Acceleration** (#51)
  - Added Pallas kernel support for voltage solver to accelerate multi-compartment simulations
  - Introduced optimized triangular matrix computation with GPU/CPU backend support
  - Added debug kernels for Pallas backend testing

- **Backward Euler Solver** (#49)
  - Added backward Euler integration method for improved numerical stability
  - Enhanced integration infrastructure with new solver options

- **Morphology Enhancements** (#41, #46, #51)
  - Added support for immutable sections
  - Implemented DHS (Diagonal Hines Solver) support
  - Added lazy loading of networkx for better performance
  - Improved morphology branch tree handling and documentation
  - Enhanced ASC/SWC file support for morphology loading

### Performance Improvements

- **Sodium Channel Integration** (da6697f, 7f91bbe, 7c218f1)
  - Refactored sodium integration from backward Euler to RK4 solver for better accuracy
  - Updated population size handling in simulations
  - Optimized voltage solver performance

- **Integration System Refactoring** (#47)
  - Refactored integrators to get time from `brainstate.environ` for better consistency
  - Streamlined solver logic and improved code structure

### Documentation

- **Expanded Chinese Documentation** (#45)
  - Added comprehensive Chinese language documentation
  - Included advanced tutorial examples and API references

- **New Documentation Structure** (#40, #42)
  - Added quickstart guides, tutorials, and advanced tutorials
  - Reorganized documentation for better navigation
  - Enhanced code documentation and type hints (#44)

### Infrastructure & Dependencies

- **Packaging Modernization**
  - Migrated from `setup.py` to modern `pyproject.toml`-only configuration
  - Updated license format to SPDX identifier (`Apache-2.0`)
  - Improved package metadata and dependency specifications

- **Dependencies**
  - Added `brainpy>=3.0.0` as core dependency
  - Added `braintools>=0.1.0` for enhanced tooling
  - Updated CI/CD configurations for Python 3.13 support

- **CI/CD Updates**
  - Added Python 3.13 support (#50, #48)
  - Updated GitHub Actions: setup-python from 5 to 6, checkout from 4 to 5

### Code Quality

- **Refactoring & Improvements** (#44)
  - Improved external current registration and error handling
  - Enhanced type hints across the codebase
  - Better code organization and readability

### Examples & Testing

- Added linear solver test notebooks
- Enhanced Golgi model simulation examples
- Updated example scripts for better demonstration of features

## Version 0.0.4

Previous release with core functionality.

## Version 0.0.1

The first release of the project.
