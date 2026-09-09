# Design TODO 历史快照

归档日期：2026-09-07。来源：`docs/design/TODO.md` 的整理前工作区版本。
以下保留原记录；其中的实现状态和测试数量没有在归档时重新验收。现行入口为 [Design TODO](../design/TODO.md)。

---

# BrainCell Project Design and TODO

> Status: living project-level progress index. Tracks macro goals, module
> milestones, major dependencies, and blockers. Detailed contracts, local
> stages, and acceptance criteria belong in their module design documents.
> Status markers in this file follow:
>
> - `[x]` shipped — implemented, covered by `*_test.py`, and consistent with
>   the applicable design and usage examples.
> - `[~]` partial — implementation exists but is missing functionality,
>   tests, or runtime integration. Specific gaps are listed inline.
> - `[ ]` planned — design agreed, code not yet written.
> - `[ ]` research — explicitly labeled investigation; the approach or
>   implementation contract is not yet settled.
>
> This document describes committed repository state. Experimental work in an
> uncommitted working tree does not become a shipped capability until its API,
> implementation, tests, and applicable examples land together.

---

## Document Navigation

- [Mission and scope](#1-mission-and-scope)
- [Top-level architecture](#2-top-level-architecture)
- [Module catalogue](#3-module-catalogue)
- [Cross-cutting concerns](#4-cross-cutting-concerns)
- [Public API contract](#6-public-api-contract)
- [End-to-end workflows](#7-end-to-end-user-workflows)

Each module TODO tracks local questions and next actions; Current documents describe
implemented behavior, Proposals compare designs, and References preserve supporting evidence.

| Topic | Collaboration entry |
| --- | --- |
| Cross-module architecture and interface consistency | [Architecture TODO](../design/architecture/TODO.md) |
| Cell frontend, runtime, and single ODE discussion | [Cell TODO](../design/cell/TODO.md) |
| Channels and template invariants | [Channel TODO](../design/channel/TODO.md) |
| Ions and shared mechanism bibliography | [Ion TODO](../design/ion/TODO.md) |
| Morphology readers and writers | [IO TODO](../design/io/TODO.md) |
| Morphology layering | [Morph TODO](../design/morph/TODO.md) |
| Spatial selections and callable parameters | [Filter TODO](../design/filter/TODO.md) |
| Populations and event routing | [Network TODO](../design/network/TODO.md) |
| Parameter learning and optimization experiments | [Optimization TODO](../design/optim/TODO.md) |
| Reduction model integration and DBNN | [Reduction TODO](../design/reduction/TODO.md) |
| Visualization and migration to BrainTools | [Vis TODO](../design/vis/TODO.md) |

Model-specific imports and numerical comparisons are tracked with the actual
[Cerebellum examples](../../examples/neuron_compare/cerebellum-import-progress.md).
Update this index when a macro milestone or blocker changes. Writing and ownership rules
are maintained in [Design conventions](../design/AGENTS.md); commit checks follow the repository
[design, code, and examples agreement](../../AGENTS.md#design-code-and-examples).

## Current Architecture Snapshot

The committed repository currently provides:

- [x] **Cerebellum channel/ion imports and tests expanded.** The channel
  catalogue now includes PC MA2024 channel variants and the calcium-ion
  catalogue includes concrete Cerebellum kinetic-ion imports such as
  `CdpStC_*`, `CdpCAM_MA2024_PC`, and `CdpCR_MA2020_GrC`, with co-located
  unit tests and NEURON-comparison notebooks under `examples/neuron_compare`.
- [x] **PC MA2024 assembly scaffold added.** `examples/neuron_compare/cell/pc_ma2024`
  contains the simplified NEURON assembly, the matching BrainCell assembly,
  shared parameter loading, debug variants, and `run.ipynb` for side-by-side
  simulation.
- [x] **Direct multi-compartment runtime.** `Cell` owns declaration and runtime
  state, is initialized with `init_state()`, and advances directly with
  `run()`. The former `Cell -> RunnableCell` build boundary no longer exists.
- [x] **Population network runtime.** `braincell.network` owns population
  registration, event routing, initialization and result aggregation while
  synapses, connections and recordings remain owned by their target `Cell`.
- [x] **Trainable parameter mappings.** `braincell.trainable` maps selected
  channel fields from direct, shared-scale or latent parameter sources into
  runtime values. Optimizers, losses and training loops remain user-owned.
- [x] **SWC writing and structural round trips.** `Morphology.to_swc()` writes
  branch trees through `braincell.io.swc`; focused tests cover shared branch
  endpoints, soma attachments, reversed branches and validation failures.
- [x] **NEURON-style ion-current snapshot mode added.** `Cell(...,
  cache_ion_total_current=True)` caches the total ion current at the start of
  the staggered step, before voltage or ion state advances, so current-driven
  ion mechanisms can read the same precomputed current snapshot that
  NEURON-style scheduling expects.
- [x] **Frozen voltage channel variants added where needed.** Some PC calcium
  channels now have `_Frozen` variants which stop differentiation through the
  voltage used inside the current expression, matching the intended NEURON
  semantics for those mechanisms during the comparison.
- [x] **Two ion/channel update schedules are available.**
  `ion_channel_update_order="family"` restores the NEURON-like family
  ordering for ion/channel updates; `"integration"` keeps the previous
  BrainCell integration-oriented ordering.
- [x] **Homogeneous multi-compartment `Cell` populations now support
  multi-dimensional `pop_size`.** `Cell(..., pop_size=(...))` expands
  runtime state to `pop_size + (n_cv,)`, point-space runtime arrays to
  `pop_size + (n_point,)`, and supports population-specific
  `CurrentClamp(...)` amplitudes such as `(2,)` or `(2, 2)`-shaped
  current grids. Regression coverage includes `(2,)` and `(2, 2)`
  populations.
- [x] **The population axis is mandatory.** `pop_size` defaults to `1`
  and an explicitly empty `pop_size=()` is rejected, so every `Cell`
  hidden state is at least two-dimensional and its trailing axis always
  enumerates compartments or points. That invariant is what lets `Cell`
  states be `brainstate.HiddenGroupState` (`Cell.V` is a
  `braincell.DiffEqGroupState`) while `SingleCompartment`, which has no
  spatial axis, keeps the plain `brainstate.HiddenState`. See
  `docs/specs/2026-08-13-cell-hidden-group-state.md`.
- [x] **The channel template layer validates at class-definition time.**
  `HH` and `Markov` resolve and check `gates` / `pairs` in
  `__init_subclass__`, so a mistyped gate name, a duplicate, a gate
  defining neither (or both) rate forms, a transition naming a missing
  rate method, and a `dependent_state` outside the state set are all
  rejected when the class is created rather than at `reset_state()`.
  `init_state` refuses to bind a gate over a non-`DiffEqState`
  attribute, which used to silently replace a constructor parameter.
- [x] **Gate and transition rates carry real units.** `Gate.time_unit`
  (default `u.ms`) says what a bare `f_*_tau` / `f_*_alpha` / `f_*_beta`
  return means; a united return is used as given and a wrong dimension
  is rejected against the gate by name. Markov transition rates accept
  the same two forms against a fixed `u.ms`. Every state derivative is
  asserted to be an inverse time before it reaches the integrator,
  which is what catches a dimensioned `phi`.
- [x] **`OhmicHH` carries the ohmic driving force.** 63 channels that
  restated `g_max * conductance_factor(...) * (E - V)` verbatim now
  inherit it; a channel reading a fixed `self.E` overrides
  `reversal_potential()`. GHK-flux and permeability-scaled channels
  keep inheriting `HH` and writing their own `current()`.
- [x] **Gate metadata binds by attribute name.** `Gate(q10="q10")`
  replaces the 75 `lambda self: self.q10` closures, which were
  unpicklable and invisible to tooling. The callable form still works.
- [~] **Gate/state clipping is an explicit policy.** `Gate.clip`
  defaults to `False` (NEURON does not clip HH gates, and the catalogue
  is validated against those mechanisms); `Markov.clip_states` defaults
  to `True`. Both project only the value fed to the conductance product
  or the kinetics, never the stored state. Remaining gap: the implicit
  `dependent_state` fallback still exists behind a `DeprecationWarning`
  and is slated for removal.

## 1. Mission and Scope

BrainCell is a JAX-native library for **biologically detailed cell and network
modelling**. It targets the same workload as NEURON, Arbor, and BluePyOpt but
expresses models as differentiable, vectorized JAX programs so that
multi-compartment populations can be simulated, connected, batched, and
parameterized inside the broader `brain*` ecosystem (`brainstate`,
`brainunit`, `brainevent`, `braintools`, `brainpy`).

The library owns seven concerns end-to-end:

1. **Morphology ingestion** — read SWC / ASC / NeuroML2, validate, cache.
2. **Geometry & discretization** — turn a morphology + a CV policy into
   immutable control-volume (CV) arrays suitable for vectorized solvers.
3. **Mechanism declaration** — paint cable properties, density mechanisms,
   and ion channels onto regions; place point mechanisms onto locsets.
4. **Runtime lowering** — initialize `Cell` with resolved ion species,
   channel state, point-mechanism storage, and a DHS-ordered node tree.
5. **Numerical integration** — provide a registry of explicit, implicit,
   exponential, and staggered step functions, including a custom DHS
   voltage solver for branched cables.
6. **Network execution** — connect event sources to Cell-owned synapses,
   schedule delayed delivery, and aggregate samples and sparse events.
7. **Parameterization** — expose selected physical fields through stable,
   unit-aware trainable parameter mappings.

Out of scope (for this iteration): a BrainCell-owned optimizer or Trainer,
plasticity learning rules, trainable topology, NEURON HOC compatibility, GUI
tools, and stand-alone NMODL execution. The previous `mech/nmodl/` research
tree has been removed; if NMODL support returns, it will be a separate codegen
design targeting the mechanism registry.

---

## 2. Top-Level Architecture

[System Overview](../design/architecture/current/system-overview.md) describes module responsibilities,
dependencies, dataflow, state ownership, lifecycle, and execution paths. Cross-module design
questions and next actions are tracked in [Architecture TODO](../design/architecture/TODO.md).

---

## 3. Module Catalogue

Each subsection lists: **purpose · key types · public API surface ·
internal dependencies · status · open work**.

### 3.1 `braincell.morph` — morphology data model

- **Purpose** — owns the canonical in-memory representation of a neuron's
  geometry. Splits cleanly into immutable per-branch geometry (`Branch`)
  and a mutable owning tree (`Morphology`).
- **Key types**
  - `Branch` (frozen dataclass) and typed subclasses `Soma`, `Dendrite`,
    `Axon`, `BasalDendrite`, `ApicalDendrite`, `CustomBranch`.
    Built via `Branch.from_lengths` / `Branch.from_points`.
  - `branch_class_for_type(type_str)` factory used by IO readers.
  - `Morphology` — mutable owning tree, root attachment, attribute-style
    children (`morpho.soma.dendrite = ...`), `topo()` text rendering,
    `branches`, `edges`, `branch_by_order`.
  - `MorphoBranch` — node view exposing parent / children navigation.
  - `MorphoEdge` — frozen, read-only directed edge between two
    `MorphoBranch` nodes.
  - `MorphoMetric` — frozen snapshot of `n_branches`, `total_length`,
    `total_area`, `total_volume`, `max_path_distance`,
    `max_euclidean_distance`, `max_branch_order`, range boxes, etc.
- **Status**
  - [x] Branch geometry, area, volume, point/length constructors.
  - [x] Morphology root construction, `attach`, sugar attribute API,
    topology queries, `topo()` text tree.
  - [x] `Morphology.from_swc` / `Morphology.from_asc` constructors.
  - [x] `save_checkpoint` / `load_checkpoint` (`.bcm` self-contained
    format) plus `pickle` / `copy.deepcopy` support.
  - [x] `MorphoMetric` covering total length / area / volume, branch
    order, path distance, Euclidean distance.
  - [ ] **Tree editing primitives**: delete subtree, splice subtree,
    merge two morphologies at a chosen attachment point, swap a branch
    with another while preserving orientation.
  - [ ] **In-place geometry transforms**: translate / rotate / scale /
    align principal axis, with corresponding metric invalidation.
- **Open risks**
  - Mutability of `Morphology` versus the immutability of `Branch`
    (and downstream caches in `Cell`) makes accidental aliasing easy.
    Tree-edit operations must follow the existing
    `Morphology.clone()` discipline used by `Cell`.

### 3.2 `braincell.io` — file-format ingestion

- **Purpose** — read morphologies from common neuroscience formats and
  produce a `Morphology` plus a structured report describing parsing
  decisions and validation issues.
- **Key types**
  - `swc.SwcReader`, `SwcReadOptions`, `SwcReport`, `SwcIssue` plus
    rulebook (`rules.py`) and soma reconstruction (`soma.py`).
  - `asc.AscReader`, `AscReport`, `AscIssue`, `AscMetadata`.
  - `neuroml2.NeuroMlReader`.
  - `neuromorpho` package — three-tier NeuroMorpho.Org integration:
    - Tier 1: `load_neuromorpho` (also re-exported as
      `braincell.load_neuromorpho`), `fetch_neuromorpho`, and the
      `Morphology.from_neuromorpho` classmethod sibling to `from_swc` /
      `from_asc`.
    - Tier 2: `NeuroMorphoClient` (typed `search` / `iter_search`,
      `get_neuron`, `get_measurement`, `describe`, `download` with
      `dry_run=True`, configurable `retries` / `backoff_base`).
    - Tier 3: `NeuroMorphoCache`, `NeuroMorphoCacheLayout`,
      `NeuroMorphoQuery`, `NeuroMorphoMeasurement`, `NeuroMorphoFilePlan`,
      `NeuroMorphoUrls`, `NeuroMorphoCacheStatus`,
      `NeuroMorphoSearchPage`, `NeuroMorphoDetail`,
      `NeuroMorphoDownloadItem`, `NeuroMorphoDownloadRecord`,
      `NeuroMorphoNeuron`, plus pure URL helpers
      (`build_standard_swc_url`, `build_original_file_url`,
      `infer_original_extension`, `plan_neuron_files`).
    - Errors: `NeuroMorphoError`, `NeuroMorphoHTTPError`,
      `NeuroMorphoNotFoundError`.
  - `io.checkpoint` — `save_branch` / `load_branch` /
    `save_morpho` / `load_morpho` and the `.bcm` single-file format.
- **Status**
  - [x] SWC import + rulebook validation + report.
  - [x] SWC export through `Morphology.to_swc()` / `swc.write_swc()`,
    with structural round-trip coverage for branch endpoint duplication,
    soma interior attachments, reversed branches, and invalid geometry.
  - [~] ASC import: most Neurolucida trees, metadata, and
    `Morphology.from_asc(..., return_report=True)` work; **gaps**:
    spine markers, contour-only somas, and multi-tree files are still
    handled minimally — see `io/asc/reader_test.py` skips.
  - [ ] NeuroML2 import — reader stub exists; needs cell, segment-group,
    biophysics decoding and round-trip tests.
  - [x] NEURON-based diff harness via `examples/neuron_compare/morph/neuron_diff.py`.
  - [x] NeuroMorpho.Org integration: Tier 1 `load_neuromorpho` /
    `fetch_neuromorpho` one-liners, Tier 2 `NeuroMorphoClient` with
    typed `iter_search` / `download` / retries, Tier 3 `NeuroMorphoCache`
    plus pure URL helpers, full NumPy-doc docstrings, and
    `Morphology.from_neuromorpho` classmethod. Notebook walkthrough at
    `examples/multi_compartment/neuromorpho.ipynb` shows the full search → cache →
    metric-diff loop.
  - [ ] Automated metric diff against published NeuroMorpho reference
    statistics promoted from the notebook into a pytest case (so the
    NeuroMorpho corpus becomes a wide regression net).
  - [x] Checkpoint API and `.bcm` format with notebook tutorial
    (`examples/multi_compartment/morphology-checkpoint.ipynb`).
  - [ ] **NMODL parsing compiler** — deferred. The previous
    `mech/nmodl/` research tree has been removed from the working
    copy; if NMODL support returns it will land as a codegen pass
    targeting the mechanism registry (see §3.4 / M5 Phase 4).
- **Open risks**
  - Format heterogeneity is the dominant source of bugs. Every reader
    must produce a `Report` so user-facing tools can surface issues
    instead of silently massaging geometry.

### 3.3 `braincell.filter` — region & locset selection

- **Purpose** — declarative, composable selection of regions of a
  morphology and points on it. The cell layer consumes these to map
  user intent onto control volumes.
- **Key types**
  - `RegionExpr` family: `BranchSlice`, `branch_in(...)` predicates for
    branch metadata / topology, `branch_range(...)` for scalar branch
    properties and metrics, set operations
    (union / intersection / difference / complement).
  - `LocsetExpr` family: root, branch points, terminals, region-driven
    uniform sampling, region-driven random sampling.
  - `SelectionCache` — memoizes resolved index sets for stable
    Morphology objects.
- **Status**
  - [x] BranchSlice, broadcasted inputs, set algebra.
  - [x] Discrete predicates (type / name / branch_order / parent_id /
    n_children / n_tapers / branch_id).
  - [x] Continuous `branch_range(...)` with both numeric and `Quantity`
    bounds.
  - [x] Branch scalar metric filters: `length`, `mean_radius`, `area`,
    `volume`.
  - [ ] **Radius-range filter** (e.g., `radius_range(0.5*u.um, 2*u.um)`).
  - [ ] **Path-distance filter** (graph distance from soma along the
    tree).
  - [ ] **Euclidean-distance filter** (3-D distance from a chosen
    anchor point).
  - [ ] **Subtree region** — everything reachable below a given branch
    or locset; needs to interoperate with the planned
    `Morphology` subtree-edit operations.
  - [x] Locset: root, branch points, terminals.
  - [x] Locset: uniform / random sampling driven by a region.
  - [~] **Locset anchors and fixed-step sampling**: `RegionAnchors` and
    explicit `at(branch, x)` locations are implemented; `StepSamples`
    remains a reserved expression that raises `NotImplementedError`.
- **Open risks**
  - The reserved distance/radius/subtree expressions must reuse the existing
    morphology spatial metrics and `SelectionCache`; they must not introduce
    a second geometry cache with different invalidation semantics.

### 3.4 `braincell.mech` — mechanism declarations

- **Purpose** — strongly-typed, purely-declarative containers used by
  the `Cell` frontend. Everything here describes *what to install*, not
  *how to integrate*: no `brainstate`, no JAX, no runtime state. The
  concrete ion species, ion channels, and synapses live in peer
  top-level modules (`braincell.ion`, `braincell.channel`,
  `braincell.synapse`) and register themselves with the
  `MechanismRegistry` at import time via class-level decorators; the
  runtime lowering in `braincell._compute` resolves a
  `Density.class_name` through the registry when it installs channels
  on a cell.
- **Key files & types**
  - `mech/_base.py` — `Mechanism` marker base class. Every mechanism
    declaration (density or point) inherits from it, so consumers can
    check `isinstance(x, Mechanism)` without having to know whether
    they hold a `Density` or a `Point`.
  - `mech/_registry.py` — `MechanismEntry(category, name, cls,
    aliases)` frozen dataclass, `MechanismRegistry` with
    `register` / `unregister` / `add_alias` / `contains` / `get` /
    `entry` / `names` / `items` / `clear`, the `_REGISTRY` singleton
    accessed via `get_registry()`, and the three class-level
    decorators `register_channel` / `register_ion` /
    `register_synapse`. Unknown-name lookups raise `KeyError` with a
    `difflib`-based "did you mean ...?" suggestion (same pattern as
    `braincell.quad._registry`). Three valid categories:
    `"channel"`, `"ion"`, `"synapse"`.
  - `mech/_params.py` — `Params(Mapping[str, Any])` frozen hashable
    mapping. `__hash__` uses `frozenset(self._items.items())`, so
    `Channel("IL", g_max=..., E=...)` and `Channel("IL", E=...,
    g_max=...)` deduplicate into a single paint-layout group. Iteration
    order is the declared order so `repr()` is stable. Accepts
    `Mapping`, `(k,v)` tuples, or another `Params` in the constructor
    (`Params.coerce(value)`), supports `**params` unpacking via the
    `Mapping` protocol, and exposes non-mutating `with_updates(...)` /
    `without(...)`.
  - `mech/_density.py` — `Density(Mechanism)` abstract base plus the
    concrete subclasses `Channel(Density)` and `Ion(Density)`. `Density`
    is a manually-immutable `__slots__` class (not a dataclass) with a
    `category: ClassVar[str]` discriminator set by each subclass
    (`"channel"` / `"ion"`). The constructor accepts `class_name` as
    either a string **or** a class (`braincell.channel.IL`); types are
    resolved to their canonical registry name via reverse lookup.
    `coverage_area_fraction` is a dedicated first-class field, not a
    pseudo-parameter. `instance_name` falls back to `class_name`,
    `identity = (instance_name, class_name)` drives paint-layout
    grouping, and `with_params(...)` / `with_coverage(...)` /
    `with_name(...)` return non-mutating copies via an internal
    `object.__new__` + `object.__setattr__` bypass. `Channel` and
    `Ion` collect parameters via `**params` kwargs.
  - `mech/_point.py` — `Point(Mechanism)` plain base class (not a
    `Union`; use `isinstance(x, Point)` in consumers) plus concrete
    frozen-dataclass subclasses `CurrentClamp`, `SineClamp`,
    `FunctionClamp`, `ProbeMechanism`, and `Synapse`. `CurrentClamp`
    has one canonical form `(start, durations, amplitudes)` and a
    `CurrentClamp(delay=..., durations=duration, amplitudes=amplitude)` classmethod
    shortcut. `Synapse` is itself a frozen dataclass
    (`synapse_type`, `params`, `name`); there is no separate factory
    function.
  - `mech/_junction.py` — `Junction(Point)` frozen dataclass for
    gap-junction coupling declarations. Placeholder implementation
    (`params` field only); lives in its own module so downstream
    work on gap-junction state and partner wiring has a clean home.
  - `mech/_cable.py` — `CableProperty` frozen dataclass
    (`resting_potential`, `membrane_capacitance`, `axial_resistivity`,
    `temperature`, all `brainunit` quantities; temperature defaults to
    36 °C via a `default_factory` and is coerced to kelvin in
    `__post_init__`). Exposes non-mutating `with_updates(**kwargs)`.
  - `mech/__init__.py` — re-exports the public surface
    (`Mechanism`, `Density`, `Channel`, `Ion`, `Point`, `CurrentClamp`,
    `SineClamp`, `FunctionClamp`, `ProbeMechanism`, `Synapse`,
    `Junction`, `CableProperty`, `Params`, registry API).
  - Co-located tests: `_base_test.py`, `_registry_test.py`,
    `_params_test.py`, `_density_test.py`, `_point_test.py`,
    `_junction_test.py`, `_cable_test.py`.
- **Status**
  - [x] `CableProperty`, `Density` (with `Channel` / `Ion`
    subclasses), and the full `Point` family (`CurrentClamp`,
    `SineClamp`, `FunctionClamp`, `ProbeMechanism`, `Synapse`,
    `Junction`) with `brainunit`-typed fields and co-located tests.
    Everything inherits from a shared `Mechanism` marker base class.
  - [x] **One type per concept.** The legacy `MechanismSpec` /
    `DensityMechanism` duality and the eight `density_*` isinstance-
    dispatch helpers in `spec.py` are gone. Every density declaration
    is a `Density` subclass (`Channel` or `Ion`) carrying a
    `category` `ClassVar`; every point declaration is a `Point`
    subclass.
  - [x] **Class-based `Channel` / `Ion`.** `braincell.mech.Channel`
    and `braincell.mech.Ion` are real classes (not factory functions)
    inheriting from `Density`. They accept the target class as either
    a string name (`"IL"`) or the concrete class object
    (`braincell.channel.IL`); the class form is reverse-looked-up in
    the registry to produce the canonical name so aliases continue to
    collapse into one identity. Top-level `braincell.Channel` /
    `braincell.Ion` still point at the runtime base classes from
    `_base_channel.py` / `_base_ion.py`; the declaration-layer classes are
    reached via
    `braincell.mech.Channel` / `braincell.mech.Ion` to avoid the
    name collision.
  - [x] **Mechanism registry.** `MechanismRegistry` + the
    `@register_channel` / `@register_ion` / `@register_synapse`
    decorators ship in `mech/_registry.py`. ~49 concrete classes in
    `braincell.channel`, `braincell.ion`, and `braincell.synapse`
    self-register at import time. `get_registry().get(category,
    class_name)` is the single lookup path used by
    `_compute/parameters.py` and `_compute/bindings.py` to resolve
    `Density.class_name` into a
    runtime class. Channel-to-ion binding is inferred from
    `issubclass(cls.root_type, Sodium / Potassium / Calcium)`, not
    from hardcoded class-name matching. Abstract base classes
    (`LeakageChannel`, `SodiumChannel`, `Calcium`, …) are deliberately
    **not** decorated.
  - [x] **Hash-stable Params.** `Params.__hash__` uses
    `frozenset(items)` so two `Channel(...)` calls with the same
    parameters in different keyword order compare equal and
    deduplicate into the same paint-layout group. Only `params` is
    hash-insensitive; `class_name`, `name`, `category`, and
    `coverage_area_fraction` remain position-sensitive.
  - [x] **`coverage_area_fraction` as a first-class field** on
    `Density`. The old abstraction leak where coverage was smuggled
    through ordinary mechanism parameters is gone; `_discretization`
    and `_compute` preserve it as geometry metadata.
  - [x] **Unified `CurrentClamp`.** One canonical frozen-dataclass
    form `(delay, durations, amplitudes)`. The old
    `CurrentClamp(amplitude=, delay=, duration=)` compatibility form
    is gone; use `CurrentClamp(delay=..., durations=duration, amplitudes=amplitude)`.
  - [x] **Consumer simplification.** `_discretization/mechanism.py` and
    the `_compute` layout, binding, parameter, and table modules operate
    directly on the declaration types without a parallel spec hierarchy.
  - [ ] **Parameter-unit validation** — `Params` currently stores
    values untyped. Needs compile-time validation that each value
    carries the brainunit dimension the target channel declares
    (e.g. `g_max` must be in `S/cm²`, `E` in `mV`), with an error
    that points at the offending `paint(...)` call. The infrastructure
    for this lives on the mechanism registry: each entry can declare
    the expected unit per parameter name.
  - [ ] **`Junction` runtime wiring** — `Junction` currently ships
    as a placeholder frozen dataclass with only a `params` field.
    It needs a `partner` reference (locset or another placed
    `Junction`), symmetric pair resolution in the runtime, and a
    gap-junction current contribution in the voltage solve. Tracked
    as the first sub-task in milestone M5 Phase 3.
  - [ ] **`ProbeMechanism` variable taxonomy** — `variable` is
    currently a free-form string. Promote it to a typed enum of known
    probes (`"v"`, `"ina"`, `"ik"`, `"ica"`, `"cai"`, `"cao"`,
    channel gate names, …) so user typos fail at declaration time
    rather than silently producing empty traces.
  - [ ] **Mechanism validation harness** — a structured comparison
    against NEURON `.mod` reference traces for every channel in
    `braincell.channel`. The previous `mech/mod_validate/` tree has
    been removed from the working copy; the harness needs to be
    re-introduced as a package under `braincell/mech/` (or a sibling
    test package) and promoted to automated pytest cases. Tracked in
    milestone M5.
  - [ ] **NMODL ingestion** — deferred. If NMODL support returns it
    must target the mechanism registry so generated channels land
    under the standard naming convention in `braincell.channel`
    rather than creating a parallel hierarchy.
- **Open risks**
  - **Hash-insensitive `Params` equality** only kicks in for the
    `params` field; `class_name`, `name`, `category`, and
    `coverage_area_fraction` stay position-sensitive. Do not extend
    the hash-insensitive treatment to other fields without first
    understanding the paint-layout grouping contract in
    `_discretization/mechanism.py`.
  - **Class-level decorator ordering.** Registration is a side
    effect of importing `braincell.channel` / `braincell.ion` /
    `braincell.synapse`. If a user imports `braincell.mech` alone
    (without importing the concrete modules) the registry is empty —
    by design. The canonical entry points in `braincell/__init__.py`
    already import all three, so normal users never see this.
  - **Ion binding inference** uses
    `issubclass(cls.root_type, Sodium/Potassium/Calcium)` in
    `_compute/bindings.py`. New ion species must either set
    `root_type` on their channels or we extend the dispatch to walk
    a lookup table — do not hardcode class-name matching.
  - **Name collision with runtime `Channel` / `Ion` bases.** The
    declaration-layer `Channel` / `Ion` classes live under
    `braincell.mech`, not at the top level of `braincell`, because
    `braincell.Channel` / `braincell.Ion` already resolve to the
    runtime base classes from `_base_channel.py` / `_base_ion.py`.
    Tutorials and user code
    should use the fully-qualified `braincell.mech.Channel` /
    `braincell.mech.Ion` when declaring mechanisms on a `Cell`.
  - The module is intentionally free of `brainstate` / JAX state —
    keeping `mech` purely declarative makes importing `braincell.mech`
    cheap and keeps the declaration frontend usable even in
    environments where the numerical runtime is absent. Do not
    import `brainstate`, `jax`, or any concrete channel/ion/synapse
    class inside `braincell/mech/`. The one permitted dynamic
    import is inside `_density._resolve_class_name`, which consults
    the registry via a lazy `from ._registry import get_registry`
    local import when a user passes a class object instead of a
    name string.

### 3.5 `_discretization` / `_compute` / `_multi_compartment` — Cell runtime

- **Purpose** — turn *(Morphology, CVPolicy, paint/place declarations)*
  into an initialized, directly runnable `Cell(HHTypedNeuron)`:
  - `braincell._discretization` owns immutable CV geometry, policies,
    mechanism rules, `CVTree`, and declaration-time `NodeTree` data.
  - `braincell._compute` owns runtime layouts, bindings, CV/point bridges,
    scheduling, tables, and `CellRuntimeState`.
  - `braincell._multi_compartment` owns `Cell`, its spatial and mechanism
    views, clamps, synapses, probes, and `RunResult`.
- **Status**
  - [x] `Cell(morpho, pop_size=..., cv_policy=...)`, `paint`, and `place`
    form the declaration phase; declarations freeze after initialization.
  - [x] `Cell.init_state()` lowers the declaration and installs runtime
    state on the same object. `Cell.run(dt=..., duration=...)` advances it
    directly; there is no public build phase or `RunnableCell`.
  - [x] CV policies, geometry, axial-resistance partitioning, mechanism
    lowering, point topology, DHS scheduling, and CV/point conversion.
  - [x] Homogeneous populations with mandatory population axes and
    multi-dimensional `pop_size`.
  - [x] Cell, Channel, Ion, Synapse and Clamp views with Cell-owned
    connection, recording, and trainable-parameter storage.
  - [x] Fixed-step clamps retain their exact continuous interval at runtime;
    density parameters are materialized on CVs rather than non-CV points.
  - [x] NEURON-compatible ion-current snapshots and selectable
    `"family"` / `"integration"` ion-channel update ordering.
  - [ ] **SingleCompartment ODE unification.** Design discussion; no new
    single mode is implemented. Local decisions, open questions, and next
    steps are tracked in [Cell TODO](../design/cell/TODO.md).
- **Open risks**
  - Declaration shapes and ownership must remain fixed after
    `init_state()` so JIT state trees and network routing stay stable.
  - Parameter materialization may change values without changing runtime
    layout, topology, units, or state shape.

### 3.6 `braincell.quad` — numerical integrators

- **Purpose** — provide a uniform registry of step functions over
  `DiffEqModule` targets, plus the specialized branched-cable voltage
  solver.
- **Key types**
  - `IntegratorRegistry`, `IntegratorEntry`, `register_integrator`,
    `get_registry`, `get_integrator`. Decorator-based registration with
    canonical name, aliases, category, order, description, deprecation.
  - `_RegistryDictView` exposes a read-only `all_integrators` mapping
    for legacy callers.
  - `DiffEqModule`, `DiffEqState`, `IndependentIntegration` —
    structural protocols and helpers for step functions.
  - **Explicit families**: `euler_step`, `rk2/3/4_step`, `heun2/3_step`,
    `midpoint_step`, `ralston2/3/4_step`, `ssprk3_step`.
  - **Implicit / mixed**: `backward_euler_step`, `implicit_euler_step`.
  - **Exponential Euler**: `exp_euler_step`, `ind_exp_euler_step`.
  - **Staggered**: `staggered_step` (DHS voltage solve +
    `ind_exp_euler` for ion-channel state, the workhorse for full
    cells).
  - **Voltage solvers**: `dhs_voltage_step` (DHS branched Hines),
    `dense_voltage_step`, `sparse_voltage_step`.
- **Status**
  - [x] Registry, alias resolution, "did you mean ...?" suggestions.
  - [x] Backwards-compatible `all_integrators` mapping view.
  - [x] All explicit RK / Heun / Ralston / Midpoint / SSPRK families.
  - [x] Backward Euler and implicit Euler. The six cell-only variants
    (`implicit_rk4`, `implicit_exp_euler`, `cn_rk4`, `cn_exp_euler`,
    `exp_exp_euler`, `splitting`) were removed: they had rotted against
    several `brainstate` / `Cell` API generations and none could be
    invoked successfully. `braincell/quad/_implicit_test.py` pins their
    absence from the registry.
  - [x] Exponential Euler (`exp_euler_step`, `ind_exp_euler_step`).
  - [x] Staggered solver (`staggered_step`).
  - [x] The staggered full-cell path calls
    `cache_ion_total_currents(...)` when the target supports it, so
    NEURON-compatible ion-current snapshot semantics can be selected at
    the `Cell` level without changing the integrator API.
  - [x] DHS voltage solver (`dhs_voltage_step`).
  - [ ] **Adaptive timestep wrapper** that produces a registered
    integrator from any embedded RK pair.
  - [x] **Convergence test matrix** — pytest-driven order-of-accuracy
    checks for every registered integrator on a small set of
    reference ODEs (passive cable, single HH spike, two-branch Y).
  - [ ] **Performance benchmarks** vs NEURON / Arbor on the standard
    Mainen / Hay / L5PC cells, run nightly via `CI-daily.yml`.

### 3.7 `braincell.vis` — visualization

`braincell.vis` 已提供 2D/3D 形态图、数值着色、轨迹与动画、拓扑分析、交互和导出。
当前讨论将可视化迁入 braintools，由同一模块提供简单绘图和 GUI 两类入口；下一步确定
模块归属、共享数据接口及依赖适配方式。

- 模块事项与下一步：[Vis TODO](../design/vis/TODO.md)。
- 已有功能、后端差异与验证现状：[Visualization](../design/vis/current/visualization.md)。
- 迁移方案与准备工作：[Braintools Migration](../design/vis/proposals/braintools-migration.md)。

### 3.8 `braincell.ion` — ion species

- **Purpose** — concrete `Ion` subclasses modelling intra/extracellular
  concentration, reversal potential, and the container of ion-bearing
  channels that consume the species' `IonInfo`. Lives as a peer
  top-level module (not under `mech`) because the classes are runtime
  objects with JAX state, not declarations.
- **Key files & types**
  - `braincell/ion/_base.py` — reusable `FixedIon`, `InitNernstIon`,
    `DynamicNernstIon`, and `KineticIon` lifecycle templates.
  - `braincell/ion/sodium.py` — `Sodium` (abstract base with
    `root_type = HHTypedNeuron`), `SodiumFixed`, and `SodiumInitNernst`.
  - `braincell/ion/potassium.py` — `Potassium` abstract base and
    fixed and initialized-Nernst variants.
  - `braincell/ion/calcium.py` — `Calcium` base class,
    fixed/initialized-Nernst variants, and two concrete dynamics models:
    - `CalciumDetailed` — Destexhe et al. 1993 thin-shell model with
      tunable `d`, `tau`, `C_rest`, `C0`, `T`.
    - `CalciumFirstOrder` — Bazhenov et al. 1998 first-order pool
      (`Ca' = α I_Ca − β Ca`).
      Both expose `C` as a `DiffEqState`, compute the Nernst reversal
      `E = (RT/2F) log(C0/C)` as a property, and forward
      `compute_derivative` to every attached `Channel` child.
  - Co-located tests: `sodium_test.py`, `potassium_test.py`,
    `calcium_test.py`.
- **Status**
  - [x] `SodiumFixed` / `PotassiumFixed` / `CalciumFixed` parameter
    storage, container (`**channels`) attachment, and `pack_info()`
    returning an `IonInfo(C, E)` tuple.
  - [x] `CalciumDetailed` / `CalciumFirstOrder` with Nernst reversal
    and full derivative wiring to child calcium channels.
  - [x] `KineticIon`-based Cerebellum calcium-pool mechanisms imported
    for the current comparison work, including `CdpStC_MA2020_GoC`,
    `CdpStC_NoCAM_MA2020_GoC`, `CdpStC_CAMOnly_MA2020_GoC`,
    `CdpStC_MA2025_BC`, `CdpStC_RI2021_SC`, `CdpCAM_MA2024_PC`, and
    `CdpCR_MA2020_GrC`.
  - [x] Co-located unit tests (~75) covering defaults, custom
    parameters, callable broadcasts, `init_state` /
    `reset_state` / `compute_derivative`, `pack_info`,
    external-current registration, Nernst formula edge cases, and
    child-channel forwarding.
  - [ ] **`SodiumDetailed` / `SodiumFirstOrder`** — activity-
    dependent Na⁺ accumulation (e.g., for spike-frequency adaptation
    driven by a Na/K pump). Parallel to the calcium dynamics pair
    and needed to reproduce several of the published cortical
    models in `examples/`.
  - [ ] **`PotassiumDetailed` / `PotassiumFirstOrder`** — activity-
    dependent intracellular / extracellular K⁺ accumulation for
    network-level effects and K-pump dynamics, with the same
    Nernst-reversal property as the calcium path.
  - [ ] **`Chloride` ion** (`Chloride`, `ChlorideFixed`,
    `ChlorideDynamics`) in a new `braincell/ion/chloride.py` plus a
    sibling `chloride_test.py`. Needed for quantitative GABAa
    modelling and developmental E_Cl shifts.
  - [x] **Shared ion lifecycle templates** — package-private `FixedIon`,
    `InitNernstIon`, `DynamicNernstIon`, and `KineticIon` mixins own the
    reusable initialization, reversal, and kinetics contracts.
  - [x] **`__init__.py` hygiene** — ion and channel re-export sets are
    explicit, deduplicated, and guarded by package-level re-export tests.
  - [x] **Mechanism-registry plumbing** — every concrete `Ion`
    subclass now self-registers via `@register_ion("CalciumFixed")` /
    `@register_ion("CalciumDetailed")` / `@register_ion("CalciumFirstOrder")` /
    `@register_ion("SodiumFixed")` / `@register_ion("PotassiumFixed")`
    at import time, and `braincell.mech.Ion("CalciumFixed")` resolves
    through the registry described in §3.4.
  - [x] **Current-driven ion dynamics can use cached ion current.**
    Kinetic ions that consume total calcium current can receive the
    runtime snapshot created by `cache_ion_total_current=True`, matching
    the NEURON-style separation between channel-current evaluation and
    ion-state integration.
  - [ ] **Consistent external-current registration** — audit that
    every dynamics class honours `include_external=True` in its
    `derivative` (the existing `CalciumDetailed.derivative` already
    does; the contract must stay alive across future refactors).
- **Open risks**
  - **Nernst unit trap.** Nernst factors resolve correctly only when every
    term remains a `brainunit` quantity; changes to the shared ion templates
    must preserve units through graph flattening and materialization.
  - **Shared lifecycle contracts.** New ion families must use the common
    template hooks and contract tests so child-channel reset and derivative
    forwarding cannot diverge by species.
  - **Test-side coupling with `braincell.channel`.** The calcium
    tests instantiate `CaT_HM1992` to exercise child-channel
    forwarding, so a heavy top-level import in `braincell.channel`
    would drag through the ion suite. Keep the channel package
    tree-shakable (see §3.9 risks).

### 3.9 `braincell.channel` — concrete ion channels

- **Purpose** — the library's catalogue of ready-to-use HH-style and
  Markov-kinetics ion channels. Every class is a subclass of
  `Channel` from `_base_channel.py` (so every instance is an `IonChannel`
  that registers its gate state as `DiffEqState`s) and declares
  `root_type = HHTypedNeuron`. Channels are container children of
  an `Ion` species or of a `SingleCompartment` / `Cell` directly.
- **Key families**
  - `sodium.py` — `Na_Ba2002`, `Na_TM1991`, `Na_HH1952`, persistent,
    resurgent, and cell-specific Nav families.
  - `potassium.py` — delayed rectifier, A-type, inward rectifier, Kv,
    and M-current families such as `KDR_Ba2002`, `K_HH1952`, and the
    MA2020/MA2024 cell-specific variants.
  - `calcium.py` — T/L/HVA/LVA and Cav families, including frozen-gradient
    variants used by controlled NEURON comparisons.
  - `braincell/channel/leaky.py` — `LeakageChannel` base and the
    passive leak `IL`.
  - `hyperpolarization_activated.py`, `potassium_calcium.py`, and
    `potassium_sodium.py` — HCN and mixed-ion channel families.
- **Status**
  - [x] Concrete channel families use current-free mechanism names such as
    `Na_HH1952`, `K_HH1952`, `CaT_HM1992`, and `HCN_HM1992`; the removed
    leading-`I` compatibility aliases are not public API.
  - [x] Co-located tests cover kinetics, current sign and shape, lifecycle,
    template invariants, and representative reference voltages.
  - [x] Concrete classes self-register with the mechanism registry at import
    time; abstract family bases are deliberately not registered.
  - [x] **PC MA2024 channel set imported.** Sodium, potassium,
    calcium, calcium-activated potassium, and HCN PC variants have been
    added and covered by targeted tests. The calcium channel set also
    includes `_Frozen` variants for the NEURON-comparison path where the
    current expression must treat voltage as fixed with respect to
    differentiation.
  - [ ] **Parameter metadata** — each channel should declare the
    unit of every user-facing parameter (`g_max` in `S/cm²`, `E` in
    `mV`, time constants in `ms`, …) so that `Density.params`
    validation can produce an actionable error at paint time rather
    than an opaque JAX trace failure. Store the per-parameter unit
    on `MechanismEntry.metadata` and consult it during
    `Density.__init__`.
  - [~] **GHK current formulation** — `GhkHH` and `ghk_flux` are implemented,
    tested, and used by selected Cav channels; the remaining work is a
    catalogue-wide audit of which published mechanisms require GHK rather
    than an ohmic driving force.
  - [~] **Q10 temperature scaling audit** — shared `q10_factor` and
    `cached_q10_factor` helpers exist and most gates use the template path;
    remaining family-specific temperature assumptions need documentation.
  - [ ] **NEURON `.mod` validation** — for every channel in the
    catalogue, compare voltage-clamp and current-clamp traces
    against the reference `.mod` implementation within a tight
    tolerance. Requires re-introducing the `mech/mod_validate/`
    harness (see §3.4) and wiring it into milestone M5.
  - [ ] **Chloride channels** — add a `braincell/channel/chloride.py`
    module once `braincell.ion.Chloride` lands, covering the passive
    leak plus GABAa-reversal-driven phasic conductance.
  - [ ] **Stiff-channel integrator audit** — run the convergence matrix
    over every channel to identify models that require a dedicated
    integration path.
  - [ ] **Gate-variable naming convention** — most channels use
    `p`/`q` for activation / inactivation and a handful use bespoke
    names (`m`, `h`, `n`, `s`, …). Tests already rely on the
    `p`/`q` convention; unifying the rest will need a deprecation
    path because downstream code reaches into `channel.p.value`.
- **Open risks**
  - **Import cost.** The package has thirty-plus classes and pulls
    `braintools.init`, `brainunit`, and `jax.numpy` at import time.
    New families should stay in their own module so the package
    remains tree-shakable, and should avoid importing numpy at
    module top level beyond what is already there.
  - **Cross-ion channels.** `potassium_calcium.py` channels depend
    on the attached calcium pool's `C` state. Compile-time checks
    that the parent `Cell` actually has a calcium ion attached would
    prevent silent `KeyError` / `AttributeError` at simulate time;
    this belongs on the mechanism registry in §3.4.
  - **API drift vs NEURON naming.** Upstream `.mod` files use lowercase
    suffixes (`ih`, `ik`, `ikdr`), while BrainCell names mechanisms by
    family/model and provenance. Any validation harness
    needs a stable alias table so the diff does not become a
    renaming exercise every time a new channel lands.

### 3.10 `braincell` package root — neuron base classes

- `_base_neuron.py`, `_base_ion.py`, and `_base_channel.py` define the
  runtime bases composed by concrete neurons and mechanisms.
- `_single_compartment/` owns `SingleCompartment`, the simplest concrete
  neuron and a numerical sanity surface.
- `_multi_compartment/` owns the directly initialized and executed `Cell`,
  its views, point-mechanism stores, probes, and `RunResult` (see §3.5).
- `_misc.py` — `normalize_param` (the brainunit gatekeeper), helpers,
  decorators (`set_module_as`, `deprecation_getattr`), `Container`.
- `_typing.py` — type aliases (`Initializer`, `ArrayLike`, `T`, `DT`).

### 3.11 `braincell.network` — population and event runtime

- **Purpose** — register Cells and event sources, connect source outputs to
  Cell-owned synapses, coordinate lifecycle and delayed delivery, and
  aggregate immutable sample and sparse-event results.
- **Status**
  - [x] Direct `Network`, `Population`, `NetworkConnections`, and
    `NetworkResult` model with no separate public build phase.
  - [x] Named connection calls, explicit or sampled endpoint pairing,
    heterogeneous delays, split runs, reset semantics, and cached schedules.
  - [x] Static recording schemas with regular `SampleBlock` outputs and
    sparse `EventSeries` outputs.
  - [ ] Runtime scalability and topology extensions; local questions and
    acceptance boundaries are tracked in [Network TODO](../design/network/TODO.md).
- **Design authority** — [Network TODO](../design/network/TODO.md)
  and its linked current contracts, proposals, and references.

### 3.12 `braincell.trainable` — parameter ownership and mapping

- **Purpose** — bind optimizer-facing parameter roots to selected physical
  runtime fields while preserving units, sharing semantics, and stable JAX
  state trees. It does not own optimizers, losses, datasets, or training loops.
- **Status**
  - [x] `ParameterSource`, `ParameterBinding`, `ParameterSet`, and
    `TrainableManager`, plus direct, shared-scale, and callable latent sources.
  - [x] Cell-local ChannelView mappings for the initial supported channel
    families, with transactional validation and differentiable materialization.
  - [ ] Ion, Synapse and Connection parameters, Network aggregation, and
    broader parameter families.
- **Design authority** — [Optimization TODO](../design/optim/TODO.md)
  and its linked current contracts, proposals, and references. Its working-tree
  capabilities do not change this index's committed-state milestones.

---

## 4. Cross-Cutting Concerns

| Concern | Maintained reference |
| --- | --- |
| Units, state ownership, and structural freeze | [System Overview](../design/architecture/current/system-overview.md#数据归属与生命周期), [units and dependencies](../design/architecture/current/system-overview.md#单位参数与外部依赖) |
| Cell initialization and reset | [Cell lifecycle](../design/cell/current/api.md#生命周期) |
| Network execution and event timing | [Network architecture](../design/network/current/architecture.md) |
| Trainable parameters and runtime materialization | [Trainable architecture](../design/optim/current/architecture.md) |
| Test layout and shared fixtures | [Repository testing conventions](../../AGENTS.md#testing) |
| Design documents and interface specifications | [Design conventions](../design/AGENTS.md) |

---

## 5. Data-Model Summary

The [ownership table](../design/architecture/current/system-overview.md#数据归属与生命周期)
tracks declarations, geometry, runtime states, connections, recordings, and parameter roots.
[State axes and spatial mappings](../design/architecture/current/system-overview.md#状态轴与空间映射)
explain how population, CV, point, and recording rows relate.

---

## 6. Public API Contract

Public entry points are exported by [braincell/__init__.py](../../braincell/__init__.py)
and the domain packages' explicit exports. Internal implementation paths can host public
objects re-exported from these entry points.

Use the [module documentation map](../design/architecture/current/system-overview.md#模块与核心对象)
for signatures, units, shapes, return values, and lifecycle requirements. Cross-module naming
and export questions are compared in the [interface proposal](../design/architecture/proposals/interface-consistency.md).

---

## 7. End-to-End User Workflows

The [runnable system example](../design/architecture/current/system-overview.md#从声明到结果)
constructs a morphology and passive Cell, connects an event source, reads recordings,
plots Cell topology and traces, and performs one parameter update.

Detailed workflows: [Cell](../design/cell/current/api.md), [Network](../design/network/current/api.md),
[Trainable](../design/optim/current/api.md), and [Vis](../design/vis/current/api.md).

---

## 8. External Dependencies

[pyproject.toml](../../pyproject.toml) defines dependency floors and extras.
JAX and SciPy remain unpinned there; the supported JAX floor and tested versions are
maintained by the [repository compatibility convention](../../AGENTS.md#working-agreement)
and [daily CI matrix](../../.github/workflows/CI-daily.yml).
Dependency roles and optional visualization backends are described in the
[system overview](../design/architecture/current/system-overview.md#单位参数与外部依赖).

| Issue | Status | Next action |
| --- | --- | --- |
| Python version coverage | Pending | Classifiers advertise 3.11 through 3.14, while [CI](../../.github/workflows/CI.yml) and daily CI test only 3.13. Expand the matrix or align the advertised support. |

---

## 9. Glossary

CV, point, population, and recording rows are defined in
[state axes and spatial mappings](../design/architecture/current/system-overview.md#状态轴与空间映射).
Paint/place are illustrated in the [declaration example](../design/architecture/current/system-overview.md#构造-cell-和事件输入);
staggered and DHS are described in [execution paths](../design/architecture/current/system-overview.md#时间推进与求解路径).
Morphology checkpoint format details live with [IO](../design/io/TODO.md).
