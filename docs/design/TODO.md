# BrainCell Project Design and TODO

> Status: living document. Tracks both the architectural intent of the
> `braincell` package and the current implementation state of every major
> subsystem. Status markers in this file follow:
>
> - `[x]` shipped — implemented, covered by `*_test.py`, and documented at
>   its intended public or internal surface.
> - `[~]` partial — implementation exists but is missing functionality,
>   tests, or runtime integration. Specific gaps are listed inline.
> - `[ ]` planned — design agreed, code not yet written.
>
> This document describes committed repository state. Experimental work in an
> uncommitted working tree does not become a shipped capability until its API,
> implementation, and tests land together.

---

## Document Navigation

- [Mission and scope](#1-mission-and-scope)
- [Top-level architecture](#2-top-level-architecture)
- [Module catalogue](#3-module-catalogue)
- [Cross-cutting concerns](#4-cross-cutting-concerns)
- [Public API contract](#6-public-api-contract)
- [End-to-end workflows](#7-end-to-end-user-workflows)

Detailed network and parameter-training contracts live in
[`network/`](network/design-overview.md) and [`optim/`](optim/design-overview.md).
Those topic directories are authoritative when their details are more specific
than this project-level summary.

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

```
┌──────────────────────────────────────────────────────────────────────┐
│                         braincell.io                                 │
│   SWC / ASC / NeuroML2 readers · checkpoints · NeuroMorpho client    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ Morphology
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       braincell.morph                                │
│           Branch (frozen) · Morphology (mutable tree)                │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │ Morphology                        │
               ▼                                   ▼
┌─────────────────────────────┐    ┌────────────────────────────────────┐
│      braincell.filter       │    │           braincell.mech           │
│  RegionExpr · LocsetExpr    │    │  Mechanism · CableProperty         │
│  SelectionCache             │    │  Density · Point · Junction        │
│                             │    │  MechanismRegistry                 │
└──────────────┬──────────────┘    └─────────────────┬──────────────────┘
               │ selection                           │ declarations
               ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ braincell._discretization   │        braincell._compute              │
│ CV/CVTree · policies      │ layouts · bindings · scheduling  │
│ geometry · mechanism rules│ CellRuntimeState · bridge · table │
├─────────────────────────────┴────────────────────────────────────────┤
│                braincell._multi_compartment (Cell)                   │
│ declaration + initialized runtime · views · run · recording           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HHTypedNeuron
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        braincell.quad                                │
│   IntegratorRegistry · explicit / implicit / exp_euler / staggered   │
│   steps · dhs_voltage_step (branched-cable Hines solver)             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ DiffEqState
                               ▼
                  brainstate / JAX execution
```

```
┌──────────────────────────────────────────────────────────────────────┐
│        braincell.ion · braincell.channel · braincell.synapse         │
│   concrete Ion species (Na, K, Ca) · IonChannel implementations      │
│   (Na, K, Ca, Ih, K_Ca, leaky) · exponential synapse models          │
└──────────────────────────────────────────────────────────────────────┘
   (supply concrete mechanism objects consumed by mech.Density /
    mech.Point declarations and installed inside braincell.Cell)
```

```
┌──────────────────────────────────────────────────────────────────────┐
│                         braincell.vis                                │
│   2D / 3D scenes · matplotlib & PyVista backends ·                   │
│   region / locset / value overlays                                   │
└──────────────────────────────────────────────────────────────────────┘
       (consumes Branch / Morphology / Cell / RegionExpr / LocsetExpr)
```

The directional rule of thumb:
**`io -> morph -> {filter, mech} -> _discretization -> _compute -> Cell -> quad`**,
with `ion` / `channel` / `synapse` as peer top-level modules supplying concrete
mechanism implementations that `mech` wraps into `Density` (`Channel` /
`Ion`) and `Point` (`CurrentClamp`, `Synapse`, `Junction`, …) declarations
at paint/place time. `network` orchestrates initialized Cells and event
sources; `trainable` binds parameter sources into Cell-owned runtime fields;
`vis` reads anything from `morph` upward. Shared runtime bases live in
`_base_neuron`, `_base_ion`, and `_base_channel`.

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

- **Purpose** — render morphologies and cell-level data with both an
  interactive 3D backend (PyVista) and a static / publication 2D
  backend (matplotlib), plus a dependency-light Plotly backend for
  interactive notebook 3D without VTK.
- **Key types and files**
  - `scene.py` — frozen dataclass primitives (`Polyline2D`, `Polygon2D`,
    `Circle2D`, `Label2D`, `BranchPolyline3D`, `BranchTypeBatch3D`),
    `RenderScene2D` / `RenderScene3D` containers, `RenderRequest`,
    `OverlaySpec`.
  - `scene2d.py`, `scene3d.py` — scene builders that strip brainunits
    (`.to_decimal(u.um)`) and translate morphology + layout into
    primitive tuples.
  - `plot2d.py`, `plot3d.py` — high-level user entry points.
  - `backend.py` — `RenderBackend` Protocol + `BackendChooser`.
  - `backend_matplotlib.py`, `backend_pyvista.py`, `backend_plotly.py` —
    concrete backends with lazy optional imports. The matplotlib
    backend attaches per-artist pick metadata; the PyVista backend
    attaches a point→branch lookup so `enable_point_picking` can
    resolve clicks.
  - `hooks.py` — `VisHooks(on_pick=..., on_hover=..., on_leave=...)`
    plus the `PickInfo` payload delivered to user callbacks
    (backend-agnostic; wired in both matplotlib and PyVista).
  - `export.py` — unified `save_figure(figure, path, dpi=..., transparent=...)`
    that dispatches on matplotlib `Axes`/`Figure`, pyvista `Plotter`,
    or plotly `Figure`.
  - `compare.py` — generalized `compare_morphologies([m1, m2, ...])` and
    `compare_values(morpho, [values_a, values_b, ...])` side-by-side
    helpers built on top of `plot2d`.
  - `pytest-benchmark` baselines for layout build, scene build, and
    end-to-end plot2d render on 50 / 500 / 2000-branch synthetic
    morphologies, skipped when `pytest-benchmark` is not installed.
    Filed with the module each measures: `layout/_dispatch_test.py`,
    `scene2d_test.py`, `plot2d_test.py`.
  - `layout/` — 2D tree-layout engine split across
    `_common.py` (shared dataclasses + tree helpers),
    `_geometry.py` (pure-numeric sampling and branch construction),
    `_collision.py` (spatial-hash collision scoring),
    `_config.py` (`LayoutConfig` frozen dataclass, the tunable
    knobs), `_cache.py` (`LayoutCache` LRU keyed on a morphology
    snapshot plus the layout config), `_stem.py` / `_balloon.py` /
    `_radial.py` / `_legacy.py` (layout families), and `_dispatch.py`
    (`build_layout_branches_2d` entry point, cache-aware). Each file
    ships with a sibling `*_test.py`.
  - `compare2d.py` — side-by-side comparison of layout families on the
    same morphology (legacy, specific to layout-family gallery).
  - `config.py` — `VisDefaults` dataclass singleton plus
    `configure_defaults` / `get_defaults` / `reset_defaults`,
    `theme(**overrides)` scoped context manager, and
    `PublicationTheme` / `publication_theme()` which flips both vis
    defaults and matplotlib `rcParams` for LaTeX-friendly output.
  - `_values.py` — colour-by-values normalisation (per-branch /
    per-segment / per-centerline-point → per-point scalar arrays)
    plus :mod:`brainunit` unit-label extraction.
  - `movie.py` — `plot_movie` time-varying colour-by-values
    animation (matplotlib `FuncAnimation` + pyvista
    `Plotter.open_movie`).
  - `traces.py` — `plot_traces` morphology-synchronized time-series
    panels.
  - `morphometry.py` — `plot_dendrogram`, `plot_topology`,
    `plot_sholl`, `plot_branch_order_histogram`, and the
    `compute_sholl_profile` / `ShollProfile` helpers.
  - `_testing.py` — shared morphology builders, the `FakeBackend`
    scene-capturing double, `VisDefaultsResetMixin`, and the
    `PYTEST_BENCHMARK_AVAILABLE` plugin probe.
- **Status**
  - [x] 3D rendering of `Branch` / `Morphology` with point geometry,
    scene composition, PyVista backend.
  - [x] 2D projected mode driven by real points.
  - [x] 2D tree auto-layout.
  - [x] 2D frustum auto-layout.
  - [x] Stem / balloon / radial360 layout family with matplotlib
    comparison output.
  - [x] `OverlaySpec` plumbed end-to-end for `region` / `locset` /
    `values`, with per-CV value colormaps, locset scatter markers,
    and region recolor passes consumed by both backends.
  - [x] `RenderRequest` uses a neutral `backend_options` mapping;
    backend-specific kwargs no longer pollute the shared schema.
  - [x] Backend capability registry via `supported_scene_kinds:
    frozenset[str]` so a future backend can declare multi-format
    support.
  - [x] `plot3d(mode="skeleton")` fast-preview path (centerline-only,
    no tube generation) alongside the default `"geometry"` mode.
  - [x] `RenderScene2D.draw_order` honored by the matplotlib backend
    (primitives sorted by draw_order → `zorder=` argument).
  - [x] `braincell.vis.theme(**overrides)` context manager for scoped
    style overrides; tests no longer need manual `reset_defaults()`.
  - [x] Shared `vis/_testing.py` helpers and parametrized layout-family
    tests covering the shared invariants across stem / balloon /
    radial_360.
  - [x] **`layout2d.py` refactor** into `vis/layout/` with separate
    files for `_common.py`, `_dispatch.py`, `_stem.py`, `_balloon.py`,
    `_radial.py`, `_legacy.py`, `_collision.py`, `_geometry.py`,
    and a `_config.py` holding the `LayoutConfig` frozen dataclass
    (M6 Phase 2). The legacy family now emits a `DeprecationWarning`,
    the collision backend uses a 2D spatial hash, and `plot2d`
    accepts `layout_config=` as an optional user knob.
  - [x] **Color-by-values** for 2D and 3D scenes: accept per-branch /
    per-segment / per-centerline-point scalars. The matplotlib
    backend uses vectorized `LineCollection` / `PolyCollection`
    (10–50× speedup on dense scenes), the PyVista backend writes
    `polydata.point_data["values"]` and calls
    `add_mesh(scalars=..., cmap=..., scalar_bar_args=...)`. Proper
    colorbars with unit labels, plus `vmin` / `vmax` / `cmap` /
    `norm` surfaced through `plot2d` / `plot3d` (M6 Phase 3).
  - [x] **`plot_movie`** — time-varying values over a morphology
    using matplotlib `FuncAnimation` (2D) or
    `pyvista.Plotter.open_movie` (3D). The 2D path builds the scene
    once and mutates the `LineCollection` / `PolyCollection` scalar
    array per frame; the 3D path rewrites
    `polydata.point_data["values"]` and writes one frame per
    timestep.
  - [x] **`plot_traces`** — stacked time-series panels at `locset`
    locations, color-synced with markers on a left-hand morphology
    view (optional).
  - [x] **Morphometry / topology plots**: `plot_dendrogram`,
    `plot_topology`, `plot_sholl` (with `compute_sholl_profile` and
    `ShollProfile` helpers), `plot_branch_order_histogram`.
  - [x] **Layout caching** — `LayoutCache` LRU keyed on a stable
    morphology snapshot plus the `LayoutConfig` hash. The
    dispatcher consults `get_default_layout_cache()` on every call;
    callers can pass a scoped `cache=LayoutCache(...)` or opt out
    with `use_cache=False`.
  - [ ] **Visual regression tests** — the `pytest-mpl` suite was
    removed in 2026-08. Its baseline directory was never committed and
    CI never passed `--mpl`, so no comparison had ever run; the 12
    tests were figure constructors with no assertions. Eight duplicated
    existing coverage and were dropped, four were rewritten as real
    matplotlib-artist assertions in `backend_matplotlib_test.py`. See
    `docs/specs/2026-08-19-vis-baselines-and-coverage-gaps.md`.
    Reinstating pixel regression needs committed baselines plus a
    Linux-only CI job that actually passes `--mpl`.
  - [x] **Generalized comparison**: `compare_morphologies([m1, m2, ...])`
    and `compare_values(morpho, [values_a, values_b, ...])` in
    `vis/compare.py` (M6 Phase 4).
  - [x] **Interactivity**: `VisHooks(on_pick=, on_hover=, on_leave=)` +
    `PickInfo` in `vis/hooks.py`. The matplotlib backend attaches
    per-artist pick metadata and wires `pick_event` /
    `motion_notify_event` handlers; the PyVista backend builds a
    point→branch lookup and calls `enable_point_picking`
    (M6 Phase 4).
  - [x] **Plotly backend**: `backend_plotly.py` renders value scenes
    as `Scatter3d` traces with per-point `line.color` / `colorscale`
    and a shared scalar bar; gated on
    `importlib.util.find_spec("plotly")` so the base install stays
    dependency-free (M6 Phase 4).
  - [x] **Export polish**: unified `save_figure(figure, path, ...)` in
    `vis/export.py` that dispatches on matplotlib `Axes`/`Figure`,
    pyvista `Plotter`, or plotly `Figure`; `PublicationTheme` preset
    plus `publication_theme()` context manager in `config.py` that
    flips both vis defaults and matplotlib `rcParams` (serif font,
    thicker lines, no grid, print-friendly palette) (M6 Phase 4).
  - [x] **Performance baselines** via `pytest-benchmark`, co-located
    with the modules they measure (`vis/layout/_dispatch_test.py`,
    `vis/scene2d_test.py`, `vis/plot2d_test.py`) — layout build, scene
    build, and plot2d render on 50 / 500 / 2000-branch synthetic
    morphologies, skipped when the plugin is absent (M6 Phase 4).
  - [x] **Narrative tutorial**: `examples/multi_compartment/vis.ipynb` — quick start,
    layout gallery, styling/themes, color-by-values, overlays, movie,
    trace panels, morphometry, interactivity, publication export,
    comparison (M6 Phase 4).
  - [x] **Sphinx autodoc wiring**: `docs/apis/vis.rst` exposes the
    whole public surface (plot entry points, morphometry helpers,
    comparison helpers, hooks, themes, layout engine) through
    `autosummary` and is linked from `docs/index.rst` (M6 Phase 4).
- **Open risks**
  - The stem layout family still holds the most bug-prone code
    (heuristic collision avoidance, the multi-weight scoring
    function). After the Phase 2 split it lives in `vis/layout/_stem.py`
    but remains the largest file in the package. Tuning individual
    scoring weights now goes through `LayoutConfig` rather than
    editing module-level constants, which makes experiments safer.
  - Optional dependencies (`matplotlib`, `pyvista`, `plotly`,
    `pytest-benchmark`) must stay lazy-imported inside
    the backend that uses them. The import-time test from §4.5 /
    risk #5 should grow to assert that none of the heavy optional
    deps are loaded after `import braincell.vis`.
  - `VisHooks` on the matplotlib backend relies on `pick_event` and
    `motion_notify_event`, which only fire with an interactive
    matplotlib backend. Notebook users should pick a GUI backend
    (e.g. `%matplotlib widget`) — the Agg backend used in tests
    will register the handlers but never deliver events, which the
    tests explicitly cover.

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
  - [ ] Chunked large-N pairing, automatic sparse/dense delay queues,
    post-initialization topology mutation, and network batch runtime.
- **Design authority** — [`network/design-overview.md`](network/design-overview.md)
  and its linked API, architecture, issues, and implementation documents.

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
- **Design authority** — [`optim/design-overview.md`](optim/design-overview.md)
  and its linked API, architecture, implementation plan, and references.

---

## 4. Cross-Cutting Concerns

### 4.1 Units

`brainunit` is non-negotiable. Every public API that takes a physical
quantity routes through `_misc.normalize_param`, which **rejects bare
numerics with `TypeError`**. New modules must:

- accept inputs as `python_number/np.ndarray/jax.Array * brainunit_unit`;
- store quantities in canonical SI units internally;
- expose values back to users with units attached, never raw floats.

### 4.2 Immutability discipline

- `Branch`, `CV`, `MorphoEdge`, `MorphoMetric`, `IntegratorEntry`,
  `PaintRule`, `PlaceRule`, `CableProperty` are frozen dataclasses.
- `Morphology` is mutable and carries a monotonic `revision`. Before
  initialization, `Cell` keys its discretization cache by morphology identity
  and revision; structural mutation after initialization must not silently
  reshape runtime state.
- `IntegratorRegistry` is the single mutable global; entries are
  added at import time via decorators and never mutated afterwards.

### 4.3 Cell declaration and initialization

`Cell` owns both its mutable declaration and, after initialization, its JAX
runtime state. Structural declarations are accepted only before
`init_state()`:

```
Cell(morpho, policy)
  -> cell.paint(region, density_mech)
  -> cell.place(locset, point_mech)
  -> cell.init_state()
  -> cell.run(dt=..., duration=...)         # returns RunResult
```

After initialization, topology-changing paint/place/connect/recording calls
are rejected. Parameter mappings may materialize new values into the existing
layout, but must not change the state-tree structure. `reset_state()` resets
runtime values without reopening the declaration phase.

### 4.4 Testing

- pytest with `unittest.TestCase`; tests live next to source as
  `<module>_test.py`, with no exceptions. The `*` must name a real
  sibling module; the one sanctioned exception is a package-scope guard
  in `<package>/__init___test.py`.
- `conftest.py` forces `JAX_PLATFORMS=cpu` and `MPLBACKEND=Agg`.
- IO test fixtures live in `examples/multi_compartment/morpho_files/`.
- New code is expected to ship with co-located tests and to keep
  per-module test runtime under a few seconds on CPU.

### 4.5 Documentation

- All public classes / methods / functions use **NumPy-style
  docstrings** (see CLAUDE.md for the canonical template).
- Examples must be `.. code-block:: python` blocks compatible with
  doctest.
- High-level narrative documentation lives under `docs/`; design
  notebooks live under `examples/multi_compartment/`.

---

## 5. Data-Model Summary

| Layer | Type | Mutability | Lifetime | Owner |
|---|---|---|---|---|
| Geometry | `Branch`, `Soma`, `Dendrite`, ... | frozen | morphology lifetime | user / IO reader |
| Geometry | `Morphology` | mutable tree | until edited | user |
| Geometry view | `MorphoBranch`, `MorphoEdge` | frozen view | follows tree | `Morphology` |
| Metrics | `MorphoMetric` | frozen snapshot | recomputed on demand | `Morphology` |
| Selection | `RegionExpr`, `LocsetExpr` | frozen expression | reusable | user |
| Selection cache | `SelectionCache` | mutable | per-Morphology | filter layer |
| Mechanisms | `CableProperty`, `Density` (`Channel`, `Ion`), `Point*` (`CurrentClamp`, `Synapse`, `Junction`, …) | frozen dataclass / slots | declaration | user |
| Mechanisms | `Ion`, `Channel`, `IonChannel`, `MixIons` | hybrid (JAX state) | per-initialized Cell | `Cell` |
| Discretization | `CV` | frozen | declaration cache / initialization | `Cell` |
| Discretization | `PaintRule`, `PlaceRule` | frozen | declaration | `Cell` |
| Topology | `CVTree`, `NodeTree`, `Node`, `NodeEdge` | frozen | declaration cache / initialization | `Cell` |
| Scheduling | `NodeScheduling` | frozen | initialized runtime | `Cell` |
| Runtime | `CellRuntimeState` and mechanism stores | brainstate-managed | initialized runtime | `Cell` |
| Network | `Population`, connection/recording stores | mixed | network lifecycle | source / target owner |
| Parameters | `ParameterSet`, `ParameterBinding` | stable structure, mutable values | training lifecycle | `TrainableManager` |
| Numerics | `IntegratorEntry` | frozen | process lifetime | `IntegratorRegistry` |
| Numerics | `DiffEqState`, `IndependentIntegration` | brainstate-managed | per-step | step function |

---

## 6. Public API Contract

The list below is the *intended* stable surface. Anything not on it is
internal and may change without deprecation.

- **Morphology layer**: `Branch`, `Soma`, `Dendrite`, `Axon`,
  `BasalDendrite`, `ApicalDendrite`, `CustomBranch`,
  `branch_class_for_type`, `Morphology`, `MorphoBranch`, `MorphoEdge`,
  `MorphoMetric`. The `Morphology` class also exposes the
  `from_swc` / `from_asc` / `from_neuromorpho` classmethod constructors.
- **External-data entry points**: `braincell.io.load_neuromorpho` and
  `Morphology.from_neuromorpho`.
  Tier-2 / Tier-3 NeuroMorpho.Org symbols (`NeuroMorphoClient`,
  `NeuroMorphoCache`, `NeuroMorphoQuery`, `NeuroMorphoMeasurement`,
  `NeuroMorphoError`, …) live under `braincell.io.neuromorpho` and
  `braincell.io`.
- **Filter layer**: `RegionExpr`, `LocsetExpr`, `SelectionCache`.
- **Mechanism declaration layer** (`braincell.mech`): `Mechanism`
  (marker base), `CableProperty`, `Density` (and its concrete
  subclasses `Channel` / `Ion`, which accept the target as either a
  string or a class object), `Point` (and its concrete subclasses
  `CurrentClamp`, `SineClamp`, `FunctionClamp`, `ProbeMechanism`,
  `Synapse`, `Junction`), the frozen `Params` mapping, and the
  registry API (`MechanismRegistry`, `MechanismEntry`,
  `get_registry`, `register_channel`, `register_ion`,
  `register_synapse`).
- **Ion species** (`braincell.ion`): `Sodium`, `SodiumFixed`,
  `Potassium`, `PotassiumFixed`, `Calcium`, `CalciumFixed`,
  `CalciumDetailed`, `CalciumFirstOrder`.
- **Ion channels** (`braincell.channel`): concrete Na, K, Ca, leak, HCN,
  calcium-activated potassium, and mixed-ion families exported by the
  channel package, plus their documented template bases.
- **Synapses** (`braincell.synapse`): `ExpSyn`, `Exp2Syn` from
  `synapse.exponential`.
- **Cell and discretization layer**: `Cell`, `MultiCompartment`, `CellView`,
  `ChannelView`, `IonView`, `SynapseView`, `ClampView`, `RunResult`, `CV`,
  `CVTree`, `CVPolicy`, `CVPerBranch`, `CVPerBranchList`, `MaxCVLen`,
  `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`, `Node`,
  `NodeTree`, and `PointPlacement`. Internal runtime and scheduling records
  are not part of the top-level contract.
- **Network layer**: `Network`, `NetworkResult`, `NetworkConnections`,
  `ConnectionView`, event-source and event-table types, recording schemas,
  `SampleBlock`, `EventSeries`, `connect`, and `observe`. The deliberately
  small `braincell.network.__all__` is separate from top-level convenience
  exports; specialized constructors remain available from their submodules.
- **Trainable parameter layer** (`braincell.trainable`):
  `ParameterSource`, `ParameterBinding`, `ParameterSet`, `TrainableManager`,
  `parameter`, `parameterized`, and `scale`.
- **Numerics layer**: `register_integrator`, `get_integrator`,
  `get_registry`, `IntegratorEntry`, `IntegratorRegistry`,
  `all_integrators`, every `*_step` function listed in
  `braincell/quad/__init__.py::__all__`, `DiffEqModule`,
  `DiffEqState`, `IndependentIntegration`.
- **Neuron base**: `HHTypedNeuron`, `IonChannel`, `Ion`, `IonInfo`,
  `Channel`, `MixIons`, `mix_ions`, `SingleCompartment`.
- **Visualization**: top-level `braincell.vis.plot2d` / `plot3d`
  entry points (the imperative scene API stays internal until it
  stabilizes).

---

## 7. End-to-End User Workflows

### 7.1 Build and inspect a morphology

```python
import braincell
import brainunit as u

morpho, report = braincell.Morphology.from_swc("cell.swc", return_report=True)
print(morpho.topo())
print(morpho.metric)                  # MorphoMetric snapshot
soma_region = braincell.filter.branch_in("type", {"soma"})
distal_region = braincell.filter.branch_range("length", (50 * u.um, None))
```

### 7.2 Discretize and declare mechanisms

```python
import braincell.mech as mech

cell = braincell.Cell(morpho, cv_policy=braincell.DLambda(0.1))

cell.paint(
  braincell.filter.AllRegion(),
  mech.CableProperty(
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
    resting_potential=-65 * u.mV,
  ),
)
cell.paint(soma_region, mech.Ion("SodiumFixed"))
# mech.Channel / mech.Ion accept either a registry name string or the
# concrete class itself — both route through the mechanism registry.
cell.paint(soma_region, mech.Channel(braincell.channel.Na_Ba2002, g_max=0.12 * u.S / u.cm ** 2))
cell.place(
    braincell.filter.at("soma", 0.5),
    mech.CurrentClamp(delay=10 * u.ms, durations=50 * u.ms, amplitudes=0.2 * u.nA),
)
cell.place(braincell.filter.at("soma", 0.5), mech.StateProbe(name="soma_v"))
```

### 7.3 Run a simulation

```python
cell.init_state()
result = cell.run(dt=0.025 * u.ms, duration=100 * u.ms)
print(result.traces["soma_v"].shape)
```

`cell.init_state()` freezes structural declarations and installs runtime state
on the same `Cell`. Subsequent runtime inspection, reset, recording, and
continued runs use that initialized object.

### 7.4 Compare two morphologies visually

```python
braincell.vis.compare2d(morpho_a, morpho_b, layout="frustum")
```

---

## 8. External Dependencies

| Package | Floor | Role |
|---|---|---|
| `python` | 3.11 | language; classifiers claim 3.11–3.14 (see note) |
| `jax` | recent | autodiff, vmap, jit, GPU/TPU — deliberately unpinned |
| `brainunit` | >= 0.0.8 | units (mandatory at every API boundary) |
| `brainstate` | >= 0.5.4 | stateful simulation framework |
| `brainevent` | >= 0.0.7 | sparse event / CSR ops |
| `braintools` | >= 0.1.0 | brain modeling utilities |
| `brainpy` | >= 2.7.5 | brain dynamics library |
| `numpy` | >= 2.0 | arrays |
| `scipy` | recent | scientific helpers |
| `pyvista` | optional | 3D visualization backend |
| `matplotlib` | optional | 2D visualization backend |
| `NEURON` | dev only | reference comparator under `examples/multi_compartment/` |

This table and `[project].dependencies` in `pyproject.toml` are kept in
sync; `pyproject.toml` is the machine-readable source of truth, and the
`requirements*.txt` files are thin pointers to its extras.

`pyproject.toml` is the source of truth for dependency floors. In particular,
`brainstate>=0.5.4` is required for current JAX compatibility and
`numpy>=2.0` reflects the tested support policy. Experimental dependencies in
an uncommitted worktree are not part of this table.

Optional dependencies must be **lazily imported** so the base install
stays small — use `importlib.util.find_spec` plus PEP 562
`__getattr__` for the visualization backends.

> **Note — Python version coverage.** The `classifiers` list advertises
> 3.11 through 3.14, but `CI.yml` and `CI-daily.yml` both run a
> single-entry `python-version: ["3.13"]` matrix. Three of the four
> advertised versions are therefore untested. Either widen the CI matrix
> or narrow the classifiers.

---

## 9. Glossary

- **CV (control volume)** — atomic spatial unit produced by the
  discretization layer; the array-of-CVs is what the integrator sees.
- **CV policy** — rule that turns a `Branch` into a sequence of CVs
  (e.g., `DLambda(0.1)`, `MaxCVLen(10*u.um)`, `CVPerBranch(n)`).
- **Paint** — install a *distributed* mechanism (cable or density)
  onto a `RegionExpr`.
- **Place** — install a *point* mechanism (clamp, probe, synapse,
  gap junction) onto a `LocsetExpr`.
- **DHS** — Dependent Hines Solver: parent-pointer-driven elimination
  ordering used by `dhs_voltage_step`, designed to vectorize the
  classic Hines solver across batched cells.
- **Staggered step** — split integrator that solves the voltage
  system implicitly (DHS) and the gating variables with exponential
  Euler in alternating half-steps.
- **`.bcm` file** — BrainCell Morphology, the self-contained
  checkpoint format produced by `io/checkpoint.py`.
