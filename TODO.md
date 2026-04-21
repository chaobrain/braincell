# BrainCell – Project Design Document

> Status: living document. Tracks both the architectural intent of the
> `braincell` package and the current implementation state of every major
> subsystem. Status markers in this file follow:
>
> - `[x]` shipped — implemented, covered by `*_test.py`, exported from the
>   public API.
> - `[~]` partial — implementation exists but is missing functionality,
>   tests, or runtime integration. Specific gaps are listed inline.
> - `[ ]` planned — design agreed, code not yet written.

---

## 1. Mission and Scope

BrainCell is a JAX-native library for **biologically detailed single-cell
modelling**. It targets the same workload as NEURON, Arbor, and BluePyOpt
but expresses everything as differentiable, vectorized JAX programs so that
multi-compartment cells can be simulated, batched, and trained inside the
broader `brain*` ecosystem (`brainstate`, `brainunit`, `brainevent`,
`braintools`, `brainpy`).

The library owns five concerns end-to-end:

1. **Morphology ingestion** — read SWC / ASC / NeuroML2, validate, cache.
2. **Geometry & discretization** — turn a morphology + a CV policy into
   immutable control-volume (CV) arrays suitable for vectorized solvers.
3. **Mechanism declaration** — paint cable properties, density mechanisms,
   and ion channels onto regions; place point mechanisms onto locsets.
4. **Compilation** — lower the declaration into a `HHTypedNeuron` with
   resolved ion species, channel state, and a DHS-ordered point tree.
5. **Numerical integration** — provide a registry of explicit, implicit,
   exponential, staggered, and diffrax-backed step functions, including a
   custom DHS voltage solver for branched cables.

Out of scope (for this iteration): network simulation, plasticity learning
rules, NEURON HOC compatibility, GUI tools, and stand-alone NMODL execution
(the previous `mech/nmodl/` research tree has been removed; NMODL support,
if it returns, lives behind milestone M5 Phase 4).

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
│   braincell.cv              │           braincell.compute            │
│   CV · CVPolicy ·           │   PointTree · PointScheduling ·        │
│   PaintRule / PlaceRule     │   CellRuntimeState · assignment table  │
├─────────────────────────────┴────────────────────────────────────────┤
│                braincell._multi_compartment (Cell)                   │
│    declaration frontend · lazy rebuild · runtime facade              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HHTypedNeuron
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        braincell.quad                                │
│   IntegratorRegistry · explicit / implicit / exp_euler / staggered / │
│   diffrax steps · dhs_voltage_step (branched-cable Hines solver)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ DiffEqState
                               ▼
                  brainstate / JAX execution
```

```
┌──────────────────────────────────────────────────────────────────────┐
│        braincell.ion · braincell.channel · braincell.synapse         │
│   concrete Ion species (Na, K, Ca) · IonChannel implementations      │
│   (Na, K, Ca, Ih, K_Ca, leaky) · Markov synapse models               │
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
**`io → morph → {filter, mech} → cv → compute → _multi_compartment → quad`**,
with `ion` / `channel` / `synapse` as peer top-level modules supplying concrete
mechanism implementations that `mech` wraps into `Density` (`Channel` /
`Ion`) and `Point` (`CurrentClamp`, `Synapse`, `Junction`, …) declarations
at paint/place time, `vis` reading anything from `morph` upward, and
`_base` providing shared abstract types (`HHTypedNeuron`, `IonChannel`,
`Ion`, `Channel`, `MixIons`) for everything below `Cell`.

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
  - [~] ASC import: most Neurolucida trees, metadata, and
    `Morphology.from_asc(..., return_report=True)` work; **gaps**:
    spine markers, contour-only somas, and multi-tree files are still
    handled minimally — see `io/asc/test.py` skips.
  - [ ] NeuroML2 import — reader stub exists; needs cell, segment-group,
    biophysics decoding and round-trip tests.
  - [x] NEURON-based diff harness via `examples/multi_compartment/neuron_diff.py`.
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
  - `RegionExpr` family: `BranchSlice`, type/name/branch_order /
    parent_id / n_children predicates, `branch_range(...)` for
    continuous-coordinate slicing, set operations
    (union / intersection / difference / complement).
  - `LocsetExpr` family: root, branch points, terminals, region-driven
    uniform sampling, region-driven random sampling.
  - `SelectionCache` — memoizes resolved index sets for stable
    Morphology objects.
- **Status**
  - [x] BranchSlice, broadcasted inputs, set algebra.
  - [x] Discrete predicates (type / name / branch_order / parent_id /
    n_children).
  - [x] Continuous `branch_range(...)` with both numeric and `Quantity`
    bounds.
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
  - [ ] **Locset: explicit anchors** and **fixed-step sampling** along
    a region.
- **Open risks**
  - Distance-based filters require a precomputed
    `path_distance` / `euclidean_distance` array per branch; this needs
    to live on `MorphoMetric` (or a sibling object) and invalidate
    cleanly when the `Morphology` is edited.

### 3.4 `braincell.mech` — mechanism declarations

- **Purpose** — strongly-typed, purely-declarative containers used by
  the `Cell` frontend. Everything here describes *what to install*, not
  *how to integrate*: no `brainstate`, no JAX, no runtime state. The
  concrete ion species, ion channels, and synapses live in peer
  top-level modules (`braincell.ion`, `braincell.channel`,
  `braincell.synapse`) and register themselves with the
  `MechanismRegistry` at import time via class-level decorators; the
  runtime lowering in `braincell.compute._runtime` resolves a
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
    `CurrentClamp.step(amplitude, duration, delay=...)` classmethod
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
    `_base.py`; the declaration-layer classes are reached via
    `braincell.mech.Channel` / `braincell.mech.Ion` to avoid the
    name collision.
  - [x] **Mechanism registry.** `MechanismRegistry` + the
    `@register_channel` / `@register_ion` / `@register_synapse`
    decorators ship in `mech/_registry.py`. ~49 concrete classes in
    `braincell.channel`, `braincell.ion`, and `braincell.synapse`
    self-register at import time. `get_registry().get(category,
    class_name)` is the single lookup path used by
    `compute/_runtime.py` to resolve `Density.class_name` into a
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
    `Density`. The old abstraction leak where
    `cv/_mech._scale_density_for_coverage` smuggled the coverage
    fraction through `params["coverage_area_fraction"]` and
    `compute/_runtime._runtime_constructor_params` had to filter it
    back out is gone.
  - [x] **Unified `CurrentClamp`.** One canonical frozen-dataclass
    form `(start, durations, amplitudes)`. The old
    `CurrentClamp(amplitude=, delay=, duration=)` compatibility form
    is gone; use `CurrentClamp.step(amplitude, duration, delay=...)`.
  - [x] **Consumer simplification.** `cv/_mech.py`,
    `compute/_runtime.py`, `compute/_assignment_table.py` have been
    rewritten against the new types. `mechanism_signature` collapses
    to `(type(m).__qualname__, m)` (structural equality does the rest),
    `_resolve_runtime_channel_spec` is deleted, `mechanism_cell_key`
    is a straight attribute read.
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
    `cv/_mech.py`.
  - **Class-level decorator ordering.** Registration is a side
    effect of importing `braincell.channel` / `braincell.ion` /
    `braincell.synapse`. If a user imports `braincell.mech` alone
    (without importing the concrete modules) the registry is empty —
    by design. The canonical entry points in `braincell/__init__.py`
    already import all three, so normal users never see this.
  - **Ion binding inference** uses
    `issubclass(cls.root_type, Sodium/Potassium/Calcium)` in
    `compute/_runtime.py`. New ion species must either set
    `root_type` on their channels or we extend the dispatch to walk
    a lookup table — do not hardcode class-name matching.
  - **Name collision with `_base.Channel` / `_base.Ion`.** The
    declaration-layer `Channel` / `Ion` classes live under
    `braincell.mech`, not at the top level of `braincell`, because
    `braincell.Channel` / `braincell.Ion` already resolve to the
    runtime base classes from `_base.py`. Tutorials and user code
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

### 3.5 `braincell.cv` / `braincell.compute` / `_multi_compartment` — declaration, discretization, runtime

- **Purpose** — the orchestration layer. Three co-operating pieces turn
  *(Morphology, CVPolicy, paint/place declarations)* into a runnable
  `HHTypedNeuron`:
  - `braincell.cv` owns the pure control-volume layer (geometry +
    mechanism rules + policies).
  - `braincell.compute` owns the execution-graph / runtime lowering
    built on top of `cv`.
  - `braincell._multi_compartment` owns the public `Cell` (mutable
    declaration) and `RunnableCell` (frozen `HHTypedNeuron` runtime)
    classes. The split replaces the old monolithic, dirty-flag-driven
    `Cell` with a one-way pipeline `Cell → RunnableCell` driven by
    `cell.build()`.
- **Key files**
  - `braincell/_multi_compartment/` — package.
    - `cell.py` — `Cell` declaration frontend (mutable, no JAX state).
    - `runnable.py` — `RunnableCell(HHTypedNeuron)` frozen runtime.
    - `build.py` — `cell.build()` lowering pipeline.
    - `bridge.py` — `cv_to_point` / `point_to_cv` scatter/gather.
    - `currents.py` — `total_membrane_current` pipeline (bugs #1/#2
      fixed: external current routed through `sum_current_inputs(init=)`;
      ambiguous `(n_point,)` total current rejected).
    - `clamp_table.py` — `ClampActiveTable` precomputed once at build.
    - `probes.py` — `sample_probe(rcell, ...)` / `sample_probes(rcell)`.
    - `run.py` — `rcell.run(dt=, duration=)` and the `RunResult`
      pytree-registered dataclass.
    - `*_test.py` — one co-located test per source module.
  - `braincell/cv/_cv.py` — `CV` dataclass plus `assemble_cv` to
    materialize the array-of-CVs view.
  - `braincell/cv/_geo.py` — `build_cv_geo` reduces a `Morphology` +
    `CVPolicy` into per-CV geometry (length, area, volume, axial
    conductance, parent index).
  - `braincell/cv/_mech.py` — `PaintRule`, `PlaceRule`, default rules,
    normalization, `init_cv_mech`, paint/place application.
  - `braincell/cv/_policy.py` — `CVPolicy` ABC plus `CVPerBranch`,
    `MaxCVLen`, `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`.
  - `braincell/compute/_point_tree.py` — `PointTree`, `CVPoint`,
    `CVEdge`, `ComputePoint`, `ComputeEdge`, `PointScheduling`,
    `build_point_tree`, `build_point_scheduling` (PointScheduling +
    DHS grouping for vectorized parent traversal live here too).
  - `braincell/compute/_assignment_table.py` — `MechanismObjectCell`,
    `MechanismObjectTable` keyed by `mechanism_cell_key`.
  - `braincell/compute/_runtime.py` — `CellRuntimeState`,
    `install_cell_runtime`, `cv_value_vector`, midpoint scatter/gather
    utilities. Tests in `_runtime_test.py`.
- **Status**
  - [x] `Cell(morpho, cv_policy)` declaration entry, morphology
    snapshotting, `paint` / `place` API. `Cell` is pure declaration —
    no JAX state and no dirty flags; `cell.cvs` / `cell.n_cv` are
    lazy previews memoized on declaration state.
  - [x] CV discretization: `CVPolicy` base + concrete policies, CV
    geometry, axial-resistance partitioning across branch joints.
  - [x] Mechanism mapping: cable paint, density paint, point place
    lowered into per-CV records.
  - [x] PointTree: compute points, compute edges, attachment handling.
  - [x] Scheduling: `PointScheduling` + DHS grouping for the voltage
    solver.
  - [x] **Execution layer**: `cell.build()` lowers a declaration into
    a frozen `RunnableCell(HHTypedNeuron)` via
    `_multi_compartment/build.py`. The pipeline allocates `V` / `spike`
    and per-channel / per-ion `DiffEqState`, installs a
    `CellRuntimeState` with a precomputed `ClampActiveTable`, caches
    the axial operator, and wires `rcell.run(dt=, duration=)` on top
    of the `quad/` registry (default `staggered`). Trace recording
    uses `StateProbe` / `MechanismProbe` / `CurrentProbe` sampled via
    `rcell.sample_probe(...)` / `rcell.sample_probes()`. Two bugs
    fixed along the way: external current is now routed through
    `sum_current_inputs(init=I_ext_density)` (bug #1) and
    `(n_point,)`-shaped total current in `nA` is rejected explicitly
    instead of producing NaN (bug #2).
- **Open risks**
  - The `Cell → RunnableCell` arrow is one-way; `RunnableCell` must
    never mutate or consult the originating `Cell`. Any future
    convenience that tries to "rebuild in place" would re-introduce
    the dirty-flag surface that this refactor removed.
  - The runtime/state shape must remain stable across `jit`
    re-traces — every change to declared mechanisms on `Cell` is
    allowed to re-trace, but parameter updates on an existing
    `RunnableCell` must not.

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
  - **Implicit / mixed**: `backward_euler_step`, `implicit_euler_step`,
    `implicit_rk4_step`, `implicit_exp_euler_step`, `cn_rk4_step`,
    `cn_exp_euler_step`, `exp_exp_euler_step`, `splitting_step`.
  - **Exponential Euler**: `exp_euler_step`, `ind_exp_euler_step`.
  - **Staggered**: `staggered_step` (DHS voltage solve +
    `ind_exp_euler` for ion-channel state, the workhorse for full
    cells).
  - **Diffrax-backed**: `diffrax_euler/heun/midpoint/ralston/bosh3/
    tsit5/dopri5/dopri8/bwd_euler/kvaerno{3,4,5}_step` — gated on
    `importlib.util.find_spec('diffrax')` so the dependency is
    optional. The actual `import diffrax` is deferred via PEP 562
    `__getattr__` so importing `braincell.quad` is cheap.
  - **Voltage solvers**: `dhs_voltage_step` (DHS branched Hines),
    `dense_voltage_step`, `sparse_voltage_step`.
- **Status**
  - [x] Registry, alias resolution, "did you mean ...?" suggestions.
  - [x] Backwards-compatible `all_integrators` mapping view.
  - [x] All explicit RK / Heun / Ralston / Midpoint / SSPRK families.
  - [x] Backward Euler, implicit Euler, implicit RK4, implicit exp
    Euler, CN variants, splitting, exp-exp Euler.
  - [x] Exponential Euler (`exp_euler_step`, `ind_exp_euler_step`).
  - [x] Staggered solver (`staggered_step`).
  - [x] Diffrax bridge for explicit and implicit families with lazy
    import.
  - [x] DHS voltage solver (`dhs_voltage_step`).
  - [ ] **Adaptive timestep wrapper** that produces a registered
    integrator from any embedded RK pair (currently only available
    via diffrax).
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
  - `perf_benchmark_test.py` — `pytest-benchmark` baselines for layout
    build, scene build, and end-to-end plot2d render on 50 / 500 /
    2000-branch synthetic morphologies (skipped when
    `pytest-benchmark` is not installed).
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
  - `_test_helper.py` — `FakeBackend` scene-capturing double for unit
    tests.
  - `visual_regression_test.py` — `pytest-mpl` baseline image
    regression suite (skipped when `pytest_mpl` is not installed).
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
  - [x] **Visual regression tests** via `pytest-mpl` with 12 baseline
    slots under `braincell/vis/_baseline_images/` — the whole module
    skips when `pytest_mpl` is not installed so the base suite
    stays dependency-free.
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
  - [x] **Performance baselines** via `pytest-benchmark` in
    `vis/perf_benchmark_test.py` — layout build, scene build, and
    plot2d render on 50 / 500 / 2000-branch synthetic morphologies,
    skipped when the plugin is absent (M6 Phase 4).
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
    `pytest-mpl`, `pytest-benchmark`) must stay lazy-imported inside
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
  - `braincell/ion/sodium.py` — `Sodium` (abstract base with
    `root_type = HHTypedNeuron`) and `SodiumFixed` (constant `E`, `C`;
    parameterised via `braintools.init.param`).
  - `braincell/ion/potassium.py` — `Potassium` abstract base and
    `PotassiumFixed` with a custom `reset_state` that walks attached
    child `Channel` nodes.
  - `braincell/ion/calcium.py` — `Calcium` base class,
    `CalciumFixed`, the shared `_CalciumDynamics` parent, and two
    concrete dynamics models:
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
  - [ ] **`_IonDynamics` shared base** — the current
    `_CalciumDynamics` in `calcium.py` is already a reusable pattern
    (DiffEqState concentration + Nernst reversal + forwarded
    `compute_derivative`). Lift it to a package-private
    `braincell/ion/_dynamics.py` and have the Na / K / Cl dynamics
    subclass it with their own valence (`z`) and `derivative`.
  - [ ] **`__init__.py` hygiene** — `braincell/ion/__init__.py`
    currently uses star imports plus `_sodium_all` / `_potassium_all`
    / `_calcium_all`. Switch to explicit re-exports so
    `from braincell.ion import Calcium` is a single lookup and so
    IDEs / Sphinx autodoc can see the symbols without running the
    star-import dance.
  - [x] **Mechanism-registry plumbing** — every concrete `Ion`
    subclass now self-registers via `@register_ion("CalciumFixed")` /
    `@register_ion("CalciumDetailed")` / `@register_ion("CalciumFirstOrder")` /
    `@register_ion("SodiumFixed")` / `@register_ion("PotassiumFixed")`
    at import time, and `braincell.mech.Ion("CalciumFixed")` resolves
    through the registry described in §3.4.
  - [ ] **Consistent external-current registration** — audit that
    every dynamics class honours `include_external=True` in its
    `derivative` (the existing `CalciumDetailed.derivative` already
    does; the contract must stay alive across future refactors).
- **Open risks**
  - **Nernst unit trap.** `gas_constant * T / (2 * faraday_constant)`
    only resolves to mV when every factor carries its brainunit
    quantity. `_CalciumDynamics` caches this factor on `__init__`;
    any change to how `Ion.__init__` stores parameters must re-check
    that the cached constant survives `brainstate.graph` pytree
    flattening.
  - **`reset_state` asymmetry.** `SodiumFixed.reset_state` inherits
    from `Ion`, but `PotassiumFixed` / `CalciumFixed` override it
    with an explicit `check_hierarchies` call and
    `_CalciumDynamics.reset_state` re-samples `C.value` from the
    stored initializer. The four flavours must stay in sync or one
    species silently skips its child-channel resets. Covered by the
    existing lifecycle tests but worth watching every refactor.
  - **Test-side coupling with `braincell.channel`.** The calcium
    tests instantiate `ICaT_HM1992` to exercise child-channel
    forwarding, so a heavy top-level import in `braincell.channel`
    would drag through the ion suite. Keep the channel package
    tree-shakable (see §3.9 risks).

### 3.9 `braincell.channel` — concrete ion channels

- **Purpose** — the library's catalogue of ready-to-use HH-style and
  Markov-kinetics ion channels. Every class is a subclass of
  `Channel` from `_base.py` (so every instance is an `IonChannel`
  that registers its gate state as `DiffEqState`s) and declares
  `root_type = HHTypedNeuron`. Channels are container children of
  an `Ion` species or of a `SingleCompartment` / `Cell` directly.
- **Key families**
  - `braincell/channel/sodium.py` — `SodiumChannel` base, the
    `INa_p3q_markov` template (m³h), concrete models
    `INa_Ba2002` / `INa_TM1991` / `INa_HH1952`, and the
    resurgent-Na `INa_Rsg` which opts into `IndependentIntegration`
    because its Markov state is integrated with its own step inside
    the staggered solver.
  - `braincell/channel/potassium.py` — `PotassiumChannel` base, the
    delayed-rectifier template `IK_p4_markov` with concrete DR
    models (`IKDR_Ba2002`, `IK_TM1991`, `IK_HH1952`), A-type templates
    (`IKA_p4q_ss` with `IKA1_HM1992` / `IKA2_HM1992`), a second
    A-type (`IKK2_pq_ss` with `IKK2A_HM1992` / `IKK2B_HM1992`), the
    non-inactivating `IKNI_Ya1989`, the potassium leak `IK_Leak`,
    and a Kv sub-type family (`IKv11_Ak2007`, `IKv34_Ma2020`,
    `IKv43_Ma2020`, `IKM_Grc_Ma2020`, `IK_Kv_test`).
  - `braincell/channel/calcium.py` — `CalciumChannel` base, the
    non-specific `ICaN_IS2008`, the shared `_ICa_p2q_ss` /
    `_ICa_p2q_markov` templates, T-type models (`ICaT_HM1992`,
    `ICaT_HP1992`), high-threshold T (`ICaHT_HM1992`,
    `ICaHT_Re1993`), L-type (`ICaL_IS2008`), and the Ma et al. 2020
    Cav family (`ICav12_Ma2020`, `ICav13_Ma2020`, `ICav23_Ma2020`,
    `ICav31_Ma2020`, `ICaGrc_Ma2020`).
  - `braincell/channel/leaky.py` — `LeakageChannel` base and the
    passive leak `IL` (`g_L (E_L − V)`).
  - `braincell/channel/hyperpolarization_activated.py` —
    `Ih_HM1992` plus the Ma 2020 pair (`Ih1_Ma2020`, `Ih2_Ma2020`).
  - `braincell/channel/potassium_calcium.py` — `KCaChannel` base,
    calcium-activated afterhyperpolarization `IAHP_De1994`, and the
    SK / IK / BK-type models from Ma et al. 2020 (`IKca3_1_Ma2020`,
    `IKca2_2_Ma2020`, `IKca1_1_Ma2020`). These consume an `IonInfo`
    from the attached calcium ion alongside the membrane voltage.
- **Status**
  - [x] ~30 concrete channel classes across six families, named by
    the convention `I<species><mechanism>_<Author><Year>`, all
    documented with NumPy-doc docstrings.
  - [x] Co-located unit tests (~170 across `sodium_test.py`,
    `potassium_test.py`, `calcium_test.py`, `leaky_test.py`,
    `hyperpolarization_activated_test.py`,
    `potassium_calcium_test.py`) covering steady-state gates at
    reference voltages, current sign / shape, and `reset_state`
    idempotency.
  - [x] **Mechanism-registry plumbing** — every concrete `Channel`
    subclass (~40 classes across the six channel submodules) now
    self-registers via `@register_channel("INa_Ba2002")` at import
    time, with optional aliases (`IL` exposes `"leaky"`). The
    `(category="channel", class_name="...")` lookup in §3.4 resolves
    through `get_registry()`. Abstract bases (`SodiumChannel`,
    `PotassiumChannel`, `CalciumChannel`, `LeakageChannel`,
    `KCaChannel`) are deliberately not decorated.
  - [ ] **Parameter metadata** — each channel should declare the
    unit of every user-facing parameter (`g_max` in `S/cm²`, `E` in
    `mV`, time constants in `ms`, …) so that `Density.params`
    validation can produce an actionable error at paint time rather
    than an opaque JAX trace failure. Store the per-parameter unit
    on `MechanismEntry.metadata` and consult it during
    `Density.__init__`.
  - [ ] **GHK current formulation** — calcium channels currently
    compute `I = g_max * p^a * q^b * (E − V)` with a Nernst-derived
    `E`. Add an opt-in GHK (Goldman–Hodgkin–Katz) formulation as a
    mixin so Cav channels can switch to the thermodynamically more
    accurate form without duplicating gate state.
  - [ ] **Q10 temperature scaling audit** — a handful of channels
    (e.g., `IKA_p4q_ss`) already apply a Q10 factor while others do
    not. Normalise temperature handling via a shared helper in
    `_base.py` and document the default Q10 per family.
  - [ ] **NEURON `.mod` validation** — for every channel in the
    catalogue, compare voltage-clamp and current-clamp traces
    against the reference `.mod` implementation within a tight
    tolerance. Requires re-introducing the `mech/mod_validate/`
    harness (see §3.4) and wiring it into milestone M5.
  - [ ] **Chloride channels** — add a `braincell/channel/chloride.py`
    module once `braincell.ion.Chloride` lands, covering the passive
    leak plus GABAa-reversal-driven phasic conductance.
  - [ ] **Stiff-channel integrator audit** — `INa_Rsg` already opts
    out of the default exponential-Euler path via
    `IndependentIntegration`. Run the M7 convergence matrix over
    every channel to confirm none of the others silently need the
    same escape hatch.
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
  - **API drift vs NEURON naming.** Upstream `.mod` files use
    lowercase suffixes (`ih`, `ik`, `ikdr`); BrainCell uses
    `I<Species><Mechanism>_<Author><Year>`. Any validation harness
    needs a stable alias table so the diff does not become a
    renaming exercise every time a new channel lands.

### 3.10 `braincell` package root — neuron base classes

- `_base.py` — `HHTypedNeuron`, `IonChannel`, `Ion`, `IonInfo`,
  `Channel`, `MixIons`, `mix_ions`. These are the abstract building
  blocks every concrete cell composes.
- `_single_compartment.py` — `SingleCompartment`, the simplest
  concrete neuron, used as a sanity surface and example.
- `_multi_compartment/` — `Cell` (mutable declaration) and
  `RunnableCell` (frozen `HHTypedNeuron` runtime produced by
  `cell.build()`), composed with `cv` + `compute` (see §3.5).
- `_misc.py` — `normalize_param` (the brainunit gatekeeper), helpers,
  decorators (`set_module_as`, `deprecation_getattr`), `Container`.
- `_typing.py` — type aliases (`Initializer`, `ArrayLike`, `T`, `DT`).

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
- `Morphology` is mutable but `Cell` always works on a `clone()` of the
  morphology it was given. Tree-edit operations (planned in §3.1) must
  preserve this so that `Cell` rebuild flags can stay correct.
- `IntegratorRegistry` is the single mutable global; entries are
  added at import time via decorators and never mutated afterwards.

### 4.3 Cell → RunnableCell

`Cell` is intentionally cheap to construct and mutate, and owns
**no** JAX state. All runtime state is produced by `cell.build()`,
which returns a frozen `RunnableCell`. The expected sequence is:

```
Cell(morpho, policy)                        # declaration only
  → cell.paint(region, density_mech)        # cheap, mutates declaration
  → cell.place(locset, point_mech)          # cheap, mutates declaration
  → rcell = cell.build()                    # lower into frozen RunnableCell
  → rcell.run(dt=..., duration=...)         # JIT and step (returns RunResult)
```

Calling `cell.build()` twice produces two independent runnables. The
`Cell` remains safe to mutate afterwards; the `RunnableCell` never
consults or mutates the originating `Cell`. Execution-layer work must
preserve this one-way arrow — no dirty-flag state machine, no
in-place "rebuild on next use".

### 4.4 Testing

- pytest with `unittest.TestCase`; tests live next to source as
  `*_test.py` (exception: `io/swc/test.py`, `io/asc/test.py`).
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
| Geometry | `Branch`, `Soma`, `Dendrite`, ... | frozen | per-build | user / IO reader |
| Geometry | `Morphology` | mutable tree | until edited | user |
| Geometry view | `MorphoBranch`, `MorphoEdge` | frozen view | follows tree | `Morphology` |
| Metrics | `MorphoMetric` | frozen snapshot | recomputed on demand | `Morphology` |
| Selection | `RegionExpr`, `LocsetExpr` | frozen expression | reusable | user |
| Selection cache | `SelectionCache` | mutable | per-Morphology | filter layer |
| Mechanisms | `CableProperty`, `Density` (`Channel`, `Ion`), `Point*` (`CurrentClamp`, `Synapse`, `Junction`, …) | frozen dataclass / slots | declaration | user |
| Mechanisms | `Ion`, `Channel`, `IonChannel`, `MixIons` | hybrid (JAX state) | per-runnable | `RunnableCell` |
| Discretization | `CV` | frozen | built at `cell.build()` | `Cell` (preview) / `RunnableCell` |
| Discretization | `PaintRule`, `PlaceRule` | frozen | declaration | `Cell` |
| Topology | `PointTree`, `CVPoint`, `CVEdge` | frozen | built at `cell.build()` | `RunnableCell` |
| Scheduling | `PointScheduling` | frozen | built at `cell.build()` | `RunnableCell` |
| Runtime | `CellRuntimeState`, `ClampActiveTable` | brainstate-managed | per-step | `RunnableCell` |
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
- **External-data entry points** (top-level re-exports): `load_neuromorpho`.
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
- **Ion channels** (`braincell.channel`): every `I<Species>*`
  concrete class exported from the six channel submodules (sodium,
  potassium, calcium, leaky, hyperpolarization_activated,
  potassium_calcium). Base classes (`SodiumChannel`,
  `PotassiumChannel`, `CalciumChannel`, `LeakageChannel`,
  `KCaChannel`) are public for subclassing.
- **Synapses** (`braincell.synapse`): `AMPA`, `GABAa`, `NMDA` from
  `synapse.markov`.
- **Cell layer**: `Cell`, `RunnableCell`, `RunResult`, `CV`, `CVPolicy`,
  `CVPerBranch`, `MaxCVLen`, `DLambda`, `CVPolicyByTypeRule`,
  `CompositeByTypePolicy`, `PointTree`, `PointScheduling`.
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
soma_region   = braincell.RegionExpr.by_type("soma")
distal_region = braincell.RegionExpr.branch_range(50 * u.um, None)
```

### 7.2 Discretize and declare mechanisms

```python
import braincell.mech as mech

cell = braincell.Cell(morpho, cv_policy=braincell.DLambda(0.1))

cell.paint(
  braincell.RegionExpr.everywhere(),
  mech.CableProperty(
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
    resting_potential=-65 * u.mV,
  ),
)
cell.paint(soma_region, mech.Ion("SodiumFixed"))
# mech.Channel / mech.Ion accept either a registry name string or the
# concrete class itself — both route through the mechanism registry.
cell.paint(soma_region, mech.Channel(braincell.channel.INa_Ba2002, g_max=0.12 * u.S / u.cm ** 2))
cell.place(
    braincell.LocsetExpr.root(),
    mech.CurrentClamp.step(0.2 * u.nA, 50 * u.ms, delay=10 * u.ms),
)
cell.place(braincell.LocsetExpr.terminals(), mech.ProbeMechanism("v"))
```

### 7.3 Run a simulation

```python
rcell = cell.build()                 # frozen RunnableCell(HHTypedNeuron)
result = rcell.run(dt=0.025 * u.ms,
                   duration=100 * u.ms)
braincell.vis.plot2d(rcell, values=result.traces["soma(0.5)_v"][-1])
```

`cell.build()` lowers the declaration into a frozen `RunnableCell` —
all subsequent runtime inspection (`rcell.layouts`,
`rcell.point_tree()`, `rcell.sample_probes()`, `rcell.current_time`,
`rcell.get_state(...)`) lives on the runnable, not on `Cell`.

### 7.4 Compare two morphologies visually

```python
braincell.vis.compare2d(morpho_a, morpho_b, layout="frustum")
```

---

## 8. External Dependencies

| Package | Floor | Role |
|---|---|---|
| `python` | 3.11 | language; tested 3.11–3.14 |
| `jax` | recent | autodiff, vmap, jit, GPU/TPU |
| `brainunit` | >= 0.0.8 | units (mandatory at every API boundary) |
| `brainstate` | >= 0.2.0 | stateful simulation framework |
| `brainevent` | >= 0.0.7 | sparse event / CSR ops |
| `braintools` | >= 0.1.0 | brain modeling utilities |
| `brainpy` | >= 2.7.5 | brain dynamics library |
| `numpy` | >= 1.15 | arrays |
| `scipy` | recent | scientific helpers |
| `diffrax` | optional | extra integrator family in `quad/_diffrax.py` |
| `pyvista` | optional | 3D visualization backend |
| `matplotlib` | optional | 2D visualization backend |
| `NEURON` | dev only | reference comparator under `examples/multi_compartment/` |

Optional dependencies must be **lazily imported** so the base install
stays small. `quad/_diffrax.py` already follows this pattern via
`importlib.util.find_spec` plus PEP 562 `__getattr__`; visualization
backends should match it.

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
