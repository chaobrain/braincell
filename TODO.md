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
(NMODL parsing exists as research-only code under `mech/nmodl/`).

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
│   Branch (frozen) · Morphology (mutable tree) · MorphoMetric         │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │ Morphology                        │
               ▼                                   ▼
┌─────────────────────────────┐    ┌────────────────────────────────────┐
│      braincell.filter       │    │           braincell.mech           │
│  RegionExpr · LocsetExpr    │    │  CableProperties · DensityMech ·   │
│  SelectionCache             │    │  PointMech · ion · channel ·       │
│                             │    │  synapse                           │
└──────────────┬──────────────┘    └─────────────────┬──────────────────┘
               │ selection                           │ mechanisms
               ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        braincell.cell                                │
│   Cell (declaration + lazy rebuild) · CV · CVPolicy ·                │
│   PaintRule / PlaceRule · PointTree · PointScheduling · runtime      │
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
│                         braincell.vis                                │
│   2D / 3D scenes · matplotlib & PyVista backends ·                   │
│   region / locset / value overlays                                   │
└──────────────────────────────────────────────────────────────────────┘
       (consumes Branch / Morphology / Cell / RegionExpr / LocsetExpr)
```

The directional rule of thumb: **`io → morph → {filter, mech} → cell → quad`**,
with `vis` reading anything from `morph` upward and `_base` providing
shared abstract types for everything below `cell`.

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
  - [x] NEURON-based diff harness via `develop_doc/neuron_diff.py`.
  - [x] NeuroMorpho.Org integration: Tier 1 `load_neuromorpho` /
    `fetch_neuromorpho` one-liners, Tier 2 `NeuroMorphoClient` with
    typed `iter_search` / `download` / retries, Tier 3 `NeuroMorphoCache`
    plus pure URL helpers, full NumPy-doc docstrings, and
    `Morphology.from_neuromorpho` classmethod. Notebook walkthrough at
    `develop_doc/neuromorpho_diff.ipynb` shows the full search → cache →
    metric-diff loop.
  - [ ] Automated metric diff against published NeuroMorpho reference
    statistics promoted from the notebook into a pytest case (so the
    NeuroMorpho corpus becomes a wide regression net).
  - [x] Checkpoint API and `.bcm` format with notebook tutorial
    (`develop_doc/morpho-checkpoint.ipynb`).
  - [ ] **NMODL parsing compiler** — currently research-only under
    `mech/nmodl/`; will be promoted to a real codegen target once the
    runtime mechanism layer stabilizes.
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

- **Purpose** — strongly-typed declarative containers for cable
  properties, density mechanisms, point mechanisms, ion species, ion
  channels, and synapses. These are *what to install*, not *how to
  integrate*.
- **Key types**
  - `CableProperties` — passive cable parameters (Cm, Ra, resting
    potential, temperature).
  - `DensityMechanism` — distributed mechanism (e.g., density of an
    ion channel) parameterized by region.
  - `PointMechanism` and concrete subclasses: `CurrentClamp`,
    `SineClamp`, `FunctionClamp`, `ProbeMechanism`,
    `SynapseMechanism`, `GapJunctionMechanism`.
  - `mech.ion` — ion species (sodium, potassium, calcium) producing
    `Ion` / `IonInfo` objects from `_base.py`.
  - `mech.channel` — concrete channel implementations (`INa_Ba2002`,
    Ih, calcium-activated potassium, leaky, etc.).
  - `mech.synapse` — Markov synapse models.
  - `mech.spec` — abstract specification base classes shared by all
    mechanisms.
- **Status**
  - [x] Cable / Density / Point dataclasses and concrete clamps /
    probes / synapses.
  - [x] Top-level re-exports of `ion`, `channel`, `synapse`.
  - [~] **Runtime integration** — declarations are accepted by
    `Cell.paint` / `Cell.place` and lowered into `PaintRule` /
    `PlaceRule` records, but the closing of the loop into compiled
    JAX kernels (state allocation, current accumulation, gating
    integration) is incomplete. See §3.6.
  - [ ] **Mechanism validation** — a structured comparison harness
    versus NEURON `.mod` reference traces. Skeleton notebooks exist
    under `mech/mod_validate/`; need to be promoted to automated
    pytest cases.
  - [ ] **NMODL → braincell codegen** — parsing pipeline lives in
    `mech/nmodl/`; the missing piece is the lowering pass that emits
    `IonChannel` / `DensityMechanism` subclasses.

### 3.5 `braincell.cell` — declaration, discretization, runtime

- **Purpose** — the orchestration layer. Owns the user-facing `Cell`
  object that turns *(Morphology, CVPolicy, paint/place declarations)*
  into a runnable `HHTypedNeuron`.
- **Key types and files**
  - `cell.py` — `Cell(HHTypedNeuron)`. Three roles: declaration
    frontend, lazy rebuild owner, runtime facade.
  - `cv.py` — `CV` dataclass plus `assemble_cv` to materialize the
    array-of-CVs view.
  - `cv_geo.py` — `build_cv_geo` reduces a `Morphology` + `CVPolicy`
    into per-CV geometry (length, area, volume, axial conductance,
    parent index).
  - `cv_mech.py` — `PaintRule`, `PlaceRule`, default rules,
    normalization, `init_cv_mech`, paint/place application.
  - `cv_policy.py` — `CVPolicy` ABC plus `CVPerBranch`, `MaxCVLen`,
    `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`.
  - `point_tree.py` — `PointTree`, `CVPoint`, `CVEdge`, `ComputePoint`,
    `ComputeEdge`, `build_point_tree`, `build_point_scheduling`.
  - `point_scheduling.py` — `PointScheduling` and DHS (Dependent
    Hines Solver) grouping for vectorized parent traversal.
  - `assignment_table.py` — `MechanismObjectCell`, `MechanismObjectTable`
    keyed by `mechanism_cell_key`.
  - `runtime.py` — `CellRuntimeState`, `install_cell_runtime`,
    `cv_value_vector`, midpoint scatter/gather utilities.
- **Status**
  - [x] `Cell(morpho, cv_policy)` declaration entry, morphology
    snapshotting, `paint` / `place` API, lazy rebuild flags.
  - [x] CV discretization: `CVPolicy` base + concrete policies, CV
    geometry, axial-resistance partitioning across branch joints.
  - [x] Mechanism mapping: cable paint, density paint, point place
    lowered into per-CV records.
  - [x] PointTree: compute points, compute edges, attachment handling.
  - [x] Scheduling: `PointScheduling` + DHS grouping for the voltage
    solver.
  - [ ] **Execution layer**: `Cell.run(...)`, full `HHTypedNeuron`
    compile path, end-to-end JAX simulation. The missing pieces are:
    1. allocating and naming all `DiffEqState` channel/ion variables
       per CV;
    2. wiring `PaintRule`/`PlaceRule` outputs into a single fused
       `dV/dt` + gating update;
    3. dispatching to a `quad/` integrator selected by name;
    4. trace recording via `ProbeMechanism`;
    5. brainstate-compatible pytree flattening.
- **Open risks**
  - Two-phase build (declaration → rebuild → runtime install) needs
    very precise dirty-flag discipline; otherwise users will silently
    simulate against stale CV layouts.
  - The runtime/state shape must remain stable across `jit`
    re-traces — every change to declared mechanisms is allowed to
    re-trace, but parameter updates must not.

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
  backend (matplotlib).
- **Key types and files**
  - `scene.py`, `scene2d.py`, `scene3d.py` — scene graph builder.
  - `plot2d.py`, `plot3d.py` — high-level user entry points.
  - `backend.py`, `backend_matplotlib.py`, `backend_pyvista.py` —
    rendering backends.
  - `layout2d.py`, `compare2d.py` — automatic 2D layouts and side-by-
    side morphology comparison.
  - `config.py` — style configuration shared across backends.
- **Status**
  - [x] 3D rendering of `Branch` / `Morphology` with point geometry,
    scene composition, PyVista backend.
  - [x] 2D projected mode driven by real points.
  - [x] 2D tree auto-layout.
  - [x] 2D frustum auto-layout.
  - [x] Stem / balloon / radial360 layout family with matplotlib
    comparison output.
  - [~] Overlays: `region` / `locset` / `values` parameters are wired
    through the unified plotting entry point, but the highlight /
    coloring semantics are still primitive — gaps:
    - per-CV value colormap with proper colorbars and unit labels;
    - locset markers respecting region restriction;
    - hover/pick metadata in PyVista linking back to branch IDs.
  - [ ] **Time-series animation** of voltage / state arrays once the
    runtime layer (§3.5) lands.

### 3.8 `braincell` package root — neuron base classes

- `_base.py` — `HHTypedNeuron`, `IonChannel`, `Ion`, `IonInfo`,
  `Channel`, `MixIons`, `mix_ions`. These are the abstract building
  blocks every concrete cell composes.
- `_single_compartment.py` — `SingleCompartment`, the simplest concrete
  neuron, used as a sanity surface and example.
- `_multi_compartment.py` — legacy multi-compartment class kept for
  backwards compatibility; new work should target the
  `cell.Cell` pipeline.
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
  `PaintRule`, `PlaceRule`, `CableProperties` are frozen dataclasses.
- `Morphology` is mutable but `Cell` always works on a `clone()` of the
  morphology it was given. Tree-edit operations (planned in §3.1) must
  preserve this so that `Cell` rebuild flags can stay correct.
- `IntegratorRegistry` is the single mutable global; entries are
  added at import time via decorators and never mutated afterwards.

### 4.3 Lazy rebuild

`Cell` is intentionally cheap to construct and mutate. Heavy work
happens only when a derived view is requested. The expected sequence is:

```
Cell(morpho, policy)
  → cell.paint(region, density_mech)        # cheap, marks dirty
  → cell.place(locset, point_mech)          # cheap, marks dirty
  → cell.cv_layout                          # rebuild CVs from morpho+policy
  → cell.compile()                          # lower paint/place into JAX state
  → cell.run(integrator, t_span, recorders) # JIT and step
```

All planned execution-layer work in §3.5 must respect this contract.

### 4.4 Testing

- pytest with `unittest.TestCase`; tests live next to source as
  `*_test.py` (exception: `io/swc/test.py`, `io/asc/test.py`).
- `conftest.py` forces `JAX_PLATFORMS=cpu` and `MPLBACKEND=Agg`.
- IO test fixtures live in `develop_doc/morpho_files/`.
- New code is expected to ship with co-located tests and to keep
  per-module test runtime under a few seconds on CPU.

### 4.5 Documentation

- All public classes / methods / functions use **NumPy-style
  docstrings** (see CLAUDE.md for the canonical template).
- Examples must be `.. code-block:: python` blocks compatible with
  doctest.
- High-level narrative documentation lives under `docs/`; design
  notebooks live under `develop_doc/`.

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
| Mechanisms | `CableProperties`, `DensityMechanism`, `PointMechanism*` | frozen dataclass | declaration | user |
| Mechanisms | `Ion`, `Channel`, `IonChannel`, `MixIons` | hybrid (JAX state) | per-cell | `Cell` |
| Discretization | `CV` | frozen | rebuilt on dirty | `Cell` |
| Discretization | `PaintRule`, `PlaceRule` | frozen | rebuilt on dirty | `Cell` |
| Topology | `PointTree`, `CVPoint`, `CVEdge` | frozen | rebuilt on dirty | `Cell` |
| Scheduling | `PointScheduling` | frozen | rebuilt on dirty | `Cell` |
| Runtime | `CellRuntimeState` | brainstate-managed | per-step | `Cell` |
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
- **Mechanism layer**: `CableProperties`, `DensityMechanism`,
  `PointMechanism`, `CurrentClamp`, `SineClamp`, `FunctionClamp`,
  `ProbeMechanism`, plus `mech.ion`, `mech.channel`, `mech.synapse`
  submodules.
- **Cell layer**: `Cell`, `CV`, `CVPolicy`, `CVPerBranch`, `MaxCVLen`,
  `DLambda`, `CVPolicyByTypeRule`, `CompositeByTypePolicy`,
  `PointTree`, `PointScheduling`.
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
cell = braincell.Cell(morpho, cv_policy=braincell.DLambda(0.1))

cell.paint(
    braincell.RegionExpr.everywhere(),
    braincell.CableProperties(
        membrane_capacitance=1.0 * (u.uF / u.cm**2),
        axial_resistivity=100.0 * (u.ohm * u.cm),
        resting_potential=-65 * u.mV,
    ),
)
cell.paint(soma_region, braincell.channel.INa_Ba2002(g_max=0.12 * u.S / u.cm**2))
cell.place(braincell.LocsetExpr.root(), braincell.CurrentClamp(amp=0.2 * u.nA, dur=50 * u.ms))
cell.place(braincell.LocsetExpr.terminals(), braincell.ProbeMechanism("v"))
```

### 7.3 Run a simulation (planned — see §3.5)

```python
trace = cell.run(
    integrator="staggered",
    t_span=(0 * u.ms, 100 * u.ms),
    dt=0.025 * u.ms,
)
braincell.vis.plot2d(cell, values=trace.v[-1])
```

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
| `NEURON` | dev only | reference comparator under `develop_doc/` |

Optional dependencies must be **lazily imported** so the base install
stays small. `quad/_diffrax.py` already follows this pattern via
`importlib.util.find_spec` plus PEP 562 `__getattr__`; visualization
backends should match it.

---

## 9. Milestones

The table below sequences the planned work in §3 into shippable
increments. Each milestone closes one user-visible capability gap.

### M1 — Runtime closure (highest priority)

Goal: a user can declare mechanisms on a multi-CV cell and call
`cell.run(...)` end-to-end.

- [ ] Allocate `DiffEqState` per (CV, mechanism) entry.
- [ ] Lower `PaintRule` / `PlaceRule` into a fused `dV/dt` + gating
  update consumed by `staggered_step`.
- [ ] Wire `dhs_voltage_step` into the staggered integrator inside the
  cell runtime.
- [ ] `ProbeMechanism` becomes a first-class trace recorder.
- [ ] Smoke tests: passive cable matches analytical exponential decay;
  Hodgkin–Huxley axon spikes within 5% of NEURON reference timing.

Acceptance: `examples/SC0X.py` runs end to end on a multi-branch cell;
notebook tutorial under `develop_doc/` reproduces the result.

### M2 — Filter expressivity

Goal: filter expressions can describe everything a NEURON `SectionList`
can.

- [ ] `RegionExpr.radius_range`.
- [ ] `RegionExpr.path_distance_range` (graph distance from soma or
  named anchor).
- [ ] `RegionExpr.euclidean_distance_range`.
- [ ] `RegionExpr.subtree(branch_or_locset)` plus interaction tests
  with the M3 tree-edit operations.
- [ ] `LocsetExpr.anchors(...)` and `LocsetExpr.fixed_step(...)`.
- [ ] `MorphoMetric` (or a sibling) caches per-branch path-distance
  and Euclidean-distance arrays so distance filters are O(1) per
  query.

### M3 — Morphology editing

Goal: users can build morphologies programmatically without round-
tripping through SWC.

- [ ] `Morphology.delete_subtree(branch)`.
- [ ] `Morphology.splice_subtree(parent, subtree)`.
- [ ] `Morphology.merge(other, at=...)`.
- [ ] `Morphology.translate / rotate / scale / align_principal_axis`.
- [ ] All edit operations invalidate `MorphoMetric` and force `Cell`
  rebuild on next access.
- [ ] Property tests over random small trees: `topo()` round-trip
  stable, `MorphoMetric` invariants hold.

### M4 — IO completeness

Goal: every common morphology format reads cleanly with a structured
report.

- [ ] Finish the ASC reader gaps (spines, contour somas, multi-tree
  files).
- [ ] NeuroML2 reader: cells, segment groups, biophysics, point
  mechanisms.
- [x] NeuroMorpho.Org client (search, download, cache, typed query and
  measurement) plus `Morphology.from_neuromorpho` and notebook
  walkthrough at `develop_doc/neuromorpho_diff.ipynb`.
- [ ] Promote the NeuroMorpho metric diff from the notebook to an
  automated pytest case using a small published reference corpus.
- [ ] Add a `Morphology.from_neuroml2(...)` constructor + tests.

### M5 — Mechanism validation and codegen

Goal: BrainCell channel implementations are quantitatively trusted
against NEURON `.mod` references and new channels can be generated
from NMODL.

- [ ] Promote `mech/mod_validate/` notebooks to automated pytest
  cases comparing voltage clamp traces against NEURON within tight
  tolerances.
- [ ] Lock down the `mech.spec` interface so generated code has a
  stable target.
- [ ] Land the NMODL → braincell codegen pass with at least three
  channels generated end to end.

### M6 — Visualization polish

Goal: publication-quality static plots and interactive 3D inspection
including time-series.

- [ ] Per-CV value overlays with proper colormaps, units, colorbars.
- [ ] Locset markers respecting region restriction.
- [ ] PyVista hover/pick mapping back to branch IDs.
- [ ] Time-series animation of `Cell.run` traces.

### M7 — Numerics hardening

Goal: every integrator in the registry is correctness- and
performance-tested.

- [ ] Order-of-accuracy convergence tests for every registered
  integrator on three reference ODEs.
- [ ] Adaptive timestep wrapper that turns any embedded RK pair into
  a registered integrator.
- [ ] Nightly benchmark suite (`CI-daily.yml`) comparing single-cell
  step time against NEURON / Arbor on Mainen / Hay / L5PC.

---

## 10. Technical Risks and Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Runtime layer is the critical path.** Without M1 the entire `cell` and `mech` stack is decorative. | Blocks every downstream milestone. | Treat M1 as the only P0; defer features (M2–M6) until the run loop is closed. |
| 2 | **Two-phase build (declare → rebuild → install) drift.** Stale CV caches can silently produce wrong results. | Correctness. | Centralize dirty flags in `Cell`; cover with regression tests that mutate declarations between runs and assert recompilation. |
| 3 | **Unit-handling regressions.** A path that drops units causes incorrect physics with no exception. | Correctness. | Keep `normalize_param` as the single chokepoint; add property tests that pass mixed-unit inputs and verify canonical-unit storage. |
| 4 | **JAX retracing on harmless mutations.** Recompiles dominate wall time on parameter sweeps. | Performance. | Separate "structure" (shapes, dtypes, mechanism set) from "parameters" (numeric values) at the `Cell.compile` boundary; only the former triggers retrace. |
| 5 | **Optional-dependency hazards.** Importing `braincell.quad` must not pay diffrax / pyvista cost. | Startup time and install footprint. | Preserve and extend the lazy-import pattern in `_diffrax.py`; add an import-time test that asserts neither diffrax nor pyvista is loaded after `import braincell`. |
| 6 | **Format heterogeneity in IO.** SWC / ASC / NeuroML2 each have edge cases that silently corrupt geometry. | Correctness. | Always return a `Report`; expand fixture corpus under `develop_doc/morpho_files/`; use NeuroMorpho diff (M4) as a wide regression net. |
| 7 | **Channel correctness vs NEURON.** Subtle gating bugs can pass smoke tests but produce wrong dynamics. | Scientific validity. | Promote `mech/mod_validate/` to CI (M5); compare voltage-clamp and current-clamp traces against `.mod` references. |
| 8 | **Mutability of `Morphology` colliding with planned tree edits.** Aliasing can corrupt cached metrics or in-flight `Cell` builds. | Correctness. | Tree edits go through copy-on-write helpers; `MorphoMetric` invalidation is mandatory on every mutator. |
| 9 | **DHS voltage solver scaling.** The branched-cable solve is the inner loop; regressions here regress every cell. | Performance. | Lock down a microbenchmark for `dhs_voltage_step` and run nightly (M7). |
| 10 | **API stability vs rapid evolution.** Public re-exports already exist but the runtime layer will reshape them. | User trust. | Treat §6 as the contract; mark anything else internal; use `_misc.deprecation_getattr` for renames. |

---

## 11. Glossary

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
