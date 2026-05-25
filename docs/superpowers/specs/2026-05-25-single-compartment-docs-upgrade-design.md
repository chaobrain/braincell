# Single-Compartment Docs Upgrade — Design

Date: 2026-05-25
Status: Approved (pending spec review)

## Goal

Improve the presentation and content of `braincell`'s documentation:

1. Stop "folding" tutorial sections away in the landing page — surface them as
   expanded, captioned link-lists instead of `:hidden:` toctrees.
2. Remove the broken `advanced_tutorial/index` from `docs/index.rst` and
   relocate its two orphaned notebooks into the single-compartment tutorials.
3. Upgrade the single-compartment example notebooks: rename to descriptive
   filenames, fix bugs, and add new examples.

Non-goals: rewriting multi-compartment content, restructuring the Integration
section's internal pages, or any unrelated refactoring.

## Background (current state)

- `docs/index.rst` declares five top-level toctrees. Single-Compartment,
  Multi-Compartment, Numerical Integration, and API Documentation are all
  `:hidden:`; only "Advanced Tutorials" (`advanced_tutorial/index`) is visible.
- `docs/advanced_tutorial/index.rst` is broken: it points at
  `quickstart/concepts`, which does not exist under `advanced_tutorial/`
  (the real one lives at `single_compartment/quickstart/concepts.ipynb`).
  Its two actual notebooks — `differential_equation.ipynb` and
  `rationale.ipynb` (each a single markdown cell) — are referenced by nothing.
- `docs/single_compartment/examples/` holds `sc02`–`sc05` (no `sc01`); the
  toctree at `examples/index.rst` and the one inlined in
  `single_compartment/index.rst` both list them.
- Theme is `sphinx_book_theme`. With `:hidden:`, a section still appears in the
  left sidebar but renders no inline link-list in the page body; removing
  `:hidden:` adds the inline captioned list ("unfolding").

### Known bugs in examples (to fix, verified by execution)

- Deprecated channel aliases used throughout (`INa_*`, `IK_*`); canonical names
  are `Na_*` / `K_*` (deprecation aliases added in commit `d0a8882`). The
  landing-page quickstart snippet in `index.rst` uses the same deprecated names.
- `sc03` uses `self.na.add_elem(...)` while every other example and the
  quickstart use `.add(...)`. Both methods exist; `.add()` is the convention.
- `sc03` ends with a stray apologetic comment about a Jupyter interruption.
- `sc02`'s first markdown cell is empty.

### Constraints

- This environment runs JAX **CPU-only**. Notebooks must be executed to verify
  fixes and to capture committed cell outputs (plots) for the built docs.
- New examples must keep population sizes / durations modest so they run on CPU.
- Network-level projections (`AlignPostProj`, `Expon`, `COBA`,
  `EventFixedProb`) come from `brainpy.state` / `brainstate`; `braincell` itself
  provides only cell-level synapses (`AMPA`, `GABAa`, `NMDA`). `brainpy` 2.7.8
  is installed. Therefore the E-I network example legitimately depends on
  `brainpy` — the fix is to make imports clean and consistent, not to remove it.

## Design

### Part 1 — `docs/index.rst`: unfold tutorials (Option B)

- Remove `:hidden:` from the **Single-Compartment Modeling**,
  **Multi-Compartment Modeling**, and **Numerical Integration** toctrees so all
  three present as expanded captioned link-lists in the page body.
- Delete the **Advanced Tutorials** toctree block entirely (drops
  `advanced_tutorial/index`).
- Keep **API Documentation** `:hidden:` (an inline API file list is noise on the
  landing page; it remains in the sidebar).
- Update the landing-page quickstart code snippet to use canonical
  (non-deprecated) channel names, so the first code a reader sees is current.

### Part 2 — Relocate `advanced_tutorial/` content, delete the folder

- Move `differential_equation.ipynb` and `rationale.ipynb` into
  `docs/single_compartment/tutorial/`.
- Add both to the Tutorials toctree in `docs/single_compartment/index.rst`.
- Cleanup is light: ensure a proper top-level heading and that each renders
  correctly. Full multi-cell rewrites are out of scope (the "folded" concern was
  the hidden toctrees, not single-cell notebooks). Expand a notebook only if a
  section is clearly broken.
- Delete `docs/advanced_tutorial/` (both notebooks, the broken `index.rst`, and
  the now-empty folder).

### Part 3 — Examples: rename, fix bugs, add new

Location: `docs/single_compartment/examples/`.

**Rename (descriptive):**

| old        | new                            |
|------------|--------------------------------|
| `sc02`     | `calcium_channel_gating.ipynb` |
| `sc03`     | `ei_network.ipynb`             |
| `sc04`     | `integration_methods.ipynb`    |
| `sc05`     | `thalamic_neurons.ipynb`       |

**Bug fixes (each notebook executed to confirm):**

- Replace deprecated channel aliases with canonical names across all examples.
- Standardize `.add_elem(...)` → `.add(...)` in `ei_network`.
- Clean up `ei_network` imports (keep `brainpy` only where the network
  projection needs it) and remove the stray trailing comment.
- Fill in / remove `calcium_channel_gating`'s empty leading markdown cell.

**New examples (descriptive filenames):**

1. `hh_neuron_basics.ipynb` — build one HH neuron, inject a step current, plot
   membrane potential and spikes. The missing starter example.
2. `fi_curve.ipynb` — sweep injected current, measure firing rate, plot the
   frequency–current (F–I) curve.
3. `channel_ablation.ipynb` — scale/remove individual channels (e.g. drop `IK`
   or `Ih`) and show the effect on dynamics.
4. `spike_frequency_adaptation.ipynb` — Ca-dependent K (AHP) current driving
   spike-frequency adaptation under sustained drive.
5. `t_current_rebound.ipynb` — low-threshold (T-type) Ca rebound bursting after
   a hyperpolarizing step; hallmark of thalamic relay cells.

**Toctree ordering** (`examples/index.rst`, reflected in
`single_compartment/index.rst`), easiest → most advanced:

```
hh_neuron_basics
calcium_channel_gating
fi_curve
channel_ablation
spike_frequency_adaptation
t_current_rebound
integration_methods
thalamic_neurons
ei_network
```

## Affected files

- `docs/index.rst` — unfold toctrees, drop advanced_tutorial, fix quickstart snippet.
- `docs/single_compartment/index.rst` — add relocated tutorials, reorder examples.
- `docs/single_compartment/examples/index.rst` — new filenames + ordering.
- `docs/single_compartment/examples/*.ipynb` — renames, bug fixes, 5 new notebooks.
- `docs/single_compartment/tutorial/differential_equation.ipynb`,
  `docs/single_compartment/tutorial/rationale.ipynb` — relocated.
- `docs/advanced_tutorial/` — deleted.

## Verification

- Execute every new and modified example notebook on CPU; confirm no errors and
  no deprecation warnings; commit cell outputs.
- Build the docs (`make html` under `docs/`) and confirm: no broken toctree
  references (no `advanced_tutorial`), the three modeling/integration sections
  render unfolded on the landing page, relocated tutorials appear under
  single-compartment, and all renamed/new examples appear in order.

## Risks

- CPU-only execution may make the E-I network or sweeps slow; keep sizes modest.
- Canonical channel-name substitutions must be verified per channel (not every
  `I`-prefixed name maps 1:1) — confirm each at execution time.
