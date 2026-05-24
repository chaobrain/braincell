# Cerebellum ion import progress

## Current status

- Declarative `KineticIon` is now implemented in `braincell/ion/_base.py`.
- Public kinetic pieces exist: `Factor`, `Species`, `Reaction`, `Source`, `Conserve`, `KineticIon`.
- Runtime kinetic pieces exist: `_Specs`, `_Species`, `_Conserve`, `_Flux`.
- `MechanismProbe` now supports plain-value ion/mechanism fields in addition to `brainstate.State` fields.
- Concrete Cerebellum calcium-pool ions are now imported in
  `braincell/ion/calcium.py`: `CdpStC_MA2020_GoC`,
  `CdpStC_NoCAM_MA2020_GoC`, `CdpStC_CAMOnly_MA2020_GoC`,
  `CdpStC_MA2025_BC`, `CdpStC_RI2021_SC`, `CdpCAM_MA2024_PC`, and
  `CdpCR_MA2020_GrC`.
- PC MA2024 channel imports and targeted tests have been added across
  sodium, potassium, calcium, calcium-activated potassium, and HCN
  channel modules.
- The PC MA2024 cell-comparison scaffold lives under
  `examples/neuron_compare/cell/pc_ma2024`, with a simplified NEURON
  assembly, a matching BrainCell assembly, shared parameters, debug
  versions, and a side-by-side run notebook.

## Files changed

### Ion template and lifecycle

- `braincell/ion/_base.py`
- `braincell/_base_ion.py`

### Multi-compartment scheduling and runtime

- `braincell/_multi_compartment/cell.py`
- `braincell/_compute/runtime.py`
- `braincell/quad/_staggered.py`

### Cerebellum ion and channel imports

- `braincell/ion/calcium.py`
- `braincell/channel/sodium.py`
- `braincell/channel/potassium.py`
- `braincell/channel/calcium.py`
- `braincell/channel/potassium_calcium.py`
- `braincell/channel/hyperpolarization_activated.py`

### Probe behavior

- `braincell/_multi_compartment/probes.py`

### NEURON comparison examples

- `examples/neuron_compare/ion/`
- `examples/neuron_compare/channel_no_conc/`
- `examples/neuron_compare/cell/pc_ma2024/`
- `examples/neuron_compare/Cerebellum_mod/`

### Tests

- `braincell/ion/_base_test.py`
- `braincell/_base_ion_test.py`
- `braincell/_multi_compartment/cell_test.py`
- `braincell/_multi_compartment/probes_test.py`
- `braincell/_compute/runtime_test.py`

## What is now supported

- `KineticIon` supports diffeq/algebraic species, `Conserve`, factor-based visible/scaled conversion, and resolved full species views.
- `Ci` is a reserved species name and still feeds the standard `Ion.pack_info()` path.
- Current-driven ion dynamics can optionally reuse a precomputed total-current snapshot when one is provided by the caller.
- `Cell(..., cache_ion_total_current=True)` snapshots per-ion total
  current at the start of the staggered step, before voltage or ion
  state advances. This restores the NEURON meaning for current-driven
  calcium pools: the ion mechanism consumes the channel current from a
  stable per-step snapshot instead of recomputing through partially
  updated states.
- `Cell(..., ion_channel_update_order="family")` selects the
  NEURON-like family schedule. `"integration"` keeps the original
  BrainCell integration-oriented schedule for comparison and backwards
  behavior checks.
- Same-name channel instances painted onto disjoint layouts are kept
  distinct internally, so soma and dendrite can both use the same
  channel class/name without one layout overwriting the other inside
  `Ion.channels`.
- PC calcium channel `_Frozen` variants stop differentiation through
  the voltage used by the current expression. This is a local
  compatibility path for reproducing the NEURON scheduling semantics of
  the imported mechanisms.

## Scheduling semantics

There are now two explicit post-voltage ion/channel schedules:

- `ion_channel_update_order="family"` is the NEURON-compatibility mode.
  It updates by ion family so ion dynamics and their attached channels
  see the same grouping assumptions as the source MOD mechanisms.
- `ion_channel_update_order="integration"` is the original BrainCell
  behavior. It follows the integration grouping used before the
  NEURON-alignment work and is useful as a comparison/debug mode.

For NEURON comparison runs that include current-driven calcium pools, use
`cache_ion_total_current=True` together with
`ion_channel_update_order="family"`.

## What is still limited

- `SingleCompartment` still uses its own update path and has not been extended for the new kinetic-ion template work.
- The PC MA2024 full-cell scaffold is present, but it is still a live
  validation target: numerical differences against NEURON should be
  tracked through the notebooks before treating it as a regression
  baseline.
- Callable spatial parameter expressions are still only documented as a
  future filter/paint direction; paint values currently need explicit
  arrays or per-region calls.
- Not every Cerebellum MOD file has a BrainCell counterpart yet. The
  current completed focus is the ion/channel subset needed by the PC
  and calcium-pool comparisons.

## Tests already run

- Targeted scheduling/runtime tests around `cache_ion_total_current`,
  same-name channel layouts, and `ion_channel_update_order`.
- Targeted ion tests for the imported `CdpStC`, `CdpCAM`, and `CdpCR`
  kinetic-ion classes.
- Targeted channel tests for the PC MA2024 channel imports and the
  `_Frozen` voltage-current variants.
- NEURON-vs-BrainCell comparison notebooks/scripts under
  `examples/neuron_compare/ion` and `examples/neuron_compare/cell/pc_ma2024`
  are being used as the external validation path.

## Next step

- Keep tightening the PC MA2024 comparison in
  `examples/neuron_compare/cell/pc_ma2024/run.ipynb`.
- Promote the stable parts of the notebook comparisons into automated
  regression tests once the expected tolerances are clear.
- Continue importing the remaining Cerebellum MOD mechanisms only after
  the PC calcium-pool/channel scheduling path is stable.

## Assumptions

- This file is now a compressed progress record, not a full design notebook.
- Previous exploratory detail has been intentionally replaced by short status and next-step notes.
- PC MA2024 is the current end-to-end validation target for the
  channel/ion scheduling work.
