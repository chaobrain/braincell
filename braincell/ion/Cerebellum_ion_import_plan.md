# Cerebellum ion import progress

## Current status

- Declarative `KineticIon` is now implemented in `braincell/ion/_base.py`.
- Public kinetic pieces exist: `Factor`, `Species`, `Reaction`, `Source`, `Conserve`, `KineticIon`.
- Runtime kinetic pieces exist: `_Specs`, `_Species`, `_Conserve`, `_Flux`.
- `MechanismProbe` now supports plain-value ion/mechanism fields in addition to `brainstate.State` fields.

## Files changed

### Ion template and lifecycle

- `braincell/ion/_base.py`
- `braincell/_base_ion.py`

### Multi-compartment scheduling and runtime

- `braincell/_multi_compartment/cell.py`
- `braincell/_compute/runtime.py`

### Probe behavior

- `braincell/_multi_compartment/probes.py`

### Tests

- `braincell/ion/_base_test.py`
- `braincell/_base_ion_test.py`
- `braincell/_multi_compartment/cell_test.py`
- `braincell/_multi_compartment/probes_test.py`
- `braincell/_compute/runtime_test.py`

## What is now supported

- `KineticIon` supports diffeq/algebraic species, `Conserve`, factor-based visible/amount conversion, and resolved full species views.
- `Ci` is a reserved species name and still feeds the standard `Ion.pack_info()` path.
- Current-driven ion dynamics can optionally reuse a precomputed total-current snapshot when one is provided by the caller.

## Scheduling semantics

The current scheduling work still follows the existing BrainCell update path:
the cell solver advances the voltage step first, and `node.update(...)` then
runs independently-integrated ion or channel nodes afterwards.

## What is still limited

- `SingleCompartment` still uses its own update path and has not been extended for the new kinetic-ion template work.
- The new scheduling has only been validated with internal synthetic tests so far, not yet against a real NEURON ion-dynamics example.
- The complex Cerebellum `KINETIC` ion mechanisms (`CdpStC`, `CdpCR`, `CdpCAM`) are not yet imported as concrete models.

## Tests already run

- `braincell/_multi_compartment/cell_test.py`
- `braincell/_compute/runtime_test.py`
- `braincell/_base_test.py`
- `braincell/ion/_base_test.py`
- `braincell/_base_ion_test.py`

These all passed after the current round of changes.

## Next step

The first external validation should use a simple current-driven ion-dynamics example, not a full reaction-network pool.

Recommended first target:

- `examples/neuron_compare/Cerebellum_mod/DCN/ion/CdpHVA_SU15_DCN.mod`

This should ideally be paired with the corresponding DCN calcium channel so the comparison exercises:

- channel / ion coupling under the existing BrainCell update order

Validation goal:

- build one minimal NEURON-vs-BrainCell comparison case
- verify that scheduling-sensitive observables behave acceptably before importing larger `KINETIC` ion mechanisms

If that DCN simple ion-dynamics comparison is sound, the next concrete import target should be the first complex reaction-network mechanism:

- `CdpStC`

## Assumptions

- This file is now a compressed progress record, not a full design notebook.
- Previous exploratory detail has been intentionally replaced by short status and next-step notes.
- The first external validation target should be the simplest current-driven DCN ion-dynamics mechanism before moving on to `CdpStC`, `CdpCR`, or `CdpCAM`.
