# Cerebellum ion import progress

## Current status

- Declarative `KineticIon` is now implemented in `braincell/ion/_base.py`.
- Public kinetic pieces exist: `Factor`, `Species`, `Reaction`, `Source`, `Conserve`, `KineticIon`.
- Runtime kinetic pieces exist: `_Specs`, `_Species`, `_Conserve`, `_Flux`.
- `MechanismProbe` now supports plain-value ion/mechanism fields in addition to `brainstate.State` fields.
- `Cell` now has `update_policy="legacy"` by default and a first `family_phased` path for `solver="staggered"`.
- Current-driven ion dynamics can consume a step-start cached current snapshot instead of re-evaluating a newer current later in the step.

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
- Current-driven ion dynamics no longer need to re-evaluate current later in the step when the family-phased path is used.
- `family_phased` ordering is currently:
  - cache ion current
  - voltage phase
  - channel phase
  - ion phase
- `legacy` remains the default and unchanged path.

## Scheduling semantics

The current scheduling work is motivated by NEURON's fixed-step / staggered
semantics: currents are evaluated from the old non-`V` state, voltage is
advanced first, and non-`V` states are advanced afterwards. NEURON does not
then keep alternating gate states and concentration states again inside the
same step.

The legacy BrainCell path does not make this family ordering explicit. The
main solver advances the default path first, and `node.update(...)` then runs
`IndependentIntegration` nodes afterwards. This can interleave channel and ion
updates according to the object tree rather than mechanism family.

That interleaving matters for concentration-dependent channels and
current-driven ion dynamics. A later independent channel may otherwise read an
already-updated `cai`, and a later independent ion may re-evaluate a newer
current after gate / `E` / `V` have already changed.

`update_policy` exists to make this scheduling choice explicit. `legacy`
preserves the old behavior unchanged. `family_phased` enforces a family order:
step-start ion current cache, then voltage, then all channels, then all ions.
At the moment this policy is only implemented for `Cell` with
`solver="staggered"`.

## What is still limited

- `family_phased` is currently implemented only for `Cell` and only when `solver == "staggered"`.
- `SingleCompartment` has not been migrated to the new update policy.
- The current-cache semantics are step-start splitting semantics, not full stage-by-stage current consistency.
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

The first NEURON-aligned scheduling validation should use a simple current-driven ion-dynamics example, not a full reaction-network pool.

Recommended first target:

- `examples/neuron_compare/Cerebellum_mod/DCN/ion/CdpHVA_SU15_DCN.mod`

This should ideally be paired with the corresponding DCN calcium channel so the comparison exercises:

- channel reads old concentration
- ion dynamics reads step-start cached current

Validation goal:

- build one minimal NEURON-vs-BrainCell comparison case under the new `family_phased` policy
- verify that scheduling-sensitive observables behave as intended before importing larger `KINETIC` ion mechanisms

If that DCN simple ion-dynamics comparison is sound, the next concrete import target should be the first complex reaction-network mechanism:

- `CdpStC`

## Assumptions

- This file is now a compressed progress record, not a full design notebook.
- Previous exploratory detail has been intentionally replaced by short status and next-step notes.
- The first external validation target should be the simplest current-driven DCN ion-dynamics mechanism before moving on to `CdpStC`, `CdpCR`, or `CdpCAM`.
