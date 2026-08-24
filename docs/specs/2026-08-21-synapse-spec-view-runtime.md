# Synapse Spec, View, and Runtime

## Summary

Point synapses use three user-visible concepts:

1. `braincell.mech.SynapseSpec` is the immutable placement declaration.
2. `braincell.SynapseView` selects stable logical instances.
3. Registered runtime synapse classes own executable SoA parameters and states.

The Cell owns a private column store between declaration and runtime. Users do
not inspect or depend on that storage class.

## Public contract

```python
exp = braincell.mech.SynapseSpec(
    "ExpSyn",
    name="parallel_fiber",
    tau=2.0 * u.ms,
    e=0.0 * u.mV,
)
cell.place(locations, exp)

all_synapses = cell.synapses
pf = cell.synapses["parallel_fiber"]
cell_3_pf = cell[3].synapses["parallel_fiber"]
first = pf[0]                 # length-one SynapseView
exp_rows = cell.synapses.by_type("ExpSyn")
```

`name` is a logical view group. The same name may be extended across placement
calls only when its registered type is unchanged. Runtime grouping ignores
name, location, parameter values, and creation source: one Cell has one flat
runtime node per `synapse_type`.

Logical ids are stable and distinct even when rows share location and parameter
values. They are not required to be dense or equal to view row positions.
Connections capture those ids and resolve them to runtime rows during lowering.

`weight` and `delay` are Connection fields, not synapse parameters. A runtime
synapse declares the payload it consumes through `event_input`; for `ExpSyn`
and `Exp2Syn`, Connection weight must therefore be conductance-compatible.

## Placement broadcasting

A single locset broadcasts to all selected cells. A sequence supplies one
locset per cell and may be ragged or contain empty rows.

Rectangular parameters accept scalar, `(L,)`, `(P, 1)`, `(P, L)`, or flat
`(P * L,)` values. Ragged parameters accept scalar, `(P, 1)`, a ragged outer
sequence, or a flat `(sum(L_i),)` value. Logical rows are cell-major.

## Mutation lifecycle

Before initialization, `view.set(...)` changes declaration-time parameter
columns. After initialization it updates the corresponding runtime parameter
rows. `brainstate.nn.Param` wrappers remain in place. Dynamic state mutation is
explicit and separate:

```python
view.set(tau=3.0 * u.ms)
view.set_state(g=0.0 * u.uS)
```

`set_state()` is available only after initialization. Initialization freezes
topology, locations, names, types, and logical ownership.

The model schema describes physical role, default, unit, and validity only. It
does not decide whether a parameter is trainable or prescribe an optimizer
transform. Training code may wrap selected runtime columns in
`brainstate.nn.Param`; view updates preserve that wrapper.

## Runtime invariants

- Runtime parameter/state storage is flat over actual logical instances; it
  does not pad each cell to a maximum synapse count.
- `population_index` and `point_index` route current back to membrane arrays.
- name, type, and logical-id maps remain available through `SynapseView`.
- manual placement, explicit Connection targets, and Network-created pools all use
  the same Cell-owned logical store.
- model classes declare `parameters`, `states`, `derived`, and `event_input`
  explicitly; constructor reflection is not part of the storage contract.
- event routing accumulates into one private buffer per runtime synapse type,
  and the model consumes that payload through `apply_events()`.
- point currents are total inward-positive currents. The cell runtime divides
  them by point area exactly once before adding them to membrane equations.

`AMPA`, `GABAa`, and `NMDA` remain importable names but are temporarily
unavailable until their transmitter-pulse event semantics are expressed by the
same contract. `ExpSyn` and `Exp2Syn` are the supported v1 runtime models.
