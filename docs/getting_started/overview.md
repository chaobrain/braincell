# Overview & Roadmap

This page is a map. It explains what `braincell` does, how the pieces fit
together, and — most importantly — **where to go next** depending on what you
want to accomplish.

## What problem does `braincell` solve?

Computational neuroscience often needs to simulate the electrical behavior of
neurons in biophysical detail: the flow of ions through voltage-gated channels,
the spread of voltage along dendrites, and the resulting spikes. `braincell`
provides this in a way that is:

- **Detailed** — from a single isopotential compartment to a full
  morphological reconstruction with hundreds of compartments.
- **Fast and portable** — every model compiles with JAX and runs on CPU, GPU,
  or TPU without code changes.
- **Differentiable** — gradients flow through whole simulations, so you can fit
  parameters with gradient descent.

## One cell class, two regimes

Almost everything in `braincell` is organized around one cell class,
{class}`braincell.Cell`, used in two regimes:

```{list-table}
:header-rows: 1
:widths: 22 38 40

* - Regime
  - Use it when…
  - Built from
* - Point neuron, {meth}`braincell.Cell.point`
  - the neuron can be treated as a single isopotential point (a "ball"), or you
    are prototyping channel dynamics.
  - ion species and channels from {mod}`braincell.mech`, painted onto its one
    compartment.
* - Morphological cell, `braincell.Cell(morphology)`
  - dendritic geometry matters — voltage attenuation, distributed channels,
    synaptic integration.
  - a {class}`~braincell.Morphology` decorated declaratively with
    {mod}`braincell.mech` mechanisms via `paint` / `place`.
```

Both are declared with `paint` and `place` and share the same ion, channel,
and integrator machinery (`Cell` derives from {class}`braincell.HHTypedNeuron`)
— so concepts you learn for one transfer to the other.

## The pipeline

A multi-compartment model moves through four conceptual stages. Understanding
this pipeline makes the rest of the docs click into place:

```text
  Morphology         Cell            Discretization        Runtime + Solver       RunResult
  (geometry)  ──▶  (declaration) ──▶ (control volumes) ──▶  (time integration) ──▶ (traces)
                   paint / place
```

1. **Morphology** — the geometry (branches, radii, connectivity), loaded from a
   file or built programmatically. See {doc}`../concepts/morphology`.
2. **Declaration** — you create a {class}`~braincell.Cell` and *paint* density
   mechanisms (channels, ions, cable properties) onto **regions** and *place*
   point mechanisms (clamps, synapses, probes) at **locations**. See
   {doc}`../concepts/mechanisms` and {doc}`../concepts/regions_locsets`.
3. **Discretization** — the continuous cable is divided into **control volumes**
   (CVs) by a {class}`~braincell.CVPolicy`. See {doc}`../concepts/discretization`.
4. **Integration** — a solver from {mod}`braincell.quad` advances the system in
   time, producing a `RunResult` of probe traces. See
   {doc}`../concepts/integration`.

A point neuron skips the morphology and discretization stages: the cell *is*
one compartment.

## Where to go next

```{list-table}
:header-rows: 1
:widths: 40 60

* - If you want to…
  - Go to
* - Run something immediately
  - {doc}`first_steps`
* - Understand the vocabulary and design
  - {doc}`../concepts/architecture`
* - Build and simulate point neurons
  - {doc}`../tutorials/index`
* - Build morphological cells
  - {doc}`../tutorials/index`
* - Pick or write a numerical solver
  - {doc}`../integration/index`
* - Load a morphology from a file
  - {doc}`../file_formats/index`
* - Look up an exact class or function
  - {doc}`../apis/braincell`
* - Contribute or extend the library
  - {doc}`../developer/index`
* - Diagnose an error
  - {doc}`../developer/troubleshooting`
```
