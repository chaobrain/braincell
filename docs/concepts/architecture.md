# Architecture

`braincell` separates **what** you want to model from **how** it is executed.
This separation is the single most important idea in the library: you write a
declarative description of a neuron, and `braincell` lowers it through several
layers into a fast, differentiable JAX computation.

## The layered design

```text
   ┌─────────────────────────────────────────────────────────┐
   │  Declaration  (what to model)                            │
   │    • Morphology        geometry: branches, radii, tree   │
   │    • mech.*            channels, ions, clamps, synapses  │
   │    • filter.*          regions & locsets (where)         │
   └───────────────────────────┬─────────────────────────────┘
                               │  paint / place
   ┌───────────────────────────▼─────────────────────────────┐
   │  Discretization  (_cv)                                   │
   │    • CV               one isopotential control volume    │
   │    • CVPolicy         how many CVs each branch gets      │
   └───────────────────────────┬─────────────────────────────┘
                               │  build
   ┌───────────────────────────▼─────────────────────────────┐
   │  Runtime  (_compute)                                     │
   │    • PointTree        execution graph over CVs           │
   │    • CellRuntimeState frozen, JAX-friendly state         │
   └───────────────────────────┬─────────────────────────────┘
                               │  step
   ┌───────────────────────────▼─────────────────────────────┐
   │  Integration  (quad)                                     │
   │    • DiffEqModule     defines f(t, y)                    │
   │    • solver           advances y by dt                   │
   └─────────────────────────────────────────────────────────┘
```

Each layer has a clear job:

- **Declaration** is pure data. A {mod}`braincell.mech` `Channel` or `Ion`
  knows *nothing* about JAX, time, or state — it just records "install this
  here, with these parameters." This is what makes models easy to inspect,
  compose, and serialize.
- **Discretization** turns continuous geometry into a finite set of
  {class}`~braincell.CV` (control volumes). A {class}`~braincell.CVPolicy`
  decides the resolution. See {doc}`discretization`.
- **Runtime** binds the declared mechanisms to the discretized geometry and
  produces a frozen state object that JAX can trace and differentiate.
- **Integration** advances the state in time using a solver from
  {mod}`braincell.quad`. See {doc}`integration`.

## Why this matters

This architecture gives `braincell` several properties that are hard to get
otherwise:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Property
  - How the design delivers it
* - **Differentiability**
  - The runtime state is a JAX pytree, so gradients flow through an entire
    simulation — you can fit parameters with gradient descent.
* - **Hardware portability**
  - Integration compiles to XLA; the same model runs on CPU, GPU, or TPU.
* - **Inspectable models**
  - Because declarations are plain data, you can print, diff, and reason about
    a model before it ever runs.
* - **Composability**
  - Mechanisms are painted/placed independently, so complex cells are built up
    from small, reusable declarations.
```

## Two front ends, one engine

Both cell classes share the layers above:

- {class}`braincell.SingleCompartment` collapses the morphology and
  discretization layers — there is exactly one compartment — and exposes ions
  and channels added imperatively in `__init__`.
- {class}`braincell.Cell` uses the full pipeline: a {class}`~braincell.Morphology`
  decorated with {mod}`braincell.mech` declarations through `paint` and `place`.

They both subclass {class}`braincell.HHTypedNeuron`, so the integrator, ion, and
channel abstractions are identical. Learn one and the other is familiar.

## Where the code lives

If you want to read the source, the layers map onto packages (all internal,
re-exported through the top-level namespace):

| Layer | Package |
|-------|---------|
| Declaration — cells | `braincell._single_compartment`, `braincell._multi_compartment` |
| Declaration — mechanisms | `braincell.mech` |
| Declaration — selection | `braincell.filter` |
| Geometry | `braincell.morph` |
| Discretization | `braincell._cv` |
| Runtime | `braincell._compute` |
| Integration | `braincell.quad` |

See the {doc}`../developer/project_layout` for the full map.
