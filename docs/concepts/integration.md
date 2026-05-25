# Integration

A neuron model is a system of differential equations: the membrane potential,
gating variables, and ion concentrations all evolve in time according to a
right-hand side $f(t, y)$. **Integration** is the act of advancing that system
forward in time. `braincell` keeps this concern in its own module,
{mod}`braincell.quad`, shared by both cell types.

This page is the conceptual overview. For hands-on use — comparing solvers,
writing your own — see the full {doc}`../integration/index` guide.

## The protocol: describing the equations

Two small abstractions describe *what* to integrate:

- {class}`braincell.DiffEqState` — a state variable that carries its own
  derivative (and, for cable diffusion, a diffusion term).
- {class}`braincell.DiffEqModule` — a module that exposes a right-hand side over
  such states.

Cells implement this protocol, so any solver knows how to step them. You rarely
touch it unless you are building a new dynamical component.

## The registry: choosing a solver

`braincell.quad` holds a **registry of solvers**, each selectable by name. You
pick one when constructing a cell:

```python
import braincell
hh = braincell.SingleCompartment(size, solver='ind_exp_euler')
```

The available integrators span the usual families:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Family
  - Members (by name)
* - **Explicit Runge–Kutta**
  - `euler`, `midpoint`, `rk2`, `rk3`, `rk4`, `heun2`, `heun3`, `ssprk3`,
    `ralston2/3/4`
* - **Exponential Euler**
  - `exp_euler`, `ind_exp_euler`, `exp_exp_euler`
* - **Implicit**
  - `backward_euler`, `implicit_euler`, `implicit_exp_euler`, `implicit_rk4`
* - **Composite / cable**
  - `cn_exp_euler`, `cn_rk4` (Crank–Nicolson), `splitting`, `staggered`,
    `dhs_voltage`
```

You can list everything available at runtime:

```python
import braincell.quad as quad
sorted(quad.all_integrators)
```

## Choosing wisely: stiffness

Biophysical neuron models are **stiff** — they mix very fast (sodium spike) and
slow (calcium, adaptation) timescales. Explicit solvers like `rk4` must take
tiny time steps to stay stable on stiff systems, which is slow.

```{tip}
For single-compartment HH-style models, **exponential-Euler** variants
(`exp_euler`, `ind_exp_euler`) handle the stiff gating kinetics stably at
practical step sizes and are a good default. For multi-compartment cable
equations, the **staggered** / Crank–Nicolson solvers are designed for the
spatial coupling.
```

## Differentiability

Because every solver is written in JAX, the entire time loop is differentiable.
Gradients of an output (a spike count, a voltage trace, a loss) with respect to
parameters (conductances, time constants) flow straight back through the
integration — this is what makes gradient-based model fitting possible.

## Writing your own integrator

The registry is open: decorate a stepping function with
`@register_integrator` and it becomes selectable by name like any built-in. The
{doc}`../integration/advanced` guide walks through this.

## See also

- {doc}`../integration/index` — the full integration guide (overview, protocol,
  solvers, advanced).
- {doc}`../apis/integration` — every integrator and its signature.
- {doc}`discretization` — what the cable solvers operate on.
