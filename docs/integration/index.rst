Numerical Integration
=====================

Neural dynamics are systems of differential equations: membrane potentials,
gating variables, and ion concentrations all evolve in time according to a
right-hand side :math:`f(t, y)`. ``braincell`` turns these equations into
runnable simulations through two cooperating pieces:

- a small **protocol** — :class:`~braincell.DiffEqState` and
  :class:`~braincell.DiffEqModule` — that describes *what* to integrate, and
- a registry of **solvers** in :mod:`braincell.quad` that decide *how* to
  advance the system one step.

The same machinery drives both single-compartment and multi-compartment
models, so it lives here as its own topic rather than under either modeling
guide. The pages below build up from the mental model to writing your own
integrator.

.. toctree::
   :maxdepth: 1

   overview
   diffeq
   solvers
   advanced

For the exhaustive list of integrators and their signatures, see the
:doc:`API reference <../apis/integration>`.
