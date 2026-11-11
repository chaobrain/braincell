# Extend Models and Integrators

When adding a mechanism or integrator, choose an existing template, implement the model-specific equations, and verify it through the actual Cell execution path.
The table below gives a reading order: the public API first, then the templates and registration functions in the source. Complete signatures, equations, and runnable examples live in the docstrings.

| Extension | Public API | Templates and registration in the source |
| --- | --- | --- |
| Channel | [braincell.channel](../apis/braincell.channel.rst), [braincell.mech](../apis/braincell.mech.rst) | HH/Markov templates and a channel of the same family in `braincell/channel/`; `register_channel`; `Cell.paint` |
| Ion | [braincell.ion](../apis/braincell.ion.rst) | `braincell/ion/`, including the `KineticIon` template; `register_ion` |
| Synapse | [braincell.synapse](../apis/braincell.synapse.rst) | `braincell/synapse/exponential.py`; `register_synapse`; `Cell.place`, `Network.connect` |
| Integrator | [Integration](../apis/integration.rst) | a neighboring algorithm in `braincell/quad/`; `braintools.quad.register_method` |

## Add a Channel or Ion

1. Find the appropriate family and a similar model, then implement it in a suitable module under `braincell/channel/` or `braincell/ion/`.
2. Select a template (HH or Markov), declare parameters, states, and ion dependencies, and document the equation sources and local adjustments.
3. Register the concrete model with `register_channel` or `register_ion`. Update package exports and adjacent tests when adding it to the public model catalog.
4. First validate independent initialization, reset, derivatives, units, and current signs. Then paint the model onto a Cell and check binding, state shapes, and execution results.

For HH/Markov models, also validate steady-state gates or probability conservation. For dynamic ions, check concentrations, conservation constraints, and current-driven behavior.
Cite the equation sources in the model's docstring, in a References section.
Keep model-specific comparison configurations and results with the relevant validation workflow.

## Add a Synapse

Start with `braincell/synapse/exponential.py` and its adjacent tests. Define the internal state and event input contract, implement the dynamics, and register the model with `register_synapse`.
Deliver a real event through Cell.place and Network.connect.

Validate state changes around an event, aggregation of multiple events, current units and signs, decay trajectories, and reset.
For changes to Connection weight updates, open an issue first: whether such state belongs to the synapse model or to the connection is still an open design question.

## Add an Integrator

Choose a neighboring algorithm in `braincell/quad/` as an implementation and testing reference. A pure-function ODE stepper `step(f, y, t)` belongs in braintools (`braintools.quad.register_method(name, step, kind='ode', order=...)`); braincell wraps it automatically. A stateful step `step(target, *args) -> None` that needs braincell's models stays in `braincell/quad/` and is registered with `register_module_step` (from `braincell.quad._registry`), which declares its name, aliases and order in braintools' `'module'` kind and can be re-imported without error.
State whether it supports general ODEs or requires Cell's specialized voltage interface.

Use analytical solutions to check single-step error and multi-step convergence. Validate stage hooks, units, state shapes, and compiled loops.
Cell-specific algorithms also need coverage for multiple CVs, branching, and boundary inputs. A clamp or synapse on a branch end or branch point sits on a row without membrane capacitance: `staggered` solves that row with the CV rows, and `Cell.compute_derivative()` solves it at every call (three Newton steps, in `braincell/_multi_compartment/boundary.py`), so a step built on `compute_derivative` receives these inputs without extra work. Point inputs are assembled once, by `point_input_current` in `braincell/_multi_compartment/currents.py`; a new point mechanism that injects current adds itself there.

## Complete the contribution

Update the affected docstrings, documentation, and examples, and add a `changelog.md` entry for user-visible changes.
See the [Testing Guide](testing.md) for commands and fixtures and [Contributing](contributing.md#submit-a-pull-request) for submission requirements.
