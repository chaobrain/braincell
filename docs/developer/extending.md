# Extend Models and Integrators

When adding a mechanism or integrator, choose an existing template, implement the model-specific equations, and verify it through the actual Cell execution path.
The table below gives a reading order. The relevant Design Current documents maintain complete signatures, equations, and runnable model examples.

| Extension | Templates and contracts | Registration and integration |
| --- | --- | --- |
| Channel | [HH/Markov templates](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/current/api.md), [template validation](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/current/template-invariants.md) | [Mech registration](https://github.com/chaobrain/braincell/blob/main/docs/design/mech/current/api.md#注册与事件契约), [Cell.paint](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/api.md#paint-与-place) |
| Ion | [Public ion models](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/current/api.md), [lifecycle and KineticIon extension template](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/current/kinetic-ion-api.md) | [Current snapshots and scheduling](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/architecture.md#离子电流快照与调度) |
| Synapse | [Dynamics and event contracts](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/current/api.md), [state ownership](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/current/architecture.md) | [Event sources and connections](https://github.com/chaobrain/braincell/blob/main/docs/design/network/current/connections.md) |
| Integrator | [Integrator registration and calls](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/current/api.md), [target protocol](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/api.md#积分协议) | [Solver paths](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/current/architecture.md) |

## Add a Channel or Ion

1. Find the appropriate family and a similar model, then implement it in a suitable module under `braincell/channel/` or `braincell/ion/`.
2. Select a template from Current, declare parameters, states, and ion dependencies, and document the equation sources and local adjustments.
3. Register the concrete model with `register_channel` or `register_ion`. Update package exports and adjacent tests when adding it to the public model catalog.
4. First validate independent initialization, reset, derivatives, units, and current signs. Then paint the model onto a Cell and check binding, state shapes, and execution results.

For HH/Markov models, also validate steady-state gates or probability conservation. For dynamic ions, check concentrations, conservation constraints, and current-driven behavior.
Reusable source references belong in the
[Ion/Channel bibliography](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/references/ion-channel-bibliography.md).
Keep model-specific comparison configurations and results with the relevant validation workflow.

## Add a Synapse

Start with `braincell/synapse/exponential.py` and its adjacent tests. Define the internal state and event input contract, implement the dynamics, and register the model with `register_synapse`.
Deliver a real event through Cell.place and Network.connect.

Validate state changes around an event, aggregation of multiple events, current units and signs, decay trajectories, and reset.
For changes to Connection weight updates, first read the
[plasticity discussion](https://github.com/chaobrain/braincell/blob/main/docs/design/network/proposals/connection-plasticity.md)
and determine whether the state belongs to the synapse model or the connection.

## Add an Integrator

Choose a neighboring algorithm in `braincell/quad/` as an implementation and testing reference. Implement the step function according to the target protocol, and declare its name, aliases, and order with `register_integrator`.
State whether it supports general ODEs or requires Cell's specialized voltage interface.

Use analytical solutions to check single-step error and multi-step convergence. Validate stage hooks, units, state shapes, and compiled loops.
Cell-specific algorithms also need coverage for multiple CVs, branching, and boundary inputs. See the existing
[boundary-input proposal](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/proposals/explicit-solver-boundary-inputs.md).

## Complete the contribution

Update the relevant Current documents and actual related examples, and link completed work or remaining issues in the module TODO.
See the [Testing Guide](testing.md) for commands and fixtures and [Contributing](contributing.md#submit-a-pull-request) for submission requirements.
