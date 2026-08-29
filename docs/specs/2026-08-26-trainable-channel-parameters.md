# Trainable Channel Parameters

## Goal

Deliver a complete Cell-local trainable-parameter path for `IL`, `Na_HH1952`,
and `K_HH1952`. The path covers explicit field schemas, pre-init View access,
compact runtime parameter storage, direct/shared/function parameter sources,
BrainState-discoverable roots, and differentiable materialization before Cell
lifecycle entry points.

## Public Contract

- The three migrated Channels declare complete `parameters`, `states`, and
  `derived` mappings; trainability remains a binding policy rather than a
  property of the physical schema object.
- `ChannelView.trainable()` accepts `braincell.trainable.parameter()`,
  `scale()`, and `parameterized()` sources.
- Grouping supports `row`, `population`, `cv`, and `all`.
- `Cell.trainables.parameters()` exposes the original `ParamState` objects and
  atomic physical/raw value setters.
- `ChannelView.<field>` is a read shortcut for `get(<field>)`; string indexing
  remains mechanism selection.
- Network aggregation, Ion, Synapse, Connection, dataset, loss, optimizer, and
  checkpoint APIs are deferred.

## Runtime Invariants

- Only roots are `ParamState`; materialized physical Channel parameters are
  non-trainable long-term states.
- Parameter storage records axis semantics explicitly as uniform, population,
  CV, or row; optimizer steps never change that representation.
- Existing non-schema Channels keep the legacy declaration/lowering path.
- A binding owns its selected row/field pairs and ordinary `set()` cannot
  overwrite them.
- Materialization evaluates every binding before committing any runtime write.
- `reset_state()` does not reset roots or scale baselines; full `reset()` keeps
  binding metadata and resolves it again on the next initialization.

## Acceptance

- Existing Channel tests remain numerically unchanged.
- Schema defaults are visible before initialization even when omitted from
  `mech.Channel(...)`.
- Two-CV leak direct and scale formulations agree in forward values and
  chain-rule gradients.
- Sodium and potassium can share one factor and expose one `ParamState`.
- A context function with two scalar coefficients exposes two degrees of
  freedom and remains differentiable through a cached JIT rollout.
