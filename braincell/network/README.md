# BrainCell Network

`braincell.network` separates cell-level connectivity from synapse-level
delivery. The public API has two main layers:

1. `Network.add_edges(...)` stores reusable pre/post cell pairs.
2. `Network.add_projection(...)` maps those edges to a postsynaptic synapse
   pool with weights, delays, and target selection policy.

This keeps graph construction inspectable and reusable. An edge set may exist
without any projection, and the same edge set may drive multiple projections.

## Basic Model

```python
import brainunit as u
import braincell

net = braincell.Network(name="demo")
net.add_population("E", E_cell)
net.add_population("I", I_cell)

net.add_edges(
    name="E_to_I",
    pre="E",
    post="I",
    method=braincell.network.pairs([(0, 1), (1, 1)]),
)

net.add_projection(
    name="E_to_I_exp",
    edges="E_to_I",
    synapse_pool="exp",
    weight=[0.25, 0.75] * u.uS,
    delay=0.0 * u.ms,
)
```

`pre` and `post` are population names. They may be the same population for
recurrent connectivity. `synapse_pool` is the name used when the postsynaptic
cell placed a synapse, for example
`cell.place(locs, Synapse("ExpSyn", ..., name="exp"))`.

## Edge Sets

`add_edges(...)` defines only cell-cell pairs. It does not choose synapse
locations, weights, or delays.

Supported edge builders:

```python
braincell.network.pairs([(0, 1), (1, 2)])
braincell.network.all_to_all(pre_indices="all", post_indices="all")
braincell.network.probability(p=0.1, seed=0)
```

The stored edge set is available by name:

```python
edges = net.edge_sets["E_to_I"]
print(edges.pre_index, edges.post_index)
```

Unused edge sets are allowed. They are just reusable graph declarations.

## Projections

`add_projection(...)` maps an edge set to postsynaptic synapse targets:

```python
projection = net.add_projection(
    name="E_to_I_exp",
    edges="E_to_I",
    synapse_pool="exp",
    target_policy="per_edge",
    number=1,
    replace=True,
    weight=0.5 * u.uS,
    delay=0.0 * u.ms,
)
```

Defaults are:

```python
target_policy = "per_edge"
number = 1
replace = True
```

If `synapse_pool="exp"` contains one placed target, every edge naturally shares
that target. If it contains multiple active targets, the projection samples from
that pool.

`net.proj["E_to_I_exp"]` and `net.projections["E_to_I_exp"]` both return the
stored projection object. `Projection(...)` remains available as a lower-level
object, but the recommended user API is direct `net.add_projection(...)`.

## Target Policies

`target_policy="per_edge"` treats each cell-level edge independently. Each edge
selects `number` targets from the post cell's `synapse_pool`.

`target_policy="by_post"` groups incoming edges by post cell and selects all
contacts for that post together.

`replace=True` allows repeated local synapse targets. `replace=False` prevents
repetition at the policy's scope:

- `per_edge`: no repeats within one edge's `number` targets.
- `by_post`: no repeats across all incoming contacts for the same post cell.

When `replace=False`, the pool must be large enough:

- `per_edge`: `number <= pool_size`
- `by_post`: `incoming_edge_count * number <= pool_size`

## Weights And Delays

`weight` is the event payload delivered to the synapse model. It should carry
the unit expected by that model. For `ExpSyn`, events increment conductance, so
weights should usually be conductance quantities such as `0.1 * u.uS`.

If projection `weight=None`, lowering uses the placed synapse model's default
`weight` parameter. An explicit projection weight overrides that default.

After projection expansion, each contact has:

```text
pre_index, post_index, synapse_pool, local_synapse_index, weight, delay
```

`weight` and `delay` may be:

- scalar
- shape `(n_edge,)`
- shape `(n_contact,)`

If `number > 1` and a parameter has shape `(n_edge,)`, each edge value is
repeated over that edge's expanded contacts.

`delay=0.0 * u.ms` currently means next-step delivery. The scan runtime still
supports only this zero-delay path; multi-step delays report a clear runtime
limitation.

## Lower-Level Connection

`Connection` is still available for explicit lowered connections:

```python
conn = braincell.network.Connection(
    pre_population="E",
    post_population="I",
    pre_index=[0, 1],
    post_index=[1, 1],
    synapse="exp",
    synapse_index=[0, 0],
    weight=[0.25, 0.75] * u.uS,
    delay=0.0 * u.ms,
)
net.add_connection(conn)
```

`synapse_index` is the local active-point index within the named placed synapse
layout. It defaults to `0`, which matches the common single-target case.

## Runtime Semantics

At each fixed step:

1. Pending arrivals are written into each target synapse's `pre_spike` buffer.
2. Every population prepares synapse inputs and advances dynamics.
3. Population spikes are sampled.
4. Each connection block gathers `spike[pre_index] * weight`.
5. Events are scatter-added to `(post_index, local_synapse_index)` for the next
   step.

For a shared target receiving edges `(0, j)`, `(1, j)`, `(2, j)`, the delivered
increment is:

```text
0_spike * 0_weight + 1_spike * 1_weight + 2_spike * 2_weight
```

For distinct local targets, the same scatter happens independently for each
`local_synapse_index`.
