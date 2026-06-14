# BrainCell Network

`braincell.network` separates cell-level connectivity from synapse-level
delivery. The public API has two main layers:

1. `Network.add_edges(...)` stores reusable pre/post cell pairs.
2. `Network.add_projection(...)` maps those edges to a postsynaptic synapse
   pool with a contact-generation method, weights, and delays.

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
    synapse="exp",
    weight=[0.25, 0.75] * u.uS,
    delay=0.0 * u.ms,
)
```

`pre` and `post` are population names. They may be the same population for
recurrent connectivity. `synapse` is the name used when the postsynaptic
cell placed a synapse, for example
`cell.place(locs, Synapse("ExpSyn", ..., name="exp"))`.

## Edge Sets

`add_edges(...)` defines only cell-cell pairs. It does not choose synapse
locations, weights, or delays.

Supported edge builders:

```python
braincell.network.pairs([(0, 1), (1, 2)])
braincell.network.dense([[0, 1], [1, 0]])
braincell.network.all_pairs(pre_indices="all", post_indices="all")
braincell.network.probability(p=0.1, seed=0)
```

Custom callables are also accepted. They receive ``n_pre`` and ``n_post`` from
the selected populations and return sparse ``(pre_index, post_index)`` arrays.

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
    synapse="exp",
    method=braincell.network.per_edge(number=1, replace=True),
    weight=0.5 * u.uS,
    delay=0.0 * u.ms,
)
```

The default method is:

```python
method = braincell.network.per_edge(number=1, replace=True)
```

If `synapse="exp"` contains one placed target, every edge naturally shares
that target. If it contains multiple active targets, the projection samples from
that pool.

`net.proj["E_to_I_exp"]` and `net.projections["E_to_I_exp"]` both return the
stored projection object. `Projection(...)` remains available as a lower-level
object, but the recommended user API is direct `net.add_projection(...)`.

## Contact Methods

`method=per_edge(...)` treats each cell-level edge independently. Each edge
selects `number` targets from the post cell's `synapse`.

`method=by_post(...)` groups incoming edges by post cell and selects all
contacts for that post together. `method=explicit_contacts(...)` accepts a
ready sparse contact table.

`replace=True` allows repeated local synapse targets. `replace=False` prevents
repetition at the method's scope:

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
- callable accepting a `ProjectionContactContext` and returning one of the
  above

If `number > 1` and a parameter has shape `(n_edge,)`, each edge value is
repeated over that edge's expanded contacts.

Custom contact methods are plain callables accepting a `ProjectionEdgeContext`
and returning `ContactTable(source_edge=..., synapse_index=...)`. Parameter
callables receive `ProjectionContactContext`, which includes selected edge ids,
pre/post cell ids, contact-to-edge mapping, local synapse indices, pool size,
and the target synapse name.

`delay=0.0 * u.ms` means next-step delivery. Positive delays are delivered by a
fixed-step ring buffer. `Network.run(...)` quantizes non-grid delays with
`delay_quantization="ceil"` by default, so events are not delivered earlier
than the requested delay. Use `"floor"` for explicit earlier/compatibility
quantization, or `"strict"` to require `delay / dt` to be an integer.

## Source Layout

The network package is split by responsibility:

```text
core.py                  Population, Connection, NetworkRunResult
edges.py                 cell-cell adjacency builders
projections.py           edge-to-synapse contact expansion
lowering.py              Connection -> runtime-ready blocks
delivery.py              ring buffers and event delivery backends
engine.py                Network orchestration and run loop
```

User code should normally import from `braincell.network` rather than these
implementation modules.

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

1. The current ring-buffer slot is written into each target synapse's
   `pre_spike` buffer and cleared.
2. Every population prepares synapse inputs and advances dynamics.
3. Population spikes are sampled.
4. Each connection block projects `spike[pre_index] * weight` to
   `(post_index, local_synapse_index)`.
5. Events are added to the future ring-buffer slot selected by that contact's
   `delay_steps`.

Connection contacts are grouped by quantized `delay_steps` at runtime, so a
single connection table may contain heterogeneous per-edge or per-contact
delays. Multiple groups targeting the same `(post_population, synapse_layout)`
are summed before the synapse input is prepared.

`Network.run(..., event_backend="auto")` uses the `brainevent.coomv` sparse
matrix-vector backend when it is available, otherwise it falls back to the JAX
scatter backend. Pass `event_backend="scatter"` to force scatter, or
`event_backend="brainevent"` to require `brainevent.coomv`. When brainevent is
used, `brainevent_backend` selects the internal coomv kernel; the default is
`"jax_raw"` for a portable CPU/GPU path, while GPU installations may also
benchmark brainevent backends such as `"cuda_raw"` and `"cusparse"`.

`Network.run(..., spike_recording="full")` returns the historical full cell
spike trace for each population. For benchmarks or large multicompartment
populations, use `"population"` to record only the cell-level spike trace used
for event delivery, or `"none"` to omit spike traces from the result while still
delivering recurrent events internally.

For a shared target receiving edges `(0, j)`, `(1, j)`, `(2, j)`, the delivered
increment is:

```text
0_spike * 0_weight + 1_spike * 1_weight + 2_spike * 2_weight
```

For distinct local targets, the same scatter happens independently for each
`local_synapse_index`.
