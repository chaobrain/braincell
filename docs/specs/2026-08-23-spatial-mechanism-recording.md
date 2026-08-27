# Spatial Mechanism Views, Recording, and Network Sources

## Status

This specification locks the public behavior for the spatial/mechanism view,
recording, and independently owned event-source work. It is the implementation
contract for the corresponding Network design documents.

## Selection algebra

Selection proceeds from population members to continuous morphology, then to
CVs, then to mechanisms:

```text
Cell population -> Region/Locset/branch -> CV -> mechanism
```

`cell[index]` selects population members. `cell.on(region)` includes every CV
with positive area coverage and retains its coverage fraction. `cell.loc(locset)`
retains ordered, duplicate continuous locations while exposing their deduplicated
owning CVs. `cell.branch[index_or_name]`, `cell.branch.by_type(type)`, branch-type
sugars, and `cell.cv[index]` compose with the same immutable scope. CV indexing is
local to the current scope; `cell.cv.by_id(ids)` intersects global CV identifiers.

Spatial views own no model data. They store a root `Cell` reference and static
indices. Population-only views retain heterogeneous `place()`; a view constrained
by Region, Locset, branch, or CV supports query, shape-preserving `set()`, and
`record()`, but cannot paint or place.

## Mechanism identity and views

`scope.channels`, `scope.ions`, and `scope.synapses` apply the scope before
mechanism filtering. Channels and ions select logical owners by explicit name or
registered type; ions additionally select by species. Synapses retain stable
logical IDs and duplicate-preserving numeric row selection. Numeric indexing is
not defined for Channel/Ion views because population, owner, and CV are distinct
axes.

Within one mechanism category, a logical density owner is `(registered_type,
instance_name)`. An omitted name defaults to the registered type. One name cannot
denote two types in the same category; the same string may be reused in different
categories.

Multiple paints of one density owner are legal only when their active CV sets are
disjoint. They form one logical owner and may carry different spatial parameter
values. Any active-CV overlap is an initialization error, even when the continuous
regions are geometrically disjoint. Different names remain independent and may
overlap. Cable-property precedence is unchanged.

Channel/Ion value access is aligned with stable `(owner, population, CV)` rows.
Shape-preserving parameter changes accept a scalar or one value per selected row.
Structural mutation and heterogeneous morphology remain outside v1.

## Recording

The primary API is spatial-scope first:

```python
scope.record(name, observable, *, period=None, frequency=None, start=0 * u.ms)
```

Public observable constructors live under `braincell.observe`:

```python
observe.state("v")
observe.channel(name="nav").state("m")
observe.channel(type="Na_HH1952").current()
observe.ion(species="na").current()
observe.synapse(name="ampa").state("g")
observe.synapse(ids=[3, 8]).current(reduce="none")
observe.membrane_current()
```

Selector arguments are explicit and mutually exclusive where applicable; naked
strings never guess between type and name. State observations expand logical
owners and never sum. Current observations default to `reduce="sum"`; `none`
retains contributor rows. Channel, ion, and membrane currents use CV current
density; synapse current uses total point-process current. Full membrane current
includes channels, synapses, clamps, and current inputs.

`period` and `frequency` are mutually exclusive and default to every solver step.
The resulting period and `start` must be integer multiples of `dt`. Recording uses
a persistent global schedule and half-open run segments `[start, stop)`. The first
default sample is the initialized state at `t=0`, before the first continuous
update. Parameters are inspected through views and are not recordable observables
in v1. Arbitrary Python observable callables are deferred.

Recording declarations are frozen at Cell/Network initialization. They compile
to gathers over existing model state and never create point placements, point IDs,
mechanism layouts, or layout-cache signatures. Legacy placed Probe declarations
remain a compatibility path with their historical post-step sampling convention;
new code must use `record()`. Removing the legacy Probe layout is migration work,
not part of the new recording representation.

## Results and continuation

`SampleBlock` contains immutable values and a static `RecordingSchema` describing
logical rows, units, owners, population IDs, morphology IDs, and contributor IDs.
Regular sample times are derived lazily from the segment and recording schedule.
`EventSeries` stores sparse `(time, source_id, count)` rows plus static metadata.

`NetworkResult` contains `start_time`, `stop_time`, `dt`, population-keyed samples,
and source-port-keyed events. All discrete output, including each Cell
Population's canonical root spike output, is read from `events`; there is no
separate dense spike result. `NetworkResult.concat(parts)` requires contiguous
segments, identical runtime `dt`, and identical schemas.

Duration must be a positive integer multiple of `dt`. Without reset, two
consecutive runs preserve Network time, cell/mechanism state, live-detector
history, delay queues, recording schedules, and RNG state. A fresh 10 ms run and
a fresh 5 ms + 5 ms run are numerically equivalent after concatenation. Reset
restores the initialization baseline without mutating previously returned results.

## Population and source ownership

`Network.add_population(name, model, **metadata)` eagerly accepts a `Cell`,
`NetStim`, `EventSequence`, or zero-argument provider returning one. `Population`
is the resolved handle and exposes `name`, `model`, `size`, `event_outputs`,
custom `metadata`, and `set()`. Scalars broadcast to population size; non-scalars
require that leading dimension. Reserved attributes and metadata share one namespace.

Canonical event-output ports are `Cell: spike`, `NetStim: spike`, and
`EventSequence: event`. A Population with exactly one port can be passed directly
to `connect`; multiple ports require `population.event_outputs[name]`. Additional
named Cell voltage-crossing outputs are registered automatically after a successful
`Network.connect()` and share the Cell execution owner. Unconnected outputs may be
published explicitly with `register_event_output()`. Independently owned sources
must be registered in the same Network as their consumers; standalone `Cell.run()`
may continue to use a raw scheduled source.

The Network owns global time. A model has one Network/Population owner. For
`NetStim(seed=None)`, initialization derives an order-independent seed from
`Network.seed` and the canonical population name. An explicit source seed fully
overrides Network seed. Schedule materialization occurs at initialization and is
not repeated by split runs.

## Deferred work

Scalable endpoint-pair generators, richer automatic placement, post-initialization
structural mutation, heterogeneous morphology, arbitrary derived observables, and
trainable topology are not part of this change.
