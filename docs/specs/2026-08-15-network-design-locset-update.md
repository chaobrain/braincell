# Network Design and Locset Update

## Goal

Synchronize the Network Builder design documents with the decisions made during
the naming, population, projection, context, pair, and placement review. In the
same change, implement the locset behavior that is already part of BrainCell's
current public surface.

This change updates design specifications for the future Network Builder; it
does not implement that complete builder. Current runtime code is changed only
where needed for the locset and fork-point behavior below.

## Network design requirements

- Make naming and API vocabulary I-01 and renumber the former I-01 through I-08
  to I-02 through I-09. Keep I-10 through I-12 unchanged.
- Distinguish mutable handles, immutable specs, materialized results, and
  runtime blocks/states.
- Specify Network defaults, quaternion rotation, Population ownership and
  factory resolution, endpoint filtering, the array PairRule protocol,
  Projection inspection, NetworkContext views, and add-order dependencies.
- Remove planned public PairBatch, LocationBatch, ResolvedProjection, ctx.rows,
  and implicit locset sampling contracts.
- Keep unresolved weight, RNG, weighted sampling, delay-buffer, and mutation
  questions open instead of inventing implementation behavior.

## Locset requirements

- Preserve row order and duplicate locations by default; validation and
  coordinate canonicalization must not sort or deduplicate.
- Support deferred expression operations: concatenation with `+`, stable set
  union/intersection/difference with `|`, `&`, and `-`, and explicit stable
  deduplication with `unique()`.
- Give resolved LocsetMask a read-only columnar representation, an efficient
  column constructor, compatibility row access, optional display names, and an
  immediate `unique()` method.
- Preserve the exact requested cardinality and generation order of uniform and
  random samples.
- Add ForkPoints as the primary name and keep BranchPoints as a compatibility
  alias. Forks are topology junctions that connect at least three distinct
  branches after applying transitive parent/child attachment equivalence.
- Do not merge ordinary locset rows merely because they share XYZ, a CV, an
  electrical node, or a topology junction.

## Documentation and compatibility

- Update the five durable Network design documents together so public and
  internal contracts do not diverge.
- Update current locset concepts, API changelog, and the point-placement
  identity notebook only after the corresponding code behavior exists.
- Preserve existing `LocsetMask(points=..., display_names=...)`, `.points`, and
  `BranchPoints` imports. Do not add EmptyLocset or public multiset subtraction
  and intersection operations.

## Acceptance

- Locset unit tests cover ordering, duplicates, all public operators, explicit
  unique, column storage, sampling cardinality, and validation.
- Fork tests cover same-parent and transitive junctions, the non-fork single
  soma child case, disconnected coincident geometry, and stable output.
- Duplicate Cell.place locations lower to independent placement identities.
- Relevant notebooks execute without error, the documentation builds, and a
  repository search finds no unmarked stale Network vocabulary or locset set
  semantics in normative/current documentation.
