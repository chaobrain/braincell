# Optim Documentation Organization

## Approved Change

Reorganize all of docs/design/optim by current capability, future proposals,
experimental results, and theory/method references. Keep public API, experimental
implementation, unimplemented proposals, and unsupported targets distinct.
Status refers to the inspected working tree, not a release announcement.

Preserve historical measurements, provenance, and limitations. Archive the old
P0 implementation plan under its first tracked date, 2026-08-29. Keep existing
dated specs as historical records rather than rewriting their frozen contracts.
Move current numerical evidence into topic-specific results pages, including the
session-only bidirectional CPU timing measurements; do not invent raw artifacts.

Current contracts live in docs/design/optim. Check code and docs/examples for
consistency; link the existing root examples requested for this change without
bulk-migrating them. This change modifies only documentation and necessary links,
not Python behavior, notebook outputs, or experiment results. No experiments are
rerun and no commit or push is requested.

## Acceptance

- One overview exposes current support, experimental workflows, future work,
  result records, and references without contradictory implementation status.
- Channel, Ion, Synapse, Connection/detector, and Network support are documented
  separately, including initialization, valid zero gradients, natural errors,
  explicit exclusions, tests, and runnable example locations.
- Plasticity remains a proposal, without inventing an implemented public API.
- Moved result tables retain all values and their local conditions. Timing,
  compilation, memory estimates, process peaks, and historical snapshots remain
  distinguishable.
- Local links and anchors resolve; superseded current-document paths have no
  remaining live references. Historical specs may retain historical paths.
- Pre-existing unrelated workspace changes are preserved.

## Outcome

The current design tree now has one overview, parameter support, public API,
architecture, experimental workflow and roadmap entry, plus two proposals,
five result pages and six theory/method references. The old P0 plan was archived;
superseded current-document bodies were moved rather than duplicated.

Static inspection found and corrected stale public-export/owner descriptions,
Synapse dictionary claims, the old threshold-equality formula, the assumption
that voltage-only loss never traverses event feedback, and experimental README
wording that prematurely reserved a future public optim package.

All 163 numeric table rows from the five migrated historical result/protocol
sources were retained. Protected measured paragraphs were compared with the
pre-edit working-tree text, allowing only the removed section heading. Local
Markdown link and heading-anchor checks passed across the 19 current optim
documents, the archived plan, and three affected example navigation pages.
git diff whitespace checks passed. These are documentation checks, not new
simulation, training, performance, coverage, or external-link validation runs.

All nine docs/examples notebooks were inspected structurally for parameter
learning entrypoints; no matching tutorial was found. Existing root learning
notebooks and code were not modified. The maintained tutorial gap is recorded
in the current roadmap, not claimed to be resolved by navigation alone.
