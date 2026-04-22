# braincell Triage Review — Spec

**Date:** 2026-04-22
**Mode:** Read-only triage. No code changes in this pass.
**Owner of next step:** user selects findings → separate implementation plan via `writing-plans`.

## Goal

Produce a single ranked findings report for the `braincell/` package that the user can triage in one sitting. Findings cover correctness first, architecture second. Every entry must point to a concrete location and propose the smallest high-confidence fix.

## Non-Goals

- No source edits, no refactors, no test additions in this pass.
- No exhaustive linting. Style-only nits are excluded unless they mask correctness.
- No speculative redesigns. Architectural findings must point to a concrete boundary or duplication problem in current code.

## Scope

### Full read (core runtime hot path)

- `braincell/_base.py`
- `braincell/_misc.py`
- `braincell/_single_compartment/` (all source files)
- `braincell/_multi_compartment/` (all source files)
- `braincell/_compute/` (all source files)
- `braincell/_cv/` (all source files)
- `braincell/quad/` (all source files)
- `braincell/mech/` (all source files)
- `braincell/__init__.py`

### Surface scan (skim for smells, hygiene, duplication)

- `braincell/channel/`
- `braincell/ion/`
- `braincell/synapse/`
- `braincell/io/`
- `braincell/filter/`
- `braincell/vis/`
- `braincell/morph/`

### Out of scope

- `legacy/`, `develop/`, `examples/`, `docs/`, `dev/`
- Test files (`*_test.py`) reviewed only when they are the clearest evidence of a bug or when a finding proposes a test.

## Inspection Lens (per module)

Applied in this order. Finding quality > quantity.

1. **Correctness** — wrong operators, off-by-one, dimensional/unit errors, state aliasing, JAX-trace hazards (side effects inside `jit`, Python-level branching on traced values, leaked `Tracer` references), dtype mismatches, integer/float confusion, mutation under `vmap`.
2. **Failure modes** — missing validation at system boundaries, silent fallbacks, overly broad `except`, cryptic or swallowed errors, partial-success states, resource leaks.
3. **Hidden assumptions** — implicit ordering (dict iteration order, sort stability), implicit dtype defaults, import-time side effects (registry registration, global state), mutable default args, coupled init order.
4. **Edge cases** — empty inputs, single-branch morphology, zero-length CV, degenerate shapes, unit = dimensionless, single-cell vs batched, diffrax cold import.
5. **Architecture** — boundary bleed, duplication, unclear public vs `_private`, import cycles, `__init__.py` export hygiene, CLAUDE.md ↔ code drift.

## Deliverable

Single file: `docs/superpowers/specs/2026-04-22-braincell-triage-review.md` (this spec is overwritten by the report in the final step, OR the report is written as a sibling file `...-review.md`; final name chosen when report is produced).

**Decision:** report will be written as sibling file `docs/superpowers/specs/2026-04-22-braincell-triage-review-report.md` so the spec remains referenceable.

### Report structure

1. **Executive summary** — ≤250 words. Top 5 risks. One-sentence each.
2. **Findings table of contents** — IDs grouped by severity.
3. **Findings** — one subsection per finding, in severity order:
   - **Critical** — data corruption, silent wrong answers, unit/dimensional bug, JAX-trace hazard causing wrong output.
   - **High** — likely bug under plausible input, weak error handling that will bite, import-time crash risk.
   - **Medium** — latent bug requiring unusual input, confusing but functional code, missing validation where upstream happens to be safe today.
   - **Low** — smell only, speculative, style that masks correctness.
   - **Arch** — module boundary, duplication, unclear responsibility, CLAUDE.md drift.
4. **Appendix A — CLAUDE.md drift table** — path in CLAUDE.md vs actual path, per known mismatch.
5. **Appendix B — coverage gap** — explicit list of modules not deeply read, with one-line reason.

### Finding schema

Each finding MUST have:

| Field | Rule |
|-------|------|
| `ID` | `SEV-NN` e.g. `CRIT-01`, `HIGH-03`, `ARCH-02`. Stable across revisions. |
| `Title` | ≤80 chars, imperative or noun phrase. |
| `Location` | `file:line` or `file:line-line`. At least one concrete anchor required. |
| `Issue` | 2–4 sentences. What is wrong. Evidence from code (quote ≤10 lines if needed). |
| `Risk` | 1–2 sentences. What breaks, under what input, with what blast radius. |
| `Proposed fix` | Smallest change that removes the risk. Ideally ≤20 LOC. If larger, flag and give sketch. |
| `Confidence` | `H` / `M` / `L`. H = I will defend this as a bug. M = bug under stated assumption. L = smell / speculative; caveat required. |
| `Behavior preserved?` | `yes` / `no` (justify if `no`). |

## Severity Rubric

Severity is about blast radius under realistic use, not about how scary the code looks.

- **Critical** — produces wrong scientific output silently, crashes on common input, or corrupts state. Fixing is urgent.
- **High** — will fail for a user within weeks of normal use, or weak error handling that will mask a real bug when it occurs.
- **Medium** — needs unusual input or specific interleaving. Not urgent but not dismissible.
- **Low** — smell, duplication at small scale, speculative concern. Author judgement required.
- **Arch** — structural concern. No behavior change proposed, only clarification.

## Confidence Rubric

- **H (High)** — I have read the code path end to end and can reproduce the issue mentally. I would bet on it.
- **M (Medium)** — bug under an assumption I have stated; assumption is plausible but not verified against tests.
- **L (Low)** — I noticed a smell; a reader should verify. Findings at this level must include the word "speculative" in the issue text.

## Process

1. Exhaustive read of full-read scope. One subpackage at a time. Record candidate findings as I go.
2. Surface scan of remaining subpackages. ≤3 findings per surface-scanned subpackage unless something egregious.
3. Cross-module pass: duplication, boundary bleed, `__init__.py` hygiene, import cycles.
4. CLAUDE.md drift pass: diff paths listed in `CLAUDE.md` against actual layout; record in Appendix A.
5. Rank and write the report.
6. Self-review the report: every finding has location, risk, fix, confidence; severities calibrated; no vibes-as-bugs.

## Success Criteria

- Every finding has `file:line`, `risk`, `fix`, `confidence`.
- No finding at `H` confidence is wrong.
- Executive summary surfaces the 5 highest-impact findings.
- CLAUDE.md drift appendix is complete.
- Coverage gap appendix names every subpackage not deeply read.
- Report is triageable by the user in ≤30 minutes.

## Risks to This Process

- **Risk: I mislabel a smell as a bug.** Mitigation: confidence rubric forces `L` for unverified smells with explicit caveat.
- **Risk: report grows too long.** Mitigation: severity caps — aim ≤10 findings per severity, merge duplicates.
- **Risk: CLAUDE.md is stale in ways that look like bugs.** Mitigation: code is authoritative; CLAUDE.md drift documented in appendix, not reported as bugs.
- **Risk: surface-scanned subpackages hide critical issues.** Mitigation: Appendix B names the gap explicitly so user can request deeper review.

## Next Step

User reads the report, picks findings worth fixing. Those findings become input to `writing-plans` for a targeted implementation plan and subsequent PRs.
