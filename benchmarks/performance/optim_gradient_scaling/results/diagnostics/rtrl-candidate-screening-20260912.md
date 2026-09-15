# RTRL candidate screening protocol

This file records candidate experiments that are evaluated without changing
the accepted speed path. A candidate is adopted only when it preserves exact
RTRL correctness (`relative-L2 <= 1e-6`) and improves steady median by at least
3% at both C=5 and C=21. Candidates below that threshold remain diagnostics
only and their implementation changes are discarded.

## Fixed protocol

- full HH, x64, GPU 0, `B=S=16`, 40 ms, `dt=0.025 ms`;
- generic DHS AD and `linearize + vmap` unless the candidate changes one of
  those explicitly;
- one first execution, two extra warmups, five steady samples;
- no GPU polling during timing;
- record compile phases, StableHLO counts, XLA cost analysis, memory analysis,
  logical carry, correctness, source hash, and raw samples;
- C=5 is the fast screening point; C=21 is the scale confirmation point.

## Candidate queue

| ID | Candidate | Status | Adoption rule |
|---|---|---|---|
| LAYOUT | direction/batch tangent layout | excluded | state-major was 0.28% slower at C=5 and 0.14% slower at C=21 |
| INDEX | dense/static tangent indexing | excluded | dense tangent packing was 17.8% slower at C=5 due to per-step pack/unpack |
| FUSE | local gradient/tangent accumulation fusion | excluded | HLO keeps a small gradient accumulator in the while carry; no large standalone gradient buffer was found |
| ACTIVE | compact active transition | blocked | generic projection still requires a full tangent pytree; current compact path is slower |
| GPU | kernel/layout specialization | excluded | short Nsight trace shows dynamic-slice output history dominates visible kernels; tangent gather/scatter are minor |

No candidate implementation is part of the accepted path until its row is
marked adopted with links to an A/B result.

## INDEX initial diagnosis

The existing C=5/C=21 RTRL HLO contains respectively 118/322 gathers and
58/154 scatters, with no residual reshape or dynamic-update-slice operations
after lowering. The largest tangent-related shapes are dense four-dimensional
arrays such as `16x15x16x12xf64` at C=5 and `16x63x16x44xf64` at C=21.
Therefore compiler canonicalization has already removed simple reshape
overhead; a useful INDEX experiment must change the pre-lowered indexing/data
layout, not add another post-hoc transpose. No accepted code was changed by
this diagnosis.

## FUSE diagnosis

The C=5 and C=21 RTRL graphs each contain one scan `while`, with 195 and 463
carry components respectively. The gradient accumulator is carried as a small
direction-by-parameter array alongside the tangent state. The only
`dynamic_update_slice` is the output-history helper for the time axis, not a
large gradient workspace. No independent local-gradient buffer large enough
to target was identified, so a hand-written accumulation fusion would mainly
duplicate XLA's existing scan fusion. FUSE is excluded without changing
production code.

## LAYOUT result

The state-major experiment moved each tangent direction axis to the final
array axis during JVP propagation and restored the public shape afterward. It
passed correctness tests, but steady medians changed from 0.146248 s to
0.146659 s at C=5 and from 0.602850 s to 0.603708 s at C=21. This is a small
regression, so the experimental implementation was removed and the original
direction-major path remains unchanged.

## INDEX dense packing result

The temporary dense-packing path concatenated all tangent leaves into one vector
for the scan carry and unpacked them around every transition. It passed the
C=5 numerical check, but steady median increased from 0.146248 s to 0.172319 s
(17.8% slower). The candidate was stopped before C=21 and removed. The cost of
per-step pack/unpack is larger than the benefit of reducing carry leaf count.

## ACTIVE assessment

The existing HH active projection retains `V`, `Na.m`, `Na.h`, and `K.n`, but
the generic transition still consumes the complete state pytree. The current
compact implementation therefore performs full tangent embedding and active
extraction around every JVP. A fresh correctness/performance run completed
with zero compact-vs-full gradient error: full RTRL steady median was
0.183003 s and compact was 0.197322 s (7.8% slower) for the five-CV
correctness workload. A faster active path would require an HH-specific
transition or solver-level derivative interface, which is outside the current
semantic-preservation scope. ACTIVE remains a memory path and no production
code was changed.

## GPU profiling result

A short C=5 full-RTRL trace was collected with Nsight Systems using generic DHS
AD and `linearize + vmap`. Nsight report post-processing emitted a CUDA 13
compatibility error, but its SQLite event database was readable. The trace
contained 5,972 kernels with 7.98 ms total device kernel duration. The largest
entry was `wrapped_dynamic_slice` (4,805 launches, 5.35 ms total). Tangent
indexing kernels were small: `wrapped_gather` had 35 launches and 0.036 ms,
while `input_scatter_fusion` had 19 launches and 0.025 ms. The visible GPU
hotspot is therefore the time-history/dynamic-slice output path rather than
tangent gather/scatter. Changing RTRL tangent layout or writing a specialized
tangent kernel is not justified by this trace, so GPU is excluded without
production changes. The raw report was kept outside the repository; the
summary is the reproducible artifact for this decision.
