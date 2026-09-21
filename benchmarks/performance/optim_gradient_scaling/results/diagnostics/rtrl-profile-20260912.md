# RTRL profiling baseline (2026-09-12)

This is the fixed profiling suite for full HH, with C=1, 5, and 21 and
`Ntheta=3C`. Each isolated worker used GPU 0, x64, 40 ms / 1600 steps,
`B=S=16`, no GPU polling, one first execution, two extra warmups, and five
steady samples. StableHLO diagnostics were exported without extra model calls.

| C | method | trace s | lower s | XLA compile s | steady median s | IQR s | temporary MiB | logical carry MiB | HLO bytes | gather | scatter |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BPTT | 0.686 | 0.131 | 1.814 | 0.188825 | 0.000252 | 265.84 | — | 731604 | 44 | 33 |
| 1 | RTRL | 0.549 | 0.119 | 1.246 | 0.066623 | 0.000102 | 0.311 | 0.117 | 736972 | 52 | 34 |
| 5 | BPTT | 1.015 | 0.166 | 2.040 | 0.370697 | 0.000051 | 1328.74 | — | 2498834 | 95 | 64 |
| 5 | RTRL | 0.828 | 0.146 | 1.328 | 0.169691 | 0.000033 | 2.425 | 2.504 | 2491621 | 118 | 58 |
| 21 | BPTT | 1.806 | 0.234 | 3.297 | 0.962436 | 0.000287 | 6109.38 | — | 9489225 | 263 | 164 |
| 21 | RTRL | 1.294 | 0.241 | 1.999 | 0.723853 | 0.000060 | 37.92 | 42.74 | 9453639 | 322 | 154 |

RTRL is faster than BPTT at all three points: 2.83x, 2.18x, and 1.33x for
C=1, 5, and 21. Its compile time is also lower. The full RTRL HLO is slightly
smaller than BPTT at C=5 and C=21; its extra gather operations do not create a
large static graph increase. This baseline does not include explicit DHS JVP
or compact RTRL because the profiling suite fixes the speed path to full RTRL;
those candidates are recorded separately in the engineering comparison.

All six workers completed with finite outputs and saved StableHLO diagnostics.
Raw trials and source/environment provenance are in
`artifacts/rtrl_profile_20260912/`.

## Phase interpretation

The state-tree alignment helpers run while constructing the initial
reset/materialization carry. The rollout scan body carries tupled array values
and tangents directly; it does not call `_align_state_carry` or rebuild state
metadata at every time step. State-tree alignment is therefore not a useful
per-step optimization target for this workload. The next engineering targets
are tangent layout and the transition's tangent/local-gradient buffers.

The C=5 RTRL StableHLO scan has one `stablehlo.while` with 195 carry
components. The large dynamic components are the state tangent arrays; the
prefix-gradient accumulator is only a small `16x5xf64` value. Gradient
accumulation is therefore unlikely to be the dominant memory or copy cost.
The component count is mostly the functionalized state tree and static solver
metadata, so a blind carry-packing rewrite would risk changing solver
semantics without targeting the measured steady-time bottleneck.

## C=5 executable cost analysis

The diagnostic worker also recorded XLA cost analysis for the generic-DHS,
linearize-based C=5 RTRL path: about 5.47 million FLOPs and 26.64 MB of
bytes accessed per compiled call, giving an arithmetic intensity of roughly
0.21 FLOP/byte. This is strongly memory/traffic dominated rather than
compute dominated. The result points toward tangent/state layout and indexed
access fusion as the worthwhile GPU-level directions; adding more algebra to
the DHS JVP is unlikely to help.
