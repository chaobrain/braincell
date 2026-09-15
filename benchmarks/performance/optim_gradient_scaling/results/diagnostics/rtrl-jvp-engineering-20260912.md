# RTRL JVP engineering comparison (2026-09-12)

This diagnostic compares two implementations of the same exact RTRL recurrence:
the existing `jax.linearize` followed by batched pushforwards, and a direct
`vmap(jax.jvp)` path. It also compares the explicit DHS elimination/backsub JVP
against generic AD. The primal solver, HH model, and state update order are
unchanged.

## Workload

- five CV HH morphology (`dendrite_segments=(2, 2)`), 15 parameter directions;
- 50 ms rollout at 0.025 ms, float64;
- GPU 0, one worker per process;
- full and compact RTRL plus BPTT;
- each run uses one compile/first execution and the existing five steady samples;
- `BRAINCELL_DHS_CUSTOM_JVP=0` isolates the generic solver AD rule for the
  `linearize` versus direct-JVP comparison.

## Results

The following two runs used the same source and GPU, but were separate processes;
small differences should not be treated as confidence intervals.

| DHS rule | RTRL JVP mode | BPTT steady s | full RTRL steady s | compact RTRL steady s | full compile s | compact compile s |
|---|---|---:|---:|---:|---:|---:|
| generic AD | `linearize` + `vmap` | 0.440555 | 0.208915 | 0.225985 | 1.695 | 1.680 |
| generic AD | direct `vmap(jvp)` | 0.420876 | 0.211293 | 0.222450 | 1.495 | 2.130 |

The direct JVP path is numerically correct (full-vs-BPTT maximum absolute
gradient error `1.364e-12`), but it does not improve full RTRL. The existing
`linearize` path remains the preferred default because it has lower or equal
compile cost and comparable steady time.

The explicit DHS JVP A/B at the same five-CV workload gave full RTRL steady
times of `0.176541 s` with generic solver AD and `0.182874 s` with explicit
solver JVP in an earlier paired run. The difference is small and favors the
generic rule; the explicit rule is retained for correctness and future
structured kernels, not as a claimed speedup.

## Decision

Do not switch the RTRL engine to direct `vmap(jvp)`. The next performance target
is the full-state transition around the JVP: remove unnecessary state-tree
projection and materialization work while preserving exact parameter directions.
Any later solver-kernel change must be compared against the `linearize` baseline
with the same full workload.

## StableHLO inspection

For the same C=5, 2000-step functional step, lowering produced 327,366 bytes
for full RTRL and 342,989 bytes for compact RTRL. Operation counts were:

| path | gather | scatter |
|---|---:|---:|
| full | 111 | 55 |
| compact | 115 | 62 |

The compact graph is only modestly larger, so its roughly 8% slower execution
is not explained by a large increase in static indexing operations. The likely
remaining issue is layout/fusion and the repeated embed/extract of direction
arrays inside the scan; this needs a direct active-transition implementation or
GPU profiling before another algorithmic change.

## Fixed-profile generic-DHS A/B

A second fixed `rtrl_profile` run used the same C=1/5/21, `Ntheta=3C`, `B=S=16`, 1600-step, x64, 1+2+5 protocol. The only changed setting was `BRAINCELL_DHS_CUSTOM_JVP`: explicit in the original profile and generic AD in `rtrl_profile_generic_20260912`.

| C | Method | Explicit DHS JVP (s) | Generic DHS AD (s) | Generic / explicit |
|---:|---|---:|---:|---:|
| 1 | BPTT | 0.188825 | 0.185618 | 0.983 |
| 1 | RTRL | 0.066623 | 0.067147 | 1.008 |
| 5 | BPTT | 0.370697 | 0.364895 | 0.984 |
| 5 | RTRL | 0.169691 | 0.146248 | 0.862 |
| 21 | BPTT | 0.962436 | 0.933516 | 0.970 |
| 21 | RTRL | 0.723853 | 0.602850 | 0.833 |

The generic path is materially faster for RTRL at C=5 and C=21, while BPTT changes are small. This rejects explicit DHS JVP as the default speed path for the current full RTRL transition. The explicit implementation remains available for numerical diagnostics through `--dhs-jvp-mode explicit`; the default profiling and future speed work should use generic DHS AD.

## Fixed-profile RTRL JVP A/B

With generic DHS AD fixed, `linearize + vmap` was compared with direct
`vmap(jvp)` using the same six-worker protocol.

| C | linearize + vmap (s) | direct vmap(jvp) (s) | direct / linearize |
|---:|---:|---:|---:|
| 1 | 0.067147 | 0.066228 | 0.986 |
| 5 | 0.146248 | 0.145832 | 0.997 |
| 21 | 0.602850 | 0.602675 | 1.000 |

The sub-percent differences are below the 3% acceptance threshold and do not
justify changing the default `linearize + vmap` path. Direct mode remains an
engineering control only.
