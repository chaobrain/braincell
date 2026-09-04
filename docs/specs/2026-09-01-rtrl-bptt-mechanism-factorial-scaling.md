# RTRL/BPTT Mechanism Factorial Scaling

## Goal

Separate the empirical contributions of active dynamical state size and
trainable parameter directions. The existing CV sweep follows
`N_x proportional to N_theta`; this experiment adds mechanism/trainable-subset
combinations that break that collinearity without changing the public API.

## Fixed Protocol

- CV counts: `3, 5, 9, 17, 33`.
- Duration/batch/seeds: `T40/B16/S16`.
- A100, x64, `dt=0.025 ms`, recursive backsub, ten synchronized steady calls.
- One deterministic full-HH target voltage is shared by all six candidates at
  each CV count. Target generation is outside gradient timing.

## Factorial Cases

| Case | Painted channels | Trainable channels | States/CV | Parameters/CV |
| --- | --- | --- | ---: | ---: |
| `l_fit_l` | Leak | Leak | 1 | 1 |
| `lk_fit_l` | Leak, K | Leak | 2 | 1 |
| `lk_fit_lk` | Leak, K | Leak, K | 2 | 2 |
| `lkn_fit_l` | Leak, K, Na | Leak | 4 | 1 |
| `lkn_fit_lk` | Leak, K, Na | Leak, K | 4 | 2 |
| `lkn_fit_lkn` | Leak, K, Na | Leak, K, Na | 4 | 3 |

Every selected trainable channel uses `group_by="cv"`. Painted but untrained
channels retain their fixed baseline conductance. Sodium and potassium ions
remain present in every candidate so fixed infrastructure is comparable.

## Measurements And Analysis

Each method/case runs in an isolated worker. Results add mechanism metadata,
`n_x`, `n_theta`, full traced state count, logical RTRL carry, compiler memory,
NVML process peak, timing distribution, throughput, and paired numerical
agreement.

Adding K/Na changes both state count and the transition operators; changing the
trainable subset also changes which reverse/JVP branches are active. Therefore
these rows are a mechanism/trainable-subset ablation, not an independently
controlled `N_x` or `N_theta` complexity experiment. The notebook reports
within-C normalized runtime and temporary-memory responses without theoretical
exponent fits. RTRL carry is still checked against its exact logical byte
formula.

## Acceptance

- Exactly 30 static configurations and 60 method trials.
- BPTT/RTRL losses and gradients satisfy the existing x64 tolerances for every
  factorial case.
- Existing pilot/full/large-CV/backsub suites and ids are unchanged.
- Generated factorial CSV/NPZ/JSON artifacts remain ignored by Git.
- The publication notebook executes without errors and exports Supplementary
  Figure S3 as PDF and 300 dpi PNG.
