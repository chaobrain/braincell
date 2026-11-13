# Scheduled delivery on the production entry: six-model regression

This page holds the CELL-10 measurement of `scheduled_delivery="direct"` and `"padded"` against the default path (`scheduled_delivery=None`, called `default` below). Both run through the production entries, `Network.run` and `Network.prepare_run` + `Network.update`, on CPU and GPU. Raw worker JSON, logs and the generated tables stay on the runner's machine and are not in Git; every number this page uses is reproduced here.

## Summary

- **Correctness.** On every model, at both precisions and on both entries, `direct` and `padded` give the default path's V, g, scheduled input and spikes: bit-identical apart from summation order where 100 rows share one synapse. small and long are beyond `padded`'s padding limit (ratios 40 and 12 > 8), so `padded` is not applicable there.
- **The better method depends on the device.** On the CPU, both methods are 3.7–5.2× faster on shared; `direct` is 1.75–1.81× faster on long's `update` entry, and about 3× per step on long's `run` entry. Gains elsewhere are 1.1–1.3× or indistinguishable from the default. On the GPU, `direct` is slower than the default on every model (0.65–0.92×). `padded` is faster wherever it applies (1.05–1.50×), and stable on shared, active and mixed.
- **Cold cost is small.** Plan preparation takes at most 0.025 s and plans hold at most 2 MiB, except long's `update` plan: 23 MiB, built in about 0.2 s, because `prepare_run` plans all 10⁶ events. On the CPU most method first calls are no slower than the default's, which compiles the schedule in; on the GPU some are up to 0.6 s slower. Every recommended method pays back within 9 runs of 100 ms, except long `update` `direct`: 9 (supplementary pass) to 28 (protocol run).
- **The default stays `scheduled_delivery=None` on every device.** The tables only recommend; choosing a method is the user's, by measuring.

Recommendation per device, model and entry: the method with a stable gain (faster in every round, median ratio ≥ 1.10), its median paired ratio (default ÷ method) and break-even runs; otherwise `default`, with the methods' ratios. CPU large, long and mixed combine the protocol run and the supplementary pass (below); the GPU shared `update` and active `run` cells use the mini pass for round 0.

| Model | CPU, `run` | CPU, `update` | GPU, `run` | GPU, `update` |
| --- | --- | --- | --- | --- |
| small | default (direct 1.04) | default (direct 1.08) | default (direct 0.65) | default (direct 0.76) |
| large | default (padded 1.20–1.24, limited by noise; direct 1.03–1.09) | padded 1.28, 0 | default (padded 1.06, direct 0.89) | default (padded 1.05, direct 0.92) |
| long | default (direct 1.27–1.30, per step 2.7–3.1, limited by noise) | direct 1.75–1.81, 9–28 | default (direct 0.80) | default (direct 0.82) |
| shared | padded 3.90, 0 (direct 3.74, 0) | padded 5.17, 0 (direct 5.02, 0) | padded 1.50, 0 (direct 0.67) | padded 1.38, 1.9 (direct 0.72) |
| active | padded 1.15, 0 (direct 1.13, one slow round) | default (direct 1.03, padded 0.95) | padded 1.13, 3.5 (direct 0.73) | padded 1.11, 0 (direct 0.79) |
| mixed | default (direct 1.05–1.06, padded 1.06–1.07) | default (direct 0.99–1.06, padded 1.07–1.08) | padded 1.12, 8.9 (direct 0.77) | padded 1.11, 0 (direct 0.77) |

"Limited by noise": faster in the quiet rounds of both CPU runs, but one noisy round in each run fails the every-round test.

## Environment

| Item | Value |
| --- | --- |
| Measured trees | Library `77e0290` (`Network.run`/`prepare_run` with `scheduled_delivery`) in every run. Driver `7722084` (CPU protocol run), `5877e20` (GPU protocol run), `8f00f9c` (CPU supplementary pass), `1bc87cf` (GPU mini pass); the later driver commits change only the re-run triggers and the selection of groups. No round mixes trees. |
| CPU | Intel Core i7-14700KF, 28 logical CPUs. Workers pinned to CPUs 0 and 2, two performance cores; their hyperthread siblings are CPUs 1 and 3. XLA CPU single-threaded. |
| GPU | NVIDIA GeForce RTX 3090 (24 GiB), GPU 1 (PCI 02:00.0), driver 580.178.04, CUDA 13.0, `jax-cuda13-plugin` 0.12.0.1; `XLA_PYTHON_CLIENT_PREALLOCATE=false`, compilation cache off |
| Software | Linux 7.0.0-42, Python 3.13.18, jax/jaxlib 0.12.0.1, brainstate 0.6.1, brainunit 0.6.0, numpy 2.5.4; float32 (`brainstate` default) |
| Machine state | Shared. Benchmark jobs serialize on one lock; tests do not take it. From 04:42 (UTC+8) this project's other jobs ran their tests on CPUs 4–27, so the CPU protocol run (02:29–04:36) is entirely before that and the supplementary pass entirely after; another project's tests and CI runner used any CPU, and GPU 1, throughout. |

Load (1-minute average at worker start and end) and the busy share of the sibling CPUs over each accepted worker's run:

| Run | Load, median (max) | CPU 1 busy, median (max) | CPU 3 busy, median (max) | Peak RSS MB |
| --- | --- | --- | --- | --- |
| CPU protocol | 6.4 (23.8) | 0.02 (1.00) | 0.02 (0.87) | 813–1108 |
| GPU protocol | 14.6 (78.1) | 0.29 (1.00) | 0.32 (1.00) | 1088–1395 |
| CPU supplementary | 3.9 (61.2) | 0.01 (0.91) | 0.01 (0.90) | 829–1115 |
| GPU mini | 2.8 (3.9) | 0.01 (0.08) | 0.02 (0.16) | 1155–1237 |

## Models

All six models run 100 ms at dt = 0.025 ms (4000 steps) in float32. Every scheduled row has its own NetStim source, firing every 10 ms from a start phase drawn from U[0, 10) ms (`brainstate.random.RandomState(7)`, drawn at float32 on the CPU), seed 7. Weights are 0.001 uS and the delays zero. `event_backend="scatter"`.

| Model | Cells × scheduled rows per cell | Events per source | Arrivals in 100 ms | Cell and synapses |
| --- | --- | ---: | ---: | --- |
| small | 1 × 10 | 10 | 100 | passive, one CV (IL 0.1 mS/cm², −65 mV); one ExpSyn (τ 2 ms, e 0 mV) per row |
| large | 100 × 100 | 10 | 100 000 | as small |
| long | 10 × 10 | 10 000 | 1 000 (10⁶ generated) | as small |
| shared | 100 × 100 | 10 | 100 000 | as small, the 100 rows of a cell on one ExpSyn |
| active | 20 × 50 | 11 | 11 000 | HH cable: one 200 µm branch, radius 1 µm, `CVPerBranch(10)`, `V_th` 0 mV; Na_HH1952 120 and K_HH1952 36 mS/cm² (E_Na 50, E_K −77 mV), IL 0.3 mS/cm² at −54.3 mV; 50 ExpSyn at x = 0.05 … 0.95 |
| mixed | as active | 11 | 11 000 | active, plus a live ring: the spike of cell i reaches the first 5 ExpSyn of cell i + 1, delay 1 ms, 0.003 uS |

In 100 ms every active cell fires 8–10 times and every mixed cell 2–11 times, in float32 and float64 alike: the start phases are drawn in float32 for both, so the schedule is the same.

## Protocol

The protocol was fixed before any timing ran. Each device has 3 rounds × 6 models × 2 entries × 3 methods = 108 rows.

- **One process per row.** Every (device, model, entry, method, round) is a fresh worker process. Within a round, the three methods of a (model, entry) run back to back, in an order shuffled from `--seed 0`.
- **Isolation.** Each worker holds `flock` on one lock file shared with the other benchmark jobs on the machine. It is pinned with `taskset` to CPUs 0 and 2, two performance cores that do not share a physical core, and runs XLA's CPU backend single-threaded. It imports JAX only once it holds the lock. The coordinator pauses 3 s after each worker. GPU rows use GPU 1, which another benchmark job also used under the same lock, one job at a time.
- **Inside a worker.** For each step count, 4000 and then 2000, the worker makes a first call, which compiles (and, on the `run` entry, builds the window's plan). It then makes 2 untimed warm-up calls and 10 timed calls. `run` below is the median of the 10 timed calls at 4000 steps, i.e. one 100 ms run. Per step is the slope between the two step counts, and per run the intercept. Plan preparation (host build and upload) and plan bytes come from the Network's preparation statistics; peak RSS is that of the whole process.
- **Re-run rule.** A worker is re-run, at most twice, if its IQR/median exceeds 0.1, if the 1-minute load average at its start or end is 16 or more (saturated), or, on a GPU, if at its start the device was at least 5 % busy or another process held at least 1 GiB of it. The last attempt is the row; a row that still breaks the rule is kept and flagged. Every attempt's file is kept.
- **Not applicable.** A `padded` plan over its limits raises `PaddingLimitError`. The worker records the message, the padding ratio and the time of the attempt; there is no fallback to `direct`.
- **Recorded per worker.** The load at start and end, the busy share of every CPU over the worker's run (from `/proc/stat`), peak RSS, `nvidia-smi` utilization and compute processes before and after (GPU), and `braincell.__file__`.
- **Paired ratio.** default ÷ method of the 100 ms run time, same round and entry; the reported ratio is the median over the 3 rounds. A method is recommended for a (device, model, entry) only with a stable gain: faster in every round and a median ratio of at least 1.10. Otherwise the recommendation is `default`.
- **Break-even.** The number of 100 ms runs after the first that repay the method's extra cold cost: (plan preparation + first call − default first call) ÷ (default run − method run). On the `run` entry the first call already includes the plan preparation. 0 means the first run already pays back; never means the method is not faster.

| Run | Driver | Time (UTC+8) | Scope | Rows | Workers | Rows flagged |
| --- | --- | --- | --- | --- | ---: | ---: |
| CPU protocol | `7722084` | 02:29–04:36 | all | 108: 96 ok, 12 not applicable | 173 | 16 |
| GPU protocol | `5877e20` | 04:41–06:15 | all | 108: 93 ok, 12 not applicable, 3 failed | 214 | 47 |
| CPU supplementary | `8f00f9c` | 06:15–08:17 | large, long, mixed; adds the sibling rule, 4 retries | 54: 48 ok, 6 not applicable | 122 | 11 |
| GPU mini | `1bc87cf` | 08:17–08:20 | round 0 of shared `update` and active `run` | 6 ok | 7 | 0 |

A first GPU run at `7722084` was stopped after 9 workers and is not used. Its GPU re-run trigger counted any other process on the device, and every worker found another project's idle 258 MiB context there. The two supplementary passes were agreed with the coordinator after the protocol runs.

## Correctness

The checks ran untimed on the CPU at the branch head `f36eefa` (library `77e0290`, clean tree), with `check_regression` for each model in float64 and float32. Each method runs against the default path on both entries. The `update` entry compares every step's V, g, scheduled input and spikes. The `run` entry compares every step's V, the final g and the spike events (step and source). All 12 (model, precision) checks pass.

| Model | `update` entry: direct / padded | `run` entry: direct / padded | Spike events (`run` entry) |
| --- | --- | --- | --- |
| small | 0 / not applicable (ratio 39.9) | 0 / not applicable (ratio 40) | 0 |
| large | 0 / 0 | 0 / 0 | 0 |
| long | 0 / not applicable (ratio 12) | 0 / not applicable (ratio 12) | 0 |
| shared | 0 / 0 | V 7.1e-15 mV, g 1.4e-17 uS (float64); V 3.8e-6 mV, g 0 (float32) / 0 | 0 |
| active | 0 / 0 | 0 / 0 | 198, identical |
| mixed | 0 / 0 | 0 / 0 | 183, identical |

The entries are the largest absolute errors against the default path; 0 means bit-identical. The two nonzero ones, shared on the `run` entry with `direct`, are within the float64 tolerance (`rtol = atol = 1e-12`) and the float32 one (`rtol = 3e-5`, V `atol = 1e-4 mV`). The `padded` plans of small and long exceed the padding ratio 8 and raise `PaddingLimitError`, so they are not applicable, at both precisions and on both entries.

Every timed worker also saves an untimed check run (final V and g, spike steps and sources). The report compares each method with the default worker of the same round, at the float32 tolerance; the largest errors are in the CPU and GPU tables below.

## CPU results

### Protocol run (driver `7722084`)

Run 02:29–04:36 (UTC+8). 108/108 rows: 96 ok and 12 not applicable (small and long `padded`); 173 workers including re-runs, 16 rows still flagged after 2 retries. Rounds are numbered 0–2. Ratios are default ÷ method of the same round, so above 1 the method is faster. Every cross-method check of the timed workers agrees exactly (V and g errors 0, spikes identical).

| Model | Entry | Method | Run ms, median (per round) | Per step µs | Per run ms | Ratio, median (per round) | Per-step ratio | Stable gain | Max IQR/median | Load | Flagged rounds |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| small | run | default | 3.03 (3.27, 3.03, 2.93) | 0.41 | 1.34 | — | — | — | 0.036 | 5.3–9.6 | — |
| small | run | direct | 2.84 (2.80, 2.91, 2.84) | 0.39 | 1.34 | 1.04 (1.17, 1.04, 1.03) | 1.04 | no | 0.058 | 4.4–10.0 | — |
| small | run | padded | not applicable (padding ratio 40) | — | — | — | — | — | — | — | — |
| small | update | default | 1.84 (1.95, 1.83, 1.84) | 0.40 | 0.24 | — | — | — | 0.029 | 4.9–9.4 | — |
| small | update | direct | 1.77 (1.79, 1.62, 1.77) | 0.42 | 0.12 | 1.08 (1.08, 1.13, 1.04) | 0.96 | no | 0.051 | 5.2–8.8 | — |
| small | update | padded | not applicable (padding ratio 39.9) | — | — | — | — | — | — | — | — |
| large | run | default | 171.47 (171.47, 166.35, 172.85) | 42.02 | 4.01 | — | — | — | 0.217 | 5.2–18.0 | 1 |
| large | run | direct | 161.12 (207.98, 161.12, 158.69) | 39.67 | 2.44 | 1.03 (0.82, 1.03, 1.09) | 0.79 | no | 0.243 | 5.6–15.0 | 0 |
| large | run | padded | 138.78 (132.66, 138.78, 181.31) | 34.63 | 2.20 | 1.20 (1.29, 1.20, 0.95) | 1.15 | no | 0.119 | 7.2–12.5 | 2 |
| large | update | default | 158.86 (157.43, 158.86, 162.86) | 40.25 | -0.49 | — | — | — | 0.023 | 5.0–12.0 | — |
| large | update | direct | 171.37 (151.62, 171.37, 191.32) | 42.83 | 0.04 | 0.93 (1.04, 0.93, 0.85) | 0.94 | no | 0.112 | 6.3–13.5 | 2 |
| large | update | padded | 123.26 (123.21, 176.50, 123.26) | 30.91 | 2.16 | 1.28 (1.28, 0.90, 1.32) | 1.28 | no | 0.076 | 5.0–18.2 | 1 |
| long | run | default | 38.23 (34.16, 41.56, 38.23) | 7.03 | 6.04 | — | — | — | 0.078 | 4.8–15.6 | — |
| long | run | direct | 27.28 (26.30, 27.28, 46.37) | 2.29 | 17.16 | 1.30 (1.30, 1.52, 0.82) | 3.08 | no | 0.341 | 4.8–23.8 | 1, 2 |
| long | run | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| long | update | default | 18.08 (18.08, 29.25, 16.11) | 4.76 | -0.95 | — | — | — | 0.149 | 3.8–16.3 | 1 |
| long | update | direct | 12.05 (9.99, 12.05, 14.29) | 4.49 | -5.92 | 1.81 (1.81, 2.43, 1.13) | 1.89 | yes | 0.454 | 4.1–21.0 | 1, 2 |
| long | update | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| shared | run | default | 88.11 (86.68, 92.71, 88.11) | 21.31 | 3.33 | — | — | — | 0.117 | 3.4–14.0 | 1 |
| shared | run | direct | 24.79 (23.13, 24.79, 27.53) | 5.25 | 3.08 | 3.74 (3.75, 3.74, 3.20) | 3.98 | yes | 0.275 | 2.9–18.0 | 2 |
| shared | run | padded | 22.23 (22.23, 21.66, 29.61) | 4.66 | 3.57 | 3.90 (3.90, 4.28, 2.98) | 4.63 | yes | 0.092 | 2.8–12.3 | — |
| shared | update | default | 77.23 (76.79, 77.23, 104.55) | 19.31 | -0.45 | — | — | — | 0.409 | 2.9–16.1 | 2 |
| shared | update | direct | 15.39 (15.36, 15.39, 19.21) | 3.94 | -0.35 | 5.02 (5.00, 5.02, 5.44) | 5.04 | yes | 0.093 | 2.9–9.7 | — |
| shared | update | padded | 14.94 (14.62, 14.94, 21.04) | 3.66 | 0.01 | 5.17 (5.25, 5.17, 4.97) | 5.27 | yes | 0.056 | 2.8–9.2 | — |
| active | run | default | 96.42 (98.32, 95.79, 96.42) | 23.50 | 2.40 | — | — | — | 0.031 | 2.7–8.0 | — |
| active | run | direct | 84.77 (84.77, 84.58, 102.84) | 20.70 | 1.80 | 1.13 (1.16, 1.13, 0.94) | 1.16 | no | 0.285 | 2.9–17.9 | 2 |
| active | run | padded | 85.30 (85.35, 85.30, 83.73) | 20.89 | 1.73 | 1.15 (1.15, 1.12, 1.15) | 1.15 | yes | 0.068 | 2.9–14.2 | — |
| active | update | default | 76.80 (76.76, 76.92, 76.80) | 19.07 | 0.65 | — | — | — | 0.010 | 2.7–7.5 | — |
| active | update | direct | 74.79 (75.29, 74.74, 74.79) | 18.60 | 0.40 | 1.03 (1.02, 1.03, 1.03) | 1.03 | no | 0.010 | 2.7–5.2 | — |
| active | update | padded | 80.92 (83.25, 80.07, 80.92) | 20.23 | -0.01 | 0.95 (0.92, 0.96, 0.95) | 0.94 | no | 0.051 | 2.7–6.4 | — |
| mixed | run | default | 108.48 (109.48, 108.28, 108.48) | 26.67 | 1.78 | — | — | — | 0.030 | 3.5–4.4 | — |
| mixed | run | direct | 102.20 (108.62, 101.92, 102.20) | 25.10 | 1.81 | 1.06 (1.01, 1.06, 1.06) | 1.06 | no | 0.061 | 3.1–4.0 | — |
| mixed | run | padded | 102.25 (102.42, 100.76, 102.25) | 25.12 | 1.95 | 1.07 (1.07, 1.07, 1.06) | 1.08 | no | 0.045 | 3.4–6.6 | — |
| mixed | update | default | 108.09 (146.14, 105.12, 108.09) | 27.04 | 0.18 | — | — | — | 0.274 | 7.8–14.6 | 0 |
| mixed | update | direct | 140.62 (147.01, 140.62, 107.02) | 41.21 | -21.55 | 0.99 (0.99, 0.75, 1.01) | 0.80 | no | 0.156 | 7.9–11.9 | 0 |
| mixed | update | padded | 98.96 (98.96, 98.49, 99.66) | 24.63 | 0.27 | 1.08 (1.48, 1.07, 1.08) | 1.09 | no | 0.046 | 3.8–10.3 | — |

Storage and cold cost (medians over rounds). Plan preparation is the host build and upload of the plan; on the `run` entry it is part of the first call. The extra cold cost is plan preparation (update entry only) + first call − default first call.

| Model | Entry | Method | Plan preparation s | Plan KiB | First call s | Peak RSS MB | Extra cold s | Break-even runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | run | default | — | — | 0.49 | 926 | — | — |
| small | run | direct | 0.001 | 17.7 | 0.49 | 918 | 0.00 | 1.4 |
| small | update | default | — | — | 0.42 | 903 | — | — |
| small | update | direct | 0.001 | 17.6 | 0.38 | 912 | -0.04 | 0 |
| large | run | default | — | — | 0.89 | 984 | — | — |
| large | run | direct | 0.014 | 1078.7 | 0.88 | 952 | -0.02 | 0 |
| large | run | padded | 0.013 | 2039.1 | 0.72 | 947 | -0.18 | 0 |
| large | update | default | — | — | 0.74 | 947 | — | — |
| large | update | direct | 0.012 | 1078.7 | 0.74 | 925 | 0.01 | never |
| large | update | padded | 0.013 | 2039.6 | 0.59 | 923 | -0.13 | 0 |
| long | run | default | — | — | 0.87 | 1084 | — | — |
| long | run | direct | 0.001 | 32.0 | 0.57 | 982 | -0.30 | 0 |
| long | update | default | — | — | 0.57 | 983 | — | — |
| long | update | direct | 0.219 | 23817.4 | 0.51 | 1040 | 0.17 | 27.6 |
| shared | run | default | — | — | 0.77 | 975 | — | — |
| shared | run | direct | 0.012 | 1078.7 | 0.67 | 941 | -0.09 | 0 |
| shared | run | padded | 0.013 | 2039.1 | 0.61 | 935 | -0.15 | 0 |
| shared | update | default | — | — | 0.70 | 940 | — | — |
| shared | update | direct | 0.012 | 1078.7 | 0.47 | 913 | -0.23 | 0 |
| shared | update | padded | 0.012 | 2039.6 | 0.46 | 923 | -0.24 | 0 |
| active | run | default | — | — | 1.39 | 1069 | — | — |
| active | run | direct | 0.002 | 147.5 | 1.22 | 1041 | -0.17 | 0 |
| active | run | padded | 0.002 | 253.9 | 1.21 | 1053 | -0.18 | 0 |
| active | update | default | — | — | 1.10 | 1067 | — | — |
| active | update | direct | 0.002 | 149.1 | 1.10 | 1047 | 0.00 | 0.7 |
| active | update | padded | 0.002 | 279.0 | 0.95 | 1045 | -0.14 | never |
| mixed | run | default | — | — | 1.38 | 1079 | — | — |
| mixed | run | direct | 0.002 | 147.5 | 1.28 | 1078 | -0.10 | 0 |
| mixed | run | padded | 0.002 | 253.9 | 1.32 | 1063 | -0.06 | 0 |
| mixed | update | default | — | — | 1.33 | 1082 | — | — |
| mixed | update | direct | 0.002 | 149.1 | 1.18 | 1054 | -0.15 | never |
| mixed | update | padded | 0.002 | 279.0 | 1.01 | 1064 | -0.32 | 0 |

Selection (stable gain: faster in every round, median ratio ≥ 1.10):

| Model | `run` entry | `update` entry |
| --- | --- | --- |
| small | default | default |
| large | default | default |
| long | default | direct (1.81×, break-even 27.6) |
| shared | padded (3.90×, break-even 0) | padded (5.17×, break-even 0) |
| active | padded (1.15×, break-even 0) | default |
| mixed | default | default |

### Supplementary pass (driver `8f00f9c`)

The CPU protocol run left large, long and mixed noisy: several of their rows were flagged, and few attempts ran with both sibling CPUs idle. These three models were measured again 06:15–08:17 (UTC+8), with the same protocol plus one rule: a worker is also re-run if CPU 1 or CPU 3 was more than 5 % busy over its run (`--sibling-cpus 1,3 --max-sibling-busy 0.05`), up to 4 times. 54 rows (48 ok, 6 not applicable), 122 workers; 11 rows were still flagged after 4 retries. They are all in rounds 0 and 1, which ran while load reached 61. Round 2 ran on a quiet machine. Cross-method checks agree exactly.

| Model | Entry | Method | Run ms, median (per round) | Per step µs | Per run ms | Ratio, median (per round) | Per-step ratio | Stable gain | Max IQR/median | Load | Flagged rounds |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| large | run | default | 169.65 (169.65, 376.86, 164.16) | 39.21 | 17.94 | — | — | — | 0.241 | 2.3–61.2 | 1 |
| large | run | direct | 174.55 (174.55, 223.47, 151.28) | 42.45 | 5.13 | 1.09 (0.97, 1.69, 1.09) | 1.07 | no | 0.138 | 1.5–47.7 | 0, 1 |
| large | run | padded | 204.75 (204.75, 223.69, 131.88) | 41.23 | 2.49 | 1.24 (0.83, 1.68, 1.24) | 1.21 | no | 0.222 | 1.8–20.4 | 0, 1 |
| large | update | default | 161.99 (161.99, 172.96, 158.05) | 40.97 | -1.56 | — | — | — | 0.075 | 1.6–11.4 | — |
| large | update | direct | 177.30 (177.30, 257.89, 148.87) | 40.44 | 0.10 | 0.91 (0.91, 0.67, 1.06) | 1.01 | no | 0.452 | 1.5–32.7 | 0, 1 |
| large | update | padded | 123.88 (127.60, 123.88, 123.10) | 30.87 | 0.21 | 1.28 (1.27, 1.40, 1.28) | 1.28 | yes | 0.066 | 1.5–11.9 | — |
| long | run | default | 32.60 (31.21, 37.30, 32.60) | 6.22 | 7.70 | — | — | — | 0.062 | 1.7–11.7 | — |
| long | run | direct | 27.76 (43.19, 27.76, 25.59) | 2.34 | 21.58 | 1.27 (0.72, 1.34, 1.27) | 2.66 | no | 0.081 | 1.5–16.7 | 0 |
| long | run | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| long | update | default | 16.79 (17.45, 16.33, 16.79) | 4.39 | -0.77 | — | — | — | 0.046 | 2.5–9.1 | — |
| long | update | direct | 9.94 (9.97, 9.32, 9.94) | 2.49 | -0.01 | 1.75 (1.75, 1.75, 1.69) | 1.64 | yes | 0.265 | 1.9–10.7 | 0 |
| long | update | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| mixed | run | default | 108.67 (146.50, 107.29, 108.67) | 26.41 | 4.54 | — | — | — | 0.279 | 2.0–15.0 | 0 |
| mixed | run | direct | 103.47 (102.38, 103.47, 103.92) | 25.80 | 0.34 | 1.05 (1.43, 1.04, 1.05) | 1.04 | no | 0.020 | 2.2–5.5 | — |
| mixed | run | padded | 101.15 (100.34, 101.15, 102.56) | 24.92 | 2.50 | 1.06 (1.46, 1.06, 1.06) | 1.07 | no | 0.084 | 1.8–2.5 | — |
| mixed | update | default | 108.38 (108.38, 104.94, 108.99) | 26.74 | 1.40 | — | — | — | 0.091 | 2.2–7.0 | — |
| mixed | update | direct | 99.80 (102.62, 99.78, 99.80) | 24.69 | 1.03 | 1.06 (1.06, 1.05, 1.09) | 1.04 | no | 0.031 | 2.1–7.7 | 0 |
| mixed | update | padded | 98.11 (97.85, 98.11, 103.32) | 24.38 | 0.45 | 1.07 (1.11, 1.07, 1.05) | 1.04 | no | 0.098 | 2.5–9.8 | — |

| Model | Entry | Method | Plan preparation s | Plan KiB | First call s | Peak RSS MB | Extra cold s | Break-even runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large | run | default | — | — | 0.86 | 968 | — | — |
| large | run | direct | 0.013 | 1078.7 | 1.15 | 952 | 0.29 | never |
| large | run | padded | 0.025 | 2039.1 | 1.25 | 954 | 0.39 | never |
| large | update | default | — | — | 0.73 | 948 | — | — |
| large | update | direct | 0.012 | 1078.7 | 0.69 | 929 | -0.02 | never |
| large | update | padded | 0.013 | 2039.6 | 0.63 | 929 | -0.08 | 0 |
| long | run | default | — | — | 0.76 | 1109 | — | — |
| long | run | direct | 0.001 | 32.0 | 0.64 | 994 | -0.12 | 0 |
| long | update | default | — | — | 0.56 | 988 | — | — |
| long | update | direct | 0.178 | 23817.4 | 0.45 | 1042 | 0.06 | 9.0 |
| mixed | run | default | — | — | 1.35 | 1078 | — | — |
| mixed | run | direct | 0.002 | 147.5 | 1.26 | 1076 | -0.09 | 0 |
| mixed | run | padded | 0.002 | 253.9 | 1.27 | 1076 | -0.08 | 0 |
| mixed | update | default | — | — | 1.09 | 1086 | — | — |
| mixed | update | direct | 0.002 | 149.1 | 1.07 | 1066 | -0.02 | 0 |
| mixed | update | padded | 0.002 | 279.0 | 1.01 | 1052 | -0.08 | 0 |

With a stable gain here: large `update` padded 1.28× (break-even 0) and long `update` direct 1.75× (break-even 9.0). The rest stay `default`.

- **Limited by noise.** large `run` padded (1.24× median) and long `run` direct (1.27×, per step 2.7×) are faster in the quiet rounds of both CPU runs, but a noisy round is below 1 in each run, so neither qualifies as a stable gain.
- **Attempts with both siblings idle.** Counting only this pass's attempts with both siblings at most 5 % busy, IQR/median ≤ 0.1 and load < 16 (every attempt, medians, not paired by round) gives the same direction. large `run`: padded 1.27× and direct 1.10×, one attempt each. long `run`: direct 1.22×, 3.2× per step. large `update`: padded 1.31×. long `update`: direct 1.74×. mixed: 1.02–1.10× on both entries.
- **mixed.** In both CPU runs, mixed is at 0.99–1.08× of the default on both entries, below the 1.10 threshold.

## GPU results

### Protocol run (driver `5877e20`)

Run 04:41–06:15 (UTC+8) on GPU 1. 108 rows: 93 ok, 12 not applicable (small and long `padded`) and 3 failed. The 3 failed rows are r0 shared `update` default and padded, and r0 active `run` padded; each worker failed at backend initialization with `CUDA_ERROR_OUT_OF_MEMORY`, because another project's processes had filled the GPU's memory. 214 workers including re-runs; 47 rows still flagged after 2 retries, mostly for load and IQR. At their start, 69 of the 108 accepted rows saw another project's idle CUDA context (258 MiB) on the GPU, and 6 rows saw a process holding 4.4–18.4 GiB. The GPU was shared with another project throughout. The cross-method checks agree within float32 rounding (largest V difference 4.1e-5 mV, g 0, spikes identical).

On the GPU, a run is dominated by per-step launch and host overhead. Even the one-CV `small` model costs 28 µs per step, 70 times its CPU cost, and every model here is slower on this GPU than in the CPU runs above (two pinned cores, single-threaded XLA).

| Model | Entry | Method | Run ms, median (per round) | Per step µs | Per run ms | Ratio, median (per round) | Per-step ratio | Stable gain | Max IQR/median | Load | Flagged rounds |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| small | run | default | 115.08 (115.08, 116.16, 114.74) | 27.81 | 3.49 | — | — | — | 0.044 | 13.3–22.1 | 2 |
| small | run | direct | 177.79 (161.39, 177.79, 187.02) | 48.88 | -13.29 | 0.65 (0.71, 0.65, 0.61) | 0.58 | no | 0.163 | 13.3–21.0 | 2 |
| small | run | padded | not applicable (padding ratio 40) | — | — | — | — | — | — | — | — |
| small | update | default | 167.14 (166.74, 167.14, 168.23) | 41.59 | 0.55 | — | — | — | 0.012 | 11.0–34.5 | 1, 2 |
| small | update | direct | 220.02 (217.75, 220.02, 226.32) | 57.16 | -2.33 | 0.76 (0.77, 0.76, 0.74) | 0.73 | no | 0.063 | 13.2–23.1 | 1, 2 |
| small | update | padded | not applicable (padding ratio 39.9) | — | — | — | — | — | — | — | — |
| large | run | default | 619.68 (616.31, 626.97, 619.68) | 160.02 | -22.37 | — | — | — | 0.090 | 9.2–48.1 | 0, 1, 2 |
| large | run | direct | 699.77 (689.00, 699.77, 727.75) | 179.86 | -19.65 | 0.89 (0.89, 0.90, 0.85) | 0.90 | no | 0.093 | 9.8–41.8 | 1 |
| large | run | padded | 586.44 (520.14, 606.61, 586.44) | 151.03 | -8.81 | 1.06 (1.18, 1.03, 1.06) | 1.06 | no | 0.129 | 12.3–42.4 | 0, 1 |
| large | update | default | 651.80 (651.08, 651.80, 658.44) | 169.38 | -24.82 | — | — | — | 0.028 | 9.5–42.3 | 1 |
| large | update | direct | 707.25 (707.25, 703.25, 728.97) | 169.55 | 25.03 | 0.92 (0.92, 0.93, 0.90) | 1.00 | no | 0.137 | 6.2–31.8 | 0, 1 |
| large | update | padded | 624.36 (618.69, 624.36, 627.53) | 164.13 | -32.18 | 1.05 (1.05, 1.04, 1.05) | 1.03 | no | 0.090 | 6.5–60.7 | 1 |
| long | run | default | 151.73 (151.73, 160.97, 146.09) | 32.59 | 26.40 | — | — | — | 0.075 | 6.2–54.8 | 1 |
| long | run | direct | 187.36 (187.36, 250.08, 183.59) | 39.57 | 25.31 | 0.80 (0.81, 0.64, 0.80) | 0.81 | no | 0.120 | 6.1–54.7 | 1 |
| long | run | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| long | update | default | 173.54 (171.34, 176.28, 173.54) | 43.10 | 0.60 | — | — | — | 0.017 | 5.0–67.6 | 1 |
| long | update | direct | 210.01 (210.01, 445.77, 208.40) | 52.26 | 0.98 | 0.82 (0.82, 0.40, 0.83) | 0.82 | no | 0.563 | 6.0–62.8 | 1 |
| long | update | padded | not applicable (padding ratio 12) | — | — | — | — | — | — | — | — |
| shared | run | default | 145.63 (145.04, 169.06, 145.63) | 34.05 | 8.85 | — | — | — | 0.135 | 8.8–78.1 | 1 |
| shared | run | direct | 215.68 (215.20, 215.68, 222.80) | 44.11 | 38.77 | 0.67 (0.67, 0.78, 0.65) | 0.72 | no | 0.091 | 14.4–60.6 | 0, 1 |
| shared | run | padded | 100.82 (100.82, 104.86, 97.31) | 21.29 | 12.68 | 1.50 (1.44, 1.61, 1.50) | 1.55 | yes | 0.122 | 4.3–39.3 | 0, 1 |
| shared | update | default | 187.80 (188.53, 187.08) | 47.01 | -0.25 | — | — | — | 0.009 | 9.0–38.6 | 1 |
| shared | update | direct | 260.25 (263.78, 260.25, 260.20) | 65.35 | -1.17 | 0.72 (0.72, 0.72) | 0.72 | no | 0.142 | 12.1–31.6 | 0, 1 |
| shared | update | padded | 136.22 (136.86, 135.57) | 33.70 | 1.42 | 1.38 (1.38, 1.38) | 1.40 | no | 0.020 | 11.9–33.5 | 1 |
| active | run | default | 248.40 (248.40, 252.99, 247.28) | 58.76 | 17.93 | — | — | — | 0.028 | 11.5–47.0 | 0, 1 |
| active | run | direct | 344.57 (344.57, 366.42, 333.61) | 83.04 | 12.42 | 0.72 (0.72, 0.69, 0.74) | 0.67 | no | 0.104 | 12.5–37.0 | 0, 1 |
| active | run | padded | 221.34 (224.02, 218.65) | 52.58 | 11.03 | 1.13 (1.13, 1.13) | 1.14 | no | 0.060 | 12.5–35.2 | 1 |
| active | update | default | 312.20 (312.51, 312.20, 308.55) | 77.71 | 1.18 | — | — | — | 0.025 | 12.5–41.3 | 0, 1 |
| active | update | direct | 394.57 (394.57, 475.67, 391.01) | 99.50 | -3.41 | 0.79 (0.79, 0.66, 0.79) | 0.78 | no | 0.245 | 14.9–43.7 | 0, 1, 2 |
| active | update | padded | 276.78 (275.86, 280.27, 276.78) | 68.69 | 2.03 | 1.11 (1.13, 1.11, 1.11) | 1.12 | yes | 0.018 | 15.5–48.5 | 0, 1 |
| mixed | run | default | 264.73 (264.53, 280.16, 264.73) | 64.42 | 8.26 | — | — | — | 0.023 | 9.9–66.1 | 1 |
| mixed | run | direct | 348.33 (344.22, 365.88, 348.33) | 86.11 | 6.66 | 0.77 (0.77, 0.77, 0.76) | 0.74 | no | 0.032 | 14.6–42.8 | 1 |
| mixed | run | padded | 236.10 (236.10, 249.05, 234.55) | 58.03 | 4.08 | 1.12 (1.12, 1.12, 1.13) | 1.11 | yes | 0.045 | 15.4–59.1 | 1 |
| mixed | update | default | 327.53 (326.06, 330.17, 327.53) | 81.51 | 2.07 | — | — | — | 0.009 | 10.8–28.3 | 1 |
| mixed | update | direct | 423.35 (423.35, 422.18, 424.50) | 106.24 | -2.78 | 0.77 (0.77, 0.78, 0.77) | 0.77 | no | 0.043 | 11.5–33.7 | 1 |
| mixed | update | padded | 295.40 (293.41, 297.02, 295.40) | 73.64 | 1.63 | 1.11 (1.11, 1.11, 1.11) | 1.11 | yes | 0.008 | 11.9–34.0 | 1 |

Storage and cold cost:

| Model | Entry | Method | Plan preparation s | Plan KiB | First call s | Peak RSS MB | Extra cold s | Break-even runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | run | default | — | — | 1.07 | 1160 | — | — |
| small | run | direct | 0.002 | 17.7 | 1.21 | 1165 | 0.14 | never |
| small | update | default | — | — | 0.99 | 1160 | — | — |
| small | update | direct | 0.002 | 17.6 | 0.87 | 1165 | -0.12 | never |
| large | run | default | — | — | 1.57 | 1190 | — | — |
| large | run | direct | 0.014 | 1078.7 | 1.67 | 1185 | 0.10 | never |
| large | run | padded | 0.016 | 2039.1 | 2.04 | 1184 | 0.47 | 14.1 |
| large | update | default | — | — | 1.36 | 1170 | — | — |
| large | update | direct | 0.013 | 1078.7 | 1.51 | 1167 | 0.17 | never |
| large | update | padded | 0.015 | 2039.6 | 1.26 | 1163 | -0.08 | 0 |
| long | run | default | — | — | 1.16 | 1328 | — | — |
| long | run | direct | 0.003 | 32.0 | 1.04 | 1236 | -0.12 | never |
| long | update | default | — | — | 0.83 | 1239 | — | — |
| long | update | direct | 0.201 | 23817.4 | 0.86 | 1392 | 0.23 | never |
| shared | run | default | — | — | 1.21 | 1193 | — | — |
| shared | run | direct | 0.023 | 1078.7 | 1.77 | 1178 | 0.57 | never |
| shared | run | padded | 0.020 | 2039.1 | 1.16 | 1181 | -0.05 | 0 |
| shared | update | default | — | — | 1.43 | 1162 | — | — |
| shared | update | direct | 0.019 | 1078.7 | 1.26 | 1160 | -0.15 | never |
| shared | update | padded | 0.019 | 2039.6 | 1.25 | 1155 | -0.16 | 0 |
| active | run | default | — | — | 2.85 | 1230 | — | — |
| active | run | direct | 0.004 | 147.5 | 2.36 | 1228 | -0.49 | never |
| active | run | padded | 0.004 | 253.9 | 2.69 | 1219 | -0.16 | 0 |
| active | update | default | — | — | 2.60 | 1242 | — | — |
| active | update | direct | 0.003 | 149.1 | 2.24 | 1228 | -0.36 | never |
| active | update | padded | 0.005 | 279.0 | 2.37 | 1231 | -0.23 | 0 |
| mixed | run | default | — | — | 2.10 | 1230 | — | — |
| mixed | run | direct | 0.004 | 147.5 | 2.46 | 1235 | 0.36 | never |
| mixed | run | padded | 0.003 | 253.9 | 2.36 | 1229 | 0.26 | 8.9 |
| mixed | update | default | — | — | 1.90 | 1257 | — | — |
| mixed | update | direct | 0.003 | 149.1 | 2.26 | 1242 | 0.37 | never |
| mixed | update | padded | 0.004 | 279.0 | 1.79 | 1235 | -0.11 | 0 |

Selection:

| Model | `run` entry | `update` entry |
| --- | --- | --- |
| small | default | default |
| large | default | default |
| long | default | default |
| shared | padded (1.50×, break-even 0) | default |
| active | default | padded (1.11×, break-even 0) |
| mixed | padded (1.12×, break-even 8.9) | padded (1.11×, break-even 0) |

shared `update` padded (1.38×) and active `run` padded (1.13×) are faster in both of their ok rounds. They are not selected here because a stable gain needs three rounds, and round 0 failed; the mini pass completes them.

### Mini pass (driver `1bc87cf`)

Round 0 of the two groups with failed workers was measured again 08:17–08:20 (UTC+8). Each GPU attempt first waited, out of a 2 h budget, until no other process held 1 GiB or more of the GPU and it was under 5 % busy (`--groups 0:shared:update,0:active:run --gpu-wait 7200`). It found the GPU idle, at load 2.4–3.9. 6 rows, all ok and none flagged; cross-method checks agree exactly.

| Model | Entry | Method | Run ms, median (per round) | Per step µs | Per run ms | Ratio, median (per round) | Per-step ratio | Stable gain | Max IQR/median | Load | Flagged rounds |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| shared | update | default | 185.90 (185.90) | 46.01 | 1.87 | — | — | — | 0.006 | 2.6–2.6 | — |
| shared | update | direct | 258.21 (258.21) | 64.38 | 0.67 | 0.72 (0.72) | 0.71 | no | 0.007 | 2.9–2.9 | — |
| shared | update | padded | 134.43 (134.43) | 33.35 | 1.04 | 1.38 (1.38) | 1.38 | no | 0.012 | 2.8–2.8 | — |
| active | run | default | 246.79 (246.79) | 60.95 | 2.98 | — | — | — | 0.004 | 3.9–3.9 | — |
| active | run | direct | 339.91 (339.91) | 83.33 | 6.59 | 0.73 (0.73) | 0.73 | no | 0.017 | 3.4–3.4 | — |
| active | run | padded | 218.69 (218.69) | 51.72 | 11.82 | 1.13 (1.13) | 1.18 | no | 0.041 | 2.6–2.6 | — |

The quiet-machine times are within 3 % of the protocol run's rounds 1 and 2 for the same rows (for example shared `update` default 185.9 against 187.1–188.5 ms). The exception is active `run` direct, 8 % in one round. So the GPU rows depend little on the host load that flagged most of them.

With this round 0 and rounds 1–2 of the protocol run, each round pairs methods measured in one process tree:

| Model | Entry | Method | Ratio, median (per round) | Stable gain | Break-even runs |
| --- | --- | --- | --- | --- | ---: |
| shared | `update` | direct | 0.72 (0.72, 0.72, 0.72) | no | never |
| shared | `update` | padded | 1.38 (1.38, 1.38, 1.38) | yes | 1.9 |
| active | `run` | direct | 0.73 (0.73, 0.69, 0.74) | no | never |
| active | `run` | padded | 1.13 (1.13, 1.13, 1.13) | yes | 3.5 |

## Not applicable, flagged and failed rows

**Not applicable** (`PaddingLimitError`, the same on both devices and in every round; the worker records the attempt and does not fall back):

| Model | Entry | Steps planned | Padding ratio | Bytes needed |
| --- | --- | --- | ---: | ---: |
| small | `run` | [0, 4000) | 40 | 32 044 |
| small | `update` | [0, 3994) | 39.9 | 31 996 |
| long | `run` | [0, 4000) | 12 | 128 404 |
| long | `update` | [0, 3999999) | 12 | 128 000 372 |

Each message ends `(limits 268435456 bytes, ratio 8); use scheduled_delivery="direct", which has no such limit.` The `update` entry plans every event (`prepare_run`), so long's plan spans its whole 100 s schedule. In this run's single window from step 0, each plan is the first the Network builds. CELL-53 revisits the ratio check for later, sparser windows; it does not change a Network's first plan, as here.

**Failed** (GPU protocol run only): `gpu_shared_update_padded_r0`, `gpu_shared_update_default_r0` and `gpu_active_run_padded_r0`, `CUDA_ERROR_OUT_OF_MEMORY` at backend initialization while another project's processes held the GPU's memory. A failed worker is not re-run by the protocol; the GPU mini pass above re-measured both groups.

**Flagged** rows are listed per run in the tables above (column "Flagged rounds"). Their numbers are kept and enter the medians; a flagged row's reason is one or more of: IQR/median > 0.1, load ≥ 16, a busy GPU, a busy sibling CPU (supplementary pass only).

## Limitations

- **One window.** The `run` entry measures a single window [0, n) from step 0, repeated after `reset_state()` (a plan cache hit). The cost of building a new plan for each window of a segmented run is in the first-call column only. The padding limit over several consecutive, sparser windows is tracked separately as CELL-53.
- **`run` overhead on long schedules.** On long, each `Network.run` call spends about 18–20 ms on the host building the result's event table (`NetStim._event_table` over the source's 10⁶ generated events), whatever the method; a cProfile of the 2000-step calls under the lock shows it for default and `direct` alike. This per-call cost is the per-run intercept of long's `run` rows. It is why `direct`'s ratio on that entry (1.30×) is far below its per-step ratio (3.1×). It is a cost of `Network.run`'s result, outside the delivery methods.
- **A shared machine.** Benchmark jobs serialize on one lock, but other projects' tests ran outside it during every run here, on any CPU. A busy hyperthread sibling of a pinned CPU slows a worker by 1.2–2× without always raising its IQR. Idle siblings help but are not enough: in the CPU protocol run, attempts of one row with CPUs 1 and 3 both at most 5 % busy usually differ by under 5 %, but by up to 25 %. The supplementary pass re-ran such workers. The GPU was shared with another project's processes (idle contexts on most rows, 4.4–18.4 GiB on six, an out-of-memory failure on three), and GPU rows also depend on the host: their per-step cost is launch and host overhead. Paired ratios carry the comparison; absolute times are this machine's under load.
- **float32 only.** Timing runs in float32, the `brainstate` default; float64 is checked for correctness only.
- **Memory.** Peak RSS is that of the whole worker process, JAX runtime included; plan bytes are the plan arrays alone.
- **Two step counts.** Per-step and per-run costs come from two points (4000 and 2000 steps). A worker whose load changed between the two can give a negative intercept, as in some rows above; the medians of 4000-step runs and the paired ratios do not depend on it.

## Reproducing

```text
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase regression \
  --devices cpu --rounds 3 --warmup 2 --repeat 10 --steps 4000,2000 --entries run,update \
  --lock <lock-file> --cpus <two cores> --single-thread --pause 3 --max-iqr 0.1 --max-load 16 --retries 2 \
  --seed 0 --timeout 900 --budget-seconds 14400 --out <new-directory>
python -m benchmarks.performance.synapse_events.delivery_report <new-directory> --regression
```

The GPU run adds `--devices gpu --gpu <index>`. The CPU supplementary pass adds `--cases large,long,mixed --sibling-cpus 1,3 --max-sibling-busy 0.05 --retries 4`, and the GPU mini pass `--groups 0:shared:update,0:active:run --gpu-wait 7200`. The correctness tests are `RegressionCorrectnessTest` in [delivery_benchmark_test.py](../delivery_benchmark_test.py) (slow tier).
