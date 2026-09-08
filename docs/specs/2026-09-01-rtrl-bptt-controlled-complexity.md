# Controlled BPTT/RTRL Complexity Validation

## Goal

Separate algorithmic complexity from mechanism choice and GPU shape effects.
The mechanism-factorial sweep remains a practical ablation, not a complexity
proof. A synthetic recurrence validates independently controlled `N_x` and
`N_theta`; a fixed full-HH graph reports application-level scaling.

## Protocol

- A100, x64, 1,600 recurrent steps, batch 16, independent seeds 16.
- Three isolated worker processes per logical configuration.
- Ten synchronized steady calls per worker.
- Both reverse BPTT and exact full forward sensitivity.

The synthetic transition is a stable nearest-neighbour recurrence. Parameters
are padded into the state vector, so varying `N_theta <= N_x` does not change the
length of the primal state update:

```text
x[t+1] = tanh(0.6 x[t] + 0.2 roll(x[t], -1) + 0.2 roll(x[t], 1)
                + 0.02 pad(theta) + drive[t])
```

- State sweep: `N_x = 32, 64, 128, 256, 512, 1024`, `N_theta = 8`.
- Parameter sweep: `N_theta = 1, 2, 4, 8, 16, 32, 64`, `N_x = 512`.

The HH state sweep always paints Leak, K, and Na and trains their three globally
shared conductance scales over the full cell. Thus `N_theta = 3` while
`C = 3, 5, 9, 17, 33, 65` and nominal dynamical `N_x = 4C`.

The HH parameter sweep fixes full-HH dynamics at `C = 33`, `N_x = 132`, keeps K
and Na fixed, and trains the same Leak scale with `all`, `population`, `cv`, and
`row` grouping. With `B = 16`, this gives `N_theta = 1, 16, 33, 528`.

## Interpretation

For the synthetic recurrence, expected leading work is `O(T N_x)` for BPTT and
`O(T N_x N_theta)` for RTRL. The exact x64 RTRL carry is
`8 S B N_x N_theta` bytes. Wall-clock exponents are fitted over the largest four
points and compared with all-point fits; compiler cost and memory are reported
separately.

Figure 4 uses the full-HH controlled axes: runtime occupies the top row and XLA
temporary memory occupies the bottom row. The row-grouped `N_theta=528` point is
retained as a diagnostic but excluded from the main figure. Supplementary Figure
S4 applies the same layout to the synthetic axes and shows expected `N_x`
exponents `(1, 1)` and `N_theta` exponents `(0, 1)` for BPTT and RTRL. HH slopes
are application-level responses, not algorithmic proofs.

## Acceptance

- 22 logical configurations, two methods, three workers: 132 successful trials.
- Every worker stores all ten steady samples and compiler/GPU metadata.
- BPTT and RTRL loss/gradients agree in x64 for every paired trial.
- Synthetic and HH controlled dimensions satisfy the fixed-axis contracts.
- The publication notebook exports full-HH controlled scaling as Figure 4,
  mechanism ablation as Supplementary Figure S3, and synthetic controlled
  complexity as Supplementary Figure S4.
