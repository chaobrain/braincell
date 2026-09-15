# RTRL speed exploration review

## Final decision

The accepted speed combination is:

```text
full RTRL
→ generic DHS AD
→ jax.linearize + vmap(linear_map)
→ existing parameter-major tangent tree
→ vectorized rollout materialization at the rollout boundary
→ staggered solver with recursive backsub
```

The compact path remains available for memory-constrained workloads. It is not
the default speed path. No solver recurrence, reset behavior, parameter
mapping, or exact RTRL definition was changed by the accepted performance
path.

## Fixed measurement protocol

The main profiling protocol used x64, GPU 0, one worker at a time, `B=S=16`,
40 ms, `dt=0.025 ms`, 1600 steps, and full Leak/K/Na HH. Each worker used one
first execution, two additional warmups, and five steady samples. Target
generation and compilation were recorded separately from steady loss-plus-
gradient timing. GPU polling was disabled during timed runs.

The fixed profiling points were C=1, 5, and 21 with `Ntheta=3C`.

## Baseline profile

| C | method | trace (s) | lower (s) | XLA compile (s) | steady median (s) | IQR (s) | temporary MiB | logical carry MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | BPTT | 0.686 | 0.131 | 1.814 | 0.188825 | 0.000252 | 265.84 | — |
| 1 | RTRL | 0.549 | 0.119 | 1.246 | 0.066623 | 0.000102 | 0.31 | 0.12 |
| 5 | BPTT | 1.015 | 0.166 | 2.040 | 0.370697 | 0.000051 | 1328.74 | — |
| 5 | RTRL | 0.828 | 0.146 | 1.328 | 0.169691 | 0.000033 | 2.43 | 2.50 |
| 21 | BPTT | 1.806 | 0.234 | 3.297 | 0.962436 | 0.000287 | 6109.38 | — |
| 21 | RTRL | 1.294 | 0.241 | 1.999 | 0.723853 | 0.000060 | 37.92 | 42.74 |

Full RTRL is faster than BPTT at all three points. Its advantage narrows as C
grows, but its compiler temporary footprint remains much smaller.

## Adopted changes

### Rollout materialization

Parameter materialization was moved to the rollout boundary when bindings are
provably fixed and read-only. The implementation keeps a guarded fallback for
step-level materialization when custom setters, writes, or reset-dependent
behavior make hoisting unsafe. This reduced repeated materialization work and
compile cost while preserving parameter mapping and reset semantics.

### Profiling and reproducibility

The benchmark now records compile phases, first/warmup/steady samples, XLA
memory analysis, logical carry bytes, StableHLO operation counts, source and
environment provenance, and explicit JVP mode settings. The suite supports
generic/explicit DHS A/B and linearize/direct RTRL JVP A/B without changing
the default path.

### Generic DHS AD default

The explicit DHS custom-JVP kernels are numerically correct, but generic DHS
AD was faster for RTRL:

| C | explicit DHS RTRL (s) | generic DHS RTRL (s) | generic / explicit |
|---:|---:|---:|---:|
| 1 | 0.066623 | 0.067147 | 1.008 |
| 5 | 0.169691 | 0.146248 | 0.862 |
| 21 | 0.723853 | 0.602850 | 0.833 |

Generic DHS AD is therefore the benchmark speed default. Explicit JVP stays
as a correctness and diagnostic switch.

## Explored and rejected candidates

### Direct `vmap(jvp)`

Direct JVP was effectively tied with `linearize + vmap`:

| C | linearize (s) | direct (s) |
|---:|---:|---:|
| 1 | 0.067147 | 0.066228 |
| 5 | 0.146248 | 0.145832 |
| 21 | 0.602850 | 0.602675 |

The sub-percent differences are below the 3% threshold.

### State-tree alignment

Alignment occurs during initialization/reset carry construction. The scan body
passes array values and tangents directly, so there is no repeated per-step
state metadata alignment to remove.

### Direction-axis layout

A temporary state-major layout passed correctness but regressed slightly:

- C=5: 0.146248 → 0.146659 s;
- C=21: 0.602850 → 0.603708 s.

The experimental code was removed.

### Dense tangent packing

Packing all tangent leaves into one vector and unpacking around every
transition passed the C=5 numerical check but changed 0.146248 → 0.172319 s,
17.8% slower. The code was removed before C=21.

### Local-gradient fusion

StableHLO already carries the small gradient accumulator inside the scan
while. No independent large local-gradient workspace was found, so a manual
fusion would duplicate compiler behavior.

### Compact active transition

The HH active projection retains `V`, `Na.m`, `Na.h`, and `K.n`, but the generic
transition still requires the complete state pytree. Current compact RTRL
therefore performs full tangent embedding and active extraction each step. A
fresh correctness run had zero gradient error, but compact was 0.197322 s
versus 0.183003 s for full RTRL, 7.8% slower. Compact remains a memory path.

### GPU/kernel specialization

A short Nsight Systems C=5 trace contained 5,972 kernels and 7.98 ms total
device kernel duration. `wrapped_dynamic_slice` accounted for 4,805 launches
and 5.35 ms. Tangent indexing was minor (`wrapped_gather`: 0.036 ms;
`input_scatter_fusion`: 0.025 ms). The visible hotspot is time-history output
writing, not RTRL tangent indexing. No specialized tangent kernel was added.
Nsight GUI post-processing reported a CUDA 13 compatibility error, but the
SQLite event database was readable and used for the summary.

## Cost model conclusion

For C=5 full RTRL, XLA cost analysis estimated approximately 5.47 million
FLOPs and 26.64 MB of bytes accessed per compiled call, or about 0.21
FLOP/byte. The workload is traffic-dominated, but the tested layout and
packing changes increased overhead rather than improving traffic. Further
optimization would require a deliberate solver/output-history redesign or a
specialized derivative interface, both outside the current semantic scope.

## Final retained combinations

| Use case | Retained path |
|---|---|
| Default speed | full RTRL + generic DHS AD + `linearize + vmap` |
| Memory constrained | compact RTRL with existing embed/extract projection |
| Reverse-mode reference | full BPTT |
| Diagnostics | explicit DHS JVP and direct JVP switches, state sensitivity inspection, compile/HLO/cost diagnostics |

All rejected candidates remain documented here and in the candidate screening
log. They should not be restored unless the solver or output-history design is
changed deliberately.
