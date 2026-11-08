# Known numerical differences

The following failures were reproduced on both the pre-migration commit
`bd8e2f9` and the reorganized checkout on 2026-09-08, using NEURON 8.2.6,
JAX 0.10.1 and double precision. The original assertions remain in place.

- **SC morphology geometry:** the
  [morphology comparison](morph/neuron_diff_test.py) reports maximum Euclidean
  distance difference `8.559982695066495 um`, above its `1e-4 um` limit.
  This uses the preserved IO fixture variant, not the whole-cell morphology.

Resolved:

- **Sinusoidal channel stimulus** (MECH-15): the
  [comparison test](channel_no_conc/tests/test_compare.py) reported voltage MAE
  `0.015916517057767133 mV`, above its `2e-5 mV` limit (reproduced again on
  2026-11-08 at `a72ca45` with NEURON 9.0.2 and JAX 0.12.0.1). BrainCell
  evaluates a current clamp once per step at `t + dt/2`; the NEURON runner
  played `I(t)` for the step starting at `t`, half a step earlier. The runner
  now plays `I(t + dt/2)`; the test's voltage MAE is `6.4e-11 mV`. DC and
  vinit cases are unchanged, because their stimulus is constant within
  every step.
- **DCN SK initialization** (MECH-14): the
  [BrainCell channel smoke test](channel_no_conc/tests/test_braincell_runner.py)
  changed a scan carry from scalar `float64[]` to `float64[1,1]`. The case
  paints no calcium ion, so `SK_SU2015_DCN`'s gate `z`, whose steady state
  reads only `Ca.Ci`, was reset to a scalar instead of the channel shape.
  `HH.reset_state` now broadcasts each gate to the shape `init_state`
  allocated. The SU15 DCN SK configuration then matches NEURON to
  `4.7e-10 mV` (voltage, vinit and dc templates) and `1.1e-19` (gate `z`).

## Untested differences in the full channel sweep

On 2026-11-08 every `channel_no_conc` configuration was compared with NEURON
under each template it declares (883 cases). With the midpoint stimulus, 77 of
the 89 configurations agree to `1.2e-7 mV` or better in every template. The
remaining 12 differ in vinit and dc as well, so the difference is in the
channel, not the stimulus. No test asserts these cases' accuracy (their smoke
tests check shapes and finiteness only). The numbers are in double precision,
as in the tests; `engine/run.py` runs in single precision unless
`JAX_ENABLE_X64=1` is set (for `cav2p1_ma24_pc`, dc case 3 reads `0.99 mV`
instead of `0.627 mV`):

| Configuration | Max voltage difference, vinit and dc | Max, ac |
|---|---|---|
| `cav2p1` in `ma24_pc`, `ma25_bc`, `ri21_sc` | `0.627 mV` | `8.0e-5 mV` |
| `cav3p1` in `ma20_goc`, `ma24_pc` (`cav3p1_ma24_pc`, `cav3p1_test2_pc24`) | `0.166 mV` | `0.180 mV` |
| `calva_su15_dcn` | `3.2e-4 mV` | `9.3e-5 mV` |
| `na_zh19_io` | `2.6e-5 mV` | `8.7e-8 mV` |
| `kca2p2` in `ma20_goc`, `ma20_grc`, `ma25_bc`, `ri21_sc` | `6.4e-6 mV` | `3.5e-6 mV` |

Run each workflow in a fresh process. Combining cell and channel suites in one
pytest process can attempt to register the same NEURON mechanism from both the
cell and channel libraries; NEURON then raises a duplicate-name error. The
separate build directories preserve each workflow's mechanism selection.

```bash
pytest -q validation/neuron/cable/tests
pytest -q validation/neuron/cell
pytest -q validation/neuron/channel_no_conc/tests
pytest -q validation/neuron/morph
```

Model-specific import and comparison status is maintained in
[import progress](cerebellum-import-progress.md). These discrepancies need their
own numerical fixes; moving files preserves the model inputs and tolerance values.
