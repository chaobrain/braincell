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

## Channel sweep differences

On 2026-11-08 every `channel_no_conc` configuration was compared with NEURON
under each template it declares. With the midpoint stimulus, most
configurations agree to `1.2e-7 mV` or better in every template. The groups
below differ for the reasons given; MECH-16 traced each one on 2026-11-09
(NEURON 9.0.2, JAX 0.12.0.1, double precision). In every case, patching only
the named cause in-process brought all three templates below `1e-9 mV`.
[`test_neuron_agreement.py`](channel_no_conc/tests/test_neuron_agreement.py)
asserts these configurations; their limits are the T1 entries of
[`tolerances.py`](channel_no_conc/tolerances.py), each naming its
deviation class and one of the headings below. The numbers
are in double precision, as in the tests; `engine/run.py` runs in single
precision unless `JAX_ENABLE_X64=1` is set (for `cav2p1_ma24_pc`, dc case 3
reads `0.99 mV` instead of `0.627 mV`).

| Configuration | Measured vinit / dc / ac (mV) | Limit vinit / dc / ac (mV) |
|---|---|---|
| `cav2p1` in `ma24_pc`, `ma25_bc`, `ri21_sc` | `2.8e-4` / `0.627` / `8.0e-5` | `5e-4` / `1.0` / `2e-4` |
| `cav3p1_ma20_goc`, `cav3p1_ma24_pc`, `cav3p1_test2_pc24` | `0.039` / `0.166` / `0.180` | `0.06` / `0.25` / `0.25` |
| `cav3p3_ma24_pc`, `cav3p3_ri21_sc` | `7.1e-15` / `6.9e-7` / `7.3e-9` | `1e-6` each |
| The six `_frozen` Cav2.1, Cav3.1 and Cav3.3 configurations | `<= 2.3e-11` | `1e-6` each |
| `calva_su15_dcn` | `3.2e-4` / `9.3e-5` / `9.3e-5` | `5e-4` / `2e-4` / `2e-4` |
| `na_zh19_io` | `2.6e-5` / `4.5e-11` / `2.1e-10` | `5e-5` / `1e-6` / `1e-6` |
| `kca2p2` in `ma20_goc`, `ma20_grc`, `ma25_bc`, `ri21_sc` | `4.2e-6` / `6.4e-6` / `3.5e-6` | `1e-5` each |

### Unfrozen GHK calcium channels

Class (d), `Cav2p1_*`, `Cav3p1_*`, `Cav3p3_*`: the
  `.mod` files compute the GHK drive (`gk`, `g`, `ghk`) inside `rates()` in
  the `DERIVATIVE` block. NEURON's finite-difference conductance therefore
  treats the drive as constant. The staggered solver linearizes the whole
  membrane current by autodiff, so the unfrozen classes include the drive's
  voltage derivative in the implicit voltage step. The `_Frozen` classes
  (`freeze_drive_gradient = True`) leave it out and match NEURON. The Cav2.1
  `_Frozen` classes also use the `.mod`'s GHK constants (`F = 9.6485e4`,
  `R = 8.3145`, `273.19 + celsius`); freezing alone leaves `7.4e-3 mV`. Both
  linearizations are first-order accurate: against a `dt = 1e-4 ms` reference
  at `dt = 0.01 ms`, the unfrozen and frozen Cav2.1 errors are `0.320` and
  `0.311 mV` (Cav3.1: `0.082` and `0.084 mV`). Both class sets are kept: the
  whole-cell models use the `_Frozen` Cav2.1 and Cav3.1 classes, and the
  unfrozen classes keep the drive in reverse-mode gradients.
  `Cav3p3_RI2021_SC` has no `_Frozen` class.

Under voltage clamp no voltage is solved, so the linearization does not
enter and the unfrozen classes are T0 (`cav3p1_ma20_goc`: `vc_steps`
`e_I = 2.9e-14`, `e_s = 7.4e-15`; `cav2p1_ma24_pc`: `5.3e-16`, `1.7e-16`,
since MECH-6a gave the unfrozen Cav2.1 classes the `.mod` constants; before
that, the CODATA constants alone gave `e_I = 1.39e-4`; MECH-17, 2026-11-11).

### CaLVA forward-difference conductance

Class (c), `calva_su15_dcn`: NEURON computes the membrane conductance as
  `(i(v + 0.001) - i(v)) / 0.001`. With `perm = 1 cm/s` the GHK current is
  stiff enough for that forward difference's truncation error to show; the
  BrainCell formula matches the `.mod`. Using the same difference in the
  BrainCell linearization gives `6.6e-10 mV`.

### Na_ZH2019_IO limit branch

Class (c), `na_zh19_io`: only the vinit case with `v_init = -50 mV`. At
  `|v + 50| < 1e-6` the `.mod` evaluates `b_h` at `v + 50.000001`; BrainCell
  takes the exact limit with `exprel` (see the `Na_ZH2019_IO` docstring), a
  relative difference of `5e-8` in `b_h(-50)`. Copying the `.mod` branch gives
  `4.5e-12 mV`.

### Kca2.2 steady-state initialization

Class (c), `kca2p2` (GoC, GrC, BC, SC): these `.mod` files initialize with
  `SOLVE kin STEADYSTATE sparse`, a backward-Euler step of `dt = 1e9` from
  zero. It leaves the initial state `6.1e-9` from the exact steady state
  (`d(state)/dt = 8e-10 /ms` at `t = 0`). BrainCell solves the steady state
  exactly, as the PC `.mod`'s `LINEAR` block does (that configuration agrees to
  `5e-9 mV`). Setting NEURON's initial state to the exact steady state with an
  `FInitializeHandler` gives `3.8e-12 mV`.

Not addressed: the current (`ix`) traces of the two runners are one step
apart, and NEURON's sample `k + 1` matches BrainCell's sample `k`. This affects
the current metrics only, not voltage or gates.

## Ion pool differences

### Cldif_JE2011 longitudinal diffusion

Class (d), `Cldif_JE2011`: ModelDB 148253's `cldif.mod` also diffuses chloride
along the cable (`LONGITUDINAL_DIFFUSION`); BrainCell has no ion flux between
control volumes, so only the radial shells are ported. On one segment the two
agree to T0 ([`chloride_test.py`](ion/chloride_test.py): `1.1e-12`). On a
200 um x 1 um dendrite (21 segments) with a 40 Hz GABA_A input at its middle
for 1 s, NEURON gives `cli = 5.277 mM` at the synapse, and `6.981 mM` without
that line; multi-compartment comparisons use the `.mod` without it.

## Locating a difference with the voltage clamp

The `channel_no_conc` voltage-clamp templates (`vc_steps`, `vc_ramp`) clamp
NEURON with an `SEClamp` and replay its voltage in BrainCell, so they compare
the channel equations alone, with currents aligned to the same sample. Read
the two protocols together:

- **Voltage clamp T0, current clamp above T0:** the difference is in the
  voltage coupling, i.e. the solver's linearization or NEURON's conductance
  estimate, not in the channel. The unfrozen Cav3.1 classes are the example
  (`0.166 mV` under `dc`, `e_I = 2.9e-14` under `vc_steps`).
- **Voltage clamp above T0:** the difference is in the channel equations,
  constants or initial values. It must be class (c) or (d), with a heading in
  this document and a T1 entry in the suite's tolerance table; otherwise it is
  a BrainCell defect (class (a)) and is fixed, not tolerated.
- **A channel-free case also differs:** the two sides receive different
  inputs (class (b)); fix the runner. Ion concentrations, for example, must be
  written before `finitialize`, or NEURON initializes concentration-dependent
  states and `eca` from its defaults (`Cav1p2_MA2020_GoC`: `e_s = 0.22`).

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
