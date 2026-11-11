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
| `cav2p1` in `ma24_pc`, `ma25_bc`, `ri21_sc` | `1.5e-4` / `0.632` / `4.6e-5` | `5e-4` / `1.0` / `1e-4` |
| `cav3p1_ma20_goc`, `cav3p1_ma24_pc`, `cav3p1_test2_pc24` | `0.039` / `0.166` / `0.180` | `0.06` / `0.25` / `0.25` |
| `cav3p3_ma24_pc`, `cav3p3_ri21_sc` | `7.1e-15` / `6.9e-7` / `7.3e-9` | `1e-6` each |
| The six `_frozen` Cav2.1, Cav3.1 and Cav3.3 configurations | `<= 2.3e-11` | `1e-6` each |
| `calva_su15_dcn` | `3.2e-4` / `9.3e-5` / `9.3e-5` | `5e-4` / `2e-4` / `2e-4` |
| `na_zh19_io` | `2.6e-5` / `4.5e-11` / `2.1e-10` | `5e-5` / `1e-6` / `1e-6` |
| `kca2p2` in `ma20_goc`, `ma20_grc`, `ma25_bc`, `ri21_sc` | `4.2e-6` / `6.4e-6` / `3.5e-6` | `1e-5` each |
| `ca_zh19_io` with `freeze_m_inf=False` | `3.9e-3` / `0.076` / `0.18` | `1e-2` / `0.2` / `0.5` |
| The five `kv1p1` configurations with `gateCurrent = 1` | `3.8e-4` / `3.4e-3` / `1.4e-3` | `1e-3` / `1e-2` / `5e-3` |
| `kv3p3_ma24_pc` with `gateCurrent = 1` | `2.5e-6` / `2.6e-3` / `6.2e-4` | `5e-6` / `5e-3` / `1e-3` |

The Cav2.1 row was measured on 2026-11-11, after MECH-6a; before it the
unfrozen classes used CODATA constants and gave `2.8e-4` / `0.627` /
`8.0e-5 mV`.

Under voltage clamp the catalogue tests (`tests/test_catalogue_*.py`, slow)
assert every configuration; the T1 entries explained in this document are
below (2026-11-11, MECH-19a; `e_I` is relative to the protocol's largest
current).

| Configuration | Measured `vc_steps` / `vc_ramp` (`e_I`, `e_s`) | Limit `vc_steps` / `vc_ramp` |
|---|---|---|
| `kca2p2` in `ma20_goc`, `ma20_grc`, `ma25_bc`, `ri21_sc` | `1.1e-6`, `3.1e-9` / `4.3e-7`, `3.1e-9` | `2e-6`, `5e-9` / `1e-6`, `5e-9` |
| `kca1p1` in `ma20_goc`, `ma20_grc`, `ma25_bc`, `ri21_sc` | `1.1e-13`, `3.1e-11` / `4.5e-10`, `2.8e-11` | T0 / `1e-9`, T0 |
| `cav3p3_ma24_pc`, `cav3p3_ma24_pc_frozen`, `cav3p3_ri21_sc` | `2.7e-14`, `3.0e-14` without the 0 mV step / `2.5e-5`, `1.1e-14` | T0 / `5e-5`, T0 |
| `cav3p1_test2_pc24` | `3.4e-6`, `7.4e-15` / `1.6e-6`, `2.6e-15` | `1e-5`, T0 / `5e-6`, T0 |
| `cahva_su15_dcn` | `6.7e-6`, `1.4e-15` / `7.2e-6`, `2.2e-16` | `2e-5`, T0 / `2e-5`, T0 |
| `calva_su15_dcn` | `4.1e-10`, `6.3e-15` / `7.4e-11`, `1.9e-15` | `1e-9`, T0 / T0 |
| `kdr_zh19_io` (MECH-19b) | `1.2e-15`, `3.3e-16` / `8.4e-10`, `6.5e-9` | T0 / `2e-9`, `1e-8` |

### Unfrozen GHK calcium channels

Class (d), `Cav2p1_*`, `Cav3p1_*`, `Cav3p3_*`: the
  `.mod` files compute the GHK drive (`gk`, `g`, `ghk`) inside `rates()` in
  the `DERIVATIVE` block. NEURON's finite-difference conductance therefore
  treats the drive as constant. The staggered solver linearizes the whole
  membrane current by autodiff, so the unfrozen classes include the drive's
  voltage derivative in the implicit voltage step. The `_Frozen` classes
  (`freeze_drive_gradient = True`) leave it out and match NEURON. Since
  MECH-6a every Cav2.1 class uses the `.mod`'s GHK constants (`F = 9.6485e4`,
  `R = 8.3145`, `273.19 + celsius`), so freezing is the whole difference:
  setting `freeze_drive_gradient = True` on `Cav2p1_MA2024_PC` gives
  `2.8e-13` / `4.7e-11` / `1.4e-12 mV` (vinit / dc / ac). Both
  linearizations are first-order accurate: against a `dt = 1e-4 ms` reference
  at `dt = 0.01 ms`, the unfrozen and frozen Cav2.1 errors are `0.320` and
  `0.311 mV` (Cav3.1: `0.082` and `0.084 mV`; measured before MECH-6a). Both class sets are kept: the
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
  BrainCell linearization gives `6.6e-10 mV`. Under voltage clamp the
  configuration is T0; its large current holds NEURON's `SEClamp` up to
  `3.3e-6 mV` from the command (`rs = 1e-9 MΩ`), which does not enter the
  comparison because BrainCell replays NEURON's voltage.

### Na_ZH2019_IO limit branch

Class (c), `na_zh19_io`: only the vinit case with `v_init = -50 mV`. At
  `|v + 50| < 1e-6` the `.mod` evaluates `b_h` at `v + 50.000001`; BrainCell
  takes the exact limit with `exprel` (see the `Na_ZH2019_IO` docstring), a
  relative difference of `5e-8` in `b_h(-50)`. Copying the `.mod` branch gives
  `4.5e-12 mV`.

### Kdr_ZH2019_IO limit branch

Class (c), `kdr_zh19_io` (`vc_ramp`): the `.mod` evaluates `a_n` at
  `v + 41.00001` when `|v + 41| < 1e-6`, the same device as `Na_ZH2019_IO`'s
  `b_h`; BrainCell takes the exact limit with `exprel`. The ramp's sample at
  `t = 64.025 ms` lies `3.5e-10 mV` from `-41 mV`; there `a_n` is off by
  `5e-6 /ms`, and the gate `n` carries the offset for the rest of the ramp
  (`e_I = 8.4e-10`, `e_s = 6.5e-9`). The steps never land within `1e-6 mV`
  of `-41 mV` and are T0. Copying the `.mod` branch into BrainCell gives
  `e_I = 1.0e-15`, `e_s = 2.2e-16`.

### Kca2.2 steady-state initialization

Class (c), `kca2p2` (GoC, GrC, BC, SC): these `.mod` files initialize with
  `SOLVE kin STEADYSTATE sparse`, a backward-Euler step of `dt = 1e9` from
  zero. It leaves the initial state `6.1e-9` from the exact steady state
  (`d(state)/dt = 8e-10 /ms` at `t = 0`). BrainCell solves the steady state
  exactly, as the PC `.mod`'s `LINEAR` block does (that configuration agrees to
  `5e-9 mV`). Setting NEURON's initial state to the exact steady state with an
  `FInitializeHandler` gives `3.8e-12 mV`. Under voltage clamp the same
  initial offset gives `e_I = 1.1e-6`, `e_s = 3.1e-9` (`vc_steps`); writing
  BrainCell's initial state into NEURON after `finitialize` gives
  `e_I = 3.3e-13`, `e_s = 1.0e-16` in both protocols.

### Kca1.1 steady-state initialization

Class (c), `kca1p1` (GoC, GrC, BC, SC): the same `SOLVE activation
  STEADYSTATE sparse` as Kca2.2, in a ten-state scheme. The initial state is
  `1.3e-10` from the exact steady state at `-80 mV` (`1.1e-10` at `-100 mV`,
  where `vc_ramp` starts). The current error under `vc_ramp`,
  `e_I = 4.5e-10`, exceeds T0 while the states stay below it
  (`e_s <= 3.1e-11`). The PC `.mod` solves a `LINEAR` system and is T0
  (`e_I = 2.9e-14`). Writing BrainCell's initial state into NEURON after
  `finitialize` gives `e_I = 2.9e-14`, `e_s = 2.8e-16` (`vc_ramp`). Under
  current clamp all three templates stay within T0 (`<= 5.0e-9 mV`).

### GHK drive near 0 mV in mod files without a limit branch

Class (c), `cav3p3_ma24_pc`, `cav3p3_ma24_pc_frozen`, `cav3p3_ri21_sc`
  (`vc_ramp`) and `cav3p1_test2_pc24` (both protocols): these `.mod` files
  compute the GHK drive with `w/(exp(w) - 1)`, `w = v*0.001*z*F/(R*T)`, and no
  branch at `w = 0`. `cahva_su15_dcn` (both protocols) and `calva_su15_dcn`
  (`vc_steps`) are the same case below. Where the clamped voltage is within `1e-11 mV` of 0 mV
  (the ramp sample at `t = 105 ms`, `v = -1.0e-11 mV`; the first sample of the
  0 mV step, `v = 4.7e-13 mV`), `exp(w) - 1` keeps only a few significant
  digits, and NEURON's current there is off by up to `2.5e-5` of the
  protocol's largest current. BrainCell switches to the series `1 - w/2`
  when `|exp(w) - 1| < 1e-6`, as `Cav3p1_MA24_PC.mod` does in its `ghk()`;
  `cav3p1_ma24_pc`, the same class against that `.mod`, is T0. Evaluating the
  `.mod`'s expression without the branch in BrainCell gives
  `e_I = 1.3e-14` (Cav3.3 `vc_ramp`), `2.9e-14` and `4.1e-15` (test2
  `vc_steps`, `vc_ramp`); the same patch raises `cav3p1_ma24_pc`'s `vc_ramp`
  to `1.6e-6`. Closer still, the expression is singular: in the Cav3.3
  `vc_steps` run, the second sample of the 0 mV step is `v = -1.3e-17 mV`,
  where `exp(w)` rounds to exactly 1, so `ica` and then `v` and the gates are
  not finite for the rest of that command (1198 of 1401 samples). The
  comparison leaves that command out (`NON_FINITE_REFERENCE` in
  `tolerances.py`; the test checks that it is still not finite); the other
  seven are T0. Before MECH-19a the metrics dropped such NaN samples
  silently and reported this protocol as T0; they now raise `ValueError` on
  any non-finite sample. The current clamp of these configurations is not
  affected (the unfrozen Cav3.1 rows above are the linearization).

  The two SU2015 DCN `.mod` files compute `(v/T)/(1 - A)` with
  `A = exp(-23.20764929 v/T)`, again without a branch. Since MECH-23a,
  BrainCell evaluates this ratio through `exprel`, so it is finite at
  0 mV and exact near it. The largest differences sit at the samples nearest
  0 mV: `v = 3.0e-12 mV` in the 0 mV step and `1.7e-11 mV` on the ramp
  (`cahva_su15_dcn`, `e_I = 6.7e-6`, `7.2e-6`), and `v = 1.5e-8 mV`
  (`calva_su15_dcn` `vc_steps`, `4.1e-10`). Evaluating the `.mod`'s form, as
  BrainCell did before MECH-23a, gives `e_I <= 8.2e-15`.

### Ca_ZH2019_IO with freeze_m_inf=False

Class (d), `ca_zh19_io` with the non-default `freeze_m_inf=False`: the `.mod`
  computes `minf` in `rates()`, called from the `DERIVATIVE` block, so
  NEURON's finite-difference conductance treats it as constant, as the default
  `freeze_m_inf=True` does (`1.3e-12 mV`). With `False`, the staggered
  solver's autodiff linearization also includes `d(minf)/dV`, the same
  mechanism as the unfrozen GHK classes above. The option does not change the
  current, and the voltage clamp stays T0 (`e_I = 1.3e-14`).

### Kv1.1 and Kv3.3 gating current

Class (d), the `kv1p1` configurations (GoC, GrC, PC, BC, SC) and
  `kv3p3_ma24_pc` with the non-default `gateCurrent = 1`: the `.mod` adds the
  gating current `igate = nc*e0*4*zn*(ninf - n)/taun` as a separate
  `NONSPECIFIC_CURRENT i`, with `ninf` and `taun` from `rates()` in the
  `DERIVATIVE` block, so NEURON's finite-difference conductance holds `igate`
  constant in `v`. BrainCell folds `igate` into the channel current, and the
  staggered solver's autodiff linearization includes `d(igate)/dV`, as for
  `freeze_m_inf=False` above. The current itself agrees: under voltage clamp,
  with NEURON's `i` added to `ik`, both protocols are T0 (`e_I <= 1.2e-15`).
  Hiding `igate`'s dependence on `V` from the linearization
  (`jax.lax.stop_gradient`) gives `<= 2.1e-13 mV` in all three templates.
  The other default-off path of this family, `Kv1p5_MA2020_GrC` with
  `gnonspec > 0` (NEURON's `ino` added to `ik`), is T0 in all five
  protocols.

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
