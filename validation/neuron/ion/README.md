# ion

Calcium-pool comparisons against NEURON. Each configuration puts one pool,
with or without one calcium channel, on a single soma (L = diam = 20 um) and
runs it under up to three protocols; the limits come from one tolerance table.

| File | Contents |
|---|---|
| `configs/<model>/<name>.json` | One pool: its `.mod` sources under `data/`, the NEURON and BrainCell names of the pool, its states and parameters, the channel and its gates (the channel_no_conc layout plus `pool_name`, `pool_states`, `ci_state`) |
| `templates/pool_*.json` | The protocols, see below |
| `runner.py` | Loads a configuration, runs both sides and returns the metrics (`compare_protocol`) |
| `tolerances.py` | The T1 entries; every other configuration and protocol is T0 |
| `runner_test.py` | Configurations, templates and table, without NEURON (pull-request tier) |
| `tolerances_test.py` | Every configuration within the table, with its root causes patched at T0, and a pump-rate counterexample (`slow`, daily tier) |
| `chloride_test.py` | `Cldif_JE2011` against ModelDB 148253 (local only, see below) |
| `gaba_test.py` | `GabaA_JE2011` with `Cldif_JE2011` against ModelDB 148253 (local only, see below) |

## Protocols

| Template | Protocol | Metrics |
|---|---|---|
| `pool_zero_current` | No channel; the membrane is held at `v_init`. The pool relaxes from its initial state or from `defaults.initial_states` (written after `finitialize`) | `e_Ci`, `e_pool` |
| `pool_vc` | `SEClamp` -70 -> -10 -> -70 mV (5, 20, 25 ms). BrainCell replays NEURON's voltage, so both pools see the current of the same voltage | `e_I`, `e_s`, `e_Ci`, `e_pool` |
| `pool_cc` | `IClamp` pulse 5-25 ms (0.005 nA unless the configuration sets `current_clamp.amp_nA`); both sides solve V | `e_V`, `e_Ci`, `e_pool` |

All run at `dt = 0.025 ms` in double precision; sample `i` of both sides is the
state at `t = i dt`, so the initial state is compared too. `e_Ci` is the
largest `Ci` difference over the protocol's largest NEURON `cai`; `e_pool` is
the same ratio for each listed pool state over its own largest value, the
largest of these; `e_I`, `e_s` and `e_V` are the channel suite's metrics
(`validation/neuron/_tolerance.py`). T0 is `1e-10` for `e_Ci` and `e_pool`.

## Results

With BrainCell's defaults, the kinetic pools differ from NEURON by `e_Ci`
`1.7e-7` to `2.0e-4`: NEURON iterates its implicit `sparse` step, BrainCell's
`backward_euler` takes one linearized step (class (d)). The other T1 entries
are the CdpHVA pool's explicit update in NEURON, NEURON's forward-difference
conductance and the unfrozen GHK channels under current clamp. Each is a
heading of [known differences](../known-differences.md); with every named
cause patched, all 22 runs are T0 (`tolerances_test.py`).

The configurations replace the earlier notebooks: GoC `CdpStC_MA2020_GoC`
with CaHVA, Cav3.1 and the SC Cav2.1; PC `CdpCAM_MA2024_PC` with Cav3.1 and
Cav3.2, and against the original `cdp5_CAM` at the whole-cell pump density;
DCN `CdpHVA_SU2015_DCN`; and the four DCN toy kinetic pools.

## Running

The runner compiles every configuration's sources into one library and loads
it once; run the suite in its own process, apart from `channel_no_conc`,
whose libraries define the same channels.

```bash
pytest -q validation/neuron/ion -m "not slow"   # pull-request tier
pytest -q validation/neuron/ion -m slow         # daily tier, about 40 s
```

```python
from validation.neuron.ion import runner

result = runner.compare_protocol("goc_ma2020/cdpstc_ma20_goc", "pool_vc")
result["e_Ci"], result["neuron"]["cai"], result["braincell"]["cai"]
```

A new configuration lists its `.mod` files, names and templates; a pool
whose NEURON state for `cai` is not `cai` itself names it in `ci_state`. A
difference above T0 needs its cause, a patch that removes it and an entry in
`tolerances.py`.

`chloride_test.py` compares `Cldif_JE2011` with ModelDB 148253's `cldifus` on
one segment, as a zero-current relaxation and with `Cl_Leak` under current
clamp (T0). `cldif.mod` has no license
statement and whether it may be committed awaits the maintainers' decision;
until then the test runs only when `BRAINCELL_JE2011_MOD_DIR` points to a
directory with that file (`ModelDBRepository/148253@1d737a0`, SHA-256 checked
by the test), and skips otherwise:

```bash
BRAINCELL_JE2011_MOD_DIR=/path/to/148253 pytest -q validation/neuron/ion/chloride_test.py
```

`gaba_test.py` compares `GabaA_JE2011` with `Cldif_JE2011` against `gaba` with
`cldifus` on the same segment (NetStim-driven, 1005 ms, T0). It also needs
`gabaA_Cl.mod` in that directory (SHA-256 checked by the test) and skips the
same way.
