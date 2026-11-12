# Parameter Learning: Past Experiments and Training Diagnostics

## Current state

This directory keeps the parameter-fitting training diagnostics and the composable 1-CV experiment
framework. It is not public API, and it no longer contains the source of the early full
nine-parameter training, dataset, ablation and demo code. Initialization and stimulus design have
moved to their own experiment directories; the code currently in this directory is:

```text
diagnostics.py
diagnostics_test.py
config.py / models.py / datasets.py / losses.py
optimizers.py / search.py / training.py / reporting.py / run.py
configs/
```

For the related DC/Sobol/Nevergrad experiments see [`optim_initialization`](../initialization); for
PRMLS, OED and the global ensemble see [`optim_stimulus_design`](../stimulus_design).

## Active code

`diagnostics.py` provides experiment-local, optimizer-agnostic multistart training diagnostics:

- optimizer-space and physical parameter history;
- total/component loss and evaluator metrics;
- gradient norm, gradient cosine and optimizer/physical step norm;
- parameter bound position;
- per-start continuous-best and spike-feasible best archives;
- failure summary, plotting and artifact serialization.

It does not handle:

- model parameter declaration and runtime materialization; `braincell.trainable` handles these;
- BPTT/RTRL or the solver; BrainState/JAX and the corresponding experiment modules handle these;
- the optimizer; `braintools.optim` is still used;
- a general Trainer, Dataset or checkpoint public API.

Current direct callers:

```text
examples/optim/parameter_learning/trainable_hh_multistart.py
examples/optim/parameter_learning/trainable_hh_multistart_test.py
examples/optim/parameter_learning/train.ipynb
```

This helper applies equally to BPTT, RTRL or any other experiment that returns the same named
gradient mapping, so the framework stays in `optim_parameter_fitting/` and is not merged into
`optim/`, which only provides model-independent gradient machinery.

## Hybrid Initialization Experiment

`dc_protocol_dataset.py` builds a 3-CV synthetic target with 9 conductance scales. Each protocol is
a constant DC over the full 50 ms; the currents are calibrated independently at the three injection
sites from the target response. The fixed split is:

| Split | Per site | Total |
| --- | --- | ---: |
| train | `-80/-110 mV`, `0/1/3 spike` | 15 |
| validation | `-90 mV`, `2 spike` | 6 |
| test | `-100 mV`, `4 spike` | 6 |

`../optim_initialization/hybrid_initialization.py` fixes raw voltage MSE and exact RTRL + Adam, and
compares only the initialization entry point:

| Method | Forward screening budget | Adam starts |
| --- | ---: | ---: |
| direct random | 16 (no screening) | 16 |
| random screen | 1024 | 16 |
| scrambled Sobol | 1024 | 16 |
| Nevergrad TwoPointsDE | `64 x 16 = 1024` | 16 |

All methods work in `[0.5, 1.5]` of the nine-dimensional relative physical scale and all generate
candidates in log space. Nevergrad is an optional dependency:

```bash
pip install -e '.[optim]'
```

First run the single-search-seed pilot:

```bash
python examples/optim/initialization/hybrid_initialization.py run \
  --stage pilot --gpu 7 \
  --python ~/anaconda3/envs/braincell_311/bin/python \
  --resume
```

Once the pilot passes, run search seeds `0/1/2`:

```bash
python examples/optim/initialization/hybrid_initialization.py run \
  --stage formal --gpu 7 \
  --python ~/anaconda3/envs/braincell_311/bin/python \
  --resume
```

Every update saves the loss, the parameters, the raw gradient, the gradient norm and the optimizer
delta; every 10 updates save the per-protocol MSE, spike counts and timing. Full voltages are saved
only for the initial, validation-best and final states. The test split never takes part in
candidate ranking or checkpoint selection.

Generated results are in the Git-ignored directory:

```text
examples/optim/initialization/artifacts/hybrid_initialization/
```

The current A100 x64 formal results use search seeds `0/1/2`, training 48 starts per method in
total:

| Method | Trace success | Parameter success | Joint success | Median validation MSE | Median parameter relative RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct random | 14/48 | 0/48 | 0/48 | 33.519 | 0.248 |
| random-1024 | 19/48 | 1/48 | 0/48 | 24.321 | 0.260 |
| Sobol-1024 | 21/48 | 0/48 | 0/48 | 24.097 | 0.251 |
| TwoPointsDE-1024 | 19/48 | 0/48 | 0/48 | 24.216 | 0.248 |

Here trace success requires every validation spike count to be correct and the aggregate RMSE to be
at most 5 mV; parameter success requires the nine-parameter relative RMS to be at most 10%. All three
1024-evaluation screenings raised trace success, with Sobol highest in this round, but no method
produced a joint success. Initialization screening therefore already raises the probability of
entering the region of fittable trajectories, while DC-only/raw-MSE data is still not enough to
identify the nine conductances reliably. The next round should first analyse the conditioning of
the observation Jacobian before deciding whether to add protocols or change the loss.

## Stimulus Design v2

`../optim_stimulus_design/dataset.py` keeps the old morphology geometry but uses `1/3/3` CVs and six
parameters: soma Leak/Na/K, and Leak/Na/K shared by all dend CVs. It generates a uniform 100 ms
current tensor:

```text
0--20 ms baseline / 20--80 ms stimulus / 80--100 ms recovery
```

The candidate inputs include feature-based steps and four-level PRMLS at `2/5/10 ms`, combined with
three injection sites: the soma, distal dend_a and distal dend_b. In total there are
`train/validation/test = 33/15/12` protocols, and the 7-CV voltage is saved for all of them.

`../optim_stimulus_design/robust_oed.py` accumulates the observation Fisher information online with
exact RTRL at the target and 16 log-space Sobol parameter points, and produces a robust D-optimal
greedy ordering of the 33 train candidates only. It does not run parameter optimization and does
not freeze the final train subset automatically.

```bash
python examples/optim/stimulus_design/dataset.py
python examples/optim/stimulus_design/robust_oed.py
```

Current A100 x64 dataset/OED results:

```text
target dataset: current (60, 4000, 3), voltage (60, 4000, 7)
PRMLS: all locations subthreshold, global A = 0.05931144 nA
OED references: target + 16 Sobol prior points
OED candidates: 33/33 finite
compile: 24.92 s
FIM execution: 0.78 s
XLA temporary: 2.99 MB
```

Under the relative rank threshold, the first protocol of the greedy prefix already reaches rank 6,
but the worst condition number is still `1.36e7`, still about `1.89e7` after all 33 protocols, and
the worst parameter-column correlation is about `0.99997`. These stimuli therefore resolve all six
directions numerically, yet strong practical compensation remains; "full rank" alone is no reason to
freeze the minimal training set. The first ten entries of the current ordering are the 1-spike,
mild-negative and 3-spike protocols at the three sites plus dend_b small-positive; PRMLS enters from
the 11th entry. Review the full conditioning curve before choosing the final subset.

`../optim_stimulus_design/global_ensemble.py` evaluates 16384 forward-only Sobol candidates in the
six-dimensional log-scale parameter box, saving the MSE of all 60 protocols and the hard spike
counts. It uses two train scores:

```text
raw MSE
prior-median protocol-normalized MSE
```

```bash
python examples/optim/stimulus_design/global_ensemble.py
```

Current A100 x64 results:

```text
candidates: 16384
compile: 21.84 s
ensemble/prior evaluation: 83.86 s
weak/strong/plane profiles: 7.16 s
XLA temporary: 875.73 MB
raw/normalized Top256 overlap: 130, Jaccard 0.340
raw Top PC1 vs FIM weak cosine: 0.871
normalized Top PC1 vs FIM weak cosine: 0.790
```

The raw Top256 has `12/51` exact validation/test spike signatures and the normalized Top256 has
`4/29`; a low voltage score does not guarantee that every hard spike count is correct. The median
parameter relative RMS of both Top256 sets is still about `0.25`, so the best sparse global samples
so far are far from the synthetic target. The target-centered weak-direction profile shows clearly
shallow directions near the target, so the 16384-point Sobol ensemble mainly serves to check the
geometric trend; it cannot stand in for a full posterior or prove that all equivalent solutions
have been found.

### Composable Python baseline zero

This directory is the current fixed experiment entry point, not BrainCell public API. A Python
preset composes the real components directly, without a JSON string registry:

```python
CONFIG = ExperimentConfig(
    model=hh_1cv_classic_bounded_direct(),
    dataset=feature_step_dataset(),
    loss=raw_voltage_mse(),
    initialization=InitializationConfig(seed=0, num_candidates=64),
    stages=(adam(epochs=180, learning_rate=0.01, gradient_method="rtrl"),),
)
```

Component responsibilities:

| Module | Replaceable content | Fixed contract |
| --- | --- | --- |
| `models.py` | morphology, channel, parameter space | physical / bounded `u` / optimizer `z` round trip |
| `datasets.py` | waveform, split, target generation | train/validation/test metadata |
| `losses.py` | local/trajectory objective | training and split evaluation mean the same thing |
| `optimizers.py` | Adam and other gradient stages | the optimizer receives only the `z` gradient |
| `search.py` | forward-only/non-gradient stage | hands over through a physical `CandidateSet` |
| `training.py` | stage pipeline, RTRL/BPTT | never splits candidates implicitly |
| `reporting.py` | success, CSV/JSON, plots | test final-only |
| `run.py` | worker, resume, GPU monitor | one fixed CLI |

A new ordinary configuration field gets a default value in the dataclass; an incompatible behaviour
change adds a versioned factory. Non-gradient stages use only the train objective; when a DFO stage
hands back to a gradient stage, `z` is recovered from the physical parameters and the optimizer
moments are rebuilt. Stages compose:

```python
stages = (sobol_screen(...), adam(...))
stages = (adam(...), differential_evolution(...))
stages = (adam(...), local_search(...), adam(...))
```

Only plain Adam is run formally at present; `ForwardSelectionStage` checks the stage hand-over
contract and does not pose as a real search algorithm.

Model zero is one 25 um soma CV that learns the bounded classic HH conductances directly:

| Parameter | Target | Bounds | Runtime mapping |
| --- | ---: | ---: | --- |
| `leak.g_max` | `0.3` | `[0.15,0.45] mS/cm^2` | direct parameter |
| `na.g_max` | `120` | `[60,180] mS/cm^2` | direct parameter |
| `k.g_max` | `36` | `[18,54] mS/cm^2` | direct parameter |

The cell runtime uses `theta=lower+(upper-lower)*sigmoid(z)`; there is no frozen baseline scale.
All data are DC steps over `0--20/20--80/80--100 ms` with a `5/2/1` split; PRMLS is not part of
experiment zero. The 64 starts are generated from one seed and enter the same RTRL kernel and Adam
optimizer together, without splitting into cohorts. The test split is evaluated only at epoch 180.

```bash
python examples/optim/parameter_fitting/run.py run \
  --config examples/optim/parameter_fitting/configs/basic_1cv_bounded_direct_adam.py \
  --gpu 1 --resume
```

Current A100 x64 formal results:

| Metric | Value |
| --- | ---: |
| trace success | `8/64 = 12.5%` |
| trace-success Wilson 95% CI | `[6.47%,22.77%]` |
| parameter / joint success | `3/64 / 1/64` |
| median train MSE | `61.2255 mV^2` |
| median validation / test RMSE | `10.6116 / 17.2393 mV` |
| median parameter relative RMS | `0.2260` |
| compile / 180-update stage / end-to-end | `2.85 / 48.88 / 82.50 s` |
| XLA temporary / monitored peak GPU memory | `2.10 MiB / 1166 MiB` |

All 64 endpoints are finite; `22/64` get every validation count right, `8/64` have an RMSE of at
most 5 mV, and the intersection is `8/64`. `35/64` get every test count right, and `5/64` meet
both the 5 mV and the count criteria. The results are in the Git-ignored directory:

```text
examples/optim/parameter_fitting/artifacts/parameter_experiments/
  20260831-210204_adam-narrow-e180_db936968/
```

The older `6/64` result (7 CVs, 6 scales, four cohorts, PRMLS mixed into the holdout) is kept as
legacy and is no longer the zero reference.

#### Epoch budget: 180 vs 300

`basic_1cv_bounded_direct_adam_e300.py` is derived with `dataclasses.replace()`; the only change in
the resolved config is `stages[0].epochs: 180 -> 300`. The initial physical/normalized/`z` values
are bitwise identical, the optimizer `z`, physical parameters and gradients of the first 180 epochs
are bitwise identical, and the epoch-180 validation is exactly the same.

| Metric | Epoch 180 | Epoch 300 |
| --- | ---: | ---: |
| trace success | `8/64` | `10/64` |
| parameter success | `3/64` | `6/64` |
| joint success | `1/64` | `1/64` |
| median train MSE | `61.2255` | `55.3886 mV^2` |
| median validation RMSE | `10.6116` | `10.1949 mV` |
| median test RMSE | `17.2393` | `17.1670 mV` |
| end-to-end | `82.50` | `112.95 s` |

Pairing by start shows that trace success is kept by `6`, gained by `4` and lost by `2`; parameter
success gains 3 and loses none; the joint success count is unchanged but moves from start 37 to
start 59. Training longer lowers the loss overall but does not guarantee that held-out success
holds monotonically on every lane. The formal comparison artifact is under the 300-epoch results in
`comparisons/20260831-210204_adam-narrow-e180_db936968/`.

#### LR 0.02 + Wide Bounds

On the 300-epoch configuration, the Adam LR changes from `0.01` to `0.02` and the transform bounds
widen from `0.5--1.5` to `0.1--2.0 x target` at the same time; the 64 physical initial values are
bitwise identical to `c00763f960b0` and still lie within `0.5--1.5 x target`.

| Metric | 300-epoch baseline | LR 0.02 + wide bounds |
| --- | ---: | ---: |
| trace success | `10/64` | `5/64` |
| parameter success | `6/64` | `14/64` |
| joint success | `1/64` | `1/64` |
| median train MSE | `55.3886` | `49.1545 mV^2` |
| median validation RMSE | `10.1949` | `8.7019 mV` |
| median test RMSE | `17.1670` | `16.4772 mV` |
| median parameter relative RMS | `0.2261` | `0.1706` |

Every final parameter is at least 10% away from the new bounds; parameter success shifts as
`4 kept/10 gained/2 lost`, and trace success as `2 kept/3 gained/8 lost`. The count of the
`3-spike` train protocol is exact for only `1/64`: the combined setting improves the continuous
error and parameter recovery but misses the target spike region more easily. Because the LR and the
bounds changed together, the result cannot be attributed to either one alone. The result digest is
`682d461a9318`.

To separate the two variables, another run uses the wide bounds but keeps `lr=0.01`, with digest
`3692d0969bca`:

| Metric | Original bounds LR 0.01 | Wide LR 0.01 | Wide LR 0.02 |
| --- | ---: | ---: | ---: |
| trace success | `10/64` | `9/64` | `5/64` |
| parameter success | `6/64` | `8/64` | `14/64` |
| joint success | `1/64` | `3/64` | `1/64` |
| validation RMSE | `10.1949` | `8.3472` | `8.7019 mV` |
| test RMSE | `17.1670` | `16.8468` | `16.4772 mV` |
| parameter relative RMS | `0.2261` | `0.1956` | `0.1706` |

Bounds alone improve the validation MSE of 42/64 and the parameter error of 45/64, and raise joint
success to 3/64; the trace total drops by only 1, but the set is `3 kept/6 gained/7 lost`. Raising
the LR to 0.02 under the same wide bounds lowers trace from 9 to 5 and joint from 3 to 1, while the
per-start median delta of the train MSE is only `+0.063 mV^2`. Wide bounds therefore help the
continuous fit and parameter recovery overall; a high LR pushes parameter recovery further but
clearly hurts spike-region stability.

#### Rprop Optimizer

With wide bounds, 300 epochs and exactly the same initial values, only Adam is replaced by
`Rprop(lr=0.01, etas=(0.5,1.2), step_sizes=(1e-6,50))`:

| Metric | Adam | Rprop |
| --- | ---: | ---: |
| trace success | `9/64` | `35/64` |
| parameter success | `8/64` | `13/64` |
| joint success | `3/64` | `11/64` |
| validation RMSE | `8.3472` | `4.4684 mV` |
| test RMSE | `16.8468` | `9.4214 mV` |
| stage time | `78.06` | `89.53 s` |

In the optimizer-only pairing, Rprop gains 28 trace successes and loses 2; the validation MSE of
49/64 and the test MSE of 47/64 improve. The validation-feasible archive further reaches validation
trace `38/64` and test trace `12/64`, against `22/64` and `7/64` for the Adam archive. Rprop is
clearly better than Adam on the current deterministic low-dimensional problem, and most of the gain
is kept at the final endpoint. The result digest is `8ae34281f8a6`.

The BrainTools Rprop wrapper applies the same LR scheduler again after `scale_by_rprop(base_lr)`, so
a nominal `lr=0.01` gives an actual initial/min step of about `1e-4/1e-8`. With a custom Optax tx
that applies the LR once and keeps initial step=`1e-4`, min step=`1e-6`:

| Metric | Wrapper Rprop | Single-scale Rprop |
| --- | ---: | ---: |
| trace / parameter / joint | `35/13/11` | `36/11/10` |
| validation / test RMSE | `4.4684/9.4214` | `4.5119/9.4196 mV` |
| archive validation / test trace | `38/12` | `39/12` |
| K median `abs(delta z)` over the last 50 epochs | `1.5e-8` | `1.2e-6` |

Lifting the `1e-8` freeze raises the physical movement of K by about 80 times, but the final and
held-out results change only slightly; about 24% of gradient sign flips still make Rprop take a
zero update on the epoch where the sign reverses. The repeated LR in the wrapper is therefore a
real bug, but it is not the only reason Rprop beats Adam or stalls late. The single-scale result
digest is `81eff6e7aa2e`.

#### Protocol-Balanced MSE

On single-scale Rprop, changing only the protocol weights to the inverse target-voltage std (with a
`5 mV` floor) raises final trace/parameter/joint from `36/11/10` to `43/15/12` and lowers the
validation/test RMSE from `4.5119/9.4196` to `3.8424/9.1728 mV`. In the pairing, trace is
`34 kept/9 gained/2 lost`, and the validation error of 47/64, the test error of 42/64 and the
parameter error of 46/64 improve. The validation archive reaches validation trace `47/64`, but its
test trace is `9/64`, below the `12/64` of the raw-loss archive; the balanced loss improves the
overall and validation fit, but strict test success does not rise with it. The result digest is
`cdc6f22c0f24`.

#### Balanced Huber Delta 5 mV

Starting from balanced MSE, only the pointwise penalty is replaced by an MSE-normalized Huber: it is
identical to MSE for `abs(error)<=5 mV`, and large errors enter the linear tail:

| Metric | Balanced MSE | Balanced Huber |
| --- | ---: | ---: |
| final trace / parameter / joint | `43/15/12` | `38/17/15` |
| validation / test RMSE | `3.8424/9.1728` | `3.3781/9.1748 mV` |
| archive validation / test trace | `47/9` | `39/13` |
| archive median test RMSE | `8.9297` | `9.0319 mV` |

Huber reduces the final and archive validation trace coverage, but raises parameter, joint and
strict archive test success. It does not lower every raw MSE: the `3-spike` median MSE drops from
`13.995` to `8.930`, but small-positive rises from `7.217` to `96.045`, so the linear tail lets some
lanes sacrifice the subthreshold trajectory to improve the spiking/parameter directions. The result
directory is:

```text
20260901-152320_balanced-huber-d5_3ee11164/
```

`checkpoint_every=10` is a historical name: it only means that a validation forward pass runs every
10 epochs; it saves no optimizer checkpoint and does not change the trajectory. New configurations
can write `validation_every=10`; the old name remains supported. Every complete run can be queried
through `plot/parameter_experiments/runs.csv`.

#### Vanilla SGD

On the balanced Huber configuration, only the optimizer is replaced by vanilla SGD `lr=1e-4`, with
no momentum, Nesterov, weight decay, clipping or scheduler. The result is much worse than
single-scale Rprop:

| Metric | Rprop | SGD |
| --- | ---: | ---: |
| final trace / parameter / joint | `38/17/15` | `0/2/0` |
| validation / test RMSE | `3.3781/9.1748` | `15.1178/21.5266 mV` |
| archive validation / test trace | `39/13` | `0/3` |
| median final objective | `6.5554` | `21.1214 mV^2` |

SGD neither diverges nor sticks to the bounds, and the median gradient norm drops from `22.7` to
`5.38`, but within 300 epochs it forms no correct validation spike signature. It lowers the Huber
objective steadily, but moves slowly along the wrong continuous basin; Rprop's exponential growth of
the step while the gradient sign holds is more effective at crossing into the target spike region.
The result directory is:

```text
20260902-143337_sgd-lr1e-4-balanced-huber_94649c2f/
```

#### Momentum and Nesterov SGD

With the same `lr=1e-4`, `momentum=0.9` and Nesterov are tested in parallel. Both clearly speed up
the decrease of the Huber objective, but neither restores final trace success:

| Metric | Vanilla | Momentum | Nesterov | Rprop |
| --- | ---: | ---: | ---: | ---: |
| final trace | `0/64` | `0/64` | `0/64` | `38/64` |
| parameter | `2/64` | `8/64` | `11/64` | `17/64` |
| joint | `0/64` | `0/64` | `0/64` | `15/64` |
| validation RMSE | `15.12` | `10.78` | `12.54` | `3.38 mV` |
| test RMSE | `21.53` | `18.87` | `19.94` | `9.17 mV` |
| final Huber objective | `21.12` | `10.69` | `11.42` | `6.56` |
| archive validation/test trace | `0/3` | `6/2` | `5/6` | `39/13` |

Momentum/Nesterov fix the slow movement of vanilla SGD, but 5/7 of the lanes end close to the
transform edge and still fall into the wrong spike basin; Nesterov is not consistently better than
plain momentum. This further indicates that Rprop's per-coordinate sign step suits the current task
better than a uniform velocity accumulation. The two parallel runs are:

```text
20260902-145308_momentum09-lr1e-4-balanced-huber_1404e6e8/
20260902-145308_nesterov09-lr1e-4-balanced-huber_2025b7f6/
```

## Historical sources

The repository history once contained the following experiment lines, but the current worktree no
longer has the corresponding `.py` source files:

- `conductance_learning_core.py`;
- `heterogeneous_nine_parameter_training.py`;
- `heterogeneous_protocol_dataset.py`;
- `heterogeneous_nine_parameter_composite_ablation.py`;
- `parameter_learning_demo.py`;
- `sine_expanded_batch_comparison.py`.

A local machine may still hold `.pyc` files of the same names; these are only the Python cache:

- they are not a code dependency;
- they are not under version control;
- they must not be decompiled and restored as the current implementation;
- tests or maintenance should remove them directly.

When design documents refer to these names, they mean the experimental evidence in a historical
snapshot, not that the files still exist on the current branch.

## Related documents

- [Experimental Optimization Work](../README.md): navigation for the exact forward sensitivity, RTRL/BPTT, correctness and scaling experiments.

## Minimal usage

The current training examples use this data flow:

```text
rollout
  -> voltage_mse_objective
  -> evaluate_voltage_protocols
  -> capture_state / capture_update
  -> optimizer.update
  -> finalize_history
  -> extract_best_archives / summarize_history
```

A new experiment should replace only one of these roles where possible, for example the objective
or the evaluator, rather than copy the whole history/archive implementation.

## Verification

```bash
pytest -q examples/optim/parameter_fitting/diagnostics_test.py
pytest -q examples/optim/parameter_learning/trainable_hh_multistart_test.py
```

If this helper later gains several stable callers and a clear public contract, decide then whether
to move it into a proper module; this README does not promise that its current location or type
names will stay stable.
