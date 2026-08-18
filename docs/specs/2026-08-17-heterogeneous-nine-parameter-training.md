# Heterogeneous Nine-Parameter Training

## Goal

Fit independent Leak, Na, and K maximum conductances in the soma, dend_a,
and dend_b compartments against the 144-protocol heterogeneous dataset.  The
experiment is example-local and does not change BrainCell's public API.

## Data flow

The saved dataset contains 108 training, 18 validation, and 18 test protocols.
One training epoch visits every training protocol exactly once in six batches
of 18.  Each batch contains, for every injection site, three DC, two paired,
and one sine protocol.  The strata are independently permuted each epoch with
`brainstate.random.RandomState(20260817)`, while every multistart run receives
the same schedule.

The model is a population of 18 identical three-compartment cells.  Saved
three-site current waveforms are converted from nA to point current density and
read by a state-backed current input.  This permits mixed protocol families in
one compiled rollout without adding a new mechanism or API.

## Parameters and starts

The physical parameter order is:

1. soma Leak, Na, K
2. dend_a Leak, Na, K
3. dend_b Leak, Na, K

There are nine independent sigmoid-bounded parameters.  The first experiment
uses eight target-scaled starts.  Leak, Na, and K are each multiplied by 0.5
or 1.5 across all three regions, giving a 2 by 2 by 2 grid in channel-family
scale rather than an impractical 2 to the ninth grid.  Starts are trained in
four compiled chunks of two to bound memory.

## Objective and optimizer

The objective is voltage-only Huber loss with delta 2 mV over all three probe
locations.  Fixed windows from 1 ms before through 3 ms after each target soma
upward zero crossing receive weight 0.1; all other samples receive weight 1.
Each protocol is divided by the corresponding canonical-initialization loss,
floored at 0.1, before batch averaging.  Hard spike counts are diagnostics and
do not enter the differentiable objective.

Adam uses learning rate 0.01 and global gradient clipping at 1.0, without
weight decay or a schedule.  The maximum is 30 epochs (180 updates).  The best
validation checkpoint is retained independently for each start.  After a
minimum of 10 epochs, a start is inactive after six validation checks without
a relative improvement of at least 0.1 percent.  Test data are evaluated once
from the retained checkpoint.

## Evaluation contract

Parameter recovery uses the scale-free relative RMS distance

```text
sqrt(mean(((parameter - target) / target) ** 2))
```

over all nine conductances. A value of `0.1` represents a typical relative
error of ten percent without allowing the sodium scale to dominate Leak and K.
Adjacent checkpoint movement uses the same normalization.

Every saved checkpoint is replayed on the validation split. The recorded
metrics are three-probe voltage RMSE, exact soma spike-count fraction, and the
fraction whose counts match and whose ordered spike times all differ by at
most `0.5 ms`. Test metrics are computed only for the saved validation-best
checkpoint; test data never select a start or checkpoint. Equal zero-spike
traces have timing error zero, while count mismatches have undefined timing
error.

The numeric archive retains per-start parameter distances and movements,
per-protocol test voltage RMSE, signed nine-parameter endpoint errors, hard
spike-count errors, and maximum spike-timing errors. These arrays are the
source of truth for plots and can be regenerated in diagnostics-only mode
without rerunning optimization.

## High-dimensional diagnostics

The primary landscape is a 15 by 15 validation-loss plane in
`log(parameter / target)` coordinates.  Its first axis is the direction from
the best endpoint to the target.  Its second axis is the leading trajectory
PCA direction after orthogonalization against the first.  Training paths,
target, and endpoint are overlaid.  Aggregate validation soma spike-count
mismatch remains in the numeric archive but is not drawn on the continuous-loss
landscape; dedicated spike diagnostics report that discrete metric.

Nine 31-point one-dimensional profiles supplement the plane.  Each parameter
is swept around both the target anchor and the best endpoint while the other
eight parameters remain fixed.  The plots report objective and aggregate
spike-count mismatch.  These views diagnose curvature and local basins without
claiming to visualize all nine dimensions.

The standard figures additionally include:

- eight target-versus-best test trace atlases, with no initial trace overlay;
- parameter-distance and checkpoint-movement histories;
- validation voltage, spike-count, and spike-timing trajectories;
- `8 x 18` protocol voltage and spike-timing heatmaps;
- an `8 x 9` signed parameter-error heatmap; and
- an endpoint Pareto view of parameter error, voltage RMSE, count success, and
  timing success.

The landscape displays only the continuous validation objective. The archived
hard spike-mismatch grid is retained, but no white spike-count boundary is
drawn over the filled loss contours.

## Prior experimental evidence

The retired shared-three-parameter experiments established why the current
dataset and diagnostics are necessary:

- A single-spike target produced trace success for `6/9` and `12/16` starts,
  but parameter recovery for only `2/9` and `4/16`. This exposed a broad
  sodium-potassium compensation valley and showed that trace fit alone does
  not identify conductances.
- Four fixed protocols improved median three-probe RMSE from `12.8944 mV` to
  `7.8201 mV` and median mean parameter error from `0.2687` to `0.1562`, but
  only `3/8` starts met the trace criterion. More stimuli improved basin
  robustness without removing local minima.
- Near spike boundaries, changing scalar versus batched evaluation produced
  small floating-point differences that could change the final integer spike
  count. Boundary conclusions therefore require hard spike diagnostics and
  precision or batch-shape controls.

The retired scripts are not part of the active example set. Their generated
plots remain under `examples/experimental/parameter_learning/plot/` as
historical evidence.

## Artifacts

The default output directory is
`examples/experimental/parameter_learning/plot/heterogeneous_nine_parameter_training/`.
It contains the configuration and minibatch schedule, numeric training arrays,
loss curves, trace-fit atlases, parameter and protocol error figures, hard
spike diagnostics, the validation plane, one-dimensional profiles, and a JSON
summary of endpoint metrics and timing.

## Acceptance checks

- The schedule is deterministic, balanced per batch, and covers each training
  protocol exactly once per epoch.
- Current playback reproduces a saved target trace at the target parameters.
- Initial values have shape 8 by 9 and remain strictly inside transforms.
- Mask construction handles zero-spike and boundary-spike protocols.
- A reduced one-epoch smoke run has finite loss, gradients, parameters, and
  voltage traces.
- Saved best checkpoints, not final optimizer states, drive test metrics and
  diagnostic plots.
- All eight initial relative distances equal `0.5`, and the target distance is
  zero.
- Per-protocol voltage RMSE averages over time and all three probes; signed
  parameter errors preserve over- versus under-estimation.
- Spike timing handles equal silent traces, equal-count traces, and count
  mismatches without conflating the cases.
- Landscape construction retains filled loss contours and projected training
  paths without drawing a hard spike-count boundary.
- Diagnostics-only execution updates archives and figures without training or
  test-set checkpoint selection.
