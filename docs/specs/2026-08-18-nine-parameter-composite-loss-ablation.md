# Nine-Parameter Composite Loss Ablation

## Goal

Compare three differentiable loss compositions under the established
heterogeneous nine-parameter training protocol.  Keep the dataset, eight
initial points, minibatch schedule, optimizer, epoch budget, checkpoint rule,
quality diagnostics, and loss-landscape display fixed.  Save each experiment
to a separate artifact directory without changing the BrainCell public API.

## Components

The component order and baseline Composite weights are:

| Component | Scope | Weight |
| --- | --- | ---: |
| voltage | all three probes | 1.00 |
| derivative | all three probes | 0.10 |
| multiscale | all three probes | 0.25 |
| event | soma | 0.75 |
| count | soma | 0.40 |
| peak | soma | 2.00 |

Voltage is the existing 2 mV Huber objective with target-defined spike windows
weighted by 0.1.  Derivative uses 0.5 mV Huber and the adjacent-sample minimum
of that mask.  Multiscale averages non-overlapping 20-sample blocks before the
2 mV Huber penalty.  Event is the squared error between 2 ms exponentially
filtered sigmoid-crossing traces.  Count is the squared difference between
the sums of sigmoid crossings; it is differentiable and is distinct from the
hard zero-crossing count used for endpoint diagnostics.  Peak is the squared
difference between smooth soma maxima over 20--100 ms.

## Configurations

The three component-weight vectors are:

```text
voltage_count             [1.00, 0.00, 0.00, 0.00, 0.40, 0.00]
without_count_composite   [1.00, 0.10, 0.25, 0.75, 0.00, 2.00]
full_composite            [1.00, 0.10, 0.25, 0.75, 0.40, 2.00]
```

For each protocol and component, the raw loss is divided by its value at the
canonical initial conductances.  Floors in component order are
`[0.1, 0.001, 0.1, 0.0001, 0.001, 1.0]`.  The per-protocol objective is the
weighted sum of normalized components divided by the sum of active weights.

## Fixed Training Contract

- Use the existing 108/18/18 train, validation, and test split.
- Use the same deterministic balanced schedule for every configuration.
- Train the same eight target-scaled starts and nine independent conductances.
- Use Adam with learning rate 0.01, gradient clipping at 1.0, and 30 epochs.
- Retain each start's minimum-validation-loss checkpoint without test-set
  selection.
- Draw the same parameter, voltage, spike, trace, profile, Pareto, and
  trajectory-informed loss figures.  The landscape has no hard-spike contour.

## Artifacts

Write complete, independent archives to:

```text
examples/experimental/parameter_learning/plot/heterogeneous_nine_parameter_voltage_count/
examples/experimental/parameter_learning/plot/heterogeneous_nine_parameter_without_count_composite/
examples/experimental/parameter_learning/plot/heterogeneous_nine_parameter_full_composite/
```

Each summary records component names, weights, normalizer floors, peak window,
training backend, and the existing endpoint quality metrics.  Each NPZ stores
the six-column normalizer matrix in addition to the standard training arrays.

## Acceptance Checks

- Configuration lookup returns exactly the three approved vectors and rejects
  unknown names and all-zero weights.
- Identical prediction and target traces produce zero for all six components.
- A dendrite-only error affects the three all-probe components but not the
  soma-only event, count, or peak components.
- The smooth count component has a finite, nonzero gradient near threshold.
- Canonical normalizers have shape `(144, 6)` and are finite and no smaller
  than their component floors.
- A short CPU smoke run produces finite objectives, gradients, parameters, and
  traces before the three full runs begin.
- All three full archives contain eight starts, 30 epochs, 18 test protocols,
  finite continuous metrics, and all requested figures.

## Completed Runs

All three CPU runs completed with identical starts, minibatch schedules, and
component normalizers.  Training wall times were 591.9 seconds for
`voltage_count`, 587.1 seconds for `without_count_composite`, and 577.5 seconds
for `full_composite`; landscape, validation replay, and figure generation are
outside those recorded training timers.

The best endpoint observed separately under each hard or continuous metric was:

| Configuration | Best parameter RMS | Best test RMSE | Best count | Best timing |
| --- | ---: | ---: | ---: | ---: |
| voltage_count | 15.22% | 9.820 mV | 17/18 | 8/18 |
| without_count_composite | 13.92% | 6.485 mV | 18/18 | 11/18 |
| full_composite | 13.77% | 8.495 mV | 17/18 | 11/18 |

These columns may select different starts and therefore describe attainable
endpoints, not one jointly best model.  The archived per-start metrics and
Pareto figures remain authoritative for joint selection.  All 18 required
figures per configuration were generated.  Landscape figures retain the
continuous objective only and do not overlay a hard spike-count contour.
