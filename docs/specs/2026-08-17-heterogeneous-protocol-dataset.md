# Heterogeneous Protocol Dataset

## Goal

Build a three-compartment Hodgkin-Huxley target whose leak, sodium, and
potassium maximum-conductance densities are independent in the soma,
``dend_a``, and ``dend_b``.  Generate a deterministic, population-batched
dataset that can later be used to fit all nine conductances without changing
the training API during this exploratory phase.

Training is deliberately out of scope for this change.  The generated traces,
protocol catalog, and coverage plots are the review boundary before choosing a
nine-parameter training setup.

## Target Model

The morphology uses one control volume per branch:

| Compartment | Length | Proximal radius | Distal radius |
| --- | ---: | ---: | ---: |
| soma | 25 um | 12.5 um | 12.5 um |
| dend_a | 100 um | 2.0 um | 1.0 um |
| dend_b | 150 um | 1.5 um | 0.75 um |

Each compartment owns independent leak, sodium, and potassium channel
instances.  The fixed target densities are:

| Compartment | Leak | Na | K |
| --- | ---: | ---: | ---: |
| soma | 0.60 | 120.0 | 36.0 |
| dend_a | 0.48 | 96.0 | 28.8 |
| dend_b | 0.42 | 84.0 | 25.2 |

Conductance-density values are expressed in ``mS/cm^2``.  Sodium and
potassium reversal potentials remain 50 mV and -77 mV; the leak reversal is
-54.387 mV.  A later fitted model will expose the same nine independent scalar
parameters with the existing sigmoid transforms and bounds repeated per
compartment.

## Protocol Catalog

Every simulation lasts 100 ms at ``dt=0.025 ms``.  Current is zero outside
20--80 ms.  All three compartments are used as injection sites and all three
voltages are recorded.  The catalog has 48 protocols per site and 144 total:

| Family | Per site | Total | Definition |
| --- | ---: | ---: | --- |
| DC | 24 | 72 | 6 negative and 18 positive levels |
| paired | 16 | 48 | 8 negative-to-positive and 8 positive-to-negative |
| sine | 8 | 24 | 10/40 Hz by four amplitudes |

Negative DC amplitudes are calibrated per site against soma minima of -70,
-80, -90, -100, -110, and -120 mV.  Positive DC amplitudes cover evoked soma
spike counts zero through five: the largest continuous amplitude interval for
each count contributes its 25%, 50%, and 75% points.

Paired protocols use 20--40 ms and 60--80 ms pulses separated by 20 ms at
zero current.  Their negative levels are the calibrated -80 and -100 mV
levels.  Their positive levels are the midpoints of the one-, two-, four-, and
five-spike DC intervals.  Both temporal orders are present.

Sine protocols use 10 and 40 Hz and the midpoint amplitudes of the zero-,
one-, three-, and five-spike DC intervals.  The offset is zero.  Phase is
centered over the finite 60 ms window so its integral is zero rather than
silently introducing a DC component.

Spike counts used for calibration and acceptance are zero-millivolt upward
crossings in 20--100 ms.  This excludes the target model's initialization
transient while including post-stimulus rebound spikes.  Every protocol must
have at most five evoked spikes.  A paired or sine candidate that exceeds the
limit repeatedly scales its nonnegative amplitude by 0.9 until it passes; the
effective amplitude is stored in the catalog.

The split is deterministic and stratified within every site/family group:
108 training, 18 validation, and 18 test protocols.  No random generator is
needed to reproduce the catalog.

## Population Execution

DC, paired, and sine protocols run as population batches of 72, 48, and 24.
One clamp is placed at each possible injection site; rows belonging to other
sites receive zero amplitude.  Model parameters are shared across the
population axis while each row has a different stimulus.  BrainCell owns the
compiled time loop; the example must not drive simulation steps with a Python
loop.

Before full generation, scalar and population rollouts of the same protocol
must agree.  Full execution starts with small batches and may chunk a family
at 128 rows or less if required by the local JAX runtime.  A crash is an error
to isolate, not a condition to hide by silently dropping protocols.

## SineClamp Contract

``SineClamp`` continues to accept scalar inputs.  In addition,
``amplitude``, ``frequency``, ``phase``, ``offset``, ``delay``, and
``duration`` may be arrays broadcastable to ``Cell.pop_size``.  Quantity
fields retain mandatory units; frequency and duration are positive
elementwise; numeric phase values are finite.  Runtime buffers preserve the
population-leading axes and the point-placement axis.

## Artifacts

The generator writes:

- ``dataset.npz`` with time, three-site current, three-probe voltage, spike
  counts, spike mask, split indices, and the target parameter matrix;
- ``protocol_catalog.csv`` and ``summary.json`` with complete stimulus and
  model metadata;
- nine trace atlases, one for every injection-site/family pair, plus coverage,
  spike-distribution, and warm-throughput figures;
- ``performance.json`` containing compile and warm execution timings.

The full numerical traces cover 0--100 ms.  Initialization-transient and
evoked spike counts are stored separately.  Loss-gradient, loss-landscape,
and parameter-path plots are not produced because no training occurs here.

## Acceptance

- The target contains three control volumes and nine distinct conductance
  channel instances.
- Scalar and batched SineClamp behavior is numerically consistent.
- Dataset arrays have 144 protocol rows, 4000 time samples, and three current
  and voltage columns.
- Split counts are exactly 108/18/18 and each site/family count matches the
  catalog above.
- All arrays are finite and all evoked soma spike counts are at most five.
- Generated current waveforms are zero outside 20--80 ms and agree with their
  catalog metadata.
