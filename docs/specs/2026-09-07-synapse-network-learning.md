# Synapse and Network Learning

## Intent

Extend the existing Channel/Ion parameter sources to signature-declared
Synapse parameters, Connection weight, and event-detector threshold. Aggregate
all supported parameters in Network without copying optimizer roots. Delay,
plasticity rules, plastic weight initializers, and new RTRL algorithms remain
outside this change. Preserve unrelated workspace edits.

## Implementation

- Replace handwritten Synapse parameter schemas with explicit constructors and
  generated metadata. Preserve default get/set, units, broadcasting, model
  validation, states, and event input contracts.
- Reuse parameter/scale/parameterized sources, grouping, transforms, root
  sharing, atomic registration, and persistent differentiable runtime values.
  Identify synapses, contacts, and detector endpoints by their logical IDs,
  not merely by their CV. Reject trainable delay explicitly.
- Unify rising detection as last < threshold <= next and falling detection as
  last > threshold >= next. Use the Cell surrogate by default, allow detector
  overrides, and retain floating event payloads through delivery. Document
  the different NEURON APCount and fixed-step NetCon equality boundaries.
- Expose Network.trainables, prepare_run(), and update(); share the existing
  simulation ordering with run(), keeping host result formatting outside AD.
  Reset dynamic state and queues without resetting trainable roots.
- First validate a single autaptic Cell, then two-Cell Network execution.
  Compare existing full-state RTRL against BPTT for voltage and spike losses,
  including queue sensitivities. Parameters remain fixed within a rollout;
  no independent-Cell sensitivity truncation or stepwise optimizer is added.

## Verification

Write regression tests before fixes. Cover equality boundaries, directions,
surrogates, signatures, units, selection, colocated synapses, shared roots,
repeated compiled execution, reset, errors, zero gradients, fixed delays,
both delivery backends, and existing Channel/Ion/Network behavior. Use finite
differences only for ordinary differentiable paths; compare JVP/VJP and
RTRL/BPTT for surrogate paths. Target meaningful changed-line coverage >90%.

Create examples/multi_compartment/synapse_learning.ipynb with classification,
gradient/error, and fit tables. Demonstrate short single-sample tau, weight,
and threshold fits, and autaptic voltage/spike losses. Tau and weight fits
must reduce MSE tenfold; threshold must improve loss without claiming unique
parameter recovery. Keep executable RTRL checks in the existing experimental
gradient-correctness area. Record actual results and unavailable environments.

## Results

Implemented on branch `reduction`. The pre-existing edits to channel_learning.ipynb
and ion_learning.ipynb were preserved, not rewritten by this change.

- JAX 0.8.0 / Python 3.11 / CPU: the related Channel, Ion, runtime, trainable,
  multi-compartment, Network and Synapse regression run passed 1509 tests with
  2 skips. Additional focused tests cover Exp2Syn finite differences and qualified
  root-name collisions.
- Focused coverage run: 381 passed, 2 skipped. Changed executable production
  lines, including new production modules: 364/378 covered (96.3%). This is
  changed-line coverage, not whole-repository coverage or a correctness proof.
- Both JAX 0.8.0 and 0.10.1 passed autaptic voltage/spike BPTT versus full-RTRL
  checks, zero and 0.1 ms delays, and two-Cell sensitivity/carry-shape checks.
  The notebook's float64 comparisons had maximum absolute gradient discrepancy
  at most 5.24e-10; spike-loss discrepancies were at most 3.33e-15.
- JAX 0.10.1 focused Synapse, Network, point-target and base-class regressions:
  173 passed, 2 skipped (plus 8 passing subtests).
- The notebook executed end to end on JAX 0.8.0. Its tau MSE went from
  0.0564968 to 2.48711e-7, weight from 3.11666 to 9.70201e-7, and threshold from
  2.5993e-5 to zero on the discrete event grid. Each target had one spike.
  All three fit acceptance checks also passed on JAX 0.10.1.
- The JAX 0.10.1 combined run exposed existing Ion float32/float64 test-order
  failures. Running `ion/_base_test.py`, `_compute/ions_test.py`, then the Ion
  cases in `trainable/_manager_test.py` reproduced five such failures in an
  unmodified HEAD archive. The manager's 17 Ion cases passed in a fresh isolated
  process. The unmodified HEAD full related run also passed 1488 tests with
  2 skips: this is execution-context-sensitive, not an unconditional failure.
  The failing modified combined run was coverage-instrumented; instrumenting
  alone has not been established as the cause. This pre-existing precision/cache
  issue was not changed here, and the full modified JAX 0.10.1 suite is not
  claimed to be uniformly green.
- Ruff and `git diff --check` passed. GPU execution was not tested.

Regression tests prevent partial physical writes after a later unit error,
premature validation of inherited constructor fields, stale compiled weights,
loss of threshold ownership, duplicate equality events, and loss of unit-based
inspection on runtime parameter buffers. Zero gradients remain valid outcomes.

### Reproduction

```bash
python -m pytest -q braincell/channel braincell/ion braincell/_compute \
  braincell/trainable braincell/_multi_compartment braincell/network braincell/synapse \
  braincell/_base_channel_test.py braincell/_base_ion_test.py braincell/_base_neuron_test.py
python -m pytest -q examples/experimental/optim_gradient_correctness/autapse_test.py
python -m pytest -q examples/multi_compartment/synapse_learning_test.py
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 examples/multi_compartment/synapse_learning.ipynb
```

Choose the intended Python environment explicitly when comparing JAX versions.
Test simultaneous dtype-switching suites in a separate process when investigating
the known JAX 0.10.1 Ion failure; do not reinterpret it as a synapse gradient error.
