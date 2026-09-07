# Bidirectional Population Learning

## Plan

Validate the existing full-state RTRL on a Network containing two independent
one-CV HH Cell populations of sizes two and three. Both populations own trainable
channel conductance/voltage shift, ion reversal, synapse kinetics/reversal and
detector threshold. Both directions have six independently addressable contacts
and trainable weights. A uses ExpSyn and B uses Exp2Syn.

Use asymmetric parameters and repeated, offset stimuli. Require real emissions
and arrivals in both directions and subsequent spikes after feedback. Compare
all scalar/vector root gradients for A-only, B-only, joint voltage and spike
losses. Diagnose event paths by stopping only event derivatives while preserving
the complete forward trajectory. Exercise zero, positive and heterogeneous fixed
delays; scatter and brainevent; shared roots; compiled reset and root replacement;
prefix gradients and full-state carry sizes. Delay is not trainable.

Train two replicas with BPTT and full RTRL from identical perturbed roots against
one synthetic spiking target. For fitting, share parameters within each
population and each connection direction, but not between populations. Use
Adam 0.01, at most 200 epochs, parameters fixed within each rollout, positive
bounded factors and nonoverlapping Exp2Syn time-constant ranges. Require voltage
MSE to decrease at least tenfold, not unique parameter recovery or identical
optimizer trajectories. Treat gradient validation and optimization separately.

Keep experiments and co-located tests under the existing gradient-correctness
directory. Extend synapse_learning.ipynb with the executed topology, gradient,
activity and fit results. Validate CPU JAX 0.8.0 and 0.10.1 in separate processes.
Existing workspace changes are preserved; no commit is requested. This is not
exhaustive coverage of arbitrary mechanisms, morphologies, or online parameter
updates inside a rollout.

## Implementation and Validation

The experiment uses 20 ms at dt = 0.025 ms. Population A has two members and B
has three; all five emit twice and receive nonzero synaptic conductance before
subsequent firing. A later, stronger clamp overcomes HH afterhyperpolarization;
the initial weaker repeated pulse was insufficient to exercise recurrent firing.
Activity assertions prevent a numerically passing but inactive-feedback fixture.

Gradient comparisons include every coordinate of 15 named vector roots (45
scalars), rather than reducing each vector to a single summary before checking:

| Loss | Fixed delays | Delivery |
| --- | --- | --- |
| Joint voltage | Heterogeneous, including zero | scatter |
| A-only voltage | Homogeneous positive | scatter |
| B-only voltage | Heterogeneous, including zero | brainevent |
| Joint spikes | Zero | brainevent |

The float64 acceptance bound is `atol=1e-8, rtol=1e-7`. In the executed notebook's
joint-voltage case, the largest absolute gradient difference is approximately
3.35e-10. This is floating-point agreement on the chosen surrogate graph, not
bitwise equality or a finite-difference derivative of hard event timing.

Additional tests check:

- Per-member model roots and six independently addressable weights per direction.
- A-only loss reaching B.gmax/B.threshold, and the converse.
- Stopping only event derivatives: identical voltages, spikes and conductances,
  but exactly zero cross-population gmax/threshold gradients.
- A shared gmax root receiving the sum of independent A and B contributions.
- Prefix gradient agreement around emissions/arrivals and at the final step.
- Cross-population voltage sensitivity and nonzero delayed-queue sensitivity.
- Reset with the same compiled function, root updates, and exact restoration.
- Full-state/sensitivity carry shapes independent of rollout duration.
- BPTT and full RTRL fitting both populations and both incoming weight groups.

The executed notebook uses 100 Adam updates of all 15 grouped roots. Both
methods reduce voltage MSE from 25.51068369 to approximately 0.10422649, a ratio
of 0.0040856. All grouped roots change. This demonstrates joint fitting, not
unique recovery of the generating parameters. Automated fitting allows 200
updates and requires at least a tenfold MSE reduction.

### Sparse Derivative Regression

Both installed JAX environments exposed the same brainevent failure:
`coomv_p_call() got an unexpected keyword argument 'weight_info'` when batching
weight JVP directions. Previous backward-only sparse-delivery checks did not
exercise that transformation. A co-located delivery test first reproduced it.
The adapter now retains the requested forward kernel and defines its exact
bilinear JVP with scatter. Tests compare batched weight/event JVPs, reverse
gradients, homogeneous/heterogeneous weights, units and boolean event inputs
against the scatter implementation. No dependency files or global derivative
registrations are modified.

### Completed Checks

| Check | Result |
| --- | --- |
| CPU JAX 0.8.0: delivery and bidirectional tests | 11 passed |
| CPU JAX 0.10.1: same tests | 11 passed |
| CPU JAX 0.8.0: network, network parameter manager, autapse regressions | 135 passed, 1 skipped |
| Selected coverage run with Python tracer, JAX 0.10.1 | 3 passed |
| Executed synapse_learning notebook | All six code cells completed without errors |
| Targeted Ruff checks and git diff whitespace check | Passed |

The new experiment has 139/146 executable lines covered (95.2%); the seven
uncovered lines are its command-line reporting entry point. The new bilinear
delivery adapter has no uncovered executable statements. These figures are
scoped to the new implementation, not a claim of whole-repository coverage.

The C-tracer coverage attempt on JAX 0.8.0 terminated with a native segmentation
fault during AD tracing. Its results are not counted. Ordinary JAX 0.8.0 tests
and Python-tracer coverage on JAX 0.10.1 subsequently passed; the native crash's
cause has not been established and no unrelated runtime changes were made.

Reproduce the main checks with the appropriate Python environment:

```bash
python -m pytest -q braincell/network/delivery_test.py examples/experimental/optim_gradient_correctness/bidirectional_test.py
python -m pytest -q braincell/network/ braincell/trainable/_network_test.py examples/experimental/optim_gradient_correctness/autapse_test.py
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 examples/multi_compartment/synapse_learning.ipynb
python -m coverage run --timid --source=braincell/network,examples/experimental/optim_gradient_correctness -m pytest -q braincell/network/delivery_test.py examples/experimental/optim_gradient_correctness/bidirectional_test.py -k 'batched or both_gradient_methods or shared_root'
```

### Scope

The four combinations above cover the intended structural boundaries, not the
Cartesian product of every loss/backend/delay option. Remaining extensions
include multi-CV cells, nonlinear/custom synapses, larger sparse topologies,
other surrogate functions, GPU execution and optimizer updates within a rollout.
Delay remains fixed and is not a trainable target.
