# HH Multistart Training Example

## Goal

Add one executable multi-compartment example that demonstrates the complete
Cell-local trainable-parameter path on the smallest useful Hodgkin-Huxley
model. The example fits the maximal conductances of `IL`, `Na_HH1952`, and
`K_HH1952` to one synthetic voltage trace with MSE and BrainTools Adam.

This is a parameter-recovery tutorial, not a claim that three conductances are
identifiable from one protocol or that fitting one trace measures
generalization.

## Simulation Protocol

- Use one 20 um soma branch and `CVPerBranch`, giving one CV.
- Use the staggered solver with an initial voltage of -65 mV.
- Paint fixed sodium and potassium ions plus `IL`, `Na_HH1952`, and
  `K_HH1952`.
- Inject 0.05 nA from 5 ms through 45 ms.
- Simulate 50 ms with a 0.025 ms time step, producing 2000 voltage samples.
- Generate the target from the channel defaults multiplied by known
  leak/Na/K factors `[1.2, 0.85, 1.25]`.
- Reject the target protocol at runtime if it does not produce at least one
  upward 0 mV crossing.

## Trainable Parameters And Parallelism

The candidate Cell has `pop_size=(32,)`. Each population member is one
independent initial point, while all members share the morphology, mechanism
layouts, current protocol, and compiled simulation.

For each of the three channels:

- draw 32 independent physical factors uniformly from `[0.5, 1.5]` with
  `brainstate.random.RandomState(123)`;
- construct an `nn.Param` with `SigmoidT(0.1, 2.0)`;
- bind it to `g_max` with `trainable.scale(..., group_by="population")`.

The model therefore has 96 scalar degrees of freedom. A candidate runtime
conductance is always its channel's frozen default baseline multiplied by its
factor. The bounds constrain the factor, not the conductance density directly.

The simulated voltage has shape `(2000, 32)`. The one target trace broadcasts
over the population axis. The differentiable scalar objective is the sum of
the 32 per-candidate time MSE values, so each candidate receives the same
gradient it would receive in an independent optimization. Reported curves use
the mean and minimum candidate MSE.

## Training And Output

- Convert voltage quantities explicitly to mV at the loss boundary.
- Use BrainTools Adam with learning rate 0.02 for 100 updates.
- Use `brainstate.transform.for_loop` for both the 2000 simulation steps and
  the optimizer updates; do not drive either loop with Python iteration.
- Record aligned histories for per-candidate MSE and physical factors,
  including the initial and final values.
- Select the candidate with the lowest final MSE.
- Plot target versus best fitted voltage, all candidate loss curves with their
  mean and minimum, and the best candidate's three factor trajectories against
  the known target factors.

The example remains device-independent. GPU selection occurs before Python
starts, for example:

```bash
CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  conda run -n braincell_311 \
  python examples/multi_compartment/trainable_hh_multistart.py
```

## Acceptance

- The default target contains at least one spike and all arrays are finite.
- The three trainable roots have shape `(num_starts,)` and remain within the
  physical factor bounds.
- A reduced smoke configuration completes an Adam update and lowers the mean
  or best MSE.
- The default 32-start, 100-update workflow runs on CPU and on physical GPU 5.
- The change adds no BrainCell Trainer, Dataset, loss, or optimizer API.
