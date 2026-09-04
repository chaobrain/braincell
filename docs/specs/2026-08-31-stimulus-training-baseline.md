# 1CV Bounded Direct-Parameter Training Baseline

## Goal

Create an experiment-local, Python-composed parameter-fitting runner whose true
zero baseline is one HH compartment with three bounded direct conductance
parameters. Run all 64 initial points in one device kernel, keep test data final-only,
and preserve stage boundaries for later gradient-free/gradient hybrids.

## Fixed Baseline

- One 25 um by 12.5 um soma CV with classical `IL`, `Na_HH1952`, and `K_HH1952`.
- Targets `(0.3, 120, 36) mS/cm^2`; bounds are target-relative `(0.5, 1.5)`.
- `braincell.trainable.parameter()` maps bounded sigmoid optimizer coordinates
  directly to physical `g_max`; no frozen baseline scale is trained.
- Five train, two validation, and one final-only test protocol, all 100 ms DC Steps.
- Raw voltage MSE, exact RTRL, Adam at `0.01`, 64 starts in one kernel, and 180 updates.
- Initialization seed 0 draws physical values uniformly inside the bounds and stores
  both physical and inverse-transformed optimizer coordinates.

## Composition Contract

Python presets compose immutable model, dataset, loss, initialization, optimizer,
and stage definitions. An optimization pipeline passes a physical `CandidateSet`
between stages; gradient stages convert through bounded `z`, while derivative-free
stages use normalized bounded coordinates. A non-gradient parameter change resets
stale optimizer moments before a later gradient stage.

## Artifacts

Each result copies the input Python config and writes resolved JSON metadata,
parameter-space and dataset manifests, initial candidates, per-stage histories,
long-form metrics, per-start endpoint rows, a Chinese report, timing/memory data,
and figures. Train is recorded every update, validation every ten updates, and test
only at the final state.

## Verification

Test parameter-coordinate round trips with units, direct runtime materialization,
Step-only split isolation, RTRL/BPTT equality, one-kernel 64-lane shape, optimizer
state isolation, stage handoff and moment reset, result/resume contracts, and a CPU
smoke. Before the formal run, compile the exact 64-start GPU kernel and record XLA
memory analysis without silently falling back to chunks.
