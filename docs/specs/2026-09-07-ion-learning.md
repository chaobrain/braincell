# Ion Learning

## Intent

Extend the Channel signature-based parameter framework to Ion without a
scientific trainability whitelist. Preserve units, default reads and writes,
regional ownership, grouping, shared roots, constructor conversions, and the
existing parameter/scale/parameterized API. Do not expose internal constants,
train morphology or Cell.V_init, or change reaction equations.

## Implementation

- Share signature metadata and compact parameter allocation with Channel.
  Resolve Ion's None defaults from model defaults. Keep configuration separate
  from numeric storage and preserve explicit species-initializer overrides.
- Use differentiable array operations and persistent runtime parameter states
  for Ion broadcasting, regional merging, synchronization, and repeated JIT.
- Keep initial parameters distinct from evolving species states. Reset reads
  current trainable initial values inside the differentiated function.
- Reevaluate derived default initializers at reset, including buffer equilibria
  and caiBase/caliBase fallbacks; explicit initial values remain independent.
- Preserve fixed, reset-time Nernst, and dynamic Nernst semantics. Remove stale
  trainable-dependent shell volume caches without changing geometry formulas.

## Verification

Write failing regressions before fixes. Cover defaults, units, inherited
signatures, regional/population ownership, shared roots, dtype conversions,
repeated compiled gradients, reset, finite differences, zero gradients, and
natural errors. Regress Channel, Ion, compute, trainable, Cell, Network, and
Synapse behavior on JAX >= 0.8.0. Measure changed executable-line coverage,
targeting over 90%, and report uncovered branches honestly.

Execute examples/multi_compartment/ion_learning.ipynb in a fresh CPU kernel.
Use independent one-CV, one-parameter fits for SodiumFixed.E,
SodiumInitNernst.temp, CalciumDetailed.Ci_initializer, CalciumDetailed.tau,
and ToyCaBindingKinetic_SU2015_DCN.kf. Fit spiking voltage for the first two
and concentration traces for the remaining three. Use short traces and at
most 100 Adam updates, compiled brainstate loops, and no external dataset.
Require finite results and final MSE at most 10% of initial MSE. Include
classification, gradient/error, and fit-results tables plus small checks for
derived defaults and explicit overrides. Record actual results and runtime.

## Execution Results

Implemented on the `reduction` branch without committing or pushing.

Channel and Ion now share signature discovery, layout parameter allocation,
sources, grouping, and root ownership. Ion initialization and updates use JAX
numeric scatters with persistent runtime states. The compute-layer import
guard explicitly includes the new `ions -> parameters` edge; the graph remains
acyclic. Registered Ion metadata covers 23 classes and 310 numeric defaults.

Regression checks cover repeatable compiled reset/gradients, partial initial
overrides, live buffer equilibria and shell factors, the caiBase/caliBase
fallbacks, row/population/CV/all grouping, split layouts, shared Channel/Ion
roots, parameterized profiles, and physical-unit validation. An additional
regression ensures initial-value scale baselines reflect pre-init regional
changes to the parameters that define those initial values.

Numeric Ion buffers now follow configured JAX/BrainState precision, as Channel
buffers do, rather than implicitly obtaining float64 from NumPy. Legacy Ion
runtime reference tests that demand twelve decimal places explicitly select
BrainState precision 64. Training integration tests and the notebook also run
with the default float32 backend. Constructor integer conversions still apply;
learning `substeps` fails naturally at the traced integer conversion, while
regional floating-point `valence` values retain gradients.

The pre-existing CalciumFirstOrder unit inconsistency remains outside this
change: its dimensionless alpha/beta defaults do not form a unit-consistent
concentration derivative with physical current input. It is not used as a
training example, and that error is not evidence that all Ion rates are
unlearnable. Species/reaction/conservation declarations remain intact.

### Notebook Results

Fresh-kernel execution on Python 3.11 / JAX 0.8.0 CPU completed nine code cells
and five fitted-trajectory figures without errors. Each example used one CV,
one scalar scale root, 800 steps of 0.025 ms, and 100 Adam updates at 0.03.
Both voltage targets and fitted traces contained a spike. All losses were
finite and all five fits exceeded the required tenfold MSE reduction.

| Parameter | Initial | Target | Fitted | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| E (mV) | 45 | 50 | 49.97645 | 14.18368 | 8.10408e-5 |
| temp (K) | 300 | 309.15 | 309.10382 | 4.34521 | 3.91733e-4 |
| Ci_initializer (mM) | 0.0008 | 0.001 | 0.0009999672 | 0.00501708 | 1.32332e-10 |
| tau (ms) | 7 | 5 | 4.998580 | 0.00462772 | 2.85074e-9 |
| kf (1/(mM ms)) | 1.6 | 2 | 2.000439 | 17.27890 | 1.51110e-5 |

Voltage MSE uses mV squared; concentration MSE uses uM squared. The observed
fit time was 23.75 seconds including compilation, not a performance guarantee.
The notebook also executes zero-gradient, invalid-unit/string, live-derived
initial-value, and independent explicit-initial-value checks.

### Reproduction

```bash
pytest -q --disable-warnings braincell/channel braincell/ion braincell/_compute braincell/trainable braincell/_multi_compartment braincell/_base_channel_test.py braincell/_base_ion_test.py braincell/network braincell/synapse
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 examples/multi_compartment/ion_learning.ipynb
```

### Final Verification

| Check | Result |
| --- | --- |
| Full related regression command above | 1,482 passed, 2 skipped |
| Additional dictionary-key-order regression | 1 passed |
| Changed executable-line coverage (implementation only) | 209 / 212 = 98.58% |
| Ruff lint and format checks | Passed |
| Git diff whitespace check | Passed |
| Fresh-kernel notebook, repeated | 9 executed code cells, 5 figures, no errors |

The supplementary check verifies that equivalent species-initializer dictionaries
compare structurally through the existing Params implementation, not by their
display representation. Its coverage was combined with the full suite.

Changed-line coverage intersects `git diff --unified=0` added/replaced line
ranges with coverage.py executable lines, excluding test modules. It is not
whole-repository coverage or a probability of correctness. The three uncovered
lines are the nonuniform scalar-configuration rejection, spatial-callback
evaluation while resolving a pre-init default, and the standalone kinetic
explicit-Ci default merge. Related regular paths have regression coverage.
Coverage data was collected under `/tmp/braincell-ion-checks.baCFcg/`; no
coverage dependency or report artifact was added to the repository.

GPU and other JAX versions have not been executed locally.
