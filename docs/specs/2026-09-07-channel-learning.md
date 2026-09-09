# Channel Learning

## Intent

Replace hand-maintained Channel parameter schemas with constructor-signature
discovery. Preserve default reads, pre-initialization overrides, physical units,
compact runtime parameters, region selection, grouping, shared optimizer roots,
and the existing parameter/scale/parameterized interfaces. Ion and Synapse
schemas and gate/Markov state declarations are out of scope.

Signature membership is not a promise of differentiability. Do not introduce a
scientific trainability whitelist or reject zero gradients. Existing conversion,
shape, unit, and static-control-flow errors remain meaningful outcomes. Keep
ordinary constructor configuration on its existing path; discovering a field
does not mean allocating a numeric buffer for a missing or nonnumeric default.

## Implementation

1. Discover named constructor arguments and defaults, including forwarded parent
   signatures, without treating arbitrary **kwargs as declared parameters.
   Generate internal metadata instead of requiring Channel.parameters dictionaries.
2. Preserve default get/set and runtime materialization. Constructor conversions
   and subclass overrides must not be silently undone by attaching runtime states.
3. Replace the four temperature-derived sodium phi assignments with properties.
   Independent explicit phi parameters remain ordinary parameters.
4. Make cached_q10_factor tracer-safe: bypass caching traced inputs and never
   retain traced outputs, including results produced from concrete inputs in JIT.
5. Add examples/multi_compartment/channel_learning.ipynb: three deterministic,
   independent one-CV, one-parameter fits of a spiking voltage trace, learning
   sodium g_max, V_sh, and temp. Keep injected current fixed. Explain observed
   finite gradients, legitimate zero gradients, and natural errors in tables.

## Verification

Write regression tests before fixes. Exercise omitted defaults, explicit values,
inheritance, units, shapes, constructor configuration, regional bindings and
unselected rows, all existing grouping/source modes, reset, and shared roots.
Test all affected phi variants and cache behavior under JIT and gradients.
Include zero-gradient switches, frozen gates, zero temperature offsets, and a
learnable floating-point concentration exponent despite its integer default.

Execute the notebook from a fresh CPU kernel with no external dataset. Each
target and fitted trace must contain a spike, all results must be finite, and
final waveform MSE must be at most 10% of initial MSE. Start with 20 ms traces
and 100 Adam updates, tuning only as needed to meet this teaching acceptance
criterion. Record actual results and wall time. Use brainstate transforms for
repeated simulation and optimization; retain JAX >= 0.8.0 compatibility.

## Execution Results

Implemented on the `reduction` worktree branch. The existing user changes in
`examples/multi_compartment/reduction.ipynb` were not edited.

Additional regressions found and fixed while opening signature parameters:

- Equal initial values had allowed independent regional bindings to collapse to
  one shared runtime scalar. Compression now accounts for selection ownership.
- Integer default buffers truncated floating-point updates and erased exponent
  gradients. Both training materialization and view writes now promote dtype.
- Constructor bool/int coercions must be distinguished from numeric passthrough:
  constructor inputs are arrays, and actual scalar conversions remain effective.
  Forced constructor settings are also reflected in parameter queries.
- Required/None-default fields supplied through pre-init overrides or explicit
  training initial values now receive runtime storage. Without a numeric default,
  every active row in the layout must be supplied rather than guessed.
- One existing K_Kv_test fixture supplied unitless `q=9.0` despite the signature's
  voltage slope default. The fixture now uses `9.0 * u.mV`.

Validation used Python 3.11 and JAX 0.8.0 on CPU:

| Check | Result |
| --- | --- |
| Channel, compute, trainable, multi-compartment, base Channel tests | 1,051 passed |
| Ion, Synapse, Network compatibility tests | 392 passed, 2 skipped |
| Registered Channel numeric defaults | 112 classes, 375 defaults validated |
| Changed executable line coverage | 135 / 140 = 96.43% |
| Ruff lint, format check, git diff whitespace check | Passed |
| Fresh-kernel notebook | 8 code cells, 4 plots, no errors |

Coverage is for changed executable lines in the implementation, measured with
coverage.py and the changed-line ranges from git diff, not whole-repository
coverage. GPU and other JAX versions were not executed locally. The existing CI
version matrix remains unchanged. Coverage tooling was installed only in a
temporary directory, not added to project dependencies.

All three independent notebook fits used one CV, one scalar optimizer root,
one target spike, a 20 ms waveform, and 100 Adam updates. Final fitted traces
also contained one spike each. MSE is measured in mV squared.

| Parameter | Initial | Target | Fitted | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| g_max (mS/cm^2) | 108 | 120 | 119.966 | 55.9342 | 0.0000709192 |
| V_sh (mV) | -44 | -45 | -45.0055 | 258.324 | 0.00785799 |
| temp (K) | 308.15 | 309.15 | 309.145 | 24.2734 | 0.000379125 |

The final notebook execution took 17.48 seconds for the three fits combined,
including their compilation; this is an observed run, not a performance promise.
The fixed-voltage gateCurrent probe verifies a changed forward value with zero
switch gradients. Its existing gating-current formula produces a very large
enabled current; that amplitude was not calibrated or changed here and was not
used to generate any training target.

Reproduce functional validation with:

```bash
pytest -q --disable-warnings braincell/channel braincell/_compute braincell/trainable braincell/_multi_compartment braincell/_base_channel_test.py
pytest -q --disable-warnings braincell/network braincell/synapse braincell/ion
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 examples/multi_compartment/channel_learning.ipynb
```
