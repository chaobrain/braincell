# Cerebellum Import Progress

This record tracks concrete Cerebellum imports and NEURON comparisons across ion, channel,
and cell examples. It is not the reusable API contract or evidence of full numerical equivalence.
The source inventory and test summaries below come from the earlier import work; the documentation
move on 2026-09-07 did not rerun comparisons or advance their validation status.

## Current status

- Declarative KineticIon and its public pieces exist. Reusable species, conservation, and current-input
  semantics are documented in the ``KineticIon`` docstrings.
- Concrete calcium pools are imported in [calcium.py](../../braincell/ion/calcium.py):
  `CdpStC_MA2020_GoC`, `CdpStC_NoCAM_MA2020_GoC`, `CdpStC_CAMOnly_MA2020_GoC`,
  `CdpStC_MA2025_BC`, `CdpStC_RI2021_SC`, `CdpCAM_MA2024_PC`, and `CdpCR_MA2020_GrC`.
- PC MA2024 channel imports and targeted tests span sodium, potassium, calcium,
  calcium-activated potassium, and HCN modules.
- The [PC MA2024 scaffold](cell/pc_ma2024/pc.md) includes simplified NEURON and BrainCell
  assemblies, shared parameters, debug variants, and a [comparison notebook](cell/pc_ma2024/run.ipynb).
- Spatial callable parameters are implemented; the former statement that paint required explicit
  arrays or per-region calls is obsolete. See the [filter API reference](../../docs/apis/filter.rst).

## Comparison entry points

| Work | Location |
| --- | --- |
| Calcium-pool and ion comparisons | [Ion pool configurations and runner](ion/README.md) |
| Channel comparisons | [Channel examples](channel_no_conc/README.md) |
| Whole-cell comparisons | [Cell examples](cell/README.md) |
| PC MA2024 validation target | [PC comparison notebook](cell/pc_ma2024/run.ipynb) |
| Imported MOD provenance | [Cerebellum MOD sources](../../data/cerebellum/mechanism-catalog.md) |

## Comparison configuration

For runs with current-driven calcium pools, use `cache_ion_total_current=True` together with
`ion_channel_update_order="family"`. Both are current Cell defaults; specifying them makes the
comparison configuration explicit. `"integration"` remains an alternative for controlled comparisons.
The timing and ownership of these options are documented on ``Cell``.

The import work also covered same-name channels on disjoint soma/dendrite layouts and local PC
calcium-channel `_Frozen` variants that stop differentiation through the voltage in the current
expression. These are specific compatibility paths, not proof that the full model matches NEURON.

## Implementation and test locations

- [Ion template](../../braincell/ion/_base.py) and [template tests](../../braincell/ion/_base_test.py).
- [Ion lifecycle](../../braincell/_base_ion.py) and [lifecycle tests](../../braincell/_base_ion_test.py).
- [Cell runtime](../../braincell/_multi_compartment/cell.py) and [Cell tests](../../braincell/_multi_compartment/cell_test.py).
- [Ion construction](../../braincell/_compute/ions.py) and [ion tests](../../braincell/_compute/ions_test.py).
- [Runtime bindings](../../braincell/_compute/bindings.py) and [binding tests](../../braincell/_compute/bindings_test.py).
- [Staggered integration](../../braincell/quad/_staggered.py).
- Channel implementations and adjacent tests under [channel](../../braincell/channel), including
  sodium, potassium, calcium, potassium_calcium, and hyperpolarization_activated modules.

The earlier record also described MechanismProbe support for plain-value fields and listed the old
probes implementation/tests. That is historical context, not a current API entry point; use the current
recording API of ``Network`` for supported observation interfaces.

## Validation limits

- The PC MA2024 scaffold remains a live validation target. Track numerical differences in the
  notebooks before promoting it to a regression baseline.
- Not every Cerebellum MOD file has a BrainCell counterpart. The completed import focus was the
  ion/channel subset needed by the PC and calcium-pool comparisons.
- These comparisons do not establish kinetic-ion equivalence for a future unified single mode.
  SingleCompartment has its own update path; compatibility scope remains an open design question
  (single/multi-compartment unification).
- The original record reported targeted scheduling/runtime, calcium-ion, and PC channel tests.
  It did not include a dated run manifest or complete tolerance table, and those results were not
  rerun during this documentation move.

## Next steps

| Work | Status | Next action |
| --- | --- | --- |
| PC MA2024 whole-cell comparison | 实施中 | Tighten the notebook comparison and record remaining discrepancies |
| Automated regression baseline | 待讨论 | Establish expected tolerances before promoting stable comparisons into tests |
| Remaining Cerebellum imports | 待讨论 | Prioritize remaining MOD mechanisms after the PC scheduling path is stable |

PC MA2024 remains the current end-to-end validation target for this import effort.
Source attribution lives in the References sections of the ion and channel docstrings.
