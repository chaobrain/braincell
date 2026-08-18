# Task 6 report — split `runtime_test.py` into four co-located test modules

`braincell/_compute/runtime_test.py` (3,200 lines, 111 tests, 13 top-level test
classes) is deleted and replaced by five new files: one private fixture module
and four `*_test.py` siblings matching the four production modules that replaced
`runtime.py` in tasks 2-5.

## Method

The split was performed by a generator script that slices exact line ranges out
of `runtime_test.py` and reassembles them, so no test body was retyped. A second
script then re-parsed both the committed original (`git show HEAD:...`) and the
four new files with `ast`, extracted every `test*` method as raw source text, and
compared them string-for-string. All 111 compare equal.

## Partition — complete and disjoint

`ast` walk of the original yields exactly 111 `test*` methods; `ast` walk of the
four new files yields exactly 111. `set(old) == set(new)` holds, and the
assembly script asserts no test name appears in two new files. The 66 methods of
`CellRuntimeStateTest` were assigned by the brief's line numbers; the script
asserts `sorted(layout + ion + binding + state) == sorted(all_method_starts)`,
which is what proves the four-way carve-up is a permutation rather than merely
the right size.

Per-file test counts, from `--collect-only`:

```
    24 braincell/_compute/bindings_test.py
     7 braincell/_compute/bridge_test.py
    28 braincell/_compute/ions_test.py
    32 braincell/_compute/layouts_test.py
     7 braincell/_compute/spatial_params_test.py
    27 braincell/_compute/state_test.py
     2 braincell/_compute/table_test.py
    14 braincell/_compute/topology_test.py
```

32 + 28 + 24 + 27 = 111, matching the brief's target table exactly.

Verifier output:

```
old methods: 111  new methods: 111
byte-identical: 111 / 111
Counter({'layouts_test.py': 32, 'ions_test.py': 28, 'state_test.py': 27, 'bindings_test.py': 24})
fixtures byte-identical: OK
```

The `fixtures byte-identical` line asserts that `_RuntimeTestTwoOwnerChannel`
(old 40-60, modulo the one-word `__module__` change), `_build_tree` (63-68),
`_quantity_set_at` (71-74) and the five clamp stubs (2670-2695) appear verbatim
in their new homes.

## File contents

### `_testing.py` (full)

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Shared fixtures for the ``braincell._compute`` runtime test modules.

This module is deliberately not named ``*_test.py``: it holds helpers, not
tests, so pytest must not collect it. ``_RuntimeTestTwoOwnerChannel`` in
particular must be defined exactly once in the repository, because
``braincell.mech.register_channel`` raises ``ValueError`` when the same
channel name is registered twice.
"""

import brainstate
import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell._base import Channel, IonInfo
from braincell.ion import NonSpecific, Potassium
from braincell.mech import register_channel


@register_channel("_RuntimeTestTwoOwnerChannel")
class _RuntimeTestTwoOwnerChannel(Channel):
    """Small multi-owner channel used by runtime binding tests."""

    __module__ = "braincell._compute._testing"
    root_type = brainstate.mixin.JointTypes[Potassium, NonSpecific]
    current_owner_types = {"k": Potassium, "no": NonSpecific}

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)

    def current(self, V, K: IonInfo, No: IonInfo):
        parts = self.current_components(V, K, No)
        return parts["k"] + parts["no"]

    def current_components(self, V, K: IonInfo, No: IonInfo):
        _ = (K, No)
        return {
            "k": 2.0 * u.math.ones_like(V.to_decimal(u.mV)) * (u.nA / u.cm**2),
            "no": 3.0 * u.math.ones_like(V.to_decimal(u.mV)) * (u.nA / u.cm**2),
        }


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def _quantity_set_at(value, index: int, replacement):
    decimal = np.array(value.to_decimal(value.unit), copy=True)
    decimal[..., int(index)] = replacement.to_decimal(value.unit)
    return u.Quantity(decimal, value.unit)
```

### `layouts_test.py` — docstring and imports

```python
"""Tests for :mod:`braincell._compute.layouts`."""

import unittest
from dataclasses import dataclass as _dataclass

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import (
    CVPerBranch,
    Cell,
    CurrentClamp,
    FunctionClamp,
    SineClamp,
)
from braincell.filter import BranchSlice, RootLocation, at
from ._testing import _build_tree
from .layouts import (
    CLAMP_KINDS,
    ClampRoutingTable,
    build_clamp_routing_table,
)


@_dataclass
```

### `ions_test.py` — docstring and imports

```python
"""Tests for :mod:`braincell._compute.ions`."""

import unittest

import braintools
import brainunit as u
import numpy as np

import braincell
from braincell import Cell
from braincell.filter import BranchSlice, at
from ._testing import _build_tree, _quantity_set_at
```

### `bindings_test.py` — docstring and imports

```python
"""Tests for :mod:`braincell._compute.bindings`.

``_RuntimeTestTwoOwnerChannel`` is looked up here by registry name only; the
``._testing`` import below is what registers it.
"""

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell import Cell
from braincell.filter import BranchSlice, at
from braincell.quad import get_integrator
from ._testing import _build_tree
```

### `state_test.py` — docstring and imports

```python
"""Tests for :mod:`braincell._compute.state`."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import (
    Cell,
    CurrentClamp,
)
from braincell.filter import BranchSlice, RootLocation, at
from ._testing import _build_tree
```

All four carry the Apache-2.0 header with year 2026 on lines 1-14, omitted above
for brevity; it is identical to the block shown in full in `_testing.py`.

### Structure of the four test modules

Every method body below is byte-identical to the cited `runtime_test.py` range.

#### `layouts_test.py`

```
_ClampStubLayout
_ClampStubCV
_ClampStubNodeTree
_clamp_node_tree
_clamp_cv
RuntimeLayoutTest
    test_density_mechanism_builds_dense_layout_with_global_shape  <- runtime_test.py:78-104 (CellRuntimeStateTest)
    test_region_limited_density_current_is_masked_outside_active_points  <- runtime_test.py:106-129 (CellRuntimeStateTest)
    test_point_mechanism_builds_sparse_layout_with_local_shape  <- runtime_test.py:131-155 (CellRuntimeStateTest)
    test_point_mechanism_can_land_on_root_endpoint  <- runtime_test.py:157-175 (CellRuntimeStateTest)
    test_channel_spec_builds_dense_layout_with_global_shape  <- runtime_test.py:222-247 (CellRuntimeStateTest)
    test_named_channel_spec_merges_across_regions_when_identity_matches  <- runtime_test.py:249-266 (CellRuntimeStateTest)
    test_same_class_different_names_build_distinct_layouts  <- runtime_test.py:268-283 (CellRuntimeStateTest)
    test_runtime_state_keeps_dense_and_sparse_layouts_together  <- runtime_test.py:285-308 (CellRuntimeStateTest)
    test_runtime_evaluates_step_sine_and_function_clamps_on_target_points  <- runtime_test.py:349-368 (CellRuntimeStateTest)
    test_function_clamp_receives_absolute_time  <- runtime_test.py:370-385 (CellRuntimeStateTest)
    test_sine_clamp_uses_delay_window  <- runtime_test.py:387-408 (CellRuntimeStateTest)
    test_probe_layouts_are_sparse_and_allocate_no_state_buffers  <- runtime_test.py:410-434 (CellRuntimeStateTest)
EvaluatePointClampsJitTest
    test_evaluate_point_clamps_jit_compiles  <- runtime_test.py:2162-2172 (EvaluatePointClampsJitTest)
    test_evaluate_point_clamps_supports_population_specific_amplitudes  <- runtime_test.py:2174-2186 (EvaluatePointClampsJitTest)
    test_evaluate_point_clamps_can_filter_midpoint_and_boundary_points  <- runtime_test.py:2188-2219 (EvaluatePointClampsJitTest)
DensityLayoutMaskingUnderJit
    test_state_buffer_mantissa_sums_under_jit  <- runtime_test.py:2225-2235 (DensityLayoutMaskingUnderJit)
TestBuildClampRoutingTable
    test_no_clamp_layouts_returns_none  <- runtime_test.py:2699-2713 (TestBuildClampRoutingTable)
    test_midpoint_current_clamp_builds_midpoint_route  <- runtime_test.py:2715-2732 (TestBuildClampRoutingTable)
    test_each_clamp_kind_is_recognized  <- runtime_test.py:2734-2735 (TestBuildClampRoutingTable)
    test_ids_are_sorted_and_unique  <- runtime_test.py:2737-2756 (TestBuildClampRoutingTable)
    test_zero_area_raises  <- runtime_test.py:2758-2772 (TestBuildClampRoutingTable)
    test_endpoint_clamp_builds_boundary_route  <- runtime_test.py:2774-2791 (TestBuildClampRoutingTable)
    test_mixed_midpoint_and_endpoint_clamps_route_separately  <- runtime_test.py:2793-2814 (TestBuildClampRoutingTable)
    test_non_clamp_point_layout_ignored  <- runtime_test.py:2816-2831 (TestBuildClampRoutingTable)
StateBufferStorageTest
    test_density_buffer_is_quantity_backed_by_jax  <- runtime_test.py:2845-2856 (StateBufferStorageTest)
    test_set_state_broadcast_scalar_and_readback  <- runtime_test.py:2858-2868 (StateBufferStorageTest)
    test_set_state_shape_mismatch_raises  <- runtime_test.py:2870-2883 (StateBufferStorageTest)
    test_no_object_dtype_in_density_buffer  <- runtime_test.py:2885-2894 (StateBufferStorageTest)
RaggedCurrentClampBufferTest
    test_identical_lambda_bodies_produce_one_layout  <- runtime_test.py:2900-2912 (RaggedCurrentClampBufferTest)
    test_three_clamps_with_varying_step_counts_pad_and_mask  <- runtime_test.py:2914-2948 (RaggedCurrentClampBufferTest)
    test_population_multistep_current_clamp_preserves_step_axis  <- runtime_test.py:2950-2976 (RaggedCurrentClampBufferTest)
FnFingerprintWarnsOnOpaqueClosureTest
    test_opaque_closure_emits_warning  <- runtime_test.py:2982-3010 (FnFingerprintWarnsOnOpaqueClosureTest)
```

#### `ions_test.py`

```
RuntimeIonTest
    test_default_ions_are_available_with_global_shape  <- runtime_test.py:609-622 (CellRuntimeStateTest)
    test_default_ions_expand_with_population_shape  <- runtime_test.py:624-630 (CellRuntimeStateTest)
    test_runtime_ions_expose_point_space_geometry_arrays  <- runtime_test.py:632-651 (CellRuntimeStateTest)
    test_runtime_ion_geometry_expands_with_population_shape  <- runtime_test.py:653-661 (CellRuntimeStateTest)
    test_single_named_ion_keeps_family_and_class_aliases  <- runtime_test.py:663-683 (CellRuntimeStateTest)
    test_explicit_init_nernst_ion_replaces_default_species_container  <- runtime_test.py:685-709 (CellRuntimeStateTest)
    test_multiple_named_ions_make_family_lookup_ambiguous  <- runtime_test.py:711-730 (CellRuntimeStateTest)
    test_dynamic_ion_lifecycle_runs_in_runtime  <- runtime_test.py:892-934 (CellRuntimeStateTest)
    test_imported_cdp_ion_relaxes_without_channel_in_runtime  <- runtime_test.py:936-957 (CellRuntimeStateTest)
    test_imported_cdp_ion_and_cahva_channel_run_together  <- runtime_test.py:959-991 (CellRuntimeStateTest)
    test_imported_cdplva_ion_relaxes_without_channel_in_runtime  <- runtime_test.py:993-1014 (CellRuntimeStateTest)
    test_imported_cdplva_ion_and_calva_channel_run_together  <- runtime_test.py:1016-1050 (CellRuntimeStateTest)
    test_toy_kinetic_ion_runs_and_exposes_species_probes  <- runtime_test.py:1052-1075 (CellRuntimeStateTest)
    test_toy_kinetic_ion_reset_restores_custom_initializers  <- runtime_test.py:1077-1103 (CellRuntimeStateTest)
    test_toy_source_kinetic_ion_runs_and_exposes_species_probes  <- runtime_test.py:1105-1140 (CellRuntimeStateTest)
    test_toy_ica_source_kinetic_ion_and_cahva_run_together  <- runtime_test.py:1142-1191 (CellRuntimeStateTest)
    test_toy_factor_kinetic_ion_and_cahva_run_together  <- runtime_test.py:1193-1238 (CellRuntimeStateTest)
    test_toy_diam_factor_kinetic_ion_runs_and_exposes_geometry_factor_species  <- runtime_test.py:1240-1271 (CellRuntimeStateTest)
    test_toy_diam_factor_kinetic_ion_reset_restores_custom_initializers  <- runtime_test.py:1273-1299 (CellRuntimeStateTest)
    test_cdpstc_goc_runs_and_exposes_species_and_geometry_probes  <- runtime_test.py:1301-1399 (CellRuntimeStateTest)
    test_cdpstc_and_cav3p1_goc_run_together  <- runtime_test.py:1401-1495 (CellRuntimeStateTest)
    test_cdpstc_and_cav2p1_run_together  <- runtime_test.py:1497-1587 (CellRuntimeStateTest)
    test_cdpstc_camonly_goc_runs_and_exposes_species_and_geometry_probes  <- runtime_test.py:1589-1654 (CellRuntimeStateTest)
    test_cdpstc_nocam_goc_runs_and_exposes_species_and_geometry_probes  <- runtime_test.py:1656-1748 (CellRuntimeStateTest)
    test_cdpcam_pc_zero_ica_runs_and_exposes_steady_species_and_geometry_probes  <- runtime_test.py:1750-1799 (CellRuntimeStateTest)
    test_cdpcam_pc_ion_params_scatter_with_population_shape  <- runtime_test.py:1801-1833 (CellRuntimeStateTest)
    test_constant_quantity_ci_initializer_stays_quantity_with_population_shape  <- runtime_test.py:1835-1876 (CellRuntimeStateTest)
    test_same_ion_instance_name_cannot_mix_different_classes  <- runtime_test.py:1905-1921 (CellRuntimeStateTest)
```

#### `bindings_test.py`

```
RuntimeBindingTest
    test_density_mechanism_leaky_builds_runtime_il_node  <- runtime_test.py:573-588 (CellRuntimeStateTest)
    test_set_state_syncs_runtime_node_param  <- runtime_test.py:590-607 (CellRuntimeStateTest)
    test_single_ion_channel_binds_to_explicit_runtime_ion  <- runtime_test.py:732-758 (CellRuntimeStateTest)
    test_same_named_single_ion_channels_in_distinct_layouts_do_not_overwrite  <- runtime_test.py:760-793 (CellRuntimeStateTest)
    test_same_named_overlapping_ion_channels_remain_distinct  <- runtime_test.py:795-817 (CellRuntimeStateTest)
    test_set_state_syncs_merged_channel_param  <- runtime_test.py:819-842 (CellRuntimeStateTest)
    test_single_ion_channel_requires_selector_when_family_is_ambiguous  <- runtime_test.py:844-864 (CellRuntimeStateTest)
    test_set_state_on_named_ion_layout_updates_only_that_instance  <- runtime_test.py:866-890 (CellRuntimeStateTest)
    test_calva_channel_binds_only_to_explicit_lva_ion_when_multiple_calcium_ions_exist  <- runtime_test.py:1878-1903 (CellRuntimeStateTest)
    test_mixed_ion_channel_binds_per_family_and_uses_owner_ion_bucket  <- runtime_test.py:1923-1948 (CellRuntimeStateTest)
    test_same_named_mixed_ion_channels_in_distinct_layouts_do_not_overwrite_owner_current  <- runtime_test.py:1950-1998 (CellRuntimeStateTest)
    test_mixed_ion_channel_probe_uses_bound_ions_and_owner_total_current  <- runtime_test.py:2000-2034 (CellRuntimeStateTest)
    test_multi_owner_mixed_ion_channel_exposes_component_currents  <- runtime_test.py:2036-2076 (CellRuntimeStateTest)
    test_family_order_integrates_mixed_ion_wrapper_channel_only  <- runtime_test.py:2078-2091 (CellRuntimeStateTest)
    test_channel_spec_ina_hh1952_builds_runtime_node_and_binds_to_na  <- runtime_test.py:2093-2117 (CellRuntimeStateTest)
    test_set_state_syncs_runtime_node_param_for_ina_hh1952  <- runtime_test.py:2119-2142 (CellRuntimeStateTest)
    test_unknown_channel_name_raises_key_error  <- runtime_test.py:2144-2156 (CellRuntimeStateTest)
IsRootLevelRuntimeNodeUnknownClassTest
    test_unknown_channel_kind_raises_value_error  <- runtime_test.py:2241-2246 (IsRootLevelRuntimeNodeUnknownClassTest)
RuntimeSubsolverScheduleTest
    test_cell_schedule_applies_to_markov_channels_and_kinetic_ions  <- runtime_test.py:3046-3074 (RuntimeSubsolverScheduleTest)
    test_local_override_has_priority_over_cell_schedule  <- runtime_test.py:3076-3105 (RuntimeSubsolverScheduleTest)
    test_one_kinetic_override_applies_to_the_shared_named_runtime  <- runtime_test.py:3107-3129 (RuntimeSubsolverScheduleTest)
    test_conflicting_shared_kinetic_overrides_are_rejected  <- runtime_test.py:3131-3153 (RuntimeSubsolverScheduleTest)
    test_non_independent_density_override_is_rejected  <- runtime_test.py:3155-3168 (RuntimeSubsolverScheduleTest)
    test_different_markov_overrides_are_not_merged  <- runtime_test.py:3170-3200 (RuntimeSubsolverScheduleTest)
```

#### `state_test.py`

```
CellRuntimeStateTest
    test_synapse_mechanism_builds_sparse_runtime_node_and_pre_spike_state  <- runtime_test.py:177-202 (CellRuntimeStateTest)
    test_synapse_pre_spike_can_be_mutated_through_runtime_state  <- runtime_test.py:204-220 (CellRuntimeStateTest)
    test_rebuild_after_place_produces_new_runtime  <- runtime_test.py:310-326 (CellRuntimeStateTest)
    test_state_mutation_updates_buffer_without_rebuild  <- runtime_test.py:328-347 (CellRuntimeStateTest)
    test_sample_probe_reads_voltage_and_channel_gate_state  <- runtime_test.py:436-465 (CellRuntimeStateTest)
    test_sample_probe_reads_mechanism_and_total_ion_current  <- runtime_test.py:467-494 (CellRuntimeStateTest)
    test_sample_probe_reads_pure_channel_current_without_ion_selector  <- runtime_test.py:496-523 (CellRuntimeStateTest)
    test_sample_probe_reads_plain_field_and_rejects_unknown_mechanism  <- runtime_test.py:525-549 (CellRuntimeStateTest)
    test_sample_probes_requires_unique_names  <- runtime_test.py:551-571 (CellRuntimeStateTest)
CellLifecycleInlineTest
    test_init_state_installs_runtime_attributes_directly  <- runtime_test.py:2252-2262 (CellLifecycleInlineTest)
    test_reset_clears_runtime_attributes  <- runtime_test.py:2264-2273 (CellLifecycleInlineTest)
    test_init_reset_init_is_idempotent  <- runtime_test.py:2275-2282 (CellLifecycleInlineTest)
PopulationResponseHeterogeneityTest
    test_population_cells_can_have_different_current_clamp_responses  <- runtime_test.py:2286-2303 (PopulationResponseHeterogeneityTest)
    test_two_dimensional_population_can_run_with_population_specific_clamp  <- runtime_test.py:2305-2320 (PopulationResponseHeterogeneityTest)
    test_population_cells_can_have_different_current_clamp_delays  <- runtime_test.py:2322-2342 (PopulationResponseHeterogeneityTest)
    test_population_delay_works_with_multistep_current_clamp  <- runtime_test.py:2344-2361 (PopulationResponseHeterogeneityTest)
    test_current_clamp_delay_uses_active_point_axis  <- runtime_test.py:2363-2383 (PopulationResponseHeterogeneityTest)
    test_unbroadcastable_current_clamp_delay_raises  <- runtime_test.py:2385-2396 (PopulationResponseHeterogeneityTest)
PointSynapseRuntimeTest
    test_synapse_compute_derivative_populates_ode_state  <- runtime_test.py:2400-2426 (PointSynapseRuntimeTest)
    test_ampa_synapse_drive_changes_state_and_voltage  <- runtime_test.py:2428-2465 (PointSynapseRuntimeTest)
    test_expsyn_drive_jumps_g_and_then_decays  <- runtime_test.py:2467-2505 (PointSynapseRuntimeTest)
    test_exp2syn_drive_updates_A_B_and_current  <- runtime_test.py:2507-2556 (PointSynapseRuntimeTest)
    test_netstim_can_drive_expsyn_through_cell_run  <- runtime_test.py:2558-2585 (PointSynapseRuntimeTest)
    test_synapse_input_preparation_sums_manual_netstim_and_bound_drive  <- runtime_test.py:2587-2624 (PointSynapseRuntimeTest)
    test_expsyn_discrete_event_applies_at_begin_step_not_post_integral  <- runtime_test.py:2626-2654 (PointSynapseRuntimeTest)
CellRuntimeStateIsMutableTest
    test_cell_runtime_state_is_not_frozen  <- runtime_test.py:3016-3022 (CellRuntimeStateIsMutableTest)
    test_no_object_setattr_on_runtime_in_hot_paths  <- runtime_test.py:3024-3042 (CellRuntimeStateIsMutableTest)
```



Whole classes moved verbatim (class name and all):

| Class | Old lines | New file |
|---|---|---|
| `EvaluatePointClampsJitTest` | 2159-2219 | `layouts_test.py` |
| `DensityLayoutMaskingUnderJit` | 2222-2235 | `layouts_test.py` |
| `TestBuildClampRoutingTable` | 2698-2831 | `layouts_test.py` |
| `StateBufferStorageTest` | 2842-2894 | `layouts_test.py` |
| `RaggedCurrentClampBufferTest` | 2897-2976 | `layouts_test.py` |
| `FnFingerprintWarnsOnOpaqueClosureTest` | 2979-3010 | `layouts_test.py` |
| `IsRootLevelRuntimeNodeUnknownClassTest` | 2238-2246 | `bindings_test.py` |
| `RuntimeSubsolverScheduleTest` | 3045-3200 | `bindings_test.py` |
| `CellLifecycleInlineTest` | 2249-2282 | `state_test.py` |
| `PopulationResponseHeterogeneityTest` | 2285-2396 | `state_test.py` |
| `PointSynapseRuntimeTest` | 2399-2654 | `state_test.py` |
| `CellRuntimeStateIsMutableTest` | 3013-3042 | `state_test.py` |

## Verification — all six steps, unedited output

### 1. Test-name fingerprint diff (primary check)

```
$ python -m pytest braincell/_compute/ --collect-only -q > /tmp/collect.txt 2>&1
$ grep '::' /tmp/collect.txt | sed 's/.*:://' | sort > /tmp/after.txt
$ diff .superpowers/sdd/compute-testnames-at-t5.txt /tmp/after.txt
$ echo "DIFF_EXIT=$?"
DIFF_EXIT=0
```

The diff produced **no output at all** and exited 0. 141 names in the baseline,
141 after, same multiset.

### 2. `python -m pytest braincell/_compute/ -q`

```
.....................................................................  [100%]
141 passed in 23.92s
```

### 3. `python -m pytest braincell/ -q`

```
2240 passed, 19 skipped, 52 warnings, 289 subtests passed in 438.21s (0:07:18)
```

Identical to the baseline (2240 passed, 19 skipped).

### 4. `_testing.py` is not collected

```
$ grep -c '_testing' /tmp/collect.txt
0
```

The string `_testing` does not occur anywhere in the collection output. Files
collected under `_compute/` are the eight listed in the count table above; no
`_testing.py`.

### 5. `_RuntimeTestTwoOwnerChannel` defined exactly once

```
$ grep -rn 'class _RuntimeTestTwoOwnerChannel' --include=*.py .
./braincell/_compute/_testing.py:36:class _RuntimeTestTwoOwnerChannel(Channel):
```

One hit, repo-wide.

### 6. ruff

```
$ .../ruff check braincell/_compute/
All checks passed!

$ .../ruff format --check braincell/_compute/
17 files already formatted
```

Also run during development with the wider `--select F,E`; clean.

### Extra: no dangling references to `runtime_test.py`

`grep -rn 'runtime_test'` over `.py`/`.md`/`.toml`/`.cfg`/`.in` finds only
(a) `braincell/network/runtime_test.py`, an unrelated file that still exists,
(b) prior task briefs and reports under `.superpowers/sdd/` (historical
records), and (c) one stale doc mention at
`docs/design/ion-cerebellum-import-plan.md:61` plus one at
`docs/specs/2026-08-13-cell-hidden-group-state.md:298`. Both are docs and fall
under task 8's scope; not touched here.

## Decisions made

1. **`_RuntimeTestTwoOwnerChannel` is never imported as a symbol.** Grepping the
   original showed every use is a *registry name string* —
   `braincell.mech.Channel("_RuntimeTestTwoOwnerChannel", ...)`,
   `layout.kind == "channel:_RuntimeTestTwoOwnerChannel"`,
   `k_main.channels["_RuntimeTestTwoOwnerChannel"]`. So no test file needs to
   import the class; what they need is for `_testing` to have been imported so
   that `@register_channel` has run. `bindings_test.py`, the only consumer,
   already does `from ._testing import _build_tree`, which triggers it. I first
   emitted an explicit `_RuntimeTestTwoOwnerChannel` import; ruff flagged it
   F401, so I removed it and instead documented the dependency in
   `bindings_test.py`'s module docstring, so a future reader who deletes the
   `_build_tree` import knows what else breaks.

2. **Method-local imports were left in place, not hoisted.** The brief's step 4
   says "hoist the mid-file imports". I hoisted the three *module-level* mid-file
   import sites (2661 `dataclasses`, 2663 `.layouts` clamp symbols, 2838-2839
   `jax`/`jax.numpy`). The imports at 2985 (`_fn_fingerprint`, `_opaque_warned`)
   and 3017 (`CellRuntimeState`) sit *inside test method bodies*; hoisting them
   would edit a body, violating the byte-identity rule, which I read as the
   stronger constraint. Both symbols still land in the file the brief's table
   assigns them to (`layouts_test.py` and `state_test.py` respectively). Same for
   the in-body `import braincell` at 223/250/269/591/610/2120, the in-body
   `import warnings` / `import pathlib`, and the `_is_root_level_runtime_node`
   import at 2242.

3. **Import blocks are per-file minimal, not the union.** Each file's imports
   were computed from the names its own tests actually reference and then checked
   with ruff's pyflakes rules: F401 (unused) and F821 (undefined) are both clean,
   so each file imports exactly what it uses, no more. Concretely,
   `state_test.py` gets `jax.numpy as jnp` but not `jax` (it never uses bare
   `jax.`); `ions_test.py` is the only file needing `braintools` and
   `_quantity_set_at`; only `layouts_test.py` imports from `.layouts`, and only
   it needs `_dataclass`, `jax`, `CVPerBranch`, `FunctionClamp`, `SineClamp`.

4. **Class docstrings added to the three new classes.** `RuntimeLayoutTest`,
   `RuntimeIonTest` and `RuntimeBindingTest` are new class names, so each gets a
   one-line summary. `CellRuntimeStateTest` keeps its name per the brief and also
   gained one (the original had none). These are additions above the method
   bodies, not edits to them.

5. **Clamp stubs placed after the import block, above every test class.** The
   brief says "top of `layouts_test.py` ... do not leave them mid-file". They now
   sit at lines 43-68, ahead of `RuntimeLayoutTest`, even though their only
   consumer `TestBuildClampRoutingTable` comes later in the file.

6. **Class ordering within each file** follows the brief's order: the carved-up
   class first, then the whole classes in their original relative order.

7. **`__module__` on the fixture channel** changed from
   `"braincell._compute.runtime_test"` to `"braincell._compute._testing"` as
   instructed. Suite stays green, confirming nothing reads it as a key.

8. **Import ordering** normalised to alphabetical within the third-party and
   relative groups (`brainunit`, `jax`, `jax.numpy`, `numpy`; `._testing` before
   `.layouts`). Ruff's isort rule (`I`) is deliberately not enabled in this repo
   (`pyproject.toml` `lint.select = ["E4", "E7", "E9", "F"]`), so this is
   cosmetic consistency, not a lint requirement.

## Concerns

None blocking. Two notes for downstream tasks:

- `docs/design/ion-cerebellum-import-plan.md:61` and
  `docs/specs/2026-08-13-cell-hidden-group-state.md:298` still name
  `braincell/_compute/runtime_test.py`. Task 8 ("update docs and comments")
  should catch these.
- `bindings_test.py`'s access to the registered `_RuntimeTestTwoOwnerChannel`
  depends on its `from ._testing import _build_tree` line as an implicit
  side-effect import. Documented in that file's module docstring.

## Post-review fixes

Review of task 6 raised one Important finding and one Minor finding. Both
fixed; no test method body touched.

### Finding 1 (Important) — fragile side-effect import in `bindings_test.py`

`_RuntimeTestTwoOwnerChannel` is registered via `@register_channel` as a side
effect of importing `_testing`, but `bindings_test.py` only referenced the
sibling symbol `_build_tree` from that module — never `_RuntimeTestTwoOwnerChannel`
itself. If a future edit dropped the last use of `_build_tree`, the import (and
the registration side effect) would silently disappear: no `ImportError` at
collection time, just a much later `KeyError: No 'channel' mechanism registered
as '_RuntimeTestTwoOwnerChannel'` the first time a test tried to resolve it by
name — and only when the file is run in isolation, since the sibling test
modules also import `._testing` and mask the problem in a full-suite run.

Fixed by importing the class explicitly:

```python
from ._testing import _RuntimeTestTwoOwnerChannel, _build_tree  # noqa: F401  # registers the channel
```

`F401` (unused import) is in `pyproject.toml`'s `lint.ignore`, so the `noqa` is
belt-and-braces against a future tightening of `lint.select`, not a requirement
for the current gate. Updated the module docstring accordingly:

```python
"""Tests for :mod:`braincell._compute.bindings`.

``_RuntimeTestTwoOwnerChannel`` is looked up here by registry name only, never
as a symbol, so it is imported explicitly below (and marked ``noqa: F401``) to
make the ``@register_channel`` side effect an ordinary, order-independent
import rather than a coincidence of also needing ``_build_tree``.
"""
```

### Finding 2 (Minor) — docstring scope in `state_test.py`

The module docstring claimed the file tests only `braincell._compute.state`,
but `CellLifecycleInlineTest` (3 tests, ~line 259) asserts on
`Cell.init_state()` / `Cell.reset()` installing and clearing runtime attributes
on the `Cell` object itself — `Cell` lifecycle behavior, not `state.py`. Left
the tests where they are (still the best home) and reworded the docstring:

```python
"""Tests for :mod:`braincell._compute.state` and the cell lifecycle that builds it."""
```

### Verification — unedited output

**1. `python -m pytest braincell/_compute/bindings_test.py -q` (run alone)**

```
........................                                                 [100%]
24 passed in 7.65s
```

**2. `python -m pytest braincell/_compute/ -q`**

```
........................................................................ [ 51%]
.....................................................................    [100%]
141 passed in 23.91s
```

**3. `python -m pytest braincell/ -q`**

```
2240 passed, 19 skipped, 52 warnings, 289 subtests passed in 480.06s (0:08:00)
```

**4. ruff (pre-commit-managed binary)**

```
$ .../ruff check braincell/_compute/
All checks passed!

$ .../ruff format --check braincell/_compute/
17 files already formatted
```

**5. Test-body identity check (`ast`, `HEAD` vs. working tree)**

Parsed every `test*` method from `bindings_test.py`, `ions_test.py`,
`layouts_test.py`, `state_test.py` at `HEAD` and in the working tree via
`ast.walk` + `ast.get_source_segment`, matched by `(file, method name)`, and
compared source text:

```
HEAD methods: 111
WT methods: 111
identical (file,name) pairs bodies: 111
differing: 0
```

All 111 test method bodies are byte-identical between `HEAD` and the working
tree — only the two import/docstring edits above were made.

## Concerns

None. Both findings addressed without touching any test body; full suite
count (2240 passed, 19 skipped) unchanged from the pre-review baseline.
