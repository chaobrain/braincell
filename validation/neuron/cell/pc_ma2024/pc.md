# PC cell template notes

This file records the current PC template conventions.  New cell templates
should follow this shape first, then specialize only the morphology, parameter
values, region definitions, and mechanism blocks that differ from PC.

## Template goal

The formal BrainCell template should mirror the corresponding NEURON template.
For PC this means `pc_braincell.py` stays close to `pc_neuron.py`:

- morphology import maps NEURON `Import3d_*` to `Morphology.from_asc`;
- NEURON `sec.nseg = 1 + 2 * int(sec.L / CV_MAX_LEN_UM)` maps to
  `CVPerBranchList`;
- NEURON `sec.Ra` / `sec.cm` maps to `mech.CableProperty`;
- NEURON `insert pas` maps to `mech.Channel("IL", ...)`;
- NEURON `insert X` plus `sec.gbar_X = value` maps to one
  `mech.Channel("X", ...)`;
- NEURON ion reversal fields and calcium pump mechanisms map to `mech.Ion(...)`.

Do not hide this mapping behind broad helper abstractions.  Keep paint blocks
close to the original soma/dendrite/region logic so readers can compare the two
backends line by line.

## File layout

- `parameters.py`: shared paths, constants, grouped parameters, loader, and the
  nseg rule.
- `pc_neuron.py`: NEURON reference implementation.
- `pc_braincell.py`: BrainCell implementation.
- `run.ipynb`: formal comparison entry point.
- `debug/`: debugging versions and notebooks.  Do not use debug files as the
  source of truth for the formal template style.

## Parameter layout

Use one top-level parameter object and split values by how the template uses
them:

- `PCParameters.channel`: channel conductance/permeability values, grouped by
  region (`soma`, `dend`).
- `PCParameters.cable`: passive cable and discretization values, including
  `ra_ohm_cm`, leak reversal/conductance, capacitance, diameter thresholds, and
  `cv_max_len_um`.
- `PCParameters.ion`: ion reversal potentials and calcium pump densities.

Channel values loaded from source data remain plain floats.  The BrainCell
template writes units explicitly at paint sites, for example
`ch.soma.nav1p6 * (u.siemens / u.cm**2)` or
`ch.soma.cav3p3_perm * (u.cm / u.second)`.  Do not add automatic unit inference
based on field names.

Keep parameter names short but semantic:

- use `morph`, not `morpho`;
- use region locals such as `ch`, `cable`, and `ion` inside methods;
- avoid long chains when a local variable makes the paint block clearer.

## BrainCell class flow

The formal BrainCell class follows this order:

1. `__init__` stores `morph_path`, `params`, `temperature_celsius`,
   `v_init_mV`, `morph`, `cell`, and `regions`.
2. `build()` reads the morphology, creates per-branch CV counts, constructs
   `Cell`, then calls the private assembly steps.
3. `_define_regions()` creates all regions used by later paint calls.
4. `_paint_cable()` paints passive cable properties.
5. `_paint_ions()` paints fixed Na/K ions and region-specific Ca dynamics.
6. `_paint_channels()` paints channels in NEURON order, grouped by region.

The private method prefix is intentional.  These methods are internal assembly
steps called by `build()`, not a public stable API.

## Morphology and CV policy

Use `self.morph` for the loaded morphology.  Local variables should also use
`morph` as the abbreviation.

Use `CVPerBranchList` for PC-style NEURON nseg matching.  The rule is:

```python
1 + 2 * int(length_um / max_len_um)
```

Keep the rule in `parameters.py` (`pc24_nseg_rule`) and call it from a small
BrainCell helper that extracts each branch length in micrometers.

## Cable and regions

Define regions before painting:

- `soma`: all soma branches;
- `dend`: all dendrite branches;
- `thick_dend`: dendrites with diameter above the thick-dendrite threshold;
- `nav_dend`: dendrites with diameter above the Nav dendrite threshold.

Paint a harmless global cable default first, then paint branch-level cable
values.  This mirrors NEURON's per-section assignments and handles dendritic
capacitance rules such as:

```python
dend.cm = 11.510294 * exp(-1.376463 * dend.diam) + 2.120503
if dend.diam >= THICK_DEND_DIAM_UM:
    dend.cm = SOMA_CM_UF_CM2
```

## Ions and channels

Paint ions before channels.  NEURON often inserts channels before assigning
`ena`, `ek`, or `eca` because the inserted mechanisms create those ion fields.
BrainCell declares ions first, then channels bind to those ions.

Channel paint blocks should be grouped by biological/NEURON region:

- soma block;
- dendrite base block;
- thick-dendrite block;
- Nav-dendrite block.

Keep units visible in each `mech.Channel(...)`, `mech.Ion(...)`, and
`mech.CableProperty(...)` call.  Do not define local aliases like
`conductance = ...` solely to hide units.

## Size support

Do not add a `size` argument to this template until the core multi-compartment
`Cell` size interface has landed.  Once that interface is merged, update this
document and the template together so `PC(size=...)` has the same meaning as the
underlying `Cell(size=...)`.

## Mechanism table

| PC | Soma | Dend(R<0.8) | Dend(0.8<=R<1.65) | Dend(R>=1.65) |
| --- | --- | --- | --- | --- |
| Leak | ✓ | ✓ | ✓ | ✓ |
| HCN1 | ✓ | ✓ | ✓ | ✓ |
| Nav1.6 | ✓ |  |  | ✓ |
| Kir2.3 | ✓ |  | ✓ | ✓ |
| Kv1.1 | ✓ |  | ✓ | ✓ |
| Kv1.5 | ✓ |  | ✓ | ✓ |
| Kv3.4 | ✓ |  |  |  |
| Kv3.3 |  | ✓ | ✓ | ✓ |
| Kv4.3 |  | ✓ | ✓ | ✓ |
| Cav2.1 | ✓ | ✓ | ✓ | ✓ |
| Cav3.1 | ✓ |  | ✓ | ✓ |
| Cav3.2 | ✓ |  | ✓ | ✓ |
| Cav3.3 | ✓ | ✓ | ✓ | ✓ |
| Kca1.1 | ✓ | ✓ | ✓ | ✓ |
| Kca2.2 | ✓ | ✓ | ✓ | ✓ |
| Kca3.1 | ✓ |  | ✓ | ✓ |
| CdpCAM | ✓ | ✓ | ✓ | ✓ |
| Na_ion | ✓ |  |  | ✓ |
