# IO ZH2019 Cell Template Notes

This file records the current formal IO template conventions.  The formal
template is intentionally small: IO has no morphology import in this cell-level
comparison and builds one manual soma in both NEURON and BrainCell.

## Source

- ModelDB: https://modeldb.science/257028?tab=1
- GitHub: https://github.com/ModelDBRepository/257028
- Local mod sources: `examples/neuron_compare/Cerebellum_mod/IO`
- Reference: Zhang X, Santaniello S. (2019), PNAS.

The local `.mod` files are treated as extracted IO mechanisms from that source
model, following the note in `Cerebellum_mod/IO/README.md`.

## File Layout

- `parameters.py`: formal source paths, soma geometry, cable, ion, and channel
  defaults.
- `io_neuron.py`: NEURON reference implementation.
- `io_braincell.py`: BrainCell implementation.
- `debug/`: toggle-enabled debugging scripts and notebook.

The debug files are useful for channel isolation, but they are not the formal
template style source.

## Morphology Policy

IO does not currently import `Cerebellum_mod/IO/morphology/IO.swc` in this
template.  Both backends manually create a single soma:

- length: `20 um`
- diameter: `20 um`
- segments/CVs: `1`

This keeps the first formal template focused on channel translation and
single-soma dynamics.  If a later IO template uses the SWC morphology, update
this document and split that implementation from the single-soma template.

## Parameter Layout

Use one top-level `IOParameters` object with four groups:

- `soma`: manual soma geometry.
- `channel`: ZH2019 channel conductance/reversal-style parameters.
- `cable`: passive cable and leak baseline.
- `ion`: fixed Na/K reversal potentials.

Parameter values remain plain floats.  BrainCell writes units explicitly at
paint sites, for example `ch.na_gbar_mS_cm2 * (u.mS / u.cm**2)`.

## Mechanism Mapping

| Logical mechanism | NEURON mod | BrainCell registry | Notes |
| --- | --- | --- | --- |
| Leak | `pas` | `IL` | Debug/formal passive baseline, not an IO `.mod`. |
| Na | `Na_ZH19_IO` | `Na_ZH2019_IO` | Uses fixed `na` ion, `ena=55 mV`. |
| Kdr | `Kdr_ZH19_IO` | `Kdr_ZH2019_IO` | Uses fixed `k` ion, `ek=-75 mV`. |
| Ca | `Ca_ZH19_IO` | `Ca_ZH2019_IO` | Nonspecific current with explicit `E=120 mV`. |
| HCN | `HCN_ZH19_IO` | `HCN_ZH2019_IO` | Nonspecific current with explicit `E=-43 mV`. |

`io_gap.mod` is not included in this single-soma formal template because it is
a point-process gap junction mechanism requiring another voltage endpoint.

## BrainCell Assembly Flow

The formal BrainCell class follows this order:

1. Build the manual soma morphology with `Branch.from_lengths`.
2. Create `Cell` with `CVPerBranchList((1,))`.
3. Define the `soma` region.
4. Paint cable properties.
5. Paint fixed Na/K ions.
6. Paint leak and four IO channels.

Keep the assembly close to `io_neuron.py`; avoid broad helper abstractions until
there is real repeated structure to remove.
