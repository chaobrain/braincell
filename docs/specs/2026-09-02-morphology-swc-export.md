# Morphology SWC Export

## Goal

Add `Morphology.to_swc(path)` for writing morphologies with complete point
geometry to a standard seven-column SWC file.

## Export Contract

- Every branch must provide proximal and distal point geometry. Export fails
  before touching the destination when any branch lacks points.
- Coordinates and radii are written in micrometres with stable, consecutive,
  one-based sample IDs and parent-before-child ordering.
- Branch types map to SWC as `custom=0`, `soma=1`, `axon=2`,
  `dendrite/basal_dendrite=3`, and `apical_dendrite=4`.
- A child attached through `child_x=1` is emitted in reverse point order.
- Every branch emits its complete point path, including a duplicate attachment
  sample that the reader can merge with its parent.
- Ordinary non-soma attachments require the child endpoint to match the
  declared parent endpoint in both position and radius. Soma endpoint
  attachments require matching positions but keep the child's radius.
- Soma `parent_x=0.5` attachments use the existing internal soma sample nearest
  the arc midpoint without validating the child endpoint. Export rejects a
  midpoint attachment when the soma has no internal sample.
- A `parent_x=0` child may attach to its parent's second point only for the
  `con2prox` shape: the parent is directly below soma, its first point can merge
  into the soma anchor, and the child endpoint exactly matches its second point.
- Unrepresentable attachments fail before the destination is touched.
- A suffixless path receives `.swc`; parent directories are created and an
  existing destination is atomically replaced.

## Format Limits

SWC does not preserve branch names or explicit boundaries between collinear,
same-type branches. Generic `dendrite` uses SWC type 3 and therefore imports as
`basal_dendrite`. The verbose attachment rows may produce safe duplicate-node
warnings. Morphologies that satisfy the export contract round-trip exactly
through the default `mode="neuron"` reader; other reader modes are not part of
this guarantee.

## Verification

Cover type mapping, verbose attachment samples, soma midpoint selection,
`con2prox`, reversed children, radius jumps, invalid-attachment rejection,
suffix and atomic write behaviour, and exact read-back of the bundled `grc`
and `io` fixtures through `SwcReader`.
