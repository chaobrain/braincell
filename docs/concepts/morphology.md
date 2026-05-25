# Morphology

A *morphology* is the geometry of a neuron: the branching tree of cable
segments, each with a length, radius, and position in space. In `braincell`
this is represented by {class}`braincell.Morphology`, built from
{class}`braincell.Branch` segments.

A morphology is required only for multi-compartment {class}`~braincell.Cell`
models. Single-compartment models have no geometry.

## The pieces

```{list-table}
:header-rows: 1
:widths: 28 72

* - Object
  - Role
* - {class}`braincell.Branch`
  - An **immutable** geometric cable segment: sample points (x, y, z, radius)
    and the topology connecting them. Typed subclasses tag what part of the
    neuron a branch is.
* - {class}`braincell.Morphology`
  - The **tree** of branches with parent/child connectivity. The object you
    pass to {class}`~braincell.Cell`.
```

### Typed branches

Branches carry a *type* so that mechanisms can target anatomically meaningful
regions ("paint sodium channels on the soma"). The built-in types are:

- {class}`~braincell.Soma`
- {class}`~braincell.Dendrite`, with {class}`~braincell.BasalDendrite` and
  {class}`~braincell.ApicalDendrite`
- {class}`~braincell.Axon`
- {class}`~braincell.CustomBranch` for anything else

These map onto the standard SWC structure identifiers, so loading an SWC file
assigns types automatically.

## Where morphologies come from

You almost never type coordinates by hand. Instead you load a reconstruction:

```python
import braincell

# from a local SWC file
morpho = braincell.Morphology.from_swc("neuron.swc")

# from a Neurolucida ASC file
morpho = braincell.Morphology.from_asc("neuron.asc")

# directly from NeuroMorpho.Org (downloaded and cached)
morpho = braincell.Morphology.from_neuromorpho("cnic_001")
```

See {doc}`../file_formats/index` for the full set of readers, reader options,
and validation reports.

## Inspecting and visualizing

Once loaded, a morphology can be explored and rendered:

```python
import braincell.vis as vis

vis.plot2d(morpho)   # 2-D dendrogram / tree layout (matplotlib)
vis.plot3d(morpho)   # 3-D rendering (PyVista or Plotly)
```

The {mod}`braincell.vis` layer also provides morphometry plots (Sholl analysis,
branch-order histograms, topology) — see the {doc}`../apis/vis` reference.

## The mutable analysis tree (`braincell.morph`)

{class}`braincell.Morphology` is the immutable object you simulate. For
programmatic construction and analysis there is a richer, mutable tree in
{mod}`braincell.morph`:

- {class}`~braincell.morph.MorphoBranch` / {class}`~braincell.morph.MorphoEdge`
  — editable branch and edge views.
- {class}`~braincell.morph.MorphoMetric` — a whole-morphology metric snapshot
  (total length, surface area, branch counts, …).

Use {mod}`braincell.morph` when you need to *modify* or *measure* a
reconstruction; use {class}`~braincell.Morphology` when you are ready to build a
cell.

## See also

- {doc}`discretization` — how the continuous geometry becomes control volumes.
- {doc}`regions_locsets` — selecting parts of a morphology to decorate.
- {doc}`../file_formats/index` — loading and saving morphologies.
