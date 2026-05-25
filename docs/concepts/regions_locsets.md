# Regions & Locsets

When you decorate a multi-compartment cell you must say **where** a mechanism
goes. `braincell` answers this with two kinds of selection expression, both
living in {mod}`braincell.filter`:

- a **region** selects an *extended set of cable* — "the soma", "all apical
  dendrites", "everything within 100 µm of the root". You `paint` density
  mechanisms onto regions.
- a **locset** selects a *set of points* — "the root", "all branch tips",
  "every 50 µm". You `place` point mechanisms onto locsets.

```python
import braincell.filter as f
```

```{note}
This mirrors the *labels* idea in simulators like Arbor: a small, composable
algebra for naming parts of a morphology, kept separate from the mechanisms you
attach to them.
```

## Regions (paint targets)

```{list-table}
:header-rows: 1
:widths: 34 66

* - Expression
  - Selects
* - {class}`~braincell.filter.AllRegion`
  - the entire cell
* - {class}`~braincell.filter.EmptyRegion`
  - nothing (useful as an identity in set operations)
* - {func}`~braincell.filter.branch_in`
  - branches matching a property, e.g. `branch_in("type", "soma")`
* - {func}`~braincell.filter.branch_range` / {class}`~braincell.filter.BranchRangeFilter`
  - a sub-interval along branches
* - {class}`~braincell.filter.SubtreeRegion`
  - a branch and everything distal to it
* - {class}`~braincell.filter.RadiusRangeRegion`
  - cable whose radius falls in a range
* - {class}`~braincell.filter.EuclideanDistanceRegion` / {class}`~braincell.filter.TreeDistanceRegion`
  - cable within a straight-line / along-the-tree distance
```

```python
from braincell.filter import AllRegion, branch_in

cell.paint(AllRegion(), mech.Channel("IL", g_max=0.0003 * u.S / u.cm**2, E=-70. * u.mV))
cell.paint(branch_in("type", "soma"), mech.Channel("Na_Ba2002", g_max=0.12 * u.S / u.cm**2))
cell.paint(
    branch_in("type", ("dendrite", "basal_dendrite", "apical_dendrite")),
    mech.Channel("CaL_IS2008", g_max=0.002 * u.S / u.cm**2),
)
```

## Locsets (place targets)

```{list-table}
:header-rows: 1
:widths: 34 66

* - Expression
  - Selects
* - {class}`~braincell.filter.RootLocation`
  - a point on the root branch, e.g. `RootLocation(0.5)` (the midpoint)
* - {class}`~braincell.filter.Terminals`
  - all terminal tips
* - {class}`~braincell.filter.BranchPoints`
  - all branch points (bifurcations)
* - {class}`~braincell.filter.UniformSamples` / {class}`~braincell.filter.StepSamples`
  - evenly / step-spaced points along the cell
* - {class}`~braincell.filter.RandomSamples`
  - randomly sampled points
* - {func}`~braincell.filter.at` / {class}`~braincell.filter.AtLocation`
  - a specific (branch, position) location
```

```python
from braincell.filter import RootLocation, Terminals

cell.place(RootLocation(0.5), mech.CurrentClamp.step(0.2 * u.nA, duration=50 * u.ms))
cell.place(Terminals(), mech.StateProbe("V"))
```

## Composing selections

Both regions and locsets support **set operations** — union, intersection,
difference — so you can build precise targets from simple parts
({class}`~braincell.filter.RegionSetOp`, {class}`~braincell.filter.LocsetSetOp`,
and the mask variants). This is the same composability that makes mechanisms
reusable: name a place once, decorate it many times.

`braincell.filter` also caches selection results
({class}`~braincell.filter.SelectionCache`) so repeated paints over the same
region don't recompute the membership.

## See also

- {doc}`mechanisms` — the things you paint and place.
- {doc}`morphology` — what the selections range over.
- {doc}`../apis/filter` — full selection API reference.
