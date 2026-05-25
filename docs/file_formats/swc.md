# SWC

[SWC](https://swc-specification.readthedocs.io/) is the most widely used
morphology format: a plain-text table where each row is a sample point with an
id, a structure type, `(x, y, z)` coordinates, a radius, and a parent id. It is
the default export of most reconstruction tools and the primary download format
on NeuroMorpho.Org.

## Loading

```python
import braincell

morpho = braincell.Morphology.from_swc("neuron.swc")
```

`from_swc` accepts:

- `path` — the file to read.
- `options` — a {class}`~braincell.io.SwcReadOptions` controlling validation.
- `mode` — a named validation preset.
- `return_report` — when `True`, also return an {class}`~braincell.io.SwcReport`.

```python
morpho, report = braincell.Morphology.from_swc("neuron.swc", return_report=True)
```

## Structure types

SWC structure identifiers map onto `braincell`'s typed branches:

| SWC id | Structure | braincell type |
|--------|-----------|----------------|
| 1 | soma | {class}`~braincell.Soma` |
| 2 | axon | {class}`~braincell.Axon` |
| 3 | (basal) dendrite | {class}`~braincell.BasalDendrite` |
| 4 | apical dendrite | {class}`~braincell.ApicalDendrite` |
| other | custom | {class}`~braincell.CustomBranch` |

So once an SWC file is loaded, region selectors like
`branch_in("type", "soma")` work immediately (see
{doc}`../concepts/regions_locsets`).

## Validation options

{class}`~braincell.io.SwcReadOptions` controls how forgiving the reader is:

```python
from braincell.io import SwcReadOptions

opts = SwcReadOptions(
    standardize_safe_fixes=True,   # apply safe automatic corrections
    unknown_type_as_custom=True,   # don't error on unknown structure ids
    require_root_type_soma=False,  # allow non-soma roots
)
morpho = braincell.Morphology.from_swc("neuron.swc", options=opts)
```

## The lower-level reader

`Morphology.from_swc` is a convenience wrapper over
{class}`braincell.io.SwcReader`, which you can use directly when you want to
*check* a file without building a morphology:

```python
from braincell.io import SwcReader

reader = SwcReader()
report = reader.check("neuron.swc")   # validate only
morpho = reader.read("neuron.swc")    # parse to a Morphology
```

See {doc}`../apis/io` for the full reader, options, and report API.
