# NeuroML2

[NeuroML2](https://neuroml.org/) is an XML-based, community standard for
describing neuronal models — including detailed morphologies — in a
simulator-independent way. `braincell` reads the morphology portion through
{class}`braincell.io.NeuroMlReader`.

## Loading

```python
from braincell.io import NeuroMlReader

reader = NeuroMlReader()
morpho = reader.read("neuron.cell.nml")
```

The reader extracts the segment groups and cable geometry from the NeuroML2
document and returns a standard {class}`braincell.Morphology`, which you then
decorate and simulate like any other.

```{note}
`braincell` reads NeuroML2 **morphology**. Channel and network definitions in a
NeuroML2 document are not imported automatically — declare mechanisms with
{mod}`braincell.mech` as usual (see {doc}`../concepts/mechanisms`).
```

## See also

- {doc}`swc` and {doc}`asc` — the other supported morphology formats.
- {doc}`../apis/io` — the full IO reference.
