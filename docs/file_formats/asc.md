# Neurolucida ASC

The Neurolucida ASCII format (`.asc`), produced by MBF Bioscience's Neurolucida
tracing software, stores morphologies as nested S-expressions with rich
metadata (spines, markers, contours). `braincell` reads it through
{class}`braincell.io.AscReader`.

## Loading

```python
import braincell

morpho = braincell.Morphology.from_asc("neuron.asc")
```

As with SWC, you can request a validation report:

```python
morpho, report = braincell.Morphology.from_asc("neuron.asc", return_report=True)
print(report)   # AscReport: issues found and metadata
```

## What the reader captures

The ASC reader understands the format's richer structure than SWC:

- typed branches (soma, axon, basal/apical dendrite) → `braincell` branch types;
- {class}`~braincell.io.AscSpineRecord` — dendritic spine annotations;
- {class}`~braincell.io.AscMetadata` — file-level metadata;
- {class}`~braincell.io.AscReport` / {class}`~braincell.io.AscIssue` — the
  validation report and any problems encountered.

## Lower-level reader

```python
from braincell.io import AscReader

reader = AscReader()
morpho = reader.read("neuron.asc")
```

See {doc}`../apis/io` for the full ASC API. For the simpler, more portable
format, see {doc}`swc`.
