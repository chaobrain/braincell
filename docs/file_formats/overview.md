# IO Overview

Loading a morphology is almost always a one-liner using a
`Morphology.from_*` constructor. Each reader parses a file (or a downloaded
record), validates it, and returns a {class}`braincell.Morphology` ready to wrap
in a {class}`~braincell.Cell`.

```python
import braincell

morpho = braincell.Morphology.from_swc("neuron.swc")
morpho = braincell.Morphology.from_asc("neuron.asc")
morpho = braincell.Morphology.from_neuromorpho(12345)   # NeuroMorpho.Org neuron id
```

## Validation reports

Reconstructions are messy: disconnected points, missing soma, unknown structure
identifiers. Instead of failing on the first problem (or silently guessing),
the SWC reader can hand back a **report** describing what it found and fixed.
Pass `return_report=True`:

```python
morpho, report = braincell.Morphology.from_swc("messy.swc", return_report=True)
print(report)        # SwcReport: issues, fixes applied, summary
```

The report types ({class}`~braincell.io.SwcReport`,
{class}`~braincell.io.SwcIssue`, and the ASC equivalents) let you decide
programmatically whether a reconstruction is clean enough to trust.

## Reader options

Readers accept an options object to control validation strictness. For SWC,
{class}`~braincell.io.SwcReadOptions` controls things like:

- `standardize_safe_fixes` — automatically apply safe corrections;
- `unknown_type_as_custom` — map unrecognized structure ids to
  {class}`~braincell.CustomBranch` instead of erroring;
- `require_root_type_soma` — enforce that the root is a soma.

```python
from braincell.io import SwcReadOptions

opts = SwcReadOptions(unknown_type_as_custom=True)
morpho = braincell.Morphology.from_swc("neuron.swc", options=opts)
```

## Saving and reloading

To persist a `braincell` morphology (after editing, say) and reload it later,
use the checkpoint helpers — see {doc}`checkpointing`.

## Where to go next

- {doc}`swc` · {doc}`asc` · {doc}`neuroml2` — per-format details.
- {doc}`neuromorpho` — searching and downloading from NeuroMorpho.Org.
- {doc}`../concepts/morphology` — what a morphology *is*, once loaded.
