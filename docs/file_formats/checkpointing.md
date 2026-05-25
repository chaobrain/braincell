# Checkpointing

Once you have loaded and possibly edited a morphology, you can persist it to
disk and reload it later without re-parsing the original reconstruction. This is
useful for caching a cleaned-up morphology or sharing a processed cell.

## Saving and loading a morphology

```python
import braincell
import braincell.io as io

morpho = braincell.Morphology.from_swc("neuron.swc")

path = io.save_morpho(morpho, "neuron.bcm")   # returns the written Path
morpho = io.load_morpho("neuron.bcm")
```

For a single branch there are matching helpers:

```python
io.save_branch(branch, "branch.bcb")
branch = io.load_branch("branch.bcb")
```

## Versioning and errors

Checkpoints are versioned. If you load a file written by an incompatible
version, the loaders raise a clear error:

- {class}`~braincell.io.CheckpointError` — a checkpoint could not be read.
- {class}`~braincell.io.CheckpointVersionError` — the checkpoint's format
  version is not supported by the installed `braincell`.

```python
from braincell.io import CheckpointVersionError

try:
    morpho = io.load_morpho("old.bcm")
except CheckpointVersionError as e:
    print("Re-export from the source file:", e)
```

## See also

- {doc}`overview` — loading from the original formats.
- {doc}`../apis/io` — the full checkpoint API.
