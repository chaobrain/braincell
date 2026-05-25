# NeuroMorpho.Org

[NeuroMorpho.Org](https://neuromorpho.org) is the largest public repository of
digitally reconstructed neurons — over 250,000 cells from hundreds of labs.
`braincell` ships a full client that searches, downloads, **caches**, and parses
these reconstructions into ready-to-use morphologies.

The client requires the IO extra:

```bash
pip install -U braincell[io]
```

## One-liner: load a neuron by id

The simplest path is to load a neuron directly by its NeuroMorpho id. The file
is downloaded once and cached locally, so subsequent calls are instant:

```python
import braincell

morpho = braincell.Morphology.from_neuromorpho(12345)
```

Equivalently, via the IO module entry point:

```python
import braincell.io as io

morpho = io.load_neuromorpho(12345)

# also get the validation report
morpho, report = io.load_neuromorpho(12345, return_report=True)
```

```{note}
The argument is the **integer neuron id** from NeuroMorpho.Org (the number in a
neuron's URL), not its name.
```

## Searching the repository

To find neurons matching criteria, use the
{class}`~braincell.io.NeuroMorphoClient`:

```python
from braincell.io import NeuroMorphoClient

client = NeuroMorphoClient()

# iterate search results (species, brain region, cell type, …)
for neuron in client.iter_search(species="mouse", brain_region="neocortex"):
    print(neuron.neuron_id, neuron.neuron_name)

# describe a single neuron
detail = client.get_neuron(12345)

# download its reconstruction file(s)
record = client.download(12345)
```

Search and query construction are typed through
{class}`~braincell.io.NeuroMorphoQuery` and return rich models
({class}`~braincell.io.NeuroMorphoNeuron`,
{class}`~braincell.io.NeuroMorphoMeasurement`,
{class}`~braincell.io.NeuroMorphoSearchPage`).

## Downloading without parsing

To fetch raw files (standard and/or original SWC) without immediately building a
morphology:

```python
import braincell.io as io

record = io.fetch_neuromorpho(12345, mode="both")   # 'standard' | 'original' | 'both'
print(record)   # NeuroMorphoDownloadRecord: what was written where
```

## Caching

Downloads are cached under a user cache directory
(`braincell.io.DEFAULT_USER_CACHE_DIR`) and managed by
{class}`~braincell.io.NeuroMorphoCache`. You can point any entry point at a
custom location with `cache_dir=...`, and inspect cache state through the
client's `get_cache_status`.

## Command-line interface

The client is also exposed as a CLI, `braincell-neuromorpho`, for searching and
downloading from a shell:

```bash
braincell-neuromorpho --help
```

## See also

- {doc}`../concepts/morphology` — what you get back.
- {doc}`../apis/io` — the complete NeuroMorpho client API.
