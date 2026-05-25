# Installation

`braincell` is a pure-Python package built on [JAX](https://github.com/jax-ml/jax)
and [brainstate](https://github.com/chaobrain/brainstate). It runs on Linux,
macOS, and Windows (via WSL), and requires **Python 3.10 or newer**.

## Quick install

The fastest way to get a working CPU install:

```bash
pip install -U braincell[cpu]
```

This pulls in `braincell` together with a CPU build of JAX.

## Choosing a hardware backend

`braincell` inherits JAX's hardware support. Pick the extra that matches your
accelerator:

````{tab-set}
```{tab-item} CPU
    pip install -U braincell[cpu]
```

```{tab-item} GPU (CUDA)
    pip install -U braincell[cuda12]

    # or, for CUDA 13
    pip install -U braincell[cuda13]
```

```{tab-item} TPU
    pip install -U braincell[tpu]
```
````

```{note}
If you install plain `braincell` without a hardware extra and no JAX build is
present, JAX falls back to CPU. If you have an NVIDIA GPU but installed only the
CPU build, you will see a message like *"An NVIDIA GPU may be present … falling
back to cpu"* — install the matching `cuda` extra to use the GPU.
```

## Optional feature groups

Several capabilities depend on optional libraries, grouped as install extras:

| Extra | Installs | Enables |
|-------|----------|---------|
| `braincell[vis]` | matplotlib, pyvista, plotly | 2-D and 3-D visualization ({mod}`braincell.vis`) |
| `braincell[io]` | requests | the NeuroMorpho.Org client ({mod}`braincell.io`) |
| `braincell[all]` | all of the above | every optional feature |

Extras combine, so you can request several at once:

```bash
pip install -U "braincell[cpu,vis,io]"
```

## Install with the full ecosystem

`braincell` is part of the [BrainX ecosystem](https://brainx.chaobrain.com/).
To get `braincell` together with the rest of the stack:

```bash
pip install -U BrainX
```

## Development install

To work on `braincell` itself, clone the repository and install it in editable
mode with the testing extras:

```bash
git clone https://github.com/chaobrain/braincell.git
cd braincell
pip install -e ".[testing]"
```

Then install the pre-commit hooks and run the test suite to confirm everything
works:

```bash
pre-commit install
pytest braincell/
```

See the {doc}`Developer Guide <../developer/index>` for the project layout,
testing conventions, and contribution workflow.

## Verifying the install

```python
import braincell
print(braincell.__version__)
```

A successful import that prints a version string means you are ready to go.
Continue to {doc}`overview` for a map of the library, or jump straight to
{doc}`first_steps` to run your first model.

## Windows + NEURON (optional)

Some validation examples compare `braincell` against the
[NEURON](https://neuron.yale.edu) simulator. If you are on Windows and want a
WSL-based development setup with `NEURON 8.2.6` and `nrnivmodl`, follow the
notes in `develop_doc/windows_wsl_neuron_setup.md` in the repository. NEURON is
**not** required to use `braincell`.
