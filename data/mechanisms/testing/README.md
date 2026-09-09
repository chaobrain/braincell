# Synthetic reference mechanisms

`Toy*.mod` are the synthetic ion/kinetic fixtures previously stored with DCN.
`kv.mod` is the potassium-channel validation fixture, copied from the converter
example's `mod_validate/mods/kv.mod` (the converter example retains its own input).

Compile the validation library from the repository root:

```bash
python -m validation.neuron._mechanisms testing --kind channel
```

The library is written to `validation/neuron/artifacts/mechanisms/testing/channel/`.
Ion notebooks compile only their requested Toy mechanisms in isolated subsets.
