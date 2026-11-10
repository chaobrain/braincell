# Synthetic reference mechanisms

`Toy*.mod` are the synthetic ion/kinetic fixtures previously stored with DCN.
`kv.mod` is the potassium-channel fixture (`Kv`, gate `n`). Its NEURON
comparison against `braincell.channel.K_Kv_test` is the channel_no_conc
configuration `validation/neuron/channel_no_conc/configs/testing/kv_testing.json`;
the converter examples in `examples/convert_mod/mod_validate/` load this library.

Compile the validation library from the repository root:

```bash
python -m validation.neuron._mechanisms testing --kind channel
```

The library is written to `validation/neuron/artifacts/mechanisms/testing/channel/`.
Ion notebooks compile only their requested Toy mechanisms in isolated subsets.
