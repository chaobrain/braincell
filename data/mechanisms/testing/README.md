# Synthetic reference mechanisms

`Toy*.mod` are the synthetic ion/kinetic fixtures previously stored with DCN.
`kv.mod` is the potassium-channel fixture (`Kv`, gate `n`). Its NEURON
comparison against `braincell.channel.K_Kv_test` is the channel_no_conc
configuration `validation/neuron/channel_no_conc/configs/testing/kv_testing.json`;
the converter examples in `examples/convert_mod/mod_validate/` load this library.
`clleak.mod` is the chloride-leak fixture (`clleak`, `icl = g (v - ecl)`), the
reference of `braincell.channel.Cl_Leak` in
`validation/neuron/channel_no_conc/configs/testing/cl_leak_testing.json`. It
only reads `ecl`, so NEURON computes the Nernst potential only under
`ion_style("cl_ion", 1, 2, 1, 1, 1)`, which the channel_no_conc runner sets.
`kleak.mod` is the potassium-leak fixture (`kleak`, `ik = g (v - ek)`), the
current that drives `braincell.ion.PotassiumDetailedFH` against Botta et al.'s
(2010) `K_conc.mod` in `validation/neuron/ion/configs/bo2010/`; that pool
writes `ki` and `ko`, so NEURON computes `ek` from them.

Compile the validation library from the repository root:

```bash
python -m validation.neuron._mechanisms testing --kind channel
```

The library is written to `validation/neuron/artifacts/mechanisms/testing/channel/`.
Ion notebooks compile only their requested Toy mechanisms in isolated subsets.

`Cav3p3_MA24_PC_finite.mod` and `Cav3p3_RI21_SC_finite.mod` are finite
references for the 0 mV voltage-clamp command (MECH-43). They are copies of
`data/cerebellum/pc_ma2024/mechanisms/channel/Cav3p3_MA24_PC.mod`
(SHA-256 `82781c90703880449bdffe2203339357d24b2bf8017993bc7efc2575809afaa6`)
and `data/cerebellum/sc_ma2021/mechanisms/channel/Cav3p3_RI21_SC.mod`
(SHA-256 `49109875a6b87dabb5ef8b20d795d0e1304bdd081f343f0abeec2cc59073f576`)
with two changes each:

- `SUFFIX Cav3p3_MA24_PC` becomes `SUFFIX Cav3p3_MA24_PC_finite`
  (`Cav3p3_RI21_SC` becomes `Cav3p3_RI21_SC_finite`), so the copy and the
  original can be loaded in one process;
- the GHK line of `rates()`,
  `ghk = -0.001*z*F*(cao-cai*exp(w))*w/(exp(w)-1)`, which is 0/0 at
  `w = 0`, becomes the small-`w` series branch that `Cav3p1_MA24_PC.mod`
  uses in its `ghk()`:

  ```
  if (fabs(exp(w)-1) < 1e-6) {
    ghk = -0.001*z*F*(cao-cai*exp(w))*(1-w/2)
  } else {
    ghk = -0.001*z*F*(cao-cai*exp(w))*w/(exp(w)-1)
  }
  ```

NMODL has no built-in `exprelr`, hence the branch. The channel_no_conc
configurations `configs/testing/cav3p3_*_finite.json` compare the BrainCell
Cav3.3 classes against these copies; `tests/test_coverage.py` rebuilds each
copy from its original and checks that the two are identical.
