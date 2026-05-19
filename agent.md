# Agent Notes

- When recompiling NEURON mod mechanisms under `examples/neuron_compare/Cerebellum_mod/DCN/x86_64`, do not use the conda toolchain from the active `braincell` environment.
- Override the compiler environment to system binaries first:
  - `CPP=/usr/bin/cpp`
  - `CC=/usr/bin/cc`
  - `CXX=/usr/bin/c++`
- A known-good rebuild command is:

```bash
CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ \
make -j4 \
  -f /home/swl/anaconda3/envs/braincell/lib/python3.10/site-packages/neuron/.data/bin/nrnmech_makefile \
  ROOT=/home/swl/anaconda3/envs/braincell/lib/python3.10/site-packages/neuron/.data \
  MODOBJFILES='./CdpHVA_SU15_DCN.o ./CdpLVA_SU15_DCN.o ./CaHVA_SU15_DCN.o ./CaLVA_SU15_DCN.o' \
  UserLDFLAGS='' UserINCFLAGS='' LinkCoreNEURON=false special
```

- This was required so `CdpLVA_SU15_DCN` and `CaLVA_SU15_DCN` could compile and load successfully with the current local NEURON installation.
