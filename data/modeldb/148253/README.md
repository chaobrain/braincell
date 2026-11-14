# ModelDB 148253: Jedlicka et al. 2011

## Source

- ModelDB: https://modeldb.science/148253
- GitHub mirror: https://github.com/ModelDBRepository/148253 at commit
  `1d737a0da00bdd3522fcd95894436955b3d1e980` (2019-05-31)
- Retrieved 2026-11-14 from
  `https://raw.githubusercontent.com/ModelDBRepository/148253/1d737a0da00bdd3522fcd95894436955b3d1e980/<file>`;
  stored unmodified.

| File | SHA-256 |
|---|---|
| `cldif.mod` | `077dc33c1f42fa85c67f6a5b3408844fd73c268f97c9c09620612b1759aa24ee` |
| `gabaA_Cl.mod` | `6a7e8e810f02c2d9de67b18e158afc54e0971f5d7d5aafe3711244940fec1e24` |

## License

No license statement: the ModelDB entry, its mirror and these files carry
none. They are committed under the maintainers' decision of 2026-11-13
(MECH-48, D13), for the comparisons below only; braincell does not relicense
them.

## Reference

Jedlicka P, Deller T, Gutkin BS, Backus KH (2011). Activity-dependent
intracellular chloride accumulation and diffusion controls GABAA
receptor-mediated synaptic transmission. Hippocampus 21:885-898.

## Use

- `cldif.mod` (`cldifus`): `Cldif_JE2011`, in `validation/neuron/ion/chloride_test.py`.
- `gabaA_Cl.mod` (`gaba`): `GabaA_JE2011` with `Cldif_JE2011`, in
  `validation/neuron/ion/gaba_test.py`.
