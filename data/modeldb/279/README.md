# ModelDB 279: Destexhe et al. 1998

## Source

- ModelDB: https://modeldb.science/279
- GitHub mirror: https://github.com/ModelDBRepository/279 at commit
  `31424439f64771344c28105e80093eecf1fc0b64` (2023-12-01); `ITGHK.mod` last
  changed in `69f1de2436155b6ead7a37c8b500dad82b2db1e0` (2011-04-11)
- Retrieved 2026-11-14 from a clone of the mirror at that commit; stored
  unmodified.

| File | SHA-256 |
|---|---|
| `ITGHK.mod` | `fa71f75d0b5440f79bb080f1b4dce7a77bc04ff8c657c95eca7f01739111d394` |

## License

No license statement: the ModelDB entry, its mirror and this file carry
none. It is committed under the maintainers' decision of 2026-11-13
(MECH-48, D13), for the comparison below only; braincell does not relicense
it.

## Reference

Destexhe A, Neubig M, Ulrich D, Huguenard JR (1998). Dendritic low-threshold
calcium currents in thalamic relay cells. J Neurosci 18:3574-3588. The file
implements the T current of Huguenard & McCormick (1992), J Neurophysiol
68:1373-1383, with a GHK drive.

## Use

`ITGHK.mod` (`itGHK`): `braincell.channel.CaT_HM1992_GHK`, in the
configuration `validation/neuron/channel_no_conc/configs/modeldb279/cat_hm1992_ghk.json`
(`identity.pinned_mod_files`; compiled into the `testing` library by
`python -m validation.neuron._mechanisms testing --kind channel`).
