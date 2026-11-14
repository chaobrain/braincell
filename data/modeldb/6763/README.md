# ModelDB 6763: Canavier 1999

## Source

- ModelDB: https://modeldb.science/6763
- GitHub mirror: https://github.com/ModelDBRepository/6763 at commit
  `ccdc2f0083c0ffdbba1f1a8fd8b68d391bff5cc1` (2023-11-28)
- Retrieved 2026-11-14 from
  `https://raw.githubusercontent.com/ModelDBRepository/6763/ccdc2f0083c0ffdbba1f1a8fd8b68d391bff5cc1/<file>`;
  stored unmodified.

| File | SHA-256 |
|---|---|
| `leak.mod` | `71d5171938be424cf9b266f2d1fade561f2e70a89f43c71dbdd1f1cde8e6fad9` |
| `nadifl.mod` | `26d1903892222e2b7e6bfdd09c3640693c35b19d4f6d496f5498d7dc5d66ab48` |
| `pump.mod` | `d1fcacba4f1ca195f8239830d0a63066fc9ebbd2865f31e56ceb914bc5591738` |

## License

No license statement: the ModelDB entry, its mirror and these files carry
none. They are committed under the maintainers' decision of 2026-11-13
(MECH-48, D13), for the comparisons below only; braincell does not relicense
them.

## Reference

Canavier CC (1999). Sodium dynamics underlying burst firing and putative
mechanisms for the regulation of the firing pattern in midbrain dopamine
neurons: a computational approach. J Comput Neurosci 6:49-69.

## Use

The configurations `validation/neuron/ion/configs/ca1999/sodium_detailed_pump_off.json`
(`nadifl`, `leak`: `SodiumDetailed`, `Na_Leak`, `K_Leak`) and
`sodium_detailed_pump.json` (adds `pump`: `NaKPump_Ca1999`), in
`validation/neuron/ion/sodium_test.py`.
