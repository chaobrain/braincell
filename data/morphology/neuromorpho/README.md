# NeuroMorpho.Org regression neurons

Seven neurons from [NeuroMorpho.Org](https://neuromorpho.org) in the
`NeuroMorphoCache` layout: one folder per NeuroMorpho id, holding the
standardized SWC (`<name>.CNG.swc`) and the `metadata.json` that
`NeuroMorphoClient().download(id, output_dir=..., mode="standard")` writes.
`metadata.json` keeps NeuroMorpho's `neuron` and `measurement` payloads
unchanged; `measurement` is the published morphometry.

`braincell/io/neuromorpho/regression_test.py` compares BrainCell's metrics
with that morphometry offline, and `examples/io/neuromorpho.ipynb` reads this
directory by default.

## Source

Retrieved on 2026-11-12 (`2026-11-11T19:27:19Z`) with
`NeuroMorphoClient.download`. The SWC URLs are under
`https://neuromorpho.org/dableFiles/`; the morphometry is
`https://neuromorpho.org/api/morphometry/id/<id>` and the neuron payload
`https://neuromorpho.org/api/neuron/id/<id>`. The files are stored byte for
byte (`.gitattributes` disables line-ending conversion here), so the SHA-256
values below hold for the checked-out files.

| id | Name | Archive | Species, cell type | SWC (after `dableFiles/`) | SWC SHA-256 | Original paper |
|---|---|---|---|---|---|---|
| 102085 | 01042017e6-000 | Nandi | mouse, dentate gyrus granule cell | `nandi/CNG%20version/01042017e6-000.CNG.swc` | `5d09c3b987329b57563bd5003cafb010c41d55f149486d53d401b58bfa277575` | doi:10.1016/j.neuroscience.2017.11.017 |
| 10074 | Purkinje-slice-ageP35-3 | Dusart | mouse, cerebellar Purkinje cell | `dusart/CNG%20version/Purkinje-slice-ageP35-3.CNG.swc` | `98ee4f752385b119c7532cce71cf5ef053030312bb9a1b5b15c64b2661f5a48c` | doi:10.1523/JNEUROSCI.2977-12.2013 |
| 107134 | Fig3i_Rat_81 | Tamas | rat, neocortical basket cell | `tamas/CNG%20version/Fig3i_Rat_81.CNG.swc` | `5cfe270e49e51f7615cdd3d807b59389f9b489727e7f6cbf1c9d50b2c6fdd2a2` | doi:10.1016/j.neuron.2016.09.032 |
| 122127 | AA0990 | MouseLight | mouse, cerebellar Purkinje cell | `mouselight/CNG%20version/AA0990.CNG.swc` | `016b1c8f1621371501025b39311159a7838e62a5274d244342b14d43bad74c76` | doi:10.2139/ssrn.3330557 |
| 100 | n419 | Turner | rat, CA1 pyramidal cell | `turner/CNG%20version/n419.CNG.swc` | `39711b83c3793466ec13c90fcbe68e25df9b6d661ea2364eca5930b48b492af6` | doi:10.1002/(SICI)1096-9861(19980216)391:3<335::AID-CNE4>3.0.CO;2-2 |
| 1013 | n10fts | Ascoli | rat, hippocampal pyramidal cell | `ascoli/CNG%20version/n10fts.CNG.swc` | `26247c9d947ace7d7bb529c0d174a649dcd34a790d3877be7bfc418858fff68f` | doi:10.2307/4134575; doi:10.1038/nn1132 |
| 10036 | 5-27-11cell1-nonFS-BC | Santhakumar | rat, dentate gyrus basket cell | `santhakumar/CNG%20version/5-27-11cell1-nonFS-BC.CNG.swc` | `cf59917cae264e397522be011e63701af5903f41f2a7fe6d7f3086e201ef60f0` | doi:10.1152/jn.00891.2012 |

If a re-download changes a hash, re-measure the regression before replacing
the file: its snapshot and known-exception values are BrainCell's numbers for
these exact files.

## License and citation

NeuroMorpho.Org's terms of use (https://neuromorpho.org/useterm.jsp,
retrieved 2026-11-11): "NeuroMorpho.Org is licensed under a Creative Commons
Attribution 4.0 International License." The data here are redistributed under
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The terms ask any
publication that uses material from NeuroMorpho.Org to include all three
citations:

1. The original paper(s) describing each reconstruction (table above).
2. NeuroMorpho.Org (RRID:SCR_002145).
3. Tecuatl C, Ljungquist B, Ascoli GA (2024) Accelerating the continuous
   community sharing of digital neuromorphology data. FASEB Bioadv.
   6(7):207-221. doi:10.1096/fba.2024-00048
