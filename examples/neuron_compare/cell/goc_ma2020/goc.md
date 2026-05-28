# GoC mechanism table

区域划分按 `goc_neuron_debug.py` / `goc_parameters.py` 当前实现：

- `Soma`
- `Dend_apical`
- `Dend_basal`
- `Axon_AIS`
- `Axon_regular`

| GoC | Soma | Dend_apical | Dend_basal | Axon_AIS | Axon_regular |
| --- | --- | --- | --- | --- | --- |
| Leak | ✓ | ✓ | ✓ | ✓ | ✓ |
| HCN1 |  |  |  | ✓ |  |
| HCN2 |  |  |  | ✓ |  |
| Nav1.6 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kv1.1 | ✓ |  |  |  |  |
| Kv3.4 | ✓ |  |  |  | ✓ |
| Kv4.3 | ✓ |  |  |  |  |
| KM |  |  |  | ✓ |  |
| Kca1.1 | ✓ | ✓ | ✓ | ✓ |  |
| Kca2.2 |  | ✓ | ✓ |  |  |
| Kca3.1 | ✓ |  |  |  |  |
| CaHVA | ✓ |  | ✓ | ✓ |  |
| Cav2.3 |  | ✓ |  |  |  |
| Cav3.1 | ✓ | ✓ |  |  |  |
| CdpStC | ✓ | ✓ | ✓ | ✓ | ✓ |
| Na_ion | ✓ | ✓ | ✓ | ✓ | ✓ |
| K_ion | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ca_ion | ✓ | ✓ | ✓ | ✓ | ✓ |
