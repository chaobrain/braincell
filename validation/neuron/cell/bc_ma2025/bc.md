# BC mechanism table

区域划分按 `debug/bc_neuron_debug.py` / `debug/bc_parameters.py` 当前实现：

- `Soma`
- `Dend`
- `Axon_AIS`
- `Axon_regular`

| BC | Soma | Dend | Axon_AIS | Axon_regular |
| --- | --- | --- | --- | --- |
| Leak | ✓ | ✓ | ✓ | ✓ |
| Nav1.1 | ✓ |  |  |  |
| Nav1.6 |  |  | ✓ | ✓ |
| Cav1.2 | ✓ | ✓ |  |  |
| Cav1.3 | ✓ | ✓ |  |  |
| Cav2.1 |  |  | ✓ | ✓ |
| Cav3.2 | ✓ | ✓ |  |  |
| Kir2.3 | ✓ |  |  |  |
| Kv1.1 |  |  |  | ✓ |
| Kv3.4 | ✓ |  | ✓ | ✓ |
| Kv4.3 | ✓ | ✓ |  |  |
| Kca1.1 |  |  | ✓ | ✓ |
| Kca2.2 |  | ✓ |  |  |
| Kca3.1 | ✓ |  |  |  |
| HCN1 | ✓ |  | ✓ | ✓ |
| CdpStC | ✓ | ✓ | ✓ | ✓ |
| Na_ion | ✓ |  | ✓ | ✓ |
| K_ion | ✓ | ✓ | ✓ | ✓ |
| Ca_ion | ✓ | ✓ | ✓ | ✓ |
