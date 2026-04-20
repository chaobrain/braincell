# Cerebellum mod classification overview

本表仅覆盖当前 `channel/ion` 目录中的机制，不包含 `synapse/other`。

## Classification rule

- `HH_no_conc`: 以 `DERIVATIVE` 为主，且不显式依赖浓度变量。
- `Markov_no_conc`: 以 `KINETIC` 为主，且不显式依赖浓度变量。
- `Ion_dyn`: 独立的离子浓度 / 缓冲 / pump 动力学模块。
- `HH_conc`: 以 `DERIVATIVE` 为主，且显式依赖浓度变量。
- `Markov_conc`: 以 `KINETIC` 为主，且显式依赖浓度变量。
- 表格中展示名已经去掉作者年份与 cell 后缀，并把内部的 `p` 还原成 `.` 仅用于阅读；真实机制名以各 `.mod` 文件中的 `SUFFIX` 为准。

## Region totals

| Region | Count |
|---|---:|
| HH_no_conc | 40 |
| Markov_no_conc | 9 |
| Ion_dyn | 7 |
| HH_conc | 23 |
| Markov_conc | 10 |

## Cell totals

| Cell | Count |
|---|---:|
| BC | 15 |
| DCN | 11 |
| GoC | 16 |
| GrC | 13 |
| IO | 4 |
| PC | 16 |
| SC | 14 |

## HH_no_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| HCN1<br>Kir2.3<br>Kv1.1<br>Kv3.4<br>Kv4.3 | HCN<br>NaF<br>NaP<br>fKdr<br>sKdr | HCN1<br>HCN2<br>KM<br>Kv1.1<br>Kv3.4<br>Kv4.3<br>CaHVA<br>Cav2.3 | KM<br>Kir2.3<br>Kv1.1<br>Kv2.2_0010<br>Kv3.4<br>Kv4.3<br>CaHVA | HCN<br>Na<br>Kdr<br>Ca | HCN1<br>Kir2.3<br>Kv1.1<br>Kv3.4<br>Kv4.3 | HCN1<br>KM<br>Kir2.3<br>Kv1.1<br>Kv3.4<br>Kv4.3 |

## HH_no_conc template fit

Fit labels:

- `direct_hh`: current `HH` template is enough; `TABLE`/`FROM ... WITH ...` only affect NEURON-side caching.
- `hh_special_current`: gate dynamics fit the template, but current/root-type handling needs an extra override.
- `manual_structure`: convert the NMODL structure first, then write the BrainCell class.

### BC
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN1_MA25_BC` | `hh_special_current` | `USEION h`; map to `HHTypedNeuron` + explicit `E_rev`; gate has `Q10`. |
| `Kir2p3_MA25_BC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kv1p1_MA25_BC` | `hh_special_current` | Extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
| `Kv3p4_MA25_BC` | `direct_hh` | Piecewise `if/else` tau functions; translate with `u.math.where`. |
| `Kv4p3_MA25_BC` | `direct_hh` | Piecewise helper formulas; active branch uses `sigm`, and the unused `linoid` helper is only a stability reference. |

### DCN
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN_SU15_DCN` | `hh_special_current` | `NONSPECIFIC_CURRENT ih`; map to `HHTypedNeuron` instead of introducing an `h` ion. |
| `NaF_SU15_DCN` | `direct_hh` |  |
| `NaP_SU15_DCN` | `direct_hh` |  |
| `fKdr_SU15_DCN` | `direct_hh` |  |
| `sKdr_SU15_DCN` | `direct_hh` |  |

### GoC
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN1_MA20_GoC` | `hh_special_current` | Two independent open-state gates; current uses `o_fast + o_slow`, not a pure gate product. |
| `HCN2_MA20_GoC` | `hh_special_current` | Two independent open-state gates plus a clamped piecewise `r(v)` corridor. |
| `KM_MA20_GoC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kv1p1_MA20_GoC` | `hh_special_current` | Extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
| `Kv3p4_MA20_GoC` | `direct_hh` | Piecewise `if/else` tau functions; translate with `u.math.where`. |
| `Kv4p3_MA20_GoC` | `direct_hh` | Piecewise helper formulas; active branch uses `sigm`, and the unused `linoid` helper is only a stability reference. |
| `CaHVA_MA20_GoC` | `direct_hh` | `derivimplicit` in NEURON, but the two gate ODEs are still independent. |
| `Cav2p3_MA20_GoC` | `manual_structure` | Uses `inf[2]`, `tau[2]`, and `FROM i=0 TO 1`; split the indexed logic into explicit `m/h` functions first. |

### GrC
| SUFFIX | Fit | Special notes |
|---|---|---|
| `KM_MA20_GrC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kir2p3_MA20_GrC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kv1p1_MA20_GrC` | `hh_special_current` | Extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
| `Kv2p2_0010_MA20_GrC` | `direct_hh` | Contains `UNITSOFF/UNITSON`; translate with explicit dimensionless intermediates. |
| `Kv3p4_MA20_GrC` | `direct_hh` | Piecewise `if/else` tau functions; translate with `u.math.where`. |
| `Kv4p3_MA20_GrC` | `direct_hh` | Piecewise helper formulas; active branch uses `sigm`, and the unused `linoid` helper is only a stability reference. |
| `CaHVA_MA20_GrC` | `direct_hh` | `derivimplicit` in NEURON, but the two gate ODEs are still independent. |

### IO
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN_ZH19_IO` | `hh_special_current` | `NONSPECIFIC_CURRENT ih` with `UNITSOFF/UNITSON`; map to `HHTypedNeuron` and keep the explicit reversal potential. |
| `Na_ZH19_IO` | `direct_hh` | Uses small-denominator guards with `if`; translate to stable `u.math.where` branches. |
| `Kdr_ZH19_IO` | `direct_hh` | Uses small-denominator guards with `if`; translate to stable `u.math.where` branches. |
| `Ca_ZH19_IO` | `hh_special_current` | Calcium-shaped gating, but current is emitted as `NONSPECIFIC_CURRENT i` rather than through a `ca` ion payload. |

### PC
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN1_MA24_PC` | `hh_special_current` | `USEION h`; map to `HHTypedNeuron` + explicit `E_rev`; gate has `Q10`. |
| `Kir2p3_MA24_PC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kv1p1_MA24_PC` | `hh_special_current` | Extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
| `Kv3p4_MA24_PC` | `direct_hh` | Piecewise `if/else` tau functions; translate with `u.math.where`. |
| `Kv4p3_MA24_PC` | `direct_hh` | Piecewise helper formulas; active branch uses `sigm`, and the unused `linoid` helper is only a stability reference. |

### SC
| SUFFIX | Fit | Special notes |
|---|---|---|
| `HCN1_RI21_SC` | `hh_special_current` | `USEION h`; map to `HHTypedNeuron` + explicit `E_rev`; gate has `Q10`. |
| `KM_RI21_SC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kir2p3_RI21_SC` | `direct_hh` | `derivimplicit` in NEURON, but the gate ODE is still independent and can be written explicitly. |
| `Kv1p1_RI21_SC` | `hh_special_current` | Extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
| `Kv3p4_RI21_SC` | `direct_hh` | Piecewise `if/else` tau functions; translate with `u.math.where`. |
| `Kv4p3_RI21_SC` | `direct_hh` | Piecewise helper formulas; active branch uses `sigm`, and the unused `linoid` helper is only a stability reference. |

## Markov_no_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| Nav1.1<br>Nav1.6 | - | Nav1.6 | Nav<br>NaFHF | - | Nav1.6<br>Kv3.3 | Nav1.1<br>Nav1.6 |

## Ion_dyn

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| CdpStC | CdpHVA<br>CdpLVA | CdpStC | CdpCR | - | CdpCAM | CdpStC |

## HH_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| Kca3.1<br>Cav1.2<br>Cav1.3<br>Cav2.1<br>Cav3.2 | SK<br>CaHVA<br>CaL<br>CaLVA | Kca3.1<br>Cav1.2<br>Cav1.3<br>Cav3.1 | Kv1.5 | - | Kca3.1<br>Cav2.1<br>Cav3.1<br>Cav3.2<br>Cav3.3<br>Kv1.5 | Cav2.1<br>Cav3.2<br>Cav3.3 |

## HH_conc reclassified manual_structure

| SUFFIX | Fit | Special notes |
|---|---|---|
| `Kv1p5_MA20_GrC` | `manual_structure` | Reads `ki/ko/nai/nao` and writes `ino`; this is a mixed-ion current, not a plain single-ion HH channel. |
| `Kv1p5_MA24_PC` | `manual_structure` | Current file comments out the extra ions, but the current law still depends on `nai/nao/ki/ko`; treat it as mixed-ion structure. |

## Markov_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| Kca1.1<br>Kca2.2 | - | Kca1.1<br>Kca2.2 | Kca1.1<br>Kca2.2 | - | Kca1.1<br>Kca2.2 | Kca1.1<br>Kca2.2 |


## total table
| Model / Condition | BC (15) | DCN (11) | GoC (16) | GrC (13) | IO (4) | PC (16) | SC (14) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HH_no_conc (40) | HCN1, Kir2.3, Kv1.1, Kv3.4, Kv4.3 | HCN, NaF, NaP, fKdr, sKdr | HCN1, HCN2, KM, Kv1.1, Kv3.4, Kv4.3, CaHVA, Cav2.3 | KM, Kir2.3, Kv1.1, Kv2.2, Kv3.4, Kv4.3, CaHVA | HCN, Na, Kdr, Ca | HCN1, Kir2.3, Kv1.1, Kv3.4, Kv4.3 | HCN1, KM, Kir2.3, Kv1.1, Kv3.4, Kv4.3 |
| Markov_no_conc (9) | Nav1.1, Nav1.6 | - | Nav1.6 | Nav, NaFHF | - | Nav1.6, Kv3.3 | Nav1.1, Nav1.6 |
| Ion_dyn (7) | CdpStC | CdpHVA,CdpLVA | CdpStC | CdpCR | - | CdpCAM | CdpStC |
| HH_conc (23) | Kca3.1, Cav1.2, Cav1.3, Cav2.1, Cav3.2 | SK, CaHVA, CaL, CaLVA | Kca3.1, Cav1.2, Cav1.3, Cav3.1 | Kv1.5 | - | Kca3.1, Cav2.1, Cav3.1, Cav3.2, Cav3.3, Kv1.5 | Cav2.1, Cav3.2, Cav3.3 |
| Markov_conc (10) | Kca1.1, Kca2.2 | - | Kca1.1, Kca2.2 | Kca1.1, Kca2.2 | - | Kca1.1, Kca2.2 | Kca1.1, Kca2.2 |
