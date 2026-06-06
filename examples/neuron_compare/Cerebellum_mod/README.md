# Cerebellum mod classification overview

本表仅覆盖当前 `channel/ion` 目录中的机制，不包含 `synapse/other`。

## Classification rule

- `HH_no_conc`: 以 `DERIVATIVE` 为主，且不显式依赖浓度变量。
- `Markov_no_conc`: 以 `KINETIC` 为主，且不显式依赖浓度变量。
- `Ion_dyn`: 独立的离子浓度 / 缓冲 / pump 动力学模块。
- `HH_conc`: 以 `DERIVATIVE` 为主，且显式依赖浓度变量。
- `Markov_conc`: 以 `KINETIC` 为主，且显式依赖浓度变量。
- 表格中展示名已经去掉作者年份与 cell 后缀，并把内部的 `p` 还原成 `.` 仅用于阅读；真实机制名以各 `.mod` 文件中的 `SUFFIX` 为准。

## TABLE status summary

本表仅覆盖 `channel/ion` 目录。`TABLE` 在 NEURON 中是有限范围插值表；输入超过 `FROM ... TO ...` 范围时会使用边界值，因此 `[-100,30]` 和浓度区间 `[0,0.01]` 这类表需要特别关注。状态栏含义：

- `仍 TABLE`: 当前 `.mod` 里仍保留 `TABLE`。
- `已连续化`: 当前工作区已移除 `TABLE`，改为每次按连续公式计算。
- `[截断]`: 表范围外会边界钳制；对动作电位可能超过 `30 mV` 的电压表、或浓度可能越界的浓度表特别标注。

| Mechanism | BC | DCN | GoC | GrC | PC | SC | Notes |
|---|---|---|---|---|---|---|---|
| `Kv4p3` | `已连续化`<br>原 `[-100,30] [截断]` |  | `已连续化`<br>原 `[-100,30] [截断]` | `已连续化`<br>原 `[-100,30] [截断]` | `已连续化`<br>原 `[-100,30] [截断]` | `已连续化`<br>原 `[-100,30] [截断]` | `a_inf/tau_a/b_inf/tau_b`; 同族参数还涉及 NMODL 默认值有效数字重写。 |
| `Kir2p3` | `已连续化`<br>原 `[-100,100]` |  |  | `已连续化`<br>原 `[-100,100]` | `已连续化`<br>原 `[-100,100]` | `已连续化`<br>原 `[-100,100]` | `d_inf/tau_d`。 |
| `Kca3p1` | `已连续化`<br>原 `V[-100,100]`<br>原 `cai[0,0.01] [截断]` |  | `已连续化`<br>原 `V[-100,100]`<br>原 `cai[0,0.01] [截断]` |  | `已连续化`<br>原 `V[-100,100]`<br>原 `cai[0,0.01] [截断]` |  | `Yvdep/Yconcdep`; 浓度表原先超过 `0.01 mM` 会边界钳制。 |
| `KM` |  |  | `已连续化`<br>原 `[-100,30] [截断]` | `已连续化`<br>原 `[-100,30] [截断]` |  | `已连续化`<br>原 `[-100,30] [截断]` | `n_inf/tau_n`。 |
| `CaHVA` |  | `已连续化`<br>原 `[-150,100]` | `已连续化`<br>原 `[-100,30] [截断]` | `已连续化`<br>原 `[-100,30] [截断]` |  |  | DCN 原有 `minf/taum` 表和 `DEPEND T` 表；GoC/GrC 为 `s_inf/tau_s/u_inf/tau_u`。 |
| `HCN1` |  |  | `已连续化`<br>原 `[-100,30] [截断]` |  |  |  | `o_fast_inf/o_slow_inf/tau_f/tau_s`。 |
| `HCN2` |  |  | `已连续化`<br>原 `[-100,30] [截断]` |  |  |  | `o_fast_inf/o_slow_inf/tau_f/tau_s`。 |
| `Cav2p3` |  |  | `已连续化`<br>原 `[-100,100]` |  |  |  | Indexed `inf/tau` table. |
| `CaLVA` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | 原 `minf/taum/hinf/tauh` 表和 `DEPEND T` 表。 |
| `CaL` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf/taum/hinf/tauh`。 |
| `HCN` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf`。 |
| `NaF` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf/taum/hinf/tauh`。 |
| `NaP` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf/hinf/tauh`。 |
| `SK` |  | `已连续化`<br>原 `[0,0.01] [截断]` |  |  |  |  | `zinf/tauz`; calcium concentration table. |
| `fKdr` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf/taum`。 |
| `sKdr` |  | `已连续化`<br>原 `[-150,100]` |  |  |  |  | `minf/taum`。 |

`synapse/other` 不纳入上表；其中 NMDA `MgBlock TABLE [-120,30]` 等表仍存在，但属于突触机制，不计入当前 channel/ion 转换状态。

## Integration method status

本表仅记录当前工作区中 `channel/ion` 目录下已从 `derivimplicit` 替换为 `cnexp` 的 HH gate ODE。原本就是 `cnexp` 的机制不标为替换；Markov/KINETIC 的 `sparse` 机制不纳入本表。

| Mechanism | BC | GoC | GrC | PC | SC | Notes |
|---|---|---|---|---|---|---|
| `CaHVA` |  |  | `derivimplicit -> cnexp` |  |  | GrC 替换；GoC 原本已是 `cnexp`，不计为替换。 |
| `KM` |  | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` |  | `derivimplicit -> cnexp` | 单 gate HH ODE。 |
| `Kir2p3` | `derivimplicit -> cnexp` |  | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | 单 gate HH ODE。 |
| `Kv1p5` |  |  | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` |  | 单 gate HH ODE。 |
| `Kv4p3` | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | `derivimplicit -> cnexp` | `a/b` 两个独立 HH gates。 |

## Rate update placement status

本表记录当前工作区中 `SOLVE state/states METHOD cnexp` 相关 HH 机制的速率函数调用位置变更。`INITIAL` 中保留的初始化调用不计为变更；Markov/KINETIC 的 `sparse` 或 `seqinitial` 机制不纳入本表。

| Mechanism | BC | GoC | IO | Old placement | New placement | Notes |
|---|---|---|---|---|---|---|
| `Cav1p2` | `已调整` | `已调整` |  | `BREAKPOINT: rates()` | `DERIVATIVE state: rates()` | `inf/tau` 在 `cnexp` 状态更新前刷新。 |
| `Cav1p3` | `已调整` | `已调整` |  | `BREAKPOINT: rates()` | `DERIVATIVE state: rates()` | `inf/tau` 在 `cnexp` 状态更新前刷新。 |
| `Ca` |  |  | `已调整` | `BREAKPOINT: rates(v)` | `DERIVATIVE states: rates(v)` | IO channel。 |
| `HCN` |  |  | `已调整` | `BREAKPOINT: rates(v)` | `DERIVATIVE states: rates(v)` | IO channel。 |
| `Kdr` |  |  | `已调整` | `BREAKPOINT: rates(v)` | `DERIVATIVE states: rates(v)` | IO channel。 |
| `Na` |  |  | `已调整` | `BREAKPOINT: rates(v)` | `DERIVATIVE states: rates(v)` | IO channel。 |

## NMODL numeric default precision

NEURON/NMODL 生成的 C 代码会把部分 `PARAMETER`/global 默认值写成约 6 位有效数字。下表记录当前需要和 BrainCell 对齐的默认值；公式内部普通字面量如 IO 的 `41.000001`、DCN GHK 的 `23.20764929`、以及 `e0 = 1.60217646e-19` 保持原值，不属于这个默认值重写问题。

| Mechanism | Cells | Names | MOD source | NEURON compiled | BrainCell status |
|---|---|---|---|---|---|
| `Kv4p3` | BC, GoC, GrC, PC, SC | `Kalpha_a`, `Kbeta_a`, `V0beta_a`, `V0alpha_b` | `-23.32708`, `19.47175`, `-18.27914`, `-111.33209` | `-23.3271`, `19.4718`, `-18.2791`, `-111.332` | `已对齐` |
| `HCN1` | GoC | `tEf`, `tEs` | `2.302585092`, `2.302585092` | `2.30259`, `2.30259` | `已对齐` |
| `CaHVA` | GoC, GrC | `Kalpha_s` | `15.87301587302` | `15.873` | `已对齐` |
| `ToyDiamFactorKinetic` | DCN | `pump_area`, `cyto` | `62.83185307179586`, `62.83185307179586` | `62.8319`, `62.8319` | `例外`: BrainCell 使用运行时 `pi * diam_mid` / `pi * diam_mid * depth` 几何派生，不做常量替换。 |

## Region totals

| Region | Count |
|---|---:|
| HH_no_conc | 41 |
| Markov_no_conc | 8 |
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
| HCN1<br>Kir2.3<br>Kv1.1<br>Kv3.4<br>Kv4.3 | HCN<br>NaF<br>NaP<br>fKdr<br>sKdr | HCN1<br>HCN2<br>KM<br>Kv1.1<br>Kv3.4<br>Kv4.3<br>CaHVA<br>Cav2.3 | KM<br>Kir2.3<br>Kv1.1<br>Kv2.2_0010<br>Kv3.4<br>Kv4.3<br>CaHVA | HCN<br>Na<br>Kdr<br>Ca | HCN1<br>Kir2.3<br>Kv1.1<br>Kv3.3<br>Kv3.4<br>Kv4.3 | HCN1<br>KM<br>Kir2.3<br>Kv1.1<br>Kv3.4<br>Kv4.3 |

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
| `Kv1p5_MA24_PC` | `direct_hh` | The PC file only enables `USEION k WRITE ik`; `ino` is calculated as a RANGE variable but its `USEION no WRITE ino` line is commented out. |
| `Kv3p3_MA24_PC` | `hh_special_current` | The comment block says `KINETIC SCHEME: Hodgkin-Huxley (n^4)`; extra `NONSPECIFIC_CURRENT i/igate`; keep the optional gating-current path. |
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
| Nav1.1<br>Nav1.6 | - | Nav1.6 | Nav<br>NaFHF | - | Nav1.6 | Nav1.1<br>Nav1.6 |

## Ion_dyn

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| CdpStC | CdpHVA<br>CdpLVA | CdpStC | CdpCR | - | CdpCAM | CdpStC |

## Ion_dyn inherited variants

| SUFFIX | Base implementation | Special notes |
|---|---|---|
| `CdpStC_MA25_BC` | `CdpStC_NoCAM_MA20_GoC` | Same pump, non-CAM buffer, and PV kinetic network; source CAM block is commented out. |
| `CdpStC_RI21_SC` | `CdpStC_NoCAM_MA20_GoC` | Same pump, non-CAM buffer, and PV kinetic network; extra `cao` read is not used by the equations. |

## Ion_dyn implementation notes

| SUFFIX | Fit | Special notes |
|---|---|---|
| `CdpCAM_MA24_PC` | `manual_structure` | Similar to `CdpStC_MA20_GoC`, but enables the CB subnetwork and places both CB and CAM states in the cytosolic compartment. |
| `CdpCR_MA20_GrC` | `manual_structure` | Similar Cdp pump/buffer scaffold, but replaces PV/CB/CAM with the GrC Calretinin network. |

## HH_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| Kca3.1<br>Cav1.2<br>Cav1.3<br>Cav2.1<br>Cav3.2 | SK<br>CaHVA<br>CaL<br>CaLVA | Kca3.1<br>Cav1.2<br>Cav1.3<br>Cav3.1 | Kv1.5 | - | Kca3.1<br>Cav2.1<br>Cav3.1<br>Cav3.2<br>Cav3.3<br>Kv1.5 | Cav2.1<br>Cav3.2<br>Cav3.3 |

## HH_conc default ik conversion

| SUFFIX | Fit | Special notes |
|---|---|---|
| `Kv1p5_MA20_GrC` | `default_ik_only` | The source mod enables `ino`, but `gnonspec` defaults to zero and no repo config sets it nonzero; BrainCell converts the default `ik` path and intentionally does not expose nonzero `ino`. |

## HH_conc inherited variants

| SUFFIX | Base implementation | Special notes |
|---|---|---|
| `Kca3p1_MA25_BC` | `Kca3p1_MA20_GoC` | Same core `Y/p` kinetics and `ik` path; BC-only `g_equiv = gkbar*Y` is a derived RANGE value and is not exposed as a BrainCell field. |
| `Kca3p1_MA24_PC` | `Kca3p1_MA20_GoC` | Same core `Y/p` kinetics and `ik` path. |

## Markov_conc

| BC | DCN | GoC | GrC | IO | PC | SC |
|---|---|---|---|---|---|---|
| Kca1.1<br>Kca2.2 | - | Kca1.1<br>Kca2.2 | Kca1.1<br>Kca2.2 | - | Kca1.1<br>Kca2.2 | Kca1.1<br>Kca2.2 |


## total table
| Model / Condition | BC (15) | DCN (11) | GoC (16) | GrC (13) | IO (4) | PC (16) | SC (14) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HH_no_conc (41) | HCN1, Kir2.3, Kv1.1, Kv3.4, Kv4.3 | HCN, NaF, NaP, fKdr, sKdr | HCN1, HCN2, KM, Kv1.1, Kv3.4, Kv4.3, CaHVA, Cav2.3 | KM, Kir2.3, Kv1.1, Kv2.2, Kv3.4, Kv4.3, CaHVA | HCN, Na, Kdr, Ca | HCN1, Kir2.3, Kv1.1, Kv3.3, Kv3.4, Kv4.3 | HCN1, KM, Kir2.3, Kv1.1, Kv3.4, Kv4.3 |
| Markov_no_conc (8) | Nav1.1, Nav1.6 | - | Nav1.6 | Nav, NaFHF | - | Nav1.6 | Nav1.1, Nav1.6 |
| Ion_dyn (7) | CdpStC | CdpHVA,CdpLVA | CdpStC | CdpCR | - | CdpCAM | CdpStC |
| HH_conc (23) | Kca3.1, Cav1.2, Cav1.3, Cav2.1, Cav3.2 | SK, CaHVA, CaL, CaLVA | Kca3.1, Cav1.2, Cav1.3, Cav3.1 | Kv1.5 | - | Kca3.1, Cav2.1, Cav3.1, Cav3.2, Cav3.3, Kv1.5 | Cav2.1, Cav3.2, Cav3.3 |
| Markov_conc (10) | Kca1.1, Kca2.2 | - | Kca1.1, Kca2.2 | Kca1.1, Kca2.2 | - | Kca1.1, Kca2.2 | Kca1.1, Kca2.2 |
