# Filter Continuous Sampling

`braincell.filter.sample` 在连续形态上抽取位置，返回可供 Cell.place 消费的 LocsetExpr。区域与位点解析见 [Filter API](api.md)，空间参数上下文见 [Callable 参数](spatial-callable-parameters.md)。

## Continuous Location Sampling

### `braincell.filter.sample`

```text
braincell.filter.sample(
    region,
    *,
    number,
    seed,
    measure="length",
    density=None,
    u_resolution=1e-10,
) -> SampleLocations
```

创建一个延迟解析的连续随机 `LocsetExpr`。表达式在获得具体 morphology 后才生成 `branch_id` 和连续
`branch_x`，因此可以直接传给 `Cell.place` 或 `Network.connect(locations=...)`。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `region` | `RegionExpr` | required | 连续 morphology 支持域。 |
| `number` | positive `int` | required | 样本数；保留抽样顺序和重复位置。 |
| `seed` | `int` | required | 该采样规则独立使用的显式随机种子。 |
| `measure` | `{"normalized", "length", "lateral_area", "area"}` | `"length"` | density 下方的基础几何测度。 |
| `density` | callable or `None` | `None` | 接收 `SamplingContext` 的非负、无量纲位置偏好。 |
| `u_resolution` | `float` | `1e-10` | 数值逆 CDF 的目标精度，范围为 `[1e-12, 1e-5]`。 |

#### Returns

| Type | Description |
| --- | --- |
| `SampleLocations` | 延迟到 morphology 已知时解析的 locset expression。 |

#### Probability measure

设所选区域为 \(R\)，用户 density 为 \(\rho\)，`measure` 指定的几何测度为 \(\mu_m\)。对任意
子区域 \(A\subseteq R\)，一个样本落入其中的概率为：
