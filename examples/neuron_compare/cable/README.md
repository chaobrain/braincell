# cable

`cable` 用来比较同一份 morphology 在 `NEURON` 和 `braincell` 两侧的纯 cable 电压响应。

- 官方输入面是 `configs/*.json + scan_templates/*.json`。
- 一个 scan template 只描述一个 `base + group`，主配置可以引用多个子模板。
- 单跑 `docs/workflow.ipynb` 默认指向 `io.swc` 的 `DC smoke` 模板。
- 推荐 notebook 入口：
  - `docs/workflow.ipynb`
  - `docs/batch_workflow.ipynb`
- `cases/` 只保留单 case 样例和 ASC fixture。
- `templates/` 是实现代码，`tests/` 是当前 harness 测试。
