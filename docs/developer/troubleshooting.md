# 开发排错

先确认当前 Python 环境、源码位置和失败阶段，再缩小到可复现的命令。
提交问题时附上命令、版本、完整异常和最小输入。

## 修改源码后结果没有变化

确认解释器实际导入的是当前 checkout：

```bash
python -m pip show braincell
python -c "import sys, braincell; print(sys.executable); print(braincell.__file__)"
```

如果路径指向另一个环境或安装副本，回到目标环境，在仓库根目录执行
`python -m pip install -e ".[dev]"`，然后重新启动已有 Python 或 notebook kernel。

## 缺少测试、绘图或文档依赖

贡献开发使用 `.[dev]`。按任务单独安装时，测试使用 `.[testing]`，文档使用 `.[doc]`，
绘图使用 `.[vis]`，NeuroMorpho 客户端使用 `.[io]`。
这些 extras 以 [pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml) 为准。

如果看到 GPU 回退 CPU 的提示，先检查当前设备：

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

根 conftest 为普通测试设置 CPU 环境；GPU 安装和设备选择见
[安装指南](../getting_started/installation.ipynb)。

## 测试未收集或找不到输入文件

从仓库根目录运行 `python -m pytest`，先用 `--collect-only` 检查目标测试文件。
新测试按相邻源码命名为 `*_test.py`，配置在 `pyproject.toml`。
形态 fixture 从 `braincell.io._testing` 导入，具体例子见 [测试指南](testing.md#复用形态-fixture)。
其他工作流的数据和可选依赖按其本地说明准备。

## 文档构建失败或链接缺页

先确认安装了文档依赖，再从仓库根目录构建：

```bash
python -m sphinx -b html docs docs/_build/html
```

按日志中的源文件和行号检查新增警告。Markdown、RST 和 notebook 的内部引用使用相同的文档页面名称；
转换文件格式时保留页面名称，并移除旧格式文件，避免同名页面冲突。
Design 当前不参与网站构建，Developer 到 Design 的链接指向 GitHub 文件。
网站构建通过不代表 notebook 代码已执行。

重新生成包含 PyVista 交互图的 notebook 时，HTML 导出还需要以下依赖：

```bash
python -m pip install ipywidgets trame trame-vtk trame-vuetify "jupyterlab>=3"
```

使用 `vis3d(notebook=True, jupyter_backend="html")` 并保存执行结果，
让发布页面包含导出的交互 HTML；调用参数见
[Vis API](https://github.com/chaobrain/braincell/blob/main/docs/design/vis/current/api.md)。

## 建模和数值问题的入口

| 现象 | 查阅位置 |
| --- | --- |
| 物理单位或参数形状不匹配 | [单位用法](../concepts/units.ipynb)、对应模块 Current 的参数说明 |
| dt/duration、记录结果或连续运行异常 | [Cell 运行接口](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/api.md#运行与结果)、[Recording](https://github.com/chaobrain/braincell/blob/main/docs/design/network/current/recording.md) |
| solver 名称或适用模型不匹配 | [Quad API](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/current/api.md) |
| 数值发散或与参照不符 | [测试验收方法](testing.md#数值结果如何验收)、[Cell 求解路径](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/architecture.md#两条电压路径) |
| SWC/ASC 读取失败 | [Reader 与诊断报告](https://github.com/chaobrain/braincell/blob/main/docs/design/io/current/api.md) |

仍不能定位时，在 [Issues](https://github.com/chaobrain/braincell/issues) 提供可运行复现及环境信息。
